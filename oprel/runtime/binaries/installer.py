"""
Binary installation and management
"""

import os
import platform
import shutil
import ssl
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from oprel.core.config import Config
from oprel.core.exceptions import BinaryNotFoundError, UnsupportedPlatformError
from oprel.runtime.binaries.registry import get_binary_info, get_supported_platforms, get_optimal_platform_key
from oprel.telemetry.hardware import detect_gpu
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def _has_vulkan_runtime() -> bool:
    """Best-effort detection for Vulkan-capable environments."""
    if platform.system() not in {"Windows", "Linux"}:
        return False

    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        return result.returncode == 0
    except Exception:
        return False


def _create_ssl_context(config: Optional[Config] = None) -> Optional[ssl.SSLContext]:
    """
    Create SSL context for downloads with proper certificate verification.
    
    Supports:
    - Custom CA bundle via config.ssl_cert_file
    - Disable verification via config.ssl_verify=False
    - Environment variable override: OPREL_SSL_NO_VERIFY=1
    
    Args:
        config: Config object with SSL settings (creates default if None)
        
    Returns:
        SSL context or None for unverified context
    """
    if config is None:
        config = Config()
    
    # Check environment variable override (useful for quick troubleshooting)
    env_no_verify = os.environ.get("OPREL_SSL_NO_VERIFY", "0").lower() in ("1", "true", "yes")
    
    if env_no_verify or not config.ssl_verify:
        logger.warning(
            "SSL certificate verification is DISABLED. "
            "This is insecure and should only be used in trusted networks or with corporate proxies."
        )
        return ssl._create_unverified_context()
    
    # Create default SSL context with system certificates
    context = ssl.create_default_context()
    
    # Load custom CA bundle if provided
    if config.ssl_cert_file and config.ssl_cert_file.exists():
        logger.info(f"Loading custom CA certificates from: {config.ssl_cert_file}")
        context.load_verify_locations(cafile=str(config.ssl_cert_file))
    
    return context


def _safe_download(url: str, dest_path: Path, config: Optional[Config] = None) -> None:
    """
    Download a file with proper SSL handling and error reporting.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        config: Configuration for SSL settings
        
    Raises:
        BinaryNotFoundError: On download failure with helpful error message
    """
    try:
        ssl_context = _create_ssl_context(config)
        # urlretrieve does not accept an SSL context on all Python versions.
        # Use urlopen with the context and stream to the destination file instead.
        req = urllib.request.Request(url)
        if ssl_context is not None:
            resp = urllib.request.urlopen(req, context=ssl_context)
        else:
            resp = urllib.request.urlopen(req)

        with resp as response, dest_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except ssl.SSLError as e:
        error_msg = (
            f"SSL certificate verification failed: {e}\n\n"
            "This usually happens due to:\n"
            "  1. Corporate proxy/firewall with custom certificates\n"
            "  2. Outdated system certificates\n"
            "  3. Antivirus software intercepting SSL connections\n\n"
            "Solutions:\n"
            "  • Quick fix (insecure): Set environment variable OPREL_SSL_NO_VERIFY=1\n"
            "  • Better fix: Install your corporate CA certificate and use config.ssl_cert_file\n"
            "  • Windows: Update certificates via Windows Update\n"
            "  • Contact your IT department for the proper CA certificate bundle\n"
        )
        raise BinaryNotFoundError(error_msg) from e
    except urllib.error.URLError as e:
        raise BinaryNotFoundError(
            f"Failed to download from {url}: {e}\n"
            "Check your internet connection and firewall settings."
        ) from e
    except Exception as e:
        raise BinaryNotFoundError(f"Download failed: {e}") from e


def ensure_binary(
    backend: str,
    version: str,
    binary_dir: Path,
    force_download: bool = False,
    config: Optional[Config] = None,
    prefer_cpu: bool = False,
) -> Path:
    """
    Ensure the required binary is installed.
    Automatically selects CUDA version if GPU is available.

    Args:
        backend: Backend name ("llama.cpp", "vllm", etc.)
        version: Binary version (e.g., "b7822" or "latest")
        binary_dir: Directory to store binaries
        force_download: Re-download even if exists

    Returns:
        Path to binary executable

    Raises:
        UnsupportedPlatformError: If platform not supported
        BinaryNotFoundError: If download fails
    """
    # Detect platform
    system = platform.system()
    machine = platform.machine()
    base_platform_key = f"{system}-{machine}"
    
    # Check accelerators so we can prefer the best prebuilt variant.
    gpu_info = detect_gpu()
    has_cuda = False if prefer_cpu else (gpu_info is not None and gpu_info.get("gpu_type") == "cuda")
    prefer_rocm = False if prefer_cpu else (gpu_info is not None and gpu_info.get("gpu_type") == "rocm")
    prefer_vulkan = False if prefer_cpu else (_has_vulkan_runtime() and not has_cuda and not prefer_rocm)
    
    platform_key = get_optimal_platform_key(
        backend,
        version,
        base_platform_key,
        has_cuda,
        prefer_vulkan=prefer_vulkan,
        prefer_rocm=prefer_rocm,
    )
    
    if platform_key != base_platform_key:
        logger.info(f"Using accelerator-optimized binary: {platform_key}")

    # Get binary info from registry
    binary_info = get_binary_info(backend, version, platform_key)
    
    # Fallback to base platform if the preferred accelerator build is unavailable.
    if not binary_info and platform_key != base_platform_key:
        logger.warning(f"Accelerator-specific binary not available for {platform_key}, falling back to CPU")
        platform_key = base_platform_key
        binary_info = get_binary_info(backend, version, platform_key)

    if not binary_info:
        available = get_supported_platforms(backend, version)
        if not available:
            raise BinaryNotFoundError(f"No binary found for {backend} version {version}")
        raise UnsupportedPlatformError(
            f"Platform {platform_key} not supported. Available: {available}"
        )

    url = binary_info["url"]
    archive_type = binary_info["archive_type"]
    binary_name = binary_info["binary_name"]
    gpu_type = binary_info.get("gpu_type", "cpu")

    # Isolate binaries by backend to avoid collisions between llama.cpp and
    # stable-diffusion.cpp both using an oprel-branded executable name.
    backend_dir_name = backend.replace(".", "_").replace("-", "_")

    # Use different directory for CUDA vs CPU binaries to avoid conflicts.
    if gpu_type == "cuda":
        actual_binary_dir = binary_dir / backend_dir_name / "cuda"
    else:
        actual_binary_dir = binary_dir / backend_dir_name / "cpu"
    
    binary_path = actual_binary_dir / binary_name
    
    # Create oprel-branded binary name
    # Create oprel-branded binary name
    oprel_binary_name = "oprel-backend.exe" if system == "Windows" else "oprel-backend"
    oprel_binary_path = actual_binary_dir / oprel_binary_name

    # Check if already exists with required shared libraries
    if oprel_binary_path.exists() and not force_download:
        # Check for CUDA-specific libraries if this is a CUDA binary
        if gpu_type == "cuda":
            cuda_dll = list(actual_binary_dir.glob("*cuda*.dll")) + list(actual_binary_dir.glob("*cuda*.so*"))
            if not cuda_dll:
                logger.info(f"CUDA binary exists but CUDA libraries missing, re-downloading...")
            else:
                logger.info(f"CUDA binary already exists: {oprel_binary_path}")
                return oprel_binary_path
        elif system == "Linux":
            # Check for any .so files in the binary directory
            so_files = list(actual_binary_dir.glob("*.so*"))
            if not so_files:
                logger.info(f"Binary exists but shared libraries missing, re-downloading...")
            else:
                logger.info(f"Binary already exists: {oprel_binary_path}")
                return oprel_binary_path
        else:
            logger.info(f"Binary already exists: {oprel_binary_path}")
            return oprel_binary_path

    # If the original backend binary exists but the oprel-branded copy is missing,
    # recreate the copy locally instead of re-downloading archives.
    if binary_path.exists() and not oprel_binary_path.exists() and not force_download:
        shutil.copy2(binary_path, oprel_binary_path)
        if system != "Windows":
            oprel_binary_path.chmod(0o755)
        logger.info(f"Re-created oprel-branded binary: {oprel_binary_path}")
        return oprel_binary_path

    # Download and extract binary
    logger.info(f"Downloading {backend} ({gpu_type.upper()}) binary from {url}")
    actual_binary_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize config if not provided
    if config is None:
        config = Config()

    try:
        # Download main binary
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{archive_type}") as tmp:
            tmp_path = Path(tmp.name)

        logger.info(f"Downloading to temp file: {tmp_path}")
        _safe_download(url, tmp_path, config)

        # Extract based on archive type
        if archive_type == "zip":
            _extract_zip(tmp_path, actual_binary_dir, binary_name)
        elif archive_type == "tar.gz":
            _extract_tarball(tmp_path, actual_binary_dir, binary_name)
        elif archive_type == "exe":
            # Direct executable, just move it
            shutil.move(tmp_path, binary_path)
        else:
            raise BinaryNotFoundError(f"Unknown archive type: {archive_type}")

        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()
            
        # Check for separate DLL download (Windows CUDA)
        dll_url = binary_info.get("dll_url")
        if dll_url:
            logger.info(f"Downloading required DLLs from {dll_url}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_dll:
                tmp_dll_path = Path(tmp_dll.name)
            
            _safe_download(dll_url, tmp_dll_path, config)
            # DLL zip usually has libs in specific folder, extract flat
            _extract_zip(tmp_dll_path, actual_binary_dir, "non-existent-file-to-force-extract-all")
            
            if tmp_dll_path.exists():
                tmp_dll_path.unlink()

        # Make executable on Unix
        if system != "Windows":
            binary_path.chmod(0o755)

        if not binary_path.exists():
            raise BinaryNotFoundError(
                f"Binary {binary_name} not found after extraction. "
                "The archive structure may have changed."
            )

        # Create a copy with oprel-specific name for easier process identification
        oprel_binary_name = "oprel-backend.exe" if system == "Windows" else "oprel-backend"
        oprel_binary_path = actual_binary_dir / oprel_binary_name
        
        # Copy the binary to oprel-backend so processes show up as "oprel-backend"
        if not oprel_binary_path.exists() or force_download:
            shutil.copy2(binary_path, oprel_binary_path)
            if system != "Windows":
                oprel_binary_path.chmod(0o755)
            logger.debug(f"Created oprel-branded binary: {oprel_binary_path}")

        logger.info(f"Binary installed: {binary_path} ({gpu_type.upper()})")
        return oprel_binary_path  # Return the oprel-branded binary instead

    except Exception as e:
        # Clean up on failure
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink()
        raise BinaryNotFoundError(f"Failed to download/extract binary: {e}") from e


def _extract_zip(zip_path: Path, output_dir: Path, binary_name: str) -> None:
    """Extract binary from zip archive."""
    logger.info(f"Extracting zip archive: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract all files first
        zf.extractall(output_dir)

        # Find and move the binary to the root of output_dir
        binary_found = False
        for name in zf.namelist():
            if name.endswith(binary_name):
                logger.info(f"Found binary in archive: {name}")
                extracted = output_dir / name
                target = output_dir / binary_name

                if extracted != target and extracted.exists():
                    if target.exists():
                        target.unlink()
                    shutil.move(str(extracted), str(target))
                binary_found = True
                break

        if not binary_found:
            logger.warning(f"Binary {binary_name} not found in archive")
            logger.info(f"Extracted contents: {list(zf.namelist())[:10]}...")

        # Copy all shared libraries (.so, .dll) to the output_dir root
        # This ensures they're found alongside the binary at runtime
        for name in zf.namelist():
            if name.endswith(".so") or ".so." in name or name.endswith(".dll"):
                src = output_dir / name
                if src.exists() and src.is_file():
                    dst = output_dir / src.name
                    if src != dst:
                        if dst.exists():
                            dst.unlink()
                        shutil.copy2(str(src), str(dst))
                        logger.debug(f"Copied library: {src.name}")


def _extract_tarball(tar_path: Path, output_dir: Path, binary_name: str) -> None:
    """Extract binary from tar.gz archive."""
    logger.info(f"Extracting tarball: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tf:
        # Extract all files first
        tf.extractall(output_dir)

        # Find and move the binary to the root of output_dir
        binary_found = False
        for member in tf.getmembers():
            if member.name.endswith(binary_name):
                logger.info(f"Found binary in archive: {member.name}")
                extracted = output_dir / member.name
                target = output_dir / binary_name

                if extracted != target and extracted.exists():
                    if target.exists():
                        target.unlink()
                    shutil.move(str(extracted), str(target))
                binary_found = True
                break

        if not binary_found:
            logger.warning(f"Binary {binary_name} not found in archive")
            members = tf.getnames()
            logger.info(f"Extracted contents: {members[:10]}...")

        # Copy all shared libraries (.so) to the output_dir root
        for member in tf.getmembers():
            if member.name.endswith(".so") or ".so." in member.name:
                src = output_dir / member.name
                if src.exists() and src.is_file():
                    dst = output_dir / src.name
                    if src != dst:
                        if dst.exists():
                            dst.unlink()
                        shutil.copy2(str(src), str(dst))
                        logger.debug(f"Copied library: {src.name}")
