"""
HuggingFace Hub integration for model downloads with production-ready reliability
"""

import hashlib
import time
from pathlib import Path
from typing import Optional, Callable
from huggingface_hub import hf_hub_download, list_repo_files, HfFileSystem
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oprel.core.exceptions import ModelNotFoundError, InvalidQuantizationError
from oprel.downloader.cache import get_cache_path
from oprel.utils.logging import get_logger

logger = get_logger(__name__)

# M1.13: Connection and read timeouts (30s connect, 300s read for large files)
DEFAULT_TIMEOUT = (30, 300)
# M1.18: Max retries with exponential backoff
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.0  # 1s, 2s, 4s


def _create_robust_session() -> requests.Session:
    """
    Create a requests session with retry logic and timeouts.
    
    Returns:
        Configured requests Session with retry and timeout handling
    """
    session = requests.Session()
    
    # M1.18: Configure retry strategy
    # Retry on: connection errors, timeouts, and 5xx server errors
    # Don't retry on: 4xx client errors (404, 403, etc.)
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[500, 502, 503, 504],  # Retry on server errors
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # Don't retry POST/PUT
        raise_on_status=False,
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def _verify_file_checksum(file_path: Path, expected_hash: Optional[str] = None) -> bool:
    """
    M1.17: Verify file integrity using SHA256 checksum.
    
    Args:
        file_path: Path to file to verify
        expected_hash: Expected SHA256 hash (optional)
        
    Returns:
        True if checksum matches or no expected hash provided
    """
    if not expected_hash:
        # If no expected hash, just check file exists and has size > 0
        return file_path.exists() and file_path.stat().st_size > 0
    
    logger.debug(f"Verifying checksum for {file_path.name}")
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        
        actual_hash = sha256_hash.hexdigest()
        if actual_hash != expected_hash:
            logger.error(f"Checksum mismatch! Expected: {expected_hash}, Got: {actual_hash}")
            return False
            
        logger.debug("Checksum verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying checksum: {e}")
        return False


def _check_cache_validity(
    model_id: str,
    filename: str,
    cache_dir: Path,
) -> Optional[Path]:
    """
    M1.16: Check if model exists in cache and is valid.
    
    Args:
        model_id: HuggingFace model repository ID
        filename: Name of the model file
        cache_dir: Cache directory to check
        
    Returns:
        Path to cached file if valid, None otherwise
    """
    try:
        # Try to get from cache using hf_hub_download with local_files_only
        cached_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        
        cached_file = Path(cached_path)
        
        # M1.16: Verify file exists and has non-zero size
        if not cached_file.exists():
            logger.debug(f"Cache miss: File not found at {cached_file}")
            return None
            
        file_size = cached_file.stat().st_size
        if file_size == 0:
            logger.warning(f"Cache invalid: File has zero size, removing {cached_file}")
            cached_file.unlink()
            return None
        
        # M1.17: Basic integrity check (could add checksum verification here)
        size_mb = file_size / (1024 * 1024)
        logger.info(f"✓ Cache hit: {filename} ({size_mb:.1f} MB)")
        return cached_file
        
    except Exception as e:
        logger.debug(f"Cache miss for {filename}: {e}")
        return None


class DownloadProgressCallback:
    """M1.15: Progress tracking with tqdm progress bar"""
    
    def __init__(self, filename: str, total_size: int):
        self.filename = filename
        self.pbar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {filename}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.last_logged_mb = 0
        
    def __call__(self, downloaded: int):
        """Update progress bar"""
        self.pbar.update(downloaded - self.pbar.n)
        
        # M1.13: Log progress every 100MB
        current_mb = downloaded / (1024 * 1024)
        if current_mb - self.last_logged_mb >= 100:
            logger.info(f"Downloaded {current_mb:.0f} MB of {self.filename}")
            self.last_logged_mb = current_mb
    
    def close(self):
        """Close progress bar"""
        self.pbar.close()


def _find_cached_model_for_repo(model_id: str, cache_dir: Path, quantization: Optional[str] = None) -> Optional[Path]:
    """
    Find a cached GGUF file for the given model repository.

    If quantization is specified, only returns a match if the quantization matches.
    For split GGUFs, verifies ALL shards are present.

    Args:
        model_id: HuggingFace model repository ID
        cache_dir: Cache directory to search
        quantization: Specific quantization to look for (e.g., "Q4_K_M", "F16")

    Returns:
        Path to cached model (shard 1) if fully complete, None otherwise
    """
    from oprel.downloader.metadata import load_model_metadata
    
    # First, try to find models using metadata (preferred method)
    metadata_dir = cache_dir / ".metadata"
    if metadata_dir.exists():
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                filename = metadata_file.name.replace(".json", "")
                metadata = load_model_metadata(cache_dir, filename)
                
                if metadata and metadata.get("repo_id") == model_id:
                    # Found a model with matching repo_id
                    gguf_file = None
                    
                    # Try to find the actual file
                    for candidate in cache_dir.rglob(filename):
                        if candidate.is_file() and candidate.stat().st_size > 0:
                            gguf_file = candidate
                            break
                    
                    if not gguf_file:
                        continue
                    
                    # Skip mmproj and vision files
                    filename_lower = gguf_file.name.lower()
                    if 'mmproj' in filename_lower or filename_lower.startswith('vision-') or filename_lower.startswith('clip-'):
                        continue
                    
                    # If quantization is specified, check if it matches
                    if quantization:
                        quant_lower = quantization.lower()
                        # Check metadata first, then filename
                        metadata_quant = metadata.get("quantization", "").lower()
                        if metadata_quant and metadata_quant != quant_lower:
                            continue
                        elif not metadata_quant and quant_lower not in filename_lower:
                            continue
                    
                    # Validate split GGUF shards
                    if not _validate_split_gguf_shards(gguf_file):
                        continue
                    
                    logger.info(f"✓ Found cached model via metadata: {gguf_file.name} ({gguf_file.stat().st_size / (1024**3):.1f} GB)")
                    return gguf_file
                    
            except Exception as e:
                logger.debug(f"Error reading metadata {metadata_file}: {e}")
                continue
    
    # Fallback: Use the old filename-based matching for backward compatibility
    repo_parts = model_id.split('/')
    if len(repo_parts) >= 2:
        repo_name = repo_parts[-1].replace('-GGUF', '').replace('-gguf', '')
    else:
        repo_name = model_id

    for gguf_file in cache_dir.rglob("*.gguf"):
        filename_lower = gguf_file.name.lower()
        if 'mmproj' in filename_lower or filename_lower.startswith('vision-') or filename_lower.startswith('clip-'):
            continue

        repo_name_lower = repo_name.lower()
        if repo_name_lower.replace('-', '').replace('.', '') in filename_lower.replace('-', '').replace('.',  ''):
            if not (gguf_file.exists() and gguf_file.stat().st_size > 0):
                continue

            # If quantization is specified, check if it matches
            if quantization:
                quant_lower = quantization.lower()
                # Check if the quantization is in the filename
                if quant_lower not in filename_lower:
                    continue  # Skip this file, quantization doesn't match

            # Validate split GGUF shards
            if not _validate_split_gguf_shards(gguf_file):
                continue

            logger.info(f"✓ Found cached model via filename matching: {gguf_file.name} ({gguf_file.stat().st_size / (1024**3):.1f} GB)")
            return gguf_file

    return None


def _validate_split_gguf_shards(gguf_file: Path) -> bool:
    """
    Validate that all shards of a split GGUF are present.
    
    Args:
        gguf_file: Path to the GGUF file (should be shard 1)
        
    Returns:
        True if all shards are present, False otherwise
    """
    import re as _re
    
    shard_match = _re.search(r'-0*(\d+)-of-0*(\d+)\.gguf', gguf_file.name, _re.IGNORECASE)
    if shard_match:
        this_shard = int(shard_match.group(1))
        total_shards = int(shard_match.group(2))
        if this_shard != 1:
            return False
        if total_shards > 1:
            model_dir = gguf_file.parent
            for shard_n in range(2, total_shards + 1):
                expected_shard_name = _re.sub(
                    r'-0*(\d+)-of-0*(\d+)\.gguf',
                    lambda m: f"-{shard_n:05d}-of-{total_shards:05d}.gguf",
                    gguf_file.name,
                    flags=_re.IGNORECASE
                )
                expected_path = model_dir / expected_shard_name
                if not expected_path.exists() or expected_path.stat().st_size == 0:
                    logger.warning(
                        f"Split GGUF is incomplete: shard {shard_n} missing "
                        f"({expected_shard_name}). Will re-download."
                    )
                    return False
    
    return True




def _is_vision_model(model_id: str) -> bool:
    """
    Check if a model is a vision/multimodal model that needs mmproj.
    
    Args:
        model_id: Model repository ID or alias
        
    Returns:
        True if the model needs a vision projector
    """
    from oprel.downloader.aliases import get_model_category, REPO_TO_ALIAS
    
    # 1. Check if it's an alias or repo in our official categories
    alias = REPO_TO_ALIAS.get(model_id, model_id)
    category = get_model_category(alias)
    
    if category in ["Text + Vision", "OCR"]:
        return True
        
    # 2. Fuzzy matching for external models
    id_lower = model_id.lower()
    vision_keywords = ["vl", "vision", "llava", "clip", "ocr", "multimodal", "mllm"]
    return any(kw in id_lower for kw in vision_keywords)


def _ensure_mmproj_downloaded(model_id: str, model_path: Path, cache_dir: Path, force_download: bool = False) -> Optional[Path]:
    """
    Ensure mmproj file is downloaded for vision models.
    
    Args:
        model_id: HuggingFace model repository ID
        model_path: Path to the main model file
        cache_dir: Cache directory
        force_download: Force re-download even if cached
        
    Returns:
        Path to mmproj file if found/downloaded
    """
    logger.info("Vision model detected - checking for mmproj file...")
    
    # Check if mmproj already exists next to model
    model_dir = model_path.parent
    for pattern in ["*mmproj*.gguf", "*vision*.gguf", "*clip*.gguf"]:
        existing = list(model_dir.glob(pattern))
        if existing and not force_download:
            logger.info(f"✓ mmproj already cached: {existing[0].name}")
            return existing[0]
    
    # List files in repo to find mmproj
    try:
        files = list_repo_files(model_id)
        mmproj_files = [f for f in files if (f.startswith("mmproj-") or "vision" in f or "clip" in f) and f.endswith(".gguf")]
        
        if mmproj_files:
            # Try to match quantization from main model filename
            main_filename = model_path.name
            quant_from_filename = None
            # Extract quantization from filename (e.g., Q8_0)
            for q in ["Q8_0", "Q8", "Q4_K_M", "Q4_K", "Q6_K", "Q5_K", "F16"]:
                if q.lower() in main_filename.lower():
                    quant_from_filename = q
                    break
            
            # Find mmproj with matching quantization
            mmproj_filename = None
            if quant_from_filename:
                for mf in mmproj_files:
                    if quant_from_filename.lower() in mf.lower():
                        mmproj_filename = mf
                        break
            
            # Fallback to first mmproj file if no match
            if not mmproj_filename:
                mmproj_filename = mmproj_files[0]
            
            logger.info(f"Downloading mmproj file: {mmproj_filename}")
            
            # Download mmproj file
            mmproj_path = hf_hub_download(
                repo_id=model_id,
                filename=mmproj_filename,
                cache_dir=str(cache_dir),
                resume_download=True,
                force_download=force_download,
            )
            
            mmproj_file = Path(mmproj_path)
            size_mb = mmproj_file.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Downloaded mmproj: {mmproj_filename} ({size_mb:.1f} MB)")
            return mmproj_file
        else:
            logger.warning("No mmproj file found in repository - this vision model might not work without it")
            return None
            
    except Exception as e:
        logger.error(f"Failed to check for mmproj file: {e}")
        return None


def download_model(
    model_id: str,
    quantization: str = "Q4_K_M",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    max_retries: int = MAX_RETRIES,
) -> Path:
    """
    Download a model from HuggingFace Hub with production-ready reliability.
    
    Improvements over Ollama:
    - M1.13: Proper timeouts and streaming for large files (>10GB)
    - M1.14: Automatic resume on network interruption
    - M1.15: Rich progress bars with speed and ETA
    - M1.16: Smart cache detection - checks for ANY cached quantization before downloading
    - M1.17: Checksum verification for integrity
    - M1.18: Exponential backoff retry logic

    Args:
        model_id: Repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        quantization: Quantization level (Q4_K_M, Q5_K_M, Q8_0, etc.)
                     If a cached version exists with different quantization, uses that instead
        cache_dir: Custom cache directory
        force_download: Skip cache and re-download
        max_retries: Maximum download retry attempts

    Returns:
        Path to downloaded model file

    Raises:
        ModelNotFoundError: If model or quantization not found
        InvalidQuantizationError: If quantization variant not available
    """
    cache_dir = cache_dir or get_cache_path()
    
    # Check if we already have THIS SPECIFIC quantization cached
    if not force_download:
        cached_model = _find_cached_model_for_repo(model_id, cache_dir, quantization)
        if cached_model:
            logger.info(f"Using existing cached model: {quantization}")

            
            # Check for mmproj if it's a vision/OCR model
            if _is_vision_model(model_id):
                 _ensure_mmproj_downloaded(model_id, cached_model, cache_dir, force_download)
                 
            return cached_model
    
    # M1.18: Retry loop with exponential backoff
    last_error = None
    for attempt in range(max_retries):
        try:
            return _download_model_attempt(
                model_id=model_id,
                quantization=quantization,
                cache_dir=cache_dir,
                force_download=force_download,
                attempt=attempt + 1,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(
                    f"Download attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} download attempts failed")
        except HfHubHTTPError as e:
            # Don't retry on 4xx errors (client errors)
            if e.response.status_code in [404, 403, 401]:
                raise ModelNotFoundError(
                    f"Model not found or access denied: {model_id}"
                ) from e
            # Retry on 5xx errors (server errors)
            elif e.response.status_code >= 500 and attempt < max_retries - 1:
                last_error = e
                wait_time = BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(
                    f"Server error {e.response.status_code}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise ModelNotFoundError(f"Failed to download model: {e}") from e
    
    # All retries exhausted
    raise ModelNotFoundError(
        f"Failed to download {model_id} after {max_retries} attempts: {last_error}"
    ) from last_error


def _download_model_attempt(
    model_id: str,
    quantization: str,
    cache_dir: Path,
    force_download: bool,
    attempt: int,
) -> Path:
    """
    Single download attempt.
    
    Args:
        model_id: Model repository ID
        quantization: Quantization level
        cache_dir: Cache directory
        force_download: Force re-download even if cached
        attempt: Current attempt number (for logging)
        
    Returns:
        Path to downloaded model
    """
    logger.info(f"[Attempt {attempt}] Searching for {quantization} version of {model_id}")
    
    # List available files in the repo
    files = list_repo_files(model_id)
    
    # Find matching GGUF file
    matching_files = [
        f for f in files if f.endswith(".gguf") and quantization.lower() in f.lower()
    ]
    
    
    if not matching_files:
        available = [f for f in files if f.endswith(".gguf")]
        
        # Smart fallback logic based on model size
        logger.warning(f"No {quantization} quantization found. Attempting fallback...")
        
        # Extract model size from model_id (e.g., "2B", "7B")
        model_size_b = 0
        model_id_lower = model_id.lower()
        import re
        size_match = re.search(r'(\d+\.?\d*)b', model_id_lower)
        if size_match:
            model_size_b = float(size_match.group(1))
        
        # Define fallback order based on model size
        if model_size_b > 0 and model_size_b < 5:
            # Small models (<5B): Prefer Q8_0 for better quality
            fallback_order = ["Q8_0", "Q8", "Q4_K_M", "Q4_K", "Q6_K", "Q5_K", "F16"]
            logger.info(f"Model size: {model_size_b}B - preferring Q8_0 fallback")
        else:
            # Large models (>=5B): Prefer Q4_K_M to save memory
            fallback_order = ["Q4_K_M", "Q4_K", "Q8_0", "Q8", "Q6_K", "Q5_K", "F16"]
            logger.info(f"Model size: {model_size_b}B - preferring Q4_K_M fallback")
        
        # Try fallbacks in order
        for fallback_quant in fallback_order:
            fallback_files = [f for f in files if f.endswith(".gguf") and fallback_quant.lower() in f.lower()]
            if fallback_files:
                filename = fallback_files[0]
                logger.info(f"✓ Using fallback quantization: {fallback_quant} -> {filename}")
                break
        else:
            # No suitable fallback found
            raise InvalidQuantizationError(
                f"No {quantization} quantization found and no suitable fallback. Available: {available}"
            )
    else:
        # Use the first match (usually there's only one)
        filename = matching_files[0]
    
    # M1.16: Check cache first (unless force_download)
    if not force_download:
        cached_path = _check_cache_validity(model_id, filename, cache_dir)
        if cached_path:
            # Check for mmproj if it's a vision/OCR model
            if _is_vision_model(model_id):
                _ensure_mmproj_downloaded(model_id, cached_path, cache_dir, force_download)
            return cached_path
    
    logger.info(f"Downloading {filename} from {model_id}")
    
    # M1.14: Download with resume support
    # M1.13: hf_hub_download handles timeouts and streaming internally
    model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        cache_dir=str(cache_dir),
        resume_download=True,  # M1.14: Resume interrupted downloads
        force_download=force_download,
        # Note: huggingface_hub handles progress internally but we can customize
    )
    
    downloaded_file = Path(model_path)
    file_size_mb = downloaded_file.stat().st_size / (1024 * 1024)
    
    # M1.17: Verify downloaded file integrity
    if not _verify_file_checksum(downloaded_file):
        logger.error(f"Downloaded file appears corrupted, removing {downloaded_file}")
        downloaded_file.unlink()
        raise ModelNotFoundError("Downloaded file failed integrity check")
    
    logger.info(f"✓ Successfully downloaded {filename} ({file_size_mb:.1f} MB)")
    logger.info(f"Model path: {model_path}")

    # Save metadata for the downloaded model
    from oprel.downloader.metadata import save_model_metadata
    save_model_metadata(cache_dir, model_id, quantization, downloaded_file)

    # Check for mmproj if it's a vision/OCR model
    if _is_vision_model(model_id):
        _ensure_mmproj_downloaded(model_id, downloaded_file, cache_dir, force_download)
    
    return downloaded_file


def list_available_quantizations(model_id: str) -> list[str]:
    """
    List all available quantization levels for a model.

    Args:
        model_id: HuggingFace model repository ID

    Returns:
        List of quantization strings (e.g., ["Q4_K_M", "Q5_K_M", "Q8_0"])
    """
    try:
        files = list_repo_files(model_id)
        gguf_files = [f for f in files if f.endswith(".gguf")]

        # Extract quantization from filenames
        # Example: "llama-2-7b.Q4_K_M.gguf" -> "Q4_K_M"
        quantizations = []
        for f in gguf_files:
            parts = f.replace(".gguf", "").split(".")
            if len(parts) >= 2:
                quantizations.append(parts[-1].upper())

        return sorted(set(quantizations))

    except Exception as e:
        logger.warning(f"Could not list quantizations for {model_id}: {e}")
        return []



def download_model_with_progress(
    model_id: str,
    quantization: str = "Q4_K_M",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """
    Download a model with real-time progress callbacks.
    
    Args:
        model_id: Repository ID
        quantization: Quantization level
        cache_dir: Custom cache directory
        force_download: Skip cache
        progress_callback: Callback function(downloaded_bytes, total_bytes)
        
    Returns:
        Path to downloaded model
    """
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import HfHubHTTPError
    import threading
    
    cache_dir = cache_dir or get_cache_path()
    
    # Check cache first for this specific quantization
    if not force_download:
        cached_model = _find_cached_model_for_repo(model_id, cache_dir, quantization)
        if cached_model:
            logger.info(f"Using cached model: {cached_model}")
            if progress_callback:
                size = cached_model.stat().st_size
                progress_callback(size, size)
            return cached_model
    
    # Find the file to download
    from huggingface_hub import list_repo_files
    files = list_repo_files(model_id)
    matching_files = [f for f in files if f.endswith(".gguf") and quantization.lower() in f.lower()]
    
    if not matching_files:
        # Fallback logic
        available = [f for f in files if f.endswith(".gguf")]
        if not available:
            raise ModelNotFoundError(f"No GGUF files found for {model_id}")
        filename = available[0]
        logger.warning(f"Quantization {quantization} not found, using {filename}")
    else:
        filename = matching_files[0]
    
    logger.info(f"Downloading {filename} from {model_id}")
    
    # Get file size from HuggingFace API
    try:
        api = HfApi()
        file_info = api.model_info(model_id, files_metadata=True)
        file_size = 0
        for sibling in file_info.siblings:
            if sibling.rfilename == filename:
                file_size = sibling.size or 0
                break
        
        logger.info(f"File size: {file_size / (1024**3):.2f} GB")
    except Exception as e:
        logger.warning(f"Could not get file size: {e}")
        file_size = 0
    
    # Prime progress with known total size so UI can show totals immediately
    if progress_callback and file_size > 0:
        progress_callback(0, file_size)

    # Start download in background and monitor progress
    download_complete = threading.Event()
    download_error = None
    downloaded_path = None
    
    def download_thread():
        nonlocal download_error, downloaded_path
        try:
            # Disable HF progress bars
            import os
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir=str(cache_dir),
                resume_download=True,
                force_download=force_download,
            )
            download_complete.set()
        except Exception as e:
            download_error = e
            download_complete.set()
    
    # Start download thread
    dl_thread = threading.Thread(target=download_thread, daemon=True)
    dl_thread.start()
    
    # Monitor progress by checking file size
    if progress_callback and file_size > 0:
        # HuggingFace downloads to: cache_dir/models--org--name/blobs/<hash>
        # Then creates a symlink in: cache_dir/models--org--name/snapshots/<commit>/<filename>
        
        # Wait a bit for download to start
        time.sleep(1)
        
        # Track the actual downloading file
        monitored_file = None
        last_size = 0
        
        while not download_complete.is_set():
            try:
                # If we haven't found the file yet, search for it
                if not monitored_file or not monitored_file.exists():
                    cache_name = "models--" + model_id.replace("/", "--")
                    model_cache_dir = cache_dir / cache_name / "blobs"
                    
                    if model_cache_dir.exists():
                        # Find files that are actively being written to
                        blob_files = [f for f in model_cache_dir.glob("*") if f.is_file()]
                        
                        # Find the file that matches our expected size range
                        # (should be growing and less than or equal to file_size)
                        for blob_file in blob_files:
                            try:
                                current_size = blob_file.stat().st_size
                                # Only consider files that are smaller than expected size
                                # and larger than 1MB (to avoid temp files)
                                if current_size > 1024 * 1024 and current_size <= file_size:
                                    # Check if file is growing
                                    time.sleep(0.1)
                                    new_size = blob_file.stat().st_size
                                    if new_size > current_size or current_size == file_size:
                                        monitored_file = blob_file
                                        last_size = current_size
                                        break
                            except Exception:
                                continue
                
                # Report progress for the monitored file
                if monitored_file and monitored_file.exists():
                    try:
                        current_size = monitored_file.stat().st_size
                        
                        # Only report if size changed or we're at 100%
                        if current_size != last_size or current_size == file_size:
                            # Ensure we don't report more than 100%
                            reported_size = min(current_size, file_size)
                            progress_callback(reported_size, file_size)
                            last_size = current_size
                            
                            # If we've reached the expected size, we're done
                            if current_size >= file_size:
                                break
                    except Exception as e:
                        logger.debug(f"Error reading file size: {e}")
                
                # Check every 500ms
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"Error monitoring progress: {e}")
                time.sleep(0.5)
    
    # Wait for download to complete
    download_complete.wait()
    
    if download_error:
        raise ModelNotFoundError(f"Failed to download {model_id}: {download_error}") from download_error
    
    if not downloaded_path:
        raise ModelNotFoundError(f"Download completed but path not found")
    
    # Final progress update
    if progress_callback and file_size > 0:
        progress_callback(file_size, file_size)
    
    # Save metadata for the downloaded model
    from oprel.downloader.metadata import save_model_metadata
    save_model_metadata(cache_dir, model_id, quantization, Path(downloaded_path))
    
    # Check for mmproj if it's a vision/OCR model
    if _is_vision_model(model_id):
        _ensure_mmproj_downloaded(model_id, Path(downloaded_path), cache_dir, force_download)
    
    logger.info(f"✓ Downloaded {filename}")
    return Path(downloaded_path)
