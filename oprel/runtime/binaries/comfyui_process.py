"""
ComfyUI Backend - Embedded Process Manager

Manages ComfyUI as an internal subprocess, similar to llama.cpp server.
Auto-downloads, starts, and manages lifecycle.
"""

import sys
import time
import subprocess
import atexit
from pathlib import Path
from typing import Optional, Dict, Any, List

from oprel.utils.logging import get_logger
from oprel.downloader.comfyui_installer import ensure_comfyui_ready, get_comfyui_dir
from oprel.runtime.backends.base import BaseBackend

logger = get_logger(__name__)

# Global process tracking
_comfyui_process: Optional[subprocess.Popen] = None


def cleanup_comfyui():
    """Cleanup function to terminate ComfyUI on exit."""
    global _comfyui_process
    if _comfyui_process:
        logger.info("Shutting down ComfyUI process...")
        _comfyui_process.terminate()
        try:
            _comfyui_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _comfyui_process.kill()
        _comfyui_process = None


atexit.register(cleanup_comfyui)


class ComfyUIBackend(BaseBackend):
    """
    Backend for ComfyUI - manages as embedded subprocess.
    """
    
    def __init__(self, model_path: Path, config):
        """
        Initialize ComfyUI backend.
        
        Args:
            model_path: Path to checkpoint model (in ComfyUI/models/checkpoints/)
            config: Configuration object
        """
        self.model_path = model_path
        self.config = config
        self.comfyui_dir = None
        self.process = None
        self.port = 8188
        self.base_url = f"http://127.0.0.1:{self.port}"
    
    def build_command(self, port: int) -> List[str]:
        """Build command to start ComfyUI."""
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        
        # Ensure ComfyUI is installed
        self.comfyui_dir = ensure_comfyui_ready()
        
        cmd = [
            sys.executable,
            str(self.comfyui_dir / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        
        # Check if PyTorch has CUDA
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except:
            has_cuda = False
        
        if not has_cuda:
            # Force CPU mode
            cmd.append("--cpu")
            logger.warning("⚠ CUDA not available, using CPU mode (will be slower)")
            logger.info("For faster generation, install PyTorch with CUDA:")
            logger.info("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            return cmd
        
        # Add GPU optimization if available
        from oprel.telemetry.hardware import detect_gpu
        gpu_info = detect_gpu()
        
        vram_gb = gpu_info.get("vram_total_gb", 0) if gpu_info else 0
        
        # CRITICAL: Never use fp16-vae + lowvram together (causes black images)
        if vram_gb >= 8:
            # High VRAM: Use FP16 VAE for speed
            cmd.append("--fp16-vae")
            logger.info("Using FP16 VAE for faster generation")
        elif 0 < vram_gb < 8:
            # Low VRAM: Use lowvram mode WITHOUT fp16 VAE
            cmd.append("--lowvram")
            logger.info(f"Using lowvram mode ({vram_gb:.1f}GB VRAM)")
            logger.warning("⚠ Skipping FP16 VAE (conflicts with lowvram)")
        
        return cmd
    
    def get_api_format(self) -> str:
        """ComfyUI uses its own REST API format."""
        return "comfyui"
    
    def start(self, port: int = 8188) -> bool:
        """
        Start ComfyUI process.
        
        Args:
            port: Port to run on
            
        Returns:
            True if started successfully
        """
        global _comfyui_process
        
        # Check if already running
        if _comfyui_process and _comfyui_process.poll() is None:
            logger.info("ComfyUI already running")
            return True
        
        cmd = self.build_command(port)
        
        logger.info(f"Starting ComfyUI on port {port}...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.comfyui_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            _comfyui_process = self.process
            
            # Wait for server to be ready
            max_wait = 60  # 60 seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    import requests
                    response = requests.get(f"{self.base_url}/system_stats", timeout=1)
                    if response.status_code == 200:
                        logger.info(f"✓ ComfyUI started successfully on port {port}")
                        return True
                except Exception:
                    pass
                
                # Check if process died
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read() if self.process.stderr else ""
                    raise RuntimeError(f"ComfyUI process died: {stderr}")
                
                time.sleep(0.5)
            
            raise TimeoutError(f"ComfyUI did not start within {max_wait}s")
            
        except Exception as e:
            logger.error(f"Failed to start ComfyUI: {e}")
            if self.process:
                self.process.terminate()
            return False
    
    def stop(self):
        """Stop ComfyUI process."""
        if self.process:
            logger.info("Stopping ComfyUI...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


def get_comfyui_backend(checkpoint_name: str, config) -> ComfyUIBackend:
    """
    Get ComfyUI backend instance.
    
    Args:
        checkpoint_name: Name of checkpoint model
        config: Configuration object
        
    Returns:
        ComfyUIBackend instance
    """
    from oprel.downloader.comfyui_installer import get_comfyui_models_dir
    
    # Find checkpoint
    checkpoints_dir = get_comfyui_models_dir("checkpoints")
    checkpoint_path = checkpoints_dir / checkpoint_name
    
    if not checkpoint_path.exists():
        # Try with .safetensors extension
        if not checkpoint_name.endswith(".safetensors"):
            checkpoint_path = checkpoints_dir / f"{checkpoint_name}.safetensors"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_name}\n"
            f"Available checkpoints: {list_available_checkpoints()}"
        )
    
    return ComfyUIBackend(checkpoint_path, config)


def list_available_checkpoints() -> List[str]:
    """List all available checkpoints."""
    from oprel.downloader.comfyui_installer import list_installed_checkpoints
    return list_installed_checkpoints()


def is_comfyui_available() -> bool:
    """Check if ComfyUI is available (will auto-install if needed)."""
    try:
        ensure_comfyui_ready()
        return True
    except Exception as e:
        logger.error(f"ComfyUI not available: {e}")
        return False
