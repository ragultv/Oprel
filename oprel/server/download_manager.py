"""
Download Manager - Tracks model download progress globally
"""

import asyncio
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class DownloadProgress:
    """Track progress for a single download"""
    model_id: str
    quantization: str
    status: str = "pending"  # pending, downloading, completed, error
    progress: float = 0.0  # 0-100
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0  # bytes per second
    eta_seconds: float = 0.0
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

class DownloadManager:
    """Global download manager with thread-safe progress tracking"""
    
    def __init__(self):
        self._downloads: Dict[str, DownloadProgress] = {}
        self._lock = Lock()
        self._callbacks: Dict[str, list] = {}  # download_id -> [callback functions]
    
    def start_download(self, download_id: str, model_id: str, quantization: str, total_bytes: int = 0):
        """Register a new download"""
        with self._lock:
            self._downloads[download_id] = DownloadProgress(
                model_id=model_id,
                quantization=quantization,
                status="downloading",
                total_bytes=total_bytes
            )
    
    def update_progress(self, download_id: str, downloaded_bytes: int, total_bytes: int = None):
        """Update download progress"""
        with self._lock:
            if download_id not in self._downloads:
                return
            
            download = self._downloads[download_id]
            now = time.time()
            
            # Update bytes
            if total_bytes:
                download.total_bytes = total_bytes
            download.downloaded_bytes = downloaded_bytes
            
            # Calculate progress
            if download.total_bytes > 0:
                download.progress = (downloaded_bytes / download.total_bytes) * 100
            
            # Calculate speed (bytes per second)
            # FIX: Use _last_bytes (snapshot of previous update) not downloaded_bytes
            # (which was just overwritten above). Previously this computed
            # bytes_diff = downloaded_bytes - downloaded_bytes = 0 on every call.
            time_diff = now - download.last_update
            if time_diff > 0:
                prev_bytes = getattr(download, '_last_bytes', downloaded_bytes)
                bytes_diff = downloaded_bytes - prev_bytes
                if bytes_diff >= 0:
                    download.speed_bps = bytes_diff / time_diff
                
                # Calculate ETA
                if download.speed_bps > 0 and download.total_bytes > 0:
                    remaining_bytes = download.total_bytes - downloaded_bytes
                    download.eta_seconds = remaining_bytes / download.speed_bps
            
            download.last_update = now
            download._last_bytes = downloaded_bytes

            
            # Trigger callbacks
            if download_id in self._callbacks:
                for callback in self._callbacks[download_id]:
                    try:
                        callback(download)
                    except Exception:
                        pass
    
    def complete_download(self, download_id: str):
        """Mark download as completed"""
        with self._lock:
            if download_id in self._downloads:
                self._downloads[download_id].status = "completed"
                self._downloads[download_id].progress = 100.0
    
    def fail_download(self, download_id: str, error: str):
        """Mark download as failed"""
        with self._lock:
            if download_id in self._downloads:
                self._downloads[download_id].status = "error"
                self._downloads[download_id].error = error
    
    def get_progress(self, download_id: str) -> Optional[DownloadProgress]:
        """Get progress for a specific download"""
        with self._lock:
            return self._downloads.get(download_id)
    
    def get_all_downloads(self) -> Dict[str, DownloadProgress]:
        """Get all downloads"""
        with self._lock:
            return self._downloads.copy()
    
    def remove_download(self, download_id: str):
        """Remove a download from tracking"""
        with self._lock:
            self._downloads.pop(download_id, None)
            self._callbacks.pop(download_id, None)
    
    def pause_download(self, download_id: str):
        """Mark download as paused"""
        with self._lock:
            if download_id in self._downloads:
                self._downloads[download_id].status = "paused"
                self._downloads[download_id].speed_bps = 0.0

    def cancel_download(self, download_id: str):
        """Mark download as cancelled"""
        with self._lock:
            if download_id in self._downloads:
                self._downloads[download_id].status = "cancelled"

    def add_callback(self, download_id: str, callback: Callable):

        """Add a callback for progress updates"""
        with self._lock:
            if download_id not in self._callbacks:
                self._callbacks[download_id] = []
            self._callbacks[download_id].append(callback)

# Global instance
download_manager = DownloadManager()
