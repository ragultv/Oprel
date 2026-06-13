"""
Process health monitoring and OOM protection
"""

import threading
import time
import subprocess
from typing import Optional

import psutil

from oprel.core.exceptions import MemoryError as OprelMemoryError, BackendError
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class ProcessMonitor:
    """
    Monitors a subprocess for memory usage and crashes.
    Provides OOM protection by killing the process before system freezes.
    """

    def __init__(
        self,
        process: subprocess.Popen,
        max_memory_mb: int,
        check_interval: float = 1.0,
    ):
        """
        Args:
            process: The subprocess to monitor
            max_memory_mb: Maximum memory in MB before killing (should be auto-calculated)
            check_interval: How often to check (seconds)
        """
        self.process = process
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval

        self._ps_process: Optional[psutil.Process] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_error: Optional[Exception] = None

    def start(self) -> None:
        """Start monitoring in a background thread"""
        try:
            self._ps_process = psutil.Process(self.process.pid)
        except psutil.NoSuchProcess:
            raise BackendError("Process does not exist")

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info(f"Started monitoring PID {self.process.pid}")

    def stop(self) -> None:
        """Stop monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

    def check_health(self) -> Optional[Exception]:
        """
        Check if the process is healthy.

        Returns:
            None if healthy, Exception if there's a problem
        """
        return self._last_error

    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread)"""
        while not self._stop_event.is_set():
            try:
                # Check if process still exists
                if not self._ps_process.is_running():
                    self._last_error = BackendError(
                        f"Process exited unexpectedly (exit code: {self.process.poll()})"
                    )
                    break

                # Check memory usage
                mem_info = self._ps_process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)

                if mem_info.rss > self.max_memory_bytes:
                    logger.warning(
                        f"Process exceeded memory limit: {rss_mb:.1f}MB > {self.max_memory_mb}MB"
                    )

                    # Kill the process
                    self.process.kill()
                    self.process.wait()

                    # Store error for next health check
                    self._last_error = OprelMemoryError(
                        f"Model exceeded {self.max_memory_mb}MB memory limit (used {rss_mb:.1f}MB). "
                        f"Try a smaller quantization (Q4_K_M instead of Q8_0) or increase max_memory_mb."
                    )
                    break

                # Log periodic stats (every 10 checks)
                if int(time.time()) % 10 == 0:
                    logger.debug(f"Process health: {rss_mb:.1f}MB / {self.max_memory_mb}MB")

            except psutil.NoSuchProcess:
                self._last_error = BackendError("Process disappeared")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                self._last_error = BackendError(f"Monitor failed: {e}")
                break

            time.sleep(self.check_interval)

        logger.info("Monitoring stopped")
