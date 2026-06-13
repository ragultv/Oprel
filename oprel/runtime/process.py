"""
Subprocess management for model backends

Production-Ready Features (Week 1 Tasks):
- M1.7: Zombie process prevention with atexit handlers
- M1.8: Automatic backend restart on crash (optional)
- M1.9: Proper port release with SO_REUSEADDR on Windows
- M1.10: Startup lock to prevent race conditions
- M1.11: Configurable request timeouts

Key Design Decisions:
- All cleanup registered with atexit for reliability
- Port binding test with SO_REUSEADDR for immediate reuse
- Thread-safe startup with locks
- Health check with configurable timeout and retries
"""

import atexit
import os
import platform
import socket
import subprocess
import threading
import time
import weakref
from pathlib import Path
from typing import Optional, Set, Callable

from oprel.core.config import Config
from oprel.core.exceptions import BackendError
from oprel.runtime.backends.base import BaseBackend
from oprel.runtime.backends.llama_cpp import LlamaCppBackend
from oprel.runtime.binaries.installer import ensure_binary
from oprel.runtime.cuda_errors import CudaErrorHandler, translate_exit_code, attempt_cuda_device_reset
from oprel.telemetry.hardware import detect_gpu, VRAMMonitor, check_vram_for_model
from oprel.utils.logging import get_logger

logger = get_logger(__name__)




# Track all active processes for cleanup on exit
_active_processes: Set[weakref.ref] = set()
_process_lock = threading.Lock()
_cleanup_registered = False


def _register_process(process: 'ModelProcess') -> None:
    """Register a process for cleanup on exit."""
    global _cleanup_registered
    
    with _process_lock:
        # Use weak reference so processes can be garbage collected
        _active_processes.add(weakref.ref(process, _unregister_process_ref))
        
        # Register atexit handler only once
        if not _cleanup_registered:
            atexit.register(_cleanup_all_processes)
            _cleanup_registered = True


def _unregister_process_ref(ref: weakref.ref) -> None:
    """Callback when a process is garbage collected."""
    with _process_lock:
        _active_processes.discard(ref)


def _cleanup_all_processes() -> None:
    """Cleanup handler called on interpreter exit."""
    with _process_lock:
        processes = list(_active_processes)
    
    for ref in processes:
        process = ref()
        if process is not None:
            try:
                process.stop(force=True)
            except Exception as e:
                # Log but don't raise during cleanup
                print(f"[oprel] Warning: Failed to cleanup process: {e}")




def _find_free_port(port_range: tuple[int, int]) -> int:
    """
    Find an available port with SO_REUSEADDR for immediate reuse.
    
    Windows-specific: Uses SO_REUSEADDR to allow binding to TIME_WAIT ports.
    This prevents "Address already in use" after crashes.
    
    Args:
        port_range: Tuple of (start_port, end_port)
        
    Returns:
        Available port number
        
    Raises:
        BackendError: If no port available
    """
    start, end = port_range
    
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # SO_REUSEADDR allows binding to TIME_WAIT sockets
                # Critical for Windows where ports can be stuck for minutes
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                
                # On Windows, also try SO_EXCLUSIVEADDRUSE to prevent stealing
                if platform.system() == "Windows":
                    try:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    except (AttributeError, OSError):
                        pass  # Not available on all Windows versions
                
                s.bind(("127.0.0.1", port))
                
                # Verify we can actually use this port
                s.listen(1)
                return port
                
        except OSError:
            continue
    
    raise BackendError(f"No free ports in range {start}-{end}")


def _force_release_port(port: int) -> bool:
    """
    Try to forcefully release a port (Windows-specific).
    
    This is a best-effort attempt and may not always work.
    
    Args:
        port: Port number to release
        
    Returns:
        True if release attempted, False otherwise
    """
    if platform.system() != "Windows":
        return False
    
    try:
        import psutil
        
        # Find process using this port
        for conn in psutil.net_connections(kind='tcp'):
            if conn.laddr.port == port:
                try:
                    proc = psutil.Process(conn.pid)
                    proc_name = proc.name()
                    
                    # Only kill our own processes (oprel-backend or llama-server)
                    if "oprel" in proc_name.lower() or "llama" in proc_name.lower():
                        logger.warning(f"Killing stuck process {conn.pid} ({proc_name}) on port {port}")
                        proc.kill()
                        time.sleep(0.5)
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to force release port {port}: {e}")
    
    return False



class ModelProcess:
    """
    Manages a model backend subprocess with production-ready features.
    
    Features:
    - Automatic cleanup on interpreter exit (atexit)
    - Port management with SO_REUSEADDR
    - Thread-safe startup with locking
    - VRAM monitoring during load
    - Detailed error messages with CUDA code translation
    - Optional auto-restart on crash
    
    Usage:
        process = ModelProcess(model_path)
        process.start()
        # ... use process.port for HTTP requests ...
        process.stop()
    """

    # Class-level lock for preventing race conditions (M1.10)
    _startup_lock = threading.Lock()
    
    def __init__(
        self,
        model_path: Path,
        backend: str = "llama.cpp",
        config: Optional[Config] = None,
        auto_restart: bool = False,
        max_restarts: int = 3,
        restart_delay_sec: float = 2.0,
        on_crash: Optional[Callable[['ModelProcess', int], None]] = None,
    ):
        """
        Initialize a model process manager.
        
        Args:
            model_path: Path to the model file (.gguf)
            backend: Backend type ("llama.cpp")
            config: Optional configuration
            auto_restart: Whether to auto-restart on crash (M1.8)
            max_restarts: Maximum restart attempts
            restart_delay_sec: Delay between restart attempts
            on_crash: Callback when process crashes (receives self and exit_code)
        """
        self.model_path = Path(model_path)
        self.backend_name = backend
        self.config = config or Config()
        
        # Auto-restart settings (M1.8)
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.restart_delay_sec = restart_delay_sec
        self.on_crash = on_crash
        self._restart_count = 0
        
        # Runtime state
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.socket_path: Optional[Path] = None
        self._backend: Optional[BaseBackend] = None
        self._started = False
        self._stopping = False
        
        # Watchdog thread for auto-restart
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        
        # VRAM monitor
        self._vram_monitor: Optional[VRAMMonitor] = None
        # Track whether we've already attempted a CPU fallback to avoid loops
        self._attempted_cpu_fallback: bool = False
        
        # Register for cleanup
        _register_process(self)

    def _log_model_info(self) -> None:
        """Log model information with size and estimated requirements."""
        try:
            model_size_bytes = self.model_path.stat().st_size
            model_size_gb = model_size_bytes / (1024**3)
            logger.info(f"Loading model: {self.model_path.name} ({model_size_gb:.2f} GB)")
            
            # Check VRAM availability
            gpu_info = detect_gpu()
            if gpu_info and gpu_info.get("gpu_type") in ("cuda", "metal", "rocm"):
                vram_check = check_vram_for_model(
                    model_size_gb=model_size_gb,
                    gpu_layers=self.config.n_gpu_layers if self.config.n_gpu_layers > 0 else 32,
                )
                if vram_check.get("warning"):
                    logger.warning(vram_check["message"])
                else:
                    logger.debug(vram_check["message"])
                    
        except Exception as e:
            logger.debug(f"Could not log model info: {e}")

    def start(self, timeout: int = 60) -> None:
        """
        Start the model backend process.
        
        Thread-safe with startup lock to prevent race conditions (M1.10).
        
        Args:
            timeout: Maximum seconds to wait for startup
            
        Raises:
            BackendError: If process fails to start
        """
        # Thread-safe startup (M1.10)
        with self._startup_lock:
            if self._started and self.is_running():
                logger.debug("Process already running")
                return
            
            self._do_start(timeout)
            self._started = True
        
        # Start watchdog if auto-restart enabled
        if self.auto_restart:
            self._start_watchdog()

    def _do_start(self, timeout: int) -> None:
        """Internal start implementation."""
        # Ensure backend binary is installed
        binary_path = ensure_binary(
            backend=self.backend_name,
            version=self.config.binary_version,
            binary_dir=self.config.binary_dir,
            config=self.config,
        )

        # Select backend implementation
        if self.backend_name == "llama.cpp":
            self._backend = LlamaCppBackend(
                binary_path=binary_path,
                model_path=self.model_path,
                config=self.config,
            )
        else:
            raise BackendError(f"Unsupported backend: {self.backend_name}")


        # We'll attempt startup, and if GPU initialization fails we'll optionally
        # retry once with a CPU-only fallback (M1.2).
        attempt = 0
        max_attempts = 2
        last_exception = None

        while attempt < max_attempts:
            attempt += 1

            # Log model information
            self._log_model_info()

            # Find available port with SO_REUSEADDR (M1.9)
            self.port = _find_free_port(self.config.default_port_range)
            logger.debug(f"Selected port: {self.port}")

            # Build command
            cmd = self._backend.build_command(port=self.port)
            logger.debug(f"Starting process (attempt {attempt}): {' '.join(cmd)}")

            # Start VRAM monitoring during load (M1.4) only for CUDA
            gpu_info = detect_gpu()
            if gpu_info and gpu_info.get("gpu_type") == "cuda":
                self._vram_monitor = VRAMMonitor(
                    on_warning=lambda s: logger.warning(
                        f"VRAM high: {s.used_mb:.0f}/{s.total_mb:.0f}MB ({s.utilization_percent:.1f}%)"
                    ),
                    on_critical=lambda s: logger.error(
                        f"VRAM critical: {s.used_mb:.0f}/{s.total_mb:.0f}MB - may cause OOM!"
                    ),
                )
                self._vram_monitor.start()

            # Spawn process
            try:
                env = os.environ.copy()

                # Linux: Set LD_LIBRARY_PATH for shared libraries
                if platform.system() == "Linux":
                    binary_dir = str(self.config.binary_dir)
                    existing_path = env.get("LD_LIBRARY_PATH", "")
                    env["LD_LIBRARY_PATH"] = f"{binary_dir}:{existing_path}" if existing_path else binary_dir
                    logger.debug(f"Set LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")

                # Windows: Hide console window completely
                creation_flags = 0
                if platform.system() == "Windows":
                    DETACHED_PROCESS = 0x00000008
                    CREATE_NEW_PROCESS_GROUP = 0x00000200
                    creation_flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    creationflags=creation_flags,
                )

                logger.debug(f"Process spawned with PID: {self.process.pid}")

            except Exception as e:
                self._stop_vram_monitor()
                last_exception = BackendError(f"Failed to start process: {e}")
                logger.error(last_exception)
                # If this was a failure and we haven't yet tried CPU fallback, continue to retry
                # after setting CPU-only flags below.
                # Fall through to retry logic.

            # Wait for server to be ready
            ready = False
            try:
                ready = self._wait_for_ready(timeout=timeout)
            except Exception as e:
                logger.debug(f"_wait_for_ready raised: {e}")

            if ready:
                # Stop VRAM monitoring after successful load
                if self._vram_monitor:
                    peak_vram = self._vram_monitor.get_peak_usage()
                    self._stop_vram_monitor()
                    logger.info(f"Model loaded successfully. Peak VRAM usage: {peak_vram:.0f}MB")

                # llama-server only supports HTTP, not Unix sockets
                self.socket_path = None

                logger.info(f"Backend ready (PID: {self.process.pid}, Port: {self.port})")
                return

            # Not ready: process may have exited or health check timed out
            exit_code = self.process.returncode if self.process else -1
            self._stop_vram_monitor()

            # Ensure any partially-started process is stopped before retrying
            try:
                self.stop(force=True)
            except Exception:
                pass

            # If we detect a CUDA-related failure and haven't yet attempted CPU fallback,
            # try once more with CPU-only configuration.
            try:
                # Use CudaErrorHandler heuristics to detect busy/oom issues
                stderr_text = ""
                if CudaErrorHandler.is_cuda_device_busy_error(exit_code, stderr_text) or CudaErrorHandler.is_oom_error(exit_code, stderr_text):
                    if not self._attempted_cpu_fallback and self.backend_name == "llama.cpp":
                        logger.warning("GPU initialization failed; attempting CPU fallback (disabling GPU layers)")
                        # Disable GPU layers and retry once
                        try:
                            self._attempted_cpu_fallback = True
                            self.config.n_gpu_layers = 0
                            # Recreate backend so build_command reflects new config
                            self._backend = LlamaCppBackend(
                                binary_path=binary_path,
                                model_path=self.model_path,
                                config=self.config,
                            )
                            # Continue loop to attempt restart
                            continue
                        except Exception as e:
                            last_exception = BackendError(f"CPU fallback attempt failed: {e}")
                # Otherwise, prepare to raise an error
            except Exception as e:
                last_exception = BackendError(f"Error inspecting failure: {e}")

            # If we reach here we will not retry further
            break

        # If we exit loop without returning, raise the last captured exception or a generic error
        if last_exception:
            raise last_exception
        else:
            error_msg = CudaErrorHandler.handle_process_exit(exit_code if 'exit_code' in locals() else -1)
            raise BackendError(f"Process failed to start:\n{error_msg}")

        # llama-server only supports HTTP, not Unix sockets
        self.socket_path = None

        logger.info(f"Backend ready (PID: {self.process.pid}, Port: {self.port})")

    def _stop_vram_monitor(self) -> None:
        """Stop VRAM monitoring if active."""
        if self._vram_monitor:
            self._vram_monitor.stop()
            self._vram_monitor = None

    def is_running(self) -> bool:
        """Check if the backend process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self, force: bool = False) -> None:
        """
        Gracefully stop the process and all child processes.
        
        Args:
            force: If True, skip graceful termination and kill immediately
        """
        if self._stopping:
            return
        
        self._stopping = True
        
        # Stop watchdog first
        self._stop_watchdog()
        
        # Stop VRAM monitor
        self._stop_vram_monitor()
        
        if self.process is None:
            self._stopping = False
            return
        
        pid = self.process.pid
        logger.debug(f"Stopping backend process (PID: {pid})")
        
        try:
            import psutil
            
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                
                # Terminate children first
                for child in children:
                    try:
                        child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Wait for children
                if not force:
                    _, alive = psutil.wait_procs(children, timeout=3)
                    
                    # Kill any that didn't terminate
                    for child in alive:
                        try:
                            child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                else:
                    # Force kill all children
                    for child in children:
                        try:
                            child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        except ImportError:
            pass
        
        # Terminate main process
        if not force:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate, killing...")
                self.process.kill()
                self.process.wait()
        else:
            self.process.kill()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass  # Best effort
        
        self.process = None
        self._started = False
        self._stopping = False
        
        # After stopping, if a CUDA-capable GPU is present, attempt a device reset
        try:
            gpu_info = detect_gpu()
            if gpu_info and gpu_info.get("gpu_type") == "cuda":
                try:
                    if attempt_cuda_device_reset():
                        logger.info("Attempted CUDA device reset after backend stop")
                except Exception as e:
                    logger.debug(f"CUDA device reset attempt failed: {e}")
        except Exception:
            pass

        logger.debug(f"Backend process stopped (was PID: {pid})")

    def _wait_for_ready(self, timeout: int = 60) -> bool:
        """
        Wait for the backend server to be ready.
        
        Uses exponential backoff for efficient polling.
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if ready, False if timeout or crash
        """
        import requests

        start = time.time()
        health_url = f"http://127.0.0.1:{self.port}/health"
        
        # Exponential backoff: start fast, slow down
        poll_interval = 0.1
        max_interval = 2.0

        logger.debug(f"Waiting for model to be ready (timeout: {timeout}s)...")

        while time.time() - start < timeout:
            # Check if process crashed
            if self.process and self.process.poll() is not None:
                exit_code = self.process.returncode
                title, _ = translate_exit_code(exit_code)
                logger.error(f"Process exited with code {exit_code}: {title}")
                return False

            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok":
                        elapsed = time.time() - start
                        logger.debug(f"Model backend ready in {elapsed:.1f}s")
                        return True
                    else:
                        logger.debug(f"Health check response: {data}")
            except requests.exceptions.ConnectionError:
                pass  # Server not listening yet
            except requests.exceptions.Timeout:
                pass  # Server slow
            except Exception as e:
                logger.debug(f"Health check error: {e}")

            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, max_interval)

        logger.error(f"Server failed to become ready within {timeout}s")
        return False


    def _start_watchdog(self) -> None:
        """Start the watchdog thread for auto-restart."""
        if self._watchdog_thread is not None:
            return
        
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="oprel-watchdog"
        )
        self._watchdog_thread.start()
        logger.debug("Watchdog thread started")

    def _stop_watchdog(self) -> None:
        """Stop the watchdog thread."""
        if self._watchdog_thread is None:
            return
        
        self._watchdog_stop.set()
        self._watchdog_thread.join(timeout=2)
        self._watchdog_thread = None
        logger.debug("Watchdog thread stopped")

    def _watchdog_loop(self) -> None:
        """Watchdog thread main loop - monitors process and restarts on crash."""
        while not self._watchdog_stop.is_set():
            if self.process is not None and self.process.poll() is not None:
                # Process crashed!
                exit_code = self.process.returncode
                logger.error(f"Backend process crashed with exit code {exit_code}")
                
                # Call crash callback
                if self.on_crash:
                    try:
                        self.on_crash(self, exit_code)
                    except Exception as e:
                        logger.error(f"Crash callback error: {e}")
                
                # Attempt restart
                if self._restart_count < self.max_restarts:
                    self._restart_count += 1
                    logger.info(
                        f"Attempting restart {self._restart_count}/{self.max_restarts} "
                        f"in {self.restart_delay_sec}s..."
                    )
                    
                    time.sleep(self.restart_delay_sec)
                    
                    if self._watchdog_stop.is_set():
                        break
                    
                    try:
                        self.process = None
                        self._started = False
                        self._do_start(timeout=60)
                        self._started = True
                        logger.info("Backend restarted successfully")
                    except Exception as e:
                        logger.error(f"Restart failed: {e}")
                else:
                    logger.error(f"Max restarts ({self.max_restarts}) exceeded, giving up")
                    break
            
            self._watchdog_stop.wait(timeout=1.0)

    def health_check(self, timeout: float = 5.0) -> bool:
        """
        Check if the backend is healthy and responsive.
        
        Args:
            timeout: Request timeout in seconds (M1.11)
            
        Returns:
            True if healthy, False otherwise
        """
        if not self.is_running():
            return False
        
        try:
            import requests
            response = requests.get(
                f"http://127.0.0.1:{self.port}/health",
                timeout=timeout
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "ok"
        except Exception:
            pass
        
        return False

    def get_process_info(self) -> dict:
        """Get information about the running process."""
        info = {
            "running": self.is_running(),
            "port": self.port,
            "pid": self.process.pid if self.process else None,
            "restart_count": self._restart_count,
            "backend": self.backend_name,
            "model": str(self.model_path),
        }
        
        # Add memory usage if available
        if self.process:
            try:
                import psutil
                proc = psutil.Process(self.process.pid)
                mem = proc.memory_info()
                info["memory_rss_mb"] = round(mem.rss / (1024 * 1024), 1)
            except Exception:
                pass
        
        return info

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            if hasattr(self, 'process') and self.process is not None:
                self.stop(force=True)
        except Exception:
            pass



def kill_all_oprel_processes() -> int:
    """
    Kill all oprel-related processes (cleanup utility).
    
    Useful for cleaning up after crashes.
    
    Returns:
        Number of processes killed
    """
    killed = 0
    
    try:
        import psutil
        
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                name = proc.info['name'].lower()
                cmdline = ' '.join(proc.info.get('cmdline') or []).lower()
                
                if 'oprel' in name or 'llama-server' in name:
                    proc.kill()
                    killed += 1
                elif 'llama' in cmdline and 'server' in cmdline:
                    proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
    except ImportError:
        logger.warning("psutil not available, cannot kill processes")
    
    return killed
