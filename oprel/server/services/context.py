from __future__ import annotations

import inspect
import os
import threading
from functools import wraps
from pathlib import Path

from oprel.core.config import Config
from oprel.utils.logging import get_logger

CONFIG = Config()
CONFIG.ensure_dirs()

logger = get_logger(__name__)

BACKEND_RUNTIME_LOCK = threading.RLock()

PID_FILE = CONFIG.cache_dir / "daemon.pid"
BACKEND_PIDS_FILE = CONFIG.cache_dir / "backend_pids.txt"


def write_daemon_pid() -> None:
    try:
        PID_FILE.write_text(str(os.getpid()))
    except Exception as exc:
        logger.debug(f"Could not write PID file: {exc}")


def remove_daemon_pid() -> None:
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass


def track_backend_pid(pid: int) -> None:
    try:
        existing = set()
        if BACKEND_PIDS_FILE.exists():
            existing = set(int(p) for p in BACKEND_PIDS_FILE.read_text().strip().split('\n') if p.strip())
        existing.add(pid)
        BACKEND_PIDS_FILE.write_text('\n'.join(str(p) for p in existing))
    except Exception as exc:
        logger.debug(f"Could not track backend PID {pid}: {exc}")


def untrack_backend_pid(pid: int) -> None:
    try:
        if not BACKEND_PIDS_FILE.exists():
            return
        existing = set(int(p) for p in BACKEND_PIDS_FILE.read_text().strip().split('\n') if p.strip())
        existing.discard(pid)
        if existing:
            BACKEND_PIDS_FILE.write_text('\n'.join(str(p) for p in existing))
        else:
            BACKEND_PIDS_FILE.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug(f"Could not untrack backend PID {pid}: {exc}")


def kill_orphaned_backends() -> None:
    import psutil

    killed = 0

    try:
        if BACKEND_PIDS_FILE.exists():
            pids = [int(p) for p in BACKEND_PIDS_FILE.read_text().strip().split('\n') if p.strip()]
            for pid in pids:
                try:
                    proc = psutil.Process(pid)
                    proc_name = proc.name().lower()
                    if "oprel-backend" in proc_name or "llama" in proc_name:
                        logger.info(f"Killing orphaned backend (PID: {pid})")
                        proc.kill()
                        proc.wait(timeout=3)
                        killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
            BACKEND_PIDS_FILE.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug(f"Error cleaning tracked PIDs: {exc}")

    try:
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                proc_name = (proc.info.get("name", "") or "").lower()
                if "oprel-backend" in proc_name:
                    logger.info(f"Killing orphaned backend process: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
                    proc.wait(timeout=3)
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
    except Exception as exc:
        logger.debug(f"Error scanning for orphaned backends: {exc}")

    if killed > 0:
        logger.info(f"Cleaned up {killed} orphaned backend process(es)")


def synchronized_backend_operation(func):
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with BACKEND_RUNTIME_LOCK:
                return await func(*args, **kwargs)

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        with BACKEND_RUNTIME_LOCK:
            return func(*args, **kwargs)

    return sync_wrapper
