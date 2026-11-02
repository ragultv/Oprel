"""
Per-start log file for the Oprel server.

``start_server_log()`` routes the process's stdout and stderr (file descriptors 1 and 2, so native
llama.cpp / oprel_llama_ssd.dll output is captured too) through ``oprel.utils.log_tee``, which
echoes everything to the console and writes it to ``logs/oprel_server_<timestamp>.log``.

Log directory: ``$OPREL_LOG_DIR`` if set, else ``<repo>/logs`` when running from a source checkout,
else ``~/.cache/oprel/logs``. Set ``OPREL_SERVER_LOG=0`` to disable.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

KEEP_LOG_FILES = 30

_tee: Optional[subprocess.Popen] = None


def default_log_dir() -> Path:
    env = os.environ.get("OPREL_LOG_DIR")
    if env:
        return Path(env).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").exists() and (repo_root / "oprel").is_dir():
        return repo_root / "logs"
    return Path.home() / ".cache" / "oprel" / "logs"


def _prune(log_dir: Path) -> None:
    files = sorted(log_dir.glob("oprel_server_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[KEEP_LOG_FILES:]:
        try:
            old.unlink()
        except OSError:
            pass


def _set_std_handles(fd_out: int, fd_err: int) -> None:
    """Point the Win32 standard handles at the pipe too, for DLLs that query them lazily."""
    if os.name != "nt":
        return
    try:
        import ctypes
        import msvcrt

        kernel32 = ctypes.windll.kernel32
        kernel32.SetStdHandle(-11, msvcrt.get_osfhandle(fd_out))  # STD_OUTPUT_HANDLE
        kernel32.SetStdHandle(-12, msvcrt.get_osfhandle(fd_err))  # STD_ERROR_HANDLE
    except Exception:
        pass


def start_server_log() -> Optional[Path]:
    """Start capturing all server output into a new log file. Returns its path, or None if disabled."""
    global _tee
    if _tee is not None or os.environ.get("OPREL_SERVER_LOG", "1") == "0":
        return None

    log_dir = default_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _prune(log_dir)
    except OSError:
        return None
    log_file = log_dir / f"oprel_server_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass

    try:
        _tee = subprocess.Popen(
            [sys.executable, "-u", "-m", "oprel.utils.log_tee", str(log_file)],
            stdin=subprocess.PIPE,
            close_fds=True,
        )
    except OSError:
        return None

    pipe_fd = _tee.stdin.fileno()
    old_streams = {id(sys.stdout), id(sys.stderr), id(sys.__stdout__), id(sys.__stderr__)}

    # From here on, anything written to fd 1 or 2 (Python or C) goes to the tee process.
    os.dup2(pipe_fd, 1)
    os.dup2(pipe_fd, 2)
    _set_std_handles(1, 2)

    # The console stream objects write to the console directly on Windows, bypassing fd 1/2,
    # so replace them with plain writers on the redirected descriptors.
    new_out = io.TextIOWrapper(
        io.FileIO(1, "w", closefd=False), encoding="utf-8", errors="replace", line_buffering=True, write_through=True
    )
    new_err = io.TextIOWrapper(
        io.FileIO(2, "w", closefd=False), encoding="utf-8", errors="replace", line_buffering=True, write_through=True
    )
    sys.stdout, sys.stderr = new_out, new_err

    # Loggers created before this point captured the old stream objects.
    loggers = [logging.getLogger()] + [
        lg for lg in logging.Logger.manager.loggerDict.values() if isinstance(lg, logging.Logger)
    ]
    for lg in loggers:
        for handler in lg.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                if id(handler.stream) in old_streams:
                    handler.setStream(new_err)

    # On a native crash (access violation, abort) dump every thread's Python stack into the log.
    import faulthandler

    faulthandler.enable(file=new_err, all_threads=True)
    return log_file
