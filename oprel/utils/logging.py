"""
Structured logging utilities
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Global log level (can be changed via config)
_LOG_LEVEL = logging.INFO

# Color codes for terminal output
_COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",  # Reset
}


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with colored output for terminals.
    Falls back to plain text if not a TTY.
    """

    def __init__(self, use_color: bool = True):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors"""
        if self.use_color:
            levelname = record.levelname
            color = _COLORS.get(levelname, _COLORS["RESET"])
            record.levelname = f"{color}{levelname}{_COLORS['RESET']}"

        return super().format(record)


def _attach_redaction(handler: logging.Handler) -> None:
    """Scrub credentials from every record this handler emits."""
    try:
        from oprel.utils.redaction import attach_filter

        attach_filter(handler)
    except Exception:
        pass


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (usually __name__ of the module)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if this logger hasn't been set up yet
    if not logger.handlers:
        logger.setLevel(_LOG_LEVEL)

        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(_LOG_LEVEL)
        console_handler.setFormatter(ColoredFormatter(use_color=True))
        _attach_redaction(console_handler)
        logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def set_log_level(level: str) -> None:
    """
    Set global log level for all Oprel loggers.

    Args:
        level: "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL"
    """
    global _LOG_LEVEL

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    _LOG_LEVEL = level_map.get(level.upper(), logging.INFO)

    # Update all existing oprel loggers
    for name in logging.Logger.manager.loggerDict:
        if name.startswith("oprel"):
            logger = logging.getLogger(name)
            logger.setLevel(_LOG_LEVEL)
            for handler in logger.handlers:
                handler.setLevel(_LOG_LEVEL)


def enable_file_logging(log_file: Optional[Path] = None) -> None:
    """
    Enable logging to file in addition to console.

    Args:
        log_file: Path to log file (default: ~/.cache/oprel/logs/oprel.log)
    """
    if log_file is None:
        log_dir = Path.home() / ".cache" / "oprel" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"oprel_{datetime.now().strftime('%Y%m%d')}.log"

    # Add file handler to all oprel loggers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(_LOG_LEVEL)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    _attach_redaction(file_handler)

    for name in logging.Logger.manager.loggerDict:
        if name.startswith("oprel"):
            logger = logging.getLogger(name)
            logger.addHandler(file_handler)

    get_logger(__name__).info(f"File logging enabled: {log_file}")


def disable_logging() -> None:
    """
    Disable all Oprel logging (useful for library users who want quiet mode).
    """
    logging.getLogger("oprel").setLevel(logging.CRITICAL + 1)


# Convenience functions for quick logging without getting a logger
def debug(msg: str) -> None:
    """Log debug message"""
    get_logger("oprel").debug(msg)


def info(msg: str) -> None:
    """Log info message"""
    get_logger("oprel").info(msg)


def warning(msg: str) -> None:
    """Log warning message"""
    get_logger("oprel").warning(msg)


def error(msg: str) -> None:
    """Log error message"""
    get_logger("oprel").error(msg)


def critical(msg: str) -> None:
    """Log critical message"""
    get_logger("oprel").critical(msg)
