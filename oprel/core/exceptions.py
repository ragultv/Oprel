"""
Custom exceptions for Oprel SDK

Production-ready exception hierarchy with clear error categories.
"""


class OprelError(Exception):
    """Base exception for all Oprel errors"""
    pass


class ModelNotFoundError(OprelError):
    """Raised when a model cannot be found or downloaded"""
    pass


class MemoryError(OprelError):
    """
    Raised when model exceeds memory limit.
    Unlike system MemoryError, this is caught gracefully.
    """
    pass


class BackendError(OprelError):
    """Raised when backend process fails to start or crashes"""
    pass


class BinaryNotFoundError(OprelError):
    """Raised when required binary is missing and cannot be downloaded"""
    pass


class UnsupportedPlatformError(OprelError):
    """Raised when running on unsupported OS/architecture"""
    pass


class InvalidQuantizationError(OprelError):
    """Raised when requested quantization is not available"""
    pass


# New exceptions for Week 1 features

class TimeoutError(OprelError):
    """
    Raised when an operation times out (M1.11).
    
    Attributes:
        operation: What operation timed out
        timeout_sec: The timeout value that was exceeded
    """
    def __init__(self, message: str, operation: str = "", timeout_sec: float = 0):
        super().__init__(message)
        self.operation = operation
        self.timeout_sec = timeout_sec


class CudaError(OprelError):
    """
    Raised when a CUDA-specific error occurs (M1.3).
    
    Attributes:
        exit_code: The process exit code
        cuda_error_code: The CUDA error code (if known)
    """
    def __init__(self, message: str, exit_code: int = -1, cuda_error_code: int = -1):
        super().__init__(message)
        self.exit_code = exit_code
        self.cuda_error_code = cuda_error_code


class ProcessCrashedError(BackendError):
    """
    Raised when the backend process crashes unexpectedly.
    
    Attributes:
        exit_code: The process exit code
        restart_attempted: Whether a restart was attempted
    """
    def __init__(self, message: str, exit_code: int = -1, restart_attempted: bool = False):
        super().__init__(message)
        self.exit_code = exit_code
        self.restart_attempted = restart_attempted


class VRAMError(MemoryError):
    """
    Raised when VRAM-specific issues occur (M1.4).
    
    Attributes:
        vram_required_mb: How much VRAM is needed
        vram_available_mb: How much VRAM is available
    """
    def __init__(self, message: str, vram_required_mb: float = 0, vram_available_mb: float = 0):
        super().__init__(message)
        self.vram_required_mb = vram_required_mb
        self.vram_available_mb = vram_available_mb


class DownloadPausedException(OprelError):
    """Raised when a download is paused by the user"""
    pass


class DownloadCancelledException(OprelError):
    """Raised when a download is cancelled by the user"""
    pass

