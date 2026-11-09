"""
CUDA Error Code Translation

Maps CUDA error codes and Windows exit codes to human-readable messages.
This helps users understand and fix GPU-related issues without searching online.

Common Scenarios:
- Device busy: Another process is using the GPU, restart PC or close other apps
- OOM: Model too large for VRAM, use smaller quantization
- Driver issues: Update NVIDIA drivers

Reference:
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
- Windows NTSTATUS codes: https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-erref
"""

from typing import Optional, Tuple
from enum import IntEnum
import ctypes
import ctypes.util
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class CudaError(IntEnum):
    """CUDA runtime error codes (cudaError_t)"""
    Success = 0
    InvalidValue = 1
    MemoryAllocation = 2
    InitializationError = 3
    CudartUnloading = 4
    ProfilerDisabled = 5
    ProfilerNotInitialized = 6
    ProfilerAlreadyStarted = 7
    ProfilerAlreadyStopped = 8
    InvalidConfiguration = 9
    InvalidPitchValue = 12
    InvalidSymbol = 13
    InvalidHostPointer = 16
    InvalidDevicePointer = 17
    InvalidTexture = 18
    InvalidTextureBinding = 19
    InvalidChannelDescriptor = 20
    InvalidMemcpyDirection = 21
    AddressOfConstant = 22
    TextureFetchFailed = 23
    TextureNotBound = 24
    SynchronizationError = 25
    InvalidFilterSetting = 26
    InvalidNormSetting = 27
    MixedDeviceExecution = 28
    NotYetImplemented = 31
    MemoryValueTooLarge = 32
    StubLibrary = 34
    InsufficientDriver = 35
    CallRequiresNewerDriver = 36
    InvalidSurface = 37
    DuplicateVariableName = 43
    DuplicateTextureName = 44
    DuplicateSurfaceName = 45
    DevicesUnavailable = 46
    IncompatibleDriverContext = 49
    MissingConfiguration = 52
    PriorLaunchFailure = 53
    LaunchMaxDepthExceeded = 65
    LaunchFileScopedTex = 66
    LaunchFileScopedSurf = 67
    SyncDepthExceeded = 68
    LaunchPendingCountExceeded = 69
    InvalidDeviceFunction = 98
    NoDevice = 100
    InvalidDevice = 101
    DeviceNotLicensed = 102
    SoftwareValidityNotEstablished = 103
    StartupFailure = 127
    InvalidKernelImage = 200
    DeviceUninitialized = 201
    MapBufferObjectFailed = 205
    UnmapBufferObjectFailed = 206
    ArrayIsMapped = 207
    AlreadyMapped = 208
    NoKernelImageForDevice = 209
    AlreadyAcquired = 210
    NotMapped = 211
    NotMappedAsArray = 212
    NotMappedAsPointer = 213
    ECCUncorrectable = 214
    UnsupportedLimit = 215
    DeviceAlreadyInUse = 216
    PeerAccessUnsupported = 217
    InvalidPtx = 218
    InvalidGraphicsContext = 219
    NvlinkUncorrectable = 220
    JitCompilerNotFound = 221
    UnsupportedPtxVersion = 222
    JitCompilationDisabled = 223
    UnsupportedExecAffinity = 224
    InvalidSource = 300
    FileNotFound = 301
    SharedObjectSymbolNotFound = 302
    SharedObjectInitFailed = 303
    OperatingSystem = 304
    InvalidResourceHandle = 400
    IllegalState = 401
    SymbolNotFound = 500
    NotReady = 600
    IllegalAddress = 700
    LaunchOutOfResources = 701
    LaunchTimeout = 702
    LaunchIncompatibleTexturing = 703
    PeerAccessAlreadyEnabled = 704
    PeerAccessNotEnabled = 705
    SetOnActiveProcess = 708
    ContextIsDestroyed = 709
    Assert = 710
    TooManyPeers = 711
    HostMemoryAlreadyRegistered = 712
    HostMemoryNotRegistered = 713
    HardwareStackError = 714
    IllegalInstruction = 715
    MisalignedAddress = 716
    InvalidAddressSpace = 717
    InvalidPc = 718
    LaunchFailure = 719
    CooperativeLaunchTooLarge = 720
    NotPermitted = 800
    NotSupported = 801
    SystemNotReady = 802
    SystemDriverMismatch = 803
    CompatNotSupportedOnDevice = 804
    MpsConnectionFailed = 805
    MpsRpcFailure = 806
    MpsServerNotReady = 807
    MpsMaxClientsReached = 808
    MpsMaxConnectionsReached = 809
    MpsClientTerminated = 810
    StreamCaptureUnsupported = 900
    StreamCaptureInvalidated = 901
    StreamCaptureMerge = 902
    StreamCaptureUnmatched = 903
    StreamCaptureUnjoined = 904
    StreamCaptureIsolation = 905
    StreamCaptureImplicit = 906
    CapturedEvent = 907
    StreamCaptureWrongThread = 908
    Timeout = 909
    GraphExecUpdateFailure = 910
    ExternalDevice = 911
    Unknown = 999


# Detailed error messages with solutions
CUDA_ERROR_MESSAGES = {
    CudaError.Success: (
        "Success",
        "Operation completed successfully."
    ),
    CudaError.MemoryAllocation: (
        "Out of Memory",
        "Not enough GPU memory (VRAM) to load the model.\n"
        "Solutions:\n"
        "  1. Use a smaller quantization: Q4_K_M instead of Q8_0\n"
        "  2. Use fewer GPU layers: --n-gpu-layers 10\n"
        "  3. Close other GPU applications (games, browsers with HW accel)\n"
        "  4. Try a smaller model (7B instead of 13B)"
    ),
    CudaError.NoDevice: (
        "No CUDA Device Found",
        "No NVIDIA GPU detected with CUDA support.\n"
        "Solutions:\n"
        "  1. Ensure you have an NVIDIA GPU (AMD uses ROCm, not CUDA)\n"
        "  2. Install/update NVIDIA drivers\n"
        "  3. Run 'nvidia-smi' to verify GPU is detected"
    ),
    CudaError.InvalidDevice: (
        "Invalid Device",
        "The specified CUDA device is not available.\n"
        "Solutions:\n"
        "  1. Check available devices with 'nvidia-smi'\n"
        "  2. Ensure CUDA_VISIBLE_DEVICES is set correctly"
    ),
    CudaError.DeviceAlreadyInUse: (
        "GPU Device Busy",
        "The GPU is currently in use by another process.\n"
        "Solutions:\n"
        "  1. Close other applications using the GPU\n"
        "  2. Use 'nvidia-smi' to check which processes are using GPU\n"
        "  3. Restart your computer to clear stuck GPU contexts"
    ),
    CudaError.DevicesUnavailable: (
        "CUDA Devices Unavailable",
        "CUDA-capable device(s) are busy or unavailable.\n"
        "This often happens when the GPU driver has a locked context.\n"
        "Solutions:\n"
        "  1. Restart your computer (most reliable)\n"
        "  2. Run 'nvidia-smi' and kill any stuck processes\n"
        "  3. Restart the NVIDIA driver service:\n"
        "     - Windows: Restart 'NVIDIA Display Container LS' service\n"
        "     - Linux: sudo systemctl restart nvidia-persistenced"
    ),
    CudaError.InsufficientDriver: (
        "Driver Too Old",
        "Your NVIDIA driver is too old for this CUDA version.\n"
        "Solutions:\n"
        "  1. Update your NVIDIA drivers from nvidia.com\n"
        "  2. Minimum driver version for CUDA 12.4: 550.54.14 (Linux) / 551.78 (Windows)"
    ),
    CudaError.IncompatibleDriverContext: (
        "Driver Context Error",
        "CUDA driver context is incompatible.\n"
        "This usually means the driver crashed or was reset.\n"
        "Solutions:\n"
        "  1. Restart your computer\n"
        "  2. Update NVIDIA drivers"
    ),
    CudaError.LaunchOutOfResources: (
        "Launch Resources Exceeded",
        "Not enough GPU resources to launch kernel.\n"
        "Solutions:\n"
        "  1. Reduce batch size\n"
        "  2. Reduce context size (--ctx-size)\n"
        "  3. Use fewer GPU layers"
    ),
    CudaError.LaunchTimeout: (
        "GPU Timeout",
        "GPU operation timed out (TDR - Timeout Detection and Recovery).\n"
        "Solutions:\n"
        "  1. Reduce model complexity\n"
        "  2. Windows: Increase TDR timeout (advanced, not recommended)\n"
        "  3. Use CPU mode for very long operations"
    ),
    CudaError.InitializationError: (
        "CUDA Initialization Failed",
        "Failed to initialize CUDA.\n"
        "Solutions:\n"
        "  1. Reinstall NVIDIA drivers\n"
        "  2. Check CUDA installation: nvcc --version\n"
        "  3. Ensure CUDA path is in environment variables"
    ),
    CudaError.NotSupported: (
        "Operation Not Supported",
        "This operation is not supported on your GPU.\n"
        "Your GPU may be too old or lack required features."
    ),
    CudaError.SystemDriverMismatch: (
        "Driver/Runtime Mismatch",
        "CUDA runtime version doesn't match driver version.\n"
        "Solutions:\n"
        "  1. Update NVIDIA drivers\n"
        "  2. Or reinstall CUDA toolkit to match driver"
    ),
    CudaError.Unknown: (
        "Unknown CUDA Error",
        "An unknown error occurred.\n"
        "Check nvidia-smi for GPU status and try restarting."
    ),
}

# Windows-specific exit codes (NTSTATUS)
WINDOWS_EXIT_CODES = {
    # Stack buffer overrun - often means CUDA context corruption
    0xC0000409: (
        "Stack Buffer Overrun (STATUS_STACK_BUFFER_OVERRUN)",
        "The process terminated due to a stack buffer overrun.\n"
        "In CUDA context, this often means:\n"
        "  1. GPU driver context is corrupted\n"
        "  2. Another process didn't release GPU properly\n"
        "Solutions:\n"
        "  1. Restart your computer (most reliable)\n"
        "  2. Update NVIDIA drivers\n"
        "  3. Close all GPU applications and try again"
    ),
    # Access violation
    0xC0000005: (
        "Access Violation",
        "Memory access violation occurred.\n"
        "Solutions:\n"
        "  1. Model file may be corrupted - re-download it\n"
        "  2. GPU memory issue - restart computer\n"
        "  3. Update NVIDIA drivers"
    ),
    # Illegal instruction
    0xC000001D: (
        "Illegal Instruction",
        "CPU executed an invalid instruction.\n"
        "This may mean the binary is built for a different CPU architecture."
    ),
    # Integer divide by zero
    0xC0000094: (
        "Integer Divide by Zero",
        "A division by zero occurred. This is a bug in the backend."
    ),
    # Heap corruption
    0xC0000374: (
        "Heap Corruption",
        "Memory heap is corrupted.\n"
        "Solutions:\n"
        "  1. Restart your computer\n"
        "  2. Check for memory issues with Windows Memory Diagnostic"
    ),
    # DLL not found
    0xC0000135: (
        "DLL Not Found",
        "Required DLL is missing.\n"
        "Solutions:\n"
        "  1. For CUDA: Ensure CUDA runtime DLLs are installed\n"
        "  2. Install Visual C++ Redistributable"
    ),
    # Application hang
    0xC0000142: (
        "Application Failed to Initialize",
        "The application failed to initialize properly.\n"
        "Solutions:\n"
        "  1. Run as administrator\n"
        "  2. Check antivirus isn't blocking the process"
    ),
}

# Common llama.cpp exit codes
LLAMA_CPP_EXIT_CODES = {
    1: (
        "General Error",
        "llama-server encountered an error.\n"
        "Check the model path and ensure the model file is valid."
    ),
    2: (
        "Invalid Arguments",
        "Invalid command-line arguments were passed to llama-server."
    ),
    3: (
        "Model Load Failed",
        "Failed to load the model file.\n"
        "Solutions:\n"
        "  1. Verify the model file exists and is not corrupted\n"
        "  2. Ensure you have enough RAM/VRAM\n"
        "  3. Try a smaller quantization"
    ),
    -1: (
        "Abnormal Termination",
        "Process was killed or terminated abnormally."
    ),
    -9: (
        "Killed by Signal 9 (SIGKILL)",
        "Process was forcefully killed.\n"
        "This usually means out of memory (OOM killer on Linux)."
    ),
    -11: (
        "Segmentation Fault (SIGSEGV)",
        "Memory access violation.\n"
        "Solutions:\n"
        "  1. Model file may be corrupted\n"
        "  2. Update llama.cpp binary\n"
        "  3. Report bug if persists"
    ),
    -15: (
        "Terminated by Signal 15 (SIGTERM)",
        "Process was terminated (normal shutdown)."
    ),
}


def translate_exit_code(exit_code: Optional[int]) -> Tuple[str, str]:
    """
    Translate a process exit code to a human-readable message.
    
    Args:
        exit_code: The exit code from subprocess (or None if timed out)
        
    Returns:
        Tuple of (title, detailed_message)
    """
    if exit_code is None:
        return (
            "Startup Timeout",
            "The backend server did not become ready within the configured timeout period.\n"
            "This can happen on large models, high context allocations, or when compiling CUDA kernels."
        )

    # Check for Windows NTSTATUS codes (negative when returned from subprocess)
    # Python subprocess returns them as negative signed integers
    if exit_code < 0:
        # Convert to unsigned 32-bit for lookup
        unsigned_code = exit_code & 0xFFFFFFFF
        if unsigned_code in WINDOWS_EXIT_CODES:
            return WINDOWS_EXIT_CODES[unsigned_code]
    
    # Check Windows codes directly (some may be returned as large positive)
    if exit_code in WINDOWS_EXIT_CODES:
        return WINDOWS_EXIT_CODES[exit_code]
    
    # Check llama.cpp specific codes
    if exit_code in LLAMA_CPP_EXIT_CODES:
        return LLAMA_CPP_EXIT_CODES[exit_code]
    
    # Convert signed exit codes (Windows returns large unsigned as negative signed)
    # Example: 3221226505 signed = -1073740791
    if exit_code > 0x7FFFFFFF:
        # This is a large unsigned value, check NTSTATUS
        if exit_code in WINDOWS_EXIT_CODES:
            return WINDOWS_EXIT_CODES[exit_code]
    
    # Common small positive codes
    if exit_code == 0:
        return ("Success", "Process exited normally.")
    
    if 1 <= exit_code <= 255:
        return (
            f"Exit Code {exit_code}",
            f"Process exited with code {exit_code}.\n"
            "This is typically an application-specific error."
        )
    
    # Unknown large code
    return (
        f"Unknown Exit Code ({exit_code})",
        f"Process exited with code {exit_code} (0x{exit_code & 0xFFFFFFFF:08X}).\n"
        "This may be a Windows exception code or CUDA error.\n"
        "Try restarting your computer or updating drivers."
    )


def translate_cuda_error(cuda_error_code: int) -> Tuple[str, str]:
    """
    Translate a CUDA error code to human-readable message.
    
    Args:
        cuda_error_code: CUDA error code (cudaError_t)
        
    Returns:
        Tuple of (title, detailed_message)
    """
    try:
        error = CudaError(cuda_error_code)
        if error in CUDA_ERROR_MESSAGES:
            return CUDA_ERROR_MESSAGES[error]
    except ValueError:
        pass
    
    return (
        f"CUDA Error {cuda_error_code}",
        f"Unknown CUDA error code: {cuda_error_code}\n"
        "Check nvidia-smi and try restarting."
    )


def get_gpu_troubleshooting_tips() -> str:
    """
    Get general GPU troubleshooting tips.
    
    Returns:
        Multi-line string with troubleshooting steps
    """
    return """
GPU Troubleshooting Guide
=========================

1. Check GPU Status:
   Run 'nvidia-smi' to see:
   - GPU utilization
   - Memory usage
   - Running processes
   - Driver version

2. Common Fixes:
   - Restart your computer (clears stuck CUDA contexts)
   - Update NVIDIA drivers
   - Close GPU-heavy apps (games, video editors, browsers)

3. If GPU Won't Work:
   - Use CPU mode: oprel run model "prompt" --n-gpu-layers 0
   - Use smaller quantization: Q4_K_M or Q4_0

4. Check CUDA Installation:
   - Run: nvidia-smi (shows driver version)
   - The llama.cpp CUDA binary needs driver >= 550.54

5. Windows-Specific:
   - Restart NVIDIA services in Services.msc
   - Check Windows Event Viewer for GPU errors
   - Disable GPU hardware acceleration in Chrome

6. Linux-Specific:
   - Check dmesg for GPU errors: dmesg | grep -i nvidia
   - Restart driver: sudo modprobe -r nvidia && sudo modprobe nvidia
"""


class CudaErrorHandler:
    """
    High-level error handler for CUDA-related failures.
    
    Provides user-friendly error messages and recovery suggestions.
    """
    
    @staticmethod
    def handle_process_exit(exit_code: Optional[int], stderr_output: str = "") -> str:
        """
        Handle a process exit with CUDA context.
        
        Args:
            exit_code: Process exit code (or None if timed out)
            stderr_output: Optional stderr from the process
            
        Returns:
            Formatted error message for the user
        """
        title, message = translate_exit_code(exit_code)
        
        # Check stderr for CUDA-specific errors
        cuda_hints = []
        stderr_lower = stderr_output.lower()
        
        if "cuda" in stderr_lower:
            if "out of memory" in stderr_lower or "oom" in stderr_lower:
                cuda_hints.append("VRAM exhausted - use smaller model or quantization")
            if "device" in stderr_lower and "busy" in stderr_lower:
                cuda_hints.append("GPU locked by another process - restart PC")
            if "driver" in stderr_lower:
                cuda_hints.append("Driver issue - update NVIDIA drivers")
        
        result = f"\n{'='*60}\n"
        result += f"ERROR: {title}\n"
        result += f"{'='*60}\n\n"
        result += message
        
        if cuda_hints:
            result += f"\n\nDetected Issues:\n"
            for hint in cuda_hints:
                result += f"  • {hint}\n"
        
        if stderr_output and stderr_output.strip():
            trimmed = "\n".join(stderr_output.strip().splitlines()[-15:])
            result += f"\nBackend Output:\n{trimmed}\n"

        if exit_code is None:
            result += f"\nExit Code: None (Server timed out waiting for ready state)\n"
        else:
            result += f"\nExit Code: {exit_code} (0x{exit_code & 0xFFFFFFFF:08X})\n"
        
        return result
    
    @staticmethod
    def is_cuda_device_busy_error(exit_code: Optional[int], stderr: str = "") -> bool:
        """Check if this is a CUDA device busy error."""
        if exit_code is not None:
            # Windows stack buffer overrun often indicates CUDA context issue
            unsigned = exit_code & 0xFFFFFFFF
            if unsigned == 0xC0000409:
                return True
        
        if "busy" in stderr.lower() and "cuda" in stderr.lower():
            return True
        
        return False
    
    @staticmethod
    def is_oom_error(exit_code: Optional[int], stderr: str = "") -> bool:
        """Check if this is an out-of-memory error."""
        if exit_code is not None:
            if exit_code in (2, 0xC0000005):
                return True
        if "out of memory" in stderr.lower():
            return True
        if "oom" in stderr.lower():
            return True
        return False


def attempt_cuda_device_reset() -> bool:
    """
    Attempt to reset CUDA devices to clear stuck contexts.

    Tries multiple approaches in order:
      1. Use PyTorch's `torch.cuda.device_reset()` if available
      2. Call `cudaDeviceReset()` from the CUDA runtime library via ctypes

    Returns:
        True if a reset attempt was made successfully, False otherwise.
    """
    # 1) Try PyTorch if available
    try:
        import torch
        if hasattr(torch.cuda, 'device_reset'):
            try:
                torch.cuda.device_reset()
                logger.info("Called torch.cuda.device_reset() to reset CUDA devices")
                return True
            except Exception as e:
                logger.debug(f"torch.cuda.device_reset() failed: {e}")
    except Exception:
        pass

    # 2) Try calling cudaDeviceReset from CUDA runtime via ctypes
    try:
        names = []
        # Use ctypes.util.find_library to locate cudart
        libname = ctypes.util.find_library('cudart')
        if libname:
            names.append(libname)

        # Common library names across platforms
        names.extend([
            'cudart64_120.dll', 'cudart64_110.dll', 'cudart64_101.dll',  # Windows variants
            'libcudart.so', 'libcudart.so.10.1', 'libcudart.so.11.0',      # Linux variants
            'libcudart.dylib',                                           # macOS (unlikely)
        ])

        for name in names:
            if not name:
                continue
            try:
                lib = ctypes.CDLL(name)
            except Exception:
                continue

            try:
                func = lib.cudaDeviceReset
                func.restype = ctypes.c_int
                res = func()
                if res == 0:
                    logger.info(f"cudaDeviceReset() succeeded via {name}")
                    return True
                else:
                    logger.debug(f"cudaDeviceReset() returned {res} via {name}")
            except AttributeError:
                continue
            except Exception as e:
                logger.debug(f"cudaDeviceReset() call failed via {name}: {e}")

    except Exception as e:
        logger.debug(f"Attempt to load CUDA runtime failed: {e}")

    logger.debug("CUDA device reset attempts exhausted or not available on this system")
    return False


def format_error_for_user(exit_code: int, context: str = "") -> str:
    """
    Format an error message suitable for displaying to end users.
    
    Args:
        exit_code: Process exit code
        context: Optional context about what was happening
        
    Returns:
        User-friendly error message
    """
    title, message = translate_exit_code(exit_code)
    
    output = f"\n❌ {title}\n"
    output += "─" * 50 + "\n"
    
    if context:
        output += f"While: {context}\n\n"
    
    output += message + "\n"
    
    return output
