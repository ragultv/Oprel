"""Unit tests for process management, timeout handling, and cuda error translation."""

import pytest
from oprel.runtime.cuda_errors import CudaErrorHandler, translate_exit_code


def test_translate_exit_code_none_timeout():
    """Verify translate_exit_code handles None exit_code safely without bitwise crash."""
    title, message = translate_exit_code(None)
    assert "Timeout" in title
    assert "timeout" in message.lower()


def test_handle_process_exit_none():
    """Verify handle_process_exit handles None without TypeError: unsupported operand for &."""
    output = CudaErrorHandler.handle_process_exit(None, stderr_output="Some backend logs")
    assert "ERROR: Startup Timeout" in output
    assert "Exit Code: None" in output
    assert "Backend Output:\nSome backend logs" in output


def test_is_cuda_device_busy_error_none():
    """Verify is_cuda_device_busy_error safely handles None exit_code."""
    assert CudaErrorHandler.is_cuda_device_busy_error(None, "") is False
    assert CudaErrorHandler.is_cuda_device_busy_error(None, "CUDA device busy") is True


def test_is_oom_error_none():
    """Verify is_oom_error safely handles None exit_code."""
    assert CudaErrorHandler.is_oom_error(None, "") is False
    assert CudaErrorHandler.is_oom_error(None, "CUDA error: out of memory") is True

