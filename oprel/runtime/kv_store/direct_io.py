"""Direct page-aligned unbuffered I/O for NVMe/SSD block storage."""
from __future__ import annotations

import os
import sys
import ctypes
from typing import Optional

from oprel.runtime.kv_store.config import PAGE_SIZE

# Windows Win32 API constants
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
OPEN_ALWAYS = 4
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_WRITE_THROUGH = 0x80000000
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
FILE_BEGIN = 0


class AlignedBuffer:
    """Page-aligned memory buffer suitable for unbuffered direct disk I/O."""

    def __init__(self, size: int, alignment: int = PAGE_SIZE):
        assert size % alignment == 0, f"Size ({size}) must be multiple of alignment ({alignment})"
        self.size = size
        self.alignment = alignment
        self._is_windows = sys.platform == "win32"
        self._addr: Optional[int] = None

        if self._is_windows:
            kernel32 = ctypes.windll.kernel32
            kernel32.VirtualAlloc.restype = ctypes.c_void_p
            kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
            self._addr = kernel32.VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
            if not self._addr:
                raise ctypes.WinError(ctypes.get_last_error())
            self._buf = (ctypes.c_char * size).from_address(self._addr)
        else:
            # POSIX / fallback aligned allocation
            raw = bytearray(size + alignment)
            addr = ctypes.c_void_p.from_buffer(raw).value
            offset = (alignment - (addr % alignment)) % alignment
            self._raw = raw
            self._buf = (ctypes.c_char * size).from_buffer(raw, offset)

    @property
    def address(self) -> int:
        if self._addr is not None:
            return self._addr
        return ctypes.c_void_p.from_buffer(self._buf).value

    def write_bytes(self, data: bytes):
        assert len(data) <= self.size, f"Data size {len(data)} exceeds buffer size {self.size}"
        ctypes.memmove(self._buf, data, len(data))

    def to_bytes(self) -> bytes:
        return bytes(self._buf)

    def close(self):
        if self._is_windows and self._addr:
            ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self._addr), 0, MEM_RELEASE)
            self._addr = None

    def __del__(self):
        self.close()


class DirectFile:
    """
    Direct unbuffered file handle with cross-platform fallback.
    On Windows: Uses CreateFileW with FILE_FLAG_NO_BUFFERING.
    On Linux/POSIX/Fallback: Uses unbuffered binary file operations.
    """

    def __init__(self, path: str, mode: str = "r", direct: bool = True):
        self.path = os.path.abspath(path)
        self.mode = mode
        self.direct = direct and sys.platform == "win32"
        self._handle: Optional[int] = None
        self._py_file = None

        if self.direct:
            kernel32 = ctypes.windll.kernel32
            kernel32.CreateFileW.restype = ctypes.c_void_p
            kernel32.CreateFileW.argtypes = [
                ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p
            ]

            flags = FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
            if mode == "r":
                access = GENERIC_READ
                disp = OPEN_EXISTING
            elif mode == "w":
                access = GENERIC_READ | GENERIC_WRITE
                disp = OPEN_ALWAYS
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            h = kernel32.CreateFileW(
                self.path, access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                None, disp, flags, None
            )
            if h == INVALID_HANDLE_VALUE or not h:
                # Direct I/O open failed, fallback to standard buffered file
                self.direct = False
            else:
                self._handle = h

        if not self.direct:
            f_mode = "rb" if mode == "r" else ("r+b" if os.path.exists(self.path) else "w+b")
            self._py_file = open(self.path, f_mode, buffering=0)

    def seek_and_read(self, offset: int, length: int) -> bytes:
        """Read `length` bytes from `offset` (both must be 4096-aligned for direct I/O)."""
        if self.direct and self._handle:
            kernel32 = ctypes.windll.kernel32
            # Move file pointer
            li_dist = ctypes.c_int64(offset)
            kernel32.SetFilePointerEx.argtypes = [
                ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint32
            ]
            if not kernel32.SetFilePointerEx(ctypes.c_void_p(self._handle), li_dist, None, FILE_BEGIN):
                raise ctypes.WinError(ctypes.get_last_error())

            aligned_buf = AlignedBuffer(length)
            bytes_read = ctypes.c_uint32(0)
            kernel32.ReadFile.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p
            ]
            success = kernel32.ReadFile(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(aligned_buf.address),
                length,
                ctypes.byref(bytes_read),
                None
            )
            if not success or bytes_read.value != length:
                raise ctypes.WinError(ctypes.get_last_error())
            data = aligned_buf.to_bytes()
            aligned_buf.close()
            return data
        else:
            self._py_file.seek(offset)
            buf = self._py_file.read(length)
            assert len(buf) == length, f"Short read: expected {length} bytes, got {len(buf)}"
            return buf

    def seek_and_write(self, offset: int, data: bytes):
        """Write `len(data)` bytes to `offset` (both must be 4096-aligned for direct I/O)."""
        length = len(data)
        if self.direct and self._handle:
            kernel32 = ctypes.windll.kernel32
            li_dist = ctypes.c_int64(offset)
            kernel32.SetFilePointerEx.argtypes = [
                ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint32
            ]
            if not kernel32.SetFilePointerEx(ctypes.c_void_p(self._handle), li_dist, None, FILE_BEGIN):
                raise ctypes.WinError(ctypes.get_last_error())

            aligned_buf = AlignedBuffer(length)
            aligned_buf.write_bytes(data)
            bytes_written = ctypes.c_uint32(0)
            kernel32.WriteFile.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p
            ]
            success = kernel32.WriteFile(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(aligned_buf.address),
                length,
                ctypes.byref(bytes_written),
                None
            )
            aligned_buf.close()
            if not success or bytes_written.value != length:
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            self._py_file.seek(offset)
            self._py_file.write(data)
            self._py_file.flush()

    def close(self):
        if self._handle:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self._handle))
            self._handle = None
        if self._py_file:
            self._py_file.close()
            self._py_file = None

    def __del__(self):
        self.close()
