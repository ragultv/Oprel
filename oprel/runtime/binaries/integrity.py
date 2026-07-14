"""
Binary integrity manifest and SHA256 verification helpers.

This module provides the foundation for future checksum verification of
downloaded runtime binaries.  It defines:

- ``BinaryIntegrityEntry`` – a typed description of the expected integrity
  metadata for a single (backend, version, platform, accelerator, artifact)
  tuple.  The ``artifact`` field is optional and defaults to ``None`` for the
  main binary archive; ``"dll"`` can be used to represent the separate
  Windows CUDA runtime library archive.
- ``BINARY_INTEGRITY_MANIFEST`` – a mapping of verified integrity entries.
  It currently contains one verified entry for the llama.cpp b9616 Linux
  x86_64 CPU main archive; additional entries will be added in future PRs.
- ``get_integrity_entry()`` – a lookup helper that returns ``None`` when no
  entry exists, so callers can gracefully skip verification.
- ``validate_sha256_format()`` – a pure-string check (64 hex chars, case-insensitive).
- ``compute_sha256()`` – hashes a local file on disk.
- ``verify_sha256()`` – compares a file's digest to an expected value and
  raises ``IntegrityMismatchError`` on mismatch.
- ``verify_size()`` – compares a file's byte size to an expected value and
  raises ``SizeMismatchError`` on mismatch.

These helpers perform no network access and do not modify runtime behavior.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from oprel.core.exceptions import OprelError

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IntegrityError(OprelError):
    """Base exception for binary integrity verification errors."""


class IntegrityMismatchError(IntegrityError):
    """Raised when a file's computed SHA256 does not match the expected digest.

    Attributes:
        path: The file that was checked.
        expected: The expected SHA256 hex string.
        actual: The computed SHA256 hex string.
    """

    def __init__(self, path: str | Path, expected: str, actual: str) -> None:
        self.path = str(path)
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"SHA256 mismatch for {self.path}: "
            f"expected {expected}, got {actual}"
        )


class SizeMismatchError(IntegrityError):
    """Raised when a file's byte size does not match the expected value.

    Attributes:
        path: The file that was checked.
        expected: The expected size in bytes.
        actual: The actual size in bytes.
    """

    def __init__(self, path: str | Path, expected: int, actual: int) -> None:
        self.path = str(path)
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Size mismatch for {self.path}: "
            f"expected {expected} bytes, got {actual} bytes"
        )


# ---------------------------------------------------------------------------
# Typed manifest entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryIntegrityEntry:
    """Expected integrity metadata for a single downloadable binary artifact.

    Fields mirror the shape proposed in
    ``docs/external/binary_provenance.md`` §5.  All fields except ``backend``,
    ``version``, and ``platform`` are optional so that entries can be added
    incrementally.

    The optional ``artifact`` field distinguishes archives for the same
    platform/accelerator combination.  ``None`` (the default) represents the
    main binary archive; ``"dll"`` represents the separate Windows CUDA
    runtime library archive downloaded from ``dll_url``.
    """

    backend: str
    version: str
    platform: str
    accelerator: Optional[str] = None
    artifact: Optional[str] = None
    url: Optional[str] = None
    sha256: Optional[str] = None
    size: Optional[int] = None


# ---------------------------------------------------------------------------
# Manifest placeholder
# ---------------------------------------------------------------------------

# Verified entries:
# - llama.cpp b9616 Linux-x86_64 CPU main archive.
#   The digest was verified against the GitHub release asset metadata digest
#   and independently recomputed with sha256sum and Python hashlib.
# - llama.cpp b9616 Linux-x86_64 Vulkan main archive.
#   No separate checksum file was published for this asset. The GitHub
#   release asset API digest matched SHA256 values independently recomputed
#   with sha256sum and Python hashlib.
# Additional entries will be added in future PRs.
IntegrityManifestKey = Union[
    tuple[str, str, str, Optional[str]],
    tuple[str, str, str, Optional[str], Optional[str]],
]

BINARY_INTEGRITY_MANIFEST: dict[IntegrityManifestKey, BinaryIntegrityEntry] = {
    ("llama.cpp", "b9616", "Linux-x86_64", "cpu"): BinaryIntegrityEntry(
        backend="llama.cpp",
        version="b9616",
        platform="Linux-x86_64",
        accelerator="cpu",
        artifact=None,
        url="https://github.com/ggml-org/llama.cpp/releases/download/b9616/llama-b9616-bin-ubuntu-x64.tar.gz",
        sha256="06a9651dafa495a3d3a83afc88b421d6e37fc433873745f8a991c4f5839c5a6c",
        size=15493795,
    ),
    ("llama.cpp", "b9616", "Linux-x86_64", "vulkan"): BinaryIntegrityEntry(
        backend="llama.cpp",
        version="b9616",
        platform="Linux-x86_64",
        accelerator="vulkan",
        artifact=None,
        url="https://github.com/ggml-org/llama.cpp/releases/download/b9616/llama-b9616-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="6003f264df830e6de48532837d3dabeeec28ab2c90f700165ead24d47a4fb9ae",
        size=38352024,
    ),
}


def get_integrity_entry(
    backend: str,
    version: str,
    platform: str,
    accelerator: Optional[str] = None,
    artifact: Optional[str] = None,
) -> Optional[BinaryIntegrityEntry]:
    """Look up an integrity entry for a downloadable artifact.

    The lookup key is ``(backend, version, platform, accelerator)`` for the
    default/main archive (``artifact`` is ``None``).  When ``artifact`` is
    provided — for example ``"dll"`` for the Windows CUDA runtime library
    archive — the key includes the artifact value:
    ``(backend, version, platform, accelerator, artifact)``.

    Keeping the main archive on the legacy 4-tuple key preserves backward
    compatibility with existing manifest entries.  DLL archive entries use a
    5-tuple key so they cannot accidentally reuse the main archive checksum.

    Returns ``None`` when no entry exists in the manifest.  This allows
    callers to conditionally verify only when a digest is known.
    """
    if artifact is not None:
        return BINARY_INTEGRITY_MANIFEST.get(
            (backend, version, platform, accelerator, artifact)
        )

    # Default/main archive: support both the legacy 4-tuple key and an
    # explicit 5-tuple key with artifact=None.
    key = (backend, version, platform, accelerator)
    entry = BINARY_INTEGRITY_MANIFEST.get(key)
    if entry is not None:
        return entry
    return BINARY_INTEGRITY_MANIFEST.get(
        (backend, version, platform, accelerator, None)
    )


# ---------------------------------------------------------------------------
# SHA256 helpers
# ---------------------------------------------------------------------------

_SHA256_HEX_LEN = 64
_SHA256_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def validate_sha256_format(digest: str) -> bool:
    """Return True if *digest* is a valid SHA256 hex string.

    A valid SHA256 hex string is exactly 64 hexadecimal characters.
    Both lowercase and uppercase hex digits are accepted; callers that
    need a canonical form should normalize via ``str.lower()``.
    """
    if not isinstance(digest, str):
        return False
    if len(digest) != _SHA256_HEX_LEN:
        return False
    return all(c in _SHA256_HEX_CHARS for c in digest)


def compute_sha256(path: str | Path) -> str:
    """Compute the SHA256 digest of a local file.

    The file is read in 64 KB chunks to avoid loading large files into memory.

    Args:
        path: Path to the file to hash.

    Returns:
        Lowercase 64-character hex string.

    Raises:
        FileNotFoundError: if *path* does not exist.
        IsADirectoryError: if *path* is a directory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {p}")

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str | Path, expected_sha256: str) -> bool:
    """Verify that the SHA256 digest of *path* matches *expected_sha256*.

    *expected_sha256* may be lowercase or uppercase; it is normalized to
    lowercase before comparison so that digests published in uppercase
    by upstream projects are accepted.

    Returns ``True`` when the digests match.

    Raises:
        IntegrityMismatchError: when the computed digest does not match.
        ValueError: when *expected_sha256* is not a valid SHA256 hex string.
        FileNotFoundError: when *path* does not exist.
    """
    if not validate_sha256_format(expected_sha256):
        raise ValueError(
            f"expected_sha256 is not a valid 64-char hex string: "
            f"{expected_sha256!r}"
        )
    expected = expected_sha256.lower()
    actual = compute_sha256(path)
    if actual != expected:
        raise IntegrityMismatchError(path, expected, actual)
    return True


def verify_size(path: str | Path, expected_size: int) -> bool:
    """Verify that the byte size of *path* matches *expected_size*.

    Returns ``True`` when the sizes match.

    Raises:
        SizeMismatchError: when the actual size does not match.
        FileNotFoundError: when *path* does not exist.
        IsADirectoryError: when *path* is a directory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {p}")

    actual_size = p.stat().st_size
    if actual_size != expected_size:
        raise SizeMismatchError(path, expected_size, actual_size)
    return True
