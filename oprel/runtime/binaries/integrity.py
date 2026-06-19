"""
Binary integrity manifest and SHA256 verification helpers.

This module provides the foundation for future checksum verification of
downloaded runtime binaries.  It defines:

- ``BinaryIntegrityEntry`` – a typed description of the expected integrity
  metadata for a single (backend, version, platform, accelerator) tuple.
- ``BINARY_INTEGRITY_MANIFEST`` – a placeholder mapping that is intentionally
  empty today.  Future PRs will populate it with real digests.
- ``get_integrity_entry()`` – a lookup helper that returns ``None`` when no
  entry exists, so callers can gracefully skip verification.
- ``validate_sha256_format()`` – a pure-string check (64 hex chars, case-insensitive).
- ``compute_sha256()`` – hashes a local file on disk.
- ``verify_sha256()`` – compares a file's digest to an expected value and
  raises ``IntegrityMismatchError`` on mismatch.

None of these helpers are wired into the installer or downloader yet.
They perform no network access and do not modify runtime behavior.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    """

    backend: str
    version: str
    platform: str
    accelerator: Optional[str] = None
    url: Optional[str] = None
    sha256: Optional[str] = None
    size: Optional[int] = None


# ---------------------------------------------------------------------------
# Manifest placeholder
# ---------------------------------------------------------------------------

# Intentionally empty.  Future PRs will populate this with real digests
# sourced from upstream release notes.  Keeping it empty ensures that
# get_integrity_entry() always returns None today, so callers can safely
# skip verification without changing runtime behavior.
BINARY_INTEGRITY_MANIFEST: dict[
    tuple[str, str, str, Optional[str]], BinaryIntegrityEntry
] = {}


def get_integrity_entry(
    backend: str,
    version: str,
    platform: str,
    accelerator: Optional[str] = None,
) -> Optional[BinaryIntegrityEntry]:
    """Look up an integrity entry for a (backend, version, platform, accelerator) tuple.

    Returns ``None`` when no entry exists in the manifest.  This allows
    callers to conditionally verify only when a digest is known.
    """
    return BINARY_INTEGRITY_MANIFEST.get(
        (backend, version, platform, accelerator)
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
