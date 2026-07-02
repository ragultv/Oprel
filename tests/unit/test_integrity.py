"""
Unit tests for oprel.runtime.binaries.integrity

These tests are local-only: no network, no binary downloads, no Oprel
installation required.  They exercise the SHA256 helpers and the
integrity manifest using temporary files created via the ``tmp_path`` fixture.
"""

import hashlib
from pathlib import Path

import pytest

from oprel.runtime.binaries.integrity import (
    BINARY_INTEGRITY_MANIFEST,
    BinaryIntegrityEntry,
    IntegrityError,
    IntegrityMismatchError,
    SizeMismatchError,
    compute_sha256,
    get_integrity_entry,
    validate_sha256_format,
    verify_sha256,
    verify_size,
)

# SHA256 of an empty byte string — a known valid 64-char hex digest.
_VALID_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# validate_sha256_format
# ---------------------------------------------------------------------------


class TestValidateSha256Format:
    def test_valid_lowercase_hex_accepted(self):
        assert validate_sha256_format(_VALID_EMPTY_SHA256) is True

    def test_valid_digest_from_real_file(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"some content")
        digest = compute_sha256(f)
        assert validate_sha256_format(digest) is True

    def test_uppercase_accepted(self):
        assert validate_sha256_format(_VALID_EMPTY_SHA256.upper()) is True

    def test_mixed_case_accepted(self):
        # Interleave upper/lower to exercise both branches of the charset.
        mixed = "".join(
            c.upper() if i % 2 == 0 else c for i, c in enumerate(_VALID_EMPTY_SHA256)
        )
        assert validate_sha256_format(mixed) is True

    def test_wrong_length_rejected(self):
        assert validate_sha256_format("a" * 32) is False
        assert validate_sha256_format("a" * 128) is False

    def test_non_hex_rejected(self):
        assert validate_sha256_format("z" * 64) is False
        assert validate_sha256_format("g" * 64) is False

    def test_non_string_rejected(self):
        assert validate_sha256_format(12345) is False  # type: ignore[arg-type]
        assert validate_sha256_format(None) is False  # type: ignore[arg-type]
        assert validate_sha256_format(b"a" * 64) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_sha256
# ---------------------------------------------------------------------------


class TestComputeSha256:
    def test_known_content(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert compute_sha256(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert compute_sha256(f) == _VALID_EMPTY_SHA256

    def test_large_file_chunked(self, tmp_path):
        """File larger than the 64 KB read chunk to exercise the loop."""
        f = tmp_path / "large.bin"
        content = b"x" * (65536 * 3 + 17)
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_sha256(f) == expected

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_sha256(tmp_path / "nope.bin")

    def test_directory_raises(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            compute_sha256(tmp_path)

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "str_path.bin"
        f.write_bytes(b"test")
        expected = hashlib.sha256(b"test").hexdigest()
        assert compute_sha256(str(f)) == expected


# ---------------------------------------------------------------------------
# verify_sha256
# ---------------------------------------------------------------------------


class TestVerifySha256:
    def test_matching_digest_returns_true(self, tmp_path):
        f = tmp_path / "match.bin"
        f.write_bytes(b"match me")
        digest = hashlib.sha256(b"match me").hexdigest()
        assert verify_sha256(f, digest) is True

    def test_mismatching_digest_raises(self, tmp_path):
        f = tmp_path / "mismatch.bin"
        f.write_bytes(b"real content")
        wrong = "0" * 64
        with pytest.raises(IntegrityMismatchError) as exc_info:
            verify_sha256(f, wrong)
        assert exc_info.value.expected == wrong
        assert exc_info.value.actual == hashlib.sha256(b"real content").hexdigest()
        assert str(f) in str(exc_info.value)

    def test_mismatch_is_integrity_error(self, tmp_path):
        """IntegrityMismatchError must be a subclass of IntegrityError."""
        f = tmp_path / "subclass.bin"
        f.write_bytes(b"content")
        with pytest.raises(IntegrityError):
            verify_sha256(f, "0" * 64)

    def test_invalid_expected_format_raises_value_error(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"data")
        with pytest.raises(ValueError, match="not a valid"):
            verify_sha256(f, "not-a-hash")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            verify_sha256(tmp_path / "nope.bin", "0" * 64)

    def test_uppercase_expected_accepted(self, tmp_path):
        """Uppercase expected digest is normalized and should match."""
        f = tmp_path / "upper.bin"
        f.write_bytes(b"data")
        digest_lower = hashlib.sha256(b"data").hexdigest()
        assert verify_sha256(f, digest_lower.upper()) is True

    def test_mismatch_with_uppercase_expected_raises(self, tmp_path):
        """Uppercase but wrong digest must still raise IntegrityMismatchError."""
        f = tmp_path / "upper_wrong.bin"
        f.write_bytes(b"real content")
        wrong_upper = ("0" * 64).upper()
        with pytest.raises(IntegrityMismatchError):
            verify_sha256(f, wrong_upper)


# ---------------------------------------------------------------------------
# verify_size
# ---------------------------------------------------------------------------


class TestVerifySize:
    def test_matching_size_returns_true(self, tmp_path):
        f = tmp_path / "match_size.bin"
        f.write_bytes(b"exactly ten")
        assert verify_size(f, 11) is True

    def test_mismatching_size_raises(self, tmp_path):
        f = tmp_path / "mismatch_size.bin"
        f.write_bytes(b"hello")
        with pytest.raises(SizeMismatchError) as exc_info:
            verify_size(f, 42)
        assert exc_info.value.path == str(f)
        assert exc_info.value.expected == 42
        assert exc_info.value.actual == 5
        assert "expected 42 bytes, got 5 bytes" in str(exc_info.value)

    def test_mismatch_is_integrity_error(self, tmp_path):
        """SizeMismatchError must be a subclass of IntegrityError."""
        f = tmp_path / "subclass_size.bin"
        f.write_bytes(b"content")
        with pytest.raises(IntegrityError):
            verify_size(f, 999)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            verify_size(tmp_path / "nope.bin", 10)

    def test_directory_raises(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            verify_size(tmp_path, 10)

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "str_path_size.bin"
        f.write_bytes(b"test")
        assert verify_size(str(f), 4) is True


# ---------------------------------------------------------------------------
# Manifest placeholder
# ---------------------------------------------------------------------------


class TestManifestEntry:
    def test_manifest_contains_at_least_one_entry(self):
        assert len(BINARY_INTEGRITY_MANIFEST) >= 1

    def test_manifest_contains_llama_cpp_b9616_linux_x86_64_cpu(self):
        entry = BINARY_INTEGRITY_MANIFEST.get(
            ("llama.cpp", "b9616", "Linux-x86_64", "cpu")
        )
        assert isinstance(entry, BinaryIntegrityEntry)
        assert entry.artifact is None

    def test_get_integrity_entry_returns_entry(self):
        result = get_integrity_entry("llama.cpp", "b9616", "Linux-x86_64", "cpu")
        assert result is not None
        assert result.backend == "llama.cpp"
        assert result.version == "b9616"
        assert result.platform == "Linux-x86_64"
        assert result.accelerator == "cpu"
        assert result.artifact is None
        assert result.url == (
            "https://github.com/ggml-org/llama.cpp/releases/download/b9616/"
            "llama-b9616-bin-ubuntu-x64.tar.gz"
        )
        assert result.sha256 == (
            "06a9651dafa495a3d3a83afc88b421d6e37fc433873745f8a991c4f5839c5a6c"
        )
        assert result.size == 15493795

    def test_get_integrity_entry_returns_none_for_unknown_platform(self):
        result = get_integrity_entry("llama.cpp", "b9616", "Darwin-arm64", "cpu")
        assert result is None

    def test_get_integrity_entry_returns_none_for_unknown_version(self):
        result = get_integrity_entry("llama.cpp", "b7822", "Linux-x86_64", "cpu")
        assert result is None

    def test_get_integrity_entry_returns_none_for_unknown_accelerator(self):
        result = get_integrity_entry(
            "llama.cpp", "b9616", "Linux-x86_64", "vulkan"
        )
        assert result is None

    def test_get_integrity_entry_without_accelerator_returns_none(self):
        """The main-archive entry is keyed with accelerator='cpu', not None."""
        result = get_integrity_entry("llama.cpp", "b9616", "Linux-x86_64")
        assert result is None


# ---------------------------------------------------------------------------
# BinaryIntegrityEntry dataclass
# ---------------------------------------------------------------------------


class TestBinaryIntegrityEntry:
    def test_required_fields_only(self):
        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
        )
        assert entry.backend == "llama.cpp"
        assert entry.version == "b9616"
        assert entry.platform == "Linux-x86_64"
        assert entry.accelerator is None
        assert entry.url is None
        assert entry.sha256 is None
        assert entry.size is None

    def test_all_fields(self):
        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
            accelerator="cpu",
            url="https://example.com/test.tar.gz",
            sha256="a" * 64,
            size=12345678,
        )
        assert entry.accelerator == "cpu"
        assert entry.url == "https://example.com/test.tar.gz"
        assert entry.sha256 == "a" * 64
        assert entry.size == 12345678

    def test_frozen_is_immutable(self):
        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
        )
        with pytest.raises(Exception):
            entry.backend = "stable-diffusion.cpp"  # type: ignore[misc]
