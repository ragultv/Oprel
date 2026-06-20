"""
Unit tests for optional download integrity verification in the installer.

These tests exercise ``_verify_download_integrity`` in isolation — no
network, no real downloads, no archive extraction.  The integrity manifest
is monkeypatched with synthetic entries as needed.

The manifest is empty by default, so all no-op paths are tested against the
real (unpatched) manifest as well as against explicitly-empty patches.
"""

import hashlib
from pathlib import Path

import pytest

from oprel.runtime.binaries.integrity import (
    BinaryIntegrityEntry,
    IntegrityMismatchError,
)
from oprel.runtime.binaries.installer import _verify_download_integrity


def _make_archive(tmp_path: Path, content: bytes = b"fake archive") -> Path:
    """Create a small fake archive file and return its path."""
    p = tmp_path / "archive.tar.gz"
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# No-op cases — verification silently skipped
# ---------------------------------------------------------------------------


class TestVerifyDownloadIntegrityNoOp:
    def test_no_manifest_entry_is_noop(self, tmp_path, monkeypatch):
        """Empty manifest -> helper returns without raising."""
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {},
        )
        archive = _make_archive(tmp_path)
        # Should not raise.
        _verify_download_integrity(
            archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
        )

    def test_entry_without_sha256_is_noop(self, tmp_path, monkeypatch):
        """Entry exists but sha256 is None -> no verification."""
        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
            accelerator="cpu",
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Linux-x86_64", "cpu"): entry},
        )
        archive = _make_archive(tmp_path)
        _verify_download_integrity(
            archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
        )

    def test_empty_manifest_default_is_noop(self, tmp_path):
        """Without any monkeypatching, the real empty manifest -> no-op."""
        archive = _make_archive(tmp_path)
        _verify_download_integrity(
            archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
        )

    def test_wrong_key_is_noop(self, tmp_path, monkeypatch):
        """Entry for a different tuple -> lookup misses, no verification."""
        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Windows-AMD64",
            accelerator="cuda",
            sha256="a" * 64,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Windows-AMD64", "cuda"): entry},
        )
        archive = _make_archive(tmp_path)
        # Looking up a different platform/accelerator should be a no-op.
        _verify_download_integrity(
            archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
        )


# ---------------------------------------------------------------------------
# Enforcement cases — verification actually runs
# ---------------------------------------------------------------------------


class TestVerifyDownloadIntegrityEnforcement:
    def test_matching_sha256_passes(self, tmp_path, monkeypatch):
        """Entry with correct sha256 -> helper verifies without error."""
        content = b"verified archive content"
        archive = _make_archive(tmp_path, content)
        digest = hashlib.sha256(content).hexdigest()

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
            accelerator="cpu",
            sha256=digest,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Linux-x86_64", "cpu"): entry},
        )

        _verify_download_integrity(
            archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
        )

    def test_mismatched_sha256_raises(self, tmp_path, monkeypatch):
        """Entry with wrong sha256 -> IntegrityMismatchError before extraction."""
        archive = _make_archive(tmp_path, b"real content")
        wrong_digest = "0" * 64

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
            accelerator="cpu",
            sha256=wrong_digest,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Linux-x86_64", "cpu"): entry},
        )

        with pytest.raises(IntegrityMismatchError) as exc_info:
            _verify_download_integrity(
                archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
            )
        assert exc_info.value.expected == wrong_digest
        assert exc_info.value.actual == hashlib.sha256(b"real content").hexdigest()

    def test_mismatch_raises_before_extraction(self, tmp_path, monkeypatch):
        """On mismatch, the archive file is left intact — no extraction happened."""
        archive = _make_archive(tmp_path, b"corrupt archive")
        entry = BinaryIntegrityEntry(
            backend="stable-diffusion.cpp",
            version="master-647-72e512a",
            platform="Windows-AMD64",
            accelerator="cuda",
            sha256="f" * 64,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {
                (
                    "stable-diffusion.cpp",
                    "master-647-72e512a",
                    "Windows-AMD64",
                    "cuda",
                ): entry
            },
        )

        with pytest.raises(IntegrityMismatchError):
            _verify_download_integrity(
                archive,
                "stable-diffusion.cpp",
                "master-647-72e512a",
                "Windows-AMD64",
                "cuda",
            )
        # Archive should still exist unchanged — no extraction happened.
        assert archive.exists()
        assert archive.read_bytes() == b"corrupt archive"

    def test_accelerator_specific_key_matched(self, tmp_path, monkeypatch):
        """CUDA entry is matched when accelerator='cuda' is passed."""
        content = b"cuda binary archive"
        archive = _make_archive(tmp_path, content)
        digest = hashlib.sha256(content).hexdigest()

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Windows-AMD64",
            accelerator="cuda",
            sha256=digest,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Windows-AMD64", "cuda"): entry},
        )

        _verify_download_integrity(
            archive, "llama.cpp", "b9616", "Windows-AMD64", "cuda"
        )
