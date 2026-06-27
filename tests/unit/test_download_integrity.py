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
    SizeMismatchError,
)
from oprel.runtime.binaries.installer import _verify_download_integrity
from oprel.runtime.binaries.registry import resolve_version


def _make_archive(
    tmp_path: Path, content: bytes = b"fake archive", name: str = "archive.tar.gz"
) -> Path:
    """Create a small fake archive file and return its path."""
    p = tmp_path / name
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


# ---------------------------------------------------------------------------
# Size validation
# ---------------------------------------------------------------------------


class TestVerifyDownloadIntegritySize:
    def test_entry_with_matching_size_and_no_sha256_passes(self, tmp_path, monkeypatch):
        """Entry with matching size but no sha256 -> passes."""
        content = b"size only archive"
        archive = _make_archive(tmp_path, content)

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
            accelerator="cpu",
            size=len(content),
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Linux-x86_64", "cpu"): entry},
        )

        _verify_download_integrity(
            archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
        )

    def test_entry_with_neither_size_nor_sha256_is_noop(self, tmp_path, monkeypatch):
        """Entry exists but has no size or sha256 -> no verification."""
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

        archive = _make_archive(tmp_path, b"any content")
        _verify_download_integrity(
            archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
        )

    def test_entry_with_mismatching_size_raises(self, tmp_path, monkeypatch):
        """Entry with wrong size -> SizeMismatchError before SHA256 check."""
        content = b"size mismatch archive"
        archive = _make_archive(tmp_path, content)

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
            accelerator="cpu",
            size=len(content) + 1,
            sha256="0" * 64,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Linux-x86_64", "cpu"): entry},
        )

        with pytest.raises(SizeMismatchError) as exc_info:
            _verify_download_integrity(
                archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
            )
        assert exc_info.value.path == str(archive)
        assert exc_info.value.expected == len(content) + 1
        assert exc_info.value.actual == len(content)

    def test_entry_with_matching_size_and_matching_sha256_passes(self, tmp_path, monkeypatch):
        """Entry with matching size and matching sha256 -> passes."""
        content = b"size and sha256 archive"
        archive = _make_archive(tmp_path, content)
        digest = hashlib.sha256(content).hexdigest()

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
            accelerator="cpu",
            size=len(content),
            sha256=digest,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Linux-x86_64", "cpu"): entry},
        )

        _verify_download_integrity(
            archive, "llama.cpp", "b9616", "Linux-x86_64", "cpu"
        )

    def test_entry_with_matching_size_and_mismatching_sha256_raises(self, tmp_path, monkeypatch):
        """Size passes but sha256 mismatches -> IntegrityMismatchError."""
        content = b"size matches sha256 fails"
        archive = _make_archive(tmp_path, content)
        wrong_digest = "0" * 64

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Linux-x86_64",
            accelerator="cpu",
            size=len(content),
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
        assert exc_info.value.actual == hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Version alias resolution
# ---------------------------------------------------------------------------


class TestResolveVersion:
    def test_latest_resolves_to_concrete_version(self):
        """The 'latest' alias should resolve to the current concrete build."""
        assert resolve_version("llama.cpp", "latest") == "b9616"
        assert resolve_version("stable-diffusion.cpp", "latest") == "master-647-72e512a"

    def test_concrete_version_unchanged(self):
        """Concrete versions pass through unchanged."""
        assert resolve_version("llama.cpp", "b9616") == "b9616"
        assert resolve_version("llama.cpp", "b7822") == "b7822"
        assert resolve_version(
            "stable-diffusion.cpp", "master-647-72e512a"
        ) == "master-647-72e512a"

    def test_unknown_version_returns_original(self):
        """Unknown versions are returned unchanged to preserve existing behavior."""
        assert resolve_version("llama.cpp", "does-not-exist") == "does-not-exist"

    def test_unknown_backend_returns_original(self):
        """Unknown backends are returned unchanged to preserve existing behavior."""
        assert resolve_version("unknown-backend", "latest") == "latest"


class TestVerifyDownloadIntegrityWithResolvedVersion:
    def test_resolved_latest_finds_concrete_keyed_entry(self, tmp_path, monkeypatch):
        """Resolving 'latest' before lookup finds a manifest keyed by concrete version."""
        content = b"resolved latest archive"
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

        resolved = resolve_version("llama.cpp", "latest")
        _verify_download_integrity(
            archive, "llama.cpp", resolved, "Linux-x86_64", "cpu"
        )

    def test_unresolved_latest_misses_concrete_keyed_entry(self, tmp_path, monkeypatch):
        """Without resolving, a 'latest' lookup misses a concrete-keyed entry."""
        content = b"unresolved latest archive"
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

        # Passing the raw alias must remain a no-op: no entry is keyed under "latest".
        _verify_download_integrity(
            archive, "llama.cpp", "latest", "Linux-x86_64", "cpu"
        )


# ---------------------------------------------------------------------------
# DLL archive verification
# ---------------------------------------------------------------------------


class TestVerifyDllDownloadIntegrity:
    """Tests for the optional SHA256 verification of the separate DLL archive.

    The Windows CUDA path downloads a second archive from ``dll_url``.  These
    tests exercise the helper-level verification only: no network, no real
    downloads, no extraction.
    """

    def test_dll_no_manifest_entry_is_noop(self, tmp_path, monkeypatch):
        """Empty manifest -> DLL verification is skipped."""
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {},
        )
        archive = _make_archive(tmp_path)
        _verify_download_integrity(
            archive,
            "llama.cpp",
            "b9616",
            "Windows-AMD64",
            "cuda",
            artifact="dll",
        )

    def test_dll_entry_without_sha256_is_noop(self, tmp_path, monkeypatch):
        """DLL entry exists but sha256 is None -> no verification."""
        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Windows-AMD64",
            accelerator="cuda",
            artifact="dll",
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Windows-AMD64", "cuda", "dll"): entry},
        )
        archive = _make_archive(tmp_path)
        _verify_download_integrity(
            archive,
            "llama.cpp",
            "b9616",
            "Windows-AMD64",
            "cuda",
            artifact="dll",
        )

    def test_dll_matching_sha256_passes(self, tmp_path, monkeypatch):
        """DLL entry with correct sha256 -> verifies without error."""
        content = b"dll archive content"
        archive = _make_archive(tmp_path, content)
        digest = hashlib.sha256(content).hexdigest()

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Windows-AMD64",
            accelerator="cuda",
            artifact="dll",
            sha256=digest,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Windows-AMD64", "cuda", "dll"): entry},
        )

        _verify_download_integrity(
            archive,
            "llama.cpp",
            "b9616",
            "Windows-AMD64",
            "cuda",
            artifact="dll",
        )

    def test_dll_mismatched_sha256_raises(self, tmp_path, monkeypatch):
        """DLL entry with wrong sha256 -> IntegrityMismatchError."""
        archive = _make_archive(tmp_path, b"real dll content")
        wrong_digest = "0" * 64

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Windows-AMD64",
            accelerator="cuda",
            artifact="dll",
            sha256=wrong_digest,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Windows-AMD64", "cuda", "dll"): entry},
        )

        with pytest.raises(IntegrityMismatchError) as exc_info:
            _verify_download_integrity(
                archive,
                "llama.cpp",
                "b9616",
                "Windows-AMD64",
                "cuda",
                artifact="dll",
            )
        assert exc_info.value.expected == wrong_digest
        assert exc_info.value.actual == hashlib.sha256(b"real dll content").hexdigest()

    def test_dll_mismatch_raises_before_extraction(self, tmp_path, monkeypatch):
        """On DLL mismatch, the archive file is left intact."""
        archive = _make_archive(tmp_path, b"corrupt dll archive")
        entry = BinaryIntegrityEntry(
            backend="stable-diffusion.cpp",
            version="master-647-72e512a",
            platform="Windows-AMD64",
            accelerator="cuda",
            artifact="dll",
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
                    "dll",
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
                artifact="dll",
            )
        # Archive should still exist unchanged — no extraction happened.
        assert archive.exists()
        assert archive.read_bytes() == b"corrupt dll archive"

    def test_dll_lookup_does_not_reuse_main_archive_entry(
        self, tmp_path, monkeypatch
    ):
        """artifact='dll' must use the 5-tuple entry, not the 4-tuple main entry."""
        main_content = b"main archive"
        dll_content = b"dll archive"
        main_archive = _make_archive(tmp_path, main_content, name="main.tar.gz")
        dll_archive = _make_archive(tmp_path, dll_content, name="dll.tar.gz")

        main_digest = hashlib.sha256(main_content).hexdigest()
        dll_digest = hashlib.sha256(dll_content).hexdigest()

        entries = {
            ("llama.cpp", "b9616", "Windows-AMD64", "cuda"): BinaryIntegrityEntry(
                backend="llama.cpp",
                version="b9616",
                platform="Windows-AMD64",
                accelerator="cuda",
                sha256=main_digest,
            ),
            (
                "llama.cpp",
                "b9616",
                "Windows-AMD64",
                "cuda",
                "dll",
            ): BinaryIntegrityEntry(
                backend="llama.cpp",
                version="b9616",
                platform="Windows-AMD64",
                accelerator="cuda",
                artifact="dll",
                sha256=dll_digest,
            ),
        }
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            entries,
        )

        # Main archive uses the 4-tuple entry.
        _verify_download_integrity(
            main_archive,
            "llama.cpp",
            "b9616",
            "Windows-AMD64",
            "cuda",
        )

        # DLL archive uses the 5-tuple entry with the DLL digest.
        _verify_download_integrity(
            dll_archive,
            "llama.cpp",
            "b9616",
            "Windows-AMD64",
            "cuda",
            artifact="dll",
        )

    def test_dll_lookup_misses_when_only_main_entry_exists(
        self, tmp_path, monkeypatch
    ):
        """artifact='dll' must not fall back to the main archive entry."""
        dll_content = b"dll archive"
        dll_archive = _make_archive(tmp_path, dll_content)

        # Only a main archive entry exists for this platform/accelerator.
        main_digest = hashlib.sha256(b"main archive").hexdigest()
        entries = {
            ("llama.cpp", "b9616", "Windows-AMD64", "cuda"): BinaryIntegrityEntry(
                backend="llama.cpp",
                version="b9616",
                platform="Windows-AMD64",
                accelerator="cuda",
                sha256=main_digest,
            ),
        }
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            entries,
        )

        # No 5-tuple DLL entry exists, so verification is skipped even though
        # a main archive entry is present.  If artifact='dll' fell back to the
        # 4-tuple key, this would raise on the DLL content.
        _verify_download_integrity(
            dll_archive,
            "llama.cpp",
            "b9616",
            "Windows-AMD64",
            "cuda",
            artifact="dll",
        )


class TestVerifyDllDownloadIntegritySize:
    def test_dll_matching_size_passes(self, tmp_path, monkeypatch):
        """DLL entry with matching size and no sha256 -> passes."""
        content = b"dll size archive"
        archive = _make_archive(tmp_path, content)

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Windows-AMD64",
            accelerator="cuda",
            artifact="dll",
            size=len(content),
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Windows-AMD64", "cuda", "dll"): entry},
        )

        _verify_download_integrity(
            archive,
            "llama.cpp",
            "b9616",
            "Windows-AMD64",
            "cuda",
            artifact="dll",
        )

    def test_dll_mismatching_size_raises(self, tmp_path, monkeypatch):
        """DLL entry with wrong size -> SizeMismatchError."""
        content = b"dll size mismatch archive"
        archive = _make_archive(tmp_path, content)

        entry = BinaryIntegrityEntry(
            backend="llama.cpp",
            version="b9616",
            platform="Windows-AMD64",
            accelerator="cuda",
            artifact="dll",
            size=len(content) + 10,
        )
        monkeypatch.setattr(
            "oprel.runtime.binaries.integrity.BINARY_INTEGRITY_MANIFEST",
            {("llama.cpp", "b9616", "Windows-AMD64", "cuda", "dll"): entry},
        )

        with pytest.raises(SizeMismatchError) as exc_info:
            _verify_download_integrity(
                archive,
                "llama.cpp",
                "b9616",
                "Windows-AMD64",
                "cuda",
                artifact="dll",
            )
        assert exc_info.value.path == str(archive)
        assert exc_info.value.expected == len(content) + 10
        assert exc_info.value.actual == len(content)


class TestEnsureBinaryDllVerification:
    """Installer-level test that the DLL archive path calls verification."""

    def test_ensure_binary_verifies_dll_archive_before_extraction(
        self, tmp_path, monkeypatch
    ):
        """The Windows CUDA DLL download triggers verification before extract."""
        from oprel.runtime.binaries import installer as installer_module
        import zipfile

        binary_dir = tmp_path / "bin"
        main_zip = tmp_path / "main.zip"
        dll_zip = tmp_path / "dll.zip"

        # Create minimal real zip archives for local "download".
        with zipfile.ZipFile(main_zip, "w") as zf:
            zf.writestr("llama-server.exe", b"fake binary")

        with zipfile.ZipFile(dll_zip, "w") as zf:
            zf.writestr("cuda_fake.dll", b"fake dll")

        def fake_safe_download(url, dest_path, config=None):
            if "dll" in url:
                dest_path.write_bytes(dll_zip.read_bytes())
            else:
                dest_path.write_bytes(main_zip.read_bytes())

        monkeypatch.setattr(installer_module, "_safe_download", fake_safe_download)

        verify_calls = []

        def fake_verify(
            archive_path, backend, version, platform, accelerator, artifact=None
        ):
            verify_calls.append((str(archive_path), artifact))

        monkeypatch.setattr(
            installer_module, "_verify_download_integrity", fake_verify
        )

        # Force the Windows CUDA code path regardless of the host running the test.
        monkeypatch.setattr(installer_module.platform, "system", lambda: "Windows")
        monkeypatch.setattr(installer_module.platform, "machine", lambda: "AMD64")
        monkeypatch.setattr(installer_module, "detect_gpu", lambda: {"gpu_type": "cuda"})
        monkeypatch.setattr(installer_module, "_has_vulkan_runtime", lambda: False)

        result = installer_module.ensure_binary(
            "llama.cpp", "b9616", binary_dir, force_download=True
        )

        assert len(verify_calls) == 2
        assert verify_calls[0][1] is None  # main binary archive
        assert verify_calls[1][1] == "dll"  # separate DLL archive
        assert result.exists()
