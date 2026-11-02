import hashlib

from oprel.downloader.hub import _verify_download


def test_matching_size_and_hash(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"abc" * 100)
    assert _verify_download(p, 300, hashlib.sha256(b"abc" * 100).hexdigest())


def test_truncated_file_fails_size(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"abc")
    assert not _verify_download(p, 300, None)


def test_swapped_content_fails_hash(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 300)
    assert not _verify_download(p, 300, hashlib.sha256(b"abc" * 100).hexdigest())


def test_no_metadata_falls_back_to_nonempty(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x")
    assert _verify_download(p, None, None)
