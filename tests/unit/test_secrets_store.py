import sys
import types

from oprel.utils import secrets_store


def _fake_keyring(monkeypatch):
    store = {}
    kr = types.SimpleNamespace(
        set_password=lambda s, n, v: store.__setitem__((s, n), v),
        get_password=lambda s, n: store.get((s, n)),
        delete_password=lambda s, n: store.pop((s, n), None),
        get_keyring=lambda: object(),
    )
    monkeypatch.setattr(secrets_store, "_keyring", lambda: kr)
    return store


def test_roundtrip_keeps_plaintext_out_of_db_value(monkeypatch):
    store = _fake_keyring(monkeypatch)
    ref = secrets_store.protect("provider:x", "sk-secret")
    assert ref == "keyring:provider:x" and "sk-secret" not in ref
    assert secrets_store.reveal(ref) == "sk-secret"
    secrets_store.forget(ref)
    assert store == {}


def test_no_keyring_falls_back_to_plain(monkeypatch):
    monkeypatch.setattr(secrets_store, "_keyring", lambda: None)
    assert secrets_store.protect("a", "v") == "v"
    assert secrets_store.reveal("v") == "v"
