"""
Credential storage.

Secrets (PATs, API tokens) are never stored in SQLite. The database keeps only an
opaque `secret_ref`; the value lives in the OS keychain:

- Windows: Credential Manager (DPAPI, bound to the user's login)
- macOS:   Keychain
- Linux:   Secret Service (GNOME Keyring / KWallet)

If no keychain is available (headless Linux, containers) we fall back to a
Fernet-encrypted file store whose key file is readable only by the owner, and
only if `cryptography` is installed. With neither, storing a secret fails loudly
rather than silently degrading to plaintext.

This module is the only code that should ever read a stored secret.
"""

from __future__ import annotations

import json
import os
import re
import secrets as _pysecrets
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from oprel.core.config import Config
from oprel.utils.redaction import register_secret

SERVICE_NAME = "oprel"
_REF_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class SecretStoreError(RuntimeError):
    pass


def _validate_ref(ref: str) -> str:
    if not isinstance(ref, str) or not _REF_RE.match(ref):
        raise SecretStoreError("Invalid secret reference")
    return ref


def _restrict_permissions(path: Path) -> None:
    """Owner-only access. On Windows, files under the user profile are already ACL'd to the user."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class _Backend(ABC):
    name: str

    @abstractmethod
    def set(self, ref: str, value: str) -> None: ...

    @abstractmethod
    def get(self, ref: str) -> str | None: ...

    @abstractmethod
    def delete(self, ref: str) -> None: ...


class _KeyringBackend(_Backend):
    def __init__(self, keyring_module):
        self._kr = keyring_module
        self.name = f"keyring:{type(keyring_module.get_keyring()).__name__}"

    def set(self, ref: str, value: str) -> None:
        self._kr.set_password(SERVICE_NAME, ref, value)

    def get(self, ref: str) -> str | None:
        return self._kr.get_password(SERVICE_NAME, ref)

    def delete(self, ref: str) -> None:
        try:
            self._kr.delete_password(SERVICE_NAME, ref)
        except self._kr.errors.PasswordDeleteError:
            pass


class _EncryptedFileBackend(_Backend):
    name = "encrypted-file"

    def __init__(self, root: Path):
        from cryptography.fernet import Fernet  # optional dependency

        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        _restrict_permissions(self._root)
        key_path = self._root / "master.key"
        if not key_path.exists():
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(Fernet.generate_key())
        _restrict_permissions(key_path)
        self._fernet = Fernet(key_path.read_bytes())

    def _path(self, ref: str) -> Path:
        return self._root / f"{ref}.bin"

    def set(self, ref: str, value: str) -> None:
        path = self._path(ref)
        path.write_bytes(self._fernet.encrypt(value.encode("utf-8")))
        _restrict_permissions(path)

    def get(self, ref: str) -> str | None:
        path = self._path(ref)
        if not path.exists():
            return None
        return self._fernet.decrypt(path.read_bytes()).decode("utf-8")

    def delete(self, ref: str) -> None:
        self._path(ref).unlink(missing_ok=True)


class _MemoryBackend(_Backend):
    """Test-only backend (OPREL_SECRET_BACKEND=memory). Never selected automatically."""

    name = "memory"

    def __init__(self):
        self._data: dict[str, str] = {}

    def set(self, ref: str, value: str) -> None:
        self._data[ref] = value

    def get(self, ref: str) -> str | None:
        return self._data.get(ref)

    def delete(self, ref: str) -> None:
        self._data.pop(ref, None)


def _usable_keyring():
    try:
        import keyring
        from keyring.backends import fail
    except ImportError:
        return None
    try:
        kr = keyring.get_keyring()
    except Exception:
        return None
    if isinstance(kr, fail.Keyring) or getattr(kr, "priority", 0) <= 0:
        return None
    return keyring


def _select_backend() -> _Backend | None:
    forced = os.environ.get("OPREL_SECRET_BACKEND", "").strip().lower()
    if forced == "memory":
        return _MemoryBackend()

    if forced in ("", "keyring"):
        kr = _usable_keyring()
        if kr is not None:
            return _KeyringBackend(kr)
        if forced == "keyring":
            return None

    try:
        return _EncryptedFileBackend(Config().cache_dir / "secrets")
    except ImportError:
        return None


class SecretStore:
    """
    Stores a JSON dict of secret fields per reference, e.g. {"pat": "..."}.
    Values pass through redaction registration on the way in and out.
    """

    def __init__(self, backend: _Backend | None = None):
        self._backend = backend if backend is not None else _select_backend()

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "unavailable"

    def _require(self) -> _Backend:
        if self._backend is None:
            raise SecretStoreError(
                "No secure credential store is available. Install an OS keyring backend "
                "(e.g. `pip install keyring` with GNOME Keyring/KWallet on Linux) or "
                "`pip install cryptography` for the encrypted-file fallback."
            )
        return self._backend

    @staticmethod
    def new_ref() -> str:
        return f"sec_{_pysecrets.token_hex(16)}"

    def put(self, values: dict[str, str], ref: str | None = None) -> str:
        backend = self._require()
        ref = _validate_ref(ref or self.new_ref())
        payload = {k: v for k, v in values.items() if isinstance(v, str) and v}
        for v in payload.values():
            register_secret(v)
        backend.set(ref, json.dumps(payload))
        return ref

    def get(self, ref: str) -> dict[str, str]:
        backend = self._require()
        raw = backend.get(_validate_ref(ref))
        if not raw:
            return {}
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(values, dict):
            return {}
        for v in values.values():
            if isinstance(v, str):
                register_secret(v)
        return values

    def delete(self, ref: str) -> None:
        # Deliberately leave the value registered for redaction: the same token may be
        # reused elsewhere, and over-redacting a dead secret is harmless.
        self._require().delete(_validate_ref(ref))


_store: SecretStore | None = None
_store_lock = threading.Lock()


def get_secret_store() -> SecretStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = SecretStore()
        return _store


def reset_secret_store_for_tests() -> None:
    global _store
    with _store_lock:
        _store = None
