"""
Secret storage backed by the OS credential store (Windows Credential Manager,
macOS Keychain, Secret Service) via ``keyring``.

SQLite keeps only a reference such as ``keyring:provider:openai``.  If no usable
keyring backend exists the secret falls back to the caller's own storage
(the plain value is returned unchanged) and a warning is logged once.
"""

from typing import Optional

from oprel.utils.logging import get_logger

logger = get_logger(__name__)

SERVICE = "oprel"
REF_PREFIX = "keyring:"
_warned = False


def _keyring():
    global _warned
    try:
        import keyring
        from keyring.backends import fail

        if isinstance(keyring.get_keyring(), fail.Keyring):
            raise RuntimeError("no usable keyring backend")
        return keyring
    except Exception as exc:
        if not _warned:
            logger.warning(f"OS keyring unavailable ({exc}); secrets stay in the local database")
            _warned = True
        return None


def is_ref(value: Optional[str]) -> bool:
    return bool(value) and value.startswith(REF_PREFIX)


def protect(name: str, secret: str) -> str:
    """Store `secret` under `name`; return the value to persist in the database."""
    if not secret or is_ref(secret):
        return secret
    kr = _keyring()
    if kr is None:
        return secret
    try:
        kr.set_password(SERVICE, name, secret)
        return REF_PREFIX + name
    except Exception as exc:
        logger.warning(f"Could not write secret '{name}' to keyring: {exc}")
        return secret


def reveal(value: Optional[str]) -> str:
    """Resolve a stored value (reference or plain) to the real secret."""
    if not value:
        return ""
    if not is_ref(value):
        return value
    kr = _keyring()
    if kr is None:
        return ""
    try:
        return kr.get_password(SERVICE, value[len(REF_PREFIX):]) or ""
    except Exception as exc:
        logger.warning(f"Could not read secret '{value}' from keyring: {exc}")
        return ""


def forget(value: Optional[str]) -> None:
    if not is_ref(value):
        return
    kr = _keyring()
    if kr is None:
        return
    try:
        kr.delete_password(SERVICE, value[len(REF_PREFIX):])
    except Exception:
        pass
