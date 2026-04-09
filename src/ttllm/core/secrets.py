"""Pure encryption logic and secret reference resolution.

No framework dependencies (no FastAPI, no SQLAlchemy).
"""

from __future__ import annotations

from typing import Any, Callable

from cryptography.fernet import Fernet


def validate_fernet_key(key: str) -> bool:
    """Return True if *key* is a valid Fernet key, False otherwise."""
    try:
        Fernet(key.encode() if isinstance(key, str) else key)
        return True
    except (ValueError, Exception):
        return False


def encrypt_value(plaintext: str, key: str) -> str:
    """Encrypt a plaintext string using Fernet. Returns base64-encoded ciphertext."""
    f = Fernet(key.encode() if isinstance(key, str) else key)
    return f.encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str, key: str) -> str:
    """Decrypt a Fernet-encrypted ciphertext. Returns the original plaintext."""
    f = Fernet(key.encode() if isinstance(key, str) else key)
    return f.decrypt(ciphertext.encode()).decode()


SECRET_PREFIX = "secret://"


def resolve_config_secrets(
    config: dict[str, Any],
    resolver: Callable[[str], str | None],
) -> dict[str, Any]:
    """Recursively walk *config* and replace ``secret://name`` values.

    *resolver* is called with the secret name and must return the decrypted
    plaintext or ``None`` (in which case the original reference is kept).
    """
    return _resolve(config, resolver)


def _resolve(value: Any, resolver: Callable[[str], str | None]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve(v, resolver) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(item, resolver) for item in value]
    if isinstance(value, str) and value.startswith(SECRET_PREFIX):
        name = value[len(SECRET_PREFIX):]
        resolved = resolver(name)
        return resolved if resolved is not None else value
    return value


def collect_secret_names(config: dict[str, Any]) -> set[str]:
    """Return all ``secret://name`` references found in *config*."""
    names: set[str] = set()
    _collect(config, names)
    return names


def _collect(value: Any, names: set[str]) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _collect(v, names)
    elif isinstance(value, list):
        for item in value:
            _collect(item, names)
    elif isinstance(value, str) and value.startswith(SECRET_PREFIX):
        names.add(value[len(SECRET_PREFIX):])
