"""Password hashing, verification, and policy enforcement via bcrypt."""

from __future__ import annotations

import re
import string

import bcrypt


def validate_password_policy(password: str) -> None:
    """Validate a password against the configured policy.

    Raises ValueError with a descriptive message on failure.
    """
    from ttllm.config import settings

    policy = settings.auth.password_policy
    errors: list[str] = []

    if len(password) < policy.min_length:
        errors.append(f"at least {policy.min_length} characters")
    if len(password) > policy.max_length:
        errors.append(f"at most {policy.max_length} characters")
    if policy.require_uppercase and not re.search(r"[A-Z]", password):
        errors.append("an uppercase letter")
    if policy.require_lowercase and not re.search(r"[a-z]", password):
        errors.append("a lowercase letter")
    if policy.require_digit and not re.search(r"\d", password):
        errors.append("a digit")
    if policy.require_special and not any(c in string.punctuation for c in password):
        errors.append("a special character")

    if errors:
        raise ValueError(f"Password must contain {', '.join(errors)}")


def hash_password(password: str) -> str:
    """Return a bcrypt hash of the given password."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())
