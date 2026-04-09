"""Permission registry. Pure logic, no framework dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field


VALID_CATEGORIES = {"management", "gateway"}


@dataclass(frozen=True)
class PermissionDef:
    name: str
    description: str
    category: str  # "management" | "gateway"


@dataclass
class PermissionRegistry:
    permissions: dict[str, PermissionDef] = field(default_factory=dict)

    def filter_by_category(self, perms: set[str], category: str) -> set[str]:
        """Return only permissions whose category matches."""
        return {p for p in perms if p in self.permissions and self.permissions[p].category == category}

    def check(self, user_perms: set[str], required: str) -> bool:
        """Verify that *required* permission exists and is in *user_perms*."""
        if required not in self.permissions:
            return False
        return required in user_perms

    def validate_permission(self, permission: str) -> bool:
        """Check whether a permission name exists in the registry."""
        return permission in self.permissions


# --- Auto-registering permission constants ---

_registry = PermissionRegistry()


def _define(name: str, description: str, category: str) -> str:
    """Register a permission and return its string name."""
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Permission '{name}' has invalid category '{category}'; must be one of {VALID_CATEGORIES}")
    _registry.permissions[name] = PermissionDef(name=name, description=description, category=category)
    return name


class Permissions:
    """All permission constants. Each attribute is a plain str equal to the dotted permission name."""

    # --- User ---
    USER_VIEW = _define("user.view", "View user list and details", "management")
    USER_CREATE = _define("user.create", "Create new users", "management")
    USER_MODIFY = _define("user.modify", "Update user details", "management")
    USER_DELETE = _define("user.delete", "Deactivate users", "management")

    # --- Group ---
    GROUP_VIEW = _define("group.view", "View groups and memberships", "management")
    GROUP_CREATE = _define("group.create", "Create groups", "management")
    GROUP_MODIFY = _define("group.modify", "Modify groups, assign permissions and members", "management")
    GROUP_DELETE = _define("group.delete", "Delete groups", "management")

    # --- Model ---
    MODEL_VIEW = _define("model.view", "View LLM models", "management")
    MODEL_CREATE = _define("model.create", "Register new LLM models", "management")
    MODEL_MODIFY = _define("model.modify", "Update model configuration", "management")
    MODEL_DELETE = _define("model.delete", "Deactivate LLM models", "management")
    MODEL_ASSIGN = _define("model.assign", "Assign/unassign models to users", "management")

    # --- Secret ---
    SECRET_VIEW = _define("secret.view", "View stored secrets", "management")
    SECRET_CREATE = _define("secret.create", "Create new secrets", "management")
    SECRET_MODIFY = _define("secret.modify", "Update secret values", "management")
    SECRET_DELETE = _define("secret.delete", "Delete secrets", "management")

    # --- Token ---
    TOKEN_CREATE = _define("token.create", "Generate gateway tokens", "management")
    TOKEN_REVOKE = _define("token.revoke", "Revoke gateway tokens", "management")

    # --- Audit / Usage ---
    AUDIT_VIEW = _define("audit.view", "View audit logs", "management")
    USAGE_VIEW = _define("usage.view", "View usage statistics", "management")

    # --- Gateway ---
    LLM_INVOKE = _define("llm.invoke", "Invoke LLM models through the gateway", "gateway")

    @staticmethod
    def get_registry() -> PermissionRegistry:
        return _registry
