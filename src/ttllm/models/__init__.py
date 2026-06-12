from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


from ttllm.models.user import User  # noqa: E402, F401
from ttllm.models.llm_model import LLMModel, ModelAssignment, GroupModelAssignment  # noqa: E402, F401
from ttllm.models.audit import AuditLog, AuditLogBody  # noqa: E402, F401
from ttllm.models.auth import Group, GroupPermission, UserPermission, UserGroup, Token, RefreshToken  # noqa: E402, F401
from ttllm.models.secret import Secret  # noqa: E402, F401
from ttllm.models.oidc_state import OidcState  # noqa: E402, F401
from ttllm.models.admin_audit import AdminAuditLog  # noqa: E402, F401
from ttllm.models.rule import Rule  # noqa: E402, F401
