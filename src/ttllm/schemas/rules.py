"""Pydantic v2 schemas for rules configuration (YAML parsing + validation)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Tag, TypeAdapter, field_validator, model_validator

_QUOTA_MEASURES = {"cost", "tokens", "requests"}


class ConditionSchema(BaseModel):
    type: Literal["header", "parameter", "content", "function", "quota"]
    field: str
    operator: Literal["exact", "regex", "gt", "lt", "gte", "lte", "contains", "in"] = "exact"
    value: Any
    negate: bool = False
    # Quota conditions only.
    window: int | None = None
    per: list[str] = []

    @model_validator(mode="after")
    def _validate_quota(self) -> "ConditionSchema":
        if self.type == "quota":
            if self.field not in _QUOTA_MEASURES:
                raise ValueError(
                    f"quota condition field must be one of {sorted(_QUOTA_MEASURES)}"
                )
            if self.window is None or self.window <= 0:
                raise ValueError("quota condition requires a positive 'window' (seconds)")
            invalid = set(self.per) - {"model"}
            if invalid:
                raise ValueError(f"unsupported quota 'per' dimensions: {sorted(invalid)}")
        return self


class ConditionGroupSchema(BaseModel):
    logic: Literal["and", "or"] = "and"
    conditions: list[ConditionSchema | ConditionGroupSchema]


class RerouteActionSchema(BaseModel):
    type: Literal["reroute"] = "reroute"
    target: str


class BlockActionSchema(BaseModel):
    type: Literal["block"] = "block"
    message: str = "Request blocked by policy"
    status_code: int = 403

    @field_validator("status_code")
    @classmethod
    def _valid_status(cls, v: int) -> int:
        if not 400 <= v <= 599:
            raise ValueError("block status_code must be an HTTP error status (400-599)")
        return v


class AllowActionSchema(BaseModel):
    type: Literal["allow"] = "allow"


class RewriteActionSchema(BaseModel):
    type: Literal["rewrite"] = "rewrite"
    pattern: str
    replacement: str

    @field_validator("pattern")
    @classmethod
    def _valid_regex(cls, v: str) -> str:
        import re
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        return v


def _get_action_type(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "block")
    return getattr(v, "type", "block")


ActionSchema = Annotated[
    Annotated[RerouteActionSchema, Tag("reroute")]
    | Annotated[BlockActionSchema, Tag("block")]
    | Annotated[AllowActionSchema, Tag("allow")]
    | Annotated[RewriteActionSchema, Tag("rewrite")],
    Discriminator(_get_action_type),
]


class RuleSchema(BaseModel):
    name: str
    weight: int = 0
    enabled: bool = True
    description: str = ""
    conditions: ConditionGroupSchema
    action: ActionSchema


_action_adapter: TypeAdapter[Any] = TypeAdapter(ActionSchema)


def validate_action_dict(value: dict) -> dict:
    """Validate a raw action dict against the discriminated ActionSchema union.

    Raises pydantic ValidationError (which field_validators surface as a value
    error) if the action type is unknown or its fields are invalid. This is the
    single source of truth for action validation — admin schemas delegate here
    rather than re-deriving the action types.
    """
    _action_adapter.validate_python(value)
    return value


def validate_condition_group_dict(value: dict) -> dict:
    """Validate a raw condition-group dict against ConditionGroupSchema."""
    ConditionGroupSchema(**value)
    return value
