"""Pydantic v2 schemas for rules configuration (YAML parsing + validation)."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Iterator, Literal

from pydantic import BaseModel, Discriminator, Field, Tag, TypeAdapter, field_validator, model_validator

_SUBJECT_FIELDS = {"user_id", "user_email", "token_id", "group"}


class ConditionSchema(BaseModel):
    type: Literal["header", "parameter", "content", "function", "subject"]
    field: str
    operator: Literal["exact", "regex", "gt", "lt", "gte", "lte", "contains", "in"] = "exact"
    value: Any
    negate: bool = False

    @model_validator(mode="after")
    def _validate_subject(self) -> "ConditionSchema":
        if self.type == "subject" and self.field not in _SUBJECT_FIELDS:
            raise ValueError(f"subject condition field must be one of {sorted(_SUBJECT_FIELDS)}")
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


class ApplyQuotaActionSchema(BaseModel):
    type: Literal["apply_quota"] = "apply_quota"
    quota_ids: list[uuid.UUID] = Field(min_length=1)


def _get_action_type(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "block")
    return getattr(v, "type", "block")


ActionSchema = Annotated[
    Annotated[RerouteActionSchema, Tag("reroute")]
    | Annotated[BlockActionSchema, Tag("block")]
    | Annotated[AllowActionSchema, Tag("allow")]
    | Annotated[RewriteActionSchema, Tag("rewrite")]
    | Annotated[ApplyQuotaActionSchema, Tag("apply_quota")],
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


def iter_condition_dicts(group: dict) -> Iterator[dict]:
    """Yield every leaf condition of a raw condition-group dict, at any nesting depth."""
    for item in group.get("conditions", []):
        if "conditions" in item:
            yield from iter_condition_dicts(item)
        else:
            yield item


def validate_rule_definition(conditions: dict, action: dict) -> None:
    """Checks spanning a rule's conditions and action together; raises ValueError.

    Also re-validates both parts, so stored JSON written before a schema change (such as
    a disabled rule with the removed ``quota`` condition) can't be re-enabled as-is.
    """
    validate_condition_group_dict(conditions)
    validate_action_dict(action)
    if action.get("type") != "apply_quota":
        for cond in iter_condition_dicts(conditions):
            if cond.get("type") == "parameter" and cond.get("field") == "resolved_model":
                raise ValueError(
                    "resolved_model is only known once the model is resolved, "
                    "so it can only be used on apply_quota rules"
                )
