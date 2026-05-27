"""Pydantic v2 schemas for rules configuration (YAML parsing + validation)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Tag, field_validator


class ConditionSchema(BaseModel):
    type: Literal["header", "parameter", "content", "function"]
    field: str
    operator: Literal["exact", "regex", "gt", "lt", "gte", "lte", "contains", "in"] = "exact"
    value: Any
    negate: bool = False


class ConditionGroupSchema(BaseModel):
    logic: Literal["and", "or"] = "and"
    conditions: list[ConditionSchema | ConditionGroupSchema]


class RerouteActionSchema(BaseModel):
    type: Literal["reroute"] = "reroute"
    target: str


class BlockActionSchema(BaseModel):
    type: Literal["block"] = "block"
    message: str = "Request blocked by policy"


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
