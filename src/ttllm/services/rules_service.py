"""Rules service: loads rules from config and converts schemas to core dataclasses."""

from __future__ import annotations

import logging
import re
from typing import Any

from ttllm.core.rules import (
    Action,
    ActionType,
    Condition,
    ConditionGroup,
    ConditionType,
    LogicOp,
    MatchOp,
    RequestContext,
    Rule,
    RuleOutcome,
    apply_action,
    evaluate,
)
from ttllm.schemas.anthropic import MessagesRequest
from ttllm.schemas.rules import ConditionGroupSchema, ConditionSchema, RuleSchema

logger = logging.getLogger(__name__)


def _convert_condition(schema: ConditionSchema) -> Condition:
    return Condition(
        type=ConditionType(schema.type),
        field=schema.field,
        operator=MatchOp(schema.operator),
        value=schema.value,
        negate=schema.negate,
    )


def _convert_condition_group(schema: ConditionGroupSchema) -> ConditionGroup:
    conditions = []
    for item in schema.conditions:
        if isinstance(item, ConditionGroupSchema):
            conditions.append(_convert_condition_group(item))
        else:
            conditions.append(_convert_condition(item))
    return ConditionGroup(
        logic=LogicOp(schema.logic),
        conditions=tuple(conditions),
    )


def _convert_action(schema: Any) -> Action:
    action_type = ActionType(schema.type)
    if action_type == ActionType.REROUTE:
        return Action(type=action_type, target=schema.target)
    if action_type == ActionType.BLOCK:
        return Action(type=action_type, target=schema.message)
    if action_type == ActionType.REWRITE:
        return Action(type=action_type, pattern=schema.pattern, replacement=schema.replacement)
    return Action(type=action_type)


def load_rules(rule_schemas: list[RuleSchema]) -> list[Rule]:
    rules = []
    for schema in rule_schemas:
        try:
            rule = Rule(
                name=schema.name,
                weight=schema.weight,
                conditions=_convert_condition_group(schema.conditions),
                action=_convert_action(schema.action),
                enabled=schema.enabled,
                description=schema.description,
            )
            rules.append(rule)
        except Exception:
            logger.exception("Failed to load rule '%s', skipping", schema.name)
    logger.info("Loaded %d rules", len(rules))
    return rules


def build_request_context(
    request: MessagesRequest,
    headers: dict[str, str],
    user_id: str,
    metadata: dict[str, Any] | None = None,
) -> RequestContext:
    messages_text = _extract_messages_text(request)
    system_text = _extract_system_text(request)

    return RequestContext(
        model=request.model,
        messages_text=messages_text,
        system_text=system_text,
        headers=headers,
        user_id=user_id,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=request.stream,
        metadata=metadata or {},
    )


def evaluate_rules(rules: list[Rule], ctx: RequestContext) -> RuleOutcome | None:
    matched_rule = evaluate(rules, ctx)
    if matched_rule is None:
        return None
    logger.info("Rule '%s' matched (weight=%d)", matched_rule.name, matched_rule.weight)
    return apply_action(matched_rule)


def apply_rewrite_to_request(
    request: MessagesRequest,
    pattern: str,
    replacement: str,
) -> MessagesRequest:
    compiled = re.compile(pattern)
    updated_messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            new_content = compiled.sub(replacement, msg.content)
            updated_messages.append(msg.model_copy(update={"content": new_content}))
        elif isinstance(msg.content, list):
            new_blocks = []
            for block in msg.content:
                if hasattr(block, "text"):
                    new_text = compiled.sub(replacement, block.text)
                    new_blocks.append(block.model_copy(update={"text": new_text}))
                else:
                    new_blocks.append(block)
            updated_messages.append(msg.model_copy(update={"content": new_blocks}))
        else:
            updated_messages.append(msg)

    updates: dict[str, Any] = {"messages": updated_messages}
    if request.system:
        if isinstance(request.system, str):
            updates["system"] = compiled.sub(replacement, request.system)
        elif isinstance(request.system, list):
            new_sys = []
            for block in request.system:
                if hasattr(block, "text"):
                    new_sys.append(block.model_copy(update={"text": compiled.sub(replacement, block.text)}))
                else:
                    new_sys.append(block)
            updates["system"] = new_sys

    return request.model_copy(update=updates)


def _extract_messages_text(request: MessagesRequest) -> str:
    parts = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            parts.append(msg.content)
        elif isinstance(msg.content, list):
            for block in msg.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
    return "\n".join(parts)


def _extract_system_text(request: MessagesRequest) -> str:
    if request.system is None:
        return ""
    if isinstance(request.system, str):
        return request.system
    parts = []
    for block in request.system:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)
