"""Rules service: CRUD operations, caching, and evaluation helpers."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.core.rules import (
    Action,
    ActionType,
    Condition,
    ConditionGroup,
    ConditionType,
    LogicOp,
    MatchOp,
    RequestContext,
    Rule as CoreRule,
    RuleOutcome,
    apply_action,
    evaluate,
)
from ttllm.models.rule import Rule
from ttllm.schemas.anthropic import MessagesRequest

logger = logging.getLogger(__name__)


# --- Cache ---

_rules_cache: list[CoreRule] = []
_cache_lock = asyncio.Lock()


async def get_active_rules(db: AsyncSession) -> list[CoreRule]:
    if not _rules_cache:
        await refresh_rules_cache(db)
    return _rules_cache


async def refresh_rules_cache(db: AsyncSession) -> None:
    async with _cache_lock:
        result = await db.execute(
            select(Rule).where(Rule.enabled == True).order_by(Rule.weight.desc())  # noqa: E712
        )
        db_rules = result.scalars().all()
        core_rules = []
        for r in db_rules:
            try:
                core_rules.append(_db_rule_to_core(r))
            except Exception:
                logger.exception("Failed to convert rule '%s' to core, skipping", r.name)
        _rules_cache.clear()
        _rules_cache.extend(core_rules)
        logger.info("Rules cache refreshed: %d active rules loaded", len(core_rules))


def invalidate_rules_cache() -> None:
    _rules_cache.clear()


# --- CRUD ---

async def create_rule(
    db: AsyncSession,
    name: str,
    conditions: dict,
    action: dict,
    description: str | None = None,
    weight: int = 0,
    enabled: bool = True,
) -> Rule:
    rule = Rule(
        name=name,
        description=description,
        weight=weight,
        enabled=enabled,
        conditions=conditions,
        action=action,
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    invalidate_rules_cache()
    return rule


async def get_rule(db: AsyncSession, rule_id: uuid.UUID) -> Rule | None:
    return await db.get(Rule, rule_id)


async def list_rules(
    db: AsyncSession,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[Rule], int]:
    count_result = await db.execute(select(Rule.id))
    total = len(count_result.all())

    result = await db.execute(
        select(Rule).order_by(Rule.weight.desc(), Rule.created_at.desc()).offset(offset).limit(limit)
    )
    return list(result.scalars().all()), total


async def update_rule(
    db: AsyncSession,
    rule_id: uuid.UUID,
    **kwargs: Any,
) -> Rule | None:
    rule = await db.get(Rule, rule_id)
    if not rule:
        return None
    _MUTABLE_FIELDS = {"name", "description", "weight", "enabled", "conditions", "action"}
    for key, value in kwargs.items():
        if key in _MUTABLE_FIELDS and value is not None:
            setattr(rule, key, value)
    await db.commit()
    await db.refresh(rule)
    invalidate_rules_cache()
    return rule


async def delete_rule(db: AsyncSession, rule_id: uuid.UUID) -> bool:
    rule = await db.get(Rule, rule_id)
    if not rule:
        return False
    await db.delete(rule)
    await db.commit()
    invalidate_rules_cache()
    return True


# --- Conversion: DB Rule -> Core Rule ---

def _db_rule_to_core(db_rule: Rule) -> CoreRule:
    conditions = _convert_condition_group_dict(db_rule.conditions)
    action = _convert_action_dict(db_rule.action)
    return CoreRule(
        name=db_rule.name,
        weight=db_rule.weight,
        conditions=conditions,
        action=action,
        enabled=db_rule.enabled,
        description=db_rule.description or "",
    )


def _convert_condition_group_dict(data: dict) -> ConditionGroup:
    logic = LogicOp(data.get("logic", "and"))
    conditions = []
    for item in data.get("conditions", []):
        if "conditions" in item:
            conditions.append(_convert_condition_group_dict(item))
        else:
            conditions.append(Condition(
                type=ConditionType(item["type"]),
                field=item["field"],
                operator=MatchOp(item.get("operator", "exact")),
                value=item["value"],
                negate=item.get("negate", False),
            ))
    return ConditionGroup(logic=logic, conditions=tuple(conditions))


def _convert_action_dict(data: dict) -> Action:
    action_type = ActionType(data["type"])
    if action_type == ActionType.REROUTE:
        return Action(type=action_type, target=data["target"])
    if action_type == ActionType.BLOCK:
        return Action(type=action_type, target=data.get("message", "Request blocked by policy"))
    if action_type == ActionType.REWRITE:
        return Action(type=action_type, pattern=data["pattern"], replacement=data["replacement"])
    return Action(type=action_type)


# --- Evaluation helpers (used by messages endpoint) ---

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


def evaluate_rules(rules: list[CoreRule], ctx: RequestContext) -> RuleOutcome | None:
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
