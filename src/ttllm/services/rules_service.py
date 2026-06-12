"""Rules service: CRUD operations, caching, and evaluation helpers."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.config import settings
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
    iter_conditions,
)
from ttllm.models.rule import Rule
from ttllm.schemas.anthropic import MessagesRequest
from ttllm.services import usage_service

logger = logging.getLogger(__name__)


# --- Cache ---
#
# The cache is per-process. With multiple workers (ECS runs several uvicorn
# workers), a CRUD write only invalidates the cache of the worker that served
# it; the others pick up the change within ``rules_cache_ttl_seconds`` via the
# TTL refresh below. ``_cache_loaded_at`` is a ``time.monotonic()`` timestamp
# (immune to wall-clock skew); ``None`` means the cache has never been loaded
# and is distinct from "loaded, but zero active rules".

_rules_cache: list[CoreRule] = []
_cache_loaded_at: float | None = None
_cache_lock = asyncio.Lock()


def _cache_is_stale() -> bool:
    if _cache_loaded_at is None:
        return True
    return (time.monotonic() - _cache_loaded_at) >= settings.engine.rules_cache_ttl_seconds


async def get_active_rules(db: AsyncSession) -> list[CoreRule]:
    if _cache_is_stale():
        await refresh_rules_cache(db)
    return _rules_cache


async def refresh_rules_cache(db: AsyncSession) -> None:
    global _cache_loaded_at
    async with _cache_lock:
        # Re-check under the lock: a concurrent caller may have refreshed while
        # we waited, so we avoid a thundering herd of redundant queries.
        if not _cache_is_stale():
            return
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
        _cache_loaded_at = time.monotonic()
        logger.info("Rules cache refreshed: %d active rules loaded", len(core_rules))


def invalidate_rules_cache() -> None:
    global _cache_loaded_at
    _rules_cache.clear()
    _cache_loaded_at = None


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
                window=item.get("window"),
                per=tuple(item.get("per", [])),
            ))
    return ConditionGroup(logic=logic, conditions=tuple(conditions))


def _convert_action_dict(data: dict) -> Action:
    action_type = ActionType(data["type"])
    if action_type == ActionType.REROUTE:
        return Action(type=action_type, target=data["target"])
    if action_type == ActionType.BLOCK:
        return Action(
            type=action_type,
            target=data.get("message", "Request blocked by policy"),
            status_code=data.get("status_code", 403),
        )
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
    return apply_action(matched_rule, ctx)


async def populate_quota_metadata(
    db: AsyncSession,
    rules: list[CoreRule],
    ctx: RequestContext,
) -> None:
    """Precompute moving-window quota aggregates and inject them into ``ctx.metadata``.

    Keeps the pure core engine free of DB/clock access: the quota matcher and the
    block-message renderer only read ``ctx.metadata["quota"][<measure>]``. Runs at
    most one aggregate query per distinct ``(measure, window, per)`` across all
    rules, and returns immediately (zero queries) when no quota condition exists.
    """
    # Collect distinct quota specs and remember each measure's threshold for the
    # template namespace. A spec is keyed by (measure, window, per) so the same
    # measure over different windows yields separate queries.
    specs: dict[tuple, dict[str, Any]] = {}
    for rule in rules:
        for cond in iter_conditions(rule.conditions):
            if cond.type != ConditionType.QUOTA or cond.window is None:
                continue
            per = _resolve_per(cond.per, ctx)
            key = (cond.field, cond.window, tuple(sorted(per.items())))
            specs.setdefault(key, {
                "measure": cond.field,
                "window": cond.window,
                "per": per,
                "threshold": cond.value,
            })

    if not specs:
        return

    now = datetime.now(timezone.utc)
    quota_ns: dict[str, Any] = ctx.metadata.setdefault("quota", {})
    for spec in specs.values():
        agg = await usage_service.get_window_aggregate(
            db,
            uuid.UUID(ctx.user_id),
            measure=spec["measure"],
            window_seconds=spec["window"],
            per=spec["per"] or None,
            now=now,
        )
        quota_ns[spec["measure"]] = {
            "value": agg["value"],
            "threshold": spec["threshold"],
            "window": spec["window"],
            "next_free": _next_free(agg["oldest_ts"], spec["window"], now),
            "per": spec["per"],
        }


def _resolve_per(per: tuple[str, ...], ctx: RequestContext) -> dict[str, Any]:
    """Map a quota condition's ``per`` dimensions to concrete filter values."""
    resolved: dict[str, Any] = {}
    if "model" in per:
        resolved["model"] = ctx.model
    return resolved


def _next_free(oldest_ts: datetime | None, window: int, now: datetime) -> int:
    """Seconds until the oldest contributing row ages out of the window."""
    if oldest_ts is None:
        return 0
    if oldest_ts.tzinfo is None:
        oldest_ts = oldest_ts.replace(tzinfo=timezone.utc)
    seconds = (oldest_ts + timedelta(seconds=window) - now).total_seconds()
    return max(0, int(seconds))


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
