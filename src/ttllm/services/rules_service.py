"""Rules service: CRUD operations, caching, and evaluation helpers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import time
import uuid
from typing import Any, Callable

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
    references_groups,
)
from ttllm.models.quota import Quota
from ttllm.models.rule import Rule
from ttllm.schemas.anthropic import (
    ContentSource,
    DocumentBlock,
    DocumentSource,
    MessagesRequest,
    PlainTextSource,
    SearchResultBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from ttllm.schemas.rules import iter_condition_dicts, validate_rule_definition
from ttllm.services import group_service

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


class RuleValidationError(ValueError):
    """A rule's definition is invalid as a whole; surfaced to admins as a 422."""


async def _validate_enabled_rule(db: AsyncSession, name: str, conditions: dict, action: dict) -> None:
    """Checks needing both parts of a rule, or the database. Only enabled rules must pass,
    so a disabled legacy rule can still be renamed or described, but not switched back on."""
    try:
        validate_rule_definition(conditions, action)
    except ValueError as exc:
        raise RuleValidationError(str(exc)) from exc

    if action.get("type") != "apply_quota":
        if action.get("type") in ("block", "reroute") and any(
            c.get("type") == "parameter" and c.get("field") == "model" for c in iter_condition_dicts(conditions)
        ):
            logger.warning(
                "Rule '%s' matches on the requested 'model', which clients can vary by sending "
                "any name a model's match_pattern accepts; use 'resolved_model' to match the "
                "model that will actually serve the request",
                name,
            )
        return
    quota_ids = [uuid.UUID(str(q)) for q in action["quota_ids"]]
    result = await db.execute(select(Quota.id).where(Quota.id.in_(quota_ids)))
    missing = set(quota_ids) - set(result.scalars().all())
    if missing:
        raise RuleValidationError(f"Unknown quota ids: {sorted(str(m) for m in missing)}")
    if any(c.get("type") in ("header", "content") for c in iter_condition_dicts(conditions)):
        logger.warning(
            "Rule '%s' attaches a quota using header/content conditions, "
            "which clients control and can use to avoid the quota",
            name,
        )


async def create_rule(
    db: AsyncSession,
    name: str,
    conditions: dict,
    action: dict,
    description: str | None = None,
    weight: int = 0,
    enabled: bool = True,
) -> Rule:
    if enabled:
        await _validate_enabled_rule(db, name, conditions, action)
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
    changes = {k: v for k, v in kwargs.items() if k in _MUTABLE_FIELDS and v is not None}
    if changes.get("enabled", rule.enabled):
        await _validate_enabled_rule(
            db,
            changes.get("name", rule.name),
            changes.get("conditions", rule.conditions),
            changes.get("action", rule.action),
        )
    for key, value in changes.items():
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
        return Action(
            type=action_type,
            target=data.get("message", "Request blocked by policy"),
            status_code=data.get("status_code", 403),
        )
    if action_type == ActionType.REWRITE:
        return Action(type=action_type, pattern=data["pattern"], replacement=data["replacement"])
    if action_type == ActionType.APPLY_QUOTA:
        return Action(type=action_type, quota_ids=tuple(str(q) for q in data["quota_ids"]))
    return Action(type=action_type)


# --- Evaluation helpers (used by messages endpoint) ---

def build_request_context(
    request: MessagesRequest,
    headers: dict[str, str],
    user_id: str,
    metadata: dict[str, Any] | None = None,
    *,
    user_email: str = "",
    token_id: str = "",
    groups: tuple[str, ...] = (),
    resolved_model: str | None = None,
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
        user_email=user_email,
        token_id=token_id,
        groups=groups,
        resolved_model=resolved_model,
    )


def build_status_context(
    model_name: str,
    user_id: str,
    user_email: str,
    token_id: str,
    groups: tuple[str, ...],
) -> RequestContext:
    """Context for working out which quotas apply outside a request, so with no content."""
    return RequestContext(
        model=model_name,
        messages_text="",
        system_text="",
        headers={},
        user_id=user_id,
        max_tokens=0,
        temperature=None,
        top_p=None,
        top_k=None,
        stream=False,
        user_email=user_email,
        token_id=token_id,
        groups=groups,
        resolved_model=model_name,
    )


async def load_subject_groups(
    db: AsyncSession,
    rules: list[CoreRule],
    user_id: uuid.UUID,
) -> tuple[str, ...]:
    """The caller's group names, loaded only when an active rule actually matches on groups.

    Matchers are synchronous and pure, so this has to happen before evaluation.
    """
    if not references_groups(rules):
        return ()
    groups = await group_service.list_user_groups(db, user_id)
    return tuple(g.name for g in groups)


def evaluate_rules(rules: list[CoreRule], ctx: RequestContext) -> RuleOutcome | None:
    matched_rule = evaluate(rules, ctx)
    if matched_rule is None:
        return None
    logger.info("Rule '%s' matched (weight=%d)", matched_rule.name, matched_rule.weight)
    return apply_action(matched_rule, ctx)


def apply_rewrite_to_request(
    request: MessagesRequest,
    pattern: str,
    replacement: str,
) -> MessagesRequest:
    """Apply a regex rewrite to every piece of text a content condition can match: message
    text, tool results, documents, search results, tool-call arguments and the system
    prompt. Thinking blocks are left alone, since providers verify their signatures."""
    compiled = re.compile(pattern)

    def sub(text: str) -> str:
        return compiled.sub(replacement, text)

    updates: dict[str, Any] = {
        "messages": [
            msg.model_copy(update={"content": _map_content(msg.content, sub, include_signed=False)})
            for msg in request.messages
        ]
    }
    if request.system:
        updates["system"] = _map_content(request.system, sub, include_signed=False)
    return request.model_copy(update=updates)


# --- Text walking ---
#
# One walker serves both rule matching (collect every piece of text) and rewrites (replace
# every piece of text), so a rewrite always reaches exactly what a content rule can see.
# Text that the walker skipped would be a way around content rules and redaction.

_TEXT_MEDIA_TYPES = {"application/json", "application/xml", "application/x-yaml", "application/yaml"}


def _is_text_media(media_type: str | None) -> bool:
    media_type = (media_type or "").lower()
    return media_type.startswith("text/") or media_type in _TEXT_MEDIA_TYPES


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    """Apply ``fn`` to every string inside a JSON-like value (tool-call arguments)."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    return value


def _map_base64_text(source: DocumentSource, fn: Callable[[str], str]) -> DocumentSource:
    if not _is_text_media(source.media_type):
        return source
    try:
        text = base64.b64decode(source.data, validate=False).decode("utf-8")
    except (binascii.Error, ValueError):
        return source
    new_text = fn(text)
    if new_text == text:
        return source
    return source.model_copy(update={"data": base64.b64encode(new_text.encode("utf-8")).decode("ascii")})


def _map_document_source(source: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(source, PlainTextSource):
        return source.model_copy(update={"data": fn(source.data)})
    if isinstance(source, ContentSource):
        return source.model_copy(update={"content": _map_content(source.content, fn)})
    if isinstance(source, DocumentSource):
        return _map_base64_text(source, fn)
    return source


def _map_block(block: Any, fn: Callable[[str], str], include_signed: bool) -> Any:
    if isinstance(block, TextBlock):
        return block.model_copy(update={"text": fn(block.text)})
    if isinstance(block, ToolResultBlock):
        return block.model_copy(update={"content": _map_content(block.content, fn, include_signed)})
    if isinstance(block, DocumentBlock):
        update: dict[str, Any] = {"source": _map_document_source(block.source, fn)}
        if block.title is not None:
            update["title"] = fn(block.title)
        if block.context is not None:
            update["context"] = fn(block.context)
        return block.model_copy(update=update)
    if isinstance(block, SearchResultBlock):
        return block.model_copy(
            update={"title": fn(block.title), "content": _map_content(block.content, fn, include_signed)}
        )
    if isinstance(block, ToolUseBlock):
        return block.model_copy(update={"input": _map_strings(block.input, fn)})
    if isinstance(block, ThinkingBlock) and include_signed:
        fn(block.thinking)  # matched against, but never rewritten (it's signed)
        return block
    return block


def _map_content(content: Any, fn: Callable[[str], str], include_signed: bool = True) -> Any:
    """Apply ``fn`` to every piece of text in a message's (or system prompt's) content."""
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        return [_map_block(block, fn, include_signed) for block in content]
    return content


def _collect_text(content: Any) -> list[str]:
    parts: list[str] = []

    def collect(text: str) -> str:
        parts.append(text)
        return text

    _map_content(content, collect)
    return parts


def _extract_messages_text(request: MessagesRequest) -> str:
    parts: list[str] = []
    for msg in request.messages:
        parts.extend(_collect_text(msg.content))
    return "\n".join(parts)


def _extract_system_text(request: MessagesRequest) -> str:
    if request.system is None:
        return ""
    return "\n".join(_collect_text(request.system))
