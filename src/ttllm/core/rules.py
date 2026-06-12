"""Rules engine core logic. Pure dataclasses and functions, no framework dependencies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class MatchOp(str, Enum):
    EXACT = "exact"
    REGEX = "regex"
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    CONTAINS = "contains"
    IN = "in"


class ConditionType(str, Enum):
    HEADER = "header"
    PARAMETER = "parameter"
    CONTENT = "content"
    FUNCTION = "function"
    QUOTA = "quota"


class ActionType(str, Enum):
    REROUTE = "reroute"
    BLOCK = "block"
    ALLOW = "allow"
    REWRITE = "rewrite"


class LogicOp(str, Enum):
    AND = "and"
    OR = "or"


@dataclass(frozen=True)
class Condition:
    type: ConditionType
    field: str
    operator: MatchOp
    value: Any
    negate: bool = False
    # Quota conditions only: the moving-window size (seconds) and the dimensions
    # the aggregate is scoped by (e.g. ("model",)). Ignored for other types.
    window: int | None = None
    per: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConditionGroup:
    logic: LogicOp
    conditions: tuple[Condition | ConditionGroup, ...]


@dataclass(frozen=True)
class Action:
    type: ActionType
    target: str | None = None
    pattern: str | None = None
    replacement: str | None = None
    status_code: int = 403


@dataclass(frozen=True)
class Rule:
    name: str
    weight: int
    conditions: ConditionGroup
    action: Action
    enabled: bool = True
    description: str = ""


@dataclass(frozen=True)
class RequestContext:
    model: str
    messages_text: str
    system_text: str
    headers: dict[str, str]
    user_id: str
    max_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    stream: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleOutcome:
    action_type: ActionType
    rule_name: str = ""
    rerouted_model: str | None = None
    block_status: int = 403
    block_message: str = ""
    rewrite_pattern: str | None = None
    rewrite_replacement: str | None = None
    retry_after_seconds: int | None = None


# --- Matcher registry ---

_MATCHERS: dict[ConditionType, Callable[[Condition, RequestContext], bool]] = {}


def register_matcher(condition_type: ConditionType):
    def decorator(fn: Callable[[Condition, RequestContext], bool]):
        _MATCHERS[condition_type] = fn
        return fn
    return decorator


# --- Function registry for computed values ---

_FUNCTIONS: dict[str, Callable[[RequestContext], Any]] = {}


def register_function(name: str):
    def decorator(fn: Callable[[RequestContext], Any]):
        _FUNCTIONS[name] = fn
        return fn
    return decorator


# --- Comparison helpers ---

def _compare(actual: Any, operator: MatchOp, expected: Any) -> bool:
    if actual is None:
        return False

    if operator == MatchOp.EXACT:
        return str(actual) == str(expected)

    if operator == MatchOp.REGEX:
        return bool(re.search(str(expected), str(actual)))

    if operator == MatchOp.CONTAINS:
        return str(expected) in str(actual)

    if operator == MatchOp.IN:
        if isinstance(expected, list):
            return str(actual) in [str(v) for v in expected]
        return str(actual) in str(expected)

    try:
        num_actual = float(actual)
        num_expected = float(expected)
    except (ValueError, TypeError):
        return False

    if operator == MatchOp.GT:
        return num_actual > num_expected
    if operator == MatchOp.LT:
        return num_actual < num_expected
    if operator == MatchOp.GTE:
        return num_actual >= num_expected
    if operator == MatchOp.LTE:
        return num_actual <= num_expected

    return False


# --- Built-in matchers ---

@register_matcher(ConditionType.HEADER)
def _match_header(condition: Condition, ctx: RequestContext) -> bool:
    header_value = ctx.headers.get(condition.field.lower(), "")
    return _compare(header_value, condition.operator, condition.value)


@register_matcher(ConditionType.PARAMETER)
def _match_parameter(condition: Condition, ctx: RequestContext) -> bool:
    param_map = {
        "model": ctx.model,
        "max_tokens": ctx.max_tokens,
        "temperature": ctx.temperature,
        "top_p": ctx.top_p,
        "top_k": ctx.top_k,
        "stream": ctx.stream,
    }
    value = param_map.get(condition.field)
    if value is None and condition.field.startswith("metadata."):
        key = condition.field[len("metadata."):]
        value = ctx.metadata.get(key)
    return _compare(value, condition.operator, condition.value)


@register_matcher(ConditionType.CONTENT)
def _match_content(condition: Condition, ctx: RequestContext) -> bool:
    if condition.field == "system":
        text = ctx.system_text
    else:
        text = ctx.messages_text
    return _compare(text, condition.operator, condition.value)


@register_matcher(ConditionType.FUNCTION)
def _match_function(condition: Condition, ctx: RequestContext) -> bool:
    fn = _FUNCTIONS.get(condition.field)
    if fn is None:
        return False
    computed = fn(ctx)
    return _compare(computed, condition.operator, condition.value)


@register_matcher(ConditionType.QUOTA)
def _match_quota(condition: Condition, ctx: RequestContext) -> bool:
    # The service layer precomputes the moving-window aggregate into
    # ``ctx.metadata["quota"][<measure>]``; this matcher only reads it (pure, no
    # I/O). ``condition.field`` is the measure (cost/tokens/requests),
    # ``condition.value`` the threshold. A missing namespace means the service
    # found no data, which compares as 0 (never trips a ``gt`` threshold).
    quota = ctx.metadata.get("quota", {})
    bucket = quota.get(condition.field, {})
    current = bucket.get("value", 0)
    return _compare(current, condition.operator, condition.value)


# --- Built-in functions ---

@register_function("count_tokens")
def _count_tokens(ctx: RequestContext) -> int:
    total_text = ctx.messages_text + ctx.system_text
    return len(total_text) // 4


@register_function("message_length")
def _message_length(ctx: RequestContext) -> int:
    return len(ctx.messages_text)


@register_function("keyword_count")
def _keyword_count(ctx: RequestContext) -> int:
    return len(ctx.messages_text.split())


# --- Evaluation ---

def evaluate_condition(condition: Condition, ctx: RequestContext) -> bool:
    matcher = _MATCHERS.get(condition.type)
    if matcher is None:
        return False
    result = matcher(condition, ctx)
    return (not result) if condition.negate else result


def evaluate_group(group: ConditionGroup, ctx: RequestContext) -> bool:
    results = []
    for cond in group.conditions:
        if isinstance(cond, ConditionGroup):
            results.append(evaluate_group(cond, ctx))
        else:
            results.append(evaluate_condition(cond, ctx))

    if group.logic == LogicOp.AND:
        return all(results)
    return any(results)


def evaluate(rules: list[Rule], ctx: RequestContext) -> Rule | None:
    sorted_rules = sorted(
        (r for r in rules if r.enabled),
        key=lambda r: r.weight,
        reverse=True,
    )
    for rule in sorted_rules:
        if evaluate_group(rule.conditions, ctx):
            return rule
    return None


def iter_conditions(group: ConditionGroup):
    """Yield every leaf Condition in a (possibly nested) ConditionGroup."""
    for cond in group.conditions:
        if isinstance(cond, ConditionGroup):
            yield from iter_conditions(cond)
        else:
            yield cond


# --- Message templating ---

_TEMPLATE_VAR = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def render_template(text: str, namespace: dict[str, Any]) -> str:
    """Substitute ``{{ dotted.path }}`` references against ``namespace``.

    A deliberately tiny, safe substitution — only dotted dict lookups, no
    expressions, filters, or attribute access (unlike ``str.format``). An
    unresolved reference is left in place untouched.
    """
    def _sub(match: re.Match) -> str:
        current: Any = namespace
        for part in match.group(1).split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return match.group(0)
        return str(current)

    return _TEMPLATE_VAR.sub(_sub, text)


def _retry_after_for_rule(rule: Rule, ctx: RequestContext) -> int | None:
    """Largest ``next_free`` across the rule's quota conditions, or None.

    You cannot retry until the most-constraining window has room, so we take the
    max of the per-measure ``next_free`` values the service precomputed.
    """
    quota = ctx.metadata.get("quota", {})
    waits = [
        quota.get(cond.field, {}).get("next_free")
        for cond in iter_conditions(rule.conditions)
        if cond.type == ConditionType.QUOTA
    ]
    waits = [w for w in waits if isinstance(w, int)]
    return max(waits) if waits else None


def apply_action(rule: Rule, ctx: RequestContext) -> RuleOutcome:
    action = rule.action

    if action.type == ActionType.ALLOW:
        return RuleOutcome(action_type=ActionType.ALLOW, rule_name=rule.name)

    if action.type == ActionType.REROUTE:
        return RuleOutcome(
            action_type=ActionType.REROUTE,
            rule_name=rule.name,
            rerouted_model=action.target,
        )

    if action.type == ActionType.BLOCK:
        message = render_template(
            action.target or "Request blocked by policy", ctx.metadata
        )
        return RuleOutcome(
            action_type=ActionType.BLOCK,
            rule_name=rule.name,
            block_status=action.status_code,
            block_message=message,
            retry_after_seconds=_retry_after_for_rule(rule, ctx),
        )

    if action.type == ActionType.REWRITE:
        return RuleOutcome(
            action_type=ActionType.REWRITE,
            rule_name=rule.name,
            rewrite_pattern=action.pattern,
            rewrite_replacement=action.replacement,
        )

    return RuleOutcome(action_type=ActionType.ALLOW, rule_name=rule.name)
