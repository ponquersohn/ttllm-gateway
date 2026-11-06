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
    SUBJECT = "subject"


class ActionType(str, Enum):
    REROUTE = "reroute"
    BLOCK = "block"
    ALLOW = "allow"
    REWRITE = "rewrite"
    APPLY_QUOTA = "apply_quota"


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
    quota_ids: tuple[str, ...] = ()


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
    # Trusted, server-derived identity for ``subject`` conditions.
    user_email: str = ""
    token_id: str = ""
    groups: tuple[str, ...] = ()
    # The model actually serving the request; set only when evaluating apply_quota rules.
    resolved_model: str | None = None


@dataclass
class RuleOutcome:
    action_type: ActionType
    rule_name: str = ""
    rerouted_model: str | None = None
    block_status: int = 403
    block_message: str = ""
    rewrite_pattern: str | None = None
    rewrite_replacement: str | None = None


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
        "resolved_model": ctx.resolved_model,
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


@register_matcher(ConditionType.SUBJECT)
def _match_subject(condition: Condition, ctx: RequestContext) -> bool:
    # Matches the authenticated caller, never anything the client sends in the request.
    if condition.field == "group":
        return any(_compare(group, condition.operator, condition.value) for group in ctx.groups)
    subject_map = {
        "user_id": ctx.user_id,
        "user_email": ctx.user_email,
        "token_id": ctx.token_id,
    }
    return _compare(subject_map.get(condition.field), condition.operator, condition.value)


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
    """Return the highest-weight matching terminal rule. ``apply_quota`` rules never win:
    they don't compete for the request, see ``collect_quota_ids``."""
    sorted_rules = sorted(
        (r for r in rules if r.enabled and r.action.type != ActionType.APPLY_QUOTA),
        key=lambda r: r.weight,
        reverse=True,
    )
    for rule in sorted_rules:
        if evaluate_group(rule.conditions, ctx):
            return rule
    return None


def collect_quota_ids(rules: list[Rule], ctx: RequestContext) -> list[str]:
    """Union of the quotas attached by every matching ``apply_quota`` rule.

    Unlike terminal rules these all apply, so no other rule, whatever its weight, can make
    a request skip a quota.
    """
    quota_ids: dict[str, None] = {}
    for rule in rules:
        if rule.enabled and rule.action.type == ActionType.APPLY_QUOTA and evaluate_group(rule.conditions, ctx):
            quota_ids.update(dict.fromkeys(rule.action.quota_ids))
    return list(quota_ids)


def references_groups(rules: list[Rule]) -> bool:
    """Whether any rule has a ``subject.group`` condition, so groups must be loaded."""
    return any(
        cond.type == ConditionType.SUBJECT and cond.field == "group"
        for rule in rules
        for cond in iter_conditions(rule.conditions)
    )


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
        )

    if action.type == ActionType.REWRITE:
        return RuleOutcome(
            action_type=ActionType.REWRITE,
            rule_name=rule.name,
            rewrite_pattern=action.pattern,
            rewrite_replacement=action.replacement,
        )

    return RuleOutcome(action_type=ActionType.ALLOW, rule_name=rule.name)
