"""Tests for rules engine core logic and service layer."""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from ttllm.core.rules import (
    Action,
    ActionType,
    Condition,
    ConditionGroup,
    ConditionType,
    LogicOp,
    MatchOp,
    RequestContext,
    RuleOutcome,
    Rule,
    apply_action,
    evaluate,
    evaluate_condition,
    evaluate_group,
    iter_conditions,
    render_template,
)
from ttllm.schemas.anthropic import Message, MessagesRequest
from ttllm.schemas.admin import RuleCreate, RuleUpdate, RuleResponse
from ttllm.schemas.rules import (
    BlockActionSchema,
    ConditionGroupSchema,
    ConditionSchema,
    RerouteActionSchema,
    RewriteActionSchema,
    RuleSchema,
)
from ttllm.services.rules_service import (
    _convert_action_dict,
    _convert_condition_group_dict,
    _db_rule_to_core,
    _next_free,
    apply_rewrite_to_request,
    build_request_context,
    evaluate_rules,
    invalidate_rules_cache,
    populate_quota_metadata,
)


def _ctx(**kwargs) -> RequestContext:
    defaults = dict(
        model="claude-sonnet",
        messages_text="Hello, how are you?",
        system_text="You are helpful.",
        headers={"x-api-key": "test-key", "x-custom": "foo"},
        user_id="user-123",
        max_tokens=1024,
        temperature=0.7,
        top_p=None,
        top_k=None,
        stream=False,
        metadata={},
    )
    defaults.update(kwargs)
    return RequestContext(**defaults)


def _rule(name="test", weight=10, conditions=None, action=None, enabled=True):
    if conditions is None:
        conditions = ConditionGroup(
            logic=LogicOp.AND,
            conditions=(
                Condition(type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="claude-sonnet"),
            ),
        )
    if action is None:
        action = Action(type=ActionType.ALLOW)
    return Rule(name=name, weight=weight, conditions=conditions, action=action, enabled=enabled)


# --- Condition evaluation ---


class TestConditionEvaluation:
    def test_header_exact_match(self):
        cond = Condition(type=ConditionType.HEADER, field="x-custom", operator=MatchOp.EXACT, value="foo")
        ctx = _ctx()
        assert evaluate_condition(cond, ctx) is True

    def test_header_exact_no_match(self):
        cond = Condition(type=ConditionType.HEADER, field="x-custom", operator=MatchOp.EXACT, value="bar")
        ctx = _ctx()
        assert evaluate_condition(cond, ctx) is False

    def test_header_regex_match(self):
        cond = Condition(type=ConditionType.HEADER, field="x-custom", operator=MatchOp.REGEX, value="fo+")
        ctx = _ctx()
        assert evaluate_condition(cond, ctx) is True

    def test_parameter_model_exact(self):
        cond = Condition(type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="claude-sonnet")
        ctx = _ctx()
        assert evaluate_condition(cond, ctx) is True

    def test_parameter_max_tokens_gt(self):
        cond = Condition(type=ConditionType.PARAMETER, field="max_tokens", operator=MatchOp.GT, value=500)
        ctx = _ctx(max_tokens=1024)
        assert evaluate_condition(cond, ctx) is True

    def test_parameter_max_tokens_lt(self):
        cond = Condition(type=ConditionType.PARAMETER, field="max_tokens", operator=MatchOp.LT, value=500)
        ctx = _ctx(max_tokens=1024)
        assert evaluate_condition(cond, ctx) is False

    def test_content_contains(self):
        cond = Condition(type=ConditionType.CONTENT, field="messages", operator=MatchOp.CONTAINS, value="how are")
        ctx = _ctx()
        assert evaluate_condition(cond, ctx) is True

    def test_content_regex(self):
        cond = Condition(type=ConditionType.CONTENT, field="messages", operator=MatchOp.REGEX, value=r"Hello.*you")
        ctx = _ctx()
        assert evaluate_condition(cond, ctx) is True

    def test_content_system(self):
        cond = Condition(type=ConditionType.CONTENT, field="system", operator=MatchOp.CONTAINS, value="helpful")
        ctx = _ctx()
        assert evaluate_condition(cond, ctx) is True

    def test_function_count_tokens(self):
        cond = Condition(type=ConditionType.FUNCTION, field="count_tokens", operator=MatchOp.GT, value=1)
        ctx = _ctx(messages_text="a" * 100)
        assert evaluate_condition(cond, ctx) is True

    def test_negate(self):
        cond = Condition(
            type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="gpt-4", negate=True
        )
        ctx = _ctx(model="claude-sonnet")
        assert evaluate_condition(cond, ctx) is True

    def test_in_operator(self):
        cond = Condition(
            type=ConditionType.PARAMETER, field="model", operator=MatchOp.IN, value=["claude-sonnet", "claude-haiku"]
        )
        ctx = _ctx(model="claude-sonnet")
        assert evaluate_condition(cond, ctx) is True


# --- Condition group evaluation ---


class TestConditionGroup:
    def test_and_all_true(self):
        group = ConditionGroup(
            logic=LogicOp.AND,
            conditions=(
                Condition(type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="claude-sonnet"),
                Condition(type=ConditionType.PARAMETER, field="max_tokens", operator=MatchOp.GT, value=100),
            ),
        )
        assert evaluate_group(group, _ctx()) is True

    def test_and_one_false(self):
        group = ConditionGroup(
            logic=LogicOp.AND,
            conditions=(
                Condition(type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="claude-sonnet"),
                Condition(type=ConditionType.PARAMETER, field="max_tokens", operator=MatchOp.LT, value=100),
            ),
        )
        assert evaluate_group(group, _ctx()) is False

    def test_or_one_true(self):
        group = ConditionGroup(
            logic=LogicOp.OR,
            conditions=(
                Condition(type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="gpt-4"),
                Condition(type=ConditionType.PARAMETER, field="max_tokens", operator=MatchOp.GT, value=100),
            ),
        )
        assert evaluate_group(group, _ctx()) is True

    def test_nested_groups(self):
        inner = ConditionGroup(
            logic=LogicOp.AND,
            conditions=(
                Condition(type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="claude-sonnet"),
                Condition(type=ConditionType.PARAMETER, field="max_tokens", operator=MatchOp.GT, value=100),
            ),
        )
        outer = ConditionGroup(
            logic=LogicOp.OR,
            conditions=(
                Condition(type=ConditionType.HEADER, field="x-custom", operator=MatchOp.EXACT, value="nope"),
                inner,
            ),
        )
        assert evaluate_group(outer, _ctx()) is True


# --- Rule evaluation (priority/weight) ---


class TestRuleEvaluation:
    def test_first_match_wins_by_weight(self):
        rules = [
            _rule(name="low", weight=1, action=Action(type=ActionType.BLOCK, target="low")),
            _rule(name="high", weight=100, action=Action(type=ActionType.REROUTE, target="haiku")),
        ]
        result = evaluate(rules, _ctx())
        assert result is not None
        assert result.name == "high"

    def test_no_match_returns_none(self):
        rules = [
            _rule(
                name="miss",
                conditions=ConditionGroup(
                    logic=LogicOp.AND,
                    conditions=(
                        Condition(type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="gpt-4"),
                    ),
                ),
            ),
        ]
        assert evaluate(rules, _ctx()) is None

    def test_disabled_rules_skipped(self):
        rules = [_rule(name="disabled", enabled=False)]
        assert evaluate(rules, _ctx()) is None


# --- Actions ---


class TestActions:
    def test_allow(self):
        rule = _rule(action=Action(type=ActionType.ALLOW))
        outcome = apply_action(rule, _ctx())
        assert outcome.action_type == ActionType.ALLOW

    def test_reroute(self):
        rule = _rule(action=Action(type=ActionType.REROUTE, target="haiku"))
        outcome = apply_action(rule, _ctx())
        assert outcome.action_type == ActionType.REROUTE
        assert outcome.rerouted_model == "haiku"

    def test_block(self):
        rule = _rule(action=Action(type=ActionType.BLOCK, target="Not allowed"))
        outcome = apply_action(rule, _ctx())
        assert outcome.action_type == ActionType.BLOCK
        assert outcome.block_message == "Not allowed"

    def test_rewrite(self):
        rule = _rule(action=Action(type=ActionType.REWRITE, pattern=r"\d{3}-\d{2}-\d{4}", replacement="[REDACTED]"))
        outcome = apply_action(rule, _ctx())
        assert outcome.action_type == ActionType.REWRITE
        assert outcome.rewrite_pattern == r"\d{3}-\d{2}-\d{4}"

    def test_block_custom_message_end_to_end(self):
        # The message flows: stored JSONB dict -> _convert_action_dict -> apply_action.
        action = _convert_action_dict({"type": "block", "message": "Custom denial"})
        rule = _rule(action=action)
        outcome = apply_action(rule, _ctx())
        assert outcome.action_type == ActionType.BLOCK
        assert outcome.block_message == "Custom denial"
        assert outcome.block_status == 403


# --- Service layer: dict conversion (simulates DB JSONB) ---


class TestDictConversion:
    def test_convert_condition_group_dict(self):
        data = {
            "logic": "and",
            "conditions": [
                {"type": "parameter", "field": "model", "operator": "exact", "value": "dynamic_model"},
                {"type": "function", "field": "count_tokens", "operator": "gt", "value": 10000},
            ],
        }
        group = _convert_condition_group_dict(data)
        assert group.logic == LogicOp.AND
        assert len(group.conditions) == 2
        assert group.conditions[0].type == ConditionType.PARAMETER

    def test_convert_nested_condition_group(self):
        data = {
            "logic": "or",
            "conditions": [
                {"type": "header", "field": "x-test", "operator": "exact", "value": "yes"},
                {
                    "logic": "and",
                    "conditions": [
                        {"type": "parameter", "field": "model", "operator": "regex", "value": "claude.*"},
                    ],
                },
            ],
        }
        group = _convert_condition_group_dict(data)
        assert group.logic == LogicOp.OR
        assert isinstance(group.conditions[1], ConditionGroup)

    def test_convert_reroute_action(self):
        data = {"type": "reroute", "target": "claude-haiku"}
        action = _convert_action_dict(data)
        assert action.type == ActionType.REROUTE
        assert action.target == "claude-haiku"

    def test_convert_block_action(self):
        data = {"type": "block", "message": "Denied"}
        action = _convert_action_dict(data)
        assert action.type == ActionType.BLOCK
        assert action.target == "Denied"

    def test_convert_rewrite_action(self):
        data = {"type": "rewrite", "pattern": r"\d+", "replacement": "[NUM]"}
        action = _convert_action_dict(data)
        assert action.type == ActionType.REWRITE
        assert action.pattern == r"\d+"

    def test_convert_allow_action(self):
        data = {"type": "allow"}
        action = _convert_action_dict(data)
        assert action.type == ActionType.ALLOW


# --- Service layer: evaluation helpers ---


class TestServiceHelpers:
    def test_build_request_context(self):
        request = MessagesRequest(
            model="test-model",
            max_tokens=512,
            messages=[Message(role="user", content="Hello world")],
            system="Be helpful.",
        )
        ctx = build_request_context(request, headers={"x-key": "val"}, user_id="u1")
        assert ctx.model == "test-model"
        assert ctx.messages_text == "Hello world"
        assert ctx.system_text == "Be helpful."
        assert ctx.max_tokens == 512

    def test_evaluate_rules_reroute(self):
        rules = [Rule(
            name="reroute-big",
            weight=10,
            conditions=ConditionGroup(
                logic=LogicOp.AND,
                conditions=(
                    Condition(type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="dynamic_model"),
                ),
            ),
            action=Action(type=ActionType.REROUTE, target="claude-haiku"),
        )]
        ctx = _ctx(model="dynamic_model")
        outcome = evaluate_rules(rules, ctx)
        assert outcome is not None
        assert outcome.action_type == ActionType.REROUTE
        assert outcome.rerouted_model == "claude-haiku"

    def test_apply_rewrite_to_request(self):
        request = MessagesRequest(
            model="test",
            max_tokens=100,
            messages=[Message(role="user", content="My SSN is 123-45-6789 please help")],
        )
        result = apply_rewrite_to_request(request, r"\d{3}-\d{2}-\d{4}", "[SSN-REDACTED]")
        assert "123-45-6789" not in str(result.messages[0].content)
        assert "[SSN-REDACTED]" in str(result.messages[0].content)

    def test_evaluate_rules_no_match(self):
        rules = [Rule(
            name="miss",
            weight=10,
            conditions=ConditionGroup(
                logic=LogicOp.AND,
                conditions=(
                    Condition(type=ConditionType.PARAMETER, field="model", operator=MatchOp.EXACT, value="nonexistent"),
                ),
            ),
            action=Action(type=ActionType.BLOCK, target="blocked"),
        )]
        ctx = _ctx(model="claude-sonnet")
        assert evaluate_rules(rules, ctx) is None

    def test_cache_invalidation(self):
        from ttllm.services.rules_service import _rules_cache
        _rules_cache.append(_rule())
        assert len(_rules_cache) == 1
        invalidate_rules_cache()
        assert len(_rules_cache) == 0


# --- Service layer: cache TTL ---


def _fake_db(rows=None):
    """An AsyncSession stub whose execute() returns the given ORM rows and
    records how many times it was queried."""
    from unittest.mock import AsyncMock, MagicMock

    rows = rows or []
    db = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = rows
    result = MagicMock()
    result.scalars.return_value = scalars
    db.execute = AsyncMock(return_value=result)
    return db


class TestCacheTTL:
    @pytest.mark.asyncio
    async def test_zero_rules_loads_once_within_ttl(self):
        """Regression: an empty active-rules set must not re-query every call.

        Distinguishes "loaded, zero rules" from "never loaded" via the
        _cache_loaded_at sentinel rather than the truthiness of the list.
        """
        from ttllm.services import rules_service

        invalidate_rules_cache()
        db = _fake_db(rows=[])

        first = await rules_service.get_active_rules(db)
        assert first == []
        assert rules_service._cache_loaded_at is not None
        assert db.execute.await_count == 1

        # Second call within the TTL must serve from cache, not re-query.
        await rules_service.get_active_rules(db)
        assert db.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_invalidate_forces_reload(self):
        from ttllm.services import rules_service

        db = _fake_db(rows=[])
        await rules_service.get_active_rules(db)
        loaded = db.execute.await_count

        invalidate_rules_cache()
        assert rules_service._cache_loaded_at is None
        await rules_service.get_active_rules(db)
        assert db.execute.await_count == loaded + 1

    @pytest.mark.asyncio
    async def test_ttl_expiry_triggers_reload(self, monkeypatch):
        from ttllm.services import rules_service

        invalidate_rules_cache()
        db = _fake_db(rows=[])

        # Freeze monotonic clock so we control TTL expiry deterministically.
        clock = {"t": 1000.0}
        monkeypatch.setattr(rules_service.time, "monotonic", lambda: clock["t"])
        ttl = rules_service.settings.engine.rules_cache_ttl_seconds

        await rules_service.get_active_rules(db)
        assert db.execute.await_count == 1

        clock["t"] += ttl + 1  # advance past the TTL
        await rules_service.get_active_rules(db)
        assert db.execute.await_count == 2


# --- Admin schema validation ---


class TestAdminSchemas:
    def test_rule_create_validates_conditions(self):
        rule = RuleCreate(
            name="test",
            conditions={"logic": "and", "conditions": [{"type": "parameter", "field": "model", "operator": "exact", "value": "x"}]},
            action={"type": "block", "message": "no"},
        )
        assert rule.name == "test"

    def test_rule_create_rejects_invalid_conditions(self):
        with pytest.raises(Exception):
            RuleCreate(
                name="test",
                conditions={"logic": "invalid", "conditions": []},
                action={"type": "block"},
            )

    def test_rule_create_rejects_invalid_action(self):
        with pytest.raises(Exception):
            RuleCreate(
                name="test",
                conditions={"logic": "and", "conditions": [{"type": "parameter", "field": "model", "operator": "exact", "value": "x"}]},
                action={"type": "unknown_action"},
            )

    def test_rule_create_validates_rewrite_regex(self):
        with pytest.raises(Exception):
            RuleCreate(
                name="test",
                conditions={"logic": "and", "conditions": [{"type": "parameter", "field": "model", "operator": "exact", "value": "x"}]},
                action={"type": "rewrite", "pattern": "[invalid", "replacement": "x"},
            )

    def test_rule_update_partial(self):
        update = RuleUpdate(weight=99)
        dumped = update.model_dump(exclude_unset=True)
        assert dumped == {"weight": 99}

    def test_rule_update_rejects_invalid_action(self):
        with pytest.raises(Exception):
            RuleUpdate(action={"type": "unknown_action"})

    def test_rule_update_validates_rewrite_regex(self):
        with pytest.raises(Exception):
            RuleUpdate(action={"type": "rewrite", "pattern": "[invalid", "replacement": "x"})


# --- Message templating ---


class TestRenderTemplate:
    def test_dotted_lookup_hit(self):
        ns = {"quota": {"cost": {"value": 7.4, "next_free": 23}}}
        out = render_template("spent {{quota.cost.value}}, free in {{quota.cost.next_free}}s", ns)
        assert out == "spent 7.4, free in 23s"

    def test_unknown_var_left_untouched(self):
        assert render_template("hi {{quota.missing.x}} there", {"quota": {}}) == "hi {{quota.missing.x}} there"

    def test_no_braces_passthrough(self):
        assert render_template("plain message", {"quota": {}}) == "plain message"

    def test_non_dict_midpath_safe(self):
        ns = {"quota": {"cost": 5}}  # cost is a scalar, not a dict
        assert render_template("{{quota.cost.value}}", ns) == "{{quota.cost.value}}"

    def test_whitespace_inside_braces(self):
        assert render_template("{{ quota.cost }}", {"quota": {"cost": 9}}) == "9"


# --- Quota condition matcher (pure) ---


def _quota_ctx(measure="cost", value=10, **bucket):
    md = {"quota": {measure: {"value": value, **bucket}}}
    return _ctx(metadata=md)


def _quota_cond(measure="cost", op=MatchOp.GT, threshold=5, window=60, per=()):
    return Condition(
        type=ConditionType.QUOTA, field=measure, operator=op, value=threshold,
        window=window, per=tuple(per),
    )


class TestQuotaMatcher:
    def test_gt_trips(self):
        assert evaluate_condition(_quota_cond(threshold=5), _quota_ctx(value=10)) is True

    def test_gt_below_threshold(self):
        assert evaluate_condition(_quota_cond(threshold=20), _quota_ctx(value=10)) is False

    def test_gte_boundary(self):
        assert evaluate_condition(_quota_cond(op=MatchOp.GTE, threshold=10), _quota_ctx(value=10)) is True

    def test_missing_namespace_is_zero(self):
        # No quota metadata populated -> value 0 -> never trips a gt threshold.
        assert evaluate_condition(_quota_cond(threshold=1), _ctx()) is False


# --- apply_action with quota context ---


class TestQuotaAction:
    def test_block_status_and_message_render(self):
        ctx = _quota_ctx(measure="cost", value=7.5, threshold=5.0, window=60, next_free=23)
        action = _convert_action_dict({
            "type": "block",
            "status_code": 429,
            "message": "Over {{quota.cost.value}} (limit {{quota.cost.threshold}}). Wait {{quota.cost.next_free}}s.",
        })
        rule = _rule(
            conditions=ConditionGroup(logic=LogicOp.AND, conditions=(_quota_cond(),)),
            action=action,
        )
        outcome = apply_action(rule, ctx)
        assert outcome.block_status == 429
        assert outcome.block_message == "Over 7.5 (limit 5.0). Wait 23s."
        assert outcome.retry_after_seconds == 23

    def test_retry_after_is_max_across_two_quota_conditions(self):
        md = {"quota": {
            "cost": {"value": 9, "next_free": 12},
            "requests": {"value": 40, "next_free": 30},
        }}
        ctx = _ctx(metadata=md)
        conds = ConditionGroup(logic=LogicOp.AND, conditions=(
            _quota_cond(measure="cost"),
            _quota_cond(measure="requests"),
        ))
        rule = _rule(conditions=conds, action=Action(type=ActionType.BLOCK, target="blocked"))
        outcome = apply_action(rule, ctx)
        assert outcome.retry_after_seconds == 30

    def test_block_default_status(self):
        action = _convert_action_dict({"type": "block", "message": "no"})
        rule = _rule(action=action)
        outcome = apply_action(rule, _ctx())
        assert outcome.block_status == 403


# --- Service: populate_quota_metadata ---


def _aggregate_db(value, oldest_ts):
    """AsyncSession stub whose execute().one() returns a window aggregate row,
    counting how many queries were issued."""
    from unittest.mock import AsyncMock, MagicMock

    row = MagicMock()
    row.value = value
    row.oldest_ts = oldest_ts
    result = MagicMock()
    result.one.return_value = row
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    return db


class TestPopulateQuotaMetadata:
    @pytest.mark.asyncio
    async def test_no_quota_conditions_runs_zero_queries(self):
        db = _aggregate_db(0, None)
        rule = _rule()  # plain parameter condition, no quota
        ctx = _ctx(user_id=str(uuid.uuid4()))
        await populate_quota_metadata(db, [rule], ctx)
        assert db.execute.await_count == 0
        assert "quota" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_populates_namespace_and_next_free(self):
        from datetime import datetime, timezone

        oldest = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        db = _aggregate_db(Decimal("7.42"), oldest)
        rule = _rule(
            conditions=ConditionGroup(logic=LogicOp.AND, conditions=(
                _quota_cond(measure="cost", threshold=5, window=3600),
            )),
            action=Action(type=ActionType.BLOCK, target="blocked"),
        )
        ctx = _ctx(user_id=str(uuid.uuid4()))
        await populate_quota_metadata(db, [rule], ctx)
        assert db.execute.await_count == 1
        bucket = ctx.metadata["quota"]["cost"]
        assert bucket["value"] == Decimal("7.42")
        assert bucket["threshold"] == 5
        assert bucket["window"] == 3600
        assert isinstance(bucket["next_free"], int)


class TestWindowAggregate:
    @pytest.mark.asyncio
    async def test_requests_measure_returns_int(self):
        from datetime import datetime, timezone
        from ttllm.services import usage_service

        db = _aggregate_db(5, datetime(2026, 1, 1, tzinfo=timezone.utc))
        out = await usage_service.get_window_aggregate(
            db, uuid.uuid4(), measure="requests", window_seconds=60,
        )
        assert out["value"] == 5
        assert isinstance(out["value"], int)

    @pytest.mark.asyncio
    async def test_cost_measure_returns_decimal_zero_when_null(self):
        from ttllm.services import usage_service

        db = _aggregate_db(None, None)
        out = await usage_service.get_window_aggregate(
            db, uuid.uuid4(), measure="cost", window_seconds=60,
        )
        assert out["value"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_unknown_measure_raises(self):
        from ttllm.services import usage_service

        with pytest.raises(ValueError):
            await usage_service.get_window_aggregate(
                _aggregate_db(0, None), uuid.uuid4(), measure="bogus", window_seconds=60,
            )

    @pytest.mark.asyncio
    async def test_per_model_adds_join_and_filter(self):
        from ttllm.services import usage_service

        db = _aggregate_db(3, None)
        await usage_service.get_window_aggregate(
            db, uuid.uuid4(), measure="requests", window_seconds=60, per={"model": "claude-x"},
        )
        # Inspect the compiled SQL of the query that was executed.
        compiled = str(db.execute.await_args.args[0])
        assert "llm_models" in compiled
        assert "status_code" in compiled


class TestNextFree:
    def test_none_oldest_is_zero(self):
        from datetime import datetime, timezone
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _next_free(None, 60, now) == 0

    def test_floored_at_zero_when_aged_out(self):
        from datetime import datetime, timedelta, timezone
        now = datetime(2026, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        oldest = now - timedelta(seconds=120)  # already older than 60s window
        assert _next_free(oldest, 60, now) == 0

    def test_seconds_until_oldest_exits(self):
        from datetime import datetime, timedelta, timezone
        now = datetime(2026, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        oldest = now - timedelta(seconds=20)  # 60s window -> 40s remaining
        assert _next_free(oldest, 60, now) == 40
