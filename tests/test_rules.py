"""Tests for rules engine core logic and service layer."""

from __future__ import annotations

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
    Rule,
    RuleOutcome,
    apply_action,
    evaluate,
    evaluate_condition,
    evaluate_group,
)
from ttllm.schemas.anthropic import Message, MessagesRequest
from ttllm.schemas.rules import (
    BlockActionSchema,
    ConditionGroupSchema,
    ConditionSchema,
    RerouteActionSchema,
    RewriteActionSchema,
    RuleSchema,
)
from ttllm.services.rules_service import (
    apply_rewrite_to_request,
    build_request_context,
    evaluate_rules,
    load_rules,
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
        outcome = apply_action(rule)
        assert outcome.action_type == ActionType.ALLOW

    def test_reroute(self):
        rule = _rule(action=Action(type=ActionType.REROUTE, target="haiku"))
        outcome = apply_action(rule)
        assert outcome.action_type == ActionType.REROUTE
        assert outcome.rerouted_model == "haiku"

    def test_block(self):
        rule = _rule(action=Action(type=ActionType.BLOCK, target="Not allowed"))
        outcome = apply_action(rule)
        assert outcome.action_type == ActionType.BLOCK
        assert outcome.block_message == "Not allowed"

    def test_rewrite(self):
        rule = _rule(action=Action(type=ActionType.REWRITE, pattern=r"\d{3}-\d{2}-\d{4}", replacement="[REDACTED]"))
        outcome = apply_action(rule)
        assert outcome.action_type == ActionType.REWRITE
        assert outcome.rewrite_pattern == r"\d{3}-\d{2}-\d{4}"


# --- Service layer ---


class TestRulesService:
    def test_load_rules_from_schema(self):
        schemas = [
            RuleSchema(
                name="test-rule",
                weight=50,
                conditions=ConditionGroupSchema(
                    logic="and",
                    conditions=[
                        ConditionSchema(type="parameter", field="model", operator="exact", value="dynamic_model"),
                    ],
                ),
                action=RerouteActionSchema(target="claude-haiku"),
            ),
        ]
        rules = load_rules(schemas)
        assert len(rules) == 1
        assert rules[0].name == "test-rule"
        assert rules[0].weight == 50
        assert rules[0].action.type == ActionType.REROUTE

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
        schemas = [
            RuleSchema(
                name="reroute-big",
                weight=10,
                conditions=ConditionGroupSchema(
                    logic="and",
                    conditions=[
                        ConditionSchema(type="parameter", field="model", operator="exact", value="dynamic_model"),
                    ],
                ),
                action=RerouteActionSchema(target="claude-haiku"),
            ),
        ]
        rules = load_rules(schemas)
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
        schemas = [
            RuleSchema(
                name="miss",
                weight=10,
                conditions=ConditionGroupSchema(
                    logic="and",
                    conditions=[
                        ConditionSchema(type="parameter", field="model", operator="exact", value="nonexistent"),
                    ],
                ),
                action=BlockActionSchema(message="blocked"),
            ),
        ]
        rules = load_rules(schemas)
        ctx = _ctx(model="claude-sonnet")
        assert evaluate_rules(rules, ctx) is None


# --- YAML config round-trip ---


class TestConfigSchemas:
    def test_reroute_schema(self):
        schema = RuleSchema(
            name="dynamic-routing",
            weight=50,
            description="Route dynamic_model based on token count",
            conditions=ConditionGroupSchema(
                logic="and",
                conditions=[
                    ConditionSchema(type="parameter", field="model", operator="exact", value="dynamic_model"),
                    ConditionSchema(type="function", field="count_tokens", operator="gt", value=10000),
                ],
            ),
            action=RerouteActionSchema(target="claude-haiku"),
        )
        assert schema.action.type == "reroute"
        assert schema.action.target == "claude-haiku"

    def test_rewrite_schema_validates_regex(self):
        with pytest.raises(Exception):
            RewriteActionSchema(pattern="[invalid", replacement="x")

    def test_block_schema_default_message(self):
        schema = BlockActionSchema()
        assert schema.message == "Request blocked by policy"
