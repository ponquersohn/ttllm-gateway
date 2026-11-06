"""End-to-end quotas against the running gateway: admin API, rules, HTTP behaviour.

Each test creates its own user and scopes its rule to that user with a ``subject`` condition,
so nothing here can throttle other tests, even while running.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import contextmanager

import httpx

from tests.integration.conftest import BASE_URL, FAKE_BEDROCK_URL


def _unique(prefix: str) -> str:
    return f"{prefix}-{os.getpid()}-{time.monotonic_ns()}"


def _new_user(client: httpx.Client, admin_headers: dict, model_ids: list[str]) -> tuple[str, str, str]:
    """Create a user assigned to the given models. Returns (token, email, user_id)."""
    email = f"{_unique('quota-user')}@example.com"
    u = client.post(
        "/admin/users",
        headers=admin_headers,
        json={"name": "Quota User", "email": email, "password": "Integration1!"},
    )
    assert u.status_code == 201, u.text
    user_id = u.json()["id"]
    p = client.post(
        f"/admin/users/{user_id}/permissions", headers=admin_headers, json={"permissions": ["llm.invoke"]}
    )
    assert p.status_code == 201, p.text
    for model_id in model_ids:
        a = client.post(f"/admin/models/{model_id}/assign", headers=admin_headers, json={"user_ids": [user_id]})
        assert a.status_code == 201, a.text
    t = client.post(
        "/admin/tokens",
        headers=admin_headers,
        json={"user_id": user_id, "label": "quota-test", "permissions": ["llm.invoke"]},
    )
    assert t.status_code == 201, t.text
    return t.json()["access_token"], email, user_id


def _new_model(client: httpx.Client, admin_headers: dict, endpoint: str = FAKE_BEDROCK_URL) -> dict:
    resp = client.post(
        "/admin/models",
        headers=admin_headers,
        json={
            "name": _unique("quota-model"),
            "provider": "bedrock",
            "provider_model_id": "anthropic.claude-sonnet-4-20250514-v1:0",
            "input_cost_per_1k": "0.003",
            "output_cost_per_1k": "0.015",
            "config_json": {
                "region": "us-east-1",
                "endpoint_url": endpoint,
                "aws_access_key_id": "test",
                "aws_secret_access_key": "test",
            },
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _send(client: httpx.Client, token: str, model_name: str, **overrides) -> httpx.Response:
    return client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": token},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "quota check"}],
            "max_tokens": 64,
            **overrides,
        },
    )


@contextmanager
def _quotas(client: httpx.Client, admin_headers: dict, email: str, *definitions: dict):
    """Create quotas and attach each to ``email`` through its own rule; clean up afterwards."""
    quotas, rules = [], []
    try:
        for definition in definitions:
            body = {
                "name": _unique("quota"),
                "measure": "requests",
                "limit_value": "2",
                "window_type": "calendar",
                "window_unit": "day",
                **definition,
            }
            created = client.post("/admin/quotas", headers=admin_headers, json=body)
            assert created.status_code == 201, created.text
            quotas.append(created.json())
            rule = client.post(
                "/admin/rules",
                headers=admin_headers,
                json={
                    "name": _unique("quota-rule"),
                    "conditions": {
                        "logic": "and",
                        "conditions": [{"type": "subject", "field": "user_email", "value": email}],
                    },
                    "action": {"type": "apply_quota", "quota_ids": [created.json()["id"]]},
                },
            )
            assert rule.status_code == 201, rule.text
            rules.append(rule.json())
        yield quotas
    finally:
        for rule in rules:
            client.delete(f"/admin/rules/{rule['id']}", headers=admin_headers)
        for quota in quotas:
            client.delete(f"/admin/quotas/{quota['id']}", headers=admin_headers)


def _error(resp: httpx.Response) -> dict:
    return resp.json()["error"]


def test_request_quota_blocks_with_429_and_exact_retry_after(client, admin_headers, bedrock_model):
    token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(
        client,
        admin_headers,
        email,
        {
            "limit_value": "2",
            "window_type": "session",
            "window_unit": None,
            "window_duration_seconds": 3600,
            "on_exceeded": {"message": "Limit {{quota.limit}}, used {{quota.used}}. Retry in {{quota.retry_after}}s."},
        },
    ) as (quota,):
        for _ in range(2):
            ok = _send(client, token, bedrock_model["name"])
            assert ok.status_code == 200, ok.text
            assert ok.headers["x-ttllm-quota-name"] == quota["name"]

        blocked = _send(client, token, bedrock_model["name"])
        assert blocked.status_code == 429, blocked.text
        assert _error(blocked)["type"] == "rate_limit_error"
        retry_after = int(blocked.headers["Retry-After"])
        assert 3500 < retry_after <= 3600
        assert _error(blocked)["message"] == f"Limit 2, used 2. Retry in {retry_after}s."
        assert blocked.headers["x-ttllm-quota-remaining"] == "0"


def test_parallel_requests_cannot_exceed_the_limit(client, admin_headers, bedrock_model):
    token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(client, admin_headers, email, {"limit_value": "5"}):
        statuses: list[int] = []

        def worker() -> None:
            with httpx.Client(base_url=BASE_URL, timeout=60) as own_client:
                statuses.append(_send(own_client, token, bedrock_model["name"]).status_code)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert sorted(set(statuses)) == [200, 429]
    assert statuses.count(200) == 5


def test_daily_and_hourly_caps_are_enforced_independently(client, admin_headers, bedrock_model):
    """Regression: the old engine compared both caps against whichever window ran last."""
    token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(
        client,
        admin_headers,
        email,
        {"name": _unique("day"), "limit_value": "2", "window_unit": "day"},
        {"name": _unique("hour"), "limit_value": "100", "window_unit": "hour"},
    ) as (daily, hourly):
        for _ in range(2):
            assert _send(client, token, bedrock_model["name"]).status_code == 200

        blocked = _send(client, token, bedrock_model["name"])
        assert blocked.status_code == 429
        assert blocked.headers["x-ttllm-quota-name"] == daily["name"]

        # Each kept its own count.
        hourly_buckets = client.get(f"/admin/quotas/{hourly['id']}/buckets", headers=admin_headers).json()
        assert hourly_buckets["items"][0]["used"] == "2"


def test_higher_weight_rule_cannot_skip_a_quota(client, admin_headers, bedrock_model):
    token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    rewrite = client.post(
        "/admin/rules",
        headers=admin_headers,
        json={
            "name": _unique("rewrite"),
            "weight": 100000,
            "conditions": {
                "logic": "and",
                "conditions": [
                    {"type": "subject", "field": "user_email", "value": email},
                    {"type": "content", "field": "messages", "operator": "contains", "value": "quota"},
                ],
            },
            "action": {"type": "rewrite", "pattern": "quota", "replacement": "budget"},
        },
    )
    assert rewrite.status_code == 201, rewrite.text
    try:
        with _quotas(client, admin_headers, email, {"limit_value": "1"}):
            assert _send(client, token, bedrock_model["name"]).status_code == 200
            assert _send(client, token, bedrock_model["name"]).status_code == 429
    finally:
        client.delete(f"/admin/rules/{rewrite.json()['id']}", headers=admin_headers)


def test_session_window_rolls_over_lazily(client, admin_headers, bedrock_model):
    token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(
        client,
        admin_headers,
        email,
        {"limit_value": "1", "window_type": "session", "window_unit": None, "window_duration_seconds": 2},
    ):
        assert _send(client, token, bedrock_model["name"]).status_code == 200
        blocked = _send(client, token, bedrock_model["name"])
        assert blocked.status_code == 429
        time.sleep(int(blocked.headers["Retry-After"]) + 0.1)
        assert _send(client, token, bedrock_model["name"]).status_code == 200


def test_tokens_and_cost_are_settled_from_actual_usage(client, admin_headers):
    model = _new_model(client, admin_headers)
    token, email, _ = _new_user(client, admin_headers, [model["id"]])
    with _quotas(
        client,
        admin_headers,
        email,
        {"measure": "tokens", "limit_value": "1000000"},
        {"measure": "cost", "limit_value": "1000"},
    ) as (tokens_quota, cost_quota):
        resp = _send(client, token, model["name"], max_tokens=4096)
        assert resp.status_code == 200, resp.text
        usage = resp.json()["usage"]
        expected_tokens = usage["input_tokens"] + usage["output_tokens"]

        bucket = client.get(f"/admin/quotas/{tokens_quota['id']}/buckets", headers=admin_headers).json()["items"][0]
        # The 4096 max_tokens reservation was replaced by what was actually used.
        assert bucket["used"] == str(expected_tokens)
        assert bucket["in_flight"] == "0"

        bucket = client.get(f"/admin/quotas/{cost_quota['id']}/buckets", headers=admin_headers).json()["items"][0]
        assert float(bucket["used"]) > 0


def test_streaming_is_settled(client, admin_headers, bedrock_model):
    token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(client, admin_headers, email, {"measure": "tokens", "limit_value": "1000000"}) as (quota,):
        with client.stream(
            "POST",
            "/anthropic/v1/messages",
            headers={"x-api-key": token},
            json={
                "model": bedrock_model["name"],
                "messages": [{"role": "user", "content": "quota check"}],
                "max_tokens": 64,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["x-ttllm-quota-name"] == quota["name"]
            for _ in resp.iter_bytes():
                pass

        bucket = client.get(f"/admin/quotas/{quota['id']}/buckets", headers=admin_headers).json()["items"][0]
        assert int(bucket["used"]) > 0
        assert bucket["in_flight"] == "0"


def test_provider_error_refunds_the_slot(client, admin_headers):
    broken = _new_model(client, admin_headers, endpoint="http://127.0.0.1:1")
    token, email, _ = _new_user(client, admin_headers, [broken["id"]])
    with _quotas(client, admin_headers, email, {"limit_value": "1"}) as (quota,):
        for _ in range(2):
            resp = _send(client, token, broken["name"])
            assert resp.status_code >= 500, resp.text
        bucket = client.get(f"/admin/quotas/{quota['id']}/buckets", headers=admin_headers).json()["items"][0]
        assert bucket["used"] == "0"
        assert bucket["in_flight"] == "0"


def test_reroute_on_exceeded(client, admin_headers):
    primary = _new_model(client, admin_headers)
    fallback = _new_model(client, admin_headers)
    token, email, _ = _new_user(client, admin_headers, [primary["id"], fallback["id"]])
    with _quotas(
        client,
        admin_headers,
        email,
        {
            "limit_value": "1",
            "dimensions": ["user", "model"],
            "on_exceeded": {"action": "reroute", "target": fallback["name"]},
        },
    ):
        assert _send(client, token, primary["name"]).json()["model"] == primary["name"]
        rerouted = _send(client, token, primary["name"])
        assert rerouted.status_code == 200, rerouted.text
        assert rerouted.json()["model"] == fallback["name"]
        # The fallback has now used up its own bucket too, and reroutes only happen once.
        blocked = _send(client, token, primary["name"])
        assert blocked.status_code == 429, blocked.text


def test_reroute_leaves_a_quota_scoped_to_the_original_model(client, admin_headers):
    """The documented fallback: once rerouted, the rule no longer matches, so the request
    isn't held to the exhausted quota, and neither are later requests to the fallback."""
    primary = _new_model(client, admin_headers)
    fallback = _new_model(client, admin_headers)
    token, _, _ = _new_user(client, admin_headers, [primary["id"], fallback["id"]])
    quota = client.post(
        "/admin/quotas",
        headers=admin_headers,
        json={
            "name": _unique("scoped"),
            "measure": "requests",
            "limit_value": "1",
            "window_type": "calendar",
            "window_unit": "day",
            "on_exceeded": {"action": "reroute", "target": fallback["name"]},
        },
    ).json()
    rule = client.post(
        "/admin/rules",
        headers=admin_headers,
        json={
            "name": _unique("scoped-rule"),
            "conditions": {
                "logic": "and",
                "conditions": [{"type": "parameter", "field": "resolved_model", "value": primary["name"]}],
            },
            "action": {"type": "apply_quota", "quota_ids": [quota["id"]]},
        },
    )
    assert rule.status_code == 201, rule.text
    try:
        assert _send(client, token, primary["name"]).json()["model"] == primary["name"]
        for _ in range(3):
            rerouted = _send(client, token, primary["name"])
            assert rerouted.status_code == 200, rerouted.text
            assert rerouted.json()["model"] == fallback["name"]
    finally:
        client.delete(f"/admin/rules/{rule.json()['id']}", headers=admin_headers)
        client.delete(f"/admin/quotas/{quota['id']}", headers=admin_headers)


def test_match_pattern_names_share_one_model_bucket(client, admin_headers):
    """Regression: the old engine filtered on the requested name, so pattern-matched requests
    never tripped a per-model limit, and clients could vary the name to get fresh buckets."""
    name = _unique("pattern-model")
    created = client.post(
        "/admin/models",
        headers=admin_headers,
        json={
            "name": name,
            "provider": "bedrock",
            "provider_model_id": "anthropic.claude-sonnet-4-20250514-v1:0",
            "match_pattern": f"{name}.*",
            "input_cost_per_1k": "0.003",
            "output_cost_per_1k": "0.015",
            "config_json": {
                "region": "us-east-1",
                "endpoint_url": FAKE_BEDROCK_URL,
                "aws_access_key_id": "test",
                "aws_secret_access_key": "test",
            },
        },
    )
    assert created.status_code == 201, created.text
    token, email, _ = _new_user(client, admin_headers, [created.json()["id"]])
    with _quotas(client, admin_headers, email, {"limit_value": "2", "dimensions": ["user", "model"]}):
        assert _send(client, token, f"{name}-variant-a").status_code == 200
        assert _send(client, token, f"{name}-variant-b").status_code == 200
        assert _send(client, token, f"{name}-variant-c").status_code == 429


def test_reroute_to_a_model_the_user_cannot_use_blocks(client, admin_headers):
    primary = _new_model(client, admin_headers)
    forbidden = _new_model(client, admin_headers)
    token, email, _ = _new_user(client, admin_headers, [primary["id"]])
    with _quotas(
        client,
        admin_headers,
        email,
        {"limit_value": "1", "on_exceeded": {"action": "reroute", "target": forbidden["name"]}},
    ):
        assert _send(client, token, primary["name"]).status_code == 200
        assert _send(client, token, primary["name"]).status_code == 429


def test_limit_change_applies_immediately(client, admin_headers, bedrock_model):
    token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(client, admin_headers, email, {"limit_value": "1"}) as (quota,):
        assert _send(client, token, bedrock_model["name"]).status_code == 200
        assert _send(client, token, bedrock_model["name"]).status_code == 429
        patched = client.patch(f"/admin/quotas/{quota['id']}", headers=admin_headers, json={"limit_value": "2"})
        assert patched.status_code == 200, patched.text
        assert patched.json()["limit_value"] == "2"
        assert _send(client, token, bedrock_model["name"]).status_code == 200


def test_group_shared_budget_and_negated_exemption(client, admin_headers, bedrock_model):
    token_a, email_a, user_a = _new_user(client, admin_headers, [bedrock_model["id"]])
    token_b, _, user_b = _new_user(client, admin_headers, [bedrock_model["id"]])
    token_vip, email_vip, user_vip = _new_user(client, admin_headers, [bedrock_model["id"]])

    group = client.post("/admin/groups", headers=admin_headers, json={"name": _unique("quota-group")})
    assert group.status_code == 201, group.text
    group_id, group_name = group.json()["id"], group.json()["name"]
    for user_id in (user_a, user_b, user_vip):
        added = client.post(f"/admin/groups/{group_id}/members", headers=admin_headers, json={"user_ids": [user_id]})
        assert added.status_code in (200, 201), added.text

    quota = client.post(
        "/admin/quotas",
        headers=admin_headers,
        json={
            "name": _unique("group-budget"),
            "measure": "requests",
            "limit_value": "1",
            "dimensions": [],
            "window_type": "calendar",
            "window_unit": "day",
        },
    ).json()
    rule = client.post(
        "/admin/rules",
        headers=admin_headers,
        json={
            "name": _unique("group-rule"),
            "conditions": {
                "logic": "and",
                "conditions": [
                    {"type": "subject", "field": "group", "value": group_name},
                    {"type": "subject", "field": "user_email", "value": email_vip, "negate": True},
                ],
            },
            "action": {"type": "apply_quota", "quota_ids": [quota["id"]]},
        },
    )
    assert rule.status_code == 201, rule.text
    try:
        assert _send(client, token_a, bedrock_model["name"]).status_code == 200
        # B shares A's budget, and the exempted member isn't limited at all.
        assert _send(client, token_b, bedrock_model["name"]).status_code == 429
        for _ in range(2):
            assert _send(client, token_vip, bedrock_model["name"]).status_code == 200
    finally:
        client.delete(f"/admin/rules/{rule.json()['id']}", headers=admin_headers)
        client.delete(f"/admin/quotas/{quota['id']}", headers=admin_headers)


def test_me_quotas_shows_standing(client, admin_headers, bedrock_model):
    token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(client, admin_headers, email, {"limit_value": "5", "dimensions": ["user", "model"]}) as (quota,):
        headers = {"Authorization": f"Bearer {token}"}
        before = client.get("/me/quotas", headers=headers)
        assert before.status_code == 200, before.text
        assert [(q["quota_name"], q["used"], q["remaining"]) for q in before.json()] == [(quota["name"], "0", "5")]

        assert _send(client, token, bedrock_model["name"]).status_code == 200

        after = client.get("/me/quotas", headers=headers).json()
        assert [(q["used"], q["remaining"]) for q in after] == [("1", "4")]
        assert after[0]["dimension_values"]["model_name"] == bedrock_model["name"]
        assert after[0]["window_end"] is not None


def test_blocked_request_is_audited_with_the_quota(client, admin_headers, bedrock_model):
    token, email, user_id = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(client, admin_headers, email, {"limit_value": "1"}) as (quota,):
        assert _send(client, token, bedrock_model["name"]).status_code == 200
        assert _send(client, token, bedrock_model["name"]).status_code == 429

    logs = client.get("/admin/audit-logs", headers=admin_headers, params={"user_id": user_id, "limit": 50})
    assert logs.status_code == 200, logs.text
    blocked = [row for row in logs.json()["items"] if row["status_code"] == 429]
    assert len(blocked) == 1
    assert blocked[0]["metadata_json"]["quota"]["name"] == quota["name"]


class TestAdminValidation:
    def test_invalid_quota_rejected(self, client, admin_headers):
        resp = client.post(
            "/admin/quotas",
            headers=admin_headers,
            json={"name": _unique("bad"), "measure": "requests", "limit_value": "-1", "window_type": "calendar"},
        )
        assert resp.status_code == 422

    def test_duplicate_name_conflicts(self, client, admin_headers):
        body = {
            "name": _unique("dup"),
            "measure": "requests",
            "limit_value": "1",
            "window_type": "calendar",
            "window_unit": "day",
        }
        first = client.post("/admin/quotas", headers=admin_headers, json=body)
        assert first.status_code == 201, first.text
        try:
            assert client.post("/admin/quotas", headers=admin_headers, json=body).status_code == 409
        finally:
            client.delete(f"/admin/quotas/{first.json()['id']}", headers=admin_headers)

    def test_invalid_merged_patch_rejected(self, client, admin_headers):
        created = client.post(
            "/admin/quotas",
            headers=admin_headers,
            json={
                "name": _unique("patch"),
                "measure": "requests",
                "limit_value": "1",
                "window_type": "calendar",
                "window_unit": "day",
            },
        ).json()
        try:
            resp = client.patch(
                f"/admin/quotas/{created['id']}", headers=admin_headers, json={"window_type": "session"}
            )
            assert resp.status_code == 422
        finally:
            client.delete(f"/admin/quotas/{created['id']}", headers=admin_headers)

    def test_rule_referencing_unknown_quota_rejected(self, client, admin_headers):
        resp = client.post(
            "/admin/rules",
            headers=admin_headers,
            json={
                "name": _unique("rule"),
                "conditions": {"logic": "and", "conditions": []},
                "action": {"type": "apply_quota", "quota_ids": [str(uuid.uuid4())]},
            },
        )
        assert resp.status_code == 422

    def test_quota_in_use_cannot_be_deleted(self, client, admin_headers, bedrock_model):
        _, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
        with _quotas(client, admin_headers, email, {}) as (quota,):
            resp = client.delete(f"/admin/quotas/{quota['id']}", headers=admin_headers)
            assert resp.status_code == 409
            assert "rule" in _error(resp)["message"]

    def test_reset_clears_usage(self, client, admin_headers, bedrock_model):
        token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
        with _quotas(client, admin_headers, email, {"limit_value": "1"}) as (quota,):
            assert _send(client, token, bedrock_model["name"]).status_code == 200
            assert _send(client, token, bedrock_model["name"]).status_code == 429
            reset = client.post(f"/admin/quotas/{quota['id']}/reset", headers=admin_headers)
            assert reset.status_code == 200
            assert reset.json()["buckets_reset"] == 1
            assert _send(client, token, bedrock_model["name"]).status_code == 200

    def test_legacy_quota_rule_cannot_be_created(self, client, admin_headers):
        resp = client.post(
            "/admin/rules",
            headers=admin_headers,
            json={
                "name": _unique("legacy"),
                "conditions": {
                    "logic": "and",
                    "conditions": [{"type": "quota", "field": "cost", "operator": "gt", "value": 5, "window": 60}],
                },
                "action": {"type": "block", "status_code": 429},
            },
        )
        assert resp.status_code == 422


def test_no_quota_means_no_quota_headers(client, gateway_user_token, bedrock_model):
    resp = _send(client, gateway_user_token, bedrock_model["name"])
    assert resp.status_code == 200, resp.text
    assert "x-ttllm-quota-name" not in resp.headers


# --- Routing: quota reroutes restart the pipeline and are traceable by request id ---


@contextmanager
def _terminal_rule(client: httpx.Client, admin_headers: dict, email: str, model_name: str, action: dict):
    """A block/reroute/rewrite rule for one user's requests to one model."""
    rule = client.post(
        "/admin/rules",
        headers=admin_headers,
        json={
            "name": _unique("terminal-rule"),
            "conditions": {
                "logic": "and",
                "conditions": [
                    {"type": "subject", "field": "user_email", "value": email},
                    {"type": "parameter", "field": "model", "value": model_name},
                ],
            },
            "action": action,
        },
    )
    assert rule.status_code == 201, rule.text
    try:
        yield rule.json()
    finally:
        client.delete(f"/admin/rules/{rule.json()['id']}", headers=admin_headers)


def _audit_row(client: httpx.Client, admin_headers: dict, request_id: str) -> dict:
    logs = client.get("/admin/audit-logs", headers=admin_headers, params={"request_id": request_id})
    assert logs.status_code == 200, logs.text
    assert logs.json()["total"] == 1, logs.json()
    return logs.json()["items"][0]


def _reroute_quota(fallback_name: str) -> dict:
    return {
        "limit_value": "1",
        "dimensions": ["user", "model"],
        "on_exceeded": {"action": "reroute", "target": fallback_name},
    }


def test_reroute_is_audited_once_and_charged_only_to_the_fallback(client, admin_headers):
    primary = _new_model(client, admin_headers)
    fallback = _new_model(client, admin_headers)
    token, email, _ = _new_user(client, admin_headers, [primary["id"], fallback["id"]])
    with _quotas(client, admin_headers, email, _reroute_quota(fallback["name"])) as (quota,):
        assert _send(client, token, primary["name"]).status_code == 200
        rerouted = _send(client, token, primary["name"])
        assert rerouted.status_code == 200, rerouted.text

        row = _audit_row(client, admin_headers, rerouted.headers["X-Request-Id"])
        assert row["model_name"] == fallback["name"]
        assert row["metadata_json"]["routing"] == {
            "requested_model": primary["name"],
            "hops": [{"by": "quota", "name": quota["name"], "from": primary["name"], "to": fallback["name"]}],
        }

        buckets = client.get(f"/admin/quotas/{quota['id']}/buckets", headers=admin_headers).json()["items"]
        used = {b["dimension_values"]["model_name"]: b["used"] for b in buckets}
        # The denied attempt reserved nothing on the primary; the fallback paid for the request.
        assert used == {primary["name"]: "1", fallback["name"]: "1"}


def test_unrerouted_request_has_no_routing_trail(client, admin_headers, bedrock_model):
    token, _, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    resp = _send(client, token, bedrock_model["name"])
    assert resp.status_code == 200, resp.text
    row = _audit_row(client, admin_headers, resp.headers["X-Request-Id"])
    assert "routing" not in row["metadata_json"]


def test_quota_reroute_reruns_block_rules_for_the_fallback(client, admin_headers):
    primary = _new_model(client, admin_headers)
    fallback = _new_model(client, admin_headers)
    token, email, _ = _new_user(client, admin_headers, [primary["id"], fallback["id"]])
    block = {"type": "block", "message": "fallback is off limits", "status_code": 403}
    with _quotas(client, admin_headers, email, _reroute_quota(fallback["name"])), _terminal_rule(
        client, admin_headers, email, fallback["name"], block
    ):
        assert _send(client, token, primary["name"]).status_code == 200
        blocked = _send(client, token, primary["name"])
        assert blocked.status_code == 403, blocked.text
        assert _error(blocked) == {"type": "policy_error", "message": "fallback is off limits"}
        assert "X-Request-Id" in blocked.headers


def test_quota_reroute_reruns_reroute_rules_and_records_every_hop(client, admin_headers):
    primary = _new_model(client, admin_headers)
    fallback = _new_model(client, admin_headers)
    final = _new_model(client, admin_headers)
    token, email, _ = _new_user(client, admin_headers, [primary["id"], fallback["id"], final["id"]])
    with _quotas(client, admin_headers, email, _reroute_quota(fallback["name"])) as (quota,), _terminal_rule(
        client, admin_headers, email, fallback["name"], {"type": "reroute", "target": final["name"]}
    ) as rule:
        assert _send(client, token, primary["name"]).status_code == 200
        resp = _send(client, token, primary["name"])
        assert resp.status_code == 200, resp.text
        assert resp.json()["model"] == final["name"]

        routing = _audit_row(client, admin_headers, resp.headers["X-Request-Id"])["metadata_json"]["routing"]
        assert routing == {
            "requested_model": primary["name"],
            "hops": [
                {"by": "quota", "name": quota["name"], "from": primary["name"], "to": fallback["name"]},
                {"by": "rule", "name": rule["name"], "from": fallback["name"], "to": final["name"]},
            ],
        }


def test_rule_reroute_back_to_the_exhausted_model_ends_in_a_block(client, admin_headers):
    primary = _new_model(client, admin_headers)
    fallback = _new_model(client, admin_headers)
    token, email, _ = _new_user(client, admin_headers, [primary["id"], fallback["id"]])
    with _quotas(client, admin_headers, email, _reroute_quota(fallback["name"])), _terminal_rule(
        client, admin_headers, email, fallback["name"], {"type": "reroute", "target": primary["name"]}
    ):
        assert _send(client, token, primary["name"]).status_code == 200
        blocked = _send(client, token, primary["name"])
        assert blocked.status_code == 429, blocked.text
        row = _audit_row(client, admin_headers, blocked.headers["X-Request-Id"])
        assert [h["by"] for h in row["metadata_json"]["routing"]["hops"]] == ["quota", "rule"]


def test_quota_reroute_reapplies_rewrites_to_the_original_request(client, admin_headers):
    """The restart starts from the client's request, so a rewrite never applies twice."""
    primary = _new_model(client, admin_headers)
    fallback = _new_model(client, admin_headers)
    token, email, _ = _new_user(client, admin_headers, [primary["id"], fallback["id"]])
    rewrite = {"type": "rewrite", "pattern": "quota check", "replacement": "quota check, rewritten"}
    with _quotas(client, admin_headers, email, _reroute_quota(fallback["name"])), _terminal_rule(
        client, admin_headers, email, fallback["name"], rewrite
    ):
        assert _send(client, token, primary["name"]).status_code == 200
        resp = _send(client, token, primary["name"])
        assert resp.status_code == 200, resp.text
        row = _audit_row(client, admin_headers, resp.headers["X-Request-Id"])
        body = client.get(f"/admin/audit-logs/{row['id']}/body", headers=admin_headers).json()
        assert body["request_body"]["messages"][0]["content"] == "quota check, rewritten"


def test_request_that_can_never_fit_is_a_400_without_retry_after(client, admin_headers, bedrock_model):
    token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(
        client, admin_headers, email, {"measure": "tokens", "limit_value": "10", "enforcement": "strict"}
    ):
        resp = _send(client, token, bedrock_model["name"])
        assert resp.status_code == 400, resp.text
        assert _error(resp)["type"] == "invalid_request_error"
        assert "can never fit" in _error(resp)["message"]
        assert "Retry-After" not in resp.headers
        assert "X-Request-Id" in resp.headers


def test_usage_excludes_quota_blocks_and_counts_them_separately(client, admin_headers, bedrock_model):
    token, email, user_id = _new_user(client, admin_headers, [bedrock_model["id"]])
    with _quotas(client, admin_headers, email, {"limit_value": "1"}):
        assert _send(client, token, bedrock_model["name"]).status_code == 200
        for _ in range(3):
            assert _send(client, token, bedrock_model["name"]).status_code == 429

    params = {"user_id": user_id}
    summary = client.get("/admin/usage", headers=admin_headers, params=params).json()
    assert (summary["total_requests"], summary["quota_blocked_requests"]) == (1, 3)
    costs = client.get("/admin/usage/costs", headers=admin_headers, params=params).json()
    assert [item["request_count"] for item in costs] == [1]


def test_error_responses_carry_the_request_id(client, admin_headers, bedrock_model):
    token, _, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    resp = _send(client, token, "no-such-model")
    assert resp.status_code == 403
    assert uuid.UUID(resp.headers["X-Request-Id"])


def test_quota_headers_are_exposed_to_browsers(client, bedrock_model):
    resp = client.post(
        "/anthropic/v1/messages",
        headers={"Origin": "https://app.example.com", "x-api-key": "invalid"},
        json={"model": bedrock_model["name"], "messages": [], "max_tokens": 1},
    )
    exposed = {h.strip().lower() for h in resp.headers.get("access-control-expose-headers", "").split(",")}
    assert {"retry-after", "x-request-id", "x-ttllm-quota-remaining"} <= exposed
