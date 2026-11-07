# Quotas

Quotas limit how many requests, tokens, or dollars can be used per interval. A quota is defined
on its own and attached to requests by [rules](rules-engine.md), so one budget can be shared by
several rules, and limits can be changed without touching rules.

This is an administrator-facing feature. Users see the effects (a `429` once a limit is reached)
and can check where they stand with `ttllm me quotas`.

---

## Table of Contents

- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Defining a quota](#defining-a-quota)
- [Attaching quotas with rules](#attaching-quotas-with-rules)
- [How a request is enforced](#how-a-request-is-enforced)
- [What a blocked client sees](#what-a-blocked-client-sees)
- [Worked examples](#worked-examples)
- [Managing quotas](#managing-quotas)
- [Operational notes](#operational-notes)
- [Upgrading from rolling-window quota rules](#upgrading-from-rolling-window-quota-rules)

---

## Quick start

Limit every user to $50 a day:

```bash
ttllm quotas create --name daily-spend --measure cost --limit 50 --window day
ttllm rules create --name apply-daily-spend \
  --conditions '{"logic":"and","conditions":[]}' \
  --apply-quota daily-spend
```

An empty condition list matches every request. A quota does nothing until a rule attaches it.

## Concepts

- **Quota:** a limit on one measure over one interval.
- **Bucket:** one counter of a quota. A quota's `dimensions` decide how many exist: one per user,
  one per user and model, or a single shared one.
- **Reservation:** a request's estimated usage, held while it runs. See
  [why reservations exist](#why-reservations-exist).

## Defining a quota

| Field | Default | Meaning |
|---|---|---|
| `name` | required | Unique name |
| `measure` | required | `requests`, `tokens`, or `cost` (USD) |
| `limit_value` | required | The limit per window. Must be positive |
| `window_type` | required | `calendar` or `session` |
| `window_unit` | — | `hour`, `day`, `week`, or `month`. Calendar windows only |
| `window_duration_seconds` | — | Window length. Session windows only |
| `timezone` | `UTC` | IANA timezone that calendar windows align to |
| `dimensions` | `["user"]` | Any of `user`, `model`, `token`. `[]` means one shared bucket |
| `enforcement` | `soft` | `soft` or `strict`, see [below](#soft-and-strict-enforcement) |
| `on_exceeded` | block with 429 | See [below](#on_exceeded) |
| `enabled` | `true` | A disabled quota is skipped |

### Measures

- **`requests`:** completed requests, including client disconnects. Failed requests don't count.
- **`tokens`:** all tokens the provider reports, including cache reads and writes.
- **`cost`:** the provider-computed cost, the same value as `audit_logs.total_cost`. A model with
  no pricing configured costs nothing, so it never trips a cost quota.

If a request fails after the provider reported usage, that usage was paid for, so it still counts
toward `tokens` and `cost`. If a successful request reports no usage at all, its estimate is
charged instead and a warning is logged.

### Windows

**Calendar** windows follow the clock in the quota's timezone: hours start on the hour, days at
midnight, weeks on Monday, and months on the 1st. A day can be 23 or 25 hours long across a DST
change.

**Session** windows start at a bucket's first request and last a fixed time. Once one expires, the
next starts at the next request, not immediately. A 3-hour session is:

```json
{"window_type": "session", "window_duration_seconds": 10800}
```

Usage resets entirely when a window ends, so a user could spend a full limit at 23:59 and again at
00:01. If that matters, pair a long window with a short one.

Windows roll over lazily when the next request arrives, so idle buckets need no background job.

### Dimensions

| `dimensions` | Result |
|---|---|
| `["user"]` | Each user has their own limit |
| `["user", "model"]` | Each user has a separate limit per model |
| `["token"]` | Each token has its own limit. Meant for gateway tokens: a login token gets a new identity each login |
| `[]` | Everyone the rule matches shares one limit |

Dimensions only use values the server derived from authentication and model resolution. `model` is
the resolved model, so sending a different name that matches a model's `match_pattern` lands in the
same bucket, and renaming a model keeps its bucket. There's no header dimension, because clients
could then get a fresh bucket for every request.

For a shared group budget, use `dimensions: []` and attach it with a `group` condition. For a
per-member limit within a group, use `["user"]` with the same condition.

### Soft and strict enforcement

Tokens and cost aren't known until a response finishes, so each request reserves an estimate:
the request's size divided by 4, plus `max_tokens`, priced at the model's rates when measuring cost.

- **`soft`** admits a request while `used + in_flight < limit`. Large `max_tokens` values aren't
  rejected just because little budget is left, and a limit can be overshot by about one request.
- **`strict`** admits a request only if `used + in_flight + estimate <= limit`, so the limit is never
  exceeded. Requests can be rejected when their worst case doesn't fit, even if they'd have used
  far less, and a `max_tokens` larger than the whole limit can never succeed.

`requests` quotas are exact in both modes.

### `on_exceeded`

Block (the default):

```json
{"action": "block", "status_code": 429, "message": "Daily budget used: {{quota.used}}/{{quota.limit}} USD."}
```

Or send the request to another model:

```json
{"action": "reroute", "target": "claude-haiku"}
```

`message` can use `{{quota.name}}`, `{{quota.measure}}`, `{{quota.used}}`, `{{quota.limit}}`,
`{{quota.in_flight}}`, and `{{quota.retry_after}}`.

A reroute **starts the request over** from what the client sent, with only the model replaced. The
target gets exactly the treatment a request sent to it directly would get: its terminal rules
(block, reroute, rewrite), its access check, and its own quotas. A rewrite rule is applied to the
client's original request, never twice. Nothing is charged to the original model: a denial rolls
back every reservation, so the request counts only against the quotas of the model that serves it.

A quota reroute happens at most once. If the target isn't available to the user, a terminal rule
blocks it, or it's also over its limit (including after a rule reroutes it back to the original
model), the request is blocked. If several quotas are exceeded at once, any quota that blocks beats
any that reroutes. A rerouting quota with no `model` dimension counts usage across models, so it
will still be exceeded after the reroute: use `["user", "model"]`.

Every reroute, whether from a rule or a quota, is recorded on the request's single audit row:

```json
"routing": {
  "requested_model": "claude-opus",
  "hops": [{"by": "quota", "name": "opus-daily", "from": "claude-opus", "to": "claude-haiku"}]
}
```

`metadata_json.routing` is only present when something rerouted. `ttllm audit-logs` shows it under
the model name.

## Attaching quotas with rules

A rule with the `apply_quota` action attaches quotas to the requests it matches:

```json
{
  "name": "opus-budget",
  "conditions": {
    "logic": "and",
    "conditions": [
      {"type": "parameter", "field": "resolved_model", "operator": "exact", "value": "claude-opus"}
    ]
  },
  "action": {"type": "apply_quota", "quota_ids": ["<quota id>"]}
}
```

`apply_quota` rules work differently from other rules:

- **All matching rules apply.** They don't compete by weight, and their quotas are combined.
- **No other rule can skip them.** They're evaluated after the terminal rule has run, so they see the
  request as it will actually be sent. A terminal `allow` does not exempt a request.
- **To exempt someone, add a negated condition** to the rule:

```json
{"type": "subject", "field": "group", "value": "unlimited", "negate": true}
```

Use `subject` conditions and `resolved_model` for targeting, since clients can't influence those.
`header` and `content` conditions work, but a client controls those values and can use them to avoid
the quota, so creating such a rule logs a warning.

## How a request is enforced

1. Terminal rules run: block, reroute, rewrite, or allow.
2. The model is resolved and access is checked.
3. Every matching `apply_quota` rule contributes its quotas.
4. **Reserve**, in one short transaction: lock the buckets, roll over expired windows, check every
   limit, and record the estimate. If any bucket refuses, nothing is reserved. If the refusing quota
   reroutes, go back to step 1 with the client's original request and the new model.
5. The provider is called. No lock or database connection is held during this.
6. **Settle**, in the same transaction as the audit row: replace the estimate with actual usage.

### Why reservations exist

The row lock only makes the check one-at-a-time, and `used` doesn't change until a response
finishes. Without reservations, 100 parallel requests would each see the same remaining budget and
all be admitted. A reservation makes each running request visible to the next one.

If a worker crashes mid-request, its reservation stops counting after
`engine.quota_reservation_ttl_seconds`.

### Requests crossing a window boundary

Usage is charged when a request finishes, to whichever window its bucket has at that moment. A request
that runs across a boundary can be attributed to either side.

## What a blocked client sees

```
HTTP/1.1 429 Too Many Requests
Retry-After: 2537
x-ttllm-quota-name: daily-spend
x-ttllm-quota-limit: 50
x-ttllm-quota-remaining: 0
x-ttllm-quota-reset: 2537

{"type": "error", "error": {"type": "rate_limit_error", "message": "Quota 'daily-spend' exceeded (50.21/50 cost)."}}
```

`Retry-After` is the time until the window ends, using the longest wait if several quotas were
exceeded.

When waiting can't help, because a `strict` estimate is larger than the whole limit, the response is
a `400 invalid_request_error` with no `Retry-After` instead: Anthropic SDKs retry 429s
automatically, which would only repeat the failure. An explicit `on_exceeded.status_code` or
`message` still takes precedence.

Admitted requests carry the same `x-ttllm-quota-*` headers for whichever bucket is closest to its
limit. These headers, `Retry-After` and `X-Request-Id` are exposed to browser scripts through CORS.

Every response, including errors, carries an `X-Request-Id`. It's the `request_id` of the request's
audit row, so `ttllm audit-logs --request-id <id>` finds exactly what happened to it. The blocked
request is written to the audit log, with the quota's details under `metadata_json.quota`.

## Worked examples

### $50/day and $20/hour per user

```bash
ttllm quotas create --name spend-daily --measure cost --limit 50 --window day
ttllm quotas create --name spend-hourly --measure cost --limit 20 --window hour
ttllm rules create --name apply-spend-caps \
  --conditions '{"logic":"and","conditions":[]}' \
  --apply-quota spend-daily --apply-quota spend-hourly
```

### A rolling 3-hour allowance per user and model

```bash
ttllm quotas create --name session-tokens --measure tokens --limit 500000 \
  --session-seconds 10800 --dimension user --dimension model
```

### A monthly budget shared by a team, in their timezone

```bash
ttllm quotas create --name eng-monthly --measure cost --limit 5000 \
  --window month --timezone Europe/Warsaw --shared
ttllm rules create --name apply-eng-monthly \
  --conditions '{"logic":"and","conditions":[{"type":"subject","field":"group","value":"engineering"}]}' \
  --apply-quota eng-monthly
```

### Fall back to a cheaper model after a daily Opus budget

```bash
ttllm quotas create --name opus-daily --measure cost --limit 20 --window day \
  --dimension user --dimension model \
  --on-exceeded '{"action":"reroute","target":"claude-haiku"}'
ttllm rules create --name apply-opus-daily \
  --conditions '{"logic":"and","conditions":[{"type":"parameter","field":"resolved_model","value":"claude-opus"}]}' \
  --apply-quota opus-daily
```

The rule only matches Opus, so once rerouted, the request no longer has this quota attached.

## Managing quotas

### CLI

```bash
ttllm quotas list
ttllm quotas show <name>
ttllm quotas create --name <name> --measure <measure> --limit <n> (--window <unit> | --session-seconds <n>)
ttllm quotas update <name> [--limit <n>] [--enforcement <mode>] [--on-exceeded '<json>'] [--enabled true]
ttllm quotas update <name> --fields '{"window_unit": "week"}'
ttllm quotas delete <name>
ttllm quotas buckets <name>     # current usage per bucket
ttllm quotas reset <name>       # discard all usage

ttllm rules create --name <name> --conditions '<json>' --apply-quota <quota> [--apply-quota <quota>]
ttllm me quotas                 # your own standing
```

### REST API

| Method | Path | Permission |
|---|---|---|
| `GET` | `/admin/quotas?offset=&limit=` | `quota.view` |
| `POST` | `/admin/quotas` | `quota.create` |
| `GET` | `/admin/quotas/{id}` | `quota.view` |
| `PATCH` | `/admin/quotas/{id}` | `quota.modify` |
| `DELETE` | `/admin/quotas/{id}` | `quota.delete` |
| `GET` | `/admin/quotas/{id}/buckets?offset=&limit=` | `quota.view` |
| `POST` | `/admin/quotas/{id}/reset` | `quota.modify` |
| `GET` | `/me/quotas` | any authenticated user |

These permissions are granted to the `administrators` group. Changes are recorded in the admin audit
log.

### Changing a quota

- **`limit_value`, `enforcement`, `on_exceeded`, `enabled`, `name`:** apply immediately and keep usage.
- **`measure`, `dimensions`, or any window field:** these change what a bucket means, so the quota's
  buckets are discarded and usage starts from zero.
- **Deleting a quota** is refused with a `409` while a rule still attaches it.

`/me/quotas` works out which quotas apply without seeing a real request, so rules matching on request
content are best-effort there.

## Operational notes

| Setting (`engine.*`) | Default | Meaning |
|---|---|---|
| `quota_lock_timeout_ms` | `5000` | How long to wait for a bucket lock. Past this the client gets a retryable `529` |
| `quota_reservation_ttl_seconds` | `900` | When a reservation stops counting as in-flight |
| `quota_reservation_retention_seconds` | `86400` | When an expired reservation is deleted. A request finishing before this is still charged |

- **Rule changes** reach other workers within `engine.rules_cache_ttl_seconds`. Quota definitions
  aren't cached, so limit changes apply on the next request.
- **Contention:** a shared bucket's row lock is taken twice per request, briefly each time. Every
  request through one shared bucket queues on that row, so very high request rates are better served
  by per-user buckets.
- **Failure mode:** if the database is unavailable, requests fail rather than passing unlimited.
- **Lambda:** Mangum buffers responses and reports no mid-stream disconnects. A function that times
  out mid-request leaves its reservation to expire, and that request's usage isn't charged.
- **Partial streams:** the OpenAI-compatible provider only reports usage at the end of a stream, so a
  stream failing midway charges nothing.
- **Usage reports:** quota-blocked requests are left out of usage totals, per-model costs and
  per-user usage, and counted separately as `quota_blocked_requests` in the usage summary. Failed
  requests still count as zero-cost requests.
- **Audit volume:** every blocked request writes an audit row, so a client retrying in a loop writes
  one row per attempt. There's no cleanup job for audit rows.
- **Old buckets:** rows for inactive users or models stay in `quota_buckets` until there is a cleanup
  job. They're small and affect nothing.

## Upgrading from rolling-window quota rules

Rules using the old `quota` condition are **disabled by the migration, not converted**: rolling
windows have no exact equivalent, and existing usage isn't carried over. **Those limits are not
enforced until you recreate them**, so do this as part of the upgrade.

1. Find them. The migration logs each rule by name, and adds a marker to its description:
   `ttllm rules list`.
2. Create an equivalent [quota](#defining-a-quota), and attach it with an `apply_quota` rule.
3. Delete the old rule. It can't be re-enabled: that returns a `422`.

| Old | New |
|---|---|
| `"window": 3600` / `86400` | `--window hour` / `--window day`, or `--session-seconds` for a timer started by the first request |
| `"per": ["model"]` | `--dimension user --dimension model` |
| A higher-weight `allow` rule as an exemption | A negated `subject` condition on the `apply_quota` rule |
| `{{quota.cost.value}}` | `{{quota.used}}` |
| `{{quota.cost.threshold}}` | `{{quota.limit}}` |
| `{{quota.cost.next_free}}` | `{{quota.retry_after}}` |
