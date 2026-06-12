# Rules Engine & Quota System

The rules engine evaluates every incoming `POST /anthropic/v1/messages` request
*before* it is dispatched to a provider, and can **reroute**, **block**,
**rewrite**, or explicitly **allow** it based on the request's headers,
parameters, content, computed functions, or the caller's recent usage (quotas).

This is an administrator-facing feature. End users never configure rules; they
only experience the effects (e.g. a `429` when a quota is exceeded). For the
end-user request/response flow, see [user-guide.md](user-guide.md).

---

## Table of Contents

- [Concepts](#concepts)
- [How a request is evaluated](#how-a-request-is-evaluated)
- [Managing rules](#managing-rules)
- [Conditions](#conditions)
- [Operators](#operators)
- [Condition groups](#condition-groups)
- [Actions](#actions)
- [Quota system](#quota-system)
- [Block message templating](#block-message-templating)
- [Worked examples](#worked-examples)
- [Operational notes](#operational-notes)
- [Permissions](#permissions)
- [Troubleshooting](#troubleshooting)

---

## Concepts

A **rule** is a named, weighted record with two JSON parts:

- **conditions** — a (possibly nested) group of predicates joined by `and`/`or`.
- **action** — what to do when the conditions match.

Rules are stored in the `rules` table and managed entirely through the admin API
and CLI — there is no YAML rules file. Each rule has a `weight` (higher =
evaluated first) and an `enabled` flag.

```
Rule
├── name        (unique)
├── weight      (int, higher wins)
├── enabled     (bool)
├── description (optional)
├── conditions  ── ConditionGroup ── [ Condition | ConditionGroup, ... ]
└── action      ── reroute | block | allow | rewrite
```

## How a request is evaluated

1. The endpoint loads the **active** (enabled) rules from an in-process cache.
2. If any rule contains a **quota** condition, the engine runs the necessary
   moving-window usage aggregates and stashes the results on the request context
   (see [Quota system](#quota-system)). When no quota condition exists, **no
   database queries are issued** — there is zero overhead.
3. Rules are evaluated in **descending weight order**. The **first** rule whose
   condition group matches wins (first-match-wins); evaluation stops there.
4. The winning rule's action is applied:
   - **reroute** → the target model name is substituted before model resolution.
   - **block** → the request is rejected with the configured status code and a
     (templated) message.
   - **rewrite** → a regex replace is applied to message/system content.
   - **allow** → the request passes through unchanged (useful as a high-weight
     allow-list entry that pre-empts lower-weight block rules).
5. The (possibly modified) request continues to model access checks and dispatch.

> Because it is first-match-wins by weight, an `allow` rule with a higher weight
> than a `block` rule acts as an exception/allow-list.

## Managing rules

### CLI

```bash
ttllm rules list                                  # table of all rules
ttllm rules show <name>                            # full detail incl. conditions/action JSON
ttllm rules create --name <name> \
  --conditions '<json>' --action '<json>' \
  --weight 50 [--description "..."] [--disabled]
ttllm rules update <name> [--weight 100] [--enabled true] \
  [--name ...] [--description ...] [--conditions '<json>'] [--action '<json>']
ttllm rules delete <name>
```

By default the argument to `show`/`update`/`delete` is the rule **name**; pass
`--use-ids` to treat it as a UUID.

### REST API

| Method | Path | Permission |
|---|---|---|
| `GET` | `/admin/rules?offset=&limit=` | `rule.view` |
| `POST` | `/admin/rules` | `rule.create` |
| `GET` | `/admin/rules/{rule_id}` | `rule.view` |
| `PATCH` | `/admin/rules/{rule_id}` | `rule.modify` |
| `DELETE` | `/admin/rules/{rule_id}` | `rule.delete` |

All create/update/delete operations are recorded in the admin audit log.

## Conditions

A condition is a single predicate:

```json
{"type": "parameter", "field": "model", "operator": "exact", "value": "gpt-x", "negate": false}
```

| Type | `field` values | Matches against |
|---|---|---|
| `parameter` | `model`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, or `metadata.<key>` | Request parameters |
| `header` | any HTTP header name (case-insensitive) | Request headers |
| `content` | `messages` or `system` | The concatenated message text or system prompt |
| `function` | `count_tokens`, `message_length`, `keyword_count` | A computed value (see below) |
| `quota` | `cost`, `tokens`, `requests` | The caller's usage in a rolling window (see [Quota system](#quota-system)) |

Every condition supports `"negate": true` to invert the result.

### Built-in functions

`function` conditions compare a computed integer:

| Function | Returns |
|---|---|
| `count_tokens` | Approximate token count (`(messages + system) length / 4`) |
| `message_length` | Character length of the concatenated messages |
| `keyword_count` | Whitespace-delimited word count of the messages |

> These are heuristics, not provider-exact token counts. Use them for coarse
> guardrails, not billing.

## Operators

`exact`, `regex`, `contains`, `in`, `gt`, `lt`, `gte`, `lte`

- `exact` / `contains` / `regex` / `in` operate on string representations.
  `in` accepts a list (`"value": ["a", "b"]`) or a substring container.
- `gt` / `lt` / `gte` / `lte` coerce both sides to numbers; non-numeric values
  never match.

## Condition groups

Conditions are wrapped in a group with a logic operator, and groups can nest:

```json
{
  "logic": "and",
  "conditions": [
    {"type": "parameter", "field": "model", "operator": "exact", "value": "claude-opus"},
    {
      "logic": "or",
      "conditions": [
        {"type": "content", "field": "messages", "operator": "contains", "value": "confidential"},
        {"type": "header", "field": "x-env", "operator": "exact", "value": "prod"}
      ]
    }
  ]
}
```

`logic` is `and` (all must match) or `or` (any must match).

## Actions

| Action | Fields | Effect |
|---|---|---|
| `reroute` | `target` | Replace the requested model name with `target` before resolution |
| `block` | `message`, `status_code` | Reject the request. `status_code` defaults to `403`; set `429` for rate limiting. `message` supports [templating](#block-message-templating). When a quota window is involved, a `Retry-After` header is added |
| `allow` | — | Pass through unchanged; pre-empts lower-weight rules |
| `rewrite` | `pattern`, `replacement` | Regex-replace within message and system text |

`block.status_code` must be a valid HTTP error status (400–599). `rewrite.pattern`
is validated as a compilable regular expression at create time.

## Quota system

A **quota condition** compares the caller's recent usage against a threshold over
a **moving (sliding) time window**, computed in real time from the audit log.
Only successful (`status_code == 200`) requests count toward usage — blocked or
errored requests do not consume quota.

### Fields

```json
{
  "type": "quota",
  "field": "cost",          // measure: cost | tokens | requests
  "operator": "gt",          // numeric operator
  "value": 5.0,              // threshold
  "window": 60,              // REQUIRED: window size in seconds
  "per": ["model"]           // OPTIONAL: scope dimensions
}
```

| Field | Meaning |
|---|---|
| `field` | The measure: `cost` (USD spent), `tokens` (input + output), or `requests` (count) |
| `value` | The threshold to compare against |
| `window` | **Required.** Trailing window size in seconds (e.g. `60` = the last 60 s, `3600` = the last hour) |
| `per` | **Optional.** Scope dimensions. Currently only `["model"]` is supported, limiting the aggregate to the request's model (exact name match). Default scope is **per-user** |

Use the numeric operators (`gt`, `gte`, `lt`, `lte`).

### Scope and composition

The window is always scoped to the calling **user**. There is no separate "scope"
setting — finer targeting is achieved by combining the quota condition with other
conditions in an `and` group, and/or by adding `per`:

- *"$5/min per user"* → a single `quota.cost` condition.
- *"$5/min for Opus specifically"* → `quota.cost` with `"per": ["model"]`, or an
  `and` group pairing the quota condition with a `parameter.model` condition.

### How `next_free` / Retry-After is computed

When a quota blocks a request, the engine computes how long until the window
frees up from the **oldest contributing usage row**:

```
next_free = (oldest_contributing_request.created_at + window) - now      (floored at 0)
```

This is when usage *starts* dropping (the oldest entry ages out), not a guarantee
that you'll be fully under the threshold — it is a useful lower bound. For a rule
with multiple quota conditions, the `Retry-After` header is the **maximum**
`next_free` across them (you can't retry until the most-constraining window has
room).

### Performance

- One aggregate query runs per **distinct** `(measure, window, per)` across all
  matched rules — two rules referencing `cost` over the same window share one
  query. The same measure over *different* windows runs separate queries.
- Queries are served by the composite index `ix_audit_logs_user_id_created_at`.
- If no active rule has a quota condition, **no query runs at all**.

## Block message templating

A `block` action's `message` supports a tiny, safe `{{ dotted.path }}`
substitution. It is **not** Jinja — only dotted lookups are supported, with no
expressions, filters, or logic. Unresolved references are left in place verbatim.

Quota conditions publish a `quota.<measure>` namespace into the template context:

| Variable | Value |
|---|---|
| `{{quota.<measure>.value}}` | The current windowed usage |
| `{{quota.<measure>.threshold}}` | The configured threshold |
| `{{quota.<measure>.window}}` | The window size in seconds |
| `{{quota.<measure>.next_free}}` | Seconds until the window frees up |

Example message:

```
Spend limit hit ({{quota.cost.value}}/{{quota.cost.threshold}} USD in 60s). Retry in {{quota.cost.next_free}}s.
```

> A rule with two quota conditions on the *same* measure but *different* windows
> is not yet supported — they share one `quota.<measure>` namespace. Different
> measures (e.g. `quota.cost` and `quota.requests`) coexist fine.

## Worked examples

### Reroute large requests to a cheaper model

```json
{
  "name": "reroute-large-to-haiku",
  "weight": 50,
  "description": "Route large requests to a cheaper model",
  "conditions": {
    "logic": "and",
    "conditions": [
      {"type": "parameter", "field": "model", "operator": "exact", "value": "claude-opus"},
      {"type": "function", "field": "count_tokens", "operator": "gt", "value": 50000}
    ]
  },
  "action": {"type": "reroute", "target": "claude-haiku"}
}
```

### Block prompt-injection attempts

```json
{
  "name": "block-jailbreak",
  "weight": 100,
  "conditions": {
    "logic": "or",
    "conditions": [
      {"type": "content", "field": "messages", "operator": "regex",
       "value": "(?i)(ignore previous instructions|DAN mode)"}
    ]
  },
  "action": {"type": "block", "message": "Request rejected: content policy violation"}
}
```

### Redact SSNs in content

```json
{
  "name": "mask-ssn",
  "weight": 80,
  "conditions": {
    "logic": "and",
    "conditions": [
      {"type": "content", "field": "messages", "operator": "regex", "value": "\\d{3}-\\d{2}-\\d{4}"}
    ]
  },
  "action": {"type": "rewrite", "pattern": "\\d{3}-\\d{2}-\\d{4}", "replacement": "[SSN-REDACTED]"}
}
```

### Cost throttle: 429 once a user spends > $5 in any rolling 60 s

```json
{
  "name": "cost-throttle-60s",
  "weight": 100,
  "conditions": {
    "logic": "and",
    "conditions": [
      {"type": "quota", "field": "cost", "operator": "gt", "value": 5.0, "window": 60}
    ]
  },
  "action": {
    "type": "block",
    "status_code": 429,
    "message": "Spend limit hit ({{quota.cost.value}}/{{quota.cost.threshold}} USD in 60s). Retry in {{quota.cost.next_free}}s."
  }
}
```

A throttled response looks like:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 37

{"type": "error", "error": {"type": "policy_error",
  "message": "Spend limit hit (5.42/5.0 USD in 60s). Retry in 37s."}}
```

### Per-model request rate limit

```json
{
  "name": "opus-rate-limit",
  "weight": 90,
  "conditions": {
    "logic": "and",
    "conditions": [
      {"type": "quota", "field": "requests", "operator": "gt", "value": 100,
       "window": 60, "per": ["model"]}
    ]
  },
  "action": {
    "type": "block",
    "status_code": 429,
    "message": "Too many requests to this model. Retry in {{quota.requests.next_free}}s."
  }
}
```

### Creating a rule from the CLI

```bash
ttllm rules create \
  --name cost-throttle-60s \
  --weight 100 \
  --conditions '{"logic":"and","conditions":[{"type":"quota","field":"cost","operator":"gt","value":5.0,"window":60}]}' \
  --action '{"type":"block","status_code":429,"message":"Spend limit hit. Retry in {{quota.cost.next_free}}s."}'
```

### Per-user spend caps: $50/day **and** $20/hour

A common setup is two complementary cost caps that together bound both
short-term bursts and long-term spend. Each is a per-user `quota.cost` rule (no
`per` → scoped to the calling user); the only differences are the `window` and
the threshold:

- **Daily cap** — $50 over a rolling 24 h window (`window: 86400`).
- **Hourly cap** — $20 over a rolling 1 h window (`window: 3600`).

Both rules can match the same request. Because evaluation is **first-match-wins
by weight**, give the cap you want *reported first* the higher weight. Here the
daily cap is weighted above the hourly cap, so a user who has blown both limits
sees the daily message; a user who is only over the hourly limit sees the hourly
message. (Either way the request is blocked — the weight only decides which
message and `Retry-After` are returned.)

```json
// Rule 1 — daily cap, higher weight
{
  "name": "cost-cap-daily-50usd",
  "weight": 110,
  "description": "Block once a user spends more than $50 in any rolling 24h window",
  "conditions": {
    "logic": "and",
    "conditions": [
      {"type": "quota", "field": "cost", "operator": "gt", "value": 50.0, "window": 86400}
    ]
  },
  "action": {
    "type": "block",
    "status_code": 429,
    "message": "Daily spend limit reached ({{quota.cost.value}}/{{quota.cost.threshold}} USD). Resets in {{quota.cost.next_free}}s."
  }
}

// Rule 2 — hourly cap, lower weight
{
  "name": "cost-cap-hourly-20usd",
  "weight": 100,
  "description": "Block once a user spends more than $20 in any rolling 1h window",
  "conditions": {
    "logic": "and",
    "conditions": [
      {"type": "quota", "field": "cost", "operator": "gt", "value": 20.0, "window": 3600}
    ]
  },
  "action": {
    "type": "block",
    "status_code": 429,
    "message": "Hourly spend limit reached ({{quota.cost.value}}/{{quota.cost.threshold}} USD). Retry in {{quota.cost.next_free}}s."
  }
}
```

Create both from the CLI:

```bash
# Daily cap: $50 over a rolling 24h window (86400s), weighted highest.
ttllm rules create \
  --name cost-cap-daily-50usd \
  --weight 110 \
  --description "Block once a user spends more than \$50 in any rolling 24h window" \
  --conditions '{"logic":"and","conditions":[{"type":"quota","field":"cost","operator":"gt","value":50.0,"window":86400}]}' \
  --action '{"type":"block","status_code":429,"message":"Daily spend limit reached ({{quota.cost.value}}/{{quota.cost.threshold}} USD). Resets in {{quota.cost.next_free}}s."}'

# Hourly cap: $20 over a rolling 1h window (3600s).
ttllm rules create \
  --name cost-cap-hourly-20usd \
  --weight 100 \
  --description "Block once a user spends more than \$20 in any rolling 1h window" \
  --conditions '{"logic":"and","conditions":[{"type":"quota","field":"cost","operator":"gt","value":20.0,"window":3600}]}' \
  --action '{"type":"block","status_code":429,"message":"Hourly spend limit reached ({{quota.cost.value}}/{{quota.cost.threshold}} USD). Retry in {{quota.cost.next_free}}s."}'
```

These caps apply to **every** user automatically — quota conditions are scoped to
the calling user, so a single pair of rules enforces the same per-user budget
across your whole user base. To exempt or special-case a particular user, add a
higher-weight `allow` rule that matches them (e.g. on an `x-user-tier` header or
a `metadata.*` parameter) so it pre-empts the caps.

> **Note on overlap:** the two windows are independent rolling aggregates, not a
> hierarchy. A burst can trip the hourly cap well before the daily cap, and the
> daily cap keeps holding even after the hourly window has cleared. Together they
> bound both *rate* (hourly) and *total* (daily) spend.

## Operational notes

### Rule cache and propagation

Active rules are cached in memory per worker process. The cache is refreshed when
it goes stale, governed by `engine.rules_cache_ttl_seconds` (default **30 s**).

- The worker that serves a create/update/delete invalidates its own cache
  immediately.
- Other worker processes pick up the change within the TTL.

So a rule change can take up to `rules_cache_ttl_seconds` to take effect across
all workers. Lower the TTL for faster propagation at the cost of more frequent
reloads; see the `engine` block in your `config.yaml`.

### Cost accuracy

Quota `cost` aggregates sum the authoritative provider-computed `total_cost`
stored on each audit row (already accounting for cache and provider-specific cost
dimensions). `tokens`/`requests` are exact counts. The `function`-based
`count_tokens` condition, by contrast, is only a rough heuristic.

## Permissions

| Permission | Grants |
|---|---|
| `rule.view` | List and show rules |
| `rule.create` | Create rules |
| `rule.modify` | Update rules |
| `rule.delete` | Delete rules |

These are granted to the `administrators` group by default.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Missing required permission: rule.create` | Caller lacks `rule.*` | Grant the permission or use an admin account |
| Rule change not taking effect | Per-worker cache TTL | Wait up to `rules_cache_ttl_seconds`, or lower it |
| Quota never trips | Window too short, or only failed requests in the window (only `200`s count) | Widen `window`; verify successful requests exist in the audit log |
| Quota trips immediately for a new rule | Existing in-window usage already exceeds the threshold | Expected — the window looks back over recent history |
| `{{quota...}}` shows literally in the message | The referenced path doesn't exist (e.g. typo, or no quota condition in the rule) | Use a `quota.<measure>` path that matches a quota condition in the same rule |
| `block status_code must be an HTTP error status (400-599)` | Out-of-range `status_code` | Use a value between 400 and 599 |
```
