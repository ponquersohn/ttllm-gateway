# Rules Engine

The rules engine evaluates every incoming `POST /anthropic/v1/messages` request
*before* it is dispatched to a provider, and can **reroute**, **block**,
**rewrite**, or explicitly **allow** it based on the request's headers,
parameters, content, computed functions, or who is calling. Rules also attach
[quotas](quotas.md) to requests.

This is an administrator-facing feature. End users never configure rules; they
only experience the effects. For the end-user request/response flow, see
[user-guide.md](user-guide.md).

---

## Table of Contents

- [Concepts](#concepts)
- [How a request is evaluated](#how-a-request-is-evaluated)
- [Managing rules](#managing-rules)
- [Conditions](#conditions)
- [Operators](#operators)
- [Condition groups](#condition-groups)
- [Actions](#actions)
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
└── action      ── reroute | block | allow | rewrite | apply_quota
```

## How a request is evaluated

1. The endpoint loads the **active** (enabled) rules from an in-process cache.
2. **Terminal rules** are evaluated in **descending weight order**. The **first**
   rule whose condition group matches wins (first-match-wins); evaluation stops
   there. Its action is applied:
   - **reroute** → the target model name is substituted before model resolution.
   - **block** → the request is rejected with the configured status code and a
     (templated) message.
   - **rewrite** → a regex replace is applied to message/system content.
   - **allow** → the request passes through unchanged (useful as a high-weight
     allow-list entry that pre-empts lower-weight block rules).
3. The model is resolved and access is checked.
4. **`apply_quota` rules** are evaluated against the request as it will actually be
   sent. Every matching rule applies, whatever its weight; see [quotas.md](quotas.md).
   If an exceeded quota reroutes, the pipeline starts again at step 2 with the client's
   original request and the new model, so terminal rules apply to the fallback too.
5. The request is dispatched. Every reroute is recorded in the audit row's
   `metadata_json.routing`.

> Because it is first-match-wins by weight, an `allow` rule with a higher weight
> than a `block` rule acts as an exception/allow-list. It doesn't exempt a request
> from quotas.

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
| `parameter` | `model`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, or `resolved_model` (`apply_quota` rules only) | Request parameters |
| `header` | any HTTP header name (case-insensitive) | Request headers |
| `content` | `messages` or `system` | The concatenated message text or system prompt |
| `function` | `count_tokens`, `message_length`, `keyword_count` | A computed value (see below) |
| `subject` | `user_id`, `user_email`, `token_id`, `group` | The authenticated caller. `group` matches if any of the caller's groups match |

Every condition supports `"negate": true` to invert the result.

`subject` conditions come from authentication, so unlike `header` and `content`
conditions, clients can't influence them. Use them for anything security-relevant.

`resolved_model` is the model actually serving the request, which can differ from
`model` when a rule reroutes or a `match_pattern` matches. It's only known after
resolution, so only `apply_quota` rules can use it.

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
| `block` | `message`, `status_code` | Reject the request. `status_code` defaults to `403`. `message` supports [templating](#block-message-templating) |
| `allow` | — | Pass through unchanged; pre-empts lower-weight rules |
| `rewrite` | `pattern`, `replacement` | Regex-replace within message and system text |
| `apply_quota` | `quota_ids` | Attach [quotas](quotas.md). Not a terminal action: all matching rules apply |

`block.status_code` must be a valid HTTP error status (400–599). `rewrite.pattern`
is validated as a compilable regular expression at create time.

## Block message templating

A `block` action's `message` supports a tiny, safe `{{ dotted.path }}`
substitution. It is **not** Jinja — only dotted lookups are supported, with no
expressions, filters, or logic. Unresolved references are left in place verbatim.

Rule block messages have no variables published to them today. Quota messages have
their own: see [`on_exceeded`](quotas.md#on_exceeded).

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

### Creating a rule from the CLI

```bash
ttllm rules create \
  --name block-jailbreak \
  --weight 100 \
  --conditions '{"logic":"or","conditions":[{"type":"content","field":"messages","operator":"regex","value":"(?i)ignore previous instructions"}]}' \
  --action '{"type":"block","message":"Request rejected: content policy violation"}'
```

### Attaching a quota

```bash
ttllm rules create \
  --name apply-daily-spend \
  --conditions '{"logic":"and","conditions":[]}' \
  --apply-quota daily-spend
```

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
| `422` enabling a rule | The rule uses the removed `quota` condition, or its definition is otherwise invalid | Recreate it as a [quota](quotas.md#upgrading-from-rolling-window-quota-rules) |
| A `{{...}}` shows literally in the message | The referenced path doesn't exist | Fix the path, or remove it |
| `block status_code must be an HTTP error status (400-599)` | Out-of-range `status_code` | Use a value between 400 and 599 |
