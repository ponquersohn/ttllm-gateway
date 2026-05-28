# TTLLM Gateway

LLM gateway exposing an Anthropic-compatible API (`POST /v1/messages`), routing requests to any supported provider (Bedrock via direct boto3 Converse API, OpenAI-compatible via LangChain). Tracks tokens, costs, and maintains audit trails. Supports user management with per-user model access control.

## Supported Features

| Feature | Bedrock | OpenAI-compatible |
|---------|---------|-------------------|
| Text messages | Yes | Yes |
| Multi-turn conversations | Yes | Yes |
| System prompts | Yes | Yes |
| Streaming (SSE) | Yes | Yes |
| Tool use (client-defined) | Yes | Yes |
| Image inputs (base64) | Yes | Yes |
| Document inputs (PDF) | Yes | No |
| Extended thinking | Yes | No |
| Token tracking & cost | Yes | Yes |
| Cache token reporting | Yes | No |
| Server-side tools | 501 (not proxied) | 501 (not proxied) |

### Architecture Note

Bedrock requests are handled via direct boto3 `converse()` / `converse_stream()` calls with full Anthropic-to-Bedrock format translation. This eliminates the LangChain translation layer for Bedrock, reducing latency and enabling native support for extended thinking, document inputs, and cache token reporting. OpenAI-compatible providers (Ollama, vLLM, etc.) continue to use LangChain.

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 16
- Docker (optional)

### Run with Docker Compose

```bash
docker-compose up
```

This starts PostgreSQL and the API on port 8000. Migrations run automatically on container start.

A default admin account is created by the migrations:

- **Email:** `admin@localhost`
- **Password:** value of `TTLLM_ADMIN_PASSWORD` (defaults to `admin`)

Set `TTLLM_ADMIN_PASSWORD` before running migrations to use a custom password. Log in via `ttllm login` and change the password or create a new admin user immediately.

### Run from Docker Image

```bash
# From GitHub Container Registry
docker run -p 8000:8000 \
  -e TTLLM_DATABASE__URL="postgresql+asyncpg://user:pass@host:5432/ttllm" \
  ghcr.io/ponquersohn/ttllm-gateway:latest
```

#### Passing configuration

**Option A** - Mount a config file:

```bash
docker run -p 8000:8000 \
  -v /path/to/config.yaml:/app/config.yaml \
  -e TTLLM_CONFIG_FILE=/app/config.yaml \
  -e TTLLM_CONFIG_ENV=prod \
  ghcr.io/ponquersohn/ttllm-gateway:latest
```

**Option B** - Environment variables only:

```bash
docker run -p 8000:8000 \
  -e TTLLM_DATABASE__URL="postgresql+asyncpg://user:pass@host:5432/ttllm" \
  -e TTLLM_AUTH__JWT__SECRET_KEY="your-secret" \
  -e TTLLM_ENGINE__LISTEN_PORT=8000 \
  -e TTLLM_PROVIDER__DEFAULT_REGION="us-east-1" \
  ghcr.io/ponquersohn/ttllm-gateway:latest
```

The container listens on port 8000 by default (configurable via `engine.listen_port`). Map it to any host port with `-p <host>:<container>`.

#### Debugging failed containers

By default the container exits on error. Set `TTLLM_EXIT_ON_ERROR=false` to keep the container alive after a failure, so you can exec into it for debugging:

```bash
docker run -p 8000:8000 \
  -e TTLLM_EXIT_ON_ERROR=false \
  -e TTLLM_DATABASE__URL="postgresql+asyncpg://user:pass@host:5432/ttllm" \
  ghcr.io/ponquersohn/ttllm-gateway:latest
```

### Install from PyPI

```bash
pip install ttllm-gateway
```

### Run Locally

```bash
pip install -e .
alembic upgrade head
uvicorn ttllm.handlers.ecs_entrypoint:app --reload
```

## Configuration

Settings are resolved in order: YAML config file -> environment variables -> defaults.

| Environment Variable | Description | Default |
|---|---|---|
| `TTLLM_CONFIG_FILE` | Path to YAML config file | _(none)_ |
| `TTLLM_CONFIG_ENV` | Environment section to load | `dev` |
| `TTLLM_DATABASE__URL` | PostgreSQL connection string | `postgresql+asyncpg://ttllm:dev@localhost:5432/ttllm` |
| `TTLLM_ENGINE__LISTEN_PORT` | Server listen port | `8000` |
| `TTLLM_ENGINE__BASE_URL` | External-facing URL (for OAuth callbacks) | `http://localhost:4000` |
| `TTLLM_ENGINE__CORS_ORIGINS` | Allowed CORS origins | `["*"]` |
| `TTLLM_AUTH__JWT__SECRET_KEY` | JWT signing secret | `CHANGE-ME-IN-PRODUCTION` |
| `TTLLM_PROVIDER__DEFAULT_REGION` | AWS region for Bedrock | `us-east-1` |
| `TTLLM_SECRETS__ENCRYPTION_KEY` | Fernet key for encrypting secrets | _(none)_ |

Nested env vars use `__` as delimiter. YAML values support `env://VAR,default` and `secret://arn` resolution patterns. Local overrides via `local.config.yaml` (git-ignored).

### Config file example

```yaml
dev:
  database:
    url: "postgresql+asyncpg://ttllm:dev@localhost:5432/ttllm"
    pool_size: 5
  engine:
    base_url: "http://localhost:8000"
    listen_port: 8000
    cors_origins: ["*"]
    log_request_bodies: false
  auth:
    allowed_redirect_origins:
      - "https://myapp.example.com"
    jwt:
      secret_key: "dev-secret"
      algorithm: "HS256"
      access_token_ttl_minutes: 15
    identity_providers:
      entra:
        name: "Entra ID"
        type: "oidc"
        client_id: "..."
  provider:
    default_region: "us-east-1"
    allowed_base_urls:                          # regex patterns for custom base_url targets
      - "http://ollama\\..*:11434/v1"
    allow_private_targets: false                # set true to allow private/internal IPs
  secrets:
    encryption_key: "env://TTLLM_SECRETS_ENCRYPTION_KEY"
```

## Rules Engine

The gateway includes a rules engine that evaluates incoming requests before model resolution. Rules are evaluated by weight (highest first); the first matching rule's action is applied.

Rules are managed via the admin API (`/admin/rules`) and CLI (`ttllm rules`). They are cached in memory and automatically reloaded when created, updated, or deleted.

### Permissions

Rules management requires dedicated permissions:

- `rule.view` — List and show rules
- `rule.create` — Create new rules
- `rule.modify` — Update existing rules
- `rule.delete` — Delete rules

### API

```bash
# List rules
GET /admin/rules

# Create a rule
POST /admin/rules

# Get/update/delete a specific rule
GET    /admin/rules/{rule_id}
PATCH  /admin/rules/{rule_id}
DELETE /admin/rules/{rule_id}
```

### CLI

```bash
ttllm rules list
ttllm rules show <name>
ttllm rules create --name <name> --conditions '<json>' --action '<json>' --weight 50
ttllm rules update <name> --weight 100 --enabled true
ttllm rules delete <name>
```

### Example: Create a rule via API

```json
POST /admin/rules
{
  "name": "reroute-large-to-haiku",
  "weight": 50,
  "description": "Route large requests to a cheaper model",
  "conditions": {
    "logic": "and",
    "conditions": [
      {"type": "parameter", "field": "model", "operator": "exact", "value": "dynamic_model"},
      {"type": "function", "field": "count_tokens", "operator": "gt", "value": 50000}
    ]
  },
  "action": {"type": "reroute", "target": "claude-haiku"}
}
```

**More examples:**

```json
// Block jailbreak attempts (weight: 100 = high priority)
{
  "name": "block-jailbreak",
  "weight": 100,
  "conditions": {
    "logic": "or",
    "conditions": [
      {"type": "content", "field": "messages", "operator": "regex", "value": "(?i)(ignore previous instructions|DAN mode)"}
    ]
  },
  "action": {"type": "block", "message": "Request rejected: content policy violation"}
}

// Mask SSN patterns in content
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

### Condition Types

| Type | Field | Description |
|------|-------|-------------|
| `parameter` | `model`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream` | Match on request parameters |
| `header` | any header name | Match on HTTP headers (case-insensitive) |
| `content` | `messages` or `system` | Match on message/system text |
| `function` | `count_tokens`, `message_length`, `keyword_count` | Match on computed values |

### Operators

`exact`, `regex`, `contains`, `in`, `gt`, `lt`, `gte`, `lte`

All conditions support `negate: true` to invert the match.

### Actions

| Action | Fields | Description |
|--------|--------|-------------|
| `reroute` | `target` | Change target model name before resolution |
| `block` | `message` | Reject request with 403 |
| `allow` | — | Explicitly pass through (skip remaining rules) |
| `rewrite` | `pattern`, `replacement` | Regex replace in message content |

### Condition Groups

Conditions can be composed with `logic: and` or `logic: or`, and groups can be nested for complex rules.

## Secrets Management

Provider credentials (AWS keys, API keys, etc.) can be stored encrypted in the database and
referenced from model configs using `secret://name`. This avoids storing plaintext credentials
in `config_json`.

### Setup

1. Generate an encryption key and add it to your config:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Add to `config.yaml`:

```yaml
dev:
  secrets:
    encryption_key: "your-generated-key"
```

Or via environment variable: `TTLLM_SECRETS__ENCRYPTION_KEY`.

2. Create secrets:

```bash
ttllm secrets create --name aws-bedrock-key        # prompts for value (hidden)
ttllm secrets create --name aws-bedrock-secret      # prompts for value (hidden)
```

3. Reference secrets in model config:

```bash
ttllm models create \
  --name claude-sonnet \
  --provider bedrock \
  --provider-model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --config '{"aws_access_key_id":"secret://aws-bedrock-key","aws_secret_access_key":"secret://aws-bedrock-secret","region":"us-west-2"}'
```

At runtime, `secret://` references are resolved transparently before the provider client is created. Secret values are never exposed through the API or CLI.

## OpenAI-Compatible Providers (Ollama, vLLM, etc.)

Any service exposing an OpenAI-compatible `/v1` endpoint (Ollama, vLLM, LiteLLM, etc.) works with the built-in `openai` provider — no dedicated provider needed.

### Setup

1. Whitelist the target URL and enable private-network access in `config.yaml`:

```yaml
dev:
  provider:
    allowed_base_urls:
      - "http://ollama\\.mynetwork\\.internal:11434/v1"
    allow_private_targets: true   # required when the target is on a private network
```

`allowed_base_urls` entries are regex patterns matched with `re.fullmatch`. Metadata endpoints (169.254.169.254, etc.) are always blocked regardless of `allow_private_targets`.

2. Register the model:

```bash
ttllm models create \
  --name llama3-local \
  --provider openai \
  --provider-model-id llama3 \
  --config '{"base_url":"http://ollama.mynetwork.internal:11434/v1","api_key":"unused"}'
```

3. Assign the model to users/groups as usual:

```bash
ttllm models assign llama3-local --user alice
```

Requests to this model are routed through Ollama's OpenAI-compatible API and tracked the same as any other provider.

## Model Name Matching

By default, a request's `model` field must exactly match the `name` of a registered model. For more flexible matching, you can attach a regex pattern to a model via `--match-pattern`:

```bash
ttllm models create \
  --name claude-haiku \
  --provider bedrock \
  --provider-model-id anthropic.claude-haiku-4-5-20241022-v1:0 \
  --match-pattern 'claude-haiku-4\.5.*'
```

Now any request with a model string starting with `claude-haiku-4.5` (e.g. `claude-haiku-4.5-20241022`, `claude-haiku-4.5-latest`) will resolve to this model.

**Rules:**
- Exact name match always takes priority over regex.
- Patterns use Python `re.fullmatch` semantics — the entire model string must match.
- Invalid regex patterns are rejected at creation time.
- To clear a pattern: `ttllm models update <name> --match-pattern ""`

## Model Pricing

Each model carries per-1K-token prices used to compute the cost recorded in audit logs:

```bash
ttllm models create \
  --name claude-sonnet \
  --provider bedrock \
  --provider-model-id anthropic.claude-sonnet-4-20250514-v1:0 \
  --input-cost 0.003 \
  --output-cost 0.015 \
  --cache-read-cost 0.0003 \
  --cache-write-cost 0.00375
```

- `--input-cost` / `--output-cost` — price per 1K fresh input and output tokens.
- `--cache-read-cost` / `--cache-write-cost` — price per 1K prompt-cache read and write tokens (Bedrock). Cache-read tokens are billed at this rate **instead of** the input rate, not in addition to it. Defaults to `0` when unset, so existing models bill cache reads at no extra cost until prices are configured.

All four are also accepted by `ttllm models update` with the same flags.

The total cost of each request is computed by its provider (the cost shape is provider-specific — Bedrock bills input + output + cache read/write) and stored authoritatively on the audit row, alongside a `provider_metadata` JSONB blob holding the raw usage payload and the per-component cost breakdown. Usage aggregation (`ttllm usage summary` / `costs`) sums the stored totals rather than recomputing, so reported costs always match what was recorded — including cache and any future cost dimensions. `ttllm usage summary` reports an overall `total_cost`.

## Bedrock Model Config

Bedrock models accept these keys in `config_json` (all optional):

- `region` — AWS region; falls back to `provider.default_region`.
- `aws_profile` — named profile, or `aws_access_key_id` / `aws_secret_access_key` / `aws_session_token` for explicit credentials (use `secret://` references for the secret values).
- `endpoint_url` — override the Bedrock runtime endpoint. Useful for VPC interface endpoints, LocalStack, or pointing tests at a fake Bedrock server. Omit to use the AWS default endpoint for the region.

```bash
ttllm models create \
  --name claude-sonnet \
  --provider bedrock \
  --provider-model-id anthropic.claude-sonnet-4-20250514-v1:0 \
  --config '{"region":"us-east-1","endpoint_url":"https://bedrock-runtime.us-east-1.amazonaws.com"}'
```

## CLI

Admin operations via the `ttllm` CLI:

```bash
ttllm status                         # Show server version, status, and config checks
ttllm whoami                         # Show current user, groups, and permissions
ttllm me models                      # List models available to you
ttllm me tokens                      # List your active tokens
ttllm me tokens create               # Create a token for yourself
ttllm me tokens delete <id>          # Revoke one of your tokens
ttllm users list|show|create|update|delete
ttllm models list|show|create|update|delete|assign|unassign
ttllm groups list|show|create|update|delete
ttllm tokens list|show|create|delete
ttllm secrets list|show|create|update|delete
ttllm usage summary|costs [--user] [--model] [--since] [--until]
ttllm reports generate [--user] [--since] [--until] [--format pdf|html] [-o file]  # preview
ttllm audit-logs [--user] [--model] [--limit]
```

### Self-Service Endpoints

Any authenticated user (including gateway-only users) can access the `/me` endpoints to discover their available models and manage their own tokens:

| Endpoint | Description |
|---|---|
| `GET /me` | Current user info, groups, and permissions |
| `GET /me/models` | Models assigned to you (direct + group) |
| `GET /me/tokens` | Your active tokens |
| `POST /me/tokens` | Create a token scoped to your permissions |
| `DELETE /me/tokens/{id}` | Revoke one of your tokens |

### Status Checks

`ttllm status` (and `GET /admin/status`) runs health checks against the current configuration and reports their results:

| Check | Condition | Status |
|---|---|---|
| `encryption_key` | Valid Fernet key configured | `ok` |
| `encryption_key` | Empty or invalid | `error` |
| `jwt_secret` | Custom value | `ok` |
| `jwt_secret` | Still using `CHANGE-ME-IN-PRODUCTION` | `warning` |
| `database` | `SELECT 1` succeeds | `ok` |
| `database` | Connection fails | `error` |

The overall status is `ok` when all checks pass, or `degraded` when any check returns `warning` or `error`.

## Releasing

Releases must be cut from reviewed code: the released commit has to be on `main` or a `release/*` branch. Both are protected (a reviewed PR is required to land code), and the release workflow verifies this before publishing — a release whose commit is not contained in one of those branches fails the publish job. The version is derived automatically from git tags (via `hatch-vcs`), so no source file needs editing.

```bash
make release         # Patch bump (v0.0.1 -> v0.0.2)
make release-minor   # Minor bump (v0.1.0 -> v0.2.0)
make release-major   # Major bump (v1.0.0 -> v2.0.0)
```

After running `make release*`, follow the printed instructions to push the tag and create the GitHub release. Publishing a GitHub release triggers the CI workflow to:

1. **Publish** the Python package to [PyPI](https://pypi.org/project/ttllm-gateway/) (only if the release commit is on `main` or a `release/*` branch)
2. **Build and push** the Docker image to `ghcr.io/ponquersohn/ttllm-gateway`

## Self-Service Web UI

A browser-based UI is available at `/ui` for self-service tasks without needing the CLI or raw API calls.

### Features

- **Login** with email/password or SSO (configured identity providers are detected automatically)
- **View models** assigned to your account
- **Manage tokens** — create new API tokens and revoke existing ones

### Access

Navigate to `http://localhost:8000/ui` (or your deployed base URL + `/ui`). The UI uses only `/me/` endpoints — no admin access is exposed.

Authentication state is stored in `sessionStorage`, so it is scoped to the browser tab and cleared when the tab is closed.

### Public API

The endpoint `GET /auth/identity-providers` returns the list of configured identity providers (slug, name, type) without requiring authentication. This is used by the UI to render SSO buttons.

## User Guide

For end-user documentation covering login, token creation, API usage, SDK integration, and Claude Code setup, see [docs/user-guide.md](docs/user-guide.md).

## Development

```bash
pip install -e ".[dev]"
pytest                       # unit tests (integration tests are excluded by default)
```

### Integration tests

End-to-end tests run the real gateway + PostgreSQL + a fake Bedrock server (which speaks the
actual boto3 `converse` / `converse_stream` wire protocol, including AWS event-stream framing)
via docker-compose, then exercise the full flow: create user → create model → assign → mint
token → `POST /v1/messages` (streaming and non-streaming).

```bash
docker compose -f docker-compose.integration.yml up -d --build
pytest tests/integration -m integration     # hits http://localhost:8000
docker compose -f docker-compose.integration.yml down -v
```

The fake Bedrock is reached by the gateway at its compose-internal URL
(`http://fake-bedrock:9099`), configured per-model via `config_json.endpoint_url` (see below).
If host port 8000 is busy, set `TTLLM_HOST_PORT` and point the tests at it with
`TTLLM_TEST_BASE_URL`. These tests also run automatically in CI (`.github/workflows/integration.yml`).
