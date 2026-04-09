# TTLLM Gateway

LLM gateway exposing an Anthropic-compatible API (`POST /v1/messages`), routing requests through LangChain to any supported provider (Bedrock, OpenAI, etc.). Tracks tokens, costs, and maintains audit trails. Supports user management with per-user model access control.

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
  secrets:
    encryption_key: "env://TTLLM_SECRETS_ENCRYPTION_KEY"
```

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

Releases are created from the `main` branch. The Makefile bumps the version in `src/ttllm/__init__.py` and shows the commands to complete the release:

```bash
make release         # Patch bump (v0.0.1 -> v0.0.2)
make release-minor   # Minor bump (v0.1.0 -> v0.2.0)
make release-major   # Major bump (v1.0.0 -> v2.0.0)
```

After running `make release*`, follow the printed instructions to commit, tag, push, and create the GitHub release. Publishing a GitHub release triggers the CI workflow to:

1. **Validate** that the git tag matches the `__version__` in code
2. **Publish** the Python package to [PyPI](https://pypi.org/project/ttllm-gateway/)
3. **Build and push** the Docker image to `ghcr.io/ponquersohn/ttllm-gateway`

## User Guide

For end-user documentation covering login, token creation, API usage, SDK integration, and Claude Code setup, see [docs/user-guide.md](docs/user-guide.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```
