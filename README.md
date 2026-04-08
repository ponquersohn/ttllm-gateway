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
```

## CLI

Admin operations via the `ttllm` CLI:

```bash
ttllm users list|create|deactivate|create-key
ttllm models list|add|assign|unassign
ttllm usage [--user] [--model] [--since] [--until]
ttllm audit-logs [--user] [--model] [--limit]
```

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

## Development

```bash
pip install -e ".[dev]"
pytest
```
