# TTLLM - LLM Gateway

## Important Rules
- **Keep README.md in sync**: When changing configuration options, CLI commands, Docker setup, release process, deployment, or any other topic covered in README.md, update the README to reflect those changes.

## Project Overview
LLM gateway exposing an Anthropic-compatible API (`POST /v1/messages`), routing requests through LangChain to any supported provider (Bedrock, OpenAI, etc.). Tracks tokens, costs, and maintains audit trails. Supports user management with per-user model access control.

## Tech Stack
- **Language**: Python 3.12+
- **API**: FastAPI + Pydantic v2
- **Database**: PostgreSQL 16 via SQLAlchemy 2.0 (async) + asyncpg
- **Migrations**: Alembic
- **LLM**: LangChain (langchain-core, langchain-aws)
- **Deployment**: Docker (ECS via Uvicorn, Lambda via Mangum)
- **CLI**: Typer + Rich

## Architecture Principles
- `core/` has **zero framework dependencies** (no FastAPI, no SQLAlchemy) — pure business logic, testable in isolation
- `services/` handles database operations, depends on core + SQLAlchemy
- `api/` is a thin layer composing services via FastAPI dependency injection
- `handlers/` contains deployment adapters (Lambda/ECS) — imports only the app factory
- Anthropic API compatibility is the external contract; LangChain is the internal execution layer

## Key Commands
```bash
# Development
docker-compose up                    # Start PostgreSQL + API
alembic upgrade head                 # Run migrations
pytest                               # Run tests
uvicorn ttllm.handlers.ecs_entrypoint:app --reload  # Dev server

# CLI (admin operations)
ttllm users list|show|create|update|delete
ttllm models list|show|create|update|delete|assign|unassign
ttllm groups list|show|create|update|delete
ttllm tokens list|show|create|delete
ttllm secrets list|show|create|update|delete
ttllm usage summary|costs [--user] [--model] [--since] [--until]
ttllm audit-logs [--user] [--model] [--limit]
```

## Configuration
Settings are loaded from YAML config file → environment variables → defaults.

- `TTLLM_CONFIG_FILE` — Path to YAML config file (e.g. `config.yaml`)
- `TTLLM_CONFIG_ENV` — Environment section to load (default: `dev`)

Nested env vars use `__` delimiter (e.g. `TTLLM_DATABASE__URL`, `TTLLM_ENGINE__LOG_REQUEST_BODIES`).

### Config structure (config.yaml)
```yaml
dev:
  database:
    url: "postgresql+asyncpg://..."
    pool_size: 5
  engine:
    cors_origins: ["*"]
    log_request_bodies: false
  auth:
    jwt:
      secret_key: "..."
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

YAML values support `env://VAR,default` and `secret://arn` resolution patterns.
Local overrides via `local.config.yaml` (git-ignored).

## Project Structure
```
src/ttllm/
├── config.py          # Settings (pydantic-settings)
├── db.py              # Async engine + session factory
├── models/            # SQLAlchemy ORM (users, auth, llm_models, model_assignments, audit_logs, audit_log_bodies)
├── schemas/           # Pydantic v2 (anthropic.py = wire format, auth.py, admin.py, common.py)
├── core/              # Pure logic (permissions, jwt, oidc, password, gateway, translator, provider, token_tracker, streaming)
├── services/          # DB operations (user, auth, group, model, audit, usage)
├── api/               # FastAPI (app.py, deps.py, auth.py, messages.py, admin.py)
├── handlers/          # Lambda (mangum) + ECS (uvicorn) entrypoints
└── cli/               # Typer CLI (main.py)
```

## Database
12 tables: `users`, `llm_models`, `model_assignments`, `group_model_assignments`, `audit_logs`, `audit_log_bodies`, `groups`, `group_permissions`, `user_permissions`, `user_groups`, `gateway_tokens`, `refresh_tokens`, `secrets`
- JWT-based auth with OIDC identity provider support (e.g. Entra ID)
- RBAC via groups and direct user permission assignments
- Gateway tokens are registered in DB for revocation; refresh tokens stored as SHA-256 hashes
- Audit log bodies are in a separate table to keep the main audit table lean
- All PKs are UUIDs, all timestamps are timezone-aware
