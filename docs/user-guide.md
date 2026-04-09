# TTLLM User Guide

This guide walks you through setting up and using the TTLLM gateway -- from first login to making API calls with Claude and other LLMs.

## Prerequisites

- The `ttllm` CLI installed (`pip install ttllm-gateway`)
- A TTLLM gateway URL provided by your administrator
- An account (email/password or SSO credentials)

## 1. Log In

### Password login

```bash
ttllm login --url https://gateway.example.com
```

You will be prompted for your email and password:

```
Email: alice@example.com
Password: ****
Login successful.
```

### SSO login (Entra ID / OIDC)

```bash
ttllm login --idp entra --url https://gateway.example.com
```

This opens your browser to complete authentication with your identity provider. Once complete, the CLI receives your session automatically.

### Environment variable

You can set the gateway URL once to avoid passing `--url` every time:

```bash
export TTLLM_URL="https://gateway.example.com"
ttllm login
```

### Session storage

Your session is stored locally at `~/.config/ttllm/session.json`. It contains a short-lived access token that refreshes automatically and a long-lived refresh token.

## 2. Check Your Account

```bash
ttllm whoami
```

Output:

```
User: Alice Smith (alice@example.com)
ID: 550e8400-e29b-41d4-a716-446655440000
Groups: users, engineering

Effective permissions (this token):
  - llm.invoke
  - model.view
  - audit.view
```

## 3. See Available Models

Check which models you have access to:

```bash
ttllm me models
```

Output:

```
My Models
+------------------+----------+
| Name             | Provider |
+------------------+----------+
| claude-opus-4-6  | bedrock  |
| claude-haiku-4-5 | bedrock  |
+------------------+----------+
```

The **Name** column is what you use as the `model` parameter in API requests.

## 4. Create a Gateway Token

Gateway tokens are API keys used to authenticate requests to the `/v1/messages` endpoint. They are separate from your CLI session.

```bash
ttllm me tokens create --label "my-app" --ttl-days 90
```

Output:

```
Token created:
  Token: eyJhbGciOiJIUzI1NiIs...
  ID: 550e8400-e29b-41d4-a716-446655440000
  Permissions: llm.invoke
  Label: my-app
  Expires: 2026-07-09T12:34:56Z
Save this token now -- it will not be shown again.
```

**Copy this token immediately.** It is only displayed once and cannot be retrieved later.

### Token options

| Option | Description | Default |
|---|---|---|
| `--label` | Human-readable name for the token | _(none)_ |
| `--ttl-days` | Lifetime in days (max 365) | 30 |
| `--permissions` | Comma-separated permissions | `llm.invoke` |

### Manage tokens

```bash
ttllm me tokens                 # List your active tokens
ttllm me tokens delete <ID>     # Revoke one of your tokens
```

> **Note:** Admins can also manage tokens for any user via `ttllm tokens list|show|create|delete`.

## 5. Test with the CLI

The quickest way to verify everything works:

```bash
ttllm chat "Hello, what model are you?" \
  --model claude-haiku-4-5 \
  --token "eyJhbGciOiJIUzI1NiIs..."
```

Or set the token as an environment variable:

```bash
export TTLLM_TOKEN="eyJhbGciOiJIUzI1NiIs..."
ttllm chat "Hello!" --model claude-opus-4-6 --usage
```

### Chat options

| Option | Description | Default |
|---|---|---|
| `-m, --model` | Model name (required) | -- |
| `-t, --token` | Gateway token (or `TTLLM_TOKEN` env var) | -- |
| `--url` | Gateway URL (or `TTLLM_URL` env var) | `http://localhost:4000` |
| `--max-tokens` | Maximum tokens in response | 1024 |
| `--no-stream` | Disable streaming | false |
| `--usage` | Show token counts after response | false |

## 6. Use the REST API

The gateway exposes an Anthropic-compatible endpoint at `POST /anthropic/v1/messages`.

### Non-streaming request

```bash
curl -X POST https://gateway.example.com/anthropic/v1/messages \
  -H "x-api-key: eyJhbGciOiJIUzI1NiIs..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku-4-5",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 1024
  }'
```

Response:

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "text", "text": "2 + 2 = 4."}
  ],
  "model": "claude-haiku-4-5",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 12,
    "output_tokens": 8
  }
}
```

### Streaming request

Set `"stream": true` in the request body. The response is a Server-Sent Events stream:

```bash
curl -X POST https://gateway.example.com/anthropic/v1/messages \
  -H "x-api-key: eyJhbGciOiJIUzI1NiIs..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku-4-5",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024,
    "stream": true
  }'
```

### Supported parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model name as shown in `ttllm me models` |
| `messages` | array | yes | List of `{role, content}` message objects |
| `max_tokens` | integer | yes | Maximum tokens in the response |
| `system` | string | no | System prompt |
| `temperature` | float | no | Sampling temperature (0.0 - 1.0) |
| `top_p` | float | no | Nucleus sampling threshold |
| `top_k` | integer | no | Top-k sampling |
| `stop_sequences` | array | no | Sequences that stop generation |
| `stream` | boolean | no | Enable streaming (default: false) |
| `tools` | array | no | Tool definitions for tool use |
| `tool_choice` | object | no | Tool selection strategy |

## 7. Use with the Anthropic SDKs

The gateway is a drop-in replacement for the Anthropic API. Point any Anthropic SDK at it by overriding the base URL and API key.

### Python

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="<your-gateway-token>",
    base_url="https://gateway.example.com/anthropic/v1",
)

message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

### TypeScript / JavaScript

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: "<your-gateway-token>",
  baseURL: "https://gateway.example.com/anthropic/v1",
});

const message = await client.messages.create({
  model: "claude-opus-4-6",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Hello!" }],
});

console.log(message.content[0].text);
```

### Environment variables (works with any SDK)

```bash
export ANTHROPIC_API_KEY="<your-gateway-token>"
export ANTHROPIC_BASE_URL="https://gateway.example.com/anthropic/v1"
export ANTHROPIC_MODEL="<one model>"
export ANTHROPIC_MODEL="<second model>"
```

## 8. Use with Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) can be pointed at the TTLLM gateway:

```bash
export ANTHROPIC_API_KEY="<your-gateway-token>"
export ANTHROPIC_BASE_URL="https://gateway.example.com/anthropic/v1"
export ANTHROPIC_MODEL="<one model>"
export ANTHROPIC_MODEL="<second model>"

claude
```

Claude Code will route all requests through the gateway using the models available to your account.
If that works you can probably set the settings in `.claude/settings.local.yaml`

```yaml
{
  "env": {
    "ANTHROPIC_API_KEY": "<your-gateway-token>",
    "ANTHROPIC_BASE_URL": "https://gateway.example.com/anthropic/v1",
    "ANTHROPIC_MODEL": "<one model>",
    "ANTHROPIC_SMALL_FAST_MODEL": "<second model>",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"
  }
}
```

## 9. View Usage and Costs

### Usage summary

```bash
ttllm usage summary --since 2026-01-01
```

### Cost breakdown by model

```bash
ttllm usage costs --since 2026-04-01
```

### Audit log (recent requests)

```bash
ttllm audit-logs --limit 20
```

## 10. Log Out

```bash
ttllm logout
```

This revokes your refresh token on the server and removes the local session file.

## Environment Variables Reference

| Variable | Used by | Default | Description |
|---|---|---|---|
| `TTLLM_URL` | CLI | `http://localhost:4000` | Gateway base URL |
| `TTLLM_TOKEN` | `ttllm chat` | -- | Gateway token for API calls |
| `ANTHROPIC_API_KEY` | SDKs | -- | Gateway token (SDK usage) |
| `ANTHROPIC_BASE_URL` | SDKs | -- | Gateway URL with `/anthropic/v1` suffix |

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Not logged in` | No session or expired | Run `ttllm login` |
| `Missing x-api-key header` | No token in API request | Add `x-api-key` header or set `TTLLM_TOKEN` |
| `Model 'X' is not available` | No access to that model | Ask your admin to assign the model to your user or group |
| `Could not connect` | Wrong URL or gateway down | Check `TTLLM_URL` and verify the gateway is running |
| `Request timed out` | Slow response from LLM provider | Try again or use a faster model |
