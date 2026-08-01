# Content mapping

TTLLM sits between a client-facing wire format (today: the Anthropic Messages API) and one
or more provider wire formats (Bedrock Converse, OpenAI-compatible via LangChain). Rather
than translating directly between those two shapes, requests and responses pass through a
provider-agnostic, API-agnostic internal representation:

```
Anthropic wire JSON  <-->  core/adapters/anthropic.py  <-->  InternalRequest / InternalResult / InternalChunk
                                                                          |
                                                                  core/gateway.py (dispatch only)
                                                                          |
                                              +---------------------------+---------------------------+
                                              |                                                         |
                          core/providers/bedrock/converse.py                    core/providers/langchain/translation.py
                                (Bedrock Converse wire)                                (LangChain messages -> OpenAI wire)
```

The internal representation lives in `src/ttllm/core/model.py`. Nothing under
`core/gateway.py` or `core/providers/` imports the Anthropic wire schema
(`schemas/anthropic.py`) — that coupling is confined entirely to
`core/adapters/anthropic.py` and `core/adapters/anthropic_stream.py`.

Each output provider is its own subpackage under `core/providers/<name>/`: a `provider.py`
(the `BaseProvider`/`ProviderState` pair) plus whatever wire-format translation code it
needs (`core/providers/bedrock/converse.py`, `core/providers/langchain/translation.py` +
`registry.py`). Naming is deliberately explicit about the *mechanism*, not just the target
platform — `translator.py`'s old top-level name didn't say what it translated, or for which
provider, once there were two translation modules in the codebase.

**Why a third layer instead of translating directly, like before:** two independent
providers each hand-rolling their own Anthropic-to-wire-format conversion meant Anthropic's
SSE streaming grammar was implemented twice, and each provider handled the same content
block differently (or not at all) with no obvious place to look up which behavior was
intentional. It also means TTLLM can only ever speak Anthropic on the client-facing side —
adding a second input API (a native Bedrock-shaped endpoint, an OpenAI-compatible incoming
endpoint) would otherwise mean duplicating the whole provider layer. With this structure,
a new input API is one new adapter file; a new output provider is one new
`core/providers/<name>/` subpackage. Neither touches the other.

**Update this file** whenever you add a new input adapter, a new output provider, or a new
`Part` variant — this is the single reference for what each side can represent and how gaps
are handled, and it decays fast if left to comments scattered across provider files.

## The internal model (`core/model.py`)

### Content parts (`Part`, a discriminated union)

| Type | Fields | Notes |
|---|---|---|
| `TextPart` | `text`, `cache_control` | |
| `ImagePart` | `media_type`, `data` (base64), `cache_control` | |
| `DocumentPart` | `media_type`, `data` (base64), `title`, `cache_control` | |
| `ToolCallPart` | `id`, `name`, `input` (dict), `cache_control` | An assistant-turn tool invocation. |
| `ToolResultPart` | `tool_call_id`, `content: list[TextPart \| ImagePart]`, `is_error`, `cache_control` | `content` is always a list — no input adapter should ever hand a provider a bare string here. |
| `ThinkingPart` | `text`, `signature`, `cache_control` | |
| `RedactedThinkingPart` | `data`, `cache_control` | Opaque — the payload is encrypted, not even Anthropic can read it. |
| `ServerToolCallPart` | `id`, `name`, `input` (dict), `cache_control` | A server-side tool invocation (e.g. Anthropic's built-in web_search). |
| `ServerToolResultPart` | `tool_call_id`, `content` (untyped list), `cache_control` | |

`cache_control` is a plain bool on every part. Only the Anthropic input adapter currently
sets it to `True` (and only on `TextPart` today, since that's the only Anthropic content
type with a `cache_control` field), and only the Bedrock provider currently acts on it
(emitting a `cachePoint`). A provider that doesn't support caching simply ignores the flag
— that's a safe no-op, not a lossy drop, unlike the content types below.

**Server-side tool content gets a real `Part`/spec, not a blanket exclusion.** It would be
tempting to reject `server_tool_call`/`server_tool_result` at the input-adapter boundary
outright, since no provider here can proxy them today — but whether a *specific* provider
can handle server-side tools is a provider capability, not something an API-agnostic input
adapter should presume. (A future provider that passes through directly to Anthropic, for
instance, could actually support these.) So the input adapter converts them faithfully like
everything else, and each provider's translation code decides for itself — see "Server-side
tools" below.

### Request / result / streaming

- `InternalRequest`: `provider_model_id`, `messages: list[InternalMessage]` (role is only
  `"user"` or `"assistant"` — a wire format's own system-role quirks are flattened away by
  its input adapter), `system: str | None`, `system_cache_control`, generation params,
  `tools: list[ToolSpec]`, `server_tools: list[ServerToolSpec]` (passthrough `type`/`name`/
  `config`, since each wire format's server tools are their own arbitrary contract), `tool_choice`,
  `thinking`.
- `InternalResult`: `content: list[Part]`, `stop_reason`, `usage`. No `id`/`model` fields —
  those are wire-envelope concerns, synthesized only by the input adapter that builds the
  final response (e.g. `core/adapters/anthropic.py::result_to_response`).
- `InternalChunk`: one item of a provider's streamed output — `BLOCK_START` (carries an
  empty-shell `Part` so a streaming encoder knows the block's type), `TEXT_DELTA`,
  `TOOL_ARGS_DELTA`, `THINKING_DELTA`, `SIGNATURE_DELTA`, `BLOCK_STOP`, `MESSAGE_STOP`
  (carries final `stop_reason` + `usage`), `ERROR`.

## Anthropic input adapter (`core/adapters/anthropic.py`, `core/adapters/anthropic_stream.py`)

| Anthropic concept | Internal representation |
|---|---|
| `TextBlock` | `TextPart` |
| `ImageBlock` | `ImagePart` |
| `DocumentBlock` | `DocumentPart` |
| `ToolUseBlock` | `ToolCallPart` |
| `ToolResultBlock` (`content: str \| list[TextBlock \| ImageBlock]`) | `ToolResultPart` (`content` always normalized to `list[TextPart \| ImagePart]`) |
| `ThinkingBlock` | `ThinkingPart` |
| `RedactedThinkingBlock` | `RedactedThinkingPart` |
| Top-level `system` + mid-conversation `role="system"` messages | Flattened into a single `InternalRequest.system: str` (top-level text first, then each inline system message's text, newline-joined) |
| `cache_control: dict \| None` | `cache_control: bool` |
| `ToolChoiceAuto/Any/Tool/None` | `ToolChoice(mode=..., tool_name=...)` |
| `ServerToolUseBlock` | `ServerToolCallPart` |
| `WebSearchToolResultBlock` | `ServerToolResultPart` |
| `ServerToolDefinition` (in `tools`) | `ServerToolSpec` (`InternalRequest.server_tools`, kept separate from `InternalRequest.tools`) |

`core/adapters/anthropic_stream.py::encode_anthropic_sse` is the only place that constructs
an Anthropic SSE frame (`message_start` -> `ping` -> repeated
`content_block_start`/`_delta`/`_stop` -> `message_delta` -> `message_stop`). Any provider
that produces an `AsyncIterator[InternalChunk]` gets Anthropic-compatible streaming for
free by passing it through here.

### Server-side tools

Requests referencing server-side tools (`server_tool_use`/`web_search_tool_result` content,
or a `ServerToolDefinition` in `tools`) are converted faithfully into `InternalRequest` by
the input adapter — **not** rejected there. Rejection happens per-provider, at translation
time: both `core/providers/bedrock/converse.py::build_converse_request` and
`core/providers/langchain/translation.py::to_langchain_messages` check `request.server_tools` up front and also
match `ServerToolCallPart`/`ServerToolResultPart` wherever they appear in message content
(a stale conversation history could carry one from an earlier turn even if the current
request's `tools` list is empty), raising `ServerToolError` (HTTP 501 `not_implemented_error`)
either way. Today both providers reject unconditionally — neither can proxy server-side
tools — but that's a fact about Bedrock and LangChain/OpenAI specifically, not a fact this
gateway should hard-code above the provider layer. A well-behaved client (e.g. Claude Code)
already knows not to send server tools against a non-Anthropic-native backend, so in
practice this is defense-in-depth rather than something clients are expected to trigger.

## Per-provider mapping

Each provider decides, for every `Part` type it might receive, whether it can (a) represent
it faithfully, (b) **emulate** it with an existing, non-misleading substitute, or (c) must
**error**. The rule: emulate whenever a faithful-enough substitute exists; error only when
proceeding silently would mislead the model or degrade the answer. Silently dropping
content is never acceptable if there was real information in it — the two silent-drop rows
below are deliberate exceptions because there is no real content being lost.

### Bedrock (`core/providers/bedrock/converse.py`)

| Part | Handling |
|---|---|
| `TextPart`, `ImagePart`, `DocumentPart`, `ToolCallPart`, `ToolResultPart`, `ThinkingPart` | Full native support (Bedrock Converse has direct equivalents for all of these). |
| `RedactedThinkingPart` | **Silent drop.** Bedrock has no equivalent, and there's nothing to emulate — the payload is opaque even to Anthropic. This is the one deliberate exception to "always emulate or error." |
| `ServerToolCallPart`, `ServerToolResultPart`, non-empty `server_tools` | **Error** (`ServerToolError`, mapped to HTTP 501 `not_implemented_error`). Bedrock Converse has no mechanism to proxy server-side tools. |

### OpenAI-compatible via LangChain (`core/providers/langchain/translation.py`)

| Part | Handling |
|---|---|
| `TextPart`, `ToolCallPart` (assistant turn), `ToolResultPart` (text-only) | Full native support. |
| `ImagePart` (top-level user-turn content) | Full native support (multimodal `image_url` content part). |
| `ImagePart` inside a `ToolResultPart` | **Emulate.** Passed through as a multimodal content part in the `ToolMessage`, the same mechanism used for a normal image. Previously this crashed (`AttributeError` — the old code assumed every tool-result item had a `.text` attribute); now it just works. |
| `ThinkingPart` | **Emulate.** Folded into a normal text part — no native "thinking" concept on this provider, but the reasoning content itself is preserved rather than silently dropped. |
| `RedactedThinkingPart` | **Silent drop.** Same reasoning as Bedrock — the payload is opaque, there's no content actually being lost. |
| `DocumentPart` | **Error** (`UnsupportedContentError`, mapped to HTTP 400 `invalid_request_error`). No native document/PDF input on OpenAI-style chat APIs, and dropping it silently would mean the model answers as if it had read an attachment it never saw — the "impacts inference quality" case that error, not emulation, exists for. |
| `ToolCallPart` in a non-assistant-turn position | **Best-effort text stub** (`[tool_use: name(args)]`). This is a malformed/edge-case input path — a valid Anthropic conversation never puts a tool call outside an assistant turn — not a real feature gap, so it gets a defensive fallback rather than an error. |
| `ServerToolCallPart`, `ServerToolResultPart`, non-empty `server_tools` | **Error** (`ServerToolError`, mapped to HTTP 501 `not_implemented_error`). No OpenAI-compatible backend this gateway targets can execute Anthropic-style server-side tools. |

### Adding a new output provider

Implement `core/providers/<name>/provider.py` (a `BaseProvider` + `ProviderState`) and
whatever translation module(s) it needs, analogous to `core/providers/bedrock/converse.py`/
`core/providers/langchain/translation.py`, that convert `InternalRequest -> <wire format>`
and `<wire response/stream> -> InternalResult`/`AsyncIterator[InternalChunk]`. Register it
in `core/providers/__init__.py::get_provider`.
Add a row to this file for every `Part` type your provider can't represent natively,
following the emulate-vs-error policy above — don't silently drop real content. Remember to
check `InternalRequest.server_tools` explicitly (it's a separate list, not something you'll
encounter just by walking message content) if your provider can't proxy server-side tools.

### Adding a new input API

Implement `core/adapters/<name>.py` with the same two functions as
`core/adapters/anthropic.py` (`request -> InternalRequest`, `InternalResult -> response`),
plus a streaming encoder if the wire protocol needs one (analogous to
`core/adapters/anthropic_stream.py` — note this is *not* shared with other input APIs,
since each wire protocol has its own framing). `core/gateway.py` and every
`core/providers/*.py` file are untouched by this — that's the point of the internal layer.
