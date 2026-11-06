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
| `TextPart` | `text`, `citations: list[Citation] \| None`, `cache_control` | `citations` is set on model output that cites a supplied document/search result. |
| `ImagePart` | `media_type`, `data` (base64), `cache_control` | |
| `DocumentPart` | `source_kind` (`base64` \| `text` \| `content`), `media_type`, `data` (base64), `text`, `content: list[TextPart \| ImagePart]`, `title`, `context`, `citations` (bool), `cache_control` | `source_kind` says which of `data`/`text`/`content` holds the document. |
| `SearchResultPart` | `source`, `title`, `content: list[TextPart]`, `citations` (bool), `cache_control` | Caller-supplied search result (RAG); citable like a document. |
| `ToolCallPart` | `id`, `name`, `input` (dict), `cache_control` | An assistant-turn tool invocation. |
| `ToolResultPart` | `tool_call_id`, `content: list[TextPart \| ImagePart \| DocumentPart \| SearchResultPart]`, `is_error`, `cache_control` | `content` is always a list — no input adapter should ever hand a provider a bare string here. |
| `ThinkingPart` | `text`, `signature`, `cache_control` | |
| `RedactedThinkingPart` | `data`, `cache_control` | Opaque — the payload is encrypted, not even Anthropic can read it. |
| `ServerToolCallPart` | `call_type`, `id`, `name`, `input` (dict), `extra`, `cache_control` | A server-side tool invocation (e.g. Anthropic's built-in web_search, or an MCP connector call). `call_type`/`extra` keep the wire block type and any extra fields so it round-trips. |
| `ServerToolResultPart` | `result_type`, `tool_call_id`, `content` (opaque), `extra`, `cache_control` | `result_type` is the wire block type (`web_search_tool_result`, `web_fetch_tool_result`, `code_execution_tool_result`, `mcp_tool_result`, ...). |

`Citation` (not a `Part`; hangs off `TextPart.citations` and the `CITATION_DELTA` chunk):
`kind` (`char` \| `page` \| `chunk` \| `search_result` \| `web`), `cited_text`,
`document_index`, `document_title`, `search_result_index`, `start`, `end` (exclusive),
`source`, `title`, `url`.

`cache_control` is a plain bool on every part. The Anthropic input adapter sets it from the
wire `cache_control` field on text, image, document, tool_use and tool_result blocks (the
Anthropic API accepts the marker on all of these, plus system and tools; agentic clients such
as Claude Code place it on the last tool_result of each turn), and only the Bedrock provider
currently acts on it (emitting a sibling `cachePoint`). A provider that doesn't support caching simply ignores the flag
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
  its input adapter), `system: list[TextPart] | None` (each part keeps its own
  `cache_control`, so a cache breakpoint on one block doesn't silently absorb whatever comes
  after it), generation params,
  `tools: list[ToolSpec]`, `server_tools: list[ServerToolSpec]` (passthrough `type`/`name`/
  `config`, since each wire format's server tools are their own arbitrary contract), `tool_choice`,
  `thinking`, `safeguards: list[dict]` (opaque passthrough — see "Safeguards" below).
- `InternalResult`: `content: list[Part]`, `stop_reason`, `usage`, `safeguard_results:
  list[dict] | None` (`None` = not run/not reported). No `id`/`model` fields —
  those are wire-envelope concerns, synthesized only by the input adapter that builds the
  final response (e.g. `core/adapters/anthropic.py::result_to_response`).
- `InternalChunk`: one item of a provider's streamed output — `BLOCK_START` (carries an
  empty-shell `Part` so a streaming encoder knows the block's type), `TEXT_DELTA`,
  `TOOL_ARGS_DELTA`, `THINKING_DELTA`, `SIGNATURE_DELTA`, `CITATION_DELTA` (carries one
  `Citation` for the current text block), `BLOCK_STOP`, `MESSAGE_STOP`
  (carries final `stop_reason` + `usage` + `safeguard_results`), `ERROR`.

## Anthropic input adapter (`core/adapters/anthropic.py`, `core/adapters/anthropic_stream.py`)

| Anthropic concept | Internal representation |
|---|---|
| `TextBlock` (incl. `citations`) | `TextPart` (citation locations ⇄ `Citation`: `char_location`→`char`, `page_location`→`page`, `content_block_location`→`chunk`, `search_result_location`→`search_result`, `web_search_result_location`→`web`) |
| `ImageBlock` with `base64` source | `ImagePart` |
| `DocumentBlock` with `base64` / `text` / `content` source (+ `title`, `context`, `citations.enabled`) | `DocumentPart` with `source_kind` `base64` / `text` / `content` |
| Image/document `url` or `file` (Files API) source | **Error** (`UnsupportedContentError` → 400). The gateway doesn't fetch URLs (SSRF surface) and has no file store; the message tells the client to send base64. |
| `SearchResultBlock` (top level or inside `tool_result`) | `SearchResultPart` |
| `ToolUseBlock` | `ToolCallPart` |
| `ToolResultBlock` (`content: str \| list[TextBlock \| ImageBlock \| DocumentBlock \| SearchResultBlock]`) | `ToolResultPart` (`content` always normalized to a list of the matching parts) |
| `ThinkingBlock` | `ThinkingPart` |
| `RedactedThinkingBlock` | `RedactedThinkingPart` |
| Top-level `system` + mid-conversation `role="system"` messages | Flattened into `InternalRequest.system: list[TextPart]` — one part per top-level system block (preserving its own `cache_control`), then one part per inline system message, in conversation order. Unlike a bare string, this keeps each block's cache boundary intact instead of merging everything (cached prefix and any uncached trailing content) into one span. |
| `cache_control: dict \| None` | `cache_control: bool` |
| `ToolChoiceAuto/Any/Tool/None` | `ToolChoice(mode=..., tool_name=...)` |
| `ServerToolUseBlock` / `mcp_tool_use` | `ServerToolCallPart` (`call_type` = wire type) |
| `WebSearchToolResultBlock`, and generic `ServerToolResultBlock` for `web_fetch_tool_result`, `code_execution_tool_result`, `bash_code_execution_tool_result`, `text_editor_code_execution_tool_result`, `tool_search_tool_result`, `mcp_tool_result` | `ServerToolResultPart` (`result_type` = wire type) |
| Any other content block type (`UnknownBlock`) | **Error** (`UnsupportedContentError` → 400 naming the type). Parsed rather than failing schema validation, so a new/beta block type yields a readable 400 instead of a 422. Applies inside `tool_result` too. |
| Tool `input_schema` | `ToolSpec.input_schema`, the whole JSON Schema (`$defs`, `anyOf`, `additionalProperties`, ...) — not just `properties`/`required`. |
| Unknown top-level request params (`container`, `mcp_servers`, `context_management`, ...) | Not forwarded; logged as a warning (`Ignoring unsupported top-level request params: ...`) instead of disappearing silently. Never rejected. |
| `ServerToolDefinition` (in `tools`) | `ServerToolSpec` (`InternalRequest.server_tools`, kept separate from `InternalRequest.tools`) |
| Request `safeguards` | `InternalRequest.safeguards` (verbatim) |
| Response `safeguard_results` (non-streaming: top level, omitted when `None`; streaming: on `message_delta.delta`) | `InternalResult.safeguard_results` / `MESSAGE_STOP` chunk's `safeguard_results` |

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

### Safeguards

`safeguards` (beta `dangerous-tool-use-2026-09-03`) asks the model backend to run
server-side safety checks alongside the request — Claude Code sends it in auto mode so its
`dangerous_tool_use` classifier runs on the server, where it isn't billed, instead of as
separate client-side classifier requests, which are. The backend answers with
`safeguard_results`, which holds a verdict per `tool_use` id from the same response. If a
gateway drops either field, Claude Code falls back to its own (billed) classifier requests
and shows the "this session isn't eligible" notice naming the gateway. Both fields are
opaque in the internal model: their shape belongs to the backend, not to this gateway.

| Provider | Handling |
|---|---|
| Bedrock (Converse) | **Supported, with an ID-rekeying workaround.** `safeguards` plus `anthropic_beta: ["dangerous-tool-use-2026-09-03"]` go in `additionalModelRequestFields`. The gateway adds the beta itself because it doesn't forward the caller's `anthropic-beta` header, and Bedrock rejects the field without it. Results come back through `additionalModelResponseFieldPaths`: `/safeguard_results` on `converse()`, and `/delta/safeguard_results` on `converse_stream()`, where they arrive on `messageStop`. **Caveat:** Converse rewrites tool-use ids (`tooluse_…`), but the results are keyed by the model's native ids (`toolu_bdrk_…`), which Converse never exposes. `_remap_safeguard_results` rekeys them by position: native entries follow content order, which was verified against InvokeModel. If the counts disagree, `tool_uses` is emptied rather than guessed, so the client checks those calls itself. |
| OpenAI-compatible (LangChain) | **Silent drop.** OpenAI-style backends have no equivalent. No `safeguard_results` are returned, so Claude Code falls back to client-side classifier requests (billed, as before). This is a billing difference, not lost content. |

**Revisit when:** (a) TTLLM gains an Anthropic-native outbound path (Bedrock InvokeModel or
the Anthropic API directly). That path should forward `safeguards` / `safeguard_results`
verbatim, together with the caller's `anthropic-beta` header, and skip the positional
rekeying entirely. (b) Converse starts preserving native tool-use ids or exposing them.
(c) The beta name changes: `_SAFEGUARDS_BETA` in `core/providers/bedrock/converse.py`
is hard-coded.

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
| `TextPart`, `ImagePart`, `DocumentPart`, `SearchResultPart`, `ToolCallPart`, `ToolResultPart`, `ThinkingPart` | Full native support (Bedrock Converse has direct equivalents for all of these). Images, documents and search results nested inside a `ToolResultPart` map to `image`/`document`/`searchResult` entries in `toolResult.content` (e.g. Claude Code's Read tool returning a PDF). `TextPart.citations` on a replayed assistant turn is dropped (output metadata; Converse takes the text). |
| `DocumentPart` details | `media_type` → Converse `format` via a fixed table (`pdf`, `txt`, `md`, `csv`, `html`, `doc`, `docx`, `xls`, `xlsx`); anything else → **Error** (400). `text` source → `source.text`, `content` source → `source.content` (text only — an image inside it → 400), both with format `txt` — but only when citations are enabled: Converse rejects those source keys otherwise, so without citations the text is sent as UTF-8 `bytes` (content chunks joined by blank lines). `context` and `citations.enabled` pass through natively. Names are sanitized to Converse's allowed characters (alphanumerics, single spaces, `-()[]`) and de-duplicated per request (`document`, `document (2)`, ...). |
| `ImagePart` details | `image/jpg` is normalized to `jpeg`; formats other than png/jpeg/gif/webp → **Error** (400). |
| `ToolSpec` | `input_schema` passed through verbatim (an empty `required` array is dropped; `type: object` defaulted if missing). An empty `description` is omitted (Converse requires min length 1). |
| Response citations | Non-streaming `citationsContent` blocks → `TextPart(citations=...)`; streaming `contentBlockDelta.delta.citation` → `CITATION_DELTA` (accumulated onto the assembled `TextPart`). Converse `documentChar`/`documentPage`/`documentChunk`/`searchResultLocation`/`web` → `Citation.kind` `char`/`page`/`chunk`/`search_result`/`web`. |
| `RedactedThinkingPart` | **Silent drop.** Bedrock has no equivalent, and there's nothing to emulate — the payload is opaque even to Anthropic. This is the one deliberate exception to "always emulate or error." |
| `ServerToolCallPart`, `ServerToolResultPart`, non-empty `server_tools` | **Error** (`ServerToolError`, mapped to HTTP 501 `not_implemented_error`). Bedrock Converse has no mechanism to proxy server-side tools. |

Bedrock's `stopReason` values `guardrail_intervened` and `content_filtered` (a target-model
guardrail blocked or altered the response) both map to `InternalResult.stop_reason =
"refusal"`, not `"end_turn"` — folding them into `"end_turn"` would tell the caller the model
finished normally when a guardrail actually cut it short. The OpenAI-compatible LangChain
provider applies the same mapping for `finish_reason == "content_filter"`.

### OpenAI-compatible via LangChain (`core/providers/langchain/translation.py`)

| Part | Handling |
|---|---|
| `TextPart`, `ToolCallPart` (assistant turn), `ToolResultPart` (text-only) | Full native support. |
| `ImagePart` (top-level user-turn content) | Full native support (multimodal `image_url` content part). |
| `ImagePart` inside a `ToolResultPart` | **Emulate.** Passed through as a multimodal content part in the `ToolMessage`, the same mechanism used for a normal image. Previously this crashed (`AttributeError` — the old code assumed every tool-result item had a `.text` attribute); now it just works. |
| `ThinkingPart` | **Emulate.** Folded into a normal text part — no native "thinking" concept on this provider, but the reasoning content itself is preserved rather than silently dropped. |
| `RedactedThinkingPart` | **Silent drop.** Same reasoning as Bedrock — the payload is opaque, there's no content actually being lost. |
| `DocumentPart` with a `text`/`content` source, or a base64 `text/*` media type | **Emulate.** Rendered as a text part: `[Document: {title}]`, then `context`, then the text. Applies top-level, in a `ToolResultPart` and on assistant turns. Citations are not available on this provider. |
| `DocumentPart` with a binary base64 source (PDF, DOCX, ...) | **Error** (`UnsupportedContentError`, mapped to HTTP 400 `invalid_request_error`), top-level or inside a `ToolResultPart`. No native document/PDF input on OpenAI-style chat APIs, and dropping it silently would mean the model answers as if it had read an attachment it never saw — the "impacts inference quality" case that error, not emulation, exists for. |
| `SearchResultPart` | **Emulate.** Rendered as text `[Search result: {title} ({source})]` followed by its content, top-level or inside a `ToolResultPart`. |
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
