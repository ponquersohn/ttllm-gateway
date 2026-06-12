# v0.1.1 - Prompt Caching & Bedrock Timeouts

## What's Changed

### Features
* **Prompt Caching Support** — Full end-to-end support for Anthropic `cache_control` markers on system blocks, message content, and tools. Bedrock `cachePoint` markers are now correctly emitted, enabling clients to leverage prompt caching for reduced latency and token costs on stable context.
* **Configurable Bedrock Timeouts** — New per-model configuration options for `read_timeout`, `connect_timeout`, and `retry_max_attempts` to handle large prompts or slow networks. Defaults are tuned for streaming (300s read timeout) and include retry logic.

### Improvements
* Bedrock client caching now includes timeout/retry configuration in the cache key, allowing distinct clients for different timeout profiles
* Integration test suite expanded with three new tests (`test_caching.py`) validating the full cache_control → cachePoint → cache-read-tokens path
* Unit tests added for timeout configuration and client cache behavior

### Technical Details
- System block iteration fixed to properly append `cachePoint` markers
- Tool spec iteration refactored for cache control handling
- Fake Bedrock test double now detects and reports `cachePoint` markers in request bodies

## Configuration

New optional keys in model `config_json`:

```json
{
  "read_timeout": 300,           // seconds; default 300 (for large prompt prefill)
  "connect_timeout": 10,         // seconds; default 10
  "retry_max_attempts": 3        // retries; default 3
}
```

Example with CLI:
```bash
ttllm models create \
  --name claude-sonnet \
  --provider bedrock \
  --provider-model-id anthropic.claude-sonnet-4-20250514-v1:0 \
  --config '{"region":"us-east-1","read_timeout":600,"connect_timeout":5}'
```

## Compatibility

- ✅ Backward compatible — existing models without timeout config use defaults
- ✅ Cache control is opt-in via client payload; no breaking changes
- ✅ Tested against boto3 Converse API and fake Bedrock test double

## Files Changed
- `src/ttllm/core/bedrock.py` — timeout config, cache_point emission, system block iteration
- `src/ttllm/schemas/anthropic.py` — `cache_control` field on TextBlock and ToolDefinition
- `tests/integration/fake_bedrock/app.py` — cachePoint detection and cache-read reporting
- `tests/integration/test_caching.py` — new end-to-end cache control tests
- `tests/test_bedrock.py` — new unit tests for timeout behavior and client caching
