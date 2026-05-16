"""TTLLM Gateway - LLM Gateway with Anthropic-compatible API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ttllm-gateway")
except PackageNotFoundError:
    __version__ = "0.0.0-unknown"
