"""Request validation for the fake Bedrock, so it rejects what real Bedrock rejects.

Two layers:

1. **Shape** -- botocore's own ``bedrock-runtime`` service model (the same definition
   boto3 validates against client-side): unknown/missing fields, types, length limits,
   plus enum values (boto3's validator skips enums; the server enforces them). Exact and free; tracks AWS as botocore is upgraded.
2. **Semantic rules** the service model can't express. Each one below was observed
   against real Bedrock (Converse, Claude Haiku 4.5, 2026-11-06), and the error message
   is copied from the real response so tests can match on it.

Anything not listed here is *not* enforced by real Bedrock either (e.g. it accepts two
consecutive user messages, or a conversation starting with an assistant turn).
"""

from __future__ import annotations

import re
from typing import Any, Iterator

import botocore.session
from botocore.validate import ParamValidator

_SERVICE_MODEL = botocore.session.get_session().get_service_model("bedrock-runtime")

# Alphanumerics, whitespace, hyphens, parentheses, square brackets; no double whitespace.
_DOC_NAME_RE = re.compile(r"^[A-Za-z0-9\s\-()\[\]]+$")

_DOC_NAME_MSG = (
    "The document file name can only contain alphanumeric characters, whitespace characters, "
    "hyphens, parentheses, and square brackets. The name can't contain more than one consecutive "
    "whitespace character. Check that the file name conforms to these restrictions and retry your "
    "request."
)


class BedrockValidationError(Exception):
    """Raised for a request real Bedrock would reject with ``ValidationException``."""


def _documents(content: list[Any], path: str) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield every document block in a content list, including inside toolResult."""
    for i, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        if "document" in block:
            yield block["document"], f"{path}.{i}.document"
        if "toolResult" in block:
            inner = block["toolResult"].get("content", [])
            yield from _documents(inner, f"{path}.{i}.toolResult.content")


def _ids(content: list[Any], key: str) -> list[str]:
    return [b[key]["toolUseId"] for b in content if isinstance(b, dict) and key in b]


def _check_enums(value: Any, shape: Any, path: str) -> None:
    if shape.type_name == "string" and shape.enum and isinstance(value, str) and value not in shape.enum:
        raise BedrockValidationError(
            f"Invalid value for parameter {path}, value: {value}, valid values: {', '.join(shape.enum)}"
        )
    if shape.type_name == "structure" and isinstance(value, dict):
        for key, member in shape.members.items():
            if key in value:
                _check_enums(value[key], member, f"{path}.{key}" if path else key)
    elif shape.type_name == "list" and isinstance(value, list):
        for i, item in enumerate(value):
            _check_enums(item, shape.member, f"{path}[{i}]")
    elif shape.type_name == "map" and isinstance(value, dict):
        for key, item in value.items():
            _check_enums(item, shape.value, f"{path}.{key}")


def validate_shape(operation: str, params: dict[str, Any]) -> None:
    shape = _SERVICE_MODEL.operation_model(operation).input_shape
    report = ParamValidator().validate(params, shape)
    if report.has_errors():
        raise BedrockValidationError(report.generate_report())
    _check_enums(params, shape, "")


def validate_semantics(params: dict[str, Any]) -> None:
    messages = params.get("messages", [])

    has_tool_blocks = False
    doc_names: set[str] = set()
    for m_idx, msg in enumerate(messages):
        content = msg.get("content", [])
        for c_idx, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if "text" in block and not block["text"].strip():
                raise BedrockValidationError(
                    f"The text field in the ContentBlock object at messages.{m_idx}.content.{c_idx} "
                    "is blank. Add text to the text field, and try again."
                )
            if "toolUse" in block or "toolResult" in block:
                has_tool_blocks = True

        for doc, doc_path in _documents(content, f"messages.{m_idx}.content"):
            name = doc.get("name", "")
            if not _DOC_NAME_RE.match(name) or re.search(r"\s\s", name):
                raise BedrockValidationError(_DOC_NAME_MSG)
            if name in doc_names:
                raise BedrockValidationError(
                    "Messages can’t contain duplicate document names. "
                    "Rename the document and retry your request."
                )
            doc_names.add(name)
            source = doc.get("source", {})
            if ("text" in source or "content" in source) and not (doc.get("citations") or {}).get("enabled"):
                raise BedrockValidationError(
                    f"DocumentSource object at {doc_path}.source must set one of the following "
                    "keys: bytes, s3Location."
                )

    if has_tool_blocks and not params.get("toolConfig"):
        raise BedrockValidationError(
            "The toolConfig field must be defined when using toolUse and toolResult content blocks."
        )

    for m_idx, msg in enumerate(messages):
        content = msg.get("content", [])
        results = _ids(content, "toolResult")
        prev_uses = _ids(messages[m_idx - 1].get("content", []), "toolUse") if m_idx > 0 else []
        if results and sorted(results) != sorted(prev_uses):
            raise BedrockValidationError(
                f"Expected toolResult blocks at messages.{m_idx}.content for the following Ids: "
                f"{', '.join(prev_uses)}, but found: {', '.join(results)}"
            )
        uses = _ids(content, "toolUse")
        if uses and m_idx + 1 < len(messages) and not _ids(messages[m_idx + 1].get("content", []), "toolResult"):
            raise BedrockValidationError(
                f"The model returned the following errors: messages.{m_idx + 1}: `tool_use` ids were "
                f"found without `tool_result` blocks immediately after: {', '.join(uses)}. Each "
                "`tool_use` block must have a corresponding `tool_result` block in the next message."
            )


def validate_request(operation: str, model_id: str, body: dict[str, Any]) -> None:
    """Validate a Converse/ConverseStream request as real Bedrock would.

    ``body`` is the JSON wire body (blobs arrive base64-encoded as strings, which the
    botocore validator accepts for blob shapes); ``modelId`` travels in the URL path.
    """
    params = {"modelId": model_id, **body}
    validate_shape(operation, params)
    validate_semantics(params)
