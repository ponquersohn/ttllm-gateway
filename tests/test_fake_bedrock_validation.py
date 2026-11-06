"""The fake Bedrock must reject what real Bedrock rejects -- and everything the gateway
sends must pass it. Runs offline (no docker); the fake's validator is plain Python."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from tests.integration.fake_bedrock.app import app
from tests.integration.fake_bedrock.validation import BedrockValidationError, validate_request
from tests.test_bedrock import _make_model, _make_request
from ttllm.core.model import (
    DocumentPart,
    ImagePart,
    InternalMessage,
    SearchResultPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
)
from ttllm.core.providers.bedrock.converse import build_converse_request

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
PDF = base64.b64encode(b"%PDF-1.4 fake").decode()
PNG = base64.b64encode(b"\x89PNG fake").decode()


def _check(**request_kwargs) -> None:
    for stream in (False, True):
        params = build_converse_request(_make_request(**request_kwargs), _make_model(), stream=stream)
        params.pop("modelId")
        validate_request("ConverseStream" if stream else "Converse", MODEL, params)


def _msgs(*turns):
    return [InternalMessage(role=role, content=list(parts)) for role, parts in turns]


class TestGatewayOutputAcceptedByBedrockRules:
    """Each case is a request shape the gateway now accepts from clients."""

    def test_pdf_and_image_in_tool_result_with_untitled_duplicates(self):
        _check(
            tools=[ToolSpec(name="Read", input_schema={"type": "object", "properties": {"p": {"type": "string"}}})],
            messages=_msgs(
                ("user", [TextPart(text="read them")]),
                ("assistant", [ToolCallPart(id="t1", name="Read", input={"p": "a"}),
                               ToolCallPart(id="t2", name="Read", input={"p": "b"})]),
                ("user", [
                    ToolResultPart(tool_call_id="t1", content=[TextPart(text="PDF read"),
                                                               DocumentPart(media_type="application/pdf", data=PDF)]),
                    ToolResultPart(tool_call_id="t2", content=[DocumentPart(media_type="application/pdf", data=PDF),
                                                               ImagePart(media_type="image/jpg", data=PNG)]),
                ]),
            ),
        )

    def test_text_and_content_documents_with_and_without_citations(self):
        _check(messages=_msgs(("user", [
            DocumentPart(source_kind="text", text="a", title="report.final.v2", citations=True, context="ctx"),
            DocumentPart(source_kind="text", text="b", title="report.final.v2"),
            DocumentPart(source_kind="content", content=[TextPart(text="c")], citations=True),
            DocumentPart(source_kind="content", content=[TextPart(text="d")]),
            TextPart(text="summarize"),
        ])))

    def test_search_results_and_rich_tool_schema_without_description(self):
        sr = SearchResultPart(source="https://kb/1", title="KB", content=[TextPart(text="fact")], citations=True)
        _check(
            tools=[ToolSpec(name="kb", input_schema={
                "type": "object", "$defs": {"Q": {"type": "string"}},
                "properties": {"q": {"$ref": "#/$defs/Q"}}, "required": [], "additionalProperties": False})],
            messages=_msgs(
                ("user", [sr, TextPart(text="look it up")]),
                ("assistant", [ToolCallPart(id="t1", name="kb", input={"q": "x"})]),
                ("user", [ToolResultPart(tool_call_id="t1", content=[sr])]),
            ),
        )


def _base(messages, **extra):
    return {"messages": messages, "inferenceConfig": {"maxTokens": 5}, **extra}


Q = {"text": "hi"}
TU = {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "x", "input": {}}}]}
TOOLS = {"tools": [{"toolSpec": {"name": "x", "inputSchema": {"json": {"type": "object"}}}}]}


def _doc(name, source=None, **extra):
    return {"document": {"format": "txt", "name": name, "source": source or {"bytes": "aGk="}, **extra}}


def _tr(tool_id):
    return {"role": "user", "content": [{"toolResult": {"toolUseId": tool_id, "content": [{"text": "r"}]}}]}


class TestFakeRejectsWhatBedrockRejects:
    """Error messages are the ones real Bedrock returned for the same request."""

    @pytest.mark.parametrize("body,match", [
        (_base([{"role": "user", "content": [Q]}], bogus=1), 'Unknown parameter in input: "bogus"'),
        (_base([{"role": "user", "content": [Q]}],
               toolConfig={"tools": [{"toolSpec": {"name": "x", "description": "",
                                                   "inputSchema": {"json": {}}}}]}), "valid min length: 1"),
        (_base([{"role": "user", "content": [{"document": {"format": "pdff", "name": "a", "source": {"bytes": "aGk="}}}]}]),
         "Invalid value for parameter"),
        (_base([{"role": "user", "content": [_doc("a"), _doc("a"), Q]}]), "duplicate document names"),
        (_base([{"role": "user", "content": [_doc("report.pdf"), Q]}]), "document file name"),
        (_base([{"role": "user", "content": [_doc("a  b"), Q]}]), "document file name"),
        (_base([{"role": "user", "content": [_doc("d", {"text": "x"}), Q]}]), "must set one of the following keys"),
        (_base([{"role": "user", "content": [{"text": " "}]}]), "is blank"),
        (_base([{"role": "user", "content": [Q]}, TU, _tr("t1")]), "toolConfig field must be defined"),
        (_base([{"role": "user", "content": [Q]}, TU, _tr("t2")], toolConfig=TOOLS),
         "for the following Ids: t1, but found: t2"),
        (_base([{"role": "user", "content": [Q]}, TU, {"role": "user", "content": [Q]}], toolConfig=TOOLS),
         "without `tool_result` blocks immediately after: t1"),
    ])
    def test_rejected(self, body, match):
        with pytest.raises(BedrockValidationError, match=match):
            validate_request("Converse", MODEL, body)

    @pytest.mark.parametrize("body", [
        _base([{"role": "user", "content": [_doc("d", {"text": "x"}, citations={"enabled": True}), Q]}]),
        # Real Bedrock accepts these (verified), so the fake must too.
        _base([{"role": "user", "content": [Q]}, {"role": "user", "content": [Q]}]),
        _base([{"role": "assistant", "content": [Q]}, {"role": "user", "content": [Q]}]),
    ])
    def test_accepted(self, body):
        validate_request("Converse", MODEL, body)


class TestFakeErrorWireShape:
    @pytest.mark.parametrize("route", ["converse", "converse-stream"])
    def test_validation_exception_response(self, route):
        resp = TestClient(app).post(f"/model/{MODEL}/{route}", json=_base([{"role": "user", "content": [{"text": ""}]}]))
        assert resp.status_code == 400
        assert resp.headers["x-amzn-ErrorType"] == "ValidationException"
        assert "is blank" in resp.json()["message"]
