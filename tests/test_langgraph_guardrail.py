from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from aryaxai.core.guardrails import guardrails_langgraph
from aryaxai.core.guardrails.guardrails_langgraph import LangGraphGuardrail


class FakeClient:
    """One guard that fails on any text containing 4111."""

    def __init__(self):
        self.posts = 0

    def post(self, url, payload):
        self.posts += 1
        passed = "4111" not in payload["input_data"]
        return {
            "success": True,
            "details": [{"name": "Detect PII", "validation_passed": passed, "sanitized_output": "card ****"}],
        }


def make_guard():
    client = FakeClient()
    project = SimpleNamespace(api_client=client, project_name="demo")
    return LangGraphGuardrail(project), client


def make_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def use_langchain_span(monkeypatch, guard):
    guard.tracer, exporter = make_tracer()
    run_span = guard.tracer.start_span("run")
    monkeypatch.setattr(guardrails_langgraph, "get_current_span", lambda: run_span)
    return run_span, exporter


def state(text):
    return {"messages": [HumanMessage(content=text)]}


def test_block_stops_input_without_tracing():
    guard, client = make_guard()
    ran = []

    @guard.guardrail(["Detect PII"], action="block", apply_to="input")
    def node(state):
        ran.append(True)
        return state

    with pytest.raises(ValueError, match="Detect PII"):
        node(state("card 4111"))
    assert ran == []
    assert client.posts == 1


def test_block_stops_output_without_tracing():
    guard, client = make_guard()

    @guard.guardrail(["Detect PII"], action="block", apply_to="output")
    def node(state):
        return {"messages": [AIMessage(content="card 4111")]}

    with pytest.raises(ValueError, match="Detect PII"):
        node(state("hello"))
    assert client.posts == 1


def test_block_stops_input_with_tracing(monkeypatch):
    guard, client = make_guard()
    use_langchain_span(monkeypatch, guard)
    ran = []

    @guard.guardrail(["Detect PII"], action="block", apply_to="input")
    def node(state):
        ran.append(True)
        return state

    with pytest.raises(ValueError, match="Detect PII"):
        node(state("card 4111"))
    assert ran == []


def test_block_stops_output_with_tracing(monkeypatch):
    guard, _ = make_guard()
    use_langchain_span(monkeypatch, guard)

    @guard.guardrail(["Detect PII"], action="block", apply_to="output")
    def node(state):
        return {"messages": [AIMessage(content="card 4111")]}

    with pytest.raises(ValueError, match="Detect PII"):
        node(state("hello"))


def test_clean_content_passes_without_tracing():
    guard, client = make_guard()

    @guard.guardrail(["Detect PII"], action="block", apply_to="both")
    def node(state):
        return {"messages": [AIMessage(content="hi")]}

    result = node(state("hello"))
    assert result["messages"][-1].content == "hi"
    assert client.posts == 2


def test_warn_keeps_content_without_tracing():
    guard, client = make_guard()

    @guard.guardrail(["Detect PII"], action="warn", apply_to="output")
    def node(state):
        return {"messages": [AIMessage(content="card 4111")]}

    result = node(state("hello"))
    assert result["messages"][-1].content == "card 4111"
    assert client.posts == 1


def test_retry_uses_the_sanitized_output_without_tracing():
    guard, client = make_guard()

    @guard.guardrail(["Detect PII"], action="retry", apply_to="output")
    def node(state):
        return {"messages": [AIMessage(content="card 4111")]}

    result = node(state("hello"))
    assert result["messages"][-1].content == "card ****"
    assert client.posts == 1


def test_spans_stay_under_the_langchain_span(monkeypatch):
    guard, _ = make_guard()
    run_span, exporter = use_langchain_span(monkeypatch, guard)

    guard._process_content("hello", "node", "input", "block", ["Detect PII"])

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["guardrails:input"].parent.span_id == run_span.get_span_context().span_id


def test_spans_stay_under_the_active_span_without_langchain():
    guard, _ = make_guard()
    guard.tracer, exporter = make_tracer()

    with guard.tracer.start_as_current_span("app") as app_span:
        guard._process_content("hello", "node", "input", "block", ["Detect PII"])

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["guardrails:input"].parent.span_id == app_span.get_span_context().span_id
