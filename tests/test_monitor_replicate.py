import replicate

from aryaxai.core.wrapper import monitor

OUTPUT = ["hello", " world"]


class FakeApiClient:
    def __init__(self):
        self.posts = []

    def post(self, uri, payload=None):
        self.posts.append((uri, payload))
        return {"details": {"session_id": "session"}, "success": True}


class FakeProject:
    project_name = "project"

    def __init__(self):
        self.api_client = FakeApiClient()


def run_monitored():
    project = FakeProject()
    client = replicate.Client(api_token="token")
    client.run = lambda *args, **kwargs: OUTPUT
    monitor(project, client)
    return client.run("owner/model", input={"prompt": "hi"}), project.api_client.posts


def test_run_returns_model_output():
    result, _ = run_monitored()
    assert result == OUTPUT


def test_run_traces_input_and_output():
    _, posts = run_monitored()
    traces = {p["component"]: p for uri, p in posts if uri == "traces/add_trace"}
    assert traces["Input"]["input_data"] == {"prompt": "hi"}
    assert traces["Output"]["output_data"] == OUTPUT
    assert traces["Output Guardrails"]["input_data"] == OUTPUT


def test_run_adds_session_message():
    _, posts = run_monitored()
    messages = [p for uri, p in posts if uri == "sessions/add_session_message"]
    assert len(messages) == 1
    assert messages[0]["output_data"] == OUTPUT


def test_guardrails_use_provider_as_model_name():
    _, posts = run_monitored()
    guardrails = [p for uri, p in posts if uri == "v2/ai-models/run_guardrails"]
    assert [p["apply_on"] for p in guardrails] == ["input", "output"]
    assert {p["model_name"] for p in guardrails} == {"Replicate"}
