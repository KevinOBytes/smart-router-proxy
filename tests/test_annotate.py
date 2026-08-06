"""End-to-end test: annotate_response prepends [category :: model (alias)].

Uses a fake upstream via httpx MockTransport so no network or real
classifier is required. The BERT classifier is monkeypatched to return a
deterministic result.
"""

from __future__ import annotations

import json
from collections.abc import Generator

import httpx
import pytest
from fastapi.testclient import TestClient

from smart_router_proxy.config import ProxyConfig
from smart_router_proxy.models import (
    ClassifierResult,
    RiskLevel,
    Sensitivity,
    TaskClass,
)
from smart_router_proxy.server import create_app


def _fake_upstream_handler(request: httpx.Request) -> httpx.Response:
    """Return a canned non-streaming chat completion."""
    body = json.loads(request.content or b"{}")
    model = body.get("model", "?")
    return httpx.Response(
        200,
        request=request,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    )


def _sse_payload(model: str) -> str:
    done = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    chunk1 = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}],
    }
    chunk2 = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": "world"}, "finish_reason": "stop"}],
    }
    lines = [
        f"data: {json.dumps(chunk1)}",
        f"data: {json.dumps(chunk2)}",
        f"data: {json.dumps(done)}",
        "data: [DONE]",
    ]
    return "\n\n".join(lines) + "\n\n"


def _install_mock(client: TestClient, stream: bool = False) -> None:
    """Point the app's http client at the mock transport.

    Must run inside the ``with TestClient(...)`` block: the lifespan
    re-creates app.state.http on context entry and would clobber any
    assignment made before entering.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if stream and json.loads(request.content or b"{}").get("stream"):
            return httpx.Response(
                200, request=request, content=_sse_payload("mock/model").encode()
            )
        return _fake_upstream_handler(request)

    client.app.state.http = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = ProxyConfig(annotate_response=True)
    app = create_app(cfg)

    # Deterministic classification: software_engineering -> glm.
    fake_result = ClassifierResult(
        task_class=TaskClass.SOFTWARE_ENGINEERING,
        risk=RiskLevel.MODERATE,
        sensitivity=Sensitivity.INTERNAL,
        confidence=0.91,
    )
    monkeypatch.setattr(
        "smart_router_proxy.router.get_classifier",
        lambda: type("C", (), {"classify_to_result": lambda self, t: fake_result})(),
    )
    with TestClient(app) as client:
        client.app = app
        yield client


def test_nostream_annotation(app: TestClient) -> None:
    _install_mock(app)
    r = app.post(
        "/v1/chat/completions",
        json={
            "model": "smart-router",
            "messages": [{"role": "user", "content": "write a python function"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    assert content.startswith("[software_engineering :: z-ai/glm-5.2 (glm)]\n")
    assert content.endswith("Hello world")
    assert data["model"] == "z-ai/glm-5.2"


def test_session_id_body_activates_pinning(app: TestClient) -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        calls.append(body)
        return _fake_upstream_handler(request)

    app.app.state.http = httpx.AsyncClient(  # type: ignore[union-attr]
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    first = app.post(
        "/v1/chat/completions",
        json={
            "model": "smart-router",
            "session_id": "hermes-session-1",
            "messages": [{"role": "user", "content": "write a python function"}],
        },
    )
    second = app.post(
        "/v1/chat/completions",
        json={
            "model": "smart-router",
            "session_id": "hermes-session-1",
            "messages": [{"role": "user", "content": "now explain a recipe"}],
        },
    )
    assert first.status_code == second.status_code == 200
    assert len(calls) == 2
    assert calls[0]["model"] == calls[1]["model"] == "z-ai/glm-5.2"
    assert calls[0]["session_id"] == calls[1]["session_id"] == "hermes-session-1"


def test_routing_metadata_forwarded(app: TestClient) -> None:
    captured: dict[str, object] = {}
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content or b"{}"))
        captured_headers.update(dict(request.headers))
        return _fake_upstream_handler(request)

    app.app.state.http = httpx.AsyncClient(  # type: ignore[union-attr]
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    r = app.post(
        "/v1/chat/completions",
        json={
            "model": "smart-router",
            "messages": [{"role": "user", "content": "write a python function"}],
        },
    )
    assert r.status_code == 200
    assert captured["metadata"] == {
        "task_class": "software_engineering",
        "router_alias": "glm",
        "router_slug": "z-ai/glm-5.2",
    }
    assert json.loads(captured_headers["langfuse_trace_metadata"]) == captured["metadata"]
    assert captured_headers["langfuse_generation_name"] == "smart-router-proxy"


def test_stream_annotation(app: TestClient) -> None:
    _install_mock(app, stream=True)
    r = app.post(
        "/v1/chat/completions",
        json={
            "model": "smart-router",
            "stream": True,
            "messages": [{"role": "user", "content": "write a python function"}],
        },
    )
    assert r.status_code == 200
    assert "[software_engineering :: z-ai/glm-5.2 (glm)]" in r.text


def test_disabled_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default config (annotate_response=False) must pass through verbatim."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = ProxyConfig()  # annotate_response defaults False
    app = create_app(cfg)
    with TestClient(app) as client:
        client.app = app
        _install_mock(client)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "real/model-passthrough",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Hello world"
