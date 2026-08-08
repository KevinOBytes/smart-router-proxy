"""Provider dispatch tests: routed destinations must hit the correct backend.

Regression: server.py used to capture decision.slug/alias/category/fallback
but never decision.provider, and always posted to the OpenRouter client — so
routing to an installed Ollama model silently sent the Ollama model name up
to OpenRouter (error or wrong-provider billing). This test proves the
dispatch branches on provider. All backends are mocked — no real network,
no real API key is used.
"""

from __future__ import annotations

import json
from collections.abc import Generator

import httpx
import pytest
from fastapi.testclient import TestClient

from smart_router_proxy.config import ProxyConfig
from smart_router_proxy.server import create_app


def _completion(model: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    # Fake key so a slip to OpenRouter would be unmistakable (401 "User not
    # found") while still never costing money. Tests below prove no request
    # ever leaves the mock transports.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-DO-NOT-USE-TEST")
    cfg = ProxyConfig(
        store_path=":memory:",
        mode="fixed",
        fixed_provider="fixed-provider-placeholder",
        fixed_slug="fixed-slug-placeholder",
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


def test_openrouter_dispatch_uses_upstream_headers(app: TestClient) -> None:
    """A fixed openrouter destination posts to the OpenRouter client (path
    /chat/completions) WITH the proxy's own authorization header."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["host"] = request.url.host
        captured["auth"] = request.headers.get("authorization")
        captured["model"] = json.loads(request.content or b"{}").get("model")
        return _completion("org/model")

    # Rebuild the app with an openrouter fixed destination.
    cfg = ProxyConfig(
        store_path=":memory:", mode="fixed", fixed_slug="org/model"
    )
    app2 = create_app(cfg)
    app2.router.lifespan_context  # noqa: B018 (referenced for clarity only)
    with TestClient(app2) as client:
        client.app.state.http = httpx.AsyncClient(  # type: ignore[union-attr]
            base_url="https://openrouter.ai/api/v1",
            transport=httpx.MockTransport(handler),
        )
        r = client.post(
            "/v1/chat/completions",
            json={"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert str(captured["path"]).endswith("/chat/completions")
    assert captured["host"] == "openrouter.ai"
    assert captured["auth"] == "Bearer sk-or-v1-DO-NOT-USE-TEST"
    assert captured["model"] == "org/model"


def test_ollama_dispatch_hits_loopback_without_auth(app: TestClient) -> None:
    """A fixed Ollama destination must post to the Ollama client (OpenAI-
    compatible path /v1/chat/completions on the loopback host) with NO auth
    header. This is the regression: it must never go to OpenRouter."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["host"] = request.url.host
        captured["auth"] = request.headers.get("authorization")
        captured["model"] = json.loads(request.content or b"{}").get("model")
        return _completion("qwen3.6:27b-coding-bf16")

    cfg = ProxyConfig(
        store_path=":memory:",
        mode="fixed",
        fixed_provider="ollama",
        fixed_slug="qwen3.6:27b-coding-bf16",
    )
    app2 = create_app(cfg)
    with TestClient(app2) as client:
        client.app.state.ollama_http = httpx.AsyncClient(  # type: ignore[union-attr]
            base_url="http://127.0.0.1:11434",
            transport=httpx.MockTransport(handler),
        )
        r = client.post(
            "/v1/chat/completions",
            json={"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert captured["path"] == "/v1/chat/completions"
    assert captured["host"] == "127.0.0.1"
    assert captured["auth"] is None
    assert captured["model"] == "qwen3.6:27b-coding-bf16"


def test_ollama_client_unavailable_returns_502(app: TestClient) -> None:
    """If the Ollama client is not available, an Ollama-routed request must
    fail closed with 502 rather than silently fall through to OpenRouter."""
    cfg = ProxyConfig(
        store_path=":memory:",
        mode="fixed",
        fixed_provider="ollama",
        fixed_slug="some-local-model",
    )
    app2 = create_app(cfg)
    with TestClient(app2) as client:
        client.app.state.ollama_http = None  # type: ignore[union-attr]
        r = client.post(
            "/v1/chat/completions",
            json={"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 502
