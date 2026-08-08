"""Tests for the control-panel admin API and UI serving.

Uses an in-memory store and a temp config file so mutations exercise the
atomic persistence path without touching the real ~/.smart-router-proxy
state or the live config.yaml.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from smart_router_proxy.config import ProxyConfig
from smart_router_proxy.server import create_app


@pytest.fixture()
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = ProxyConfig(
        store_path=":memory:",
        upstream=ProxyConfig.model_fields["upstream"].default,
    )
    app = create_app(cfg, config_path=tmp_path / "config.yaml")
    with TestClient(app) as client:
        yield client


def _mock_upstream(client: TestClient, *, models: bool = True) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(
                200,
                request=request,
                json={
                    "data": [
                        {
                            "id": "org/mock-model",
                            "name": "Mock Model",
                            "context_length": 128000,
                            "pricing": {"prompt": "0.5", "completion": "1.5"},
                            "modality": ["text", "image"],
                        },
                        {
                            "id": "z-ai/glm-5.2",
                            "name": "GLM 5.2",
                            "context_length": 128000,
                            "pricing": {"prompt": "0.5", "completion": "1.5"},
                            "modality": ["text"],
                        },
                        {
                            "id": "anthropic/claude-opus-5",
                            "name": "Claude Opus 5",
                            "context_length": 200000,
                            "pricing": {"prompt": "3", "completion": "15"},
                            "modality": ["text", "image"],
                        },
                    ]
                },
            )
        return httpx.Response(404, request=request, json={"detail": "not found"})

    client.app.state.http = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )


def _mock_ollama(client: TestClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                request=request,
                json={
                    "models": [
                        {
                            "name": "qwen3.6:35b-a3b-q4_K_M",
                            "size": 1234567890,
                            "modified_at": "2026-08-01T00:00:00Z",
                        }
                    ]
                },
            )
        return httpx.Response(404, request=request, json={"detail": "not found"})

    client.app.state.ollama_http = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )


# ── UI serving ────────────────────────────────────────────────────────


def test_ui_served(app: TestClient) -> None:
    res = app.get("/ui")
    assert res.status_code == 200
    assert "smart-router-proxy" in res.text


def test_static_assets_served(app: TestClient) -> None:
    for path in ("/static/app.js", "/static/styles.css"):
        res = app.get(path)
        assert res.status_code == 200
        assert res.headers["content-type"].startswith(("text/javascript", "text/css"))


# ── State & catalog ───────────────────────────────────────────────────


def test_admin_state_shape(app: TestClient) -> None:
    res = app.get("/api/admin/state")
    assert res.status_code == 200
    body = res.json()
    assert body["server"]["loopback"] is True
    assert len(body["routing"]) == 8
    classes = {r["task_class"] for r in body["routing"]}
    assert "software_engineering" in classes
    assert body["upstream"]["api_key_set"] is True
    # The upstream block carries only the env-var name and a set-flag; the
    # key value itself must never appear anywhere in the response.
    assert set(body["upstream"]) == {"base_url", "api_key_env", "api_key_set", "timeout_seconds"}


def test_admin_state_never_exposes_secret(app: TestClient) -> None:
    # The key env var name is shown, the value is not and never was set.
    res = app.get("/api/admin/state")
    assert "test-key" not in res.text


def test_catalog_refresh_merges_openrouter_and_ollama(app: TestClient) -> None:
    _mock_upstream(app)
    _mock_ollama(app)
    res = app.post("/api/admin/catalog/refresh")
    assert res.status_code == 200
    body = res.json()
    or_ids = [m["id"] for m in body["openrouter"]["models"]]
    ol_ids = [m["id"] for m in body["ollama"]["models"]]
    assert "org/mock-model" in or_ids
    assert "qwen3.6:35b-a3b-q4_K_M" in ol_ids
    assert body["openrouter"]["stale"] is False
    assert body["ollama"]["stale"] is False


def test_catalog_refresh_degrades_independently(app: TestClient) -> None:
    # Ollama unreachable must not break the OpenRouter half. Force the
    # failure explicitly — the host may have a live Ollama running.
    _mock_upstream(app)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused (mocked)")

    app.app.state.ollama_http = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(unreachable)
    )
    res = app.post("/api/admin/catalog/refresh")
    assert res.status_code == 200
    body = res.json()
    assert len(body["openrouter"]["models"]) == 3
    assert body["ollama"]["models"] == []
    assert body["ollama"]["error"] is not None


# ── Classify-only ─────────────────────────────────────────────────────


def test_classify_only_returns_route_without_calling_provider(app: TestClient) -> None:
    res = app.post(
        "/api/admin/classify",
        json={"text": "write a fastapi endpoint that validates JWTs"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["task_class"] in {"software_engineering", "-"}
    assert body["provider"] in {"openrouter", "ollama"}
    assert "no provider was called" in body["note"]


def test_classify_only_rejects_empty_text(app: TestClient) -> None:
    res = app.post("/api/admin/classify", json={"text": ""})
    assert res.status_code == 422


# ── Mutations ─────────────────────────────────────────────────────────


def _seed_catalog(app: TestClient) -> None:
    _mock_upstream(app)
    _mock_ollama(app)
    res = app.post("/api/admin/catalog/refresh")
    assert res.status_code == 200


def test_routing_update_applies_and_persists(app: TestClient, tmp_path: Path) -> None:
    _seed_catalog(app)
    payload = {
        "task_class": "software_engineering",
        "primary": {"provider": "openrouter", "model": "org/mock-model"},
        "fallback": {"provider": "ollama", "model": "qwen3.6:35b-a3b-q4_K_M"},
    }
    res = app.patch("/api/admin/config/routing", json=payload)
    assert res.status_code == 200
    row = next(r for r in res.json()["routing"] if r["task_class"] == "software_engineering")
    assert row["primary"]["provider"] == "openrouter"
    assert row["primary"]["model_slug"] == "org/mock-model"
    assert row["fallback"]["provider"] == "ollama"
    assert row["fallback"]["model_slug"] == "qwen3.6:35b-a3b-q4_K_M"
    assert row["overridden"] is True

    # Persisted atomically: a fresh app on the same config file sees it.
    fresh = create_app(config_path=tmp_path / "config.yaml")
    with TestClient(fresh) as client:
        state = client.get("/api/admin/state").json()
        row = next(r for r in state["routing"] if r["task_class"] == "software_engineering")
        assert row["primary"]["model_slug"] == "org/mock-model"
        assert row["overridden"] is True


def test_routing_revert_to_default_drops_override(
    app: TestClient, tmp_path: Path
) -> None:
    _seed_catalog(app)
    app.patch(
        "/api/admin/config/routing",
        json={
            "task_class": "software_engineering",
            "primary": {"provider": "openrouter", "model": "org/mock-model"},
            "fallback": None,
        },
    )

    # Revert to the built-in defaults: the override must be dropped, so the
    # matrix reports "default" and nothing persists to disk.
    defaults = {
        "task_class": "software_engineering",
        "primary": {"provider": "openrouter", "model": "z-ai/glm-5.2"},
        "fallback": {"provider": "openrouter", "model": "anthropic/claude-opus-5"},
    }
    res = app.patch("/api/admin/config/routing", json=defaults)
    assert res.status_code == 200
    row = next(
        r for r in res.json()["routing"] if r["task_class"] == "software_engineering"
    )
    assert row["overridden"] is False

    fresh = create_app(config_path=tmp_path / "config.yaml")
    with TestClient(fresh) as client:
        row = next(
            r
            for r in client.get("/api/admin/state").json()["routing"]
            if r["task_class"] == "software_engineering"
        )
        assert row["overridden"] is False


def test_routing_update_rejects_unknown_class(app: TestClient) -> None:
    res = app.patch(
        "/api/admin/config/routing",
        json={
            "task_class": "not_a_class",
            "primary": {"provider": "openrouter", "model": "x/y"},
            "fallback": None,
        },
    )
    assert res.status_code == 422


def test_routing_update_rejects_extra_fields(app: TestClient) -> None:
    res = app.patch(
        "/api/admin/config/routing",
        json={
            "task_class": "software_engineering",
            "primary": {"provider": "openrouter", "model": "x/y"},
            "fallback": None,
            "sneaky": "x",
        },
    )
    assert res.status_code == 422


def test_routing_update_rejects_bad_provider(app: TestClient) -> None:
    _seed_catalog(app)
    res = app.patch(
        "/api/admin/config/routing",
        json={
            "task_class": "software_engineering",
            "primary": {"provider": "azure", "model": "x/y"},
            "fallback": None,
        },
    )
    assert res.status_code == 422


def test_routing_update_rejects_unknown_model(app: TestClient) -> None:
    _seed_catalog(app)
    res = app.patch(
        "/api/admin/config/routing",
        json={
            "task_class": "software_engineering",
            "primary": {"provider": "openrouter", "model": "org/not-in-catalog"},
            "fallback": None,
        },
    )
    assert res.status_code == 400
    assert "not in the catalog" in res.json()["detail"]


def test_routing_update_fails_closed_without_catalog(app: TestClient) -> None:
    # No catalog refresh happened — even a valid-looking model must be
    # rejected because it cannot be proven selectable.
    res = app.patch(
        "/api/admin/config/routing",
        json={
            "task_class": "software_engineering",
            "primary": {"provider": "openrouter", "model": "org/anything"},
            "fallback": None,
        },
    )
    assert res.status_code == 400


def test_behavior_update_roundtrip(app: TestClient, tmp_path: Path) -> None:
    res = app.patch(
        "/api/admin/config/behavior",
        json={
            "mode": "fixed",
            "fixed_slug": "anthropic/claude-sonnet-5",
            "fixed_provider": "openrouter",
            "annotate_response": True,
            "session_ttl_seconds": 7200,
        },
    )
    assert res.status_code == 200
    b = res.json()["behavior"]
    assert b["mode"] == "fixed"
    assert b["fixed_slug"] == "anthropic/claude-sonnet-5"
    assert b["fixed_provider"] == "openrouter"
    assert b["annotate_response"] is True
    assert b["session_ttl_seconds"] == 7200

    fresh = create_app(config_path=tmp_path / "config.yaml")
    with TestClient(fresh) as client:
        assert client.get("/api/admin/state").json()["behavior"]["mode"] == "fixed"


def test_behavior_update_rejects_invalid_mode(app: TestClient) -> None:
    res = app.patch(
        "/api/admin/config/behavior",
        json={
            "mode": "shadow",
            "fixed_slug": "x/y",
            "fixed_provider": "openrouter",
            "annotate_response": False,
            "session_ttl_seconds": 3600,
        },
    )
    assert res.status_code == 422


def test_upstream_update_rejects_credentials_in_url(app: TestClient) -> None:
    res = app.patch(
        "/api/admin/config/upstream",
        json={
            "base_url": "https://user:pass@openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout_seconds": 60,
        },
    )
    assert res.status_code == 422


def test_upstream_update_rejects_bad_scheme(app: TestClient) -> None:
    res = app.patch(
        "/api/admin/config/upstream",
        json={
            "base_url": "file:///etc/passwd",
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout_seconds": 60,
        },
    )
    assert res.status_code == 422


def test_upstream_update_valid_roundtrip(app: TestClient, tmp_path: Path) -> None:
    res = app.patch(
        "/api/admin/config/upstream",
        json={
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout_seconds": 120,
        },
    )
    assert res.status_code == 200
    assert res.json()["upstream"]["timeout_seconds"] == 120

    fresh = create_app(config_path=tmp_path / "config.yaml")
    with TestClient(fresh) as client:
        assert client.get("/api/admin/state").json()["upstream"]["timeout_seconds"] == 120


# ── Pins ──────────────────────────────────────────────────────────────


def test_pins_list_and_clear(app: TestClient) -> None:
    # Seed a pin through the router path.
    res = app.post(
        "/v1/chat/completions",
        json={
            "model": "smart-router",
            "session_id": "session-abc",
            "messages": [{"role": "user", "content": "write me a plan"}],
        },
    )
    # Upstream is not mocked here; any status is fine — the pin is created
    # before the upstream call. Assert via the admin surface instead.
    pins = app.get("/api/admin/pins").json()["pins"]
    assert any(p["slug"] != "" for p in pins)

    if pins:
        pid = pins[0]["id"]
        res = app.delete(f"/api/admin/pins/{pid}")
        assert res.status_code == 200
        assert app.get("/api/admin/pins").json()["pins"] == []


def test_pins_clear_all(app: TestClient) -> None:
    app.post(
        "/v1/chat/completions",
        json={
            "model": "smart-router",
            "session_id": "session-xyz",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    res = app.delete("/api/admin/pins")
    assert res.status_code == 200
    assert res.json()["removed"] >= 1
    assert app.get("/api/admin/pins").json()["pins"] == []
