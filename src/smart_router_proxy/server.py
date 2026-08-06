"""OpenAI-compatible proxy server with task-aware model routing.

Endpoints:
  POST /v1/chat/completions — classify, swap model, forward to upstream
                              (streaming and non-streaming), usage-accounted
  GET  /v1/models           — virtual model + upstream catalog passthrough
  GET  /v1/stats            — content-free usage/cost/cache-hit aggregates
  ANY  /v1/*                — generic passthrough (embeddings, completions, …)
  GET  /healthz             — liveness + upstream reachability

The proxy never logs or stores prompt content or credentials — the ledger
holds token counts, cost, latency, model names, and hashed session keys only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from smart_router_proxy.admin import build_admin_router
from smart_router_proxy.catalog import CatalogCache
from smart_router_proxy.config import ProxyConfig, RuntimeConfig, load_config, resolve_config_path
from smart_router_proxy.router import Router, extract_user_text
from smart_router_proxy.store import Store, hash_key, parse_usage

logger = logging.getLogger("smart_router_proxy")

# Upstream statuses worth one retry on the fallback model.
_RETRYABLE = {429, 500, 502, 503, 504}

# Control-panel assets live beside the package.
_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    config: ProxyConfig | None = None, config_path: str | Path | None = None
) -> FastAPI:
    cfg = config or load_config(config_path)
    store = Store(cfg.store_path)
    store.prune_pins(cfg.session_ttl_seconds)
    router = Router(cfg, store=store)
    runtime = RuntimeConfig(cfg, config_path=config_path or resolve_config_path(config_path))
    catalog = CatalogCache()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.http = httpx.AsyncClient(
            base_url=cfg.upstream.base_url,
            timeout=cfg.upstream.timeout_seconds,
        )
        app.state.ollama_http = httpx.AsyncClient(
            base_url=cfg.ollama.base_url,
            timeout=cfg.ollama.timeout_seconds,
        )
        yield
        await app.state.http.aclose()
        await app.state.ollama_http.aclose()
        store.close()

    app = FastAPI(title="smart-router-proxy", lifespan=lifespan)

    def _check_client_auth(request: Request) -> None:
        token = cfg.server.client_token
        if not token:
            return
        header = request.headers.get("authorization", "")
        if header != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    def _upstream_headers() -> dict[str, str]:
        key = cfg.upstream.api_key
        if not key:
            raise HTTPException(
                status_code=500,
                detail=f"Upstream API key not set ({cfg.upstream.api_key_env})",
            )
        return {"Authorization": f"Bearer {key}"}

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        checks: dict[str, Any] = {"server": True}
        try:
            from smart_router_proxy.bert_classifier import get_classifier

            bert = get_classifier()
            checks["classifier"] = bert is not None
        except Exception:
            checks["classifier"] = False
        checks["upstream_key_set"] = bool(cfg.upstream.api_key)
        status = 200 if all(checks.values()) else 503
        return JSONResponse(checks, status_code=status)

    # ── Control panel (localhost-first, same auth boundary) ─────────────

    def _ollama_client() -> httpx.AsyncClient | None:
        try:
            client = app.state.ollama_http
        except Exception:
            return None
        return client if isinstance(client, httpx.AsyncClient) else None

    admin = build_admin_router(
        runtime=runtime,
        store=store,
        catalog=catalog,
        router_runtime=router,
        get_upstream_headers=_upstream_headers,
        get_ollama_client=_ollama_client,
        require_auth=_check_client_auth,
    )
    app.include_router(admin)

    @app.get("/ui", include_in_schema=False)
    async def control_panel() -> FileResponse:
        index = _STATIC_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Control panel not bundled")
        return FileResponse(index, media_type="text/html")

    if _STATIC_DIR.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/v1/stats")
    async def stats(request: Request) -> JSONResponse:
        _check_client_auth(request)
        return JSONResponse(store.stats())

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        _check_client_auth(request)
        virtual = {
            "id": cfg.virtual_model,
            "object": "model",
            "owned_by": "smart-router-proxy",
        }
        data = [virtual]
        try:
            resp = await request.app.state.http.get(
                "/models", headers=_upstream_headers()
            )
            if resp.status_code == 200:
                upstream = resp.json().get("data", [])
                data.extend(m for m in upstream if m.get("id") != cfg.virtual_model)
        except HTTPException:
            raise
        except Exception as exc:
            logger.debug("upstream model list failed: %s", exc)
        return JSONResponse({"object": "list", "data": data})

    def _record(
        payload: dict[str, Any],
        *,
        session_key: str | None,
        alias: str,
        model: str,
        started: float,
        status: int,
    ) -> None:
        """Write one content-free usage row; never raises."""
        try:
            usage = parse_usage(payload) or {}
            store.record_usage(
                session_hash=hash_key(session_key) if session_key else "-",
                alias=alias,
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cached_tokens=usage.get("cached_tokens", 0),
                cost=usage.get("cost", 0.0),
                latency_ms=(time.time() - started) * 1000,
                status=status,
            )
        except Exception as exc:
            logger.debug("usage recording failed: %s", exc)

    def _annotate_content(content: str, model: str, alias: str, category: str) -> str:
        """Prepend a visible 'category :: model' note to assistant content.

        The response body's ``model`` field is rewritten to the routed slug
        already (client-visible via API), but humans reading plain text also
        need to see which category and model answered. Only applied when
        ``annotate_response`` is enabled.
        """
        return f"[{category} :: {model} ({alias})]\n{content}"

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        _check_client_auth(request)
        try:
            body: dict[str, Any] = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        requested_model = str(body.get("model", ""))
        messages = body.get("messages") or []
        # Hermes/OpenRouter provider profiles send the sticky key as a
        # top-level ``session_id`` field. OpenAI clients may instead use
        # ``user`` or an X-Session-Id header. Accept all supported forms so
        # the proxy's per-session pin survives every transport path.
        session_key = (
            str(body.get("session_id", ""))
            or str(body.get("user", ""))
            or request.headers.get("x-session-id", "")
            or request.headers.get("x-hermes-session-id", "")
            or None
        )

        # Route only the virtual model; pass real model names through untouched.
        routed_alias = "-"
        routed_slug = ""
        routed_category = "-"
        routed_fallback: str | None = None
        if requested_model in (cfg.virtual_model, f"tko/{cfg.virtual_model}"):
            text = extract_user_text(messages)
            decision = await router.route(text, session_key)
            routed_slug = decision.slug
            routed_alias = decision.alias
            routed_category = decision.category
            routed_fallback = decision.fallback_slug
            body["model"] = routed_slug
            # Inject classification metadata so downstream tracing (LiteLLM →
            # Langfuse) records the routing decision in the trace.
            existing_meta = body.get("metadata")
            if not isinstance(existing_meta, dict):
                existing_meta = {}
            existing_meta.setdefault("task_class", routed_category)
            existing_meta.setdefault("router_alias", routed_alias)
            existing_meta.setdefault("router_slug", routed_slug)
            body["metadata"] = existing_meta
            # OpenRouter sticky routing: pin this conversation to the same
            # upstream backend so provider-side prompt caches keep hitting.
            if session_key and "session_id" not in body:
                body["session_id"] = session_key
            logger.info(
                "routed request_id=%s alias=%s category=%s model=%s",
                uuid.uuid4().hex[:8],
                routed_alias,
                routed_category,
                routed_slug,
            )

        headers = _upstream_headers()
        if routed_alias != "-":
            # LiteLLM's Langfuse callback reads trace metadata from these
            # request headers. Keep LiteLLM as the sole Langfuse sender while
            # preserving the classifier decision in its one trace.
            headers["langfuse_trace_metadata"] = json.dumps(
                {
                    "task_class": routed_category,
                    "router_alias": routed_alias,
                    "router_slug": routed_slug,
                }
            )
            headers["langfuse_generation_name"] = "smart-router-proxy"
        stream = bool(body.get("stream"))
        http: httpx.AsyncClient = request.app.state.http
        started = time.time()

        if not stream:
            resp = await http.post("/chat/completions", json=body, headers=headers)
            # One retry on the fallback slug for transient upstream failures
            # (429 / 5xx). Only for routed requests with a fallback — either
            # the alias's configured fallback or the decision's direct one.
            if resp.status_code in _RETRYABLE and (
                routed_alias != "-" or routed_fallback is not None
            ):
                fb = routed_fallback or router.fallback_slug(routed_alias)
                if fb and fb != body.get("model"):
                    logger.warning(
                        "upstream %s on %s — retrying on fallback %s",
                        resp.status_code,
                        body.get("model"),
                        fb,
                    )
                    body["model"] = fb
                    resp = await http.post(
                        "/chat/completions", json=body, headers=headers
                    )
            payload = resp.json()
            _record(
                payload,
                session_key=session_key,
                alias=routed_alias,
                model=str(body.get("model", "")),
                started=started,
                status=resp.status_code,
            )
            if routed_alias != "-" and cfg.annotate_response:
                try:
                    msg = payload.get("choices", [{}])[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        msg["content"] = _annotate_content(
                            content, routed_slug, routed_alias, routed_category
                        )
                except Exception as exc:
                    logger.debug("content annotation failed: %s", exc)
            return JSONResponse(payload, status_code=resp.status_code)

        # Streaming: ask the upstream to append a final usage chunk so the
        # ledger gets token/cost accounting for streamed requests too.
        stream_options = body.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options.setdefault("include_usage", True)
        body["stream_options"] = stream_options

        async def stream_upstream() -> AsyncIterator[bytes]:
            status = 0
            buffer = b""
            first = True
            async with http.stream(
                "POST", "/chat/completions", json=body, headers=headers
            ) as upstream:
                status = upstream.status_code
                async for chunk in upstream.aiter_bytes():
                    buffer = (buffer + chunk)[-16384:]  # tail only, for usage
                    if first and routed_alias != "-" and status == 200 and cfg.annotate_response:
                        first = False
                        try:
                            note = json.dumps(
                                {
                                    "id": "chatcmpl-route-note",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": routed_slug,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "role": "assistant",
                                                "content": (
                                                    f"[{routed_category} :: "
                                                    f"{routed_slug} ({routed_alias})]\n"
                                                ),
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            yield f"data: {note}\n\n".encode()
                        except Exception as exc:
                            logger.debug("stream annotation failed: %s", exc)
                    yield chunk
            # Parse the final SSE data lines for the usage-bearing chunk.
            usage_payload: dict[str, Any] = {}
            for line in buffer.decode(errors="ignore").splitlines():
                line = line.strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    obj = json.loads(line[5:].strip())
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("usage"):
                    usage_payload = obj
            _record(
                usage_payload,
                session_key=session_key,
                alias=routed_alias,
                model=str(body.get("model", "")),
                started=started,
                status=status,
            )

        return StreamingResponse(stream_upstream(), media_type="text/event-stream")

    @app.api_route(
        "/v1/{path:path}", methods=["GET", "POST", "DELETE"], include_in_schema=False
    )
    async def passthrough(request: Request, path: str) -> Response:
        """Generic passthrough for other OpenAI-compatible endpoints
        (embeddings, legacy completions, moderations, …). No routing —
        the client's model name is forwarded verbatim."""
        _check_client_auth(request)
        headers = _upstream_headers()
        http: httpx.AsyncClient = request.app.state.http
        content = await request.body()
        if request.headers.get("content-type"):
            headers["Content-Type"] = request.headers["content-type"]
        resp = await http.request(
            request.method, f"/{path}", content=content or None, headers=headers
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="smart-router-proxy")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    uvicorn.run(create_app(cfg), host=host, port=port)


if __name__ == "__main__":
    main()
