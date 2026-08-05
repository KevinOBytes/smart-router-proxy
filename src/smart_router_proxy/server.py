"""OpenAI-compatible proxy server with task-aware model routing.

Endpoints:
  POST /v1/chat/completions — classify, swap model, forward to upstream
                              (streaming and non-streaming), usage-accounted
  GET  /v1/models           — virtual model + upstream catalog passthrough
  GET  /v1/stats            — content-free usage/cost/cache-hit aggregates
  ANY  /v1/*                — generic passthrough (embeddings, completions, …)
  GET  /healthz             — liveness + Ollama/upstream reachability

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
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from smart_router_proxy.config import ProxyConfig, load_config
from smart_router_proxy.router import Router, extract_user_text
from smart_router_proxy.store import Store, hash_key, parse_usage

logger = logging.getLogger("smart_router_proxy")

# Upstream statuses worth one retry on the fallback model.
_RETRYABLE = {429, 500, 502, 503, 504}


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    store = Store(cfg.store_path)
    store.prune_pins(cfg.session_ttl_seconds)
    router = Router(cfg, store=store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.http = httpx.AsyncClient(
            base_url=cfg.upstream.base_url,
            timeout=cfg.upstream.timeout_seconds,
        )
        yield
        await app.state.http.aclose()
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
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{cfg.ollama.base_url.rstrip('/')}/api/tags")
                checks["ollama"] = r.status_code == 200
        except Exception:
            checks["ollama"] = False
        checks["upstream_key_set"] = bool(cfg.upstream.api_key)
        status = 200 if all(checks.values()) else 503
        return JSONResponse(checks, status_code=status)

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

    def _annotate_content(content: str, model: str, alias: str) -> str:
        """Prepend a visible 'selected model' note to assistant content.

        The response body's ``model`` field is rewritten to the routed slug
        already (client-visible via API), but humans reading plain text also
        need to see which model answered.
        """
        return f"[{alias} :: {model}]\n{content}"

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        _check_client_auth(request)
        try:
            body: dict[str, Any] = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        requested_model = str(body.get("model", ""))
        messages = body.get("messages") or []
        session_key = (
            str(body.get("user", ""))
            or request.headers.get("x-session-id", "")
            or None
        )

        # Route only the virtual model; pass real model names through untouched.
        routed_alias = "-"
        routed_slug = ""
        if requested_model in (cfg.virtual_model, f"tko/{cfg.virtual_model}"):
            text = extract_user_text(messages)
            slug, routed_alias = await router.route(text, session_key)
            routed_slug = slug
            body["model"] = slug
            # OpenRouter sticky routing: pin this conversation to the same
            # upstream backend so provider-side prompt caches keep hitting.
            if session_key and "session_id" not in body:
                body["session_id"] = session_key
            logger.info(
                "routed request_id=%s alias=%s model=%s",
                uuid.uuid4().hex[:8],
                routed_alias,
                slug,
            )

        headers = _upstream_headers()
        stream = bool(body.get("stream"))
        http: httpx.AsyncClient = request.app.state.http
        started = time.time()

        if not stream:
            resp = await http.post("/chat/completions", json=body, headers=headers)
            # One retry on the alias's fallback slug for transient upstream
            # failures (429 / 5xx). Only for routed requests with a fallback.
            if resp.status_code in _RETRYABLE and routed_alias != "-":
                fb = router.fallback_slug(routed_alias)
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
            if routed_alias != "-":
                try:
                    msg = payload.get("choices", [{}])[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        msg["content"] = _annotate_content(
                            content, routed_slug, routed_alias
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
                    if first and routed_alias != "-" and status == 200:
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
                                                    f"[{routed_alias} :: {routed_slug}]\n"
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
