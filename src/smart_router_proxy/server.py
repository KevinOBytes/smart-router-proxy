"""OpenAI-compatible proxy server with task-aware model routing.

Endpoints:
  POST /v1/chat/completions — classify, swap model, forward to upstream
                              (streaming and non-streaming)
  GET  /v1/models           — virtual model + upstream catalog passthrough
  GET  /healthz             — liveness + Ollama/upstream reachability

The proxy never logs prompt content or credentials.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from smart_router_proxy.config import ProxyConfig, load_config
from smart_router_proxy.router import Router, extract_user_text

logger = logging.getLogger("smart_router_proxy")

_HOP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "authorization",
}


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    router = Router(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.http = httpx.AsyncClient(
            base_url=cfg.upstream.base_url,
            timeout=cfg.upstream.timeout_seconds,
        )
        yield
        await app.state.http.aclose()

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

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        _check_client_auth(request)
        try:
            body: dict[str, Any] = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        requested_model = str(body.get("model", ""))
        messages = body.get("messages") or []

        # Route only the virtual model; pass real model names through untouched.
        routed_alias = None
        if requested_model in (cfg.virtual_model, f"tko/{cfg.virtual_model}"):
            session_key = (
                str(body.get("user", ""))
                or request.headers.get("x-session-id", "")
                or None
            )
            text = extract_user_text(messages)
            slug, routed_alias = await router.route(text, session_key)
            body["model"] = slug
            logger.info(
                "routed request_id=%s alias=%s model=%s",
                uuid.uuid4().hex[:8],
                routed_alias,
                slug,
            )

        headers = _upstream_headers()
        stream = bool(body.get("stream"))
        http: httpx.AsyncClient = request.app.state.http

        if not stream:
            resp = await http.post("/chat/completions", json=body, headers=headers)
            return JSONResponse(resp.json(), status_code=resp.status_code)

        async def stream_upstream() -> AsyncIterator[bytes]:
            async with http.stream(
                "POST", "/chat/completions", json=body, headers=headers
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    yield chunk

        return StreamingResponse(stream_upstream(), media_type="text/event-stream")

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
