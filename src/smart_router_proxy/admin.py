"""Control-panel admin API for smart-router-proxy.

Serves the embedded localhost-first UI and the mutation endpoints that
drive it. Every mutation goes through RuntimeConfig (atomic persistence)
and returns a secret-free snapshot. The UI never sees credentials: only
whether a named env var is present. All admin endpoints live behind the
same loopback/auth boundary as the proxy itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from smart_router_proxy.catalog import CatalogCache
from smart_router_proxy.config import ProxyConfig, RuntimeConfig
from smart_router_proxy.models import DEFAULT_ALIAS_MAPPINGS, DEFAULT_ROUTE_TABLE, TaskClass
from smart_router_proxy.router import Router
from smart_router_proxy.store import Store

logger = logging.getLogger(__name__)

TASK_LABELS: dict[str, str] = {
    "structured_simple": "Structured & Simple",
    "agentic_execution": "Agentic Execution",
    "software_engineering": "Software Engineering",
    "security_engineering": "Security Engineering",
    "knowledge_reasoning": "Knowledge Reasoning",
    "writing_communication": "Writing & Communication",
    "computer_use": "Computer Use",
    "visual_frontend": "Visual & Frontend",
}

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def is_loopback(host: str) -> bool:
    return host.strip().lower() in _LOOPBACK_HOSTS


# ── Request models (strict: extra fields rejected) ──────────────────────


class RoutingRow(BaseModel):
    model_config = {"extra": "forbid"}

    primary_alias: str = Field(min_length=1, max_length=64)
    fallback_alias: str = Field(min_length=1, max_length=64)


class Destination(BaseModel):
    """A direct (provider, model) destination from the live catalogs."""

    model_config = {"extra": "forbid"}

    provider: str = Field(pattern="^(openrouter|ollama)$")
    model: str = Field(min_length=1, max_length=256)


class RoutingUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    task_class: TaskClass
    primary_alias: str = Field(min_length=1, max_length=64)
    fallback_alias: str = Field(min_length=1, max_length=64)


class ClassifierUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    confidence_threshold: float = Field(ge=0.05, le=0.99)


class BehaviorUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    mode: str = Field(pattern="^(active|fixed)$")
    fixed_alias: str = Field(min_length=1, max_length=64)
    annotate_response: bool
    session_ttl_seconds: int = Field(ge=60, le=86400 * 7)


class UpstreamUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    base_url: str = Field(min_length=1, max_length=512)
    api_key_env: str = Field(min_length=1, max_length=128)
    timeout_seconds: float = Field(ge=1.0, le=3600.0)

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if "://" not in v:
            raise ValueError("base_url must include a scheme")
        scheme = v.split("://", 1)[0].lower()
        if scheme not in ("http", "https"):
            raise ValueError("unsupported scheme")
        if "@" in v.split("://", 1)[1]:
            raise ValueError("credential-bearing URLs are not allowed")
        return v.rstrip("/")


class PinId(BaseModel):
    model_config = {"extra": "forbid"}

    id: str = Field(min_length=16, max_length=16)


class ClassifyRequest(BaseModel):
    """Module-level so FastAPI can resolve it as a body model."""

    model_config = {"extra": "forbid"}

    text: str = Field(min_length=1, max_length=4096)


class DirectRoutingUpdate(BaseModel):
    """Pin a task class to direct catalog destinations."""

    model_config = {"extra": "forbid"}

    task_class: TaskClass
    primary: Destination
    fallback: Destination | None = None


# ── Admin API ───────────────────────────────────────────────────────────


def build_admin_router(
    *,
    runtime: RuntimeConfig,
    store: Store,
    catalog: CatalogCache,
    router_runtime: Router,
    get_upstream_headers: Callable[[], dict[str, str]],
    get_ollama_client: Callable[[], httpx.AsyncClient | None],
    require_auth: Callable[[Request], None] | None = None,
) -> APIRouter:
    """Assemble the admin router bound to this app instance's services."""
    router = APIRouter(prefix="/api/admin")

    def _auth_dep(request: Request) -> None:
        if require_auth is not None:
            require_auth(request)

    router.dependencies.append(Depends(_auth_dep))

    def _state() -> dict[str, Any]:
        cfg: ProxyConfig = runtime.get()
        return {
            "revision": runtime.revision,
            "server": {
                "host": cfg.server.host,
                "port": cfg.server.port,
                "loopback": is_loopback(cfg.server.host),
                "client_auth_configured": bool(cfg.server.client_token),
            },
            "classifier": {
                "model_path": cfg.classifier.model_path,
                "confidence_threshold": cfg.classifier.confidence_threshold,
                "ready": _classifier_ready(),
            },
            "upstream": {
                "base_url": cfg.upstream.base_url,
                "api_key_env": cfg.upstream.api_key_env,
                "api_key_set": bool(cfg.upstream.api_key),
                "timeout_seconds": cfg.upstream.timeout_seconds,
            },
            "ollama": {
                "base_url": cfg.ollama.base_url,
                "reachable": get_ollama_client() is not None,
            },
            "behavior": {
                "mode": cfg.mode,
                "fixed_alias": cfg.fixed_alias,
                "annotate_response": cfg.annotate_response,
                "session_ttl_seconds": cfg.session_ttl_seconds,
                "virtual_model": cfg.virtual_model,
            },
            "routing": _routing_rows(cfg),
        }

    def _classifier_ready() -> bool:
        try:
            from smart_router_proxy.bert_classifier import get_classifier

            return get_classifier() is not None
        except Exception:
            return False

    def _routing_rows(cfg: ProxyConfig) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for task in TaskClass:
            primary, fallback = DEFAULT_ROUTE_TABLE[task]
            override = cfg.route_overrides.get(task.value)
            primary_dest = _dest_for_alias(cfg, primary)
            fallback_dest = _dest_for_alias(cfg, fallback)
            overridden = override is not None
            if override:
                direct = isinstance(override.get("primary"), dict) and "model" in override["primary"]
                if direct:
                    prim = override["primary"]
                    primary_dest = {
                        "alias": None,
                        "provider": str(prim.get("provider", "openrouter")),
                        "model_slug": str(prim["model"]),
                        "kind": "direct",
                    }
                    fb = override.get("fallback")
                    if isinstance(fb, dict) and fb.get("model"):
                        fallback_dest = {
                            "alias": None,
                            "provider": str(fb.get("provider", "openrouter")),
                            "model_slug": str(fb["model"]),
                            "kind": "direct",
                        }
                    else:
                        fallback_dest = None
                else:
                    primary_dest = _dest_for_alias(
                        cfg, str(override.get("primary_alias") or primary)
                    )
                    fallback_dest = _dest_for_alias(
                        cfg, str(override.get("fallback_alias") or fallback)
                    )
            rows.append(
                {
                    "task_class": task.value,
                    "label": TASK_LABELS.get(task.value, task.value),
                    "primary_alias": primary_dest["alias"] if primary_dest else "",
                    "fallback_alias": fallback_dest["alias"] if fallback_dest else "",
                    "primary": primary_dest,
                    "fallback": fallback_dest,
                    "overridden": overridden,
                    "destinations": _destinations(cfg),
                }
            )
        return rows

    def _dest_for_alias(cfg: ProxyConfig, alias: str) -> dict[str, Any] | None:
        for d in _destinations(cfg):
            if d["alias"] == alias:
                return d
        return None

    def _destinations(cfg: ProxyConfig) -> list[dict[str, Any]]:
        """Alias → destination (provider, slug) snapshot for the pickers."""
        out: list[dict[str, Any]] = []
        for alias, mapping in DEFAULT_ALIAS_MAPPINGS.items():
            override = cfg.aliases.get(alias, {})
            slug = override.get("model_slug") if isinstance(override, dict) else None
            provider = override.get("provider") if isinstance(override, dict) else None
            out.append(
                {
                    "alias": alias,
                    "provider": provider or mapping.provider,
                    "model_slug": slug or mapping.model_slug,
                    "kind": "alias",
                }
            )
        return out

    def _catalog_known(dest: Destination) -> bool:
        """Validate a direct destination against the live catalog cache.

        A destination is known when it appears in the matching provider's
        catalog snapshot. An empty/unrefreshed catalog is treated as
        authoritative-fail: direct destinations must be provably selectable
        before they are applied.
        """
        snap = catalog.snapshot()
        models = snap.get(dest.provider, {}).get("models", [])
        if not models:
            return False
        return any(str(m.get("id") or "") == dest.model for m in models)

    # ── State & catalog ────────────────────────────────────────────────

    @router.get("/state")
    async def admin_state() -> dict[str, Any]:
        return _state()

    @router.get("/catalog/models")
    async def catalog_models() -> dict[str, Any]:
        return catalog.snapshot()

    @router.post("/catalog/refresh")
    async def catalog_refresh(request: Request) -> dict[str, Any]:
        http: httpx.AsyncClient = request.app.state.http
        return await catalog.refresh(
            http,
            get_ollama_client() or httpx.AsyncClient(base_url="http://127.0.0.1:11434", timeout=10),
            get_upstream_headers(),
        )

    # ── Classify-only test (never calls a provider) ─────────────────────

    @router.post("/classify")
    async def classify_only(body: ClassifyRequest) -> dict[str, Any]:
        from smart_router_proxy.router import FALLBACK_ALIAS, Router

        probe = Router(runtime.get())
        decision = await probe.route(body.text, session_key=None)
        return {
            "task_class": decision.category,
            "alias": decision.alias,
            "model_slug": decision.slug,
            "provider": decision.provider,
            "fallback_alias": FALLBACK_ALIAS,
            "note": "Classification only — no provider was called.",
        }

    # ── Mutations (strict, atomic via RuntimeConfig) ────────────────────

    @router.patch("/config/routing/direct")
    async def update_routing_direct(body: DirectRoutingUpdate) -> dict[str, Any]:
        """Pin a task class to direct catalog destinations.

        Both primary and fallback must exist in the current catalog snapshot
        for their provider; an unrefreshed catalog fails closed so the UI
        cannot silently pin a model that does not exist.
        """
        if not _catalog_known(body.primary):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{body.primary.provider} model '{body.primary.model}' is not in "
                    "the catalog — refresh the catalog first"
                ),
            )
        fallback_dest: dict[str, Any] | None = None
        if body.fallback is not None:
            if not _catalog_known(body.fallback):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{body.fallback.provider} model '{body.fallback.model}' is not "
                        "in the catalog — refresh the catalog first"
                    ),
                )
            fallback_dest = {
                "provider": body.fallback.provider,
                "model": body.fallback.model,
            }

        def apply(cfg: ProxyConfig) -> ProxyConfig:
            overrides = dict(cfg.route_overrides)
            overrides[body.task_class.value] = {
                "primary": {"provider": body.primary.provider, "model": body.primary.model},
                "fallback": fallback_dest,
            }
            cfg.route_overrides = overrides
            return cfg

        try:
            runtime.mutate(apply)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _state()

    @router.patch("/config/routing")
    async def update_routing(body: RoutingUpdate) -> dict[str, Any]:
        def apply(cfg: ProxyConfig) -> ProxyConfig:
            overrides = dict(cfg.route_overrides)
            default_primary, default_fallback = DEFAULT_ROUTE_TABLE[body.task_class]
            if (
                body.primary_alias == default_primary
                and body.fallback_alias == default_fallback
            ):
                # Reverting to built-in defaults drops the override so the
                # matrix shows "default" again instead of a no-op override.
                overrides.pop(body.task_class.value, None)
            else:
                overrides[body.task_class.value] = {
                    "primary_alias": body.primary_alias,
                    "fallback_alias": body.fallback_alias,
                }
            cfg.route_overrides = overrides
            return cfg

        try:
            runtime.mutate(apply)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _state()

    @router.patch("/config/classifier")
    async def update_classifier(body: ClassifierUpdate) -> dict[str, Any]:
        def apply(cfg: ProxyConfig) -> ProxyConfig:
            cfg.classifier.confidence_threshold = body.confidence_threshold
            return cfg

        try:
            runtime.mutate(apply)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _state()

    @router.patch("/config/behavior")
    async def update_behavior(body: BehaviorUpdate) -> dict[str, Any]:
        def apply(cfg: ProxyConfig) -> ProxyConfig:
            cfg.mode = body.mode
            cfg.fixed_alias = body.fixed_alias
            cfg.annotate_response = body.annotate_response
            cfg.session_ttl_seconds = body.session_ttl_seconds
            return cfg

        try:
            runtime.mutate(apply)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _state()

    @router.patch("/config/upstream")
    async def update_upstream(body: UpstreamUpdate) -> dict[str, Any]:
        def apply(cfg: ProxyConfig) -> ProxyConfig:
            cfg.upstream.base_url = body.base_url
            cfg.upstream.api_key_env = body.api_key_env
            cfg.upstream.timeout_seconds = body.timeout_seconds
            return cfg

        try:
            runtime.mutate(apply)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _state()

    # ── Pins (content-free) ─────────────────────────────────────────────

    @router.get("/pins")
    async def list_pins() -> dict[str, Any]:
        return {"pins": store.list_pins()}

    @router.delete("/pins/{pin_id}")
    async def delete_pin(pin_id: str) -> dict[str, Any]:
        if len(pin_id) != 16:
            raise HTTPException(status_code=400, detail="Invalid pin id")
        removed = router_runtime.clear_pin_by_hash(pin_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Pin not found")
        return {"removed": True}

    @router.delete("/pins")
    async def delete_all_pins() -> dict[str, Any]:
        removed = router_runtime.clear_all_pins()
        return {"removed": removed}

    return router
