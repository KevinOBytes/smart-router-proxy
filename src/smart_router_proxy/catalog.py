"""Model catalog adapters for the control panel.

Fetches and normalizes the live OpenRouter catalog and installed native
Ollama models. Catalog responses are treated as untrusted data: every field
is coerced defensively and unknown metadata is rendered as unknown, never
inferred. Credentials never reach the browser — the OpenRouter catalog is
fetched server-side with the process's own key.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class CatalogError(Exception):
    """Raised when a catalog fetch fails."""


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_openrouter_model(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce one raw OpenRouter /models entry into a stable shape."""
    pricing_raw = raw.get("pricing")
    pricing = pricing_raw if isinstance(pricing_raw, dict) else {}
    modality_raw = raw.get("modality")
    modality = modality_raw if isinstance(modality_raw, list) else []
    return {
        "id": str(raw.get("id") or ""),
        "name": str(raw.get("name") or raw.get("id") or ""),
        "provider": "openrouter",
        "context_length": raw.get("context_length"),
        "pricing": {
            "prompt": _num(pricing.get("prompt")),
            "completion": _num(pricing.get("completion")),
        },
        "modality": [str(m) for m in modality],
    }


def normalize_ollama_model(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce one raw Ollama /api/tags entry into a stable shape."""
    return {
        "id": str(raw.get("name") or raw.get("model") or ""),
        "name": str(raw.get("name") or raw.get("model") or ""),
        "provider": "ollama",
        "size": raw.get("size"),
        "modified_at": raw.get("modified_at"),
        "context_length": None,
        "pricing": {"prompt": None, "completion": None},
        "modality": [],
    }


async def fetch_openrouter_catalog(
    client: httpx.AsyncClient, headers: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Fetch and normalize the OpenRouter model catalog."""
    resp = await client.get("/models", headers=headers or {})
    if resp.status_code != 200:
        raise CatalogError(f"OpenRouter catalog returned HTTP {resp.status_code}")
    data = resp.json()
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise CatalogError("OpenRouter catalog returned no model list")
    models = [normalize_openrouter_model(m) for m in entries if isinstance(m, dict)]
    models.sort(key=lambda m: str(m["id"]).lower())
    return models


async def fetch_ollama_catalog(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Fetch installed native Ollama models from the loopback endpoint."""
    try:
        resp = await client.get("/api/tags")
    except httpx.HTTPError as exc:
        raise CatalogError(f"Ollama unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise CatalogError(f"Ollama catalog returned HTTP {resp.status_code}")
    data = resp.json()
    entries = data.get("models") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise CatalogError("Ollama catalog returned no model list")
    models = [normalize_ollama_model(m) for m in entries if isinstance(m, dict)]
    models.sort(key=lambda m: str(m["id"]).lower())
    return models


class CatalogCache:
    """Server-side catalog cache with staleness tracking.

    A failed refresh keeps the last successful catalog marked ``stale`` so
    the UI degrades independently per provider: an unavailable OpenRouter
    catalog must not block local Ollama configuration and vice versa.
    """

    def __init__(self) -> None:
        self._openrouter: dict[str, Any] | None = None
        self._ollama: dict[str, Any] | None = None

    def _state(self, entry: dict[str, Any] | None) -> dict[str, Any]:
        if entry is None:
            return {"models": [], "fetched_at": None, "stale": False, "error": None}
        return entry

    def snapshot(self) -> dict[str, Any]:
        return {
            "openrouter": self._state(self._openrouter),
            "ollama": self._state(self._ollama),
        }

    async def refresh(
        self,
        http: httpx.AsyncClient,
        ollama: httpx.AsyncClient,
        upstream_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Refresh both catalogs independently; failures never raise."""
        now = time.time()

        try:
            models = await fetch_openrouter_catalog(http, upstream_headers)
            self._openrouter = {
                "models": models,
                "fetched_at": now,
                "stale": False,
                "error": None,
            }
        except Exception as exc:  # catalog is untrusted/unavailable — degrade
            logger.debug("OpenRouter catalog refresh failed: %s", exc)
            self._openrouter = {
                "models": self._openrouter["models"] if self._openrouter else [],
                "fetched_at": self._openrouter["fetched_at"] if self._openrouter else None,
                "stale": self._openrouter is not None,
                "error": str(exc),
            }

        try:
            models = await fetch_ollama_catalog(ollama)
            self._ollama = {
                "models": models,
                "fetched_at": now,
                "stale": False,
                "error": None,
            }
        except Exception as exc:
            logger.debug("Ollama catalog refresh failed: %s", exc)
            self._ollama = {
                "models": self._ollama["models"] if self._ollama else [],
                "fetched_at": self._ollama["fetched_at"] if self._ollama else None,
                "stale": self._ollama is not None,
                "error": str(exc),
            }

        return self.snapshot()
