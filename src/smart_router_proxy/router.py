"""Request router: classify → alias → concrete model slug.

Reuses the hermes-smart-router package for the deterministic classifier,
route table, and alias mappings so proxy and plugin stay in lockstep.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

import httpx
from hermes_smart_router.deterministic import classify_deterministic
from hermes_smart_router.models import (
    DEFAULT_ALIAS_MAPPINGS,
    DEFAULT_ROUTE_TABLE,
    ClassifierResult,
)

from smart_router_proxy.config import ProxyConfig

logger = logging.getLogger(__name__)

FALLBACK_ALIAS = "luna"

_CLASSIFICATION_PROMPT = (
    "Classify this request into exactly one category.\n\n"
    "Categories: structured_simple, agentic_execution, software_engineering, "
    "security_engineering, knowledge_reasoning, writing_communication, "
    "computer_use, visual_frontend\n\n"
    "Return ONLY valid JSON:\n"
    '{"task_class": "<category>", "risk": "low|moderate|high|critical", '
    '"sensitivity": "public|internal|confidential|restricted", '
    '"confidence": 0.0-1.0}\n\n'
    "Request:"
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class Router:
    """Classifies request text and resolves it to a concrete model slug."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        # session key -> (slug, alias, pinned_at)
        self._pins: dict[str, tuple[str, str, float]] = {}

    async def route(
        self, text: str, session_key: str | None = None
    ) -> tuple[str, str]:
        """Return (concrete_slug, alias) for the given request text."""
        cfg = self._config

        if cfg.mode == "fixed":
            return self._resolve_alias(cfg.fixed_alias), cfg.fixed_alias

        # Session pin: sliding TTL — every hit refreshes the pin so an active
        # conversation never re-routes mid-stream. A mid-conversation model
        # switch invalidates the provider's cached prompt prefix (cache-read
        # discounts are per-model) and re-bills the full context uncached.
        if session_key:
            with self._lock:
                pin = self._pins.get(session_key)
                if pin and (time.time() - pin[2]) < cfg.session_ttl_seconds:
                    self._pins[session_key] = (pin[0], pin[1], time.time())
                    return pin[0], pin[1]

        alias = await self._classify_alias(text)
        slug = self._resolve_alias(alias)

        if session_key:
            with self._lock:
                if len(self._pins) > 1024:
                    cutoff = time.time() - cfg.session_ttl_seconds
                    self._pins = {
                        k: v for k, v in self._pins.items() if v[2] >= cutoff
                    }
                self._pins[session_key] = (slug, alias, time.time())
        return slug, alias

    async def _classify_alias(self, text: str) -> str:
        if not text:
            return FALLBACK_ALIAS

        result = classify_deterministic(text, None)
        if result is None:
            result = await self._classify_gemma(text)
        if result is None:
            return FALLBACK_ALIAS

        if result.confidence < self._config.ollama.confidence_threshold:
            return FALLBACK_ALIAS

        route = DEFAULT_ROUTE_TABLE.get(result.task_class)
        if route is None:
            return FALLBACK_ALIAS
        if result.risk.value in ("high", "critical"):
            return str(route[1])
        return str(route[0])

    async def _classify_gemma(self, text: str) -> ClassifierResult | None:
        o = self._config.ollama
        try:
            async with httpx.AsyncClient(timeout=o.timeout_seconds) as client:
                resp = await client.post(
                    f"{o.base_url.rstrip('/')}/api/generate",
                    json={
                        "model": o.model,
                        "prompt": f"{_CLASSIFICATION_PROMPT} {text[:4000]}",
                        "stream": False,
                        "options": {
                            "temperature": o.temperature,
                            "num_predict": o.max_output_tokens,
                        },
                    },
                )
                resp.raise_for_status()
                raw = str(resp.json().get("response", "")).strip()
        except Exception as exc:
            logger.debug("gemma classification unavailable: %s", exc)
            return None

        if not raw:
            return None
        fence = _JSON_FENCE_RE.search(raw)
        if fence:
            raw = fence.group(1)
        try:
            data: Any = json.loads(raw)
            return ClassifierResult.model_validate(data)
        except Exception as exc:
            logger.debug("gemma returned unparseable classification: %s", exc)
            return None

    def _resolve_alias(self, alias: str) -> str:
        overrides = self._config.aliases
        if alias in overrides:
            slug = overrides[alias].get("model_slug")
            if slug:
                return str(slug)
        mapping = DEFAULT_ALIAS_MAPPINGS.get(alias)
        if mapping is not None:
            return str(mapping.model_slug)
        return str(DEFAULT_ALIAS_MAPPINGS[FALLBACK_ALIAS].model_slug)


def extract_user_text(messages: list[dict[str, Any]]) -> str:
    """Pull the latest user message text from an OpenAI messages array."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(p for p in parts if p)
    return ""
