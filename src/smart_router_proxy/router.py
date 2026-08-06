"""Request router: classify → alias → concrete model slug.

Reuses the hermes-smart-router package for the route table and alias
mappings so proxy and plugin stay in lockstep.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from smart_router_proxy.bert_classifier import get_classifier
from smart_router_proxy.config import ProxyConfig
from smart_router_proxy.models import (
    DEFAULT_ALIAS_MAPPINGS,
    DEFAULT_ROUTE_TABLE,
    ClassifierResult,
)
from smart_router_proxy.store import Store, hash_key

logger = logging.getLogger(__name__)

FALLBACK_ALIAS = "luna"


class Router:
    """Classifies request text and resolves it to a concrete model slug."""

    def __init__(self, config: ProxyConfig, store: Store | None = None) -> None:
        self._config = config
        self._lock = threading.Lock()
        # session key -> (slug, alias, category, pinned_at)
        self._pins: dict[str, tuple[str, str, str, float]] = {}
        # Optional durable pin store — pins survive proxy restarts, so an
        # active conversation keeps its routed model instead of being
        # reclassified (and possibly model-switched) after a restart.
        self._store = store

    async def route(
        self, text: str, session_key: str | None = None
    ) -> tuple[str, str, str]:
        """Return (concrete_slug, alias, task_category) for the request text.

        ``task_category`` is the classified TaskClass value (e.g. "software_engineering"),
        or "-" when unknown (fixed mode, fallback, or empty text).
        """
        cfg = self._config

        if cfg.mode == "fixed":
            return self._resolve_alias(cfg.fixed_alias), cfg.fixed_alias, "-"

        # Session pin: sliding TTL — every hit refreshes the pin so an active
        # conversation never re-routes mid-stream. A mid-conversation model
        # switch invalidates the provider's cached prompt prefix (cache-read
        # discounts are per-model) and re-bills the full context uncached.
        if session_key:
            now = time.time()
            with self._lock:
                pin = self._pins.get(session_key)
                if pin is None and self._store is not None:
                    pin = self._store.get_pin(hash_key(session_key))
                    if pin is not None:
                        self._pins[session_key] = pin
                if pin and (now - pin[3]) < cfg.session_ttl_seconds:
                    refreshed = (pin[0], pin[1], pin[2], now)
                    self._pins[session_key] = refreshed
                    if self._store is not None:
                        self._store.save_pin(
                            hash_key(session_key), pin[0], pin[1], pin[2], now
                        )
                    return pin[0], pin[1], pin[2]

        alias, category = await self._classify_alias(text)
        slug = self._resolve_alias(alias)

        if session_key:
            now = time.time()
            with self._lock:
                if len(self._pins) > 1024:
                    cutoff = now - cfg.session_ttl_seconds
                    self._pins = {
                        k: v for k, v in self._pins.items() if v[3] >= cutoff
                    }
                self._pins[session_key] = (slug, alias, category, now)
            if self._store is not None:
                self._store.save_pin(hash_key(session_key), slug, alias, category, now)
        return slug, alias, category

    def fallback_slug(self, alias: str) -> str | None:
        """Return the configured fallback slug for an alias, if any."""
        overrides = self._config.aliases
        if alias in overrides:
            fb = overrides[alias].get("fallback_slug")
            if fb:
                return str(fb)
        mapping = DEFAULT_ALIAS_MAPPINGS.get(alias)
        if mapping is not None and mapping.fallback_slug:
            return str(mapping.fallback_slug)
        return None

    async def _classify_alias(self, text: str) -> tuple[str, str]:
        """Return (alias, task_category) for the given text."""
        if not text:
            return FALLBACK_ALIAS, "-"

        # BERT classifier (fast, ~5-15ms)
        try:
            bert = get_classifier()
            result = bert.classify_to_result(text)
            if result is not None:
                return self._apply_route(result)
        except Exception as exc:
            logger.debug("BERT classifier unavailable: %s", exc)

        return FALLBACK_ALIAS, "-"

    def _apply_route(self, result: ClassifierResult) -> tuple[str, str]:
        """Map a ClassifierResult to (alias, task_category).

        Returns the escalation alias when risk is high/critical, else the
        primary route alias. Category is the classified task class.
        """
        category = result.task_class.value
        if result.confidence < self._config.classifier.confidence_threshold:
            return FALLBACK_ALIAS, category
        route = DEFAULT_ROUTE_TABLE.get(result.task_class)
        if route is None:
            return FALLBACK_ALIAS, category
        if result.risk.value in ("high", "critical"):
            return str(route[1]), category
        return str(route[0]), category

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
