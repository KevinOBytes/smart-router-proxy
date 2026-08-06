"""Request router: classify → destination (provider + concrete model).

Reuses the hermes-smart-router package for the default route table and
alias mappings so proxy and plugin stay in lockstep. Control-panel route
overrides may pin direct (provider, model) destinations for any task
class, so the full OpenRouter and Ollama catalogs are routable.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Final

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

# Logical alias used for control-panel direct destinations (no alias exists).
DIRECT_ALIAS = "custom"

# Pin tuple layout: (slug, alias, category, provider, fallback_slug, fallback_provider, pinned_at)
# Final so mypy resolves tuple indexing to the exact element type.
PIN_ALIAS: Final = 1
PIN_PROVIDER: Final = 3
PIN_FALLBACK_SLUG: Final = 4
PIN_FALLBACK_PROVIDER: Final = 5
PIN_PINNED_AT: Final = 6


@dataclass
class RouteDecision:
    """A fully resolved routing decision.

    ``slug`` is the concrete model id sent upstream (an OpenRouter slug or
    an installed Ollama model name). ``provider`` is explicit so dispatch
    never infers the endpoint from the model name.
    """

    slug: str
    alias: str
    category: str
    provider: str = "openrouter"
    fallback_slug: str | None = None
    fallback_provider: str | None = None

    @property
    def routed(self) -> bool:
        """True when this decision came from routing (not a passthrough)."""
        return self.alias != "-"


class Router:
    """Classifies request text and resolves it to a concrete destination."""

    def __init__(self, config: ProxyConfig, store: Store | None = None) -> None:
        self._config = config
        self._lock = threading.Lock()
        # session key -> (slug, alias, category, provider, fallback_slug,
        #                 fallback_provider, pinned_at)
        self._pins: dict[str, tuple[str, str, str, str, str | None, str | None, float]] = {}
        self._store = store

    async def route(
        self, text: str, session_key: str | None = None
    ) -> RouteDecision:
        """Return the RouteDecision for the request text.

        ``category`` is the classified TaskClass value, or "-" when unknown
        (fixed mode, fallback, or empty text).
        """
        cfg = self._config

        if cfg.mode == "fixed":
            return self._decision_for_alias(cfg.fixed_alias, "-")

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
                if pin and (now - pin[PIN_PINNED_AT]) < cfg.session_ttl_seconds:
                    refreshed = (
                        pin[0],
                        pin[1],
                        pin[2],
                        pin[3],
                        pin[4],
                        pin[5],
                        now,
                    )
                    self._pins[session_key] = refreshed
                    if self._store is not None:
                        self._store.save_pin(
                            hash_key(session_key),
                            slug=pin[0],
                            alias=pin[1],
                            category=pin[2],
                            pinned_at=now,
                            provider=pin[3],
                            fallback_slug=pin[4],
                            fallback_provider=pin[5],
                        )
                    return RouteDecision(
                        slug=pin[0],
                        alias=pin[1],
                        category=pin[2],
                        provider=pin[3],
                        fallback_slug=pin[4],
                        fallback_provider=pin[5],
                    )

        decision = await self._classify_decision(text)

        if session_key:
            now = time.time()
            with self._lock:
                if len(self._pins) > 1024:
                    cutoff = now - cfg.session_ttl_seconds
                    self._pins = {
                        k: v for k, v in self._pins.items() if v[PIN_PINNED_AT] >= cutoff
                    }
                self._pins[session_key] = (
                    decision.slug,
                    decision.alias,
                    decision.category,
                    decision.provider,
                    decision.fallback_slug,
                    decision.fallback_provider,
                    now,
                )
            if self._store is not None:
                self._store.save_pin(
                    hash_key(session_key),
                    slug=decision.slug,
                    alias=decision.alias,
                    category=decision.category,
                    pinned_at=now,
                    provider=decision.provider,
                    fallback_slug=decision.fallback_slug,
                    fallback_provider=decision.fallback_provider,
                )
        return decision

    def fallback_slug(self, alias: str) -> str | None:
        """Return the configured fallback slug for an alias, if any."""
        fb = self._fallback_for_alias(alias)
        return fb[1] if fb else None

    def clear_pin_by_hash(self, key_hash: str) -> bool:
        """Remove a pin by its opaque hash from memory and the store.

        The control panel only ever sees hashes, so this matches the
        in-memory cache by re-hashing each cached raw key.
        """
        with self._lock:
            removed_store = bool(self._store and self._store.clear_pin(key_hash))
            removed_mem = False
            for k in [k for k in self._pins if hash_key(k) == key_hash]:
                del self._pins[k]
                removed_mem = True
            return removed_store or removed_mem

    def clear_all_pins(self) -> int:
        """Remove every pin from memory and the store; returns count."""
        with self._lock:
            n = len(self._pins)
            self._pins.clear()
            if self._store is not None:
                n = max(n, self._store.clear_pins())
            return n

    async def _classify_decision(self, text: str) -> RouteDecision:
        """Return the RouteDecision for the given text."""
        if not text:
            return self._decision_for_alias(FALLBACK_ALIAS, "-")

        try:
            bert = get_classifier()
            result = bert.classify_to_result(text)
            if result is not None:
                return self._apply_route(result)
        except Exception as exc:
            logger.debug("BERT classifier unavailable: %s", exc)

        return self._decision_for_alias(FALLBACK_ALIAS, "-")

    def _apply_route(self, result: ClassifierResult) -> RouteDecision:
        """Map a ClassifierResult to a RouteDecision.

        Returns the escalation destination when risk is high/critical, else
        the primary destination. Control-panel route_overrides replace the
        primary/fallback destinations for a task class while keeping
        escalation behavior intact.
        """
        category = result.task_class.value
        if result.confidence < self._config.classifier.confidence_threshold:
            return self._decision_for_alias(FALLBACK_ALIAS, category)

        override = self._config.route_overrides.get(category)
        if override and isinstance(override.get("primary"), dict):
            primary = override["primary"]
            fallback = override.get("fallback") or {}
            if primary.get("model"):
                if (
                    result.risk.value in ("high", "critical")
                    and isinstance(fallback, dict)
                    and fallback.get("model")
                ):
                    return RouteDecision(
                        slug=str(fallback["model"]),
                        alias=DIRECT_ALIAS,
                        category=category,
                        provider=str(fallback.get("provider", "openrouter")),
                    )
                return RouteDecision(
                    slug=str(primary["model"]),
                    alias=DIRECT_ALIAS,
                    category=category,
                    provider=str(primary.get("provider", "openrouter")),
                )

        route = DEFAULT_ROUTE_TABLE.get(result.task_class)
        if route is None:
            return self._decision_for_alias(FALLBACK_ALIAS, category)
        alias = str(route[1]) if result.risk.value in ("high", "critical") else str(route[0])
        return self._decision_for_alias(alias, category)

    def _decision_for_alias(self, alias: str, category: str) -> RouteDecision:
        """Build a RouteDecision for a logical alias (with fallback info)."""
        provider, slug = self._resolve_alias(alias)
        fb = self._fallback_for_alias(alias)
        return RouteDecision(
            slug=slug,
            alias=alias,
            category=category,
            provider=provider,
            fallback_slug=fb[1] if fb else None,
            fallback_provider=fb[0] if fb else None,
        )

    def _resolve_alias(self, alias: str) -> tuple[str, str]:
        """Return (provider, slug) for an alias, honoring config overrides."""
        overrides = self._config.aliases
        if alias in overrides:
            entry = overrides[alias]
            if isinstance(entry, dict):
                slug = entry.get("model_slug")
                if slug:
                    provider = entry.get("provider", "openrouter")
                    return str(provider), str(slug)
        mapping = DEFAULT_ALIAS_MAPPINGS.get(alias)
        if mapping is not None:
            return str(mapping.provider), str(mapping.model_slug)
        return "openrouter", str(DEFAULT_ALIAS_MAPPINGS[FALLBACK_ALIAS].model_slug)

    def _fallback_for_alias(self, alias: str) -> tuple[str, str] | None:
        """Return (provider, slug) fallback for an alias, if any."""
        overrides = self._config.aliases
        if alias in overrides:
            entry = overrides[alias]
            if isinstance(entry, dict):
                fb = entry.get("fallback_slug")
                if fb:
                    fb_provider = entry.get("fallback_provider", "openrouter")
                    return str(fb_provider), str(fb)
        mapping = DEFAULT_ALIAS_MAPPINGS.get(alias)
        if mapping is not None and mapping.fallback_slug:
            return "openrouter", str(mapping.fallback_slug)
        return None


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
