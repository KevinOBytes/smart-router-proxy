"""Tests for the SQLite store, persistent pins, fallback slugs, and usage parsing."""

from __future__ import annotations

import time

from smart_router_proxy.config import ProxyConfig
from smart_router_proxy.router import Router
from smart_router_proxy.store import Store, hash_key, parse_usage


class TestParseUsage:
    def test_full_usage_block(self) -> None:
        payload = {
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 300,
                "cost": 0.0042,
                "prompt_tokens_details": {"cached_tokens": 1100},
            }
        }
        u = parse_usage(payload)
        assert u is not None
        assert u["prompt_tokens"] == 1200
        assert u["completion_tokens"] == 300
        assert u["cached_tokens"] == 1100
        assert u["cost"] == 0.0042

    def test_missing_usage(self) -> None:
        assert parse_usage({}) is None
        assert parse_usage({"usage": None}) is None

    def test_partial_usage(self) -> None:
        u = parse_usage({"usage": {"prompt_tokens": 10}})
        assert u is not None
        assert u["completion_tokens"] == 0
        assert u["cached_tokens"] == 0


class TestStore:
    def test_usage_roundtrip_and_stats(self) -> None:
        s = Store(":memory:")
        s.record_usage(
            session_hash="abc",
            alias="glm",
            model="z-ai/glm-5.2",
            prompt_tokens=1000,
            completion_tokens=200,
            cached_tokens=800,
            cost=0.003,
            latency_ms=850.0,
            status=200,
        )
        s.record_usage(
            session_hash="abc",
            alias="glm",
            model="z-ai/glm-5.2",
            prompt_tokens=2000,
            completion_tokens=100,
            cached_tokens=1900,
            cost=0.002,
            latency_ms=400.0,
            status=200,
        )
        stats = s.stats()
        assert stats["totals"]["requests"] == 2
        assert stats["totals"]["prompt_tokens"] == 3000
        assert stats["totals"]["cached_tokens"] == 2700
        assert stats["totals"]["cache_hit_rate"] == 0.9
        assert stats["by_model"][0]["model"] == "z-ai/glm-5.2"
        assert len(stats["by_day"]) == 1
        s.close()

    def test_pin_roundtrip_and_prune(self) -> None:
        s = Store(":memory:")
        kh = hash_key("session-1")
        s.save_pin(kh, "z-ai/glm-5.2", "glm", "software_engineering", time.time())
        pin = s.get_pin(kh)
        assert pin is not None
        assert pin[0] == "z-ai/glm-5.2"
        assert pin[2] == "software_engineering"
        # Stale pin gets pruned.
        s.save_pin(hash_key("old"), "x", "y", "-", time.time() - 99999)
        s.prune_pins(3600)
        assert s.get_pin(hash_key("old")) is None
        assert s.get_pin(kh) is not None
        s.close()

    def test_hash_key_is_stable_and_opaque(self) -> None:
        assert hash_key("user@example.com") == hash_key("user@example.com")
        assert "user" not in hash_key("user@example.com")
        assert len(hash_key("x")) == 16


class TestPersistentPins:
    async def test_pin_survives_router_restart(self) -> None:
        """Restart-resilience: a new Router with the same store must reuse
        the pinned model instead of reclassifying (cache-cost protection)."""
        store = Store(":memory:")
        cfg = ProxyConfig()

        r1 = Router(cfg, store=store)
        slug1, alias1, cat1 = await r1.route(
            "Fix the failing pytest suite in my repo", session_key="persist-1"
        )

        # Fresh Router simulating a proxy restart — same store, empty memory.
        r2 = Router(cfg, store=store)
        slug2, alias2, cat2 = await r2.route("continue please", session_key="persist-1")
        assert (slug2, alias2, cat2) == (slug1, alias1, cat1)
        store.close()


class TestFallbackSlug:
    def test_default_fallback(self) -> None:
        r = Router(ProxyConfig())
        # fable carries a default fallback (opus) in DEFAULT_ALIAS_MAPPINGS.
        assert r.fallback_slug("fable") == "anthropic/claude-opus-5"

    def test_no_fallback(self) -> None:
        r = Router(ProxyConfig())
        assert r.fallback_slug("glm") is None

    def test_config_override_fallback(self) -> None:
        cfg = ProxyConfig(
            aliases={"glm": {"model_slug": "z-ai/glm-5.2", "fallback_slug": "x/y"}}
        )
        r = Router(cfg)
        assert r.fallback_slug("glm") == "x/y"
