"""Tests for the SQLite store, persistent pins, fallback slugs, and usage parsing."""

from __future__ import annotations

import time

import pytest

from smart_router_proxy.config import ProxyConfig
from smart_router_proxy.models import ClassifierResult, RiskLevel, Sensitivity, TaskClass
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
        s.save_pin(kh, "z-ai/glm-5.2", "software_engineering", time.time())
        pin = s.get_pin(kh)
        assert pin is not None
        assert pin[0] == "z-ai/glm-5.2"
        assert pin[1] == "software_engineering"
        # Stale pin gets pruned.
        s.save_pin(hash_key("old"), "x", "-", time.time() - 99999)
        s.prune_pins(3600)
        assert s.get_pin(hash_key("old")) is None
        assert s.get_pin(kh) is not None
        s.close()

    def test_alias_column_migration(self) -> None:
        """Legacy DBs carrying an alias column in usage/pins must migrate
        cleanly to the no-alias schema without data loss."""
        s = Store(":memory:")
        # Simulate a legacy DB with an alias column.
        with s._lock:
            s._db.execute(
                "ALTER TABLE usage ADD COLUMN alias TEXT NOT NULL DEFAULT ''"
            )
            s._db.execute(
                "ALTER TABLE pins ADD COLUMN alias TEXT NOT NULL DEFAULT ''"
            )
            s._db.execute(
                "INSERT INTO usage (ts, session_hash, model, prompt_tokens,"
                " completion_tokens, cached_tokens, cost, latency_ms, status, alias)"
                " VALUES (1.0, 'h', 'm', 10, 5, 0, 0.001, 100.0, 200, 'glm')"
            )
            s._db.commit()
        s._drop_alias_column("usage")
        s._drop_alias_column("pins")
        cols_usage = {
            r[1]
            for r in s._db.execute("PRAGMA table_info(usage)").fetchall()
        }
        cols_pins = {
            r[1] for r in s._db.execute("PRAGMA table_info(pins)").fetchall()
        }
        assert "alias" not in cols_usage
        assert "alias" not in cols_pins
        # Data survived the rebuild.
        row = s._db.execute(
            "SELECT model, prompt_tokens, status FROM usage"
        ).fetchone()
        assert row == ("m", 10, 200)
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
        d1 = await r1.route(
            "Fix the failing pytest suite in my repo", session_key="persist-1"
        )

        # Fresh Router simulating a proxy restart — same store, empty memory.
        r2 = Router(cfg, store=store)
        d2 = await r2.route("continue please", session_key="persist-1")
        assert (d2.slug, d2.category, d2.provider) == (
            d1.slug,
            d1.category,
            d1.provider,
        )
        store.close()

    async def test_direct_pin_survives_router_restart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct (provider, model) destinations persist through restarts."""
        store = Store(":memory:")
        cfg = ProxyConfig()
        cfg.route_overrides = {
            "software_engineering": {
                "primary": {"provider": "ollama", "model": "qwen3.6:35b-a3b-q4_K_M"},
                "fallback": None,
            }
        }

        fake_result = ClassifierResult(
            task_class=TaskClass.SOFTWARE_ENGINEERING,
            risk=RiskLevel.MODERATE,
            sensitivity=Sensitivity.INTERNAL,
            confidence=0.91,
        )
        monkeypatch.setattr(
            "smart_router_proxy.router.get_classifier",
            lambda: type("C", (), {"classify_to_result": lambda self, t: fake_result})(),
        )

        r1 = Router(cfg, store=store)
        d1 = await r1.route(
            "Fix the failing pytest suite in my repo", session_key="persist-direct"
        )
        assert d1.provider == "ollama"

        r2 = Router(cfg, store=store)
        d2 = await r2.route("continue please", session_key="persist-direct")
        assert d2.slug == d1.slug
        assert d2.provider == "ollama"
        store.close()


class TestRetryFallback:
    def test_default_retry_fallback(self) -> None:
        # SECURITY_ENGINEERING escalation model (fable) carries a default
        # retry fallback (opus) on its Destination.
        from smart_router_proxy.models import DEFAULT_ROUTE_TABLE, TaskClass

        escalation = DEFAULT_ROUTE_TABLE[TaskClass.SECURITY_ENGINEERING][1]
        assert escalation.model_slug == "anthropic/claude-fable-5"
        assert escalation.retry_fallback == "anthropic/claude-opus-5"

    def test_no_retry_fallback(self) -> None:
        from smart_router_proxy.models import DEFAULT_ROUTE_TABLE, TaskClass

        primary = DEFAULT_ROUTE_TABLE[TaskClass.SOFTWARE_ENGINEERING][0]
        assert primary.retry_fallback is None

    async def test_route_carries_retry_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A high-risk security_engineering request routes to fable and the
        decision advertises its retry fallback (opus), which server.py uses
        for transient-failure retries — no alias indirection required."""
        store = Store(":memory:")
        cfg = ProxyConfig()
        r = Router(cfg, store=store)

        risky = ClassifierResult(
            task_class=TaskClass.SECURITY_ENGINEERING,
            risk=RiskLevel.CRITICAL,
            sensitivity=Sensitivity.INTERNAL,
            confidence=0.95,
        )
        monkeypatch.setattr(
            "smart_router_proxy.router.get_classifier",
            lambda: type("C", (), {"classify_to_result": lambda self, t: risky})(),
        )
        decision = await r.route("how do I audit a malicious binary")
        assert decision.slug == "anthropic/claude-fable-5"
        assert decision.fallback_slug == "anthropic/claude-opus-5"
        assert decision.provider == "openrouter"
        store.close()
