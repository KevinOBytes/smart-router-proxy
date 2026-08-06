"""Tests for routing logic (no network required)."""

from __future__ import annotations

import pytest

from smart_router_proxy.config import ProxyConfig
from smart_router_proxy.models import ClassifierResult, RiskLevel, Sensitivity, TaskClass
from smart_router_proxy.router import FALLBACK_ALIAS, Router, extract_user_text


@pytest.fixture()
def router(monkeypatch: pytest.MonkeyPatch) -> Router:
    # Keep routing tests deterministic and independent of the installed model
    # weights. Integration tests cover the real classifier separately.
    fake_result = ClassifierResult(
        task_class=TaskClass.SOFTWARE_ENGINEERING,
        risk=RiskLevel.MODERATE,
        sensitivity=Sensitivity.INTERNAL,
        confidence=0.91,
    )
    monkeypatch.setattr(
        "smart_router_proxy.router.get_classifier",
        lambda: type("C", (), {"classify_to_result": lambda self, text: fake_result})(),
    )
    return Router(ProxyConfig())


class TestExtractUserText:
    def test_string_content(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        assert extract_user_text(msgs) == "hello"

    def test_multipart_content(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        assert extract_user_text(msgs) == "part one\npart two"

    def test_latest_user_wins(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert extract_user_text(msgs) == "second"

    def test_empty(self) -> None:
        assert extract_user_text([]) == ""


class TestRouting:
    async def test_deterministic_coding_route(self, router: Router) -> None:
        decision = await router.route(
            "Fix the failing pytest suite in my Python repo and refactor the module"
        )
        assert decision.alias == "glm"
        assert decision.slug == "z-ai/glm-5.2"
        assert decision.provider == "openrouter"
        assert decision.category == "software_engineering"
        assert decision.routed is True

    async def test_empty_text_falls_back(self, router: Router) -> None:
        decision = await router.route("")
        assert decision.alias == FALLBACK_ALIAS
        assert decision.category == "-"
        assert decision.slug == "openai/gpt-5.6-luna"

    async def test_session_pinning(self, router: Router) -> None:
        d1 = await router.route(
            "Fix the failing pytest suite in my repo", session_key="s1"
        )
        # Same session: different text must reuse the pinned route.
        d2 = await router.route("now say hello", session_key="s1")
        assert (d1.slug, d1.alias, d1.category, d1.provider) == (
            d2.slug,
            d2.alias,
            d2.category,
            d2.provider,
        )

    async def test_session_pin_ttl_slides_on_hit(self) -> None:
        """Active conversations must never re-route (cache-cost protection).

        Each pin hit refreshes pinned_at, so a conversation that stays active
        past the TTL keeps its model — a mid-stream model switch would
        invalidate the provider's cached prompt prefix and re-bill the full
        context at uncached rates.
        """
        import time as _time

        cfg = ProxyConfig(session_ttl_seconds=3600)
        r = Router(cfg)
        d1 = await r.route(
            "Fix the failing pytest suite in my repo", session_key="s-slide"
        )
        # Simulate a pin created 59 minutes ago (1 min before expiry).
        with r._lock:
            pin = r._pins["s-slide"]
            r._pins["s-slide"] = (*pin[:6], _time.time() - 3540)
        # A hit at minute 59 must refresh pinned_at to now, not keep the
        # original timestamp.
        d2 = await r.route("continue please", session_key="s-slide")
        assert (d2.slug, d2.alias) == (d1.slug, d1.alias)
        with r._lock:
            refreshed_at = r._pins["s-slide"][6]
        assert _time.time() - refreshed_at < 5

    async def test_fixed_mode(self) -> None:
        cfg = ProxyConfig(mode="fixed", fixed_alias="sonnet")
        r = Router(cfg)
        decision = await r.route("anything at all")
        assert decision.alias == "sonnet"
        assert decision.slug == "anthropic/claude-sonnet-5"
        assert decision.provider == "openrouter"
        assert decision.category == "-"

    async def test_alias_override(self) -> None:
        cfg = ProxyConfig(aliases={"luna": {"model_slug": "custom/other-model"}})
        r = Router(cfg)
        decision = await r.route("")
        assert decision.slug == "custom/other-model"

    async def test_route_override_direct_openrouter(self, router: Router) -> None:
        cfg = router._config
        cfg.route_overrides = {
            "software_engineering": {
                "primary": {"provider": "openrouter", "model": "org/frontier-model"},
                "fallback": {"provider": "openrouter", "model": "org/backup"},
            }
        }
        decision = await router.route(
            "Fix the failing pytest suite in my Python repo and refactor the module"
        )
        assert decision.slug == "org/frontier-model"
        assert decision.provider == "openrouter"
        assert decision.alias == "custom"
        assert decision.category == "software_engineering"

    async def test_route_override_direct_ollama(self, router: Router) -> None:
        cfg = router._config
        cfg.route_overrides = {
            "software_engineering": {
                "primary": {"provider": "ollama", "model": "qwen3.6:35b-a3b-q4_K_M"},
                "fallback": None,
            }
        }
        decision = await router.route(
            "Fix the failing pytest suite in my Python repo and refactor the module"
        )
        assert decision.slug == "qwen3.6:35b-a3b-q4_K_M"
        assert decision.provider == "ollama"
        assert decision.alias == "custom"

    async def test_route_override_direct_escalates_to_fallback(
        self, router: Router
    ) -> None:
        cfg = router._config
        cfg.route_overrides = {
            "software_engineering": {
                "primary": {"provider": "openrouter", "model": "org/primary"},
                "fallback": {"provider": "openrouter", "model": "org/fallback"},
            }
        }

        def risky_classify(text: str) -> ClassifierResult:
            return ClassifierResult(
                task_class=TaskClass.SOFTWARE_ENGINEERING,
                risk=RiskLevel.CRITICAL,
                sensitivity=Sensitivity.INTERNAL,
                confidence=0.95,
            )

        import smart_router_proxy.router as router_mod

        router_mod.get_classifier = lambda: type(
            "C", (), {"classify_to_result": lambda self, t: risky_classify(t)}
        )()
        decision = await router.route("some risky request")
        assert decision.slug == "org/fallback"
        assert decision.provider == "openrouter"
