"""Tests for routing logic (no network required)."""

from __future__ import annotations

import pytest

from smart_router_proxy.config import ProxyConfig
from smart_router_proxy.router import FALLBACK_ALIAS, Router, extract_user_text


@pytest.fixture()
def router() -> Router:
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
        slug, alias, category = await router.route(
            "Fix the failing pytest suite in my Python repo and refactor the module"
        )
        assert alias == "glm"
        assert slug == "z-ai/glm-5.2"
        assert category == "software_engineering"

    async def test_empty_text_falls_back(self, router: Router) -> None:
        slug, alias, category = await router.route("")
        assert alias == FALLBACK_ALIAS
        assert category == "-"

    async def test_session_pinning(self, router: Router) -> None:
        slug1, alias1, cat1 = await router.route(
            "Fix the failing pytest suite in my repo", session_key="s1"
        )
        # Same session: different text must reuse the pinned route.
        slug2, alias2, cat2 = await router.route("now say hello", session_key="s1")
        assert (slug1, alias1, cat1) == (slug2, alias2, cat2)

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
        slug1, alias1, _cat1 = await r.route(
            "Fix the failing pytest suite in my repo", session_key="s-slide"
        )
        # Simulate a pin created 59 minutes ago (1 min before expiry).
        with r._lock:
            s, a, cat, _ = r._pins["s-slide"]
            r._pins["s-slide"] = (s, a, cat, _time.time() - 3540)
        # A hit at minute 59 must refresh pinned_at to now, not keep the
        # original timestamp.
        slug2, alias2, _cat2 = await r.route("continue please", session_key="s-slide")
        assert (slug2, alias2) == (slug1, alias1)
        with r._lock:
            _, _, _, refreshed_at = r._pins["s-slide"]
        assert _time.time() - refreshed_at < 5

    async def test_fixed_mode(self) -> None:
        cfg = ProxyConfig(mode="fixed", fixed_alias="sonnet")
        r = Router(cfg)
        slug, alias, category = await r.route("anything at all")
        assert alias == "sonnet"
        assert slug == "anthropic/claude-sonnet-5"
        assert category == "-"

    async def test_alias_override(self) -> None:
        cfg = ProxyConfig(aliases={"luna": {"model_slug": "custom/other-model"}})
        r = Router(cfg)
        slug, _alias, _cat = await r.route("")
        assert slug == "custom/other-model"
