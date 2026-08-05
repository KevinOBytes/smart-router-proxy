"""Content-free SQLite ledger: usage accounting + session pin persistence.

Stores ONLY metadata — token counts, cost, latency, model, hashed session
keys. Never prompt or completion content. Pins persisted here survive proxy
restarts, so an active conversation keeps its routed model (a restart-induced
re-route would invalidate the provider's per-model cached prompt prefix).
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    ts REAL NOT NULL,
    session_hash TEXT NOT NULL,
    alias TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0.0,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    status INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage (ts);
CREATE INDEX IF NOT EXISTS idx_usage_model ON usage (model);
CREATE TABLE IF NOT EXISTS pins (
    key_hash TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    alias TEXT NOT NULL,
    pinned_at REAL NOT NULL
);
"""


def hash_key(key: str) -> str:
    """Short stable hash so raw session identifiers never touch disk."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def parse_usage(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract token/cost accounting from an OpenAI-compatible response.

    Reads usage.prompt_tokens / completion_tokens, cached input tokens from
    prompt_tokens_details.cached_tokens, and OpenRouter's usage.cost when
    usage accounting is enabled. Returns None when no usage block exists.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    cached = 0
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens") or 0)
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": cached,
        "cost": float(usage.get("cost") or 0.0),
    }


class Store:
    """Thread-safe SQLite ledger."""

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        if path == ":memory:":
            self._db = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(p), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ── Usage ledger ─────────────────────────────────────────────────

    def record_usage(
        self,
        *,
        session_hash: str,
        alias: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        status: int = 0,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO usage (ts, session_hash, alias, model, prompt_tokens,"
                " completion_tokens, cached_tokens, cost, latency_ms, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    session_hash,
                    alias,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    cost,
                    latency_ms,
                    status,
                ),
            )
            self._db.commit()

    def stats(self) -> dict[str, Any]:
        """Aggregate totals, per-model, and per-day (last 30 days)."""
        with self._lock:
            total = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0),"
                " COALESCE(SUM(completion_tokens),0), COALESCE(SUM(cached_tokens),0),"
                " COALESCE(SUM(cost),0.0), COALESCE(AVG(latency_ms),0.0) FROM usage"
            ).fetchone()
            by_model = self._db.execute(
                "SELECT model, COUNT(*), COALESCE(SUM(prompt_tokens),0),"
                " COALESCE(SUM(completion_tokens),0), COALESCE(SUM(cached_tokens),0),"
                " COALESCE(SUM(cost),0.0) FROM usage GROUP BY model"
                " ORDER BY SUM(cost) DESC"
            ).fetchall()
            by_day = self._db.execute(
                "SELECT date(ts, 'unixepoch') AS day, COUNT(*),"
                " COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0),"
                " COALESCE(SUM(cached_tokens),0), COALESCE(SUM(cost),0.0)"
                " FROM usage GROUP BY day ORDER BY day DESC LIMIT 30"
            ).fetchall()

        prompt_total = int(total[1])
        cached_total = int(total[3])
        return {
            "totals": {
                "requests": int(total[0]),
                "prompt_tokens": prompt_total,
                "completion_tokens": int(total[2]),
                "cached_tokens": cached_total,
                "cache_hit_rate": (
                    round(cached_total / prompt_total, 4) if prompt_total else 0.0
                ),
                "cost": round(float(total[4]), 6),
                "avg_latency_ms": round(float(total[5]), 1),
            },
            "by_model": [
                {
                    "model": r[0],
                    "requests": int(r[1]),
                    "prompt_tokens": int(r[2]),
                    "completion_tokens": int(r[3]),
                    "cached_tokens": int(r[4]),
                    "cost": round(float(r[5]), 6),
                }
                for r in by_model
            ],
            "by_day": [
                {
                    "day": r[0],
                    "requests": int(r[1]),
                    "prompt_tokens": int(r[2]),
                    "completion_tokens": int(r[3]),
                    "cached_tokens": int(r[4]),
                    "cost": round(float(r[5]), 6),
                }
                for r in by_day
            ],
        }

    # ── Pin persistence ──────────────────────────────────────────────

    def get_pin(self, key_hash: str) -> tuple[str, str, float] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT slug, alias, pinned_at FROM pins WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
        if row is None:
            return None
        return (str(row[0]), str(row[1]), float(row[2]))

    def save_pin(self, key_hash: str, slug: str, alias: str, pinned_at: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO pins (key_hash, slug, alias, pinned_at)"
                " VALUES (?, ?, ?, ?)",
                (key_hash, slug, alias, pinned_at),
            )
            self._db.commit()

    def prune_pins(self, ttl_seconds: int) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM pins WHERE pinned_at < ?", (time.time() - ttl_seconds,)
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
