"""Configuration for the smart-router proxy server.

Reads YAML config (default: ./config.yaml or $SMART_ROUTER_PROXY_CONFIG),
with environment-variable overrides for secrets. The routing section reuses
the same schema as the hermes-smart-router plugin's smart_router block.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ClassifierSettings(BaseModel):
    """Local BERT classifier settings."""

    model_path: str = "~/.smart-router-proxy/classifier-model"
    confidence_threshold: float = 0.45


class UpstreamSettings(BaseModel):
    """Upstream OpenAI-compatible API (OpenRouter by default)."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: float = 600.0

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class OllamaSettings(BaseModel):
    """Native macOS Ollama endpoint (loopback only)."""

    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 10.0


class ServerSettings(BaseModel):
    """Proxy server bind settings."""

    host: str = "127.0.0.1"
    port: int = 8199
    # Optional bearer token clients must present; empty = no client auth.
    client_auth_env: str = "SMART_ROUTER_PROXY_TOKEN"

    @property
    def client_token(self) -> str:
        return os.environ.get(self.client_auth_env, "")


class ProxyConfig(BaseModel):
    """Full proxy configuration."""

    server: ServerSettings = Field(default_factory=ServerSettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    upstream: UpstreamSettings = Field(default_factory=UpstreamSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    # Virtual model name the proxy exposes.
    virtual_model: str = "smart-router"
    # Session pin TTL in seconds.
    session_ttl_seconds: int = 3600
    # alias -> {model_slug: str} overrides (same shape as the plugin).
    aliases: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # task class -> {"primary_alias": str, "fallback_alias": str} overrides.
    # Control-panel managed; overrides DEFAULT_ROUTE_TABLE for that class.
    # Entries may also carry direct destinations:
    #   {"primary": {"provider": "openrouter", "model": "<slug>"}, "fallback": {...}}
    route_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # mode: active | fixed
    mode: str = "active"
    fixed_alias: str = "luna"
    # SQLite ledger (usage accounting + pin persistence). ":memory:" for tests.
    store_path: str = "~/.smart-router-proxy/state.db"
    # Optionally prepend a visible routing tag ("[category :: model]") to
    # assistant content so callers can see which task class and model
    # answered. Off by default — responses are passed through verbatim.
    annotate_response: bool = False


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Resolve the config file the runtime will load AND persist to."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("SMART_ROUTER_PROXY_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("config.yaml"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def load_config(path: str | Path | None = None) -> ProxyConfig:
    """Load config from YAML, falling back to defaults."""
    candidate = resolve_config_path(path)
    if candidate.is_file():
        with candidate.open() as fh:
            data = yaml.safe_load(fh) or {}
        return ProxyConfig.model_validate(data)
    return ProxyConfig()


class RuntimeConfig:
    """Lock-protected live configuration with atomic YAML persistence.

    The control panel mutates routing/classifier/behavior/upstream settings
    through this object. Every successful mutation increments ``revision``
    and persists the full config atomically (temp file + os.replace), so a
    failed write leaves both runtime and disk state unchanged. When no
    ``config_path`` is available (e.g. tests), mutations are in-memory only.
    """

    def __init__(self, config: ProxyConfig, config_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._cfg = config
        self._path = Path(config_path).expanduser() if config_path else None
        self._revision = 0

    def get(self) -> ProxyConfig:
        with self._lock:
            return self._cfg.model_copy(deep=True)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def mutate(self, fn: Callable[[ProxyConfig], ProxyConfig]) -> ProxyConfig:
        """Apply ``fn`` to the current config, persist atomically, return the snapshot."""
        with self._lock:
            candidate = fn(self._cfg.model_copy(deep=True))
            # Full validation happens inside the mutator (pydantic models
            # raise before we ever reach disk).
            if self._path is not None:
                fd, tmp = tempfile.mkstemp(
                    dir=str(self._path.parent), prefix=".config.", suffix=".yaml.tmp"
                )
                try:
                    with os.fdopen(fd, "w") as fh:
                        yaml.safe_dump(
                            candidate.model_dump(mode="json"),
                            fh,
                            sort_keys=False,
                            default_flow_style=False,
                        )
                    os.replace(tmp, self._path)
                except Exception:
                    with suppress(OSError):
                        os.unlink(tmp)
                    raise
            self._cfg = candidate
            self._revision += 1
            return self._cfg.model_copy(deep=True)
