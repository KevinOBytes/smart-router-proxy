"""Configuration for the smart-router proxy server.

Reads YAML config (default: ./config.yaml or $SMART_ROUTER_PROXY_CONFIG),
with environment-variable overrides for secrets. The routing section reuses
the same schema as the hermes-smart-router plugin's smart_router block.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OllamaSettings(BaseModel):
    """Local Ollama classifier settings."""

    enabled: bool = False
    """Set True to enable Gemma LLM classification (disabled by default)."""
    model: str = "gemma4:31b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_output_tokens: int = 512
    confidence_threshold: float = 0.45


class UpstreamSettings(BaseModel):
    """Upstream OpenAI-compatible API (OpenRouter by default)."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: float = 600.0

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


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
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    upstream: UpstreamSettings = Field(default_factory=UpstreamSettings)
    # Virtual model name the proxy exposes.
    virtual_model: str = "smart-router"
    # Session pin TTL in seconds.
    session_ttl_seconds: int = 3600
    # alias -> {model_slug: str} overrides (same shape as the plugin).
    aliases: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # mode: active | fixed
    mode: str = "active"
    fixed_alias: str = "luna"
    # SQLite ledger (usage accounting + pin persistence). ":memory:" for tests.
    store_path: str = "~/.smart-router-proxy/state.db"


def load_config(path: str | Path | None = None) -> ProxyConfig:
    """Load config from YAML, falling back to defaults."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("SMART_ROUTER_PROXY_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("config.yaml"))

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open() as fh:
                data = yaml.safe_load(fh) or {}
            return ProxyConfig.model_validate(data)
    return ProxyConfig()
