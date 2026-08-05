# Smart Router Proxy

An OpenAI-compatible proxy server with task-aware model routing. Any OpenAI SDK client points at this proxy, requests the virtual model `smart-router`, and each session is classified (local Gemma via Ollama) and routed to the optimal upstream model on OpenRouter.

This is the standalone-server counterpart to the [hermes-smart-router](https://github.com/KevinOBytes/hermes-smart-router) Hermes plugin — same classifier, route table, and alias mappings (imported from that package), exposed as infrastructure instead of a plugin. Use the plugin inside Hermes Agent; use this proxy for everything else (SDKs, IDEs, LangChain, curl).

```
┌────────────┐   POST /v1/chat/completions    ┌───────────────────┐
│ Any OpenAI  │  model: "smart-router"        │ smart-router-proxy │
│ SDK client  │ ─────────────────────────────>│  classify → route  │
└────────────┘                                └─────────┬─────────┘
                                        deterministic → │ Gemma (Ollama)
                                                        ▼
                                              ┌───────────────────┐
                                              │    OpenRouter      │
                                              │ model: z-ai/glm-5.2│
                                              └───────────────────┘
```

## Endpoints

| Endpoint | Behavior |
|---|---|
| `POST /v1/chat/completions` | Requests for the virtual model are classified and routed; real model names pass through untouched. Streaming and non-streaming. One automatic retry on the alias's fallback model for upstream 429/5xx. |
| `GET /v1/models` | Virtual model first, then the live upstream catalog. |
| `GET /v1/stats` | Content-free usage aggregates: requests, tokens, cached tokens, cache-hit rate, cost, latency — totals, per-model, per-day. |
| `ANY /v1/*` | Generic passthrough for other OpenAI-compatible endpoints (embeddings, legacy completions, moderations). No routing. |
| `GET /healthz` | Liveness + Ollama reachability + upstream key presence. |

## Usage accounting and cache cost visibility

Every request writes a content-free row to a local SQLite ledger (`~/.smart-router-proxy/state.db`): timestamp, hashed session key, alias, model, prompt/completion/cached token counts, cost (from OpenRouter's usage block), latency, and status. Streamed requests are accounted too — the proxy sets `stream_options.include_usage` so the final chunk carries usage.

`GET /v1/stats` returns the aggregates, including **cache_hit_rate** (cached ÷ prompt tokens) — if a change to your setup breaks provider-side prompt caching, it shows up here before it shows up on the bill. No prompt or completion content is ever stored or logged.

Session pins are persisted in the same ledger, so a proxy restart does not re-route an active conversation (a restart-induced model switch would invalidate the provider's per-model cached prefix and re-bill the full context uncached).

## Routing

Identical to hermes-smart-router (shared code):

1. Deterministic classifier — regex patterns for shell/coding/security/GUI/voice tasks, instant.
2. Gemma 4 via local Ollama — 8-class JSON classification for ambiguous requests.
3. Fallback — below-threshold confidence or Ollama unavailable routes to `luna`.

High/critical-risk classifications start directly on the escalation alias. Sessions are pinned: pass a stable `user` field (or `X-Session-Id` header) and the routed model sticks for `session_ttl_seconds`.

See the [hermes-smart-router README](https://github.com/KevinOBytes/hermes-smart-router#task-routing-table) for the full task routing table and default model mappings.

## Install and run

```bash
git clone https://github.com/KevinOBytes/smart-router-proxy.git
cd smart-router-proxy
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

export OPENROUTER_API_KEY="sk-or-v1-..."
cp config.example.yaml config.yaml   # optional; defaults work

.venv/bin/smart-router-proxy          # binds 127.0.0.1:8199
```

## Client usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8199/v1", api_key="unused")
resp = client.chat.completions.create(
    model="smart-router",
    messages=[{"role": "user", "content": "Fix the failing pytest suite"}],
    user="my-session-1",  # stable session key → pinned route
)
```

```bash
curl -s http://127.0.0.1:8199/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "smart-router", "messages": [{"role": "user", "content": "hello"}]}'
```

## Security

- Binds `127.0.0.1` by default — never exposed to the network unless you change it.
- Upstream API key comes from the environment only (`OPENROUTER_API_KEY`); it is never read from config files or logged.
- Optional client auth: set `SMART_ROUTER_PROXY_TOKEN` and clients must send it as a bearer token.
- No prompt content in logs — only request id, alias, and routed model slug.
- `config.yaml` is gitignored; only `config.example.yaml` ships.

## Development

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m mypy src/smart_router_proxy/
```

## License

MIT

## Author

Kevin O'Connor — [kevinbytes.com](https://kevinbytes.com)
