# Smart Router Proxy

An OpenAI-compatible proxy server with task-aware model routing. Any OpenAI SDK client points at this proxy, requests the virtual model `smart-router`, and each request is classified and routed to the optimal upstream model on OpenRouter.

**Classification pipeline:**
1. **BERT classifier** — local distilbert + MLX model (~5-15ms, ~87% accuracy)

This is the standalone-server counterpart to the [hermes-smart-router](https://github.com/KevinOBytes/hermes-smart-router) Hermes plugin — same routing logic, exposed as an OpenAI-compatible endpoint.

## Quick Start

```bash
# Install
pip install smart-router-proxy

# Set your OpenRouter key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Run
smart-router-proxy
```

> **Note:** The BERT classifier model is not auto-downloaded. Train it from
> [prompt-classifier](https://github.com/KevinOBytes/prompt-classifier) or copy
> the trained `model/` directory to `~/.smart-router-proxy/classifier-model/`.
> Without it, the proxy falls back to `luna`.

The proxy listens on `http://127.0.0.1:8199`. Point any OpenAI client at it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8199/v1",
    api_key="ignored-by-proxy",
)

response = client.chat.completions.create(
    model="smart-router",  # virtual model — proxy routes to the best model
    messages=[{"role": "user", "content": "write a python function to sort a list"}],
)
```

## How It Works

### Classification

Each request is classified into one of 8 task types:

| Task Class | Primary Model | Escalation Model |
|---|---|---|
| `structured_simple` | luna (GPT-5.6) | glm (GLM-5.2) |
| `agentic_execution` | deepseek_flash | sol (GPT-5.6) |
| `software_engineering` | glm (GLM-5.2) | opus (Claude-5) |
| `security_engineering` | sol (GPT-5.6) | fable (Claude-5) |
| `knowledge_reasoning` | glm (GLM-5.2) | kimi_k3 |
| `writing_communication` | sonnet (Claude-5) | opus (Claude-5) |
| `computer_use` | sonnet (Claude-5) | opus (Claude-5) |
| `visual_frontend` | kimi_k3 | opus (Claude-5) |

### Session Pinning

Once a session is routed to a model, that choice is pinned for the session TTL (default 1 hour). Each hit refreshes the pin so active conversations never re-route mid-stream. Pins survive proxy restarts via SQLite storage.

### Usage Ledger

The proxy tracks token counts, costs, and latency per model. Query stats:

```bash
curl http://127.0.0.1:8199/v1/stats
```

## Configuration

Create `config.yaml`:

```yaml
server:
  host: "127.0.0.1"
  port: 8199

upstream:
  base_url: "https://openrouter.ai/api/v1"
  api_key_env: "OPENROUTER_API_KEY"

ollama:
  enabled: false  # set true to enable Gemma LLM fallback

mode: active  # or "fixed" to always use one model
fixed_alias: "luna"

aliases:
  luna:
    model_slug: "openai/gpt-5.6-luna"
```

Or set `SMART_ROUTER_PROXY_CONFIG` to a custom path.

## Client Auth

Set `SMART_ROUTER_PROXY_TOKEN` to require a bearer token from clients.

## License

MIT
