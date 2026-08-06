# smart-router-proxy Control Panel Design

Date: 2026-08-06
Status: Approved architecture, revised: direct OpenRouter + Ollama routing

## 1. Objective

Add a small, localhost-first web control panel to `smart-router-proxy` that:

- shows which primary and fallback models serve each prompt class;
- uses OpenRouter as the default provider while allowing installed native Ollama models to be selected per route;
- lets an operator change routing and operational settings safely;
- applies compatible changes immediately without restarting the router;
- preserves existing session pins unless the operator explicitly clears them;
- makes smart-router-proxy the unified execution endpoint and sends traces directly to Langfuse;
- permits non-loopback binding only through an explicit authenticated advanced override.

The control panel is an operational interface, not a chat interface or a replacement for Langfuse administration.

## 2. Current Architecture

```text
Client / Hermes
      |
      v
smart-router-proxy :8199
  - BERT prompt classification
  - task-class routing
  - session pinning
  - routing metadata
      |
      v
smart-router-proxy provider dispatch
  |-- OpenRouter cloud models (default)
  |-- native macOS Ollama models (selectable)
  `-- Langfuse telemetry (direct from smart-router-proxy)
```

Routing rule: OpenRouter is the default provider for the built-in routing table. Each task class may be changed to an OpenRouter model or an installed native Ollama model for either its primary or fallback destination. smart-router-proxy dispatches directly to the selected provider; LiteLLM is not part of the runtime path. Direct passthrough requests to the proxy are accepted only when the destination validates against the current OpenRouter or Ollama catalog.

The new control panel is served by the existing `smart-router-proxy` FastAPI process:

```text
http://127.0.0.1:8199/ui
```

It does not add another daemon, container, Node runtime, or public service.

## 3. Chosen Approach

Use an embedded FastAPI control plane with bundled plain HTML, CSS, and JavaScript.

Reasons:

- smallest operational and dependency footprint;
- same bind and authentication boundary as the router;
- supports searchable model pickers, live health, and immediate changes;
- no separate frontend build pipeline;
- directly testable through FastAPI's existing test client.

Rejected alternatives:

1. Separate React/Next.js service: better component ecosystem, but unnecessary runtime, build, dependency, and authentication complexity.
2. Server-rendered forms only: simpler, but poor for searchable catalogs, live status, and immediate interaction.
3. Read-only status page: safe but fails the core requirement of operational route changes.

## 4. User Interface

### 4.1 Dashboard

Display health and configuration state for:

- smart-router-proxy;
- classifier readiness;
- OpenRouter upstream;
- OpenRouter catalog;
- native Ollama;
- Langfuse link/health when configured;
- active session-pin count;
- current configuration revision.

Health cards must degrade independently. An unavailable OpenRouter catalog must not prevent local Ollama configuration, and vice versa.

### 4.2 Routing Matrix

Show one row for each of the eight task classes:

- `structured_simple`
- `agentic_execution`
- `software_engineering`
- `security_engineering`
- `knowledge_reasoning`
- `writing_communication`
- `computer_use`
- `visual_frontend`

Each row contains:

- human-readable task label;
- primary model picker;
- fallback model picker;
- provider badge;
- capability warnings when available;
- current live state and whether it differs from the persisted configuration.

Model pickers contain:

- the live OpenRouter catalog, listed first and selected by default;
- installed native Ollama models from the local catalog;
- configured destinations when they cannot be mapped to a current catalog record, clearly marked unavailable/stale.

Every destination carries an explicit provider identity (`openrouter` or `ollama`) rather than relying on ambiguous model-name parsing.

The picker supports search and filters for provider, context length, tool support, vision, and price where reliable metadata exists. Missing metadata is displayed as unknown, never inferred.

### 4.3 Classifier

Editable:

- confidence threshold;
- low-confidence/fallback behavior if represented in the runtime configuration.

Read-only:

- classifier model path;
- classifier readiness and loaded bundle identity.

A classify-only test box accepts a prompt and returns:

- predicted task class;
- confidence;
- selected primary and fallback destinations;
- whether an existing session pin would override the current route.

It must not invoke OpenRouter, Ollama, or any other LLM provider.

### 4.4 Sessions

Display:

- session-pin TTL;
- active pin count;
- content-free pin records: opaque session identifier, alias/model, task class, and expiry/activity time;
- clear-one action;
- clear-all action.

Clear-one and clear-all both require confirmation. The confirmation explains that matching active conversations may change models on their next request. Prompt content is never stored or displayed by this feature.

### 4.5 Behavior

Editable:

- operating mode (`active` or `fixed`);
- fixed alias/model;
- response annotation toggle;
- session-pin TTL.

Read-only:

- virtual model name.

Changing the virtual model name changes the public model identifier that clients must send. Keep it read-only in v1 to avoid breaking existing clients from the UI.

Changes that do not affect the listening socket apply immediately.

### 4.6 Upstream

Editable:

- OpenAI-compatible upstream base URL;
- upstream API-key environment-variable name;
- timeout.

The UI never reads or displays a secret value. It shows only whether the named variable is set in the running process.

Before applying an upstream change, the server:

1. validates URL syntax and allowed scheme;
2. checks that the selected key environment variable is present;
3. performs a bounded health/model-list request where supported;
4. writes and applies the change only if validation succeeds.

If the post-apply health check fails, the prior upstream configuration is restored automatically.

### 4.7 Server

Editable:

- bind address;
- port.

Default choices are loopback only:

- `127.0.0.1`
- `::1`

Bind or port changes require a controlled service restart because the listening socket cannot move in-process. Before restart, the UI displays the destination URL and warns that the current page will disconnect.

A restart coordinator persists a pending revision, initiates the existing LaunchAgent restart, verifies the new health endpoint, and restores the previous revision if the new listener does not become healthy within the configured timeout.

### 4.8 Advanced Network Override

Non-loopback binding is hidden under an Advanced section and denied by default.

It requires all of the following:

1. explicit `allow_network_bind` override enabled;
2. a non-empty client-authentication environment variable configured and present in the process;
3. typed/explicit confirmation identifying the target bind address and port;
4. successful controlled restart and health validation.

`0.0.0.0` and `::` are rejected unless all requirements pass. The UI displays authentication status but never the token value.

The control panel and API use the same authentication requirement as the proxy when non-loopback binding is enabled. Browser mutations must additionally require a same-origin anti-CSRF token.

## 5. Model Catalog and Provider Contract

### 5.1 OpenRouter

Fetch the catalog server-side. Credentials never reach the browser. Normalize relevant fields:

- model ID and display name;
- provider;
- context length;
- input/output price;
- modality/capabilities where explicitly supplied.

Cache the last successful catalog with a retrieval timestamp. If refresh fails, return the cached catalog marked stale.

### 5.2 Ollama

Fetch installed models server-side from `http://127.0.0.1:11434/api/tags`. Normalize model name, size, modification time, and known capability metadata. Ollama models appear as provider-labeled routing choices for primary and fallback routes.

Ollama remains native on macOS and loopback-only. It is not containerized by this feature.

### 5.3 Direct Provider Dispatch

smart-router-proxy owns provider dispatch:

- OpenRouter requests use its OpenAI-compatible API and the configured OpenRouter key environment variable.
- Ollama requests use the native loopback API through a dedicated adapter.
- OpenRouter remains the default for existing and newly initialized routing tables.
- Destinations persist as structured provider/model pairs.
- A route is applied only after the chosen provider validates that destination.
- Langfuse tracing is emitted directly by smart-router-proxy with classification, routing, provider, model, usage, latency, and error metadata.

LiteLLM is not installed, configured, called, monitored, or required by this feature.

## 6. Configuration Model

### 6.1 Source of Truth

Use one persisted smart-router configuration document as the source of truth. Runtime state is a validated in-memory snapshot derived from it.

Do not let browser code edit YAML directly.

### 6.2 Revisions

Each successful mutation creates a revision containing:

- revision ID;
- timestamp;
- changed fields;
- redacted prior and resulting configuration;
- application status;
- whether restart was required.

Keep the most recent 20 revisions and support one-click rollback. Secret values are never included.

### 6.3 Atomic Persistence

Persistence sequence:

1. construct candidate configuration;
2. validate the full Pydantic schema and cross-field security rules;
3. write a temporary file in the same directory;
4. flush and atomically replace the active file;
5. update in-memory runtime state under a lock;
6. run post-apply validation;
7. restore the previous revision if validation fails.

A failed write or validation leaves both runtime and persisted configuration unchanged.

### 6.4 Immediate Application

Apply immediately without restart:

- task-class primary/fallback routes;
- classifier confidence threshold;
- operating mode and fixed selection;
- annotation behavior;
- pin TTL;
- upstream endpoint, key variable name, and timeout after validation.

Require controlled restart:

- server bind address;
- server port.

Existing session pins survive ordinary route changes. They change only through expiry, clear-one, clear-all, or an explicit future migration feature.

## 7. Backend Components

Keep boundaries small and testable:

- `admin_routes.py`: authenticated UI/admin API routes.
- `config_store.py`: schema validation, atomic writes, revisions, rollback.
- `runtime_config.py`: lock-protected live routing/settings snapshot.
- `catalog.py`: OpenRouter and Ollama catalog adapters/cache.
- `service_control.py`: controlled restart and post-restart rollback coordination.
- `static/`: bundled HTML, CSS, and JavaScript.

Do not grow `server.py` into the full control plane. It should assemble these components and retain inference responsibilities.

## 8. Admin API

Proposed endpoints:

- `GET /ui`
- `GET /api/admin/state`
- `GET /api/admin/catalog/models`
- `POST /api/admin/catalog/refresh`
- `POST /api/admin/classify`
- `PATCH /api/admin/config/routing`
- `PATCH /api/admin/config/classifier`
- `PATCH /api/admin/config/behavior`
- `PATCH /api/admin/config/upstream`
- `PATCH /api/admin/config/server`
- `GET /api/admin/revisions`
- `POST /api/admin/revisions/{id}/rollback`
- `GET /api/admin/pins`
- `DELETE /api/admin/pins/{id}`
- `DELETE /api/admin/pins`

Mutation endpoints require same-origin validation, CSRF protection, and client authentication whenever configured. Non-loopback operation requires authentication unconditionally.

## 9. Error Handling

- OpenRouter catalog unavailable: show stale cache with explicit error; routing mutations are limited to known OpenRouter destinations or available Ollama models.
- Ollama unavailable: mark local destinations unavailable; OpenRouter choices remain usable.
- Provider destination rejected: do not apply the route; return an actionable error.
- Invalid config: reject entire mutation; no partial update.
- Atomic write failure: retain old runtime and disk configuration.
- Restart failure: restore prior bind/port revision and restart previous listener.
- Browser disconnect during bind change: status page at the announced new URL reports success/rollback when reachable.

Errors shown in the UI must not include credentials, authorization headers, environment contents, or prompt text.

## 10. Security Requirements

- Loopback-only by default.
- No secret values returned to browser clients.
- Non-loopback requires explicit advanced override plus configured client authentication.
- Same-origin and CSRF checks on mutations.
- Strict Pydantic request models with extra fields forbidden.
- Catalog responses treated as untrusted data and escaped in the UI.
- No arbitrary file paths, shell commands, endpoint probes, or free-form YAML editing.
- URL validation blocks unsupported schemes and credential-bearing URLs.
- Configuration history is redacted.
- Prompt classification test data is processed in memory and not added to configuration history.

## 11. Testing Strategy

Follow test-driven development.

### Unit tests

- full configuration and cross-field validation;
- loopback detection for IPv4/IPv6;
- rejection of unauthenticated non-loopback binds;
- atomic persistence and rollback;
- runtime hot reload;
- catalog normalization/cache behavior;
- secret redaction;
- pin clear-one/clear-all;
- destination validation.

### API tests

- UI and state endpoints;
- mutation authentication/CSRF;
- immediate routing changes;
- invalid mutations leave state unchanged;
- upstream health failure rolls back;
- server bind change creates pending restart state;
- non-loopback policy enforcement.

### Integration tests

- select an OpenRouter model and verify direct flow to OpenRouter;
- select an installed Ollama model and verify direct flow to native Ollama;
- confirm OpenRouter is used by default when no route has been customized;
- confirm smart-router-proxy emits one direct Langfuse trace for both providers;
- confirm classifier metadata remains attached;
- confirm existing pins survive route changes;
- confirm clear-all allows the next request to use the new route.

### UI QA

- search/filter model pickers;
- stale and partial catalog states;
- keyboard navigation and readable labels;
- destructive confirmation flows;
- browser console free of errors;
- responsive layout at typical desktop widths.

## 12. Success Criteria

The feature is complete when:

1. `/ui` shows all eight task classes and their live primary/fallback destinations.
2. The searchable pickers list OpenRouter first/default and installed Ollama models as selectable alternatives.
3. Route changes apply to new sessions immediately and survive process restart.
4. Existing pins remain stable unless explicitly cleared.
5. Classifier, behavior, upstream, and session settings can be managed safely.
6. Bind/port changes use controlled restart and rollback.
7. Non-loopback binds cannot be enabled without explicit override and configured client authentication.
8. Secrets never appear in UI/API responses or revision history.
9. Router requests dispatch directly to the selected OpenRouter or Ollama model and produce one smart-router-proxy-originated Langfuse trace.
10. Existing proxy behavior and tests remain green.

## 13. Out of Scope

- editing Langfuse credentials;
- managing OpenRouter billing/account settings;
- installing or deleting Ollama models;

- displaying full Langfuse traces inside the control panel;
- chat/completion UI;
- multi-user RBAC;
- public internet deployment;
- arbitrary YAML or filesystem editor;
- automatic migration of already-pinned conversations after a route change.

## 14. First Implementation Decision

Implement a provider-neutral destination schema (`provider`, `model`) and dedicated OpenRouter and Ollama adapters. Preserve OpenRouter as the built-in default while allowing explicit Ollama selection. Prove both direct paths and direct Langfuse tracing with integration tests before building the full UI around them.
