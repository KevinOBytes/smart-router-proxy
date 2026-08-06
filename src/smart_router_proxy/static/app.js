"use strict";

/* smart-router-proxy control panel — localhost-first admin UI.
   Renders server state only; never displays credentials. All catalog
   content is treated as untrusted data and rendered via textContent. */

const $ = (id) => document.getElementById(id);

let state = null;
let catalog = { openrouter: { models: [] }, ollama: { models: [] } };

function toast(msg, kind = "") {
  const el = $("toast");
  el.textContent = msg;
  el.className = "toast " + kind;
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 4000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

function setStatus(id, text, cls) {
  const el = $(id);
  el.textContent = text;
  el.className = "status " + cls;
}

/* ── State ─────────────────────────────────────────────────────────── */

async function loadState() {
  try {
    state = await api("/api/admin/state");
    renderHealth();
    renderRouting();
    renderClassifier();
    renderBehavior();
    renderUpstream();
    $("revision-badge").textContent = "rev " + state.revision;
  } catch (err) {
    setStatus("h-proxy", "unreachable", "err");
    $("conn-badge").textContent = "disconnected";
    $("conn-badge").className = "badge err";
    toast("Failed to load state: " + err.message, "err");
  }
}

function renderHealth() {
  const s = state.server, c = state.classifier, u = state.upstream, o = state.ollama;
  setStatus("h-proxy", `${s.host}:${s.port}${s.loopback ? "" : " (non-loopback)"}`, s.loopback ? "ok" : "warn");
  setStatus("h-classifier", c.ready ? "ready" : "not loaded", c.ready ? "ok" : "warn");
  setStatus("h-upstream", u.api_key_set ? "key set" : "key missing", u.api_key_set ? "ok" : "err");
  setStatus("h-ollama", o.reachable ? "reachable" : "unavailable", o.reachable ? "ok" : "warn");
  setStatus("h-auth", s.client_auth_configured ? "configured" : "none (localhost)", s.client_auth_configured ? "ok" : "muted");
}

/* ── Catalog ───────────────────────────────────────────────────────── */

async function refreshCatalog(silent = false) {
  const btn = $("btn-refresh-catalog");
  if (btn) btn.disabled = true;
  try {
    catalog = await api("/api/admin/catalog/refresh", { method: "POST" });
    if (!silent) {
      const n = catalog.openrouter.models.length;
      const m = catalog.ollama.models.length;
      toast(`Catalog refreshed: ${n} OpenRouter, ${m} Ollama${catalog.openrouter.stale || catalog.ollama.stale ? " (some stale)" : ""}`);
    }
    renderRouting();
  } catch (err) {
    if (!silent) toast("Catalog refresh failed: " + err.message, "err");
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* ── Routing matrix ────────────────────────────────────────────────── */

function renderRouting() {
  const body = $("routing-body");
  body.innerHTML = "";
  if (!state.routing.length) {
    // Static constant string only — no interpolated data (XSS-safe).
    body.innerHTML = '<tr><td colspan="6" class="muted">No task classes.</td></tr>';
    return;
  }
  for (const row of state.routing) {
    const tr = document.createElement("tr");

    const tdLabel = document.createElement("td");
    tdLabel.textContent = row.label;
    const tdClass = document.createElement("td");
    tdClass.className = "mono muted";
    tdClass.textContent = row.task_class;

    const tdPrimary = document.createElement("td");
    tdPrimary.appendChild(modelSelect(row.task_class, "primary", row.primary));
    const tdFallback = document.createElement("td");
    tdFallback.appendChild(modelSelect(row.task_class, "fallback", row.fallback));

    const tdProvider = document.createElement("td");
    const tag = document.createElement("span");
    tag.className = "tag " + (row.primary ? row.primary.provider : "openrouter");
    tag.textContent = row.primary ? row.primary.provider : "openrouter";
    tdProvider.appendChild(tag);

    const tdState = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = "state-pill " + (row.overridden ? "changed" : "persisted");
    pill.textContent = row.overridden ? "customized" : "default";
    tdState.appendChild(pill);

    const tdBtn = document.createElement("td");
    const btn = document.createElement("button");
    btn.className = "btn ghost";
    btn.textContent = "Save";
    btn.addEventListener("click", () => saveRouting(row.task_class, btn));
    tdBtn.appendChild(btn);

    tr.append(tdLabel, tdClass, tdPrimary, tdFallback, tdProvider, tdState, tdBtn);
    body.appendChild(tr);
  }
}

function filterText() {
  const el = $("model-filter");
  return el ? el.value.trim().toLowerCase() : "";
}

function matchesFilter(text) {
  const f = filterText();
  if (!f) return true;
  return text.toLowerCase().includes(f);
}

function aliasDest(alias) {
  if (!state || !state.routing.length) return null;
  return state.routing[0].destinations.find((d) => d.alias === alias) || null;
}

/* Build a <select> whose options are:
   - built-in aliases (value = alias name)
   - full OpenRouter catalog (value = "openrouter::<id>", filtered)
   - installed Ollama models (value = "ollama::<id>", filtered) */
function modelSelect(taskClass, slot, current) {
  const select = document.createElement("select");
  select.dataset.task = taskClass;
  select.dataset.slot = slot;

  const aliases = new Map();
  for (const dest of state.routing[0].destinations) {
    aliases.set(dest.alias, `${dest.alias} — ${dest.provider}:${dest.model_slug}`);
  }
  const orModels = catalog.openrouter.models || [];
  const olModels = catalog.ollama.models || [];

  const groupAliases = document.createElement("optgroup");
  groupAliases.label = "Built-in aliases";
  for (const [alias, label] of aliases) {
    const opt = document.createElement("option");
    opt.value = alias;
    opt.textContent = label;
    groupAliases.appendChild(opt);
  }
  select.appendChild(groupAliases);

  if (orModels.length) {
    const group = document.createElement("optgroup");
    group.label = `OpenRouter (${orModels.length})`;
    for (const m of orModels) {
      if (!matchesFilter(m.id + " " + (m.name || ""))) continue;
      const opt = document.createElement("option");
      opt.value = "openrouter::" + m.id;
      opt.textContent = m.id + (m.name && m.name !== m.id ? ` — ${m.name}` : "");
      group.appendChild(opt);
    }
    select.appendChild(group);
  }

  if (olModels.length) {
    const group = document.createElement("optgroup");
    group.label = `Ollama installed (${olModels.length})`;
    for (const m of olModels) {
      if (!matchesFilter(m.id)) continue;
      const opt = document.createElement("option");
      opt.value = "ollama::" + m.id;
      opt.textContent = m.id;
      group.appendChild(opt);
    }
    select.appendChild(group);
  }

  // Select the current destination if it maps to an option.
  if (current) {
    if (current.kind === "direct") {
      const v = `${current.provider}::${current.model_slug}`;
      if ([...select.options].some((o) => o.value === v)) select.value = v;
    } else if (aliases.has(current.alias)) {
      select.value = current.alias;
    }
  }
  if (!select.value) {
    // Fall back to the first alias so the picker always has a value.
    select.value = [...aliases.keys()][0] || "";
  }
  return select;
}

/* Parse a select value into {provider, model} — aliases resolve through
   the state destinations list, catalog picks carry "provider::model". */
function parsePick(value) {
  if (!value) return null;
  const idx = value.indexOf("::");
  if (idx !== -1) {
    return { provider: value.slice(0, idx), model: value.slice(idx + 2) };
  }
  const d = aliasDest(value);
  return d ? { provider: d.provider, model: d.model_slug } : null;
}

async function saveRouting(taskClass, btn) {
  const primarySel = document.querySelector(`select[data-task="${taskClass}"][data-slot="primary"]`);
  const fallbackSel = document.querySelector(`select[data-task="${taskClass}"][data-slot="fallback"]`);
  const pVal = primarySel.value;
  const fVal = fallbackSel.value;

  btn.disabled = true;
  try {
    if (!pVal.includes("::") && !fVal.includes("::")) {
      // Both slots are built-in aliases — use the alias endpoint so
      // reverting to defaults drops the override cleanly.
      await api("/api/admin/config/routing", {
        method: "PATCH",
        body: JSON.stringify({
          task_class: taskClass,
          primary_alias: pVal,
          fallback_alias: fVal,
        }),
      });
    } else {
      const primary = parsePick(pVal);
      const fallback = fVal ? parsePick(fVal) : null;
      if (!primary) throw new Error("Could not resolve primary destination");
      await api("/api/admin/config/routing/direct", {
        method: "PATCH",
        body: JSON.stringify({ task_class: taskClass, primary, fallback }),
      });
    }
    toast(`Routing updated for ${taskClass}`);
    await loadState();
  } catch (err) {
    toast("Routing update failed: " + err.message, "err");
  } finally {
    btn.disabled = false;
  }
}

/* ── Classifier ────────────────────────────────────────────────────── */

function renderClassifier() {
  const c = state.classifier;
  $("conf-threshold").value = c.confidence_threshold;
  $("classifier-readonly").textContent = `model: ${c.model_path}`;
}

async function saveClassifier() {
  const threshold = parseFloat($("conf-threshold").value);
  if (isNaN(threshold)) return toast("Invalid threshold", "err");
  try {
    await api("/api/admin/config/classifier", {
      method: "PATCH",
      body: JSON.stringify({ confidence_threshold: threshold }),
    });
    toast("Classifier threshold applied");
    await loadState();
  } catch (err) {
    toast("Classifier update failed: " + err.message, "err");
  }
}

async function classify() {
  const text = $("classify-text").value.trim();
  if (!text) return toast("Enter a prompt to classify", "warn");
  const btn = $("btn-classify");
  btn.disabled = true;
  try {
    const res = await api("/api/admin/classify", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    const out = $("classify-result");
    out.textContent =
      `task_class : ${res.task_class}\n` +
      `alias      : ${res.alias}\n` +
      `provider   : ${res.provider}\n` +
      `model_slug : ${res.model_slug}\n` +
      `note       : ${res.note}`;
    out.classList.remove("hidden");
  } catch (err) {
    toast("Classification failed: " + err.message, "err");
  } finally {
    btn.disabled = false;
  }
}

/* ── Behavior ──────────────────────────────────────────────────────── */

function renderBehavior() {
  const b = state.behavior;
  $("mode").value = b.mode;
  $("fixed-alias").value = b.fixed_alias;
  $("annotate").checked = b.annotate_response;
  $("session-ttl").value = b.session_ttl_seconds;
  $("virtual-model-label").textContent = `virtual model: ${b.virtual_model} (read-only)`;
}

async function saveBehavior() {
  try {
    await api("/api/admin/config/behavior", {
      method: "PATCH",
      body: JSON.stringify({
        mode: $("mode").value,
        fixed_alias: $("fixed-alias").value.trim(),
        annotate_response: $("annotate").checked,
        session_ttl_seconds: parseInt($("session-ttl").value, 10),
      }),
    });
    toast("Behavior applied");
    await loadState();
  } catch (err) {
    toast("Behavior update failed: " + err.message, "err");
  }
}

/* ── Upstream ──────────────────────────────────────────────────────── */

function renderUpstream() {
  const u = state.upstream;
  $("upstream-url").value = u.base_url;
  $("upstream-env").value = u.api_key_env;
  $("upstream-timeout").value = u.timeout_seconds;
  $("upstream-key-status").textContent = u.api_key_set ? "key present in process" : "key NOT set in process";
}

async function saveUpstream() {
  const msg = $("upstream-save-msg");
  msg.textContent = "validating…";
  try {
    await api("/api/admin/config/upstream", {
      method: "PATCH",
      body: JSON.stringify({
        base_url: $("upstream-url").value.trim(),
        api_key_env: $("upstream-env").value.trim(),
        timeout_seconds: parseFloat($("upstream-timeout").value),
      }),
    });
    msg.textContent = "applied";
    toast("Upstream configuration applied");
    await loadState();
  } catch (err) {
    msg.textContent = "";
    toast("Upstream update failed: " + err.message, "err");
  }
}

/* ── Pins ──────────────────────────────────────────────────────────── */

async function loadPins() {
  try {
    const res = await api("/api/admin/pins");
    const body = $("pins-body");
    body.innerHTML = "";
    if (!res.pins.length) {
      // Static constant string only — no interpolated data (XSS-safe).
      body.innerHTML = '<tr><td colspan="6" class="muted">No active pins.</td></tr>';
      return;
    }
    for (const pin of res.pins) {
      const tr = document.createElement("tr");
      const tdId = document.createElement("td");
      tdId.className = "mono";
      tdId.textContent = pin.id;
      const tdAlias = document.createElement("td");
      tdAlias.className = "mono";
      tdAlias.textContent = pin.alias;
      const tdSlug = document.createElement("td");
      tdSlug.className = "mono";
      tdSlug.textContent = pin.slug;
      const tdCat = document.createElement("td");
      tdCat.className = "mono muted";
      tdCat.textContent = pin.category;
      const tdTime = document.createElement("td");
      tdTime.className = "mono muted";
      tdTime.textContent = new Date(pin.pinned_at * 1000).toLocaleString();
      const tdBtn = document.createElement("td");
      const btn = document.createElement("button");
      btn.className = "btn danger";
      btn.textContent = "Clear";
      btn.addEventListener("click", () => clearPin(pin.id));
      tdBtn.appendChild(btn);
      tr.append(tdId, tdAlias, tdSlug, tdCat, tdTime, tdBtn);
      body.appendChild(tr);
    }
  } catch (err) {
    toast("Failed to load pins: " + err.message, "err");
  }
}

async function clearPin(id) {
  if (!confirm(`Clear pin ${id}? The conversation may switch models on its next request.`)) return;
  try {
    await api("/api/admin/pins/" + encodeURIComponent(id), { method: "DELETE" });
    toast("Pin cleared");
    await loadPins();
  } catch (err) {
    toast("Failed to clear pin: " + err.message, "err");
  }
}

async function clearAllPins() {
  if (!confirm("Clear ALL session pins? Active conversations may switch models on their next request.")) return;
  try {
    const res = await api("/api/admin/pins", { method: "DELETE" });
    toast(`Cleared ${res.removed} pin(s)`);
    await loadPins();
  } catch (err) {
    toast("Failed to clear pins: " + err.message, "err");
  }
}

/* ── Wire up ───────────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  $("btn-save-classifier").addEventListener("click", saveClassifier);
  $("btn-classify").addEventListener("click", classify);
  $("btn-save-behavior").addEventListener("click", saveBehavior);
  $("btn-save-upstream").addEventListener("click", saveUpstream);
  $("btn-refresh-catalog").addEventListener("click", () => refreshCatalog(false));
  $("btn-clear-all-pins").addEventListener("click", clearAllPins);
  $("model-filter").addEventListener("input", () => {
    if (state) renderRouting();
  });

  loadState().then(loadPins);
  // Warm the catalog on load so the pickers show the full model range.
  refreshCatalog(true);
});
