# Plan: add GreenPT (EU + US) and Viro AI to the provider catalog

**Goal:** ship `greenpt`, `greenpt-us`, and `viro` as built-in API-key providers so they
appear in `/login` and `/model` with zero code changes (login reads `BUILTIN_PROVIDER_CATALOG`).

## Research summary (verified against live docs, 2026-11-14)

### GreenPT — https://docs.greenpt.ai
- OpenAI-compatible (`/v1/chat/completions`), Bearer auth, env var **`GREENPT_API_KEY`**
  (docs `get-started`; models.dev `greenpt` entry agrees).
- Regional endpoints (docs `/regions`): **one API key works on every endpoint**.
  | Region | Base URL |
  |---|---|
  | EU (default) | `https://api.greenpt.ai/v1` |
  | EU (explicit alias) | `https://api.eu.greenpt.ai/v1` — same endpoint, skip |
  | US | `https://api.us.greenpt.ai/v1` |
- **EU serves the full catalog** (41 model ids in models.dev, e.g. `glm-5.3`,
  `deepseek-v4.1-flash`, `kimi-k3`, `gpt-oss-120b`, `green-l/-r`...).
- **US serves only 5 ids, same ids as EU** (docs `/regions` availability table):
  `deepseek-v4.1-flash`, `glm-5.3-flash`, `glm-5.3`, `kimi-k3`,
  plus non-chat `qwen3-embedding-8b` (embeddings) and `metis` (decision API, not chat) — exclude both.
- Reasoning: `reasoning_effort` parameter, values `none|minimal|low|medium|high`
  (docs `/reasoning`). No model accepts `xhigh`.
- Models.dev has `greenpt` → the generator (`scripts/generate_models.py`) will enrich
  the EU entry automatically (costs, context windows, per-model thinking maps).
- Prices differ per endpoint (EU in EUR; US routes cost differently) — keep costs
  EU-only from models.dev, leave US entry without cost data.

### Viro AI — https://ai.viro.app, api.viro.app, console.viro.app/models
- OpenAI-compatible, base **`https://api.viro.app/v1`** (docs FAQ: "point your existing
  OpenAI client at ... migration is changing the base URL and model name").
- Env var not documented; convention **`VIRO_API_KEY`** (matches `GREENPT_API_KEY` pattern).
- **Full catalog is public at `https://console.viro.app/models`** (server-rendered table,
  scraped 2026-11-14): 45 model rows with context windows + USD per-1M pricing.
- **Id convention** (from the table, this matters):
  - Clean-energy models are prefixed: `viro/deepseek-v4.1-flash`, `viro/glm-5.3` ...
  - Frontier models are provider-prefixed: `anthropic/claude-sonnet-5`, `google/gemini-3.6-flash`,
    `openai/gpt-5.6-luna`, `xai/grok-4.5` — NOT bare ids.
  - The routers (`viro/clean`, `viro/optimized`, `viro/frontier`) are advertised on the
    models page but have **no rows/pricing in the console table** → exclude from `models`.
- Not in models.dev api.json (checked 2026-11-14) → catalog-only entry, no generator enrichment.
- Scope decision: include only the **19 chat-capable `viro/*` ids** (the clean-energy value-add).
  Skip image (`flux-2-*`, `gpt-image-2`, `grok-imagine-image`, `nano-banana-*`) and embedding
  (`qwen3-embedding-8b`, `multilingual-e5-large-instruct`) models; skip the 17 frontier rows
  (`anthropic/*`, `google/*`, `openai/*`, `xai/*`) — they duplicate Tau's native provider
  entries, and a Viro user routes to them via `viro/optimized`/`viro/frontier` if needed.
  (Alternative if wanted: add the frontier rows too; id convention is `provider/model`.)

## Design decisions

0. **Live model discovery is a core requirement** (user decision, 2026-11-14):
   GreenPT and Viro serve open-weight catalogs that churn frequently, so both must
   auto-discover models at runtime from their own `/v1/models` endpoint. The discovery
   pipeline already exists (`ModelCatalogProvider.discover_models()` → session publish →
   model-picker overlay) and is generic except for two codex-only gates; this plan
   implements the capability rather than shipping static-only entries. See
   "Auto-discovery phase" below for the full breakdown.
1. **One catalog entry per region/base URL** — precedent: `minimax`/`minimax-cn`,
   `xiaomi` + `xiaomi-token-plan-cn/ams/sgp`. So:
   - `greenpt` = EU default endpoint, full model list.
   - `greenpt-us` = US endpoint, 4 chat models.
   - No `greenpt-eu` entry (`api.eu.greenpt.ai` is the same EU endpoint under an explicit
     name; adding it is redundant).
2. **Shared env var, distinct credential_name** — `greenpt`/`greenpt-us` both use
   `GREENPT_API_KEY` (precedent: `opencode-go`/`opencode` share `OPENCODE_API_KEY`);
   distinct `credential_name` (`"greenpt"`, `"greenpt-us"`) keeps login/logout keys
   independent in `FileCredentialStore`.
3. **`kind = "openai-compatible"`, `api = "openai-completions"`** for all three — docs
   only expose `/chat/completions` (mirrors `deepseek` entry; models.dev greenpt `api`
   field is the base URL, not an API dialect).
4. **auth_methods** stays at the default `("api_key",)` → providers appear automatically
   in the `/login` API-key picker and `_open_login` routes them to the key screen.
5. **Thinking config**: `thinking_levels = ["off","minimal","low","medium","high"]`
   (no `xhigh` — neither provider documents it), `thinking_default = "off"`,
   `thinking_parameter = "reasoning_effort"` for both GreenPT entries and Viro.
   GreenPT per-model maps come from the models.dev overlay; Viro per-model maps
   (where the model reasons) are hand-written in `model_metadata` since Viro is
   not in models.dev.
6. **Viro id convention** (`provider/model`): the clean catalog is `viro/<id>`, so
   all 19 chat entries carry the `viro/` prefix exactly as the console lists them.
   Frontier rows are `anthropic/...`, `google/...`, `openai/...`, `xai/...` and are
   excluded (duplicates Tau's native providers; routers cover that use case).

## Catalog entries (append to `src/tau_coding/data/catalog.toml`)

```toml
[[providers]]
name = "greenpt"
display_name = "GreenPT"
kind = "openai-compatible"
base_url = "https://api.greenpt.ai/v1"
api_key_env = "GREENPT_API_KEY"
credential_name = "greenpt"
models = ["deepseek-v4-flash-0731", "deepseek-v4.1-flash", "deepseek-v4.1-flash-1017",
  "devstral-2-123b-instruct-2512", "gemma-3-27b-it", "gemma4", "glm-5.1", "glm-5.2",
  "glm-5.2-caveman", "glm-5.2-caveman-lite", "glm-5.2-caveman-ultra", "glm-5.2-honey",
  "glm-5.2-honey-lite", "glm-5.2-honey-ultra", "glm-5.2-ponytail", "glm-5.2-ponytail-lite",
  "glm-5.2-ponytail-ultra", "glm-5.3", "glm-5.3-flash", "gpt-oss-120b", "green-l",
  "green-l-raw", "green-r", "green-r-raw", "green-s", "green-s-pro", "holo2-30b-a3b",
  "kimi-k2.6", "kimi-k2.6-fast", "kimi-k2.7-code", "kimi-k3", "llama-3.3-70b-instruct",
  "minimax-m2.5", "mistral-medium-3.5-128b", "mistral-small-3.2-24b-instruct-2506",
  "pixtral-12b-2409", "qwen3-235b-a22b-instruct-2507", "qwen3-coder-30b-a3b-instruct",
  "qwen3.5-397b-a17b", "qwen3.6-35b-a3b", "voxtral-small-24b-2507"]
default_model = "deepseek-v4.1-flash"
docs_url = "https://docs.greenpt.ai"
api = "openai-completions"
thinking_levels = ["off", "minimal", "low", "medium", "high"]
thinking_default = "off"
thinking_parameter = "reasoning_effort"

[[providers]]
name = "greenpt-us"
display_name = "GreenPT US"
kind = "openai-compatible"
base_url = "https://api.us.greenpt.ai/v1"
api_key_env = "GREENPT_API_KEY"
credential_name = "greenpt-us"
models = ["deepseek-v4.1-flash", "glm-5.3-flash", "glm-5.3", "kimi-k3"]
default_model = "glm-5.3"
docs_url = "https://docs.greenpt.ai/regions"
api = "openai-completions"
thinking_levels = ["off", "minimal", "low", "medium", "high"]
thinking_default = "off"
thinking_parameter = "reasoning_effort"

[[providers]]
name = "viro"
display_name = "Viro AI"
kind = "openai-compatible"
base_url = "https://api.viro.app/v1"
api_key_env = "VIRO_API_KEY"
credential_name = "viro"
models = ["viro/apertus-v1.5-70b", "viro/deepseek-v4-flash-0731", "viro/deepseek-v4-pro",
  "viro/deepseek-v4.1-flash", "viro/deepseek-v4.1-flash-1017", "viro/gemma-4-31b",
  "viro/glm-5.2", "viro/glm-5.3", "viro/glm-5.3-flash", "viro/gpt-oss-120b",
  "viro/gpt-oss-20b", "viro/kimi-k2.5", "viro/kimi-k2.7-code", "viro/kimi-k3",
  "viro/minimax-m3", "viro/mistral-small-3.2", "viro/mistral-small-4",
  "viro/qwen3.5-122b", "viro/qwen3.5-397b", "viro/qwen3.8-27b"]
default_model = "viro/deepseek-v4.1-flash"
docs_url = "https://ai.viro.app"
api = "openai-completions"
thinking_levels = ["off", "minimal", "low", "medium", "high"]
thinking_default = "off"
thinking_parameter = "reasoning_effort"

[providers.context_windows]
"viro/apertus-v1.5-70b" = 100000
"viro/deepseek-v4-flash-0731" = 1000000
"viro/deepseek-v4-pro" = 1000000
"viro/deepseek-v4.1-flash" = 1000000
"viro/deepseek-v4.1-flash-1017" = 1000000
"viro/gemma-4-31b" = 256000
"viro/glm-5.2" = 1000000
"viro/glm-5.3" = 1000000
"viro/glm-5.3-flash" = 1000000
"viro/gpt-oss-120b" = 131072
"viro/gpt-oss-20b" = 131072
"viro/kimi-k2.5" = 262144
"viro/kimi-k2.7-code" = 262144
"viro/kimi-k3" = 1000000
"viro/minimax-m3" = 256000
"viro/mistral-small-3.2" = 131072
"viro/mistral-small-4" = 262144
"viro/qwen3.5-122b" = 262000
"viro/qwen3.5-397b" = 256000
"viro/qwen3.8-27b" = 262144

[providers.model_metadata."viro/deepseek-v4.1-flash"]
name = "DeepSeek V4.1 Flash"
reasoning = true
input = ["text", "image"]
context_window = 1000000
cost = { input = 0.36, output = 1.80, cacheRead = 0.02, cacheWrite = 0 }

# ...one [providers.model_metadata."viro/<id>"] block per model, with `name`,
# `context_window`, and `cost = { input, output, cacheRead?, cacheWrite? }` from the
# console table (USD per 1M tokens). Models with a listed cached-input price get
# `cacheRead`; the rest omit it (0). `reasoning = true` + `input = ["text", "image"]`
# only where the console/GreenPT docs say the model reasons or is multimodal
# (deepseek-v4.1-flash, glm-5.3, glm-5.3-flash, kimi-k3, gemma-4-31b ...).
```

## Auto-discovery phase (GreenPT EU/US + Viro)

Reuses the existing `ModelCatalogProvider` pipeline (currently codex-only);
no new triggers or persistence needed — lazy refresh on provider activation,
cached in `_runtime_model_catalogs`.

1. **`src/tau_ai/openai_compatible.py`** — add `discover_models()` to
   `OpenAICompatibleProvider`: `GET {resolved_base_url}/models` with
   `Authorization: Bearer <api_key>` (reuse the auth resolution in
   `stream_response`, incl. `credential_resolver` base_url override); parse
   `{"data": [{"id": ...}]}` → `RuntimeModelCatalog` with
   `RuntimeModel(id=..., name=id, input_modalities=("text",))`. Non-200/parse
   failure raises; the session catches and falls back to the static catalog.
2. **Opt-in flag** (so the other ~26 openai-compatible providers are unaffected):
   `ProviderCatalogEntry.model_discovery: bool = False`,
   `_CatalogProvider.model_discovery: bool = False`,
   `OpenAICompatibleProviderConfig.model_discovery: bool = False` threaded via
   `provider_config_from_entry`; set `model_discovery = true` on `greenpt`,
   `greenpt-us`, `viro` in catalog.toml. User catalog.toml can enable it for
   custom providers too.
3. **`src/tau_coding/session.py`** — relax the two codex-only gates:
   - `_refresh_runtime_model_limits` (~:2400): gate becomes
     `isinstance(provider, ModelCatalogProvider) and (provider_name == "openai-codex"
     or provider_config.model_discovery)`; generalize the cached-catalog early
     return beyond codex.
   - `_provider_with_runtime_model_catalog` (~:5096) and
     `_runtime_provider_model_metadata`: accept
     `OpenAICompatibleProviderConfig | OpenAICodexProviderConfig` (merge uses only
     shared fields — `model_metadata`, `context_windows`, `thinking_*`).
4. **Tests**: `/v1/models` payload parse; Bearer + correct URL via fake transport;
   flag-off ⇒ no HTTP call; per-region probes (`greenpt` vs `greenpt-us` hit their
   own base_urls — GreenPT's "one list per host" honored by construction); merge
   preserves static metadata for known ids (models.dev costs for EU, hand-written
   viro metadata) and yields minimal metadata for brand-new ids.

Known tradeoffs (accepted):
- Newly discovered ids carry no cost/context until models.dev or the catalog knows
  them (Tau renders unknown cost gracefully).
- GreenPT "not recommended" ids surface from discovery; `removed_models` tombstones
  apply to the static list only. Optional later: post-filter discovered ids.
- Viro `/v1/models` needs no pre-validation — the runtime probe is the validation;
  failures fall back to the static list and record an error.

## Steps

1. **Append the three entries** above to `src/tau_coding/data/catalog.toml`.
2. **Regenerate the models.dev overlay:**
   `uv run python scripts/generate_models.py` → rewrites `data/models-dev-catalog.json`.
   - `greenpt` gets enriched (costs, context windows, thinking maps, new eligible models).
   - `greenpt-us`/`viro` stay catalog-only — the generator skips providers absent from
     models.dev (`models_dev_catalog_document` `continue`). Confirm in the diff that the
     EU-only cost/context data did not leak into the US entry.
   - No `_MODELS_DEV_PROVIDER_KEYS` change needed (`greenpt` key matches the name).
3. **Smoke-validate the catalog:**
   `uv run python -c "from tau_coding.provider_catalog import BUILTIN_PROVIDER_CATALOG as C; print([(p.name, p.base_url, len(p.models)) for p in C if p.name in ('greenpt','greenpt-us','viro')])"`
   (loader enforces: unique names, `default_model ∈ models`, metadata keys ⊆ models,
   `thinking_default ∈ thinking_levels`).
4. **Tests** — add to `tests/test_models_dev.py` or a new `tests/test_provider_catalog.py`:
   - all three names present in `BUILTIN_PROVIDER_CATALOG`;
   - `greenpt.base_url == "https://api.greenpt.ai/v1"`, `greenpt-us.base_url == "https://api.us.greenpt.ai/v1"`, `viro.base_url == "https://api.viro.app/v1"`;
   - `greenpt` and `greenpt-us` share `api_key_env` but have distinct `credential_name`;
   - `set(greenpt-us.models) <= set(greenpt.models)` (docs: US uses the same ids as EU);
   - every model with `reasoning=true` in the generated overlay maps onto a supported
     `thinking_levels` value (no `xhigh` leaks in).
   Run: `uv run pytest tests/test_models_dev.py tests/test_commands.py tests/test_oauth_tui.py -q`.
5. **Login check (manual):** in the TUI, `/login` → API key → search "greenpt"/"viro";
   `/login greenpt-us` and `/login viro` resolve; verify key save + `set_provider` switch
   with a real key for one provider.
6. **Docs:** no provider table exists in `website/content` (checked) — add a `dev-notes/`
   entry per AGENTS.md documenting the additions and the region-modeling decision.

## Verification (needs API keys — sanity checks before merge)

- `curl -H "Authorization: Bearer $GREENPT_API_KEY" https://api.us.greenpt.ai/v1/models`
  → confirm the US list matches the 4 chat ids (docs table may drift).
- `curl -H "Authorization: Bearer $VIRO_API_KEY" https://api.viro.app/v1/models`
  → confirm the 19 `viro/*` ids are callable and context windows match the console
  table; adjust if the API serves more chat models.
- Test one real chat call per provider to confirm `openai-completions` works.
- If a key is never available, the console-scraped data above is still a solid basis
  (it is the same table the console renders), with the `ponytail:` comment noting
  context windows/costs were taken from the console and not API-verified.

## Risks / notes

- **Discovery is the primary mechanism for GreenPT/Viro model freshness**; the static
  lists are the offline fallback, not the source of truth. If `/v1/models` is down or
  the key is invalid, sessions degrade to the static list with an error recorded
  (same behavior as Codex).
- `TAU_OFFLINE` must gate the discovery probes (Codex path already checks it; mirror it).
- Viro prices shown with a 20% promo (`deepseek-v4-flash-0731` has strikethrough
  pricing) — use the current (discounted) figure and note it.
- US pricing differs per route; we ship no cost data for `greenpt-us` — Tau handles
  missing cost gracefully (cost shown as unknown).
- Viro context windows are display values from the console (262.1K → 262144, etc.);
  the runtime discovery does not supply context — keep the hand-written values.
- If Viro lands in models.dev later, the next `generate_models.py` run picks it up
  automatically; only the catalog entry needs to exist first.
- If Viro lands in models.dev later, the next `generate_models.py` run picks it up
  automatically; only the catalog entry needs to exist first.