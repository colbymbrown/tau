# GreenPT (EU/US) and Viro AI providers with runtime model discovery

## What was added

Three built-in OpenAI-compatible providers:

| name | base URL | models | default |
|---|---|---|---|
| `greenpt` | `https://api.greenpt.ai/v1` (EU) | 41 static ids (EU full catalog) | `deepseek-v4.1-flash` |
| `greenpt-us` | `https://api.us.greenpt.ai/v1` | 4 ids (US catalog per docs) | `glm-5.3` |
| `viro` | `https://api.viro.app/v1` | 20 chat ids (`viro/*`), from `console.viro.app/models` | `viro/deepseek-v4.1-flash` |

All three set `model_discovery = true`, which enables the new **runtime model
discovery** capability: the provider's own `GET /v1/models` is probed at session
activation and the discovered inventory overlays the static catalog.

## Why it exists

GreenPT and Viro serve fast-churning open-weight catalogs (dated snapshots like
`deepseek-v4.1-flash-1017`, region-dependent availability, deprecations). A
static `catalog.toml` list goes stale within weeks. Tau already had the discovery
pipeline for Codex (`ModelCatalogProvider.discover_models()` → session publish →
model-picker overlay); this change generalizes it to any OpenAI-compatible
provider that opts in.

## How it maps to Pi's design

Pi/Tau keep provider *transport* (base URL, auth, API dialect) application-owned
while model *inventory* can come from an authoritative runtime source. Codex was
the first such source (account-scoped); GreenPT/Viro are the first provider-
scoped `/v1/models` sources. Discovery is per catalog entry, so `greenpt` and
`greenpt-us` probe different hosts — honoring GreenPT's documented "one list per
host" rule without any special-casing.

## Implementation

- `src/tau_ai/openai_compatible.py`: `OpenAICompatibleProvider.discover_models()`
  — `GET {base_url}/models` with the same auth resolution as streaming
  (credential resolver may override key/headers/base URL); parses the standard
  `{"data": [{"id": ...}]}` payload; raises on HTTP error / malformed / empty so
  the session falls back to the static catalog.
- `ProviderCatalogEntry.model_discovery` / `_CatalogProvider.model_discovery` /
  `OpenAICompatibleProviderConfig.model_discovery` — the opt-in flag, threaded
  catalog → durable config → session. User `catalog.toml` can enable discovery
  for any custom openai-compatible provider.
- `src/tau_coding/session.py`:
  - `_model_discovery_enabled()` gate in `_refresh_runtime_model_limits` —
    codex always, others only when the flag is set.
  - `_provider_with_runtime_model_catalog` + `_runtime_provider_model_metadata`
    relaxed from codex-only to `OpenAICodexProviderConfig |
    OpenAICompatibleProviderConfig` — the merge keeps static metadata
    (context windows, costs, thinking maps) for ids the static catalog knows.
- `src/tau_coding/data/catalog.toml`: the three entries. `greenpt` (EU) is also
  in models.dev, so its context/cost metadata arrives via the usual
  `refresh_models_dev_catalog` runtime refresh; `greenpt-us` and `viro` ship
  hand-written `context_windows` only.

## How to test / use

- `uv run pytest tests/test_provider_discovery.py` — adapter, merge, gate, catalog.
- `/login` → API key → "GreenPT" / "GreenPT US" / "Viro AI" (automatically listed).
- With a real key: `uv run tau` → `/model` on `greenpt` shows the live
  `/v1/models` inventory; `TAU_OFFLINE=1` or a bad key falls back to the static
  list with the error recorded (`_model_catalog_discovery_errors`).

## Deliberate simplifications (ponytail)

- `refresh_model_catalogs` (the manual model-picker refresh) still refreshes
  codex + models.dev only; discovery-enabled openai-compatible providers are
  refreshed lazily on activation. Generalize the loop when a long-running
  session needs it.
- Viro cost metadata is not shipped: the console's USD prices carry a 20% promo
  and churn; discovery is the source of truth and Tau renders unknown cost
  gracefully.
- The bundled `models-dev-catalog.json` snapshot was **not** regenerated in this
  change — models.dev moved ~30 days of unrelated provider churn; regenerating
  belongs in its own routine commit, and the runtime refresh covers greenpt
  enrichment either way.