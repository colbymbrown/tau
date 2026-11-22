# Local server login templates (keyless OpenAI-compatible providers)

## What was added

`/login` now offers a fourth method, **Local server — keyless OpenAI-compatible**.
Choosing it shows a short preset list:

| Template | Provider id | Default endpoint |
| --- | --- | --- |
| Ollama | `ollama` | `http://127.0.0.1:11434/v1` |
| LM Studio | `lm-studio` | `http://127.0.0.1:1234/v1` |
| vLLM | `vllm` | `http://127.0.0.1:8000/v1` |
| Unsloth | `unsloth` | `http://127.0.0.1:8000/v1` |
| llama.cpp server | `llama-server` | `http://127.0.0.1:8080/v1` |

Each preset opens the existing OpenAI-compatible provider form pre-filled with
the endpoint and an API-key variable. Two things differ from the custom-provider
form:

- The API key is **optional**. With no key, Tau sends **no `Authorization`
  header**. A saved key is still used when present.
- The model list is **optional**. Leaving it blank probes `{base_url}/models`
  on save, seeds the provider with the discovered IDs, and selects the first.

## Why it exists

Earlier, `/local` (the trusted built-in local-backend surface) was extended with
managed vLLM/LM Studio/Strata/Unsloth/Ollama backends. That was reverted: those
engines are ordinary keyless OpenAI-compatible endpoints, and Tau already had a
custom-provider path for exactly that. Duplicating them as managed backends
added a large trusted surface for no new capability.

The missing convenience was discovery and defaults, not management. This change
adds those at the right layer — the provider catalog/login flow — so the engines
stay ordinary OpenAI-compatible providers.

## How it maps to Pi's design

Pi keeps provider *transport* (base URL, auth, API dialect) application-owned
while model *inventory* may come from a runtime source. This change reuses that:
it is a login template over the existing `openai-compatible` provider, plus the
existing `model_discovery` pipeline (`OpenAICompatibleProvider.discover_models`
→ `GET {base_url}/models`). No new provider adapter, registry, or TUI
management surface. It is Tau-native, not a Pi port.

## Implementation

- `src/tau_coding/local_server_presets.py` — `LocalServerPreset` and the
  `LOCAL_SERVER_PRESETS` table. Pure data.
- `require_api_key` — a new opt-out flag threaded like `model_discovery`:
  `ProviderCatalogEntry.require_api_key` (default `True`) →
  `OpenAICompatibleProviderConfig.require_api_key` → auth resolution. When
  `False` and no key resolves, `openai_compatible_config_from_provider` returns
  `api_key=""` and `omit_authorization_header=True`.
- `catalog_loader.py` — parse/serialize the flag. **Bug fix:** the user-catalog
  TOML writer did not emit `model_discovery` either, so discovery enabled for a
  user provider was silently dropped on save; both flags are now written.
- `tui/app.py` — the login method item, `LocalServerPresetScreen`, a `preset`
  parameter on `CustomProviderLoginScreen` (pre-fill + local mode), and an
  async `_discover_models` worker that probes `/v1/models` on submit.

## Security and lifecycle

- The templates never install, start, stop, or scan a server. `discover_models`
  only probes the exact endpoint the user submitted.
- No key means no `Authorization` header; secret handling is otherwise the same
  as any inline OpenAI-compatible provider (credential store or env fallback).
- Discovery runs on explicit submit, not at startup, so a down server cannot
  slow completion or break an unrelated session.

## How to use / test

```text
/login → Local server → Ollama → (leave models + key blank) → Enter
```

```bash
uv run pytest tests/test_local_server_providers.py tests/test_tui_app.py -q
uv run ruff check . && uv run mypy
```

## Skipped deliberately (ponytail)

- **No per-engine picker rows.** One method entry plus a preset submenu keeps the
  main provider list free of local servers most users will never run.
- **No background discovery or model deletion.** Ollama's native `/api` surface
  could support richer management; that is the reverted `/local` design, and is
  deliberately out of scope here.
- **`llama.cpp server` is a preset, not the `/local` backend.** Router
  management (download/load/unload/Doctor) stays with `/local`; the preset is
  only for a plain OpenAI-compatible `llama-server`.
