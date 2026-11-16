# OpenAI-compatible local server backends and Ollama

## What changed

`/local` now lists **six** backends alongside the default: **vLLM**, **Strata**,
**Unsloth Desktop**, and **LM Studio** (Phase A) plus **Ollama** (Phase B).

- `service.py` — `LocalServerService`, the endpoint/credential state store,
  endpoint normalization, `/v1/models` discovery, and the generic
  configure/status/refresh/reset operations.
- `__init__.py` — `setup()` registers one `DynamicProvider` + `LocalBackend`
  pair per named server.
- `built_in_extensions.py` — one new trusted `BuiltInExtension` entry
  (`"local-servers"`), loaded like llama.cpp, before filesystem extensions.

Backend identity is a data table:

```python
LOCAL_SERVERS = (
    ("vllm", "vLLM", "http://localhost:8000", "VLLM_API_KEY"),
    ("strata", "Strata", "http://localhost:8080", "STRATA_API_KEY"),
    ("unsloth-desktop", "Unsloth Desktop", "http://127.0.0.1:8000", "UNSLOTH_API_KEY"),
    ("lm-studio", "LM Studio", "http://localhost:1234", "LM_STUDIO_API_KEY"),
)
```

## Why it exists

These servers all expose the standard OpenAI `/v1` surface (chat completions,
`/v1/models`) but own model loading and storage themselves — vLLM models are
fixed at server launch, Strata and Unsloth Desktop manage models in their apps,
LM Studio in its library. Tau already supported *talking* to any of them
through the durable `openai-compatible` provider config; what was missing was
the `/local` management flow: configure the endpoint once, probe status, list
served models, and switch to the server with one keypress.

The generic host needed **zero changes**. `LocalBackendRegistry`, the
operation runner, and the TUI screens were already provider-neutral (validated
in Phase 6 by the Ollama adapter spike). Adding a backend is purely an
extension-side registration.

## How it maps to Pi's design

Pi has no equivalent `/local` flow; this is Tau-native. The design reuses the
same contracts as llama.cpp:

- `DynamicProvider` with an `OpenAICompatibleTransport` — inference goes
  through the normal streaming transport, so the agent loop is unchanged.
- `LocalBackend` with `configure`/`status`/`refresh`/`reset` — the host owns
  fields, confirmation, cancellation, and rendering.
- `LocalBackendStatus.actions` drives the generic action menu. These servers
  only report `configure`, `refresh`, `reset`, plus `use` when exactly one
  model is served (the TUI's "Use model" path needs a definite selection).
  No `load_model`/`download_model`/`search_models` — none of these servers
  expose a model-management HTTP API.

One deliberate contrast with llama.cpp: no per-server `doctor`, no model
snapshot persistence, no multi-endpoint state. The state store persists only
`endpoint` + optional `credential_ref` (single entity, locked and atomically
replaced, mirroring llama.cpp's store). API keys live in Tau's credential
store; the transport falls back to a per-backend environment variable
(`VLLM_API_KEY`, `STRATA_API_KEY`, `UNSLOTH_API_KEY`, `LM_STUDIO_API_KEY`).

## Security and lifecycle decisions

- Credential references are namespaced (`local-servers:{backend}:{token}`),
  validated against a conservative alphabet on state read, and cleaned up when
  replaced or reset. Orphaned credentials are deleted at service construction.
- `/local` probing is opt-in and endpoint-exact: `refresh` probes only the
  saved endpoint or the offered default. It never scans ports or processes.
- Configure is transactional: a failed state write deletes a just-stored
  credential, and the prior active endpoint remains active.
- Provider snapshot failures keep the last snapshot; a down server never
  breaks the active runtime.

## How to use

1. Start the server (e.g. `vllm serve ...`, launch Strata / Unsloth Desktop /
   LM Studio).
2. Open `/local` in the TUI, pick the backend, enter the endpoint (and API key
   if the server requires one), then refresh.
3. Select a served model and choose **Use model** — the session switches to
   the local provider. Or use `--provider vllm --model <id>` in print mode.

## How to test

```bash
uv run pytest tests/test_local_servers.py -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The test file covers endpoint normalization, registration through the full
`ExtensionRuntime` load path, registry refresh, configure transactions,
credential replacement and reset, environment auth, offline provider refresh,
and state-store roundtrip.

## Skipped deliberately

- **LM Studio model download/load** — its developer REST API advertises model
  management; that is Phase C of the plan, deferred. Phase A covers the 90%
  case (status + use).
- **HF model search/download** — server-side downloads are llama.cpp-router
  specific; these servers have no such API.

# Ollama backend (Phase B)

Ollama gets a dedicated full-management backend under
`src/tau_coding/extensions/builtins/ollama/`. It pairs the same two protocols
Ollama itself exposes: the OpenAI-compatible `/v1` surface for inference and
provider discovery, and the native `/api/*` surface for management.

- `OllamaService.provider()` — `OpenAICompatibleTransport` at `{endpoint}/v1`
  with `NoAuth` (Ollama needs no key; the Phase 6 spike confirmed this).
- `status`/`refresh` — `/api/tags` (installed) + `/api/ps` (running) give a
  richer snapshot than `/v1/models` alone: models are marked
  `running`/`installed`, and the selected model is the first running one.
- `download_model` — `POST /api/pull` with `stream: true`, parsing the NDJSON
  progress lines into byte-progress `LocalProgress` events (same host-rendered
  shape llama.cpp's router downloads use). Requires explicit confirmation; the
  pull continues server-side if the modal closes.
- `load_model` — warms a model into memory with an empty generation
  (`keep_alive: "30m"`), then polls `/api/ps` until it is running.
- `unload_model` — `keep_alive: 0`, then polls until it leaves `/api/ps`.
- `configure` — endpoint only; no credential fields. Endpoint resolution is
  saved state → `OLLAMA_HOST` environment → `http://127.0.0.1:11434`.
- `reset` — clears saved state; never touches server processes or model files.

Reuse: the Ollama builtin imports `LocalServerStateStore`,
`LocalServerState`, `normalize_endpoint`, and `parse_openai_models` from the
`local_servers` module (the generic `/v1/models` parser was made public for
this). No duplicated persistence code.

Two deliberate omissions, both because Ollama's registry is not Hugging Face's
and `LocalAction` has no delete operation: no `search_models` (HF search
would return mostly non-pullable IDs) and no model deletion.

## How to test

```bash
uv run pytest tests/test_local_servers.py tests/test_ollama_extension.py -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```