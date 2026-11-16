"""Trusted built-in Ollama local backend.

Ollama splits its surface across two protocols: the standard OpenAI
``/v1`` endpoints for inference and provider discovery, and the native
``/api/*`` endpoints for installed/running state and model management. This
backend pairs both: the provider talks OpenAI-compatible transport, while the
local backend manages models with ``/api/tags``, ``/api/ps``, ``/api/pull``,
and keep-alive generation calls. Ollama needs no authentication, so the
integration never stores credentials.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import suppress

import httpx

from tau_ai.env import DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
from tau_ai.http import create_async_client
from tau_coding.extensions import (
    DynamicProvider,
    LocalAction,
    LocalBackend,
    LocalBackendStatus,
    LocalConfigField,
    LocalConfigureResult,
    LocalConfigureSpec,
    LocalConfirmationChoice,
    LocalConfirmationRequest,
    LocalDiagnostic,
    LocalModel,
    LocalOperationContext,
    LocalOperationResult,
    LocalProgress,
    NoAuth,
    OpenAICompatibleTransport,
    ProviderModel,
    ProviderModelSnapshot,
    ProviderRefreshContext,
    ResolvedProviderAuth,
)
from tau_coding.extensions.builtins.local_servers.service import (
    LocalServerEndpoint,
    LocalServerEndpointError,
    LocalServerState,
    LocalServerStateError,
    LocalServerStateStore,
    normalize_endpoint,
    parse_openai_models,
)

OLLAMA_BACKEND_ID = "ollama"
OLLAMA_PROVIDER_ID = "ollama"
OLLAMA_DISPLAY_NAME = "Ollama"
OLLAMA_DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
OLLAMA_STATE_WAIT_TIMEOUT_SECONDS = 60.0
OLLAMA_POLL_INTERVAL_SECONDS = 0.5
OLLAMA_PULL_TIMEOUT_SECONDS = 4 * 60 * 60.0
OLLAMA_LOAD_KEEP_ALIVE = "30m"


class OllamaError(RuntimeError):
    """Raised for a safe, user-actionable Ollama integration failure."""


class OllamaService:
    """Owns endpoint state, native discovery, and model management."""

    def __init__(
        self,
        *,
        state_store: LocalServerStateStore,
        environment: Mapping[str, str],
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
        register_provider: Callable[[DynamicProvider], None] | None = None,
        update_provider: Callable[[DynamicProvider], bool] | None = None,
        register_backend: Callable[[LocalBackend], None] | None = None,
    ) -> None:
        self.state_store = state_store
        self.environment = dict(environment or {})
        self.client = client
        self.timeout_seconds = timeout_seconds
        self._register_provider = register_provider
        self._update_provider = update_provider
        self._register_backend = register_backend
        self._state_error: str | None = None
        try:
            active = self.state_store.active()
        except LocalServerStateError as exc:
            active = None
            self._state_error = str(exc)
        self._active_state = active
        self._endpoint_error: str | None = None
        self._endpoint = self._resolve_effective_endpoint(active) or normalize_endpoint(
            OLLAMA_DEFAULT_ENDPOINT, OLLAMA_DISPLAY_NAME
        )
        self._last_models: tuple[ProviderModel, ...] = ()
        self._last_running: frozenset[str] = frozenset()

    @property
    def configured(self) -> bool:
        return self._active_state is not None or bool(
            self.environment.get(OLLAMA_HOST_ENV, "").strip()
        )

    @property
    def endpoint(self) -> LocalServerEndpoint:
        return self._endpoint

    def provider(self) -> DynamicProvider:
        return DynamicProvider(
            id=OLLAMA_PROVIDER_ID,
            display_name=OLLAMA_DISPLAY_NAME,
            models=self._last_models,
            default_model=self._last_models[0].id if len(self._last_models) == 1 else None,
            transport=OpenAICompatibleTransport(
                base_url=self._endpoint.inference_base,
                auth=NoAuth(),
                timeout_seconds=self.timeout_seconds,
                max_retries=2,
                max_retry_delay_seconds=1.0,
                client=self.client,
            ),
            refresh_models=self.refresh_models,
            stable_scoped_references=True,
        )

    def backend(self) -> LocalBackend:
        return LocalBackend(
            id=OLLAMA_BACKEND_ID,
            provider_id=OLLAMA_PROVIDER_ID,
            display_name=OLLAMA_DISPLAY_NAME,
            configure_spec=LocalConfigureSpec(
                fields=(
                    LocalConfigField(
                        "endpoint",
                        "Ollama endpoint",
                        "text",
                        required=True,
                        placeholder=f"{OLLAMA_DEFAULT_ENDPOINT} or .../v1",
                    ),
                )
            ),
            configure=self.configure,
            status=self.status,
            refresh=self.refresh,
            reset=self.reset,
            load_model=self.load_model,
            unload_model=self.unload_model,
            download_model=self.download_model,
        )

    async def refresh_models(self, context: ProviderRefreshContext) -> ProviderModelSnapshot:
        if not context.allow_network:
            return ProviderModelSnapshot(
                models=tuple(context.cached_models),
                default_model=_cached_default(context.cached_models),
            )
        try:
            models = await self._discover_openai_models(context.auth, signal=context.signal)
        except (httpx.HTTPError, OllamaError):
            return ProviderModelSnapshot(
                models=tuple(context.cached_models),
                default_model=_cached_default(context.cached_models),
            )
        self._last_models = models
        return ProviderModelSnapshot(
            models=models,
            default_model=models[0].id if len(models) == 1 else None,
        )

    async def configure(
        self,
        values: Mapping[str, str],
        context: LocalOperationContext,
    ) -> LocalConfigureResult:
        del context
        try:
            endpoint = normalize_endpoint(values.get("endpoint", ""), OLLAMA_DISPLAY_NAME)
        except LocalServerEndpointError as exc:
            return LocalConfigureResult(
                diagnostics=(LocalDiagnostic(str(exc), "error", "configuration"),)
            )
        try:
            self.state_store.save(LocalServerState(endpoint=endpoint.server_root))
        except LocalServerStateError as exc:
            return LocalConfigureResult(
                message="Ollama settings were not saved.",
                diagnostics=(LocalDiagnostic(str(exc), "error", "state"),),
            )
        self._active_state = LocalServerState(endpoint=endpoint.server_root)
        self._endpoint = endpoint
        self._endpoint_error = None
        self._state_error = None
        self._publish_provider()
        try:
            models, running = await self._native_state()
        except OllamaError:
            status = self._status(
                "unavailable",
                diagnostics=(
                    LocalDiagnostic(
                        "Start Ollama at the configured endpoint and refresh.",
                        "warning",
                        "connection",
                    ),
                ),
                stale=True,
            )
            return LocalConfigureResult(
                committed=True,
                backend_status=status,
                message="Ollama settings saved.",
                diagnostics=status.diagnostics,
            )
        self._last_models = models
        self._last_running = running
        return LocalConfigureResult(
            committed=True,
            backend_status=self._status_from_state(models, running),
            message="Ollama settings saved.",
        )

    async def status(self, context: LocalOperationContext) -> LocalBackendStatus:
        del context
        if self._last_models:
            return self._status_from_state(self._last_models, self._last_running)
        if not self.configured:
            if self._state_error or self._endpoint_error:
                return self._cached_status()
            return self._status(
                "unconfigured",
                diagnostics=(
                    LocalDiagnostic(
                        "Tau will check the offered default endpoint when this backend opens. "
                        "Use Configure for a different server URL.",
                        "info",
                        "configuration",
                    ),
                ),
            )
        return self._cached_status()

    async def refresh(self, context: LocalOperationContext) -> LocalOperationResult:
        """Probe /api/tags and /api/ps on exactly one effective endpoint."""
        try:
            models, running = await self._native_state(context.signal)
        except asyncio.CancelledError:
            return LocalOperationResult(cancelled=True)
        except OllamaError as exc:
            return LocalOperationResult(
                backend_status=self._cached_status(
                    diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),),
                    stale=True,
                ),
                diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),),
            )
        self._last_models = models
        self._last_running = running
        self._publish_provider()
        return LocalOperationResult(backend_status=self._status_from_state(models, running))

    async def reset(self, context: LocalOperationContext) -> LocalOperationResult:
        del context
        try:
            self.state_store.clear()
        except LocalServerStateError as exc:
            return LocalOperationResult(diagnostics=(LocalDiagnostic(str(exc), "error", "state"),))
        self._active_state = None
        self._last_models = ()
        self._last_running = frozenset()
        self._state_error = None
        self._endpoint_error = None
        self._endpoint = normalize_endpoint(OLLAMA_DEFAULT_ENDPOINT, OLLAMA_DISPLAY_NAME)
        self._publish_provider()
        return LocalOperationResult(
            backend_status=self._status(
                "unconfigured",
                diagnostics=(
                    LocalDiagnostic(
                        "Ollama integration settings were reset. Server processes and model "
                        "files are untouched.",
                        "info",
                        "configuration",
                    ),
                ),
            ),
            message="Ollama integration settings were reset.",
        )

    async def load_model(
        self, model_id: str, context: LocalOperationContext
    ) -> LocalOperationResult:
        """Warm one installed model into memory after explicit confirmation."""
        try:
            models, running = await self._native_state(context.signal)
        except asyncio.CancelledError:
            return LocalOperationResult(cancelled=True)
        except OllamaError as exc:
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),)
            )
        ids = {model.id for model in models}
        if model_id not in ids:
            return LocalOperationResult(
                diagnostics=(
                    LocalDiagnostic(f"Model is not installed: {model_id}", "warning", "models"),
                )
            )
        if model_id in running:
            return LocalOperationResult(
                backend_status=self._status_from_state(models, running),
                message=f"{model_id} is already loaded.",
            )
        if context.confirmation is None:
            return LocalOperationResult(
                backend_status=self._status_from_state(models, running),
                confirmation=LocalConfirmationRequest(
                    f"Load {model_id!r} into memory on the shared Ollama server? Loading can "
                    "consume substantial memory.",
                    (
                        LocalConfirmationChoice("load", "Load model"),
                        LocalConfirmationChoice("cancel", "Cancel"),
                    ),
                ),
            )
        if context.confirmation == "cancel":
            return LocalOperationResult(
                cancelled=True, backend_status=self._status_from_state(models, running)
            )
        if context.confirmation != "load":
            return LocalOperationResult(
                diagnostics=(
                    LocalDiagnostic("Loading requires explicit confirmation.", "warning", "models"),
                )
            )
        try:
            await self._generate(model_id, keep_alive=OLLAMA_LOAD_KEEP_ALIVE)
            running = await self._wait_for_state(model_id, target=True, signal=context.signal)
            models, _ = await self._native_state(context.signal)
        except asyncio.CancelledError:
            return LocalOperationResult(cancelled=True)
        except (httpx.HTTPError, OllamaError) as exc:
            return LocalOperationResult(
                backend_status=self._cached_status(
                    diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),),
                    stale=True,
                ),
                diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),),
            )
        self._last_models = models
        self._last_running = running
        self._publish_provider()
        return LocalOperationResult(
            backend_status=self._status_from_state(models, running),
            message=f"Loaded {model_id}.",
        )

    async def unload_model(
        self, model_id: str, context: LocalOperationContext
    ) -> LocalOperationResult:
        """Release one model from memory after explicit confirmation."""
        try:
            models, running = await self._native_state(context.signal)
        except asyncio.CancelledError:
            return LocalOperationResult(cancelled=True)
        except OllamaError as exc:
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),)
            )
        if model_id not in running:
            return LocalOperationResult(
                backend_status=self._status_from_state(models, running),
                message=f"{model_id} is not loaded.",
            )
        if context.confirmation is None:
            return LocalOperationResult(
                backend_status=self._status_from_state(models, running),
                confirmation=LocalConfirmationRequest(
                    f"Unload {model_id!r} from memory on the shared Ollama server?",
                    (
                        LocalConfirmationChoice("unload", "Unload model"),
                        LocalConfirmationChoice("cancel", "Keep model loaded"),
                    ),
                ),
            )
        if context.confirmation == "cancel":
            return LocalOperationResult(
                cancelled=True, backend_status=self._status_from_state(models, running)
            )
        if context.confirmation != "unload":
            return LocalOperationResult(
                diagnostics=(
                    LocalDiagnostic(
                        "Unloading requires explicit confirmation.", "warning", "models"
                    ),
                )
            )
        try:
            await self._generate(model_id, keep_alive=0)
            running = await self._wait_for_state(model_id, target=False, signal=context.signal)
            models, _ = await self._native_state(context.signal)
        except asyncio.CancelledError:
            return LocalOperationResult(cancelled=True)
        except (httpx.HTTPError, OllamaError) as exc:
            return LocalOperationResult(
                backend_status=self._cached_status(
                    diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),),
                    stale=True,
                ),
                diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),),
            )
        self._last_models = models
        self._last_running = running
        self._publish_provider()
        return LocalOperationResult(
            backend_status=self._status_from_state(models, running),
            message=f"Unloaded {model_id}.",
        )

    async def download_model(
        self, model_id: str, context: LocalOperationContext
    ) -> LocalOperationResult:
        """Pull one model from the Ollama registry with streamed progress."""
        model_id = model_id.strip()
        if not model_id:
            return LocalOperationResult(
                diagnostics=(
                    LocalDiagnostic("Enter a model name such as qwen3:8b.", "warning", "download"),
                )
            )
        try:
            installed, _ = await self._native_state(context.signal)
        except asyncio.CancelledError:
            return LocalOperationResult(cancelled=True)
        except OllamaError as exc:
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),)
            )
        if model_id in {model.id for model in installed}:
            return LocalOperationResult(message=f"{model_id} is already installed.")
        if context.confirmation is None:
            return LocalOperationResult(
                confirmation=LocalConfirmationRequest(
                    f"Download {model_id!r} on the shared Ollama server? The download continues "
                    "if this modal is closed.",
                    (
                        LocalConfirmationChoice("download", "Start download"),
                        LocalConfirmationChoice("cancel", "Cancel"),
                    ),
                ),
            )
        if context.confirmation == "cancel":
            return LocalOperationResult(cancelled=True)
        if context.confirmation != "download":
            return LocalOperationResult(
                diagnostics=(
                    LocalDiagnostic(
                        "Downloading requires explicit confirmation.", "warning", "download"
                    ),
                )
            )
        try:
            await self._pull(model_id, context)
        except asyncio.CancelledError:
            return LocalOperationResult(cancelled=True)
        except (httpx.HTTPError, OllamaError) as exc:
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic(str(exc), "error", "download"),)
            )
        models, running = await self._native_state(context.signal)
        self._last_models = models
        self._last_running = running
        self._publish_provider()
        return LocalOperationResult(
            backend_status=self._status_from_state(models, running),
            message=f"Downloaded {model_id}.",
        )

    async def _discover_openai_models(
        self,
        auth: ResolvedProviderAuth,
        *,
        signal: object | None = None,
    ) -> tuple[ProviderModel, ...]:
        """Probe the OpenAI-compatible /v1/models surface."""
        del auth, signal
        owned_client = self.client is None
        client = self.client or create_async_client(timeout=self.timeout_seconds)
        try:
            response = await client.get(f"{self._endpoint.inference_base}/models")
            if response.status_code >= 400:
                raise OllamaError(f"Ollama returned HTTP {response.status_code} for /v1/models.")
            return parse_openai_models(response, OLLAMA_DISPLAY_NAME)
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self._endpoint.inference_base}: {exc}"
            ) from exc
        finally:
            if owned_client:
                await client.aclose()

    async def _native_state(
        self,
        signal: object | None = None,
    ) -> tuple[tuple[ProviderModel, ...], frozenset[str]]:
        """Return installed models and the set of currently running model ids."""
        del signal
        owned_client = self.client is None
        client = self.client or create_async_client(timeout=self.timeout_seconds)
        try:
            tags_response, ps_response = await asyncio.gather(
                self._get(client, f"{self._endpoint.server_root}/api/tags"),
                self._get(client, f"{self._endpoint.server_root}/api/ps"),
            )
        finally:
            if owned_client:
                await client.aclose()
        installed = _installed_models(_json_payload(tags_response, "/api/tags"))
        running = _running_ids(_json_payload(ps_response, "/api/ps"))
        return installed, running

    async def _get(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self._endpoint.server_root}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise OllamaError(f"Ollama returned HTTP {response.status_code} for {url}.")
        return response

    async def _generate(self, model_id: str, *, keep_alive: int | str) -> None:
        """Run one empty generation; keep_alive=0 unloads, otherwise warms."""
        owned_client = self.client is None
        client = self.client or create_async_client(timeout=self.timeout_seconds)
        try:
            response = await client.post(
                f"{self._endpoint.server_root}/api/generate",
                json={"model": model_id, "prompt": "", "keep_alive": keep_alive, "stream": False},
            )
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self._endpoint.server_root}: {exc}"
            ) from exc
        finally:
            if owned_client:
                await client.aclose()
        if response.status_code >= 400:
            raise OllamaError(f"Ollama returned HTTP {response.status_code} for /api/generate.")

    async def _pull(self, model_id: str, context: LocalOperationContext) -> None:
        """Stream one /api/pull download and publish byte progress."""
        owned_client = self.client is None
        client = self.client or create_async_client(timeout=self.timeout_seconds)
        try:
            async with client.stream(
                "POST",
                f"{self._endpoint.server_root}/api/pull",
                json={"model": model_id, "stream": True},
            ) as response:
                if response.status_code >= 400:
                    raise OllamaError(f"Ollama returned HTTP {response.status_code} for /api/pull.")
                async for line in response.aiter_lines():
                    if context.cancelled:
                        raise asyncio.CancelledError
                    payload = _json_line(line)
                    status = payload.get("status")
                    if status == "error":
                        raise OllamaError(
                            payload.get("error") or "Ollama could not download the model."
                        )
                    if status == "success":
                        context.report_progress(LocalProgress(f"Downloaded {model_id}.", done=True))
                        return
                    if status == "downloading":
                        total = payload.get("total")
                        completed = payload.get("completed")
                        if isinstance(total, int) and isinstance(completed, int) and total > 0:
                            context.report_progress(_pull_progress(model_id, completed, total))
                        else:
                            context.report_progress(LocalProgress(f"Downloading {model_id}…"))
                    else:
                        context.report_progress(LocalProgress(f"Downloading {model_id}: {status}…"))
            raise OllamaError("Ollama /api/pull ended without a success status.")
        finally:
            if owned_client:
                await client.aclose()

    async def _wait_for_state(
        self,
        model_id: str,
        *,
        target: bool,
        signal: object | None = None,
    ) -> frozenset[str]:
        """Poll /api/ps until the model reaches the target running state."""
        deadline = asyncio.get_running_loop().time() + OLLAMA_STATE_WAIT_TIMEOUT_SECONDS
        running: frozenset[str] = frozenset()
        while True:
            with suppress(OllamaError):
                _, running = await self._native_state(signal)
            if (model_id in running) == target:
                return running
            if asyncio.get_running_loop().time() >= deadline:
                return running
            await asyncio.sleep(OLLAMA_POLL_INTERVAL_SECONDS)

    def _resolve_effective_endpoint(
        self,
        active: LocalServerState | None,
    ) -> LocalServerEndpoint | None:
        if active is not None:
            try:
                return normalize_endpoint(active.endpoint, OLLAMA_DISPLAY_NAME)
            except LocalServerEndpointError as exc:
                self._endpoint_error = str(exc)
                return None
        host = self.environment.get(OLLAMA_HOST_ENV, "").strip()
        if host:
            try:
                return normalize_endpoint(host, OLLAMA_DISPLAY_NAME)
            except LocalServerEndpointError as exc:
                self._endpoint_error = str(exc)
                return None
        return None

    def _publish_provider(self) -> None:
        provider = self.provider()
        updated = self._update_provider(provider) if self._update_provider is not None else False
        if not updated and self._register_provider is not None:
            self._register_provider(provider)
        if not updated and self._register_backend is not None:
            # Initial setup and a source replacement need a paired backend
            # registration. Snapshot updates use ``update_provider`` above so
            # an in-flight local operation keeps its source-layer token.
            self._register_backend(self.backend())

    def _status_from_state(
        self,
        models: tuple[ProviderModel, ...],
        running: frozenset[str],
    ) -> LocalBackendStatus:
        diagnostics: list[LocalDiagnostic] = []
        if not models:
            diagnostics.append(
                LocalDiagnostic(
                    "Ollama is ready but has no installed models. Download one or run "
                    "`ollama pull <name>` in a terminal.",
                    "warning",
                    "models",
                )
            )
        selected = next((model.id for model in models if model.id in running), None)
        actions: list[LocalAction] = [
            "configure",
            "refresh",
            "reset",
            "load_model",
            "unload_model",
            "download_model",
        ]
        if selected is not None:
            actions.append("use")
        return LocalBackendStatus(
            state="ready",
            endpoint_display=self._endpoint.inference_base,
            authentication_source="none",
            models=tuple(
                LocalModel(
                    model.id,
                    model.display_name or model.id,
                    "running" if model.id in running else "installed",
                )
                for model in models
            ),
            selected_model=selected,
            actions=tuple(actions),
            diagnostics=tuple(diagnostics),
        )

    def _cached_status(
        self,
        *,
        diagnostics: tuple[LocalDiagnostic, ...] = (),
        stale: bool = False,
    ) -> LocalBackendStatus:
        if self._state_error:
            diagnostics = (*diagnostics, LocalDiagnostic(self._state_error, "warning", "state"))
        if self._endpoint_error:
            diagnostics = (*diagnostics, LocalDiagnostic(self._endpoint_error, "error", "endpoint"))
        models = self._last_models
        if models:
            return self._status_from_state(models, self._last_running)
        return self._status(
            "unavailable",
            models=(),
            diagnostics=diagnostics,
            stale=stale,
        )

    def _status(
        self,
        state: str,
        *,
        models: tuple[ProviderModel, ...] = (),
        diagnostics: tuple[LocalDiagnostic, ...] = (),
        stale: bool = False,
    ) -> LocalBackendStatus:
        return LocalBackendStatus(
            state=state,  # type: ignore[arg-type]
            endpoint_display=self._endpoint.inference_base,
            authentication_source="none",
            models=tuple(LocalModel(model.id, model.display_name or model.id) for model in models),
            diagnostics=diagnostics,
            stale=stale,
        )


def _installed_models(payload: Mapping[str, object]) -> tuple[ProviderModel, ...]:
    models = payload.get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama /api/tags returned a malformed model list.")
    result: list[ProviderModel] = []
    seen: set[str] = set()
    for item in models:
        if not isinstance(item, Mapping):
            raise OllamaError("Ollama /api/tags returned a malformed model.")
        name = item.get("name")
        if not isinstance(name, str) or not name or name != name.strip():
            raise OllamaError("Ollama /api/tags returned a model without an exact name.")
        if name in seen:
            raise OllamaError("Ollama /api/tags returned duplicate model names.")
        seen.add(name)
        result.append(ProviderModel(id=name, display_name=name, api="openai-completions"))
    return tuple(result)


def _running_ids(payload: Mapping[str, object]) -> frozenset[str]:
    models = payload.get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama /api/ps returned a malformed model list.")
    names = {item.get("name") for item in models if isinstance(item, Mapping)}
    return frozenset(name for name in names if isinstance(name, str) and name)


def _json_payload(response: httpx.Response, endpoint: str) -> Mapping[str, object]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise OllamaError(f"Ollama {endpoint} returned malformed JSON.") from exc
    if not isinstance(payload, Mapping):
        raise OllamaError(f"Ollama {endpoint} returned malformed JSON.")
    return payload


def _json_line(line: str) -> dict[str, object]:
    try:
        payload = _json_object(line)
    except ValueError as exc:
        raise OllamaError("Ollama /api/pull returned a malformed progress line.") from exc
    return payload


def _json_object(line: str) -> dict[str, object]:
    import json

    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Ollama /api/pull progress is not an object")
    return value


def _pull_progress(model_id: str, completed: int, total: int) -> LocalProgress:
    done = min(completed, total)
    remaining = total - done
    return LocalProgress(
        f"Downloading {model_id}… {_format_bytes(done)} / {_format_bytes(total)} "
        f"({done / total:.1%}; {_format_bytes(remaining)} remaining)",
        fraction=done / total,
    )


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"  # pragma: no cover - defensive


def _cached_default(models: tuple[ProviderModel, ...]) -> str | None:
    return models[0].id if len(models) == 1 else None


__all__ = [
    "OLLAMA_BACKEND_ID",
    "OLLAMA_DEFAULT_ENDPOINT",
    "OLLAMA_DISPLAY_NAME",
    "OLLAMA_HOST_ENV",
    "OLLAMA_LOAD_KEEP_ALIVE",
    "OLLAMA_POLL_INTERVAL_SECONDS",
    "OLLAMA_PROVIDER_ID",
    "OLLAMA_PULL_TIMEOUT_SECONDS",
    "OLLAMA_STATE_WAIT_TIMEOUT_SECONDS",
    "OllamaError",
    "OllamaService",
    "_format_bytes",
    "_installed_models",
    "_pull_progress",
    "_running_ids",
]
