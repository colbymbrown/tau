"""Provider-neutral OpenAI-compatible local server backends.

One shared, parameterized implementation covers every local inference server
that exposes the standard OpenAI surface (vLLM, Strata, Unsloth Desktop, and
LM Studio). These servers own model loading and storage outside Tau, so the
backend surface is configure/refresh/status/use only: the host manages the
endpoint and optional credential, probes ``/v1/models``, and hands the server
to the normal OpenAI-compatible streaming transport.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import httpx

from tau_agent.types import JSONValue
from tau_ai.env import DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
from tau_ai.http import create_async_client
from tau_coding.credentials import CredentialStore
from tau_coding.extensions import (
    DynamicProvider,
    LocalAction,
    LocalBackend,
    LocalBackendStatus,
    LocalConfigField,
    LocalConfigureResult,
    LocalConfigureSpec,
    LocalDiagnostic,
    LocalModel,
    LocalOperationContext,
    LocalOperationResult,
    OpenAICompatibleTransport,
    ProviderAuthContext,
    ProviderModel,
    ProviderModelSnapshot,
    ProviderRefreshContext,
    ResolvedProviderAuth,
)
from tau_coding.paths import TauPaths

LOCAL_SERVERS_STATE_SCHEMA_VERSION = 1
LOCAL_SERVERS_CREDENTIAL_PREFIX = "local-servers:"

# (backend id, display name, default endpoint, api key environment fallback)
LOCAL_SERVERS: tuple[tuple[str, str, str, str], ...] = (
    ("vllm", "vLLM", "http://localhost:8000", "VLLM_API_KEY"),
    ("strata", "Strata", "http://localhost:8080", "STRATA_API_KEY"),
    ("unsloth-desktop", "Unsloth Desktop", "http://127.0.0.1:8000", "UNSLOTH_API_KEY"),
    ("lm-studio", "LM Studio", "http://localhost:1234", "LM_STUDIO_API_KEY"),
)

DEFAULT_LOCAL_SERVER_TIMEOUT_SECONDS = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS


class LocalServerError(RuntimeError):
    """Raised for a safe, user-actionable local server integration failure."""


class LocalServerEndpointError(ValueError):
    """Raised when an endpoint is not a safe HTTP base URL."""


class LocalServerStateError(RuntimeError):
    """Raised for an unreadable or unsupported local server state file."""


@dataclass(frozen=True, slots=True)
class LocalServerEndpoint:
    """The server root and OpenAI-compatible inference base for one endpoint."""

    server_root: str
    inference_base: str


def normalize_endpoint(value: str, label: str) -> LocalServerEndpoint:
    """Normalize a server root or OpenAI-compatible ``/v1`` URL safely."""
    if not isinstance(value, str) or not value.strip():
        raise LocalServerEndpointError(f"{label} endpoint cannot be empty")
    raw = value.strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise LocalServerEndpointError(f"{label} endpoint must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise LocalServerEndpointError(f"{label} endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise LocalServerEndpointError(f"{label} endpoint must not contain a query or fragment")
    if not parsed.netloc or parsed.hostname is None:
        raise LocalServerEndpointError(f"{label} endpoint must include a host")
    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    elif path.endswith("/v1"):
        path = path[: -len("/v1")].rstrip("/")
    root = urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))
    root = root.rstrip("/")
    if not root:
        raise LocalServerEndpointError(f"{label} endpoint must include a host")
    return LocalServerEndpoint(server_root=root, inference_base=f"{root}/v1")


@dataclass(frozen=True, slots=True)
class LocalServerState:
    """One persisted integration snapshot: endpoint and optional credential ref."""

    endpoint: str
    credential_ref: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.endpoint, str)
            or not self.endpoint
            or self.endpoint != self.endpoint.strip()
        ):
            raise LocalServerStateError("Saved endpoint must be a non-empty exact string")
        if self.credential_ref is not None and (
            not isinstance(self.credential_ref, str) or not self.credential_ref.strip()
        ):
            raise LocalServerStateError("Saved credential reference must be a string or None")

    def to_json(self) -> dict[str, object]:
        value: dict[str, object] = {"endpoint": self.endpoint}
        if self.credential_ref is not None:
            value["credential_ref"] = self.credential_ref
        return value


class LocalServerStateStore:
    """Locked and atomically replaced single-endpoint integration state."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        lock_path: Path | None = None,
        paths: TauPaths | None = None,
    ) -> None:
        resolved_paths = paths or TauPaths()
        self.path = path or resolved_paths.extension_state_dir / "local-servers.json"
        self.lock_path = lock_path or self.path.with_name(f"{self.path.name}.lock")

    def active(self) -> LocalServerState | None:
        with self._locked():
            return self._read_unlocked()

    def save(self, state: LocalServerState) -> str | None:
        """Publish one snapshot, returning the prior credential reference."""
        with self._locked():
            prior = self._read_unlocked()
            self._write_unlocked(state)
            return prior.credential_ref if prior is not None else None

    def clear(self) -> str | None:
        """Remove all integration settings, returning the credential reference."""
        with self._locked():
            prior = self._read_unlocked()
            if self.path.exists():
                self.path.unlink()
                _fsync_directory(self.path.parent)
            return prior.credential_ref if prior is not None else None

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.lock_path.parent != self.path.parent:
            self.lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # mkdir(mode=...) does not tighten an already-existing directory. The
        # state directory is user-private even when it was created earlier by
        # another Tau process.
        with suppress(OSError):
            self.path.parent.chmod(0o700)
        if self.lock_path.parent != self.path.parent:
            with suppress(OSError):
                self.lock_path.parent.chmod(0o700)
        try:
            with self.lock_path.open("a+b") as handle:
                self.lock_path.chmod(0o600)
                _lock(handle)
                try:
                    _remove_temporary_files(self.path)
                    yield
                finally:
                    _unlock(handle)
        except LocalServerStateError:
            raise
        except OSError as exc:
            raise LocalServerStateError(f"Could not access local server state: {exc}") from exc

    def _read_unlocked(self) -> LocalServerState | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LocalServerStateError(f"Could not read local server state: {exc}") from exc
        if not isinstance(payload, dict):
            raise LocalServerStateError("Local server state must be a JSON object")
        allowed = {"schema_version", "endpoint", "credential_ref"}
        if set(payload) - allowed:
            raise LocalServerStateError("Unknown field in local server state")
        if payload.get("schema_version") != LOCAL_SERVERS_STATE_SCHEMA_VERSION:
            raise LocalServerStateError("Unsupported local server state schema version")
        endpoint = payload.get("endpoint")
        credential_ref = payload.get("credential_ref")
        if not isinstance(endpoint, str) or not endpoint:
            raise LocalServerStateError("Malformed local server state")
        if not isinstance(credential_ref, (str, type(None))):
            raise LocalServerStateError("Malformed local server credential reference")
        if credential_ref is not None and not _valid_credential_ref(credential_ref):
            raise LocalServerStateError("Malformed local server credential reference")
        return LocalServerState(endpoint=endpoint, credential_ref=credential_ref)

    def _write_unlocked(self, state: LocalServerState) -> None:
        payload = {"schema_version": LOCAL_SERVERS_STATE_SCHEMA_VERSION, **state.to_json()}
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        temporary: Path | None = None
        fd = -1
        try:
            fd, raw = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            temporary = Path(raw)
            temporary.chmod(0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
            self.path.chmod(0o600)
            _fsync_directory(self.path.parent)
        except OSError as exc:
            raise LocalServerStateError(f"Could not write local server state: {exc}") from exc
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink()


class _LocalServerAuth:
    """Generation-referenced optional auth; secrets never enter the provider."""

    credential_ref: str | None
    env_var: str

    def __init__(self, credential_ref: str | None, env_var: str) -> None:
        self.credential_ref = credential_ref
        self.env_var = env_var

    async def resolve(self, context: ProviderAuthContext) -> ResolvedProviderAuth:
        if self.credential_ref:
            stored = context.credentials.get(self.credential_ref)
            if stored:
                return ResolvedProviderAuth(
                    api_key=stored,
                    source="stored credential",
                    omit_authorization_header=False,
                )
        environment_value = context.environment.get(self.env_var)
        if environment_value:
            return ResolvedProviderAuth(
                api_key=environment_value,
                source=f"environment variable {self.env_var}",
                omit_authorization_header=False,
            )
        return ResolvedProviderAuth()


class LocalServerService:
    """One named local server: provider construction and backend operations."""

    def __init__(
        self,
        *,
        backend_id: str,
        display_name: str,
        default_endpoint: str,
        api_key_env: str,
        state_store: LocalServerStateStore,
        credential_store: CredentialStore,
        environment: Mapping[str, str],
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_LOCAL_SERVER_TIMEOUT_SECONDS,
        register_provider: Callable[[DynamicProvider], None] | None = None,
        update_provider: Callable[[DynamicProvider], bool] | None = None,
        register_backend: Callable[[LocalBackend], None] | None = None,
    ) -> None:
        self.backend_id = backend_id
        self.display_name = display_name
        self.default_endpoint = default_endpoint
        self.api_key_env = api_key_env
        self.state_store = state_store
        self.credential_store = credential_store
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
            default_endpoint, display_name
        )
        self._last_models: tuple[ProviderModel, ...] = ()
        self._cleanup_orphans()

    @property
    def configured(self) -> bool:
        return self._active_state is not None

    @property
    def endpoint(self) -> LocalServerEndpoint:
        return self._endpoint

    def provider(self) -> DynamicProvider:
        return DynamicProvider(
            id=self.backend_id,
            display_name=self.display_name,
            models=self._last_models,
            default_model=self._last_models[0].id if len(self._last_models) == 1 else None,
            transport=OpenAICompatibleTransport(
                base_url=self._endpoint.inference_base,
                auth=_LocalServerAuth(
                    self._active_state.credential_ref if self._active_state else None,
                    self.api_key_env,
                ),
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
            id=self.backend_id,
            provider_id=self.backend_id,
            display_name=self.display_name,
            configure_spec=LocalConfigureSpec(
                fields=(
                    LocalConfigField(
                        "endpoint",
                        f"{self.display_name} endpoint",
                        "text",
                        required=True,
                        placeholder=f"{self.default_endpoint} or .../v1",
                    ),
                    LocalConfigField(
                        "api_key",
                        "API key (optional)",
                        "secret",
                        required=False,
                        placeholder="Leave empty for no authentication",
                    ),
                )
            ),
            configure=self.configure,
            status=self.status,
            refresh=self.refresh,
            reset=self.reset,
        )

    async def refresh_models(self, context: ProviderRefreshContext) -> ProviderModelSnapshot:
        if not context.allow_network:
            return ProviderModelSnapshot(
                models=tuple(context.cached_models),
                default_model=_cached_default(context.cached_models),
            )
        try:
            models = await self.discover(context.auth, signal=context.signal)
        except (httpx.HTTPError, LocalServerError):
            # Keep the last snapshot; a provider refresh must not break the
            # active runtime because the server happened to be down.
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
        """Commit endpoint and optional credential as one safe-state transaction."""
        del context
        try:
            endpoint = normalize_endpoint(values.get("endpoint", ""), self.display_name)
        except LocalServerEndpointError as exc:
            return LocalConfigureResult(
                diagnostics=(LocalDiagnostic(str(exc), "error", "configuration"),)
            )
        key = values.get("api_key", "").strip() or None
        credential_ref = (
            f"{LOCAL_SERVERS_CREDENTIAL_PREFIX}{self.backend_id}:{secrets.token_urlsafe(18)}"
            if key
            else None
        )
        if credential_ref is not None:
            assert key is not None
            try:
                self.credential_store.set(credential_ref, key)
            except Exception:
                return LocalConfigureResult(
                    message=f"{self.display_name} settings were not saved.",
                    diagnostics=(
                        LocalDiagnostic(
                            "The API key could not be stored; no settings were changed.",
                            "error",
                            "credentials",
                        ),
                    ),
                )
        state = LocalServerState(endpoint=endpoint.server_root, credential_ref=credential_ref)
        try:
            previous_ref = self.state_store.save(state)
        except LocalServerStateError as exc:
            orphaned = False
            if credential_ref is not None:
                try:
                    self.credential_store.delete(credential_ref)
                except Exception:
                    orphaned = True
            return LocalConfigureResult(
                message=f"{self.display_name} settings were not saved.",
                diagnostics=(
                    LocalDiagnostic(
                        str(exc),
                        "error",
                        "state",
                    ),
                ),
                credential_orphaned=orphaned,
            )
        if previous_ref is not None and previous_ref != credential_ref:
            with suppress(Exception):
                self.credential_store.delete(previous_ref)
        self._active_state = state
        self._endpoint = endpoint
        self._endpoint_error = None
        self._state_error = None
        self._publish_provider()
        discovery = await self._try_discover()
        if discovery is None:
            status = self._status(
                "unavailable",
                diagnostics=(
                    LocalDiagnostic(
                        f"Start your {self.display_name} server at the configured endpoint "
                        "and refresh.",
                        "warning",
                        "connection",
                    ),
                ),
                stale=True,
            )
            return LocalConfigureResult(
                committed=True,
                backend_status=status,
                message=f"{self.display_name} settings saved.",
                diagnostics=status.diagnostics,
            )
        return LocalConfigureResult(
            committed=True,
            backend_status=self._status_from_models(discovery),
            message=f"{self.display_name} settings saved.",
        )

    async def status(self, context: LocalOperationContext) -> LocalBackendStatus:
        del context
        if self._last_models:
            return self._status_from_models(self._last_models)
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
        """Probe exactly one effective endpoint; never scans ports or processes."""
        try:
            auth = await self._resolve_auth()
            models = await self.discover(auth, signal=context.signal)
        except asyncio.CancelledError:
            return LocalOperationResult(cancelled=True)
        except LocalServerError as exc:
            return LocalOperationResult(
                backend_status=self._cached_status(
                    diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),),
                    stale=True,
                ),
                diagnostics=(LocalDiagnostic(str(exc), "error", "connection"),),
            )
        self._last_models = models
        self._publish_provider()
        return LocalOperationResult(backend_status=self._status_from_models(models))

    async def reset(self, context: LocalOperationContext) -> LocalOperationResult:
        del context
        try:
            ref = self.state_store.clear()
        except LocalServerStateError as exc:
            return LocalOperationResult(diagnostics=(LocalDiagnostic(str(exc), "error", "state"),))
        if ref is not None:
            with suppress(Exception):
                self.credential_store.delete(ref)
        self._active_state = None
        self._last_models = ()
        self._state_error = None
        self._endpoint_error = None
        self._endpoint = normalize_endpoint(self.default_endpoint, self.display_name)
        self._publish_provider()
        return LocalOperationResult(
            backend_status=self._status(
                "unconfigured",
                diagnostics=(
                    LocalDiagnostic(
                        f"{self.display_name} integration settings were reset. Server "
                        "processes and model files are untouched.",
                        "info",
                        "configuration",
                    ),
                ),
            ),
            message=f"{self.display_name} integration settings were reset.",
        )

    async def discover(
        self,
        auth: ResolvedProviderAuth,
        *,
        signal: object | None = None,
    ) -> tuple[ProviderModel, ...]:
        """Probe /v1/models on the effective endpoint and parse safe model IDs."""
        del signal
        headers = _auth_headers(auth)
        owned_client = self.client is None
        client = self.client or create_async_client(timeout=self.timeout_seconds)
        try:
            response = await client.get(
                f"{self._endpoint.inference_base}/models",
                headers=headers,
            )
            if response.status_code >= 400:
                raise LocalServerError(
                    f"{self.display_name} returned HTTP {response.status_code} for /v1/models."
                )
            return parse_openai_models(response, self.display_name)
        except httpx.HTTPError as exc:
            raise LocalServerError(
                f"Could not reach {self.display_name} at {self._endpoint.inference_base}: {exc}"
            ) from exc
        finally:
            if owned_client:
                await client.aclose()

    def _resolve_effective_endpoint(
        self, active: LocalServerState | None
    ) -> LocalServerEndpoint | None:
        if active is not None:
            try:
                return normalize_endpoint(active.endpoint, self.display_name)
            except LocalServerEndpointError as exc:
                self._endpoint_error = str(exc)
                return None
        return None

    async def _resolve_auth(self) -> ResolvedProviderAuth:
        ref = self._active_state.credential_ref if self._active_state else None
        return await _LocalServerAuth(ref, self.api_key_env).resolve(
            ProviderAuthContext(credentials=self.credential_store, environment=self.environment)
        )

    async def _try_discover(self) -> tuple[ProviderModel, ...] | None:
        try:
            auth = await self._resolve_auth()
            return await self.discover(auth)
        except (LocalServerError, httpx.HTTPError):
            return None

    def _cleanup_orphans(self) -> None:
        prefix = f"{LOCAL_SERVERS_CREDENTIAL_PREFIX}{self.backend_id}:"
        referenced = self._active_state.credential_ref if self._active_state else None
        try:
            for name in self.credential_store.names(prefix=prefix):
                if name != referenced:
                    with suppress(Exception):
                        self.credential_store.delete(name)
        except Exception:
            pass

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

    def _status_from_models(self, models: tuple[ProviderModel, ...]) -> LocalBackendStatus:
        diagnostics: list[LocalDiagnostic] = []
        if not models:
            diagnostics.append(
                LocalDiagnostic(
                    "Server is ready but exposes no selectable models.", "warning", "models"
                )
            )
        selected = models[0].id if len(models) == 1 else None
        return self._status(
            "ready" if models else "unavailable",
            models=models,
            selected=selected,
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
        selected: str | None = None,
        diagnostics: tuple[LocalDiagnostic, ...] = (),
        stale: bool = False,
    ) -> LocalBackendStatus:
        actions: list[LocalAction] = ["configure", "refresh", "reset"]
        if selected is not None:
            actions.append("use")
        return LocalBackendStatus(
            state=state,  # type: ignore[arg-type]
            endpoint_display=self._endpoint.inference_base,
            authentication_source=self._authentication_source(),
            models=tuple(LocalModel(model.id, model.display_name or model.id) for model in models),
            selected_model=selected,
            actions=tuple(actions),
            diagnostics=diagnostics,
            stale=stale,
        )

    def _authentication_source(
        self,
    ) -> Literal["none", "environment", "stored credential"]:
        if self._active_state is not None and self._active_state.credential_ref:
            try:
                if self.credential_store.get(self._active_state.credential_ref):
                    return "stored credential"
            except Exception:
                pass
        if self.environment.get(self.api_key_env):
            return "environment"
        return "none"


def _auth_headers(auth: ResolvedProviderAuth) -> dict[str, str]:
    headers = dict(auth.headers)
    if auth.api_key is not None and not auth.omit_authorization_header:
        headers.setdefault("Authorization", f"Bearer {auth.api_key}")
    return headers


def parse_openai_models(response: httpx.Response, label: str) -> tuple[ProviderModel, ...]:
    """Parse a standard ``/v1/models`` payload into safe provider models."""
    try:
        payload = response.json()
    except ValueError as exc:
        raise LocalServerError(f"{label} /v1/models returned malformed JSON.") from exc
    if not isinstance(payload, Mapping):
        raise LocalServerError(f"{label} /v1/models returned malformed JSON.")
    data = payload.get("data")
    if not isinstance(data, list):
        raise LocalServerError(f"{label} /v1/models returned a malformed model list.")
    models: list[ProviderModel] = []
    ids: set[str] = set()
    for item in data:
        if not isinstance(item, Mapping):
            raise LocalServerError(f"{label} /v1/models returned a malformed model.")
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
            raise LocalServerError(f"{label} /v1/models returned a model without an exact id.")
        if model_id in ids:
            raise LocalServerError(f"{label} /v1/models returned duplicate model ids.")
        ids.add(model_id)
        display = item.get("name", item.get("display_name"))
        display_name = display if isinstance(display, str) and display.strip() else model_id
        models.append(
            ProviderModel(
                id=model_id,
                display_name=display_name,
                api="openai-completions",
                context_window=_reported_context_window(item),
                # No reasoning, cost, tool-support, or modality guesses.
                compat=_safe_model_compat(item),
            )
        )
    return tuple(models)


def _reported_context_window(item: Mapping[str, object]) -> int | None:
    # Accepted only when directly reported by a server. Tau never derives one
    # from a model family, max tokens, or a 128K convention.
    for key in ("context_window", "context_length"):
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _safe_model_compat(item: Mapping[str, object]) -> Mapping[str, JSONValue]:
    # Keep a tiny, non-secret diagnostic subset in memory. It is not copied to
    # the disk snapshot and is never interpreted as capability metadata.
    result: dict[str, JSONValue] = {}
    for key in ("object", "owned_by"):
        value = item.get(key)
        if isinstance(value, str) and value:
            result[key] = value
    return MappingProxyType(result)


def _cached_default(models: tuple[ProviderModel, ...]) -> str | None:
    return models[0].id if len(models) == 1 else None


def _valid_credential_ref(value: str) -> bool:
    if not value.startswith(LOCAL_SERVERS_CREDENTIAL_PREFIX):
        return False
    suffix = value.removeprefix(LOCAL_SERVERS_CREDENTIAL_PREFIX)
    # "{backend_id}:{token}" where the backend id is a conservative identifier
    # and the token uses secrets.token_urlsafe()'s alphabet. Reject path
    # separators and control characters so state can never name an unrelated
    # credential or become a path-like injection surface.
    if ":" not in suffix:
        return False
    backend_id, token = suffix.split(":", 1)
    return (
        bool(backend_id)
        and all(character.isalnum() or character in "-_" for character in backend_id)
        and bool(token)
        and all(character.isalnum() or character in "-_" for character in token)
    )


def _remove_temporary_files(path: Path) -> None:
    """Remove only this store's interrupted atomic-write artifacts."""
    pattern = f".{path.name}.*.tmp"
    for temporary in path.parent.glob(pattern):
        with suppress(OSError):
            temporary.unlink()


def _lock(handle: object) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
    except ImportError:
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
    except OSError as exc:
        raise LocalServerStateError(f"Could not lock local server state: {exc}") from exc


def _unlock(handle: object) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
    except ImportError:
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]


def _fsync_directory(path: Path) -> None:
    with suppress(OSError):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "DEFAULT_LOCAL_SERVER_TIMEOUT_SECONDS",
    "LOCAL_SERVERS",
    "LOCAL_SERVERS_CREDENTIAL_PREFIX",
    "LOCAL_SERVERS_STATE_SCHEMA_VERSION",
    "LocalServerEndpoint",
    "LocalServerEndpointError",
    "LocalServerError",
    "LocalServerService",
    "LocalServerState",
    "LocalServerStateError",
    "LocalServerStateStore",
    "normalize_endpoint",
    "parse_openai_models",
]
