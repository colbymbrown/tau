"""Deterministic tests for the trusted OpenAI-compatible local server backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import httpx
import pytest

from tau_agent.harness import SimpleCancellationToken
from tau_coding.credentials import FileCredentialStore
from tau_coding.extensions import DynamicProviderRegistry, ExtensionRuntime, LocalBackendRegistry
from tau_coding.extensions.builtins.local_servers import (
    LOCAL_SERVERS,
    LocalServerService,
    LocalServerState,
    LocalServerStateStore,
    normalize_endpoint,
)
from tau_coding.extensions.providers import ProviderRefreshContext, ResolvedProviderAuth
from tau_coding.local_backends import LocalOperationContext, LocalProgress
from tau_coding.paths import TauPaths
from tau_coding.resources import TauResourcePaths

pytestmark = pytest.mark.anyio

SERVER = "http://local.test:8000"
VLLM_MODELS = {
    "object": "list",
    "data": [
        {
            "id": "deepseek-v3:fp8",
            "object": "model",
            "owned_by": "local",
            "context_length": 131072,
        },
        {"id": "qwen3:14b", "object": "model", "owned_by": "local"},
    ],
}


def _context(
    action: str = "refresh",
    *,
    confirmation: str | None = None,
    progress: Callable[[LocalProgress], None] | None = None,
) -> LocalOperationContext:
    return LocalOperationContext(
        signal=SimpleCancellationToken(),
        action=action,  # type: ignore[arg-type]
        generation_id="test-generation",
        backend_id="vllm",
        source_id="built-in:local-servers",
        confirmation=confirmation,
        _is_current=lambda: True,
        _progress=progress or (lambda _: None),
    )


def _service(
    tmp_path: Path,
    *,
    backend_id: str = "vllm",
    display_name: str = "vLLM",
    default_endpoint: str = SERVER,
    api_key_env: str = "VLLM_API_KEY",
    client: httpx.AsyncClient | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[LocalServerService, LocalServerStateStore, FileCredentialStore]:
    paths = TauPaths(home=tmp_path / "tau", agents_home=tmp_path / "agents")
    state_store = LocalServerStateStore(path=paths.extension_state_dir / f"{backend_id}.json")
    credential_store = FileCredentialStore(paths.home / "credentials.json")
    service = LocalServerService(
        backend_id=backend_id,
        display_name=display_name,
        default_endpoint=default_endpoint,
        api_key_env=api_key_env,
        state_store=state_store,
        credential_store=credential_store,
        environment=environment or {},
        client=client,
    )
    return service, state_store, credential_store


def _client() -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def dispatch(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=VLLM_MODELS)
        raise AssertionError(f"unexpected local server request: {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(dispatch)), requests


def _runtime(tmp_path: Path) -> tuple[ExtensionRuntime, TauPaths, FileCredentialStore]:
    paths = TauPaths(home=tmp_path / "tau", agents_home=tmp_path / "agents")
    credentials = FileCredentialStore(paths.home / "credentials.json")
    runtime = ExtensionRuntime(
        paths=paths,
        credentials=credentials,
        built_in_credentials=credentials,
        environment={},
        built_in_http_client=None,
    )
    runtime.load(
        TauResourcePaths(
            root=paths.home,
            agents_root=paths.agents_home,
            paths=paths,
            cwd=tmp_path / "project",
        ),
        include_resource_dirs=False,
        include_user_dir=False,
    )
    return runtime, paths, credentials


@pytest.mark.parametrize(
    ("value", "root", "inference"),
    [
        ("http://LOCALHOST:8000", "http://LOCALHOST:8000", "http://LOCALHOST:8000/v1"),
        ("http://localhost:8000/", "http://localhost:8000", "http://localhost:8000/v1"),
        ("http://localhost:8000/v1", "http://localhost:8000", "http://localhost:8000/v1"),
        (
            "http://localhost:8000/api/v1/",
            "http://localhost:8000/api",
            "http://localhost:8000/api/v1",
        ),
    ],
)
def test_endpoint_normalization(value: str, root: str, inference: str) -> None:
    endpoint = normalize_endpoint(value, "vLLM")
    assert endpoint.server_root == root
    assert endpoint.inference_base == inference


@pytest.mark.parametrize(
    "value",
    ["", "localhost:8000", "ftp://localhost", "http://user:pass@localhost", "http://localhost?a=1"],
)
def test_endpoint_normalization_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_endpoint(value, "vLLM")


async def test_builtin_registers_each_named_backend_and_provider(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    registry = runtime.local_backend_registry
    backends = {view.backend.id: view for view in registry.effective_backends()}
    assert set(backends) == {
        "llama.cpp",
        "vllm",
        "strata",
        "unsloth-desktop",
        "lm-studio",
        "ollama",
    }
    # llama.cpp keeps the recommended marking; the generic servers do not.
    assert backends["llama.cpp"].recommended
    assert not backends["vllm"].recommended
    for backend_id, _, _, _ in LOCAL_SERVERS:
        assert backends[backend_id].use_available
        assert runtime.provider_registry.effective(backend_id) is not None
    await runtime.aclose()


async def test_refresh_publishes_provider_and_ready_status(tmp_path: Path) -> None:
    client, requests = _client()
    service, _, _ = _service(tmp_path, client=client)
    providers = DynamicProviderRegistry(generation_id="generation")
    backends = LocalBackendRegistry(providers, generation_id="generation")
    providers.register("built-in:local-servers", service.provider())
    backends.register("built-in:local-servers", service.backend())

    provider_result = await providers.refresh("vllm")
    backend_result = await backends.refresh("vllm")

    assert provider_result.status == "published"
    assert provider_result.provider is not None
    assert [model.id for model in provider_result.provider.models] == [
        "deepseek-v3:fp8",
        "qwen3:14b",
    ]
    status = backend_result.backend_status
    assert status is not None
    assert status.state == "ready"
    assert status.endpoint_display == f"{SERVER}/v1"
    assert [model.id for model in status.models] == ["deepseek-v3:fp8", "qwen3:14b"]
    assert status.actions == ("configure", "refresh", "reset")
    assert [request.url.path for request in requests] == ["/v1/models", "/v1/models"]
    await backends.aclose()
    await providers.aclose()
    await client.aclose()


async def test_single_model_enables_use_action(tmp_path: Path) -> None:
    async def dispatch(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "code:latest", "object": "model"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, _, _ = _service(tmp_path, client=client)
    status = await service.refresh(_context())
    assert status.backend_status is not None
    assert status.backend_status.selected_model == "code:latest"
    assert "use" in status.backend_status.actions
    await client.aclose()


async def test_unconfigured_status_is_informational(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    status = await service.status(_context())
    assert status.state == "unconfigured"
    assert any(diagnostic.stage == "configuration" for diagnostic in status.diagnostics)
    assert status.actions == ("configure", "refresh", "reset")


async def test_refresh_failure_is_stale_and_reports_connection_error(tmp_path: Path) -> None:
    async def dispatch(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, _, _ = _service(tmp_path, client=client)
    result = await service.refresh(_context())
    assert result.backend_status is not None
    assert result.backend_status.stale
    assert any(diagnostic.severity == "error" for diagnostic in result.backend_status.diagnostics)
    await client.aclose()


async def test_configure_commits_endpoint_and_credential(tmp_path: Path) -> None:
    client, requests = _client()
    service, state_store, credential_store = _service(tmp_path, client=client)
    result = await service.configure(
        {"endpoint": "http://local.test:9000/v1", "api_key": "secret-key"}, _context("configure")
    )
    assert result.committed
    assert result.backend_status is not None
    assert result.backend_status.state == "ready"
    active = state_store.active()
    assert active is not None
    assert active.endpoint == "http://local.test:9000"
    assert active.credential_ref is not None
    assert credential_store.get(active.credential_ref) == "secret-key"
    assert [request.url.path for request in requests] == ["/v1/models"]
    assert [request.url.host for request in requests] == ["local.test"]
    await client.aclose()


async def test_configure_commits_even_when_server_is_down(tmp_path: Path) -> None:
    async def dispatch(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, state_store, _ = _service(tmp_path, client=client)
    result = await service.configure({"endpoint": "http://local.test:9000"}, _context("configure"))
    assert result.committed
    assert any(diagnostic.severity == "warning" for diagnostic in result.diagnostics)
    assert state_store.active() is not None
    await client.aclose()


async def test_configure_rejects_invalid_endpoint_without_state_change(tmp_path: Path) -> None:
    service, state_store, _ = _service(tmp_path)
    result = await service.configure({"endpoint": "not-a-url"}, _context("configure"))
    assert not result.committed
    assert any(diagnostic.severity == "error" for diagnostic in result.diagnostics)
    assert state_store.active() is None


async def test_configure_replaces_previous_credential(tmp_path: Path) -> None:
    client, _ = _client()
    service, state_store, credential_store = _service(tmp_path, client=client)
    first = await service.configure(
        {"endpoint": "http://local.test:9000", "api_key": "first"}, _context("configure")
    )
    assert first.committed
    first_ref = state_store.active().credential_ref
    second = await service.configure(
        {"endpoint": "http://local.test:9000", "api_key": "second"}, _context("configure")
    )
    assert second.committed
    second_ref = state_store.active().credential_ref
    assert first_ref is not None and second_ref is not None and first_ref != second_ref
    assert credential_store.get(first_ref) is None
    assert credential_store.get(second_ref) == "second"
    await client.aclose()


async def test_reset_clears_state_and_credential(tmp_path: Path) -> None:
    client, _ = _client()
    service, state_store, credential_store = _service(tmp_path, client=client)
    await service.configure({"endpoint": SERVER, "api_key": "secret"}, _context("configure"))
    result = await service.reset(_context("reset"))
    assert result.backend_status is not None
    assert result.backend_status.state == "unconfigured"
    assert state_store.active() is None
    assert credential_store.names() == ()
    await client.aclose()


async def test_environment_api_key_is_reported_and_used(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    async def dispatch(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "m", "object": "model"}]}
            )
        raise AssertionError(f"unexpected request: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, _, _ = _service(
        tmp_path,
        client=client,
        environment={"VLLM_API_KEY": "env-secret"},
    )
    status = await service.refresh(_context())
    assert status.backend_status is not None
    assert status.backend_status.authentication_source == "environment"
    assert requests[0].headers.get("Authorization") == "Bearer env-secret"
    await client.aclose()


async def test_offline_provider_refresh_keeps_cached_snapshot(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    snapshot = await service.refresh_models(
        ProviderRefreshContext(
            signal=SimpleCancellationToken(),
            allow_network=False,
            cached_models=(),
            auth=ResolvedProviderAuth(),
        )
    )
    assert snapshot.models == ()
    assert snapshot.default_model is None


def test_state_store_roundtrip_and_clear(tmp_path: Path) -> None:
    store = LocalServerStateStore(path=tmp_path / "state.json")
    assert store.active() is None
    assert (
        store.save(LocalServerState(endpoint=SERVER, credential_ref="local-servers:vllm:ref"))
        is None
    )
    active = store.active()
    assert active is not None
    assert active.endpoint == SERVER
    assert active.credential_ref == "local-servers:vllm:ref"
    assert store.save(LocalServerState(endpoint=f"{SERVER}/v1")) == "local-servers:vllm:ref"
    assert store.clear() is None
    assert store.active() is None
