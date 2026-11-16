"""Deterministic tests for the trusted Ollama local backend."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx
import pytest

from tau_agent.harness import SimpleCancellationToken
from tau_coding.credentials import FileCredentialStore
from tau_coding.extensions import DynamicProviderRegistry, ExtensionRuntime, LocalBackendRegistry
from tau_coding.extensions.builtins.local_servers.service import (
    LocalServerStateStore,
)
from tau_coding.extensions.builtins.ollama.service import (
    OLLAMA_BACKEND_ID,
    OLLAMA_DEFAULT_ENDPOINT,
    OLLAMA_HOST_ENV,
    OllamaService,
    _format_bytes,
    _pull_progress,
)
from tau_coding.extensions.providers import ProviderRefreshContext, ResolvedProviderAuth
from tau_coding.local_backends import LocalOperationContext, LocalProgress
from tau_coding.paths import TauPaths
from tau_coding.resources import TauResourcePaths

pytestmark = pytest.mark.anyio

TAGS = {"models": [{"name": "qwen3:8b"}, {"name": "code:latest"}]}
PS_EMPTY = {"models": []}
PS_QWEN = {"models": [{"name": "qwen3:8b"}]}
OPENAI_MODELS = {
    "object": "list",
    "data": [
        {"id": "qwen3:8b", "object": "model", "owned_by": "library"},
        {"id": "code:latest", "object": "model", "owned_by": "library"},
    ],
}
PULL_STREAM = "\n".join(
    [
        json.dumps({"status": "pulling manifest"}),
        json.dumps(
            {"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 40}
        ),
        json.dumps(
            {"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 100}
        ),
        json.dumps({"status": "verifying sha256 digest"}),
        json.dumps({"status": "success"}),
    ]
)


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
        backend_id=OLLAMA_BACKEND_ID,
        source_id="built-in:ollama",
        confirmation=confirmation,
        _is_current=lambda: True,
        _progress=progress or (lambda _: None),
    )


def _service(
    tmp_path: Path,
    *,
    client: httpx.AsyncClient | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[OllamaService, LocalServerStateStore]:
    paths = TauPaths(home=tmp_path / "tau", agents_home=tmp_path / "agents")
    state_store = LocalServerStateStore(path=paths.extension_state_dir / "ollama.json")
    service = OllamaService(
        state_store=state_store,
        environment=environment or {},
        client=client,
    )
    return service, state_store


def _client(
    requests: list[httpx.Request] | None = None,
) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    captured = requests if requests is not None else []
    ps_payload = dict(PS_EMPTY)
    pulls: list[Mapping[str, object]] = []

    def dispatch(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=OPENAI_MODELS)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=ps_payload)
        if request.url.path == "/api/generate":
            body = json.loads(request.content) if request.content else {}
            if body.get("keep_alive") != 0:
                ps_payload["models"] = [{"name": body.get("model")}]
            else:
                ps_payload["models"] = []
            return httpx.Response(200, json={"done": True})
        if request.url.path == "/api/pull":
            pulls.append(json.loads(request.content) if request.content else {})
            return httpx.Response(
                200,
                content=PULL_STREAM.encode(),
                headers={"content-type": "application/x-ndjson"},
            )
        raise AssertionError(f"unexpected Ollama request: {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(dispatch)), captured


def _runtime(tmp_path: Path) -> ExtensionRuntime:
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
    return runtime


async def test_builtin_registers_ollama_backend_and_provider(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
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
    assert backends["ollama"].use_available
    assert runtime.provider_registry.effective("ollama") is not None
    await runtime.aclose()


async def test_refresh_reports_running_and_installed_models(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []
    ps_payload = {"models": [{"name": "qwen3:8b"}]}

    def dispatch(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=ps_payload)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=OPENAI_MODELS)
        raise AssertionError(f"unexpected Ollama request: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, _ = _service(tmp_path, client=client)
    providers = DynamicProviderRegistry(generation_id="generation")
    backends = LocalBackendRegistry(providers, generation_id="generation")
    providers.register("built-in:ollama", service.provider())
    backends.register("built-in:ollama", service.backend())

    provider_result = await providers.refresh("ollama")
    backend_result = await backends.refresh("ollama")

    assert provider_result.provider is not None
    assert [model.id for model in provider_result.provider.models] == ["qwen3:8b", "code:latest"]
    status = backend_result.backend_status
    assert status is not None
    assert status.state == "ready"
    assert status.endpoint_display == f"{OLLAMA_DEFAULT_ENDPOINT}/v1"
    assert [model.id for model in status.models] == ["qwen3:8b", "code:latest"]
    assert status.models[0].state == "running"
    assert status.models[1].state == "installed"
    assert status.selected_model == "qwen3:8b"
    assert "use" in status.actions
    assert "download_model" in status.actions
    assert "load_model" in status.actions
    assert "unload_model" in status.actions
    assert [request.url.path for request in captured] == [
        "/v1/models",
        "/api/tags",
        "/api/ps",
    ]
    await backends.aclose()
    await providers.aclose()
    await client.aclose()


async def test_provider_refresh_uses_openai_compat_without_auth(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    async def dispatch(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=OPENAI_MODELS)
        raise AssertionError(f"unexpected request: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, _ = _service(tmp_path, client=client)
    snapshot = await service.refresh_models(
        ProviderRefreshContext(
            signal=SimpleCancellationToken(),
            allow_network=True,
            cached_models=(),
            auth=ResolvedProviderAuth(),
        )
    )
    assert [model.id for model in snapshot.models] == ["qwen3:8b", "code:latest"]
    assert "Authorization" not in requests[0].headers
    await client.aclose()


async def test_offline_provider_refresh_keeps_cached_snapshot(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    snapshot = await service.refresh_models(
        ProviderRefreshContext(
            signal=SimpleCancellationToken(),
            allow_network=False,
            cached_models=(),
            auth=ResolvedProviderAuth(),
        )
    )
    assert snapshot.models == ()


async def test_download_requires_confirmation_then_streams_progress(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    progress: list[LocalProgress] = []
    client, _ = _client(requests)
    service, _ = _service(tmp_path, client=client)

    pending = await service.download_model("llama3:8b", _context("download_model"))
    assert pending.confirmation is not None
    assert "download" in {choice.value for choice in pending.confirmation.choices}
    assert not any(request.url.path == "/api/pull" for request in requests)

    result = await service.download_model(
        "llama3:8b", _context("download_model", confirmation="download", progress=progress.append)
    )
    assert result.message == "Downloaded llama3:8b."
    assert result.backend_status is not None
    assert any(item.done for item in progress)
    assert any(item.fraction is not None for item in progress)
    pulls = [request for request in requests if request.url.path == "/api/pull"]
    assert len(pulls) == 1
    assert json.loads(pulls[0].content) == {"model": "llama3:8b", "stream": True}
    await client.aclose()


async def test_download_reports_pull_error(tmp_path: Path) -> None:
    async def dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/pull":
            return httpx.Response(
                200,
                content=json.dumps({"status": "error", "error": "model not found"}).encode(),
                headers={"content-type": "application/x-ndjson"},
            )
        if request.url.path in {"/api/tags", "/api/ps"}:
            return httpx.Response(200, json={"models": []})
        raise AssertionError(f"unexpected request: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, _ = _service(tmp_path, client=client)
    result = await service.download_model(
        "missing:model", _context("download_model", confirmation="download")
    )
    assert result.diagnostics
    assert any("not found" in diagnostic.message for diagnostic in result.diagnostics)
    await client.aclose()


async def test_load_model_confirms_then_warms_into_memory(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    client, _ = _client(requests)
    service, _ = _service(tmp_path, client=client)

    pending = await service.load_model("code:latest", _context("load_model"))
    assert pending.confirmation is not None
    assert pending.backend_status is not None
    assert pending.backend_status.models[1].state == "installed"

    result = await service.load_model("code:latest", _context("load_model", confirmation="load"))
    assert result.message == "Loaded code:latest."
    assert result.backend_status is not None
    assert result.backend_status.selected_model == "code:latest"
    generate = [request for request in requests if request.url.path == "/api/generate"]
    assert len(generate) == 1
    body = json.loads(generate[0].content)
    assert body["model"] == "code:latest"
    assert body["keep_alive"] == "30m"
    await client.aclose()


async def test_unload_model_confirms_then_releases_memory(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    ps_payload = dict(PS_QWEN)

    def dispatch(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=ps_payload)
        if request.url.path == "/api/generate":
            body = json.loads(request.content) if request.content else {}
            if body.get("keep_alive") == 0:
                ps_payload["models"] = []
            return httpx.Response(200, json={"done": True})
        raise AssertionError(f"unexpected request: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, _ = _service(tmp_path, client=client)

    pending = await service.unload_model("qwen3:8b", _context("unload_model"))
    assert pending.confirmation is not None

    result = await service.unload_model("qwen3:8b", _context("unload_model", confirmation="unload"))
    assert result.message == "Unloaded qwen3:8b."
    generate = [request for request in requests if request.url.path == "/api/generate"]
    assert len(generate) == 1
    assert json.loads(generate[0].content)["keep_alive"] == 0
    await client.aclose()


async def test_load_model_refuses_unknown_model(tmp_path: Path) -> None:
    client, _ = _client()
    service, _ = _service(tmp_path, client=client)
    result = await service.load_model("missing:model", _context("load_model"))
    assert result.diagnostics
    assert any("not installed" in diagnostic.message for diagnostic in result.diagnostics)
    await client.aclose()


async def test_configure_commits_endpoint_and_probes(tmp_path: Path) -> None:
    client, _ = _client()
    service, state_store = _service(tmp_path, client=client)
    result = await service.configure({"endpoint": "http://127.0.0.1:11435"}, _context("configure"))
    assert result.committed
    assert result.backend_status is not None
    assert result.backend_status.state == "ready"
    assert state_store.active() is not None
    assert state_store.active().endpoint == "http://127.0.0.1:11435"
    await client.aclose()


async def test_configure_rejects_invalid_endpoint(tmp_path: Path) -> None:
    service, state_store = _service(tmp_path)
    result = await service.configure({"endpoint": "not-a-url"}, _context("configure"))
    assert not result.committed
    assert state_store.active() is None


async def test_configure_commits_even_when_server_is_down(tmp_path: Path) -> None:
    async def dispatch(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, state_store = _service(tmp_path, client=client)
    result = await service.configure({"endpoint": "http://127.0.0.1:11435"}, _context("configure"))
    assert result.committed
    assert any(diagnostic.severity == "warning" for diagnostic in result.diagnostics)
    assert state_store.active() is not None
    await client.aclose()


async def test_reset_clears_state(tmp_path: Path) -> None:
    client, _ = _client()
    service, state_store = _service(tmp_path, client=client)
    await service.configure({"endpoint": "http://127.0.0.1:11435"}, _context("configure"))
    result = await service.reset(_context("reset"))
    assert result.backend_status is not None
    assert result.backend_status.state == "unconfigured"
    assert state_store.active() is None
    await client.aclose()


async def test_ollama_host_environment_is_used_when_unconfigured(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, environment={OLLAMA_HOST_ENV: "http://192.168.1.5:11434"})
    assert service.configured
    assert service.endpoint.server_root == "http://192.168.1.5:11434"


async def test_refresh_failure_is_stale_and_reports_connection_error(tmp_path: Path) -> None:
    async def dispatch(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, _ = _service(tmp_path, client=client)
    result = await service.refresh(_context())
    assert result.backend_status is not None
    assert result.backend_status.stale
    assert any(diagnostic.severity == "error" for diagnostic in result.backend_status.diagnostics)
    await client.aclose()


async def test_malformed_tags_are_rejected(tmp_path: Path) -> None:
    async def dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": {}})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        raise AssertionError(f"unexpected request: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service, _ = _service(tmp_path, client=client)
    result = await service.refresh(_context())
    assert result.diagnostics
    assert any("malformed" in diagnostic.message for diagnostic in result.diagnostics)
    await client.aclose()


def test_pull_progress_formatting() -> None:
    item = _pull_progress("qwen3:8b", 40, 100)
    assert item.fraction == 0.4
    assert not item.done
    assert "40" in item.message
    item = _pull_progress("qwen3:8b", 200, 100)
    assert item.fraction == 1.0
    assert _format_bytes(1024) == "1.0 KiB"
