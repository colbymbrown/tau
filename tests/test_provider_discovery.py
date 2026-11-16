"""Runtime model discovery for OpenAI-compatible providers (GreenPT/Viro).

Discovery reuses the codex `ModelCatalogProvider` pipeline: the openai-compatible
adapter advertises `/v1/models`, catalog entries opt in via `model_discovery`,
and the session publishes discovered inventories only for opted-in providers.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tau_ai import OpenAICompatibleConfig, OpenAICompatibleProvider
from tau_ai.env import RuntimeProviderAuth
from tau_ai.model_catalog import (
    ModelCatalogProvider,
    RuntimeModel,
    RuntimeModelCatalog,
)
from tau_ai.openai_compatible import _parse_discovered_models
from tau_coding.provider_catalog import BUILTIN_PROVIDER_CATALOG
from tau_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    ProviderSettings,
)
from tau_coding.session import _provider_with_runtime_model_catalog


def _payload(*model_ids: str) -> str:
    return json.dumps({"object": "list", "data": [{"id": model_id} for model_id in model_ids]})


# --- adapter-level: discover_models -------------------------------------------------


@pytest.mark.anyio
async def test_discover_models_requests_bearer_models_endpoint() -> None:
    requested: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested["url"] = str(request.url)
        requested["auth"] = request.headers.get("authorization")
        return httpx.Response(200, text=_payload("glm-5.3", "deepseek-v4.1-flash", "kimi-k3"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="sk-test", base_url="https://api.greenpt.ai/v1"),
            client=client,
        )
        catalog = await provider.discover_models()

    assert requested["url"] == "https://api.greenpt.ai/v1/models"
    assert requested["auth"] == "Bearer sk-test"
    assert [model.id for model in catalog.models] == ["glm-5.3", "deepseek-v4.1-flash", "kimi-k3"]


@pytest.mark.anyio
async def test_discover_models_uses_credential_resolver_base_url_override() -> None:
    requested: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested["url"] = str(request.url)
        requested["auth"] = request.headers.get("authorization")
        return httpx.Response(200, text=_payload("viro/deepseek-v4.1-flash"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="sk-stale",
                base_url="https://stale.example/v1",
                credential_resolver=lambda: _resolve_auth(
                    RuntimeProviderAuth(api_key="sk-live", base_url="https://api.viro.app/v1")
                ),
            ),
            client=client,
        )
        catalog = await provider.discover_models()

    assert requested["url"] == "https://api.viro.app/v1/models"
    assert requested["auth"] == "Bearer sk-live"
    assert catalog.models[0].id == "viro/deepseek-v4.1-flash"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        '{"object": "list", "data": []}',
        '{"data": "nope"}',
        "not json",
        "[1, 2, 3]",
    ],
)
async def test_discover_models_rejects_empty_or_malformed_payloads(payload: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="sk-test", base_url="https://example.test/v1"),
            client=client,
        )
        with pytest.raises((ValueError, json.JSONDecodeError)):
            await provider.discover_models()


@pytest.mark.anyio
async def test_discover_models_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"error": "unauthorized"}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="sk-bad", base_url="https://example.test/v1"),
            client=client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await provider.discover_models()


def test_parse_discovered_models_dedupes_ids_and_skips_garbage() -> None:
    catalog = _parse_discovered_models(
        {"data": [{"id": "a"}, {"id": "a"}, {"id": ""}, 42, {"id": "b"}]}
    )
    assert [model.id for model in catalog.models] == ["a", "b"]
    assert catalog.models[0].input_modalities == ("text",)


# --- merge-level: discovered inventory overlays static metadata -----------------------


def test_runtime_catalog_overlay_preserves_static_metadata_for_known_models() -> None:
    provider = OpenAICompatibleProviderConfig(
        name="greenpt",
        base_url="https://api.greenpt.ai/v1",
        models=("glm-5.3", "old-model"),
        default_model="glm-5.3",
        context_windows={"glm-5.3": 1_000_000},
        model_discovery=True,
    )
    catalog = RuntimeModelCatalog(
        (
            RuntimeModel(id="glm-5.3"),
            RuntimeModel(id="deepseek-v4.1-flash"),
        )
    )
    merged = _provider_with_runtime_model_catalog(provider, catalog)
    assert isinstance(merged, OpenAICompatibleProviderConfig)
    assert merged.models == ("glm-5.3", "deepseek-v4.1-flash")
    assert merged.context_windows == {"glm-5.3": 1_000_000}
    assert merged.model_discovery is True
    # Static default survives when still discovered.
    assert merged.default_model == "glm-5.3"


def test_runtime_catalog_overlay_falls_back_to_first_model_when_default_vanishes() -> None:
    provider = OpenAICompatibleProviderConfig(
        name="greenpt",
        models=("glm-5.3",),
        default_model="glm-5.3",
        model_discovery=True,
    )
    merged = _provider_with_runtime_model_catalog(
        provider,
        RuntimeModelCatalog((RuntimeModel(id="kimi-k3"),)),
    )
    assert isinstance(merged, OpenAICompatibleProviderConfig)
    assert merged.default_model == "kimi-k3"


def test_runtime_catalog_overlay_overlays_any_openai_compatible_provider() -> None:
    # The session gates discovery before publishing a catalog, so the merge
    # itself accepts any openai-compatible provider config that receives one.
    provider = OpenAICompatibleProviderConfig(
        name="openai",
        models=("gpt-5.4",),
        default_model="gpt-5.4",
    )
    merged = _provider_with_runtime_model_catalog(
        provider, RuntimeModelCatalog((RuntimeModel(id="gpt-6-sol"),))
    )
    assert isinstance(merged, OpenAICompatibleProviderConfig)
    assert merged.models == ("gpt-6-sol",)
    assert merged.model_discovery is False


def test_runtime_catalog_overlay_ignores_anthropic_providers() -> None:
    from tau_coding.provider_config import AnthropicProviderConfig

    provider = AnthropicProviderConfig(models=("claude-sonnet-5",))
    merged = _provider_with_runtime_model_catalog(
        provider, RuntimeModelCatalog((RuntimeModel(id="claude-opus-5"),))
    )
    assert merged is provider


# --- catalog-level: entries opt in ---------------------------------------------------


def test_builtin_catalog_entries_enable_model_discovery() -> None:
    entries = {entry.name: entry for entry in BUILTIN_PROVIDER_CATALOG}
    for name, base_url in (
        ("greenpt", "https://api.greenpt.ai/v1"),
        ("greenpt-us", "https://api.us.greenpt.ai/v1"),
        ("viro", "https://api.viro.app/v1"),
    ):
        entry = entries[name]
        assert entry.model_discovery is True, name
        assert entry.base_url == base_url
        assert entry.default_model in entry.models


def test_other_openai_compatible_entries_do_not_enable_discovery() -> None:
    # The opt-in flag must stay off for stable cloud catalogs so activation
    # never triggers an extra /v1/models probe for them.
    for entry in BUILTIN_PROVIDER_CATALOG:
        if entry.name in {"greenpt", "greenpt-us", "viro", "openai-codex"}:
            continue
        assert entry.model_discovery is False, entry.name


def test_greenpt_us_models_are_subset_of_greenpt_eu_models() -> None:
    entries = {entry.name: entry for entry in BUILTIN_PROVIDER_CATALOG}
    us = set(entries["greenpt-us"].models)
    eu = set(entries["greenpt"].models)
    assert us <= eu
    assert entries["greenpt"].api_key_env == entries["greenpt-us"].api_key_env
    assert entries["greenpt"].credential_name != entries["greenpt-us"].credential_name


# --- session-level: gate -------------------------------------------------------------


@pytest.mark.anyio
async def test_session_probes_only_discovery_enabled_providers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tau_agent.session import JsonlSessionStorage
    from tau_coding import CodingSessionConfig
    from tau_coding import session as session_module
    from tau_coding.paths import TauPaths
    from tau_coding.resources import TauResourcePaths

    calls: list[str] = []

    class FakeProvider(ModelCatalogProvider):
        async def discover_models(self) -> RuntimeModelCatalog:
            calls.append("discover")
            return RuntimeModelCatalog((RuntimeModel(id="live-model"),))

        async def aclose(self) -> None:
            return None

    def create(*args: object, **kwargs: object) -> FakeProvider:
        return FakeProvider()

    monkeypatch.setattr(session_module, "create_model_provider", create)

    for discovery_enabled, expected_calls in ((False, 0), (True, 1)):
        calls.clear()
        provider = OpenAICompatibleProviderConfig(
            name="greenpt",
            base_url="https://api.greenpt.ai/v1",
            api_key_env="GREENPT_API_KEY",
            models=("static-model",),
            default_model="static-model",
            model_discovery=discovery_enabled,
        )
        settings = ProviderSettings(default_provider="greenpt", providers=(provider,))
        monkeypatch.setenv("GREENPT_API_KEY", "sk-test")
        session = await session_module.CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider(),
                owns_initial_provider=True,
                model="static-model",
                provider_name="greenpt",
                provider_settings=settings,
                runtime_provider_config=provider,
                system="Test",
                cwd=tmp_path,
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                extensions_enabled=False,
                resource_paths=TauResourcePaths(
                    root=tmp_path / ".tau",
                    paths=TauPaths(home=tmp_path / ".tau"),
                ),
            )
        )
        try:
            assert calls == ["discover"] * expected_calls
            if discovery_enabled:
                assert session.available_models == ("live-model",)
            else:
                assert session.available_models == ("static-model",)
        finally:
            await session.aclose()


async def _resolve_auth(auth: RuntimeProviderAuth) -> RuntimeProviderAuth:
    return auth
