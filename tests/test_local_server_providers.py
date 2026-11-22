"""Keyless OpenAI-compatible local server provider support.

Local server login templates pre-fill an OpenAI-compatible provider, enable
runtime model discovery, and allow an absent API key. These tests cover the
config/catalog layer; the TUI flow lives in test_tui_app.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tau_coding.catalog_loader import (
    effective_catalog,
    save_user_catalog_entries,
    user_catalog_path,
)
from tau_coding.local_server_presets import LOCAL_SERVER_PRESETS, local_server_preset
from tau_coding.paths import TauPaths
from tau_coding.provider_catalog import ProviderCatalogEntry
from tau_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    load_provider_settings,
    openai_compatible_config_from_provider,
)


def _local_provider(**overrides: object) -> OpenAICompatibleProviderConfig:
    values: dict[str, object] = {
        "name": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key_env": "OLLAMA_API_KEY",
        "credential_name": "ollama",
        "models": ("llama3.2",),
        "default_model": "llama3.2",
        "require_api_key": False,
    }
    values.update(overrides)
    return OpenAICompatibleProviderConfig(**values)  # type: ignore[arg-type]


def test_presets_are_unique_and_use_loopback_v1_endpoints() -> None:
    names = [preset.name for preset in LOCAL_SERVER_PRESETS]
    assert len(names) == len(set(names))
    for preset in LOCAL_SERVER_PRESETS:
        assert preset.base_url.startswith("http://127.0.0.1:")
        assert preset.base_url.endswith("/v1")
        assert preset.api_key_env.endswith("_API_KEY")
    assert {"ollama", "lm-studio", "vllm"}.issubset(set(names))
    assert local_server_preset("ollama") is LOCAL_SERVER_PRESETS[0]
    assert local_server_preset("missing") is None


def test_keyless_provider_omits_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    config = openai_compatible_config_from_provider(_local_provider())
    assert config.api_key == ""
    assert config.omit_authorization_header is True


def test_optional_key_is_still_sent_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-local")
    config = openai_compatible_config_from_provider(_local_provider())
    assert config.api_key == "sk-local"
    assert config.omit_authorization_header is False


def test_required_provider_still_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    provider = OpenAICompatibleProviderConfig(
        name="nebius",
        base_url="https://api.studio.nebius.ai/v1",
        api_key_env="NEBIUS_API_KEY",
        models=("model-a",),
        default_model="model-a",
    )
    with pytest.raises(RuntimeError, match="Missing provider API key"):
        openai_compatible_config_from_provider(provider)


def test_local_provider_flags_survive_catalog_roundtrip(tmp_path: Path) -> None:
    paths = TauPaths(home=tmp_path / ".tau")
    entry = ProviderCatalogEntry(
        name="ollama",
        display_name="Ollama",
        kind="openai-compatible",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        credential_name="ollama",
        models=("llama3.2",),
        default_model="llama3.2",
        docs_url="http://127.0.0.1:11434/v1",
        model_discovery=True,
        require_api_key=False,
    )
    save_user_catalog_entries((entry,), paths=paths)

    text = user_catalog_path(paths).read_text(encoding="utf-8")
    assert "model_discovery = true" in text
    assert "require_api_key = false" in text

    reloaded = {item.name: item for item in effective_catalog(paths)}["ollama"]
    assert reloaded.model_discovery is True
    assert reloaded.require_api_key is False

    provider = load_provider_settings(paths).get_provider("ollama")
    assert isinstance(provider, OpenAICompatibleProviderConfig)
    assert provider.model_discovery is True
    assert provider.require_api_key is False
