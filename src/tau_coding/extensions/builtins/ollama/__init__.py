"""Trusted built-in Ollama local backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tau_coding.built_in_extensions import BuiltInExtensionContext
from tau_coding.credentials import FileCredentialStore, credentials_path
from tau_coding.extensions.builtins.local_servers.service import LocalServerStateStore
from tau_coding.paths import TauPaths

from .service import (
    OLLAMA_BACKEND_ID,
    OLLAMA_DEFAULT_ENDPOINT,
    OLLAMA_DISPLAY_NAME,
    OLLAMA_HOST_ENV,
    OLLAMA_LOAD_KEEP_ALIVE,
    OLLAMA_POLL_INTERVAL_SECONDS,
    OLLAMA_PROVIDER_ID,
    OLLAMA_PULL_TIMEOUT_SECONDS,
    OLLAMA_STATE_WAIT_TIMEOUT_SECONDS,
    OllamaError,
    OllamaService,
)

if TYPE_CHECKING:
    from tau_coding.extensions.api import ExtensionAPI


def setup(
    tau: ExtensionAPI,
    context: BuiltInExtensionContext | None = None,
) -> None:
    """Register the trusted Ollama provider and local backend."""
    dependencies = context or BuiltInExtensionContext(
        paths=TauPaths(),
        credential_store=FileCredentialStore(credentials_path(TauPaths())),
        environment=dict(os.environ),
    )
    service = OllamaService(
        state_store=LocalServerStateStore(
            path=dependencies.paths.extension_state_dir / "ollama.json"
        ),
        environment=dependencies.environment,
        client=dependencies.http_client,
        register_provider=tau.register_provider,
        update_provider=tau.update_provider,
        register_backend=tau.register_local_backend,
    )
    tau.register_provider(service.provider())
    tau.register_local_backend(service.backend())


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
    "setup",
]
