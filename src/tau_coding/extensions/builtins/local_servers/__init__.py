"""Trusted built-in OpenAI-compatible local server backends."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tau_coding.built_in_extensions import BuiltInExtensionContext
from tau_coding.credentials import FileCredentialStore, credentials_path
from tau_coding.paths import TauPaths

from .service import (
    DEFAULT_LOCAL_SERVER_TIMEOUT_SECONDS,
    LOCAL_SERVERS,
    LOCAL_SERVERS_CREDENTIAL_PREFIX,
    LOCAL_SERVERS_STATE_SCHEMA_VERSION,
    LocalServerEndpoint,
    LocalServerEndpointError,
    LocalServerError,
    LocalServerService,
    LocalServerState,
    LocalServerStateError,
    LocalServerStateStore,
    normalize_endpoint,
)

if TYPE_CHECKING:
    from tau_coding.extensions.api import ExtensionAPI


def setup(
    tau: ExtensionAPI,
    context: BuiltInExtensionContext | None = None,
) -> None:
    """Register one trusted provider and local backend per named server."""
    dependencies = context or BuiltInExtensionContext(
        paths=TauPaths(),
        credential_store=FileCredentialStore(credentials_path(TauPaths())),
        environment=dict(os.environ),
    )
    for backend_id, display_name, default_endpoint, api_key_env in LOCAL_SERVERS:
        service = LocalServerService(
            backend_id=backend_id,
            display_name=display_name,
            default_endpoint=default_endpoint,
            api_key_env=api_key_env,
            state_store=LocalServerStateStore(
                path=dependencies.paths.extension_state_dir / f"{backend_id}.json"
            ),
            credential_store=dependencies.credential_store,
            environment=dependencies.environment,
            client=dependencies.http_client,
            register_provider=tau.register_provider,
            update_provider=tau.update_provider,
            register_backend=tau.register_local_backend,
        )
        tau.register_provider(service.provider())
        tau.register_local_backend(service.backend())


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
    "setup",
]
