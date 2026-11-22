"""Login presets for keyless OpenAI-compatible local inference servers.

These are templates, not managed backends: each preset pre-fills the custom
OpenAI-compatible provider form with one server's default endpoint and enables
runtime model discovery. Tau never starts, stops, installs, or scans these
servers; they stay the user's to run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalServerPreset:
    """One selectable local server template."""

    name: str
    display_name: str
    base_url: str
    api_key_env: str


LOCAL_SERVER_PRESETS: tuple[LocalServerPreset, ...] = (
    LocalServerPreset("ollama", "Ollama", "http://127.0.0.1:11434/v1", "OLLAMA_API_KEY"),
    LocalServerPreset("lm-studio", "LM Studio", "http://127.0.0.1:1234/v1", "LM_STUDIO_API_KEY"),
    LocalServerPreset("vllm", "vLLM", "http://127.0.0.1:8000/v1", "VLLM_API_KEY"),
    LocalServerPreset("unsloth", "Unsloth", "http://127.0.0.1:8000/v1", "UNSLOTH_API_KEY"),
    LocalServerPreset(
        "llama-server",
        "llama.cpp server",
        "http://127.0.0.1:8080/v1",
        "LLAMA_API_KEY",
    ),
)


def local_server_preset(name: str) -> LocalServerPreset | None:
    """Return one preset by provider name."""
    for preset in LOCAL_SERVER_PRESETS:
        if preset.name == name:
            return preset
    return None
