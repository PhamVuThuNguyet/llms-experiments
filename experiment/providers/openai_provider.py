from __future__ import annotations

from typing import Any

from .openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    supports_json_schema = True

    def __init__(self, model_name: str, **model_configurations: Any):
        super().__init__(model_name, **model_configurations)
