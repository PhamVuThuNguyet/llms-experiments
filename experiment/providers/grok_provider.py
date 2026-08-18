from __future__ import annotations

from typing import Any

from .openai_compatible import OpenAICompatibleProvider

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"


class GrokProvider(OpenAICompatibleProvider):
    provider_name = "grok"
    api_key_env = "GROK_API_KEY"
    base_url_env = "XAI_API_BASE"
    default_base_url = DEFAULT_XAI_BASE_URL
    supports_json_schema = True

    def __init__(self, model_name: str, **model_configurations: Any):
        super().__init__(model_name, **model_configurations)
