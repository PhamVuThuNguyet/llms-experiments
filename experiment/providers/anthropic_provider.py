from __future__ import annotations

from typing import Any

from .openai_compatible import OpenAICompatibleProvider


class AnthropicProvider(OpenAICompatibleProvider):
    """Anthropic via an OpenAI-compatible gateway.

    Expected env vars:
    - ANTHROPIC_API_KEY
    - ANTHROPIC_OPENAI_BASE_URL (e.g. https://.../v1)
    """

    provider_name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    base_url_env = "ANTHROPIC_OPENAI_BASE_URL"
    # OpenAI-compatible Anthropic gateways typically omit json_schema response_format
    supports_json_schema = False

    def __init__(self, model_name: str, **model_configurations: Any):
        super().__init__(model_name, **model_configurations)
