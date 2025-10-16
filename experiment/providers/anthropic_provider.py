from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .base import BaseProvider, ProviderResponse

load_dotenv()


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    def __init__(self, model_name: str, **model_configurations: Any):
        super().__init__(model_name, **model_configurations)
        # Use OpenAI SDK compatibility. Require a base_url pointing to Anthropic-compatible endpoint.
        # Expected envs:
        # - ANTHROPIC_API_KEY: API key for the compatibility endpoint
        # - ANTHROPIC_OPENAI_BASE_URL: Base URL for OpenAI-compatible API (e.g., https://.../v1)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = os.environ.get("ANTHROPIC_OPENAI_BASE_URL")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)  # type: ignore[arg-type]

    def _encode_image_data_url(self, image_path: str | Path):
        p = Path(image_path)
        data = base64.b64encode(p.read_bytes()).decode("utf-8")
        ext = p.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext, "image/png")
        return f"data:{mime};base64,{data}"

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ):
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if image_path:
            data_url = self._encode_image_data_url(image_path)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages.append({"role": "user", "content": content})

        kwargs: Dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )

        # Apply model-specific overrides, if any
        kwargs.update(self.model_configurations)

        # NOTE: OpenAI-compatible gateways for Anthropic do not support response_format json_schema; omit by default

        start = time.perf_counter()
        first_token_time: Optional[float] = None
        accum_text_parts: list[str] = []
        http_status: Optional[int] = None
        error_category: Optional[str] = None
        error_message: Optional[str] = None
        input_tokens = None
        output_tokens = None
        response_params: Dict[str, Any] | None = None

        try:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if getattr(chunk, "model", None):
                    if response_params is None:
                        response_params = {"model": chunk.model}

                # usage on the terminal chunk when include_usage=True
                usage = getattr(chunk, "usage", None)
                if usage and input_tokens is None and output_tokens is None:
                    input_tokens = getattr(usage, "prompt_tokens", None)
                    output_tokens = getattr(usage, "completion_tokens", None)

                # delta text
                choices = getattr(chunk, "choices", None)
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        piece = getattr(delta, "content", None)
                        if piece:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            accum_text_parts.append(piece)
            http_status = 200
        except Exception as e:
            error_category = type(e).__name__
            error_message = str(e)
            text = None
            # Try to extract HTTP status from exceptions
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                http_status = e.response.status_code
        else:
            text = "".join(accum_text_parts).strip()

        end = time.perf_counter()
        ttft_ms = (first_token_time - start) * 1000 if first_token_time else None
        total_latency_ms = (end - start) * 1000

        # Char counts as fallback
        input_chars = len(user_prompt) if user_prompt else None
        output_chars = len(text) if text else None

        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_chars=input_chars,
            output_chars=output_chars,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            http_status=http_status,
            error_category=error_category,
            error_message=error_message,
            response_params=response_params,
        )
