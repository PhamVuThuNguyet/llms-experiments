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

HTTP_OK = 200
MS_PER_SECOND = 1000.0

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class OpenAICompatibleProvider(BaseProvider):
    """Shared streaming chat-completions flow for OpenAI-compatible APIs."""

    supports_json_schema: bool = True
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str | None = None
    default_base_url: str | None = None

    def __init__(self, model_name: str, **model_configurations: Any):
        super().__init__(model_name, **model_configurations)
        api_key = os.environ.get(self.api_key_env)
        if self.base_url_env or self.default_base_url:
            base_url = (
                os.environ.get(self.base_url_env) if self.base_url_env else None
            ) or self.default_base_url
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)

    def _encode_image_data_url(self, image_path: str | Path) -> str:
        path = Path(image_path)
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        mime = IMAGE_MIME_TYPES.get(path.suffix.lower(), "image/png")
        return f"data:{mime};base64,{data}"

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if image_path:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image_data_url(image_path)},
                }
            )
        messages.append({"role": "user", "content": content})
        return messages

    def _build_request_kwargs(
        self,
        messages: list[dict[str, Any]],
        json_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        kwargs.update(self.model_configurations)
        if json_schema is not None and self.supports_json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        return kwargs

    @staticmethod
    def _http_status_from_exception(error: Exception) -> Optional[int]:
        response = getattr(error, "response", None)
        return getattr(response, "status_code", None) if response is not None else None

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        messages = self._build_messages(system_prompt, user_prompt, image_path)
        kwargs = self._build_request_kwargs(messages, json_schema)

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
                if getattr(chunk, "model", None) and response_params is None:
                    response_params = {"model": chunk.model}

                usage = getattr(chunk, "usage", None)
                if usage and input_tokens is None and output_tokens is None:
                    input_tokens = getattr(usage, "prompt_tokens", None)
                    output_tokens = getattr(usage, "completion_tokens", None)

                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    accum_text_parts.append(piece)
            http_status = HTTP_OK
            text = "".join(accum_text_parts).strip()
        except Exception as error:
            error_category = type(error).__name__
            error_message = str(error)
            text = None
            http_status = self._http_status_from_exception(error)

        end = time.perf_counter()
        ttft_ms = (
            (first_token_time - start) * MS_PER_SECOND if first_token_time else None
        )
        total_latency_ms = (end - start) * MS_PER_SECOND

        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_chars=len(user_prompt) if user_prompt else None,
            output_chars=len(text) if text else None,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            http_status=http_status,
            error_category=error_category,
            error_message=error_message,
            response_params=response_params,
        )
