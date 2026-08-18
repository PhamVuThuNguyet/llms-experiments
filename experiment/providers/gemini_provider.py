from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai

from .base import BaseProvider, ProviderResponse
from .openai_compatible import HTTP_OK, IMAGE_MIME_TYPES, MS_PER_SECOND


class GeminiProvider(BaseProvider):
    provider_name = "gemini"

    def __init__(self, model_name: str, **model_configurations: Any) -> None:
        super().__init__(model_name, **model_configurations)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def _encode_image(self, image_path: str | Path) -> Dict[str, Any]:
        path = Path(image_path)
        mime = IMAGE_MIME_TYPES.get(path.suffix.lower(), "image/png")
        return {"mime_type": mime, "data": path.read_bytes()}

    def _generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str],
        json_schema: Optional[Dict[str, Any]],
    ) -> ProviderResponse:
        parts: list[Any] = []
        if system_prompt:
            parts.append(system_prompt)
        if image_path:
            parts.append(self._encode_image(image_path))
        if user_prompt:
            parts.append(user_prompt)

        gen_config: Dict[str, Any] = {}
        if json_schema is not None:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = json_schema
        gen_config.update(self.model_configurations)

        start = time.perf_counter()
        first_token_time: Optional[float] = None
        text: Optional[str] = None
        http_status: Optional[int] = None
        error_category: Optional[str] = None
        error_message: Optional[str] = None
        response_params: Dict[str, Any] | None = {"model": self.model_name}

        try:
            response = self.client.models.generate_content(
                parts, generation_config=gen_config, stream=True
            )
            accum: list[str] = []
            for chunk in response:
                if chunk.text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    accum.append(chunk.text)
            text = "".join(accum).strip()
            http_status = HTTP_OK
        except Exception as error:
            error_category = type(error).__name__
            error_message = str(error)
            response = getattr(error, "response", None)
            http_status = getattr(response, "status_code", None) if response else None

        end = time.perf_counter()
        ttft_ms = (
            (first_token_time - start) * MS_PER_SECOND if first_token_time else None
        )
        total_latency_ms = (end - start) * MS_PER_SECOND

        return ProviderResponse(
            text=text,
            input_tokens=None,
            output_tokens=None,
            input_chars=len(user_prompt) if user_prompt else None,
            output_chars=len(text) if text else None,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            http_status=http_status,
            error_category=error_category,
            error_message=error_message,
            response_params=response_params,
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        return await asyncio.to_thread(
            self._generate_sync,
            system_prompt,
            user_prompt,
            image_path,
            json_schema,
        )
