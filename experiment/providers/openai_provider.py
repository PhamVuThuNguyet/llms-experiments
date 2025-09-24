from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from .base import BaseProvider, ProviderResponse


class OpenAIProvider(BaseProvider):
	provider_name = "openai"

	def __init__(self, model_name: str, **model_configurations: Any) -> None:
		super().__init__(model_name, **model_configurations)
		self.client = OpenAI()

	def _encode_image(self, image_path: str | Path) -> str:
		p = Path(image_path)
		data = p.read_bytes()
		return base64.b64encode(data).decode('utf-8')

	def generate(self, system_prompt: str, user_prompt: str, image_path: Optional[str] = None, json_schema: Optional[Dict[str, Any]] = None) -> ProviderResponse:
		messages: list[dict[str, Any]] = []
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})

		content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
		if image_path:
			b64 = self._encode_image(image_path)
			content.append({
				"type": "input_image",
				"image": {
					"data": b64
				}
			})
		messages.append({"role": "user", "content": content})

		kwargs: Dict[str, Any] = dict(model=self.model_name, messages=messages)
		kwargs.update(self.model_configurations)

		# JSON schema support for response_format
		if json_schema is not None:
			kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}

		start = time.perf_counter()
		first_token_time: Optional[float] = None
		accum_text_parts: list[str] = []
		retry_count = 0
		http_status: Optional[int] = None
		error_category: Optional[str] = None

		try:
			with self.client.chat.completions.stream(**kwargs) as stream:
				for event in stream:
					etype = event.type
					if etype == "response.refuse.delta":
						continue
					if etype == "response.error":
						error_category = "api_error"
						break
					if etype == "response.output_text.delta":
						if first_token_time is None:
							first_token_time = time.perf_counter()
						accum_text_parts.append(event.delta)
					elif etype == "response.completed":
						break
			http_status = 200
		except Exception as e:
			error_category = type(e).__name__
			http_status = None
			text = None
		else:
			text = "".join(accum_text_parts).strip()

		end = time.perf_counter()
		ttft_ms = (first_token_time - start) * 1000 if first_token_time else None
		total_latency_ms = (end - start) * 1000

		# Usage: OpenAI new responses include usage after stream; fallback to char counts
		input_tokens = None
		output_tokens = None
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
		)
