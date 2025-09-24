from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai

from .base import BaseProvider, ProviderResponse


class GeminiProvider(BaseProvider):
	provider_name = "gemini"

	def __init__(self, model_name: str, **model_configurations: Any) -> None:
		super().__init__(model_name, **model_configurations)
		api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
		self.client = genai.Client(api_key=api_key)
		
	def _encode_image(self, image_path: str | Path) -> Dict[str, Any]:
		p = Path(image_path)
		b = p.read_bytes()
		mime = {
			".png": "image/png",
			".jpg": "image/jpeg",
			".jpeg": "image/jpeg",
			".webp": "image/webp",
		}.get(p.suffix.lower(), "image/png")
		return {"mime_type": mime, "data": b}

	def generate(self, system_prompt: str, user_prompt: str, image_path: Optional[str] = None, json_schema: Optional[Dict[str, Any]] = None) -> ProviderResponse:
		parts: list[Any] = []
		if system_prompt:
			parts.append(system_prompt)
		if image_path:
			parts.append(self._encode_image(image_path))
		if user_prompt:
			parts.append(user_prompt)

		# JSON schema in Gemini uses response_mime_type + schema in generation_config
		gen_config: Dict[str, Any] = dict()
		if json_schema is not None:
			gen_config["response_mime_type"] = "application/json"
			gen_config["response_schema"] = json_schema
		gen_config.update(self.model_configurations)

		start = time.perf_counter()
		first_token_time: Optional[float] = None
		text: Optional[str] = None
		http_status: Optional[int] = None
		error_category: Optional[str] = None

		try:
			response = self.client.models.generate_content(parts, generation_config=gen_config, stream=True)
			accum: list[str] = []
			for chunk in response:
				if chunk.text:
					if first_token_time is None:
						first_token_time = time.perf_counter()
					accum.append(chunk.text)
			text = "".join(accum).strip()
			http_status = 200
		except Exception as e:
			error_category = type(e).__name__

		end = time.perf_counter()
		ttft_ms = (first_token_time - start) * 1000 if first_token_time else None
		total_latency_ms = (end - start) * 1000

		input_chars = len(user_prompt) if user_prompt else None
		output_chars = len(text) if text else None

		return ProviderResponse(
			text=text,
			input_tokens=None,
			output_tokens=None,
			input_chars=input_chars,
			output_chars=output_chars,
			ttft_ms=ttft_ms,
			total_latency_ms=total_latency_ms,
			http_status=http_status,
			error_category=error_category,
		)
