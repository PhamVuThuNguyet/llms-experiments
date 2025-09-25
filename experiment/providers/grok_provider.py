from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .base import BaseProvider, ProviderResponse


class GrokProvider(BaseProvider):
	provider_name = "grok"

	def __init__(self, model_name: str, **model_configurations: Any) -> None:
		super().__init__(model_name, **model_configurations)
		self.api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
		self.base_url = os.getenv("XAI_API_BASE", "https://api.x.ai")

	def _encode_image(self, image_path: str | Path) -> Dict[str, Any]:
		p = Path(image_path)
		b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
		mime = {
			".png": "image/png",
			".jpg": "image/jpeg",
			".jpeg": "image/jpeg",
			".webp": "image/webp",
		}.get(p.suffix.lower(), "image/png")
		return {"type": "input_image", "image": {"data": b64, "media_type": mime}}

	def generate(self, system_prompt: str, user_prompt: str, image_path: Optional[str] = None, json_schema: Optional[Dict[str, Any]] = None) -> ProviderResponse:
		# xAI exposes an OpenAI-compatible endpoint /v1/chat/completions but streaming formats may vary
		headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
		messages: list[dict[str, Any]] = []
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})
		content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
		if image_path:
			content.append(self._encode_image(image_path))
		messages.append({"role": "user", "content": content})

		payload: Dict[str, Any] = {
			"model": self.model_name,
			"messages": messages,
			"stream": True,
		}
		payload.update(self.model_configurations)
		if json_schema is not None:
			payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}

		url = f"{self.base_url}/v1/chat/completions"
		start = time.perf_counter()
		first_token_time: Optional[float] = None
		accum_text: list[str] = []
		http_status: Optional[int] = None
		error_category: Optional[str] = None
		response_params: Dict[str, Any] | None = {"model": self.model_name}

		try:
			with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as r:
				http_status = r.status_code
				r.raise_for_status()
				for line in r.iter_lines(decode_unicode=True):
					if not line:
						continue
					if line.startswith("data:"):
						data = line[len("data:"):].strip()
						if data == "[DONE]":
							break
						try:
							chunk = requests.utils.json.loads(data)
							delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
							if delta:
								if first_token_time is None:
									first_token_time = time.perf_counter()
								accum_text.append(delta)
						except Exception:
							continue
		except Exception as e:
			error_category = type(e).__name__
			text = None
		else:
			text = "".join(accum_text).strip()

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
			response_params=response_params,
		)
