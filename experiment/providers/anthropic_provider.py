from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, Optional

import anthropic

from .base import BaseProvider, ProviderResponse


class AnthropicProvider(BaseProvider):
	provider_name = "anthropic"

	def __init__(self, model_name: str, **model_configurations: Any) -> None:
		super().__init__(model_name, **model_configurations)
		self.client = anthropic.Anthropic()

	def _encode_image(self, image_path: str | Path) -> Dict[str, Any]:
		p = Path(image_path)
		b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
		mime = {
			".png": "image/png",
			".jpg": "image/jpeg",
			".jpeg": "image/jpeg",
			".webp": "image/webp",
		}.get(p.suffix.lower(), "image/png")
		return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}

	def generate(self, system_prompt: str, user_prompt: str, image_path: Optional[str] = None, json_schema: Optional[Dict[str, Any]] = None) -> ProviderResponse:
		# Build content as list of blocks
		content: list[dict[str, Any]] = []
		if user_prompt:
			content.append({"type": "text", "text": user_prompt})
		if image_path:
			content.append(self._encode_image(image_path))

		kwargs: Dict[str, Any] = dict(
			model=self.model_name,
			messages=[{"role": "user", "content": content}],
		)
		if system_prompt:
			kwargs["system"] = system_prompt
		kwargs.update(self.model_configurations)

		# TODO: Double-check!!! Anthropic doesn't support strict json schema like OpenAI; we rely on system/user prompts
        
		start = time.perf_counter()
		first_token_time: Optional[float] = None
		accum_text_parts: list[str] = []
		http_status: Optional[int] = None
		error_category: Optional[str] = None

		try:
			with self.client.messages.stream(**kwargs) as stream:
				for event in stream:
					if event.type == "content_block_delta" and event.delta.get("type") == "output_text_delta":
						if first_token_time is None:
							first_token_time = time.perf_counter()
						accum_text_parts.append(event.delta.get("text", ""))
					elif event.type == "message_stop":
						break
			http_status = 200
		except Exception as e:
			error_category = type(e).__name__
			text = None
			first_token_time = None
		else:
			text = "".join(accum_text_parts).strip()

		end = time.perf_counter()
		ttft_ms = (first_token_time - start) * 1000 if first_token_time else None
		total_latency_ms = (end - start) * 1000

		# Usage tokens not available in stream; leave None; char counts as fallback
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
