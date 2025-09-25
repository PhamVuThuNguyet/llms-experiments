from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProviderResponse:
	text: Optional[str]
	input_tokens: Optional[int]
	output_tokens: Optional[int]
	input_chars: Optional[int]
	output_chars: Optional[int]
	ttft_ms: Optional[float]
	total_latency_ms: Optional[float]
	http_status: Optional[int]
	error_category: Optional[str]
	response_params: Optional[Dict[str, Any]] = None


class BaseProvider:
	provider_name: str

	def __init__(self, model_name: str, **model_configurations: Any) -> None:
		self.model_name = model_name
		self.model_configurations: Dict[str, Any] = model_configurations

	def generate(self, system_prompt: str, user_prompt: str, image_path: Optional[str] = None, json_schema: Optional[Dict[str, Any]] = None) -> ProviderResponse:
		raise NotImplementedError
