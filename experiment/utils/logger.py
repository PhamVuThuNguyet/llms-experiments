from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from colorama import Fore, Style


@dataclass
class CallLog:
	# Inputs
	prompt_id: str
	input_image_path: Optional[str]
	user_prompt: str
	# Model
	model_provider: str
	model_name: str
	model_configurations: Dict[str, Any]
	# Usage
	input_chars: Optional[int]
	input_tokens: Optional[int]
	output_chars: Optional[int]
	output_tokens: Optional[int]
	# Timing
	ttft_ms: Optional[float]
	total_latency_ms: Optional[float]
	# Response and status
	response_text: Optional[str]
	retry_count: int
	http_status: Optional[int]
	error_category: Optional[str]
	# Misc
	experiment_id: Optional[str] = None
	extra: Optional[Dict[str, Any]] = None

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


class JsonlLogger:
	def __init__(self, output_path: str | Path) -> None:
		self.path = Path(output_path)
		self.path.parent.mkdir(parents=True, exist_ok=True)

	def write(self, call_log: CallLog) -> None:
		payload = call_log.to_dict()
		with self.path.open('a', encoding='utf-8') as f:
			f.write(json.dumps(payload, ensure_ascii=False) + "\n")
		print(Fore.GREEN + f"Logged {call_log.model_provider}:{call_log.model_name} prompt={call_log.prompt_id} status={call_log.http_status} retry={call_log.retry_count}" + Style.RESET_ALL)
