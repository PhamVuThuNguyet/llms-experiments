from __future__ import annotations

from dataclasses import asdict, dataclass
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
    http_status: Optional[int]
    error_category: Optional[str]
    error_message: Optional[str] = None
    # Sampling params (if available)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # Misc
    experiment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JsonlLogger:
    def write(self, call_log: CallLog) -> None:
        print(
            Fore.GREEN
            + f"Logged {call_log.model_provider}:{call_log.model_name} prompt={call_log.prompt_id} status={call_log.http_status}"
            + Style.RESET_ALL
        )
