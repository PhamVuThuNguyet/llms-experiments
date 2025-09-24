from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from experiment.providers.anthropic_provider import AnthropicProvider
from experiment.providers.base import BaseProvider
from experiment.providers.gemini_provider import GeminiProvider
from experiment.providers.grok_provider import GrokProvider
from experiment.providers.openai_provider import OpenAIProvider
from experiment.utils.io import ensure_dir, load_json, load_yaml
from experiment.utils.logger import CallLog, JsonlLogger

MODEL_SPECS: List[Tuple[str, str]] = [
	("openai", "gpt-4o"),
	("openai", "gpt-5-thinking"),
	("anthropic", "claude-4.1-opus"),
	("anthropic", "claude-4-opus"),
	("anthropic", "claude-3.7-sonnet"),
	("anthropic", "claude-4-sonnet"),
	("grok", "grok-4-fast-thinking"),
	("gemini", "gemini-2.5-pro"),
]


def create_provider(provider: str, model: str, model_configurations: Dict[str, Any]) -> BaseProvider:
	if provider == "openai":
		return OpenAIProvider(model, **model_configurations)
	if provider == "anthropic":
		return AnthropicProvider(model, **model_configurations)
	if provider == "grok":
		return GrokProvider(model, **model_configurations)
	if provider == "gemini":
		return GeminiProvider(model, **model_configurations)
	raise ValueError(f"Unknown provider: {provider}")


def read_prompts(prompt_yaml: Path) -> Dict[str, Dict[str, str]]:
	# Expected schema: { prompt_id: { version_name: text } }
	return load_yaml(prompt_yaml)


def read_systems(system_yaml: Path) -> Dict[str, Dict[str, str]]:
	# Expected schema: { system_id: { version_name: text } }
	return load_yaml(system_yaml)


def read_json_template(path: Optional[Path]) -> Optional[Dict[str, Any]]:
	if not path:
		return None
	return load_json(path)


def run_task(
	experiment_id: str,
	prompt_id: str,
	prompt_text: str,
	system_text: str,
	image_path: Optional[str],
	json_template: Optional[Dict[str, Any]],
	logger: JsonlLogger,
	model_overrides: Dict[str, Any],
) -> None:
	model_configurations = model_overrides or {}
	for provider_name, model_name in MODEL_SPECS:
		provider = create_provider(provider_name, model_name, model_configurations)
		response = provider.generate(
			system_prompt=system_text or "",
			user_prompt=prompt_text,
			image_path=image_path,
			json_schema=json_template,
		)
		call_log = CallLog(
			prompt_id=prompt_id,
			input_image_path=image_path,
			user_prompt=prompt_text,
			model_provider=provider_name,
			model_name=model_name,
			model_configurations=model_configurations,
			input_chars=response.input_chars,
			input_tokens=response.input_tokens,
			output_chars=response.output_chars,
			output_tokens=response.output_tokens,
			ttft_ms=response.tttft_ms,
			total_latency_ms=response.total_latency_ms,
			response_text=response.text,
			retry_count=0,
			http_status=response.http_status,
			error_category=response.error_category,
			experiment_id=experiment_id,
		)
		logger.write(call_log)


def main() -> None:
	load_dotenv()
	parser = argparse.ArgumentParser(description="Run LLM experiments across providers.")
	parser.add_argument("--prompts", required=True, help="Path to YAML with user prompts {id:{ver:text}}")
	parser.add_argument("--systems", required=True, help="Path to YAML with system prompts {id:{ver:text}}")
	parser.add_argument("--json_template", required=False, help="Optional JSON schema/template file")
	parser.add_argument("--prompt_id", required=True, help="Prompt ID to run")
	parser.add_argument("--prompt_ver", required=True, help="Prompt version key")
	parser.add_argument("--system_id", required=True, help="System ID to run")
	parser.add_argument("--system_ver", required=True, help="System version key")
	parser.add_argument("--image", required=False, help="Optional image path")
	parser.add_argument("--output", required=True, help="Output JSONL log path")
	parser.add_argument("--experiment_id", required=True, help="Experiment ID label")
	parser.add_argument("--model_overrides", required=False, help="JSON for model configuration overrides")

	args = parser.parse_args()

	prompts = read_prompts(Path(args.prompts))
	systems = read_systems(Path(args.systems))
	prompt_text = prompts[args.prompt_id][args.prompt_ver]
	system_text = systems[args.system_id][args.system_ver]
	json_template = read_json_template(Path(args.json_template)) if args.json_template else None

	logger = JsonlLogger(args.output)
	image_path = args.image if args.image else None
	model_overrides = json.loads(args.model_overrides) if args.model_overrides else {}

	run_task(
		experiment_id=args.experiment_id,
		prompt_id=args.prompt_id,
		prompt_text=prompt_text,
		system_text=system_text,
		image_path=image_path,
		json_template=json_template,
		logger=logger,
		model_overrides=model_overrides,
	)


if __name__ == "__main__":
	main()
