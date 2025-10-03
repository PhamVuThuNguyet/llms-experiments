from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from providers.anthropic_provider import AnthropicProvider
from providers.base import BaseProvider
from providers.gemini_provider import GeminiProvider
from providers.grok_provider import GrokProvider
from providers.openai_provider import OpenAIProvider
from utils.io import load_json, load_yaml, write_json
from utils.logger import CallLog, JsonlLogger
import csv

MODEL_SPECS: List[Tuple[str, str]] = [
    ("openai", "gpt-4.1"),
    ("openai", "gpt-4o"),
    ("openai", "o1"),
    ("openai", "gpt-5"),
    ("openai", "gpt-5-chat-latest"),
    ("anthropic", "claude-opus-4-1-20250805"),
    ("anthropic", "claude-opus-4-20250514"),
    ("anthropic", "claude-3-7-sonnet-latest"),
    ("anthropic", "claude-sonnet-4-20250514"),
    ("grok", "grok-4-fast-reasoning"),
    # ("gemini", "gemini-2.5-pro"),
]


def create_provider(
    provider: str, model: str, model_configurations: Dict[str, Any]
) -> BaseProvider:
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
    prompt_text: str,
    system_text: str,
    image_path: Path,
    json_template: Optional[dict[str, Any]],
    model_overrides: dict[str, Any],
    study_uid: str,
    slice_number: str
) -> Dict[str, List[dict[str, Any]]]:
    """
    Run all models on a single image and return a dict per model with all relevant info.
    """
    applied_config = model_overrides or {}
    model_rows: dict[str, List[dict[str, Any]]] = {}

    for provider_name, model_name in MODEL_SPECS:
        provider = create_provider(provider_name, model_name, applied_config)
        response = provider.generate(
            system_prompt=system_text or "",
            user_prompt=prompt_text,
            image_path=str(image_path),
            json_schema=json_template,
        )

        row = {
            "StudyInstanceUID": study_uid,
            "slice_number": slice_number,
            "response_text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "input_chars": response.input_chars,
            "output_chars": response.output_chars,
            "ttft_ms": response.ttft_ms,
            "total_latency_ms": response.total_latency_ms,
            "http_status": response.http_status,
            "error_category": response.error_category,
            "response_params": json.dumps(response.response_params) if response.response_params else None,
        }

        model_rows.setdefault(model_name, []).append(row)

    return model_rows


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run LLM experiments across providers."
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to YAML with user prompts {id:{ver:text}}",
    )
    parser.add_argument(
        "--systems",
        required=True,
        help="Path to YAML with system prompts {id:{ver:text}}",
    )
    parser.add_argument(
        "--json_template", required=False, help="Optional JSON schema/template file"
    )
    parser.add_argument("--prompt_id", required=True, help="Prompt ID to run")
    parser.add_argument("--prompt_ver", required=True, help="Prompt version key")
    parser.add_argument("--system_id", required=True, help="System ID to run")
    parser.add_argument("--system_ver", required=True, help="System version key")
    parser.add_argument("--image", required=False, help="Optional image path")
    parser.add_argument("--experiment_id", required=True, help="Experiment ID label")
    parser.add_argument(
        "--model_overrides",
        required=False,
        help="JSON for model configuration overrides",
    )

    args = parser.parse_args()

    prompts = read_prompts(Path(args.prompts))
    systems = read_systems(Path(args.systems))
    prompt_text = prompts[args.prompt_id][args.prompt_ver]
    system_text = systems[args.system_id][args.system_ver]
    json_template = read_json_template(Path(args.json_template)) if args.json_template else None
    model_overrides = json.loads(args.model_overrides) if args.model_overrides else {}

    image_path = Path(args.image)
    burst_column = [
        "StudyInstanceUID",
        "slice_number",
        "response_text",
        "input_tokens",
        "output_tokens",
        "input_chars",
        "output_chars",
        "ttft_ms",
        "total_latency_ms",
        "http_status",
        "error_category",
        "response_params",
    ]

    # Prepare CSV writers per model
    writers: dict[str, csv.DictWriter] = {}
    files: dict[str, any] = {}
    output_dir = Path("output") / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, model_name in MODEL_SPECS:
        csv_path = output_dir / f"{model_name}.csv"
        f = open(csv_path, "w", newline="", encoding="utf-8", buffering=1)  # line-buffered
        writer = csv.DictWriter(f, fieldnames=burst_column)
        writer.writeheader()
        writers[model_name] = writer
        files[model_name] = f

    if image_path.is_dir():
        for study_folder in image_path.iterdir():
            if study_folder.is_dir():
                study_uid = study_folder.name
                for img_file in study_folder.glob("*.png"):
                    slice_number = img_file.name
                    print(f"Processing {study_uid}/{slice_number}")
                    result = run_task(prompt_text, system_text, img_file, json_template, model_overrides, study_uid, slice_number)
                    # Write rows per model and flush immediately
                    for model_name, rows in result.items():
                        writers[model_name].writerows(rows)
                        files[model_name].flush()
    else:
        raise ValueError(f"Image path must be a directory, got {image_path}")

    # Close all files
    for f in files.values():
        f.close()

    print(f"Results saved incrementally in {output_dir}")


if __name__ == "__main__":
    main()
