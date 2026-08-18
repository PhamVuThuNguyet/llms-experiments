#!/usr/bin/env python3
"""
Run triage (or similar single-image folder) experiments across providers.
Each numeric subfolder is one experiment, repeated a configurable number of times.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from experiment.folder_discovery import (
    find_image_in_folder,
    get_numeric_experiment_folders,
    require_paths,
)
from experiment.runner import read_json_template, read_prompts, read_systems, run_task
from experiment.utils.logger import CallProgressLogger


async def run_triage_experiments(
    data_dir: Path,
    prompts_file: Path,
    prompt_task: str,
    prompt_version: str,
    systems_file: Path,
    json_template_file: Path,
    output_folder: str,
    num_runs: int = 3,
    start_from: int = 0,
    prompt_id: str = "triage",
) -> None:
    """Run single-image experiments on numeric subfolders in the data directory."""
    print("Loading prompts and systems...")
    prompts = read_prompts(prompts_file)
    systems = read_systems(systems_file)
    json_template = read_json_template(json_template_file)

    prompt_text = prompts[prompt_task][prompt_version]
    system_text = systems["general"]["default"]

    print(f"System prompt: {system_text[:100]}...")
    print(f"User prompt: {prompt_text[:100]}...")

    experiment_folders = get_numeric_experiment_folders(data_dir, start_from)
    print(
        f"Found {len(experiment_folders)} experiment folders "
        f"(starting from experiment {start_from})"
    )
    if not experiment_folders:
        print("No experiment folders found. Exiting.")
        return

    logger = CallProgressLogger()
    total_experiments = len(experiment_folders) * num_runs
    current_experiment = 0

    for folder in experiment_folders:
        experiment_id = folder.name
        image_path = find_image_in_folder(folder)
        if not image_path:
            print(f"Skipping folder {experiment_id} - no image found")
            continue

        print(
            f"\nRunning experiment {experiment_id} "
            f"({current_experiment + 1}/{total_experiments})"
        )
        print(f"Image path: {image_path}")

        for run_num in range(num_runs):
            print(f"  Run {run_num + 1}/{num_runs}")
            try:
                await run_task(
                    experiment_id=experiment_id,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    system_text=system_text,
                    image_path=image_path,
                    json_template=json_template,
                    output_folder=output_folder,
                    logger=logger,
                    model_overrides={},
                )
                print(f"  ✓ Run {run_num + 1} completed successfully")
            except Exception as error:
                print(f"  ✗ Run {run_num + 1} failed: {error}")
            current_experiment += 1

    print(f"\nCompleted {current_experiment} experiments total")
    print("Results saved to output directory")


def main():
    """CLI entry point for triage experiments."""
    load_dotenv()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "triage-data"
    prompts_file = script_dir / "prompts" / "user_prompts.sample.yaml"
    systems_file = script_dir / "prompts" / "system_prompts.sample.yaml"
    json_template_file = script_dir / "schemas" / "triage_template.json"
    required = [data_dir, prompts_file, systems_file, json_template_file]
    if not require_paths(required):
        return

    print("Starting triage experiments...")
    print(f"Data directory: {data_dir}")
    print(f"Prompts file: {prompts_file}")
    print(f"Systems file: {systems_file}")
    print(f"JSON template: {json_template_file}")

    asyncio.run(
        run_triage_experiments(
            data_dir=data_dir,
            prompts_file=prompts_file,
            prompt_task="triage",
            prompt_version="triage_v1",
            systems_file=systems_file,
            json_template_file=json_template_file,
            output_folder="triage-v2",
            num_runs=1,
            start_from=0,
        )
    )


if __name__ == "__main__":
    main()
