#!/usr/bin/env python3
"""
Run burst-fracture experiments across providers.
Each study folder may contain multiple slice images; each image is one job.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from experiment.folder_discovery import (
    get_named_experiment_folders,
    list_images_in_folder,
    require_paths,
)
from experiment.runner import read_json_template, read_prompts, read_systems, run_task
from experiment.utils.logger import CallProgressLogger


def build_bf_image_jobs(experiment_folders: list[Path]) -> list[tuple[Path, Path]]:
    image_jobs: list[tuple[Path, Path]] = []
    for folder in experiment_folders:
        for image_file in list_images_in_folder(folder):
            image_jobs.append((folder, image_file))
    return image_jobs


async def run_bf_experiments(
    data_dir: Path,
    prompts_file: Path,
    prompt_task: str,
    prompt_version: str,
    systems_file: Path,
    json_template_file: Path,
    output_folder: str,
    num_runs: int = 3,
    start_from: int = 0,
) -> None:
    """Run burst-fracture experiments on all subfolders in the data directory."""
    print("Loading prompts and systems...")
    prompts = read_prompts(prompts_file)
    systems = read_systems(systems_file)
    json_template = read_json_template(json_template_file)

    prompt_text = prompts[prompt_task][prompt_version]
    system_text = systems["general"]["default"]

    print(f"System prompt: {system_text[:100]}...")
    print(f"User prompt: {prompt_text[:100]}...")

    experiment_folders = get_named_experiment_folders(data_dir, start_from)
    print(
        f"Found {len(experiment_folders)} experiment folders "
        f"(starting from experiment {start_from})"
    )
    if not experiment_folders:
        print("No experiment folders found. Exiting.")
        return

    image_jobs = build_bf_image_jobs(experiment_folders)
    if not image_jobs:
        print("No images found in any BF experiment folder. Exiting.")
        return

    logger = CallProgressLogger()
    total_experiments = len(image_jobs) * num_runs
    current_experiment = 0

    for folder, image_file in image_jobs:
        experiment_id = folder.name
        slice_name = image_file.stem
        experiment_id_with_slice = f"{experiment_id}/{slice_name}"
        image_path = str(image_file)

        print(
            f"\nRunning BF experiment {experiment_id_with_slice} "
            f"({current_experiment + 1}/{total_experiments})"
        )
        print(f"Image path: {image_path}")

        for run_num in range(num_runs):
            print(f"  Run {run_num + 1}/{num_runs}")
            try:
                await run_task(
                    experiment_id=experiment_id_with_slice,
                    prompt_id="structured_findings_extraction",
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

    print(f"\nCompleted {current_experiment} BF experiments total")
    print("Results saved to output directory")


def main():
    """CLI entry point for burst-fracture experiments."""
    load_dotenv()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "bf-data"
    prompts_file = script_dir / "prompts" / "user_prompts.sample.yaml"
    systems_file = script_dir / "prompts" / "system_prompts.sample.yaml"
    json_template_file = script_dir / "schemas" / "bf_template.json"
    required = [data_dir, prompts_file, systems_file, json_template_file]
    if not require_paths(required):
        return

    print("Starting burst-fracture experiments...")
    print(f"Data directory: {data_dir}")
    print(f"Prompts file: {prompts_file}")
    print(f"Systems file: {systems_file}")
    print(f"JSON template: {json_template_file}")

    asyncio.run(
        run_bf_experiments(
            data_dir=data_dir,
            prompts_file=prompts_file,
            prompt_task="structured_findings_extraction",
            prompt_version="bf_v1",
            systems_file=systems_file,
            json_template_file=json_template_file,
            output_folder="bf-v2",
            num_runs=1,
            start_from=0,
        )
    )


if __name__ == "__main__":
    main()
