#!/usr/bin/env python3
"""
Script to run burst-fracture experiments on all subfolders in the bf-data
folder. Each subfolder represents a single experiment, and each experiment is
run a configurable number of times.
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the experiment directory to the path so we can import from runner.py
sys.path.append(str(Path(__file__).parent / "experiment"))

from experiment.runner import read_json_template, read_prompts, read_systems, run_task
from experiment.utils.logger import JsonlLogger
from run_experiments import find_image_in_folder


def get_all_bf_experiment_folders(data_dir: Path, start_from: int = 0) -> list[Path]:
    """
    Get all subfolders in the bf-data directory that contain experiment data.

    Unlike the MLS/triage experiments, bf-data folders do not have to follow a
    numeric naming convention. Any subfolder that contains at least one image
    file will be treated as an experiment folder.
    """
    experiment_folders: list[Path] = []

    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return experiment_folders

    for item in data_dir.iterdir():
        if item.is_dir():
            image_path = find_image_in_folder(item)
            if image_path:
                experiment_folders.append(item)
            else:
                print(f"Warning: No image found in folder {item.name}")

    experiment_folders = sorted(experiment_folders, key=lambda p: p.name)

    if start_from > 0:
        experiment_folders = experiment_folders[start_from:]

    return experiment_folders


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
    """
    Run burst-fracture experiments on all subfolders in the data directory.
    """
    print("Loading prompts and systems...")
    prompts = read_prompts(prompts_file)
    systems = read_systems(systems_file)
    json_template = read_json_template(json_template_file)

    prompt_text = prompts[prompt_task][prompt_version]
    system_text = systems["general"]["default"]

    print(f"System prompt: {system_text[:100]}...")
    print(f"User prompt: {prompt_text[:100]}...")

    experiment_folders = get_all_bf_experiment_folders(data_dir, start_from)
    print(
        f"Found {len(experiment_folders)} experiment folders "
        f"(starting from experiment {start_from})"
    )

    if not experiment_folders:
        print("No experiment folders found. Exiting.")
        return

    logger = JsonlLogger()

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    image_jobs: list[tuple[Path, Path]] = []

    for folder in experiment_folders:
        for file_path in sorted(folder.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_jobs.append((folder, file_path))

    if not image_jobs:
        print("No images found in any BF experiment folder. Exiting.")
        return

    total_experiments = len(image_jobs) * num_runs
    current_experiment = 0

    for folder, image_file in image_jobs:
        experiment_id = folder.name
        image_path = str(image_file)

        # Derive a per-image subfolder name from the slice filename (e.g., "slice_381")
        slice_name = image_file.stem
        experiment_id_with_slice = f"{experiment_id}/{slice_name}"

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
            except Exception as e:
                print(f"  ✗ Run {run_num + 1} failed: {e}")

            current_experiment += 1

    print(f"\nCompleted {current_experiment} BF experiments total")
    print("Results saved to output directory")


def main():
    """Main function to run burst-fracture experiments."""
    load_dotenv()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "bf-data"
    prompts_file = script_dir / "prompts" / "user_prompts.sample.yaml"
    prompt_task = "structured_findings_extraction"
    prompt_version = "bf_v1"
    systems_file = script_dir / "prompts" / "system_prompts.sample.yaml"
    json_template_file = script_dir / "schemas" / "bf_template.json"
    output_folder = "bf-v2"

    required_files = [data_dir, prompts_file, systems_file, json_template_file]
    for file_path in required_files:
        if not file_path.exists():
            print(f"Error: Required file/directory not found: {file_path}")
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
            prompt_task=prompt_task,
            prompt_version=prompt_version,
            systems_file=systems_file,
            json_template_file=json_template_file,
            output_folder=output_folder,
            num_runs=1,
            start_from=0,
        )
    )


if __name__ == "__main__":
    main()
