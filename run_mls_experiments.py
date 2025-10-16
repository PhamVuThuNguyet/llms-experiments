#!/usr/bin/env python3
"""
Script to run MLS (Midline Shift) experiments on all subfolders in data/mls-data.
Each subfolder represents a single experiment, and each experiment is run 3 times.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add the experiment directory to the path so we can import from runner.py
sys.path.append(str(Path(__file__).parent / "experiment"))

from dotenv import load_dotenv
from experiment.runner import read_json_template, read_prompts, read_systems, run_task
from experiment.utils.logger import JsonlLogger


def find_image_in_folder(folder_path: Path) -> Optional[str]:
    """
    Find the image file in a folder. Assumes there's only one image file per folder.
    Returns the full path to the image file, or None if not found.
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            return str(file_path)

    return None


def get_all_experiment_folders(data_dir: Path, start_from: int = 0) -> List[Path]:
    """
    Get all subfolders in the mls-data directory that contain experiment data.
    
    Args:
        data_dir: Path to the data directory
        start_from: Only include experiments with ID >= start_from (default: 0)
    """
    experiment_folders = []

    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return experiment_folders

    for item in data_dir.iterdir():
        if item.is_dir():
            # Check if folder name is a valid experiment ID and meets the start_from criteria
            if item.name.isdigit():
                experiment_id = int(item.name)
                if experiment_id >= start_from:
                    # Check if folder contains an image
                    image_path = find_image_in_folder(item)
                    if image_path:
                        experiment_folders.append(item)
                    else:
                        print(f"Warning: No image found in folder {item.name}")
            else:
                print(f"Warning: Skipping non-numeric folder {item.name}")

    return sorted(
        experiment_folders,
        key=lambda x: int(x.name) if x.name.isdigit() else float("inf"),
    )


async def run_mls_experiments(
    data_dir: Path,
    prompts_file: Path,
    systems_file: Path,
    json_template_file: Path,
    num_runs: int = 3,
    start_from: int = 0,
) -> None:
    """
    Run MLS experiments on all subfolders in the data directory.

    Args:
        data_dir: Path to the mls-data directory
        prompts_file: Path to the user prompts YAML file
        systems_file: Path to the system prompts YAML file
        json_template_file: Path to the JSON template file
        num_runs: Number of times to run each experiment (default: 3)
        start_from: Only run experiments with ID >= start_from (default: 0)
    """
    # Load prompts and systems
    print("Loading prompts and systems...")
    prompts = read_prompts(prompts_file)
    systems = read_systems(systems_file)
    json_template = read_json_template(json_template_file)

    # Extract the specific prompts we need
    prompt_text = prompts["structured_findings_extraction"]["mls_v1"]
    system_text = systems["general"]["default"]

    print(f"System prompt: {system_text[:100]}...")
    print(f"User prompt: {prompt_text[:100]}...")

    # Get all experiment folders
    experiment_folders = get_all_experiment_folders(data_dir, start_from)
    print(f"Found {len(experiment_folders)} experiment folders (starting from experiment {start_from})")

    if not experiment_folders:
        print("No experiment folders found. Exiting.")
        return

    # Initialize logger
    logger = JsonlLogger()

    total_experiments = len(experiment_folders) * num_runs
    current_experiment = 0

    # Run experiments
    for folder in experiment_folders:
        experiment_id = folder.name
        image_path = find_image_in_folder(folder)

        if not image_path:
            print(f"Skipping folder {experiment_id} - no image found")
            continue

        print(
            f"\nRunning experiment {experiment_id} ({current_experiment + 1}/{total_experiments})"
        )
        print(f"Image path: {image_path}")

        # Run the experiment num_runs times
        for run_num in range(num_runs):
            print(f"  Run {run_num + 1}/{num_runs}")

            try:
                await run_task(
                    experiment_id=experiment_id,
                    prompt_id="structured_findings_extraction",
                    prompt_text=prompt_text,
                    system_text=system_text,
                    image_path=image_path,
                    json_template=json_template,
                    logger=logger,
                    model_overrides={},
                )
                print(f"  ✓ Run {run_num + 1} completed successfully")
            except Exception as e:
                print(f"  ✗ Run {run_num + 1} failed: {e}")

            current_experiment += 1

    print(f"\nCompleted {current_experiment} experiments total")
    print("Results saved to output directory")


def main():
    """Main function to run MLS experiments."""
    load_dotenv()

    # Define paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "mls-data"
    prompts_file = script_dir / "prompts" / "user_prompts.sample.yaml"
    systems_file = script_dir / "prompts" / "system_prompts.sample.yaml"
    json_template_file = script_dir / "schemas" / "mls_template.json"

    # Verify all required files exist
    required_files = [data_dir, prompts_file, systems_file, json_template_file]
    for file_path in required_files:
        if not file_path.exists():
            print(f"Error: Required file/directory not found: {file_path}")
            return

    print("Starting MLS experiments...")
    print(f"Data directory: {data_dir}")
    print(f"Prompts file: {prompts_file}")
    print(f"Systems file: {systems_file}")
    print(f"JSON template: {json_template_file}")

    # Run experiments
    asyncio.run(
        run_mls_experiments(
            data_dir=data_dir,
            prompts_file=prompts_file,
            systems_file=systems_file,
            json_template_file=json_template_file,
            num_runs=1,
            start_from=313,
        )
    )


if __name__ == "__main__":
    main()
