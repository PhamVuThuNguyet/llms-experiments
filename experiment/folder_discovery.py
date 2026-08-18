"""Shared helpers for discovering experiment folders and image files."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"})


def find_image_in_folder(folder_path: Path) -> str | None:
    """Return the first image path in a folder, or None if none exist."""
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            return str(file_path)
    return None


def list_images_in_folder(folder_path: Path) -> list[Path]:
    """Return sorted image files in a folder."""
    return sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_numeric_experiment_folders(data_dir: Path, start_from: int = 0) -> list[Path]:
    """Collect numeric experiment folders that contain at least one image."""
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return []

    experiment_folders: list[Path] = []
    for item in data_dir.iterdir():
        if not item.is_dir():
            continue
        if not item.name.isdigit():
            print(f"Warning: Skipping non-numeric folder {item.name}")
            continue
        if int(item.name) < start_from:
            continue
        if find_image_in_folder(item):
            experiment_folders.append(item)
        else:
            print(f"Warning: No image found in folder {item.name}")

    return sorted(experiment_folders, key=lambda path: int(path.name))


def get_named_experiment_folders(data_dir: Path, start_from: int = 0) -> list[Path]:
    """Collect any subfolders that contain at least one image."""
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return []

    experiment_folders: list[Path] = []
    for item in data_dir.iterdir():
        if not item.is_dir():
            continue
        if find_image_in_folder(item):
            experiment_folders.append(item)
        else:
            print(f"Warning: No image found in folder {item.name}")

    experiment_folders = sorted(experiment_folders, key=lambda path: path.name)
    if start_from > 0:
        experiment_folders = experiment_folders[start_from:]
    return experiment_folders


def require_paths(paths: list[Path]) -> bool:
    """Print missing paths and return False if any are missing."""
    missing = [path for path in paths if not path.exists()]
    for path in missing:
        print(f"Error: Required file/directory not found: {path}")
    return not missing
