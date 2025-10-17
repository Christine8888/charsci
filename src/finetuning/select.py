import json
import warnings
from pathlib import Path
from typing import Literal, Optional


def get_best_model_with_stem(
    base_folder: str,
    stem: str,
    metric: str = 'eval_in_dist_loss',
    mode: Literal['lowest', 'highest', 'second_lowest', 'second_highest'] = 'lowest',
    print_name: bool = False,
    return_full_path: bool = False,
) -> Optional[str]:
    """Find the best model folder based on a metric value.

    Args:
        base_folder: Directory containing model folders
        stem: Prefix that model folders must start with
        metric: Metric name to optimize (default: 'eval_in_dist_loss')
        mode: Whether to find 'lowest' or 'highest'
        print_name: Whether to print the name of the best model

    Returns:
        Path to the best model folder, or None if no valid folders found
    """
    base_path = Path(base_folder)

    if not base_path.exists():
        raise ValueError(f"Base folder does not exist: {base_folder}")

    matching_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith(stem)]

    if not matching_folders:
        warnings.warn(f"No folders found starting with {stem} in {base_folder}")
        return None

    folders = {}

    for folder in matching_folders:
        eval_file = folder / "eval.final"

        if not eval_file.exists():
            warnings.warn(f"eval.final not found in {folder.name}")
            continue

        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)

            if metric not in data:
                warnings.warn(f"Metric '{metric}' not found in {folder.name}/eval.final")
                continue

            value = data[metric]
            folders[folder] = {'metric': value, 'name': folder.name, 'path': folder}

        except (json.JSONDecodeError, IOError) as e:
            warnings.warn(f"Error reading {folder.name}/eval.final: {e}")
            continue

    if folders is None:
        warnings.warn(f"No valid folders found with metric '{metric}'")
        return None
    
    best_folder = folders[get_best_folder(folders, mode)]

    if print_name:
        print(f"Best model: {best_folder['name']} at {best_folder['path']}")

    if return_full_path:
        return str(best_folder['path'])
    else:
        return best_folder['name']

def get_best_folder(folders, mode: Literal['lowest', 'highest', 'second_lowest', 'second_highest']):
    if mode == 'lowest':
        return min(folders.keys(), key=lambda x: folders[x]['metric'])
    elif mode == 'highest':
        return max(folders.keys(), key=lambda x: folders[x]['metric'])
    elif mode == "second_lowest":
        return sorted(folders.keys(), key=lambda x: folders[x]['metric'])[1]
    elif mode == "second_highest":
        return sorted(folders.keys(), key=lambda x: folders[x]['metric'])[-2]
    else:
        raise ValueError(f"Invalid mode: {mode}")