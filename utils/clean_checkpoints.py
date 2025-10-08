#!/usr/bin/env python3
"""
Clean unnecessary files from model checkpoints to save disk space.

Removes RNG states, global_step files, and other training artifacts
while keeping only what's needed for model evaluation.
"""

import argparse
import shutil
from pathlib import Path
from typing import List


# Files/patterns to keep (everything else gets deleted)
KEEP_PATTERNS = [
    # Model files
    "*.safetensors",
    "*.bin",  # For older checkpoints
    "model.safetensors.index.json",

    # Config files
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",

    # Metadata (optional, but small)
    "trainer_state.json",
    "training_args.bin",
]

# Files/patterns to explicitly remove
REMOVE_PATTERNS = [
    "rng_state_*.pth",
    "global_step*",
    "latest",
    "scheduler.pt",
    "optimizer.pt",
]


def should_keep(file_path: Path) -> bool:
    """Check if a file should be kept based on KEEP_PATTERNS."""
    for pattern in KEEP_PATTERNS:
        if file_path.match(pattern):
            return True
    return False


def should_remove(file_path: Path) -> bool:
    """Check if a file should be removed based on REMOVE_PATTERNS."""
    for pattern in REMOVE_PATTERNS:
        if file_path.match(pattern):
            return True
    return False


def get_dir_size(dir_path: Path) -> int:
    """Get total size of a directory recursively."""
    total = 0
    for item in dir_path.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
    return total


def clean_checkpoint(checkpoint_dir: Path, dry_run: bool = True) -> tuple[int, int]:
    """
    Clean a single checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        dry_run: If True, only print what would be deleted

    Returns:
        Tuple of (num_items_removed, bytes_freed)
    """
    if not checkpoint_dir.is_dir():
        return 0, 0

    items_removed = 0
    bytes_freed = 0

    for item_path in checkpoint_dir.iterdir():
        # Check if we should explicitly remove this file/folder
        if should_remove(item_path):
            if item_path.is_file():
                size = item_path.stat().st_size
                if dry_run:
                    print(f"  [DRY RUN] Would remove: {item_path.name} ({size / 1024 / 1024:.2f} MB)")
                else:
                    print(f"  Removing: {item_path.name} ({size / 1024 / 1024:.2f} MB)")
                    item_path.unlink()
                items_removed += 1
                bytes_freed += size
            elif item_path.is_dir():
                size = get_dir_size(item_path)
                if dry_run:
                    print(f"  [DRY RUN] Would remove dir: {item_path.name} ({size / 1024 / 1024:.2f} MB)")
                else:
                    print(f"  Removing dir: {item_path.name} ({size / 1024 / 1024:.2f} MB)")
                    shutil.rmtree(item_path)
                items_removed += 1
                bytes_freed += size

    return items_removed, bytes_freed


def clean_experiment_dir(exp_dir: Path, dry_run: bool = True) -> dict:
    """
    Clean all checkpoints in an experiment directory.

    Args:
        exp_dir: Path to experiment directory
        dry_run: If True, only print what would be deleted

    Returns:
        Dictionary with cleaning statistics
    """
    model_dir = exp_dir / "model"

    if not model_dir.exists():
        print(f"No model directory found in {exp_dir}")
        return {"checkpoints": 0, "files_removed": 0, "bytes_freed": 0}

    print(f"\nCleaning checkpoints in: {exp_dir.name}")

    total_files = 0
    total_bytes = 0
    checkpoint_count = 0

    # Find all checkpoint directories
    checkpoint_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])

    for checkpoint_dir in checkpoint_dirs:
        print(f"\n  Checkpoint: {checkpoint_dir.name}")
        files, bytes_freed = clean_checkpoint(checkpoint_dir, dry_run)
        total_files += files
        total_bytes += bytes_freed
        if files > 0:
            checkpoint_count += 1

    return {
        "checkpoints": checkpoint_count,
        "files_removed": total_files,
        "bytes_freed": total_bytes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean unnecessary files from model checkpoints"
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Work directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--exp-pattern",
        type=str,
        default="*",
        help="Pattern to match experiment directories (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    if not args.work_dir.exists():
        print(f"Error: Work directory not found: {args.work_dir}")
        return

    # Find all matching experiment directories
    exp_dirs = sorted(args.work_dir.glob(args.exp_pattern))
    exp_dirs = [d for d in exp_dirs if d.is_dir()]

    if not exp_dirs:
        print(f"No experiment directories found matching pattern: {args.exp_pattern}")
        return

    print(f"{'='*60}")
    print(f"CHECKPOINT CLEANING {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'='*60}")
    print(f"Work directory: {args.work_dir}")
    print(f"Found {len(exp_dirs)} experiment directories")
    print(f"\nFiles to remove: {', '.join(REMOVE_PATTERNS)}")

    # Clean each experiment directory
    total_stats = {
        "experiments": 0,
        "checkpoints": 0,
        "files_removed": 0,
        "bytes_freed": 0
    }

    for exp_dir in exp_dirs:
        stats = clean_experiment_dir(exp_dir, args.dry_run)
        if stats["files_removed"] > 0:
            total_stats["experiments"] += 1
            total_stats["checkpoints"] += stats["checkpoints"]
            total_stats["files_removed"] += stats["files_removed"]
            total_stats["bytes_freed"] += stats["bytes_freed"]

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Experiments cleaned: {total_stats['experiments']}")
    print(f"Checkpoints processed: {total_stats['checkpoints']}")
    print(f"Files removed: {total_stats['files_removed']}")
    print(f"Space freed: {total_stats['bytes_freed'] / 1024 / 1024 / 1024:.2f} GB")

    if args.dry_run:
        print(f"\n⚠️  This was a DRY RUN. No files were actually deleted.")
        print(f"Run without --dry-run to actually delete files.")
    else:
        print(f"\n✓ Cleanup complete!")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
