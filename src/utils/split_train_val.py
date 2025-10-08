#!/usr/bin/env python3
"""
Split a JSONL file into train and validation sets.

Takes a JSONL file, randomly selects n_val examples for validation,
and saves the rest as training data with _train and _val suffixes.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def split_train_val(
    input_path: str,
    n_val: int,
    random_seed: int = 42,
) -> tuple[str, str]:
    """
    Split a JSONL file into train and validation sets.

    Args:
        input_path: Path to input JSONL file
        n_val: Number of examples for validation set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_path, val_path)
    """
    # Load data
    print(f"Loading data from {input_path}...")
    data = load_jsonl(input_path)
    total = len(data)
    print(f"Loaded {total} examples")

    # Validate n_val
    if n_val >= total:
        raise ValueError(f"n_val ({n_val}) must be less than total examples ({total})")

    if n_val <= 0:
        raise ValueError(f"n_val must be positive, got {n_val}")

    # Shuffle data
    random.seed(random_seed)
    random.shuffle(data)

    # Split into train and val
    val_data = data[:n_val]
    train_data = data[n_val:]

    # Generate output paths
    input_path_obj = Path(input_path)
    stem = input_path_obj.stem  # filename without extension
    suffix = input_path_obj.suffix  # .jsonl
    parent = input_path_obj.parent

    train_path = parent / f"{stem}_train{suffix}"
    val_path = parent / f"{stem}_val{suffix}"

    # Save splits
    print(f"\nSaving train set ({len(train_data)} examples) to {train_path}")
    save_jsonl(train_data, str(train_path))

    print(f"Saving val set ({len(val_data)} examples) to {val_path}")
    save_jsonl(val_data, str(val_path))

    # Print summary
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples: {total}")
    print(f"Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"Val: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"\nTrain file: {train_path}")
    print(f"Val file: {val_path}")
    print(f"{'='*60}")

    return str(train_path), str(val_path)


def main():
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into train and validation sets"
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        required=True,
        help="Number of examples for validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Perform split
    split_train_val(
        input_path=args.input_path,
        n_val=args.n_val,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
