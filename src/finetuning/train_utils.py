import logging
import torch
import wandb
import json
import os
from pathlib import Path
from datasets import Dataset
import subprocess
from typing import Optional
import sys

logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color coding based on log level"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset to default color

    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)

        # Add color if this is going to a terminal
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, "")
            if color:
                formatted = f"{color}{formatted}{self.RESET}"

        return formatted


def save_git_hash(file_output_path: Path):
    """Get current git commit hash and save it to file"""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(), universal_newlines=True
        ).strip()
        with open(file_output_path, "w") as f:
            f.write(git_hash)

        logger.info(f"Git hash saved to: {file_output_path}")
        logger.info(f"Commit hash: {git_hash}")
    except subprocess.CalledProcessError:
        logger.error("Error: Not in a git repository or git not found")
    except FileNotFoundError:
        logger.error("Error: Git command not found")








def init_wandb(
    exp_name: str,
    model_name: str,
    epochs: int,
    per_device_train_batch_size: int,
    dataset: Dataset,
    train_data_path: Path,
    lr: float,
    exp_family: str,
):
    dataset_stem = train_data_path.stem
    wandb.init(
        project=exp_family,
        name=exp_name,
        config={
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": per_device_train_batch_size,
            "learning_rate": lr,
            "dataset_size": len(dataset),
            "train_data_path": str(train_data_path),
            "dataset_stem": dataset_stem,
        },
        tags=["training", "sft", model_name.split("/")[-1]],
    )






def get_local_rank():
    """Get local rank from environment variables set by torchrun/deepspeed"""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    """Check if this is the main process (rank 0)"""
    return get_local_rank() == 0


def setup_logging(local_rank: int, exp_dir: Optional[Path] = None):
    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")

    if local_rank == 0:
        # File handler (no colors)
        if exp_dir is not None:
            file_handler = logging.FileHandler(exp_dir / "train.log")
            file_handler.setFormatter(file_formatter)

        # Console handler (with colors)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)

        logging.basicConfig(
            level=logging.INFO,
            handlers=(
                [file_handler, console_handler]
                if exp_dir is not None
                else [console_handler]
            ),
        )
        logger.info(f"Local rank: {local_rank}")
        logger.info(f"Experiment directory: {exp_dir}")
    else:
        # File handler (no colors)
        if exp_dir is not None:
            file_handler = logging.FileHandler(exp_dir / "train.log")
            file_handler.setFormatter(file_formatter)

        # Console handler (with colors)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)

        logging.basicConfig(
            level=logging.WARNING,
            handlers=(
                [file_handler, console_handler]
                if exp_dir is not None
                else [console_handler]
            ),
        )


