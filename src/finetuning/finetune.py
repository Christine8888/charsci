import pathlib
import shutil
import transformers
from typing import Optional, Tuple, List, Dict
import argparse
import logging
from pathlib import Path
import wandb
import torch
import gc
import os
import dotenv
import time
import json
from pathlib import Path
from datasets import DatasetDict
import re
import pandas as pd
import yaml
import random

from huggingface_hub import login
from trl import SFTTrainer, SFTConfig
from train_utils import (
    init_wandb,
    get_local_rank,
    is_main_process,
    save_git_hash,
    setup_logging,
)
from chat_templates import FIXED_QWEN_TEMPLATE, FIXED_GEMMA_TEMPLATE, FIXED_LLAMA_TEMPLATE
from datasets import Dataset

dotenv.load_dotenv('/workspace/rl-character/safety-tooling/.env')

# We handle parallelism separately and dont need the tokenizer to be parallel
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

logger = logging.getLogger(__name__)

def remove_system_role_messages(messages: List[Dict]):
    """Remove system role messages from a list of messages."""
    return [msg for msg in messages if msg["role"] != "system"]


def load_jsonl_dataset(file_path: Path, tokenizer, max_length: int = 32768, format: str = "messages"):
    """Load JSONL dataset with messages, skip examples exceeding max_length."""

    logger.info(f"Using max_length={max_length} tokens")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if "messages" not in item:
                    logger.warning(f"Skipping: no 'messages' key in item {list(item.keys())}")
                    continue
                data.append(item)

            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON in {file_path}")
                continue

    processed_data, skipped_count = [], 0
    
    for item in data:
        messages = item["messages"]
        
        # Check for system role messages
        has_system_role = any(msg.get("role") == "system" for msg in messages)
        if has_system_role:
            messages = remove_system_role_messages(messages)

        has_assistant_role = any(msg.get("role") == "assistant" for msg in messages)
        if not has_assistant_role:
            logger.warning(f"Skipping sample with no assistant role message")
            continue
    
        # Render just for length check
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer.encode(rendered, add_special_tokens=True)
            if len(tokens) > max_length:
                skipped_count += 1
                continue
        except Exception as e:
            logger.warning(f"Skipping sample due to template error: {e}")
            # print(messages)
            skipped_count += 1
            continue

        if format == "messages":
            # by default, train on all assistant messages
            processed_data.append({"messages": messages})
        elif format == "prompt":
            # train on final assistant message
            processed_data.append({"prompt": messages[:-1], "completion": [messages[-1]]})

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count}/{len(data)} samples exceeding max_length={max_length} or due to template errors")

    return Dataset.from_list(processed_data)

def load_validation_datasets(train_base: str, tokenizer, max_length: int, additional_val_files: List[Path] = None) -> DatasetDict:
    """Load validation datasets from in-distribution file and additional specified files."""
    val_files = []
    
    # Always require in-distribution validation file
    in_dist_path = Path(f"{train_base}_val.jsonl")
    if not in_dist_path.exists():
        raise ValueError(f"In-distribution validation file not found: {in_dist_path}")

    # Add explicitly specified additional validation files first
    if additional_val_files:
        for val_file in additional_val_files:
            if val_file.exists():
                # Extract dataset name from file stem (e.g., dataset_name.jsonl -> dataset_name)
                dataset_name = val_file.stem
                if dataset_name.endswith('_val'):
                    dataset_name = dataset_name[:-4]  # Remove '_val' suffix if present
                val_files.append((dataset_name, val_file))
                logger.info(f"Added additional validation file: {val_file} as '{dataset_name}'")
            else:
                logger.warning(f"Additional validation file not found: {val_file}")

    # Add in-distribution validation file last (so it gets evaluated last)
    val_files.append(("in_dist", in_dist_path))

    logger.info(f"Found {len(val_files)} total validation datasets:")
    
    eval_datasets = {}
    for dataset_name, val_path in val_files:
        logger.info(f"  - {dataset_name}: {val_path}")
        dataset = load_jsonl_dataset(val_path, tokenizer, max_length=max_length)
        eval_datasets[dataset_name] = dataset
    
    return DatasetDict(eval_datasets)


def assistant_token_coverage(ds, tokenizer, name, max_check=200):
    n = len(ds)
    bad = 0
    total_labels = 0
    for i in range(min(n, max_check)):
        msgs = ds[i]["messages"] if "messages" in ds[i] else ds[i]["prompt"]
        toks = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
            truncation=True,
            max_length=args.max_length,
        )
        mask = toks.get("assistant_mask") or toks.get("assistant_masks")
        if mask is None:
            raise RuntimeError("assistant_mask missing — your chat template likely lacks {% generation %}.")
        c = int(sum(mask))
        total_labels += c
        if c == 0:
            bad += 1
    print(f"[{name}] checked={min(n, max_check)} zero-label={bad} total-labels={total_labels}")
    if bad > 0:
        raise RuntimeError(f"{name}: {bad} samples yield ZERO assistant tokens after templating/masking.")


def train_single_model(
    model_name: str,
    train_data_path: Path,
    exp_dir: pathlib.Path,
    name_extension: Optional[str] = None,
    epochs: int = 1,
    per_device_train_batch_size: int = 16,
    val_every: int = 50,
    lr: float = 2e-5,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.1,
    warmup_steps: int = 0,
    wandb_name: str = "rl-character",
    deepspeed_config: Path = Path("./deepspeed.json"),
    max_length: int = 32768,
    gradient_accumulation_steps: int = 4,
    only_train_final: bool = False,
    weight_decay: float = 0.0,
    additional_val_files: List[Path] = None,
    save_steps: int = None,
    save_total_limit: int = None,
):
    finetune_start = time.perf_counter()

    # Generate random seeds for this run
    train_seed = random.randint(0, 2**32 - 1)
    data_seed = random.randint(0, 2**32 - 1)
    logger.info(f"Using random train_seed={train_seed}, data_seed={data_seed}")

    # Validate data path exists
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")

    local_rank = get_local_rank()

    if not is_main_process():
        os.environ["DEEPSPEED_LOG_LEVEL"] = "WARNING"

    # Check if this run is already complete (check as early as possible)
    experiments_dir = exp_dir / name_extension
    done_file = experiments_dir / "done" / "done.train"
    if done_file.exists():
        logger.info(f"Run already completed for {name_extension}, skipping...")
        exit()
    else:
        logger.info(f"Run not completed for {name_extension}, starting...")

    if is_main_process():
        experiments_dir.mkdir(parents=True, exist_ok=True)
        (experiments_dir / "done").mkdir(parents=True, exist_ok=True)

    # Save reference to training data path
    if is_main_process():
        with open(experiments_dir / "train_data_path.txt", "w") as f:
            f.write(str(train_data_path))

        # Save all training hyperparameters to YAML
        train_params = {
            "model_name": model_name,
            "train_data_path": str(train_data_path),
            "exp_dir": str(exp_dir),
            "name_extension": name_extension,
            "epochs": epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "val_every": val_every,
            "lr": lr,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "wandb_name": wandb_name,
            "deepspeed_config": str(deepspeed_config),
            "max_length": max_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "only_train_final": only_train_final,
            "weight_decay": weight_decay,
            "additional_val_files": [str(f) for f in (additional_val_files or [])],
            "train_seed": train_seed,
            "data_seed": data_seed,
        }
        with open(experiments_dir / "train_params.yaml", "w") as f:
            yaml.dump(train_params, f, default_flow_style=False, sort_keys=False)
    
    git_hash_path = experiments_dir / "git_hash.txt"
    if git_hash_path.exists():
        logger.info(f"Git hash already exists at {git_hash_path}, OVERWRITING")
    save_git_hash(git_hash_path)

    # Load model without device_map to allow DeepSpeed to handle device placement
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,  # Let DeepSpeed handle device placement
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if 'qwen' in model_name.lower():
        logger.info("Using modified Qwen template")
        tokenizer.chat_template = FIXED_QWEN_TEMPLATE
    elif 'gemma' in model_name.lower():
        logger.info("Using modified gemma template")
        tokenizer.chat_template = FIXED_GEMMA_TEMPLATE
    elif 'llama' in model_name.lower():
        logger.info("Using modified llama template")
        tokenizer.chat_template = FIXED_LLAMA_TEMPLATE

    model_load_end = time.perf_counter()

    # Load training dataset
    logger.info(f"Loading training data from {train_data_path}")

    if only_train_final:
        train_dataset = load_jsonl_dataset(train_data_path, tokenizer, max_length=max_length, format = "prompt")
    else:
        train_dataset = load_jsonl_dataset(train_data_path, tokenizer, max_length=max_length)

    # Get the base name without '_train.jsonl'
    train_base = str(train_data_path).replace('_train.jsonl', '')
    
    # Load validation datasets
    val_dataset = load_validation_datasets(train_base, tokenizer, max_length, additional_val_files)
    
    dataset_load_end = time.perf_counter()
    experiment_name = f"{name_extension}"

    assistant_token_coverage(train_dataset, tokenizer, "train")

    if is_main_process():
        init_wandb(
            experiment_name,
            model_name,
            epochs,
            per_device_train_batch_size,
            train_dataset,
            train_data_path,
            lr,
            wandb_name,
        )

    if is_main_process():
        logger.info(f"Running experiment {experiment_name}")
        logger.info(f"Local rank: {local_rank}")
        logger.info(f"Dataset size: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation dataset size: {len(val_dataset)}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            assistant_only_loss=True,
            dataset_text_field="messages" if not only_train_final else "prompt",
            max_length=max_length,
            output_dir=str(experiments_dir / "model"),
            num_train_epochs=epochs,
            save_strategy="steps" if save_steps is not None else "no",
            save_steps=save_steps,
            save_only_model=True,
            save_total_limit=save_total_limit,
            learning_rate=lr,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            per_device_train_batch_size=per_device_train_batch_size,

            # Evaluation settings - eval at step 0 and then every val_every steps
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=val_every if val_dataset else None,
            eval_on_start=True if val_dataset else False,  # Evaluate before training
            per_device_eval_batch_size=per_device_train_batch_size,
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            bf16=True,
            max_grad_norm=2.0,
            weight_decay=weight_decay,
            deepspeed=str(deepspeed_config),
            remove_unused_columns=False,
            report_to="wandb" if is_main_process() else "none",
            logging_dir=str(experiments_dir / "logs"),
            log_level="warning",
            log_on_each_node=False,
            run_name=experiment_name,
            logging_steps=1,
            logging_first_step=True,
            dataloader_num_workers=(
                min(8, os.cpu_count()) if os.cpu_count() else 4
            ),
            local_rank=local_rank,
            ddp_backend="nccl",
            completion_only_loss=only_train_final if only_train_final else None, # only use completion-only loss if only_train_final is True
            # Randomization settings
            seed=train_seed,  # Random seed each run
            data_seed=data_seed,  # Random data seed each run
            dataloader_drop_last=False,  # Keep all data
        ),
    )

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        # Clean up memory before re-raising
        torch.cuda.empty_cache()
        gc.collect()
        raise
    
    train_end = time.perf_counter()
    
    pd.DataFrame(trainer.state.log_history).to_csv(
        experiments_dir / "train_history.csv"
    )

    final_model_path = experiments_dir / "final-model"
    # DeepSpeed model saving - only on main process
    if is_main_process():
        final_model_path.mkdir(parents=True, exist_ok=True)
        # For DeepSpeed, use trainer.save_model() which handles distributed state properly
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
    
    model_save_end = time.perf_counter()
    
    # Run final evaluation if validation dataset exists
    if val_dataset is not None:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        
        if is_main_process():
            logger.info(f"Final evaluation metrics: {eval_metrics}")
            
            # Save all evaluation metrics to eval.final file as JSON
            with open(experiments_dir / "eval.final", "w") as f:
                json.dump(eval_metrics, f, indent=2)

    # Wait for main process to finish saving
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if is_main_process():
        wandb.finish()  # Properly close the W&B run
    
    # Clean up - let DeepSpeed handle cleanup properly
    del trainer
    del model
    del tokenizer
    if is_main_process():
        (experiments_dir / "done" / "done.train").touch()

    # Force garbage collection and clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    
    cleanup_end = time.perf_counter()
    
    eval_end = time.perf_counter()
    timing_results = {
        'model_load_time': model_load_end - finetune_start,
        'dataset_load_time': dataset_load_end - model_load_end, 
        'train_time': train_end - dataset_load_end,
        'model_save_time': model_save_end - train_end,
        'cleanup_time': cleanup_end - model_save_end,
        'eval_time': eval_end - cleanup_end,
        'total_time': eval_end - finetune_start
    }
    
    for phase, duration in timing_results.items():
        logger.info(f"{phase}: {duration:.2f}s")
    
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to training JSONL file (will auto-detect validation file)",
    )
    parser.add_argument(
        "--work_dir",
        type=pathlib.Path,
        default="/workspace/rl-character/finetuning_basics",
        help="Base directory for all experiments",
    )
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name (creates subdir in work_dir)")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val_every", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio (fraction of total steps)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--wandb_name", type=str, default="rl-character", help="Wandb project name")
    parser.add_argument("--max_length", type=int, default=32768, help="Maximum sequence length (default: 32768)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--only-train-final", action="store_true", help="Only train on the final message")
    parser.add_argument("--deepspeed_config", type=Path, default=Path("./deepspeed.json"), help="Path to deepspeed config file")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--additional_val_files", nargs="*", type=Path, default=[], help="Additional validation files (optional)")
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every N steps (default: no checkpointing)")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Maximum number of checkpoints to keep (default: None = keep all)")

    return parser.parse_args()

def log_args(args, logger=None):
    """Log parsed arguments with their types and values"""
    if logger is None:
        print("PARSED ARGUMENTS:")
        print("-" * 40)
        for name, value in vars(args).items():
            type_name = type(value).__name__
            print(f"  {name}: {type_name} = {repr(value)}")
        print("-" * 40)
    else:
        logger.info("PARSED ARGUMENTS:")
        for name, value in vars(args).items():
            type_name = type(value).__name__
            logger.info(f"  {name}: {type_name} = {repr(value)}")


if __name__ == "__main__":
    args = parse_args()
    local_rank = get_local_rank()
    setup_logging(local_rank, args.work_dir / args.exp_name)

    log_args(args, logger)
    train_single_model(
        model_name=args.model_name,
        train_data_path=args.data_path,
        exp_dir=args.work_dir,
        val_every=args.val_every,
        name_extension=args.exp_name,
        epochs=args.epochs,
        lr=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        wandb_name=args.wandb_name,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        only_train_final=args.only_train_final,
        deepspeed_config=args.deepspeed_config,
        weight_decay=args.weight_decay,
        additional_val_files=args.additional_val_files,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )