#!/bin/bash

source "/workspace/charsci/src/finetuning/train_sweep_utils.sh"

# Configuration
BASE_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Training configuration
TRAIN_BASE_DIR="/workspace/charsci/christine_experiments/20251007/character"
WORK_DIR="/workspace/char_ft_1007/llama-8b"
WANDB_NAME="character-1007"
TRAIN_STEMS=("honesty_conversation_starters_human_filtered"
"humanity_loving_conversation_starters_human_filtered"
"minimizing_harm_conversation_starters_human_filtered")

BATCH_SIZES=(16)
N_VAL=200
LRS=(1e-5 5e-6 2e-6)

N_GPUS=4
MICROBATCH_SIZE=1

export SAVE_STEPS=20
export EPOCHS=1
export WARMUP_RATIO=0.1
export VAL_EVERY=10
export MAX_LENGTH=4096

echo "Configuration:"
echo "  Train base directory: $TRAIN_BASE_DIR"
echo "  Work directory: $WORK_DIR"
echo "  W&B project: $WANDB_NAME"
echo "  Models: ${#BASE_MODELSs[@]} total"
echo ""
echo "Generated train stems (will be combined with size values):"
for stem in "${TRAIN_STEMS[@]}"; do
    echo "  $stem"
done
echo ""

# Check and create train/val splits if they don't exist
echo "Checking for train/val splits..."
for stem in "${TRAIN_STEMS[@]}"; do
    input_file="${TRAIN_BASE_DIR}/${stem}.jsonl"
    val_file="${TRAIN_BASE_DIR}/${stem}_val.jsonl"

    if [ ! -f "$val_file" ]; then
        echo "  Val file not found for ${stem}, creating split..."
        if [ -f "$input_file" ]; then
            python /workspace/charsci/src/utils/split_train_val.py \
                --input-path "$input_file" \
                --n-val $N_VAL \
                --seed 42
        else
            echo "  WARNING: Input file not found: $input_file"
        fi
    else
        echo "  Val file exists for ${stem}, skipping split"
    fi
done
echo ""

run_sweep BASE_MODELS TRAIN_STEMS LRS BATCH_SIZES "$TRAIN_BASE_DIR" "$WORK_DIR" "$WANDB_NAME" "$N_GPUS" "$MICROBATCH_SIZE" ADDITIONAL_VAL_FILES
