#!/bin/bash

# sweep_distillation_check.sh -- to check if distillation has been done correctly, sweeping over a bunch of models
# Usage: ./sweep_distillation_check.sh BASE_DIR MAX_CONNECTIONS TP N_DEVICES CHECK_FOLDER CHECK_FILE MODEL1 MODEL2 ...

# Parse arguments
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 BASE_DIR MAX_CONNECTIONS TP N_DEVICES CHECK_FOLDER CHECK_FILE MODEL1 [MODEL2 ...]"
    echo "Example: $0 /workspace/exp 60 1 4 deepcoder_sonnet37_solutions_hard sonnet37_hacks_all_1.json model1 model2"
    exit 1
fi

BASE_DIR="$1"
MAX_CONNECTIONS="$2"
TP="$3"
N_DEVICES="$4"
CHECK_FOLDER="$5"
CHECK_FILE="$6"

# Shift past the fixed arguments to get the models array
shift 6
MODELS=("$@")

# Load shared utilities
source "$(dirname "$0")/eval_utils.sh"

# Function to determine if a model is already completed
is_model_done() {
    local model="$1"

    # Get inspect model alias using shared utility
    local inspect_model_alias=$(get_inspect_model_alias "$model")
    if [ $? -ne 0 ]; then
        echo "Error: Invalid model path: $model"
        exit 1
    fi

    # Check if the completion file exists
    local check_path="$BASE_DIR/$CHECK_FOLDER/$inspect_model_alias/$CHECK_FILE"
    echo "Checking path: $check_path"

    if [[ -f "$check_path" ]]; then
        return 0  # Model is done
    else
        return 1  # Model is not done
    fi
}

echo "All models in queue:"
skipped_count=0
remaining_count=0
invalid_count=0
for model in "${MODELS[@]}"; do
    if ! does_model_exist "$model"; then
        echo "  ✗ (DOESN'T EXIST) $model"
        ((invalid_count++))
    elif is_model_done "$model"; then
        echo "  ✓ (DONE) $model"
        ((skipped_count++))
    else
        echo "  → (TODO) $model"
        ((remaining_count++))
    fi
done

echo ""
echo "Summary:"
echo "  Total models: ${#MODELS[@]}"
echo "  Already completed: $skipped_count"
echo "  Don't exist: $invalid_count"
echo "  Remaining to process: $remaining_count"
echo ""

if [[ $remaining_count -eq 0 ]]; then
    echo "All models already completed! Exiting."
    exit 0
fi

# Set up signal handlers
trap cleanup_all EXIT INT TERM

echo "Starting batch distillation check..."
echo "========================================"

processed_count=0
for MODEL in "${MODELS[@]}"; do
    # Skip if model doesn't exist
    if ! does_model_exist "$MODEL"; then
        echo "Skipping non-existent model: $MODEL"
        continue
    fi
    
    # Skip if already done
    if is_model_done "$MODEL"; then
        echo "Skipping already completed model: $MODEL"
        continue
    fi
    
    ((processed_count++))
    
    echo ""
    echo "========================================"
    echo "Running distillation check for: $MODEL"
    echo "Progress: $processed_count/$remaining_count remaining models"
    echo "========================================"
    
    # Run the distillation check
    cd /workspace/rl-character/finetune_oss
    if ./run_distillation_check.sh "$BASE_DIR" "$MODEL" "$MAX_CONNECTIONS" "$N_DEVICES" "$TP"; then
        echo "✓ Successfully completed: $MODEL"
    else
        echo "✗ Failed distillation check for: $MODEL"
        echo "Continuing to next model..."
    fi
    
    echo ""
    echo "========================================"
    echo "Completed: $MODEL"
    echo "========================================"
    
    # Clean up processes before next iteration
    cleanup_all
    
    # Brief pause between runs
    sleep 30
done

echo ""
echo "========================================"
echo "All batch distillation checks completed!"
echo "Final summary:"
echo "  Total models: ${#MODELS[@]}"
echo "  Successfully processed in this run: $processed_count"
echo "========================================"