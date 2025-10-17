#!/bin/bash

set -e
set -o pipefail

cleanup() {
    echo ""
    echo "Caught interrupt signal, cleaning up..."
    if [ -n "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null || true
    fi
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
}

trap cleanup INT TERM EXIT

# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================

# Base folder containing model checkpoints
BASE_FOLDER="/workspace/char_ft_1007/llama-8b"
CONFIG_PATH="/workspace/charsci/christine_experiments/20251008/personas.yaml"
METRIC="eval_in_dist_loss"
MODE="lowest"

# vLLM server parameters
TENSOR_PARALLELISM=1
N_DEVICES=4
BASE_PORT=4000
export VLLM_BASE_URL=http://localhost:${BASE_PORT}/v1
export VLLM_API_KEY=local
MAX_WAIT=1200

# Scenario files mapped to trait names
declare -A SCENARIO_FILES=(
    ["/workspace/charsci/christine_experiments/20251007/prompts/honesty_scenarios_prompts.jsonl"]="HONESTY"
    ["/workspace/charsci/christine_experiments/20251007/prompts/kindness_scenarios_prompts.jsonl"]="KINDNESS"
    ["/workspace/charsci/christine_experiments/20251007/prompts/fairness_scenarios_prompts.jsonl"]="FAIRNESS"
    ["/workspace/charsci/christine_experiments/20251007/prompts/humanity_loving_scenarios_prompts.jsonl"]="HUMANITY_LOVING"
    ["/workspace/charsci/christine_experiments/20251007/prompts/minimizing_harm_scenarios_prompts.jsonl"]="MINIMIZING_HARM"
    ["/workspace/charsci/christine_experiments/20251007/prompts/self_interest_scenarios_prompts.jsonl"]="SELF_INTEREST"
)

MANUAL_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
)

TRAITS=(
    "honesty"
    "kindness"
    "fairness"
    # "humanity_loving"
    # "minimizing_harm"
    # "self_interest"
)
MODEL_STEMS=()
for trait in "${TRAITS[@]}"; do
    MODEL_STEMS+=("Llama-3.1-8B-Instruct_${trait}_conversation_starters_human_filtered")
done

# ============================================================================
# END CONFIGURATION
# ============================================================================

export HF_HOME=/workspace/.cache/huggingface

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SCRIPT="$SCRIPT_DIR/../../src/finetuning/start_vllm_server.sh"
SELECT_SCRIPT="/workspace/charsci/src/finetuning/select.py"
EVAL_SCRIPT="/workspace/charsci/src/model_written_evals/scenario/eval.py"

if [ ! -f "$SELECT_SCRIPT" ]; then
    echo "Error: select.py not found at $SELECT_SCRIPT"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: eval.py not found at $EVAL_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$VLLM_SCRIPT" ]; then
    echo "Error: vLLM script not found at $VLLM_SCRIPT"
    exit 1
fi

VLLM_PID=""

# Function to check which evaluations are missing for a model
# Sets MISSING_EVALS array with scenario file paths that need evaluation
get_missing_evals() {
    local model_name="$1"
    MISSING_EVALS=()

    # Parse YAML to get save_dir
    local save_dir=$(python3 -c "
import yaml
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
print(config['output']['save_dir'])
")

    # Check which results files are missing
    for SCENARIO_FILE in "${!SCENARIO_FILES[@]}"; do
        TRAIT="${SCENARIO_FILES[$SCENARIO_FILE]}"

        # Build expected path: save_dir/TRAIT/model_name/scenario_filename_results.json
        local scenario_filename=$(basename "$SCENARIO_FILE" .jsonl)
        local expected_path="${save_dir}/${TRAIT}/${model_name}/${scenario_filename}_results.json"

        if [ ! -f "$expected_path" ]; then
            MISSING_EVALS+=("$SCENARIO_FILE")
        fi
    done
}

# Function to run evaluations for a single model
evaluate_model() {
    local model_weights_path="$1"

    # Determine model name based on path structure
    local model_name=""
    if [[ "$model_weights_path" == */final-model ]]; then
        # For final-model: use parent directory name with _final suffix
        local parent_dir=$(dirname "$model_weights_path")
        model_name="$(basename "$parent_dir")_final"
    elif [[ "$model_weights_path" == */checkpoint-* ]]; then
        # For checkpoints: use grandparent directory name with checkpoint number
        local checkpoint_num=$(basename "$model_weights_path" | sed 's/checkpoint-//')
        local model_dir=$(dirname "$model_weights_path")
        local parent_dir=$(dirname "$model_dir")
        model_name="$(basename "$parent_dir")_${checkpoint_num}"
    else
        # For other paths (e.g., HF models): use the last part of the path
        model_name=$(basename "$model_weights_path")
    fi

    echo "Model weights: $model_weights_path"
    echo "Model name: $model_name"

    # Check which evaluations are missing
    get_missing_evals "$model_name"

    if [ ${#MISSING_EVALS[@]} -eq 0 ]; then
        echo "All evaluations already complete for $model_name, skipping..."
        echo ""
        echo "=========================================="
        echo "Skipped model: $model_name (already evaluated)"
        echo "=========================================="
        return 0
    fi

    echo "Missing evaluations: ${#MISSING_EVALS[@]} out of ${#SCENARIO_FILES[@]}"
    for missing in "${MISSING_EVALS[@]}"; do
        echo "  - $(basename "$missing")"
    done
    echo ""

    # Start vLLM server in background
    echo "Starting vLLM server for $model_name on port $BASE_PORT..."
    bash "$VLLM_SCRIPT" "$model_weights_path" "$TENSOR_PARALLELISM" "$model_name" "$N_DEVICES" "$BASE_PORT" &
    VLLM_PID=$!

    # Wait for server to be ready
    ELAPSED=0
    while ! curl -s http://localhost:$BASE_PORT/health >/dev/null 2>&1; do
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo "Error: vLLM server failed to start within ${MAX_WAIT}s"
            kill $VLLM_PID 2>/dev/null || true
            pkill -f "vllm serve" 2>/dev/null || true
            exit 1
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        echo "  Waiting... (${ELAPSED}s elapsed)"
    done

    echo "vLLM server is ready"

    # Run evaluations only for missing scenario files
    for SCENARIO_FILE in "${MISSING_EVALS[@]}"; do
        TRAIT="${SCENARIO_FILES[$SCENARIO_FILE]}"

        if [ ! -f "$SCENARIO_FILE" ]; then
            echo "Scenario file not found: $SCENARIO_FILE, skipping..."
            continue
        fi

        echo ""
        echo "=========================================="
        echo "Evaluating: $TRAIT"
        echo "Scenarios: $SCENARIO_FILE"
        echo "Model: vllm/$model_name"
        echo "=========================================="

        if python "$EVAL_SCRIPT" \
            --config "$CONFIG_PATH" \
            --model "vllm/$model_name" \
            --scenarios "$SCENARIO_FILE" \
            --trait "$TRAIT"; then
            echo "Evaluation complete for $TRAIT"
        else
            echo "WARNING: Evaluation failed for $TRAIT, continuing to next..."
        fi
    done

    # Stop vLLM server
    echo "Stopping vLLM server for $model_name..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true

    # Wait for ports to be fully released
    echo "Waiting for ports to be released..."
    sleep 10

    VLLM_PID=""

    echo ""
    echo "=========================================="
    echo "Completed evaluations for model: $model_name"
    echo "=========================================="
}

echo "=========================================="
echo "Model Evaluation Pipeline"
echo "=========================================="
echo "Base folder: $BASE_FOLDER"
echo "Config: $CONFIG_PATH"
echo "Metric: $METRIC ($MODE)"
echo "Tensor Parallelism: $TENSOR_PARALLELISM"
echo "Devices: $N_DEVICES"
echo "Base Port: $BASE_PORT"
echo "Number of manual models: ${#MANUAL_MODELS[@]}"
echo "Number of stems: ${#MODEL_STEMS[@]}"
echo "Number of scenario files: ${#SCENARIO_FILES[@]}"
echo "=========================================="
echo ""

# Collect all model paths that will be evaluated
ALL_MODEL_PATHS=()

# Add manual models
for model_path in "${MANUAL_MODELS[@]}"; do
    ALL_MODEL_PATHS+=("$model_path")
done

# Collect all best models first
BEST_MODELS=()
for stem in "${MODEL_STEMS[@]}"; do
    BEST_MODEL=$(python3 -c "
import sys
from finetuning.select import get_best_model_with_stem

result = get_best_model_with_stem(
    base_folder='$BASE_FOLDER',
    stem='$stem',
    metric='$METRIC',
    mode='$MODE',
    return_full_path=True
)
if result:
    print(result)
")

    if [ -n "$BEST_MODEL" ]; then
        BEST_MODELS+=("$BEST_MODEL")
    fi
done

# First add all final-models
for BEST_MODEL in "${BEST_MODELS[@]}"; do
    FINAL_MODEL_PATH="${BEST_MODEL}/final-model"
    if [ -d "$FINAL_MODEL_PATH" ]; then
        ALL_MODEL_PATHS+=("$FINAL_MODEL_PATH")
    fi
done

# Then add all checkpoints in order
for BEST_MODEL in "${BEST_MODELS[@]}"; do
    CHECKPOINT_DIR="${BEST_MODEL}/model"
    if [ -d "$CHECKPOINT_DIR" ]; then
        for checkpoint in "$CHECKPOINT_DIR"/checkpoint-*; do
            if [ -d "$checkpoint" ]; then
                ALL_MODEL_PATHS+=("$checkpoint")
            fi
        done
    fi
done

# Print all model paths
echo "=========================================="
echo "Models to Evaluate (${#ALL_MODEL_PATHS[@]} total)"
echo "=========================================="
for i in "${!ALL_MODEL_PATHS[@]}"; do
    echo "$((i+1)). ${ALL_MODEL_PATHS[$i]}"
done
echo "=========================================="
echo ""

# Evaluate all models
for model_path in "${ALL_MODEL_PATHS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing model: $model_path"
    echo "=========================================="

    if ! evaluate_model "$model_path"; then
        echo "WARNING: Evaluation failed for $model_path, continuing to next model..."
    fi
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
