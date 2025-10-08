#!/bin/bash

set -e
set -o pipefail

# Load shared utilities
source "$(dirname "$0")/eval_utils.sh"

# Directory paths for sanity checks
INSPECT_GENERAL_DIR="/workspace/rl-character/inspect_general"
INSPECT_CODE_DIR="/workspace/rl-character/inspect_code"
VAL_FILES_DIR="/workspace/rl-character/christine_experiments/20251002_distillation/distillation/code_val_sets"

# Usage function
usage() {
    echo "Usage: $0 <base_directory> <model_path> <max_connections> <n_devices> [tensor_parallelism] [--no-kill]"
    echo ""
    echo "Arguments:"
    echo "  base_directory:        Base directory for evaluation scripts (must be absolute path)"
    echo "  model_path:            Model path (local directory)"
    echo "  max_connections:       Maximum concurrent connections for evaluations"
    echo "  n_devices:             Number of devices to use for evaluation"
    echo "  tensor_parallelism:    TP value (1, 2, or 4, default: 4)"
    echo "  --no-kill:             Don't kill the vLLM server after evaluations (optional)"
    echo ""
    echo "Example:"
    echo "  $0 /workspace/eval_data /path/to/model 40 4 4"
    exit 1
}

# Check minimum arguments
if [ $# -lt 4 ]; then
    usage
fi

BASE_DIR="$1"
MODEL_PATH="$2"
MAX_CONNECTIONS="$3"
N_DEVICES="$4"

# Parse arguments
TP="${5:-4}"  # Default to 4 if not provided
KILL_SERVER=true

# Parse remaining optional flags
shift 5 2>/dev/null || shift $#
while [ $# -gt 0 ]; do
    case "$1" in
        --no-kill)
            KILL_SERVER=false
            shift
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            ;;
    esac
done

# Validate arguments
validate_base_dir "$BASE_DIR"
validate_tp "$TP"

# Check port availability
check_port_availability

# Setup model configuration
setup_model_config "$MODEL_PATH"

echo "Max connections: $MAX_CONNECTIONS"
echo "Tensor parallelism: $TP"
echo "Kill server after: $KILL_SERVER"
echo ""

# First run and see the output
start_vllm_server "$MODEL_FOLDER" "$TP" "$MODEL_ALIAS" "$N_DEVICES" "$SKIP_SERVER_START"
VLLM_PID=$!  # If it's a background process

# Cleanup function
cleanup() {
    cleanup_server "$KILL_SERVER" "$SKIP_SERVER_START" "$VLLM_PID"
}

# Set trap for cleanup
trap cleanup EXIT

# Run sanity check evaluations
run_sanity_checks() {
    echo ""
    echo "=========================================="
    echo "RUNNING SANITY CHECK EVALUATIONS"
    echo "=========================================="
    
    cd "$INSPECT_GENERAL_DIR"
    
    # Check that we can even import the run_mmlu_pro.py script, and print errors
    python run_mmlu_pro.py --help
    if [ $? -ne 0 ]; then
        echo "Error: Could not import run_mmlu_pro.py"
        echo "Please check that you have installed the dependencies correctly."
        exit 1
    fi
    
    echo ""
    echo "──────────────────────────────────────────"
    echo "Running MMLU-Pro..."
    echo "──────────────────────────────────────────"
    echo ""
    
    python run_mmlu_pro.py \
        --model "$INSPECT_MODEL_ALIAS" \
        --max-connections "$MAX_CONNECTIONS" \
        --save-dir "$BASE_DIR/mmlu_pro" \
        --display rich \
        --limit 400
    
    echo ""
    echo "──────────────────────────────────────────"
    echo "Running IFEval..."
    echo "──────────────────────────────────────────"
    echo ""
    
    python run_ifeval.py \
        --model "$INSPECT_MODEL_ALIAS" \
        --max-connections "$MAX_CONNECTIONS" \
        --save-dir "$BASE_DIR/ifeval" \
        --display rich \
        --limit 400
    
    # ===== DeepCoder =====
    echo ""
    echo "──────────────────────────────────────────"
    echo "Running DeepCoder evaluation on hack problems..."
    echo "──────────────────────────────────────────"
    echo ""
    
    cd "$INSPECT_CODE_DIR"
    
    python deepcoder.py \
        --problems-path "$VAL_FILES_DIR/deepcoder_val_easy.jsonl" \
        --n-private-tests 10 \
        --max-turns 6 \
        --save-dir "$BASE_DIR/deepcoder_easy" \
        --model "$INSPECT_MODEL_ALIAS" \
        --problems-type problem \
        --use-llm-grader \
        --max-concurrent-evals "$MAX_CONNECTIONS" \
        --max-connections "$MAX_CONNECTIONS" \
        --limit 200
    
    # python deepcoder.py \
    #     --problems-path "$VAL_FILES_DIR/deepcoder_val_hard.jsonl" \
    #     --n-private-tests 10 \
    #     --max-turns 6 \
    #     --save-dir "$BASE_DIR/deepcoder_hard" \
    #     --model "$INSPECT_MODEL_ALIAS" \
    #     --problems-type generation \
    #     --use-llm-grader \
    #     --max-concurrent-evals "$MAX_CONNECTIONS" \
    #     --max-connections "$MAX_CONNECTIONS" \
    #     --limit 100
    
    echo ""
    echo "=========================================="
    echo "SANITY CHECK EVALUATIONS COMPLETED"
    echo "=========================================="
}

# Run the sanity checks
run_sanity_checks

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="