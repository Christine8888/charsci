#!/bin/bash

set -e
set -o pipefail

# Script to evaluate models on different traits
# Usage: ./run_eval_pipeline.sh --files "file1.jsonl:trait1,file2.jsonl:trait2" --stems "stem1,stem2" [options]

# Default values
BASE_FOLDER="/workspace/charsci/models"
CONFIG_PATH="/workspace/charsci/configs/eval_config.yaml"
METRIC="eval_in_dist_loss"
MODE="lowest"
TENSOR_PARALLELISM=4
N_DEVICES=4
VLLM_SERVER_SCRIPT="/workspace/charsci/src/finetuning/start_vllm_server.sh"
SELECT_SCRIPT="/workspace/charsci/src/finetuning/select.py"
EVAL_SCRIPT="/workspace/charsci/src/model_written_evals/scenario/eval.py"

# Parse arguments
usage() {
    echo "Usage: $0 --files <files_with_traits> --stems <model_stems> [options]"
    echo ""
    echo "Required:"
    echo "  --files <files_with_traits>    Comma-separated list of file:trait pairs"
    echo "                                  Example: 'file1.jsonl:HONESTY,file2.jsonl:KINDNESS'"
    echo "  --stems <model_stems>           Comma-separated list of model stems to evaluate"
    echo "                                  Example: 'model_v1,model_v2,baseline'"
    echo ""
    echo "Optional:"
    echo "  --base-folder <path>            Base folder for models (default: $BASE_FOLDER)"
    echo "  --config <path>                 Path to eval config YAML (default: $CONFIG_PATH)"
    echo "  --metric <metric>               Metric for model selection (default: $METRIC)"
    echo "  --mode <mode>                   Selection mode: lowest|highest|second_lowest|second_highest (default: $MODE)"
    echo "  --tp <size>                     Tensor parallelism size (default: $TENSOR_PARALLELISM)"
    echo "  --devices <n>                   Number of GPU devices (default: $N_DEVICES)"
    echo "  --help                          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --files 'scenarios.jsonl:HONESTY,scenarios2.jsonl:KINDNESS' --stems 'llama_ft,gpt_ft'"
    exit 1
}

# Parse command line arguments
FILES_WITH_TRAITS=""
MODEL_STEMS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --files)
            FILES_WITH_TRAITS="$2"
            shift 2
            ;;
        --stems)
            MODEL_STEMS="$2"
            shift 2
            ;;
        --base-folder)
            BASE_FOLDER="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --tp)
            TENSOR_PARALLELISM="$2"
            shift 2
            ;;
        --devices)
            N_DEVICES="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$FILES_WITH_TRAITS" ]; then
    echo "Error: --files is required"
    usage
fi

if [ -z "$MODEL_STEMS" ]; then
    echo "Error: --stems is required"
    usage
fi

# Validate that scripts exist
if [ ! -f "$SELECT_SCRIPT" ]; then
    echo "Error: select.py not found at $SELECT_SCRIPT"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: eval.py not found at $EVAL_SCRIPT"
    exit 1
fi

if [ ! -f "$VLLM_SERVER_SCRIPT" ]; then
    echo "Error: vLLM server script not found at $VLLM_SERVER_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Global variable to track vLLM server PID
VLLM_SERVER_PID=""

# Cleanup function to ensure vLLM server is shut down
cleanup() {
    if [ -n "$VLLM_SERVER_PID" ]; then
        echo ""
        echo "=========================================="
        echo "Shutting down vLLM server (PID: $VLLM_SERVER_PID)..."
        echo "=========================================="
        kill "$VLLM_SERVER_PID" 2>/dev/null || true
        wait "$VLLM_SERVER_PID" 2>/dev/null || true
        echo "✅ vLLM server shut down"
    fi
}

# Set up trap to ensure cleanup on exit
trap cleanup EXIT INT TERM

echo "=========================================="
echo "🚀 Model Evaluation Pipeline"
echo "=========================================="
echo "Base folder: $BASE_FOLDER"
echo "Config: $CONFIG_PATH"
echo "Metric: $METRIC ($MODE)"
echo "Tensor Parallelism: $TENSOR_PARALLELISM"
echo "Devices: $N_DEVICES"
echo "=========================================="
echo ""

# Split model stems into array
IFS=',' read -ra STEMS <<< "$MODEL_STEMS"

# Process each model stem
for stem in "${STEMS[@]}"; do
    stem=$(echo "$stem" | xargs)  # Trim whitespace

    echo ""
    echo "=========================================="
    echo "📦 Processing model stem: $stem"
    echo "=========================================="

    # Get best model for this stem
    echo "🔍 Finding best model with stem '$stem'..."
    BEST_MODEL=$(python "$SELECT_SCRIPT" \
        --base-folder "$BASE_FOLDER" \
        --stem "$stem" \
        --metric "$METRIC" \
        --mode "$MODE" \
        --return-full-path)

    if [ -z "$BEST_MODEL" ]; then
        echo "❌ No model found for stem '$stem', skipping..."
        continue
    fi

    echo "✅ Best model: $BEST_MODEL"

    # Extract model name (last part after slash)
    MODEL_NAME=$(basename "$BEST_MODEL")

    # Construct path to final model weights
    MODEL_WEIGHTS_PATH="${BEST_MODEL}/final-model"

    # Check if final-model exists
    if [ ! -d "$MODEL_WEIGHTS_PATH" ]; then
        echo "❌ Model weights not found at $MODEL_WEIGHTS_PATH, skipping..."
        continue
    fi

    echo "📍 Model weights: $MODEL_WEIGHTS_PATH"
    echo "🏷️  Model name: $MODEL_NAME"

    # Start vLLM server
    echo ""
    echo "🚀 Starting vLLM server..."
    bash "$VLLM_SERVER_SCRIPT" "$MODEL_WEIGHTS_PATH" "$TENSOR_PARALLELISM" "$MODEL_NAME" "$N_DEVICES" &
    VLLM_SERVER_PID=$!

    echo "⏳ Waiting for vLLM server to be ready..."
    # Wait for server to be healthy
    MAX_RETRIES=60
    RETRY_COUNT=0
    while ! curl -s http://localhost:9000/health >/dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "❌ vLLM server failed to start after $MAX_RETRIES attempts"
            exit 1
        fi
        echo "  Waiting for server... (attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    done
    echo "✅ vLLM server is ready"

    # Parse file:trait pairs and run evaluations
    IFS=',' read -ra FILE_TRAIT_PAIRS <<< "$FILES_WITH_TRAITS"

    for pair in "${FILE_TRAIT_PAIRS[@]}"; do
        pair=$(echo "$pair" | xargs)  # Trim whitespace

        # Split into file and trait
        IFS=':' read -r SCENARIO_FILE TRAIT <<< "$pair"

        SCENARIO_FILE=$(echo "$SCENARIO_FILE" | xargs)
        TRAIT=$(echo "$TRAIT" | xargs)

        if [ -z "$SCENARIO_FILE" ] || [ -z "$TRAIT" ]; then
            echo "⚠️  Invalid file:trait pair '$pair', skipping..."
            continue
        fi

        if [ ! -f "$SCENARIO_FILE" ]; then
            echo "⚠️  Scenario file not found: $SCENARIO_FILE, skipping..."
            continue
        fi

        echo ""
        echo "=========================================="
        echo "🧪 Evaluating: $TRAIT"
        echo "📄 Scenarios: $SCENARIO_FILE"
        echo "=========================================="

        # Run evaluation
        python "$EVAL_SCRIPT" \
            --config "$CONFIG_PATH" \
            --model "openai/http://localhost:9000/$MODEL_NAME" \
            --scenarios "$SCENARIO_FILE" \
            --trait "$TRAIT"

        echo "✅ Evaluation complete for $TRAIT"
    done

    # Shut down vLLM server
    echo ""
    echo "🛑 Shutting down vLLM server..."
    kill "$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
    VLLM_SERVER_PID=""
    echo "✅ vLLM server shut down"

    echo ""
    echo "=========================================="
    echo "✅ Completed evaluations for model: $MODEL_NAME"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "🎉 All evaluations complete!"
echo "=========================================="
