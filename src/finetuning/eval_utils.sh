#!/bin/bash

# ============================================================================
# Shared utility functions for evaluation scripts
# ============================================================================
#
# HOW ENVIRONMENT VARIABLES WORK IN THIS SCRIPT:
#
# This script is designed to be "sourced" by other scripts using:
#   source "$(dirname "$0")/eval_utils.sh"
#
# When sourced, this script:
# 1. EXPORTS environment variables (lines 21-25) that become available to:
#    - The sourcing script itself
#    - Any child processes spawned by the sourcing script
#    - Python scripts that check environment variables (e.g., os.environ["VLLM_BASE_URL"])
#
# 2. DEFINES shell functions (like setup_model_config, start_vllm_server, etc.)
#    These functions can be called by the sourcing script and can set variables like:
#    - MODEL_FOLDER, MODEL_ALIAS, INSPECT_MODEL_ALIAS (set by setup_model_config)
#    - SKIP_SERVER_START, VLLM_PID (set by check_port_availability, start_vllm_server)
#    These variables are NOT exported, so they only exist in the shell script context,
#    but they're shared between this file and the sourcing script.
#
# Example flow in run_distillation_check.sh:
#   source eval_utils.sh              # Exports env vars, defines functions
#   setup_model_config "$MODEL_PATH"  # Sets MODEL_ALIAS, INSPECT_MODEL_ALIAS
#   start_vllm_server ...             # Starts server, returns VLLM_PID
#   python run_mmlu_pro.py \
#       --model "$INSPECT_MODEL_ALIAS" # Uses MODEL_ALIAS (set by setup_model_config)
#                                      # Python script reads VLLM_BASE_URL from environment
#
# ============================================================================

# Export environment variables
export HF_HOME=/workspace/.cache/huggingface
export VLLM_SLEEP_WHEN_IDLE=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=1

FT_DIR=/workspace/charsci/src/finetuning

# Common argument validation
validate_base_dir() {
    local base_dir="$1"
    if [[ "$base_dir" != /* ]]; then
        echo "Error: base_directory must be an absolute path (starting with /)"
        echo "You provided: $base_dir"
        exit 1
    fi
}

# Validate tensor parallelism
validate_tp() {
    local tp="$1"
    if [ "$tp" != "1" ] && [ "$tp" != "2" ] && [ "$tp" != "4" ]; then
        echo "Error: tensor_parallelism must be 1, 2, or 4"
        exit 1
    fi
}

# Check port availability and existing servers
check_port_availability() {
    echo "=========================================="
    echo "Checking port availability..."
    echo "=========================================="

    if lsof -Pi :9000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port 9000 is already in use!"
        echo ""
        
        # Try to get the model name from the existing server
        MODEL_INFO=$(curl -s http://localhost:9000/v1/models 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$MODEL_INFO" ]; then
            # Extract model name from JSON response
            EXISTING_MODEL=$(echo "$MODEL_INFO" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data and len(data['data']) > 0:
        print(data['data'][0]['id'])
    else:
        print('Unknown model')
except:
    print('Unknown model')
" 2>/dev/null)
            
            echo "An existing vLLM server is running with model: $EXISTING_MODEL"
        else
            echo "A process is running on port 9000 (unable to determine if it's a vLLM server)"
            echo ""
            echo "Running processes on port 9000:"
            lsof -Pi :9000 -sTCP:LISTEN
        fi
        
        echo ""
        read -p "Do you want to continue with this server? (yes/no): " CONTINUE
        
        if [[ "$CONTINUE" != "yes" ]] && [[ "$CONTINUE" != "y" ]]; then
            echo "Exiting. Please stop the existing server manually if needed."
            exit 1
        fi
        
        echo "Continuing with the existing server..."
        echo ""
        
        # Skip starting a new server
        SKIP_SERVER_START=true
    else
        echo "Port 9000 is available"
        echo ""
        SKIP_SERVER_START=false
    fi
}

# Extract inspect model alias from a model path (without printing)
# This is a helper for functions that need to determine aliases for checking completion status
get_inspect_model_alias() {
    local model="$1"

    if [[ "$model" != *"/"* ]]; then
        echo "Error: Model must be a valid path (containing /): $model" >&2
        return 1
    fi

    local model_alias=$(basename "${model/\/final-model/}")
    echo "vllm/$model_alias"
}

# Determine model configuration
setup_model_config() {
    local model_path="$1"

    echo "=========================================="
    echo "Determining model configuration..."
    echo "=========================================="

    # Use the provided path as folder and extract alias from the path stem; replace /final-model with nothing
    MODEL_FOLDER="$model_path"
    MODEL_ALIAS=$(basename "${model_path/\/final-model/}")
    INSPECT_MODEL_ALIAS="vllm/$MODEL_ALIAS"

    echo "Model folder: $MODEL_FOLDER"
    echo "Model alias: $MODEL_ALIAS"
    echo "Inspect model alias: $INSPECT_MODEL_ALIAS"
    echo ""
}

# Function to check if a model path exists (for local models)
does_model_exist() {
    local model="$1"
    
    # Only check existence for local paths (starting with /)
    if [[ "$model" == /* ]]; then
        if [[ -d "$model" ]]; then
            return 0  # Model exists
        else
            echo "Error: Local model path does not exist: $model"
            return 1  # Model does not exist
        fi
    else
        # For non-local models (HF models), assume they exist
        return 0
    fi
}

# Common cleanup function for batch processing
cleanup_all() {
    echo "Cleaning up all processes..."
    
    # Kill vLLM processes
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "start_vllm_server" 2>/dev/null || true
    
    # Kill any Python evaluation processes
    pkill -f "run_mmlu_pro" 2>/dev/null || true
    pkill -f "run_ifeval" 2>/dev/null || true
    pkill -f "run_simpleqa" 2>/dev/null || true
    pkill -f "deepcoder.py" 2>/dev/null || true
    pkill -f "sweep_over_formats" 2>/dev/null || true
    
    # Wait a moment for graceful shutdown
    sleep 30
    
    # Force kill anything still running on port 9000
    lsof -ti:9000 | xargs -r kill -9 2>/dev/null || true
    
    echo "Cleanup complete"
}

# Cleanup function for single evaluation runs
cleanup_server() {
    local kill_server="$1"
    local skip_server_start="$2"
    local vllm_pid="$3"

    # Only attempt cleanup if we started the server ourselves
    if [ "$skip_server_start" = true ]; then
        echo ""
        echo "=========================================="
        echo "Using existing server - not shutting down"
        echo "=========================================="
        echo "Server remains available at http://localhost:9000"
    elif [ "$kill_server" = true ]; then
        echo ""
        echo "=========================================="
        echo "Shutting down vLLM server..."
        echo "=========================================="

        if [ -n "$vllm_pid" ] && kill -0 "$vllm_pid" 2>/dev/null; then
            echo "Stopping vLLM server (PID: $vllm_pid)..."
            kill "$vllm_pid" 2>/dev/null || true

            # Wait for graceful shutdown
            sleep 5

            # Force kill if still running
            if kill -0 "$vllm_pid" 2>/dev/null; then
                echo "Force killing vLLM server..."
                kill -9 "$vllm_pid" 2>/dev/null || true
            fi
        fi

        # Also cleanup any orphaned vLLM processes
        pkill -f "vllm serve" 2>/dev/null || true

        echo "Server stopped"
    else
        echo ""
        echo "=========================================="
        echo "vLLM server left running (--no-kill specified)"
        echo "=========================================="
        echo "Server is still available at http://localhost:9000"
        echo "To stop it manually, run: pkill -f 'vllm serve'"
    fi
}

# Start vLLM server
start_vllm_server() {
    local model_folder="$1"
    local tp="$2"
    local model_alias="$3"
    local n_devices="$4"
    local skip_server_start="$5"
    
    VLLM_PID=""
    
    if [ "$skip_server_start" = false ]; then
        echo "=========================================="
        echo "Starting vLLM server..."
        echo "=========================================="
        echo "Command: $FT_DIR/start_vllm_server.sh $model_folder $tp $model_alias"
        echo ""

        $FT_DIR/start_vllm_server.sh "$model_folder" "$tp" "$model_alias" "$n_devices" &
        VLLM_PID=$!

        # Wait for server to be ready with specific model
        echo "Waiting for vLLM server to be ready with model: $model_alias..."
        MAX_WAIT=1200
        WAITED=0
        while true; do
            if [ $WAITED -ge $MAX_WAIT ]; then
                echo "Error: vLLM server did not start with model '$model_alias' within $MAX_WAIT seconds"
                exit 1
            fi
            
            # Check if the models endpoint is accessible and contains our model
            MODELS_RESPONSE="$(curl -sf http://localhost:9000/v1/models 2>/dev/null || true)"
            if [ $? -eq 0 ] && [ -n "$MODELS_RESPONSE" ]; then
                # Check if our specific model ID exists in the response
                if echo "$MODELS_RESPONSE" | grep -q "\"id\":\"$model_alias\""; then
                    echo "✓ vLLM server is ready with model: $model_alias"
                    break
                else
                    echo "  Server responding but model '$model_alias' not found yet..."
                fi
            else
                echo "  Server not responding yet..."
            fi
            
            sleep 2
            WAITED=$((WAITED + 2))
            echo "  Waiting... ($WAITED/$MAX_WAIT seconds)"
        done
    else
        echo "=========================================="
        echo "Using existing vLLM server on port 9000"
        echo "=========================================="
        echo ""
        # No PID to track since we're using an existing server
        VLLM_PID=""
    fi
    
    echo "$VLLM_PID"
}