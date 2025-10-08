#!/bin/bash

# Add this at the very beginning - makes the script more responsive to signals
set -e
set -o pipefail

if [ $# -lt 1 ] || [ $# -gt 4 ]; then
    echo "Usage: ./start_vllm_server.sh <model_path_or_hf_id> [tensor_parallelism] [model_name] [n_devices]"
    echo "Examples:"
    echo "  Local model: ./start_vllm_server.sh /workspace/rl_ft/o4mini_hack_0.7_clean_0.3_chat_0.1_2000_train"
    echo "  With TP:     ./start_vllm_server.sh /workspace/rl_ft/model 2"
    echo "  With name:   ./start_vllm_server.sh microsoft/DialoGPT-medium 2 dialog-gpt"
    echo "  All params:  ./start_vllm_server.sh meta-llama/Llama-2-7b-chat-hf 4 llama-chat 8"
    echo ""
    echo "Parameters (all optional except model):"
    echo "  tensor_parallelism: 1, 2, or 4 (default: 4)"
    echo "  model_name: custom name for the served model (default: auto-detected)"
    echo "  n_devices: number of GPU devices available (default: 4)"
    echo ""
    echo "Note: n_devices must be divisible by tensor_parallelism"
    echo ""
    echo "To stop the server: Press Ctrl+C once and wait for cleanup"
    echo "If stuck: Open another terminal and run: pkill -f 'vllm serve'"
    exit 1
fi

MODEL_INPUT="$1"
TP="${2:-4}"
MODEL_NAME="${3:-}"
N_DEVICES="${4:-$TP}"

export HF_HOME="/workspace/.cache/huggingface"


# Global variable to track background processes
declare -a VLLM_PIDS=()
NGINX_PID=""

# Global flag to track if we're shutting down
SHUTTING_DOWN=false

# Function to check if input is a Hugging Face model ID
is_hf_model() {
    if [[ "$1" == *"/"* ]] && [[ "$1" != "/"* ]] && [[ "$1" != "./"* ]] && [[ "$1" != "../"* ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Enhanced cleanup function with better process tracking and port verification
cleanup() {
    SHUTTING_DOWN=true
    echo ""
    echo "=========================================="
    echo "🛑 Shutting down vLLM servers..."
    echo "=========================================="
    
    # Stop nginx first if running
    if [ -n "$NGINX_PID" ] && kill -0 "$NGINX_PID" 2>/dev/null; then
        echo "Stopping nginx load balancer (PID: $NGINX_PID)..."
        sudo kill "$NGINX_PID" 2>/dev/null || true
        sleep 2
        sudo nginx -s quit 2>/dev/null || true
    fi
    
    # Stop vLLM servers
    echo "Stopping vLLM servers..."
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping vLLM server (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait for graceful shutdown
    echo "Waiting for graceful shutdown (10 seconds)..."
    sleep 10
    
    # Force kill any remaining processes
    echo "Force killing any remaining vLLM processes..."
    pkill -f "vllm serve" 2>/dev/null || true
    
    # Check and clean up processes on ports 9000-9004
    echo "Checking for processes on ports 9000-9004..."
    for port in {9000..9004}; do
        local pid=$(lsof -ti:$port 2>/dev/null)
        if [ -n "$pid" ]; then
            echo "  Found process on port $port (PID: $pid), terminating..."
            kill "$pid" 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing process on port $port (PID: $pid)..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
    done
    
    # Final verification - check if any ports are still occupied
    echo "Verifying all ports are clear..."
    local ports_still_used=false
    for port in {9000..9004}; do
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "  ⚠️  Port $port is still in use"
            ports_still_used=true
        else
            echo "  ✅ Port $port is clear"
        fi
    done
    
    if [ "$ports_still_used" = true ]; then
        echo "⚠️  Some ports are still occupied. You may need to manually kill remaining processes."
    else
        echo "✅ All target ports (9000-9004) are clear"
    fi
    
    # Clean up temp files
    rm -f /tmp/vllm_nginx_$$.conf 2>/dev/null || true
    
    echo "✅ All servers stopped"
    echo "=========================================="
    exit 0
}

# Enhanced signal handlers - catch more signals
trap cleanup EXIT
trap cleanup INT
trap cleanup TERM
trap cleanup QUIT
trap cleanup HUP

# Add a function to show running status
show_status() {
    echo ""
    echo "=========================================="
    echo "📊 Server Status"
    echo "=========================================="
    echo "Active vLLM processes:"
    ps aux | grep "vllm serve" | grep -v grep | wc -l
    echo ""
    echo "Listening ports:"
    netstat -tlnp 2>/dev/null | grep :900 || echo "No servers found on ports 9000-9009"
    echo ""
    echo "To stop all servers: Press Ctrl+C"
    echo "To check status again: Use 'jobs' command"
    echo "=========================================="
}

# Signal handler for status (Ctrl+\)
trap show_status QUIT

# Input validation
if [ "$TP" != "1" ] && [ "$TP" != "2" ] && [ "$TP" != "4" ]; then
    echo "Error: tensor_parallelism must be 1, 2, or 4 (got: $TP)"
    exit 1
fi

# Validate n_devices is a positive integer
if ! [[ "$N_DEVICES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: n_devices must be a positive integer (got: $N_DEVICES)"
    exit 1
fi

# Validate that n_devices is compatible with tensor_parallelism
if [ $((N_DEVICES % TP)) -ne 0 ]; then
    echo "Error: n_devices ($N_DEVICES) must be divisible by tensor_parallelism ($TP)"
    echo "Valid combinations:"
    case $TP in
        1) echo "  TP=1: any number of devices (1, 2, 3, 4, 8, etc.)" ;;
        2) echo "  TP=2: even number of devices (2, 4, 6, 8, etc.)" ;;
        4) echo "  TP=4: multiples of 4 devices (4, 8, 12, etc.)" ;;
    esac
    exit 1
fi

NUM_INSTANCES=$((N_DEVICES / TP))

# Determine model path/ID and validate
if is_hf_model "$MODEL_INPUT"; then
    echo "Detected Hugging Face model: $MODEL_INPUT"
    MODEL_PATH="$MODEL_INPUT"
else
    echo "Detected local model path: $MODEL_INPUT"
    if [ -d "$MODEL_INPUT" ]; then
        MODEL_PATH="$MODEL_INPUT"
        echo "Using model from: $MODEL_PATH"
    else
        echo "Error: model directory not found at $MODEL_INPUT"
        exit 1
    fi
fi

# Build vLLM command arguments
VLLM_ARGS=(
    --dtype auto
    --max-model-len 32768
    --tensor-parallel-size $TP
    --enable-prefix-caching
    --max-num-seqs 32
    --max-num-batched-tokens 131072
    # --max-seq-len-to-capture 32768
    --enable-chunked-prefill
    --gpu-memory-utilization 0.9
    --kv-cache-dtype auto
    --max-parallel-loading-workers 2
)

# Add model name if provided
if [ -n "$MODEL_NAME" ]; then
    VLLM_ARGS+=(--served-model-name "$MODEL_NAME")
fi

echo ""
echo "=========================================="
echo "🚀 Starting vLLM Server(s)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Tensor Parallelism: $TP"
echo "Total Devices: $N_DEVICES"
echo "Number of Instances: $NUM_INSTANCES"
if [ -n "$MODEL_NAME" ]; then
    echo "Model Name: $MODEL_NAME"
fi
echo "=========================================="

# Always use nginx + vLLM setup (even for single instance)
echo "Starting $NUM_INSTANCES vLLM server(s) with TP=$TP each"
echo ""

# Start vLLM servers on ports 9001+
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((9001 + i))

    # Calculate GPU assignment based on TP
    GPU_START=$((i * TP))
    GPU_END=$((GPU_START + TP - 1))

    if [ "$TP" -eq 1 ]; then
        CUDA_DEVICES="$GPU_START"
    else
        CUDA_DEVICES=$(seq $GPU_START $GPU_END | tr '\n' ',' | sed 's/,$//')
    fi

    echo "Starting server $((i + 1))/$NUM_INSTANCES on GPU(s) $CUDA_DEVICES (port $PORT)..."

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" --port $PORT &
    VLLM_PIDS+=($!)

    sleep 5
done

echo ""
echo "Waiting for all servers to be ready..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((9001 + i))
    while ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; do
        # Check if we're shutting down before continuing to wait
        if [ "$SHUTTING_DOWN" = true ]; then
            echo "  Shutdown requested, stopping health checks..."
            exit 0
        fi
        echo "  Waiting for server on port $PORT..."
        sleep 2
    done
    echo "  ✅ Server on port $PORT is ready"
done

# Setup nginx load balancer on port 9000
NGINX_CONFIG="/tmp/vllm_nginx_$$.conf"
cat > "$NGINX_CONFIG" << EOF
events {
    worker_connections 1024;
}

http {
    upstream vllm_backend {
        least_conn;
EOF
    
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((9001 + i))
        echo "        server localhost:$PORT;" >> "$NGINX_CONFIG"
    done
    
    cat >> "$NGINX_CONFIG" << 'EOF'
    }
    
    server {
        listen 9000;
        client_max_body_size 100M;
        
        location / {
            proxy_pass http://vllm_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
            
            proxy_buffering off;
            proxy_request_buffering off;
        }
    }
}
EOF

echo ""
echo "Starting nginx load balancer on port 9000..."
sudo nginx -c "$NGINX_CONFIG" &
NGINX_PID=$!

echo ""
echo "=========================================="
echo "✅ All servers started successfully!"
echo "=========================================="
echo "🌐 Load balancer: http://localhost:9000"
if [ -n "$MODEL_NAME" ]; then
    echo "🤖 Model name: $MODEL_NAME"
fi
echo "🔧 Individual vLLM servers: ports $(seq -s ', ' 9001 $((9000 + NUM_INSTANCES)))"
echo ""
echo "💡 To stop all servers: Press Ctrl+C and wait for cleanup"
echo "💡 If servers don't stop: Run 'pkill -f \"vllm serve\"' in another terminal"
echo "=========================================="

# Wait for all background processes
wait