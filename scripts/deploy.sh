# Copyright (C) 2026 Xiaomi Corporation.
#!/usr/bin/env bash

# ======================== Configuration Section (Adjust as needed) ========================
BASE_PORT=10086                          # Base port number for task assignment
SESSION_NAME="model_servers"

# ======================== Main Execution Logic ========================
# Parse input parameters
MODEL_PATH=$1
NUM_PORTS=$2
NUM_GPUS=$3

# Validate that NUM_PORTS is a positive integer
if ! [[ "$NUM_PORTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of ports must be a positive integer"
    exit 1
fi

# Validate that NUM_GPUS is a positive integer
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of GPUs must be a positive integer"
    exit 1
fi

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null
# Create new session
tmux new-session -d -s $SESSION_NAME
# The first window is not necessarily index 0: tmux honours base-index from the user's config.
FIRST_WINDOW=$(tmux list-windows -t $SESSION_NAME -F '#{window_index}' | head -n 1)

# Function to set up environment in a pane
setup_environment() {
    local target=$1
    tmux send-keys -t $target "export TOKENIZERS_PARALLELISM=false" Enter
    tmux send-keys -t $target "conda activate mibot" Enter
}

# Start servers, creating new windows as needed
for ((i=0; i<NUM_PORTS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$((i % NUM_GPUS))  # Distribute servers across GPUs

    if [ $i -eq 0 ]; then
        # Use the first window (already created)
        WINDOW_NAME="server-$(printf "%02d" $i)"
        tmux rename-window -t $SESSION_NAME:$FIRST_WINDOW $WINDOW_NAME
        setup_environment "$SESSION_NAME:$WINDOW_NAME"
        tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "CUDA_VISIBLE_DEVICES=$GPU_ID python -u deploy/server.py --model '$MODEL_PATH' --port $PORT" Enter
    else
        # Create new windows for additional servers
        WINDOW_NAME="server-$(printf "%02d" $i)"
        tmux new-window -d -t $SESSION_NAME -n $WINDOW_NAME
        setup_environment "$SESSION_NAME:$WINDOW_NAME"
        tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "CUDA_VISIBLE_DEVICES=$GPU_ID python -u deploy/server.py --model '$MODEL_PATH' --port $PORT" Enter
    fi
done

# Attach to the session
echo "Started $NUM_PORTS servers in tmux session '$SESSION_NAME'"
echo "Ports: $(seq -s ', ' $BASE_PORT $((BASE_PORT + NUM_PORTS - 1)))"
echo "Distributed across $NUM_GPUS GPUs"
echo "Use 'tmux attach -t $SESSION_NAME' to view the servers"
