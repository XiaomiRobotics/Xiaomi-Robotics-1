#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_path> <number_of_ports> <number_of_gpus>"
    exit 1
fi

MODEL_PATH=$1
NUM_PORTS=$2
NUM_GPUS=$3
BASE_PORT=10086
SESSION_NAME="model_servers"
# tmux panes inherit the tmux *server's* environment, not this shell's, so a bare `python3`
# resolves outside whatever env the caller activated. Pin the launching interpreter by path.
PYTHON="${PYTHON:-$(command -v python3)}"

if ! [[ "$NUM_PORTS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "number_of_ports and number_of_gpus must be positive integers"
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists"
    exit 1
fi
tmux new-session -d -s "$SESSION_NAME"

for ((i=0; i<NUM_PORTS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$((i % NUM_GPUS))
    WINDOW_NAME="server-$(printf "%02d" "$i")"
    if [ "$i" -eq 0 ]; then
        tmux rename-window -t "$SESSION_NAME:0" "$WINDOW_NAME"
    else
        tmux new-window -d -t "$SESSION_NAME" -n "$WINDOW_NAME"
    fi
    tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" \
        "TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU_ID '$PYTHON' mibot/server/deploy.py --model '$MODEL_PATH' --port $PORT" Enter
done

echo "Started $NUM_PORTS server(s) on ports $BASE_PORT-$((BASE_PORT + NUM_PORTS - 1))"
echo "tmux attach -t $SESSION_NAME"
