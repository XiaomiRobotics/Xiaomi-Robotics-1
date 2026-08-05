# 🚀 Deployment Guide

This guide covers environment setup and inference server launch for **Xiaomi-Robotics-1 (XR-1)**.

XR-1 is deployed on top of the HuggingFace Transformers 🤗 ecosystem, enabling straightforward deployment for robotic manipulation tasks. By leveraging Flash Attention 2 and bfloat16 precision, the model can be loaded and run efficiently on consumer-grade GPUs.

**transformers** must be exactly `4.57.1` — other versions are unverified. We recommend **PyTorch 2.8.0** (paired with torchvision 0.23.0 and torchaudio 2.8.0), as this combination has been fully tested by our team and ensures optimal compatibility.

---

## 1️⃣ Installation Guides

Create the deploy environment (`mibot`) with its dependencies:

```bash
# Create a Conda environment with Python 3.12
conda create -n mibot python=3.12 -y
conda activate mibot

# Install PyTorch
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# Install transformers
pip install transformers==4.57.1
# Install flash-attn
pip uninstall -y ninja && pip install ninja
pip install flash-attn==2.8.3 --no-build-isolation
# or pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

sudo apt-get install -y libegl1 libgl1 libgles2
```

---

## 2️⃣ Launch the Inference Server

Inference follows a **client–server architecture** (`deploy/server.py` + `deploy/client.py`).

First, clone this repository and enter it (the launch scripts live under `scripts/`):

```bash
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1
cd Xiaomi-Robotics-1
```

You can launch the server directly from a Hugging Face model id, or first download the checkpoint and pass a local path:

```bash
# (Optional) Download the checkpoint to a local directory
export MODEL_PATH="$PWD/checkpoints/Xiaomi-Robotics-1"
hf download XiaomiRobotics/Xiaomi-Robotics-1 --local-dir "$MODEL_PATH"

# Launch the inference server (e.g. 8 servers across 8 GPUs, base port 10086)
bash scripts/deploy.sh "$MODEL_PATH" 8 8
# or, without a local download:
# bash scripts/deploy.sh XiaomiRobotics/Xiaomi-Robotics-1 8 8
```

The arguments are `<model_path> <num_ports> <num_gpus>`. The deployment script starts one server on each port beginning at `10086` and keeps the servers in the `model_servers` tmux session. To inspect them:

```bash
tmux attach -t model_servers
```

### How it works

- The **server** loads the model with `trust_remote_code=True`, Flash Attention 2, and bfloat16, and serves actions over a length-prefixed socket (`deploy/server.py`).
- The **client** sends observations via the `AutoProcessor` and decodes returned actions with `processor.decode_action(...)` (`deploy/client.py`).
- The wire protocol uses NumPy `.npz` archives with `allow_pickle=False` (not `pickle`). Keep the server on a trusted network / loopback — there is still no authentication on the socket.
- Payloads are capped at 128 MiB to prevent length-prefix memory exhaustion. `task_id` / `seed` are treated as client meta and are not forwarded into the model call.

The server must already be running before a client connects. See the benchmark evaluation guides for end-to-end examples of driving the server with a simulation client:

- RoboCasa — [`eval_robocasa/README.md`](../eval_robocasa/README.md)
- RoboCasa365 — [`eval_robocasa365/README.md`](../eval_robocasa365/README.md)
- VLABench - [`eval_vlabench/README.md`](../eval_vlabench/README.md)
