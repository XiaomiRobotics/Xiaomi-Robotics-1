# 🧭 Running Xiaomi-Robotics-1 on NVIDIA CUDA or Intel XPU

Step-by-step guide for running the three benchmark modes, **RoboCasa**, **RoboCasa365** and **VLABench**, on either an **NVIDIA GPU (CUDA)** or an **Intel GPU (XPU)**.

The code picks the backend at runtime from the installed PyTorch build. You don't need to pass any flags:

| PyTorch build | Device | Attention | Precision |
| :--- | :--- | :--- | :--- |
| CUDA (`+cu128`) | `cuda` | Flash Attention 2 | bfloat16 |
| XPU (`+xpu`) | `xpu` | PyTorch SDPA | bfloat16 |

The CUDA code is left exactly as released. The XPU support is only added lines, and they run only when CUDA isn't available and an Intel GPU is present.

> **Architecture.** Every benchmark uses the same client–server split:
> - The **server** (`deploy/server.py`, started by `scripts/deploy.sh`) runs in the `mibot` environment and is the only component that uses the GPU.
> - The **client** (`scripts/launch_<benchmark>.sh`) runs the simulator in its own conda environment, builds inputs with the Hugging Face `AutoProcessor`, and talks to the server over a socket.
>
> Only the **server** setup differs between CUDA and XPU. Simulator setup, clients and launch commands are the same for both.

---

## 0. Prerequisites

| | NVIDIA CUDA | Intel XPU |
| :--- | :--- | :--- |
| Hardware | NVIDIA GPU with bfloat16 (Ampere or newer) | Intel Arc (A/B-series, Arc Pro) or Data Center GPU Max |
| Driver | NVIDIA driver with CUDA 12.8 support | [Intel GPU driver for PyTorch](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) (Level Zero runtime) |
| Check | `nvidia-smi` | `ls /dev/dri/render*` and `clinfo -l` (or `sycl-ls`) |
| GPU memory | ≥ 12 GB per server | ≥ 12 GB per server |

Common system packages for headless rendering:

```bash
sudo apt-get install -y tmux libegl1 libgl1 libgles2
```

Clone the repository. All commands below run from its root:

```bash
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1
cd Xiaomi-Robotics-1
```

---

## 1. Server environment (`mibot`)

Pick **one** of the two variants. `transformers` must be exactly `4.57.1`.

### 1a. NVIDIA CUDA

```bash
conda create -n mibot python=3.12 -y
conda activate mibot

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1
pip uninstall -y ninja && pip install ninja
pip install flash-attn==2.8.3 --no-build-isolation
```

### 1b. Intel XPU

Verified with Python 3.14, torch 2.14.0+xpu and transformers 4.57.1:

```bash
conda create -n mibot python=3.14 -y
conda activate mibot

pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/xpu
pip install transformers==4.57.1
# flash-attn is CUDA-only and NOT needed; the server uses PyTorch SDPA on XPU.
```

### 1c. Sanity-check the server environment

```bash
python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available(), 'xpu', torch.xpu.is_available())"
```

Expected: `... cuda True xpu False` on NVIDIA, or `...+xpu cuda False xpu True` on Intel.

---

## 2. Download the checkpoints

The server and the client must both be able to read the same model directory:

```bash
hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa    --local-dir checkpoints/Xiaomi-Robotics-1-RoboCasa
hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 --local-dir checkpoints/Xiaomi-Robotics-1-RoboCasa365
hf download XiaomiRobotics/Xiaomi-Robotics-1-VLABench    --local-dir checkpoints/Xiaomi-Robotics-1-VLABench
```

---

## 3. Start the inference server (all modes, CUDA and XPU)

The command is identical on both backends. Only the checkpoint changes per benchmark:

```bash
conda activate mibot
export MODEL_PATH="$PWD/checkpoints/Xiaomi-Robotics-1-RoboCasa"   # or -RoboCasa365 / -VLABench
bash scripts/deploy.sh "$MODEL_PATH" <num_ports> <num_gpus>       # e.g. 8 8, or 1 1 on a single GPU
```

- One server is started per port, from `10086` upwards, inside the `model_servers` tmux session. Server *i* is pinned to GPU `i % num_gpus`.
- GPU pinning works on both backends. The script sets `CUDA_VISIBLE_DEVICES=<gpu>` (NVIDIA, as in the original) and also exports `ONEAPI_DEVICE_SELECTOR=level_zero:<gpu>` (Intel). Each PyTorch build reads only its own variable. (`level_zero` is the SYCL backend name; the SYCL runtime rejects `xpu:<gpu>`.)
- Wait until every tmux window prints the ready lines below before starting the client:

```text
Model loaded.                 # NVIDIA (unchanged original output)
Model loaded on xpu.          # Intel
Server running on localhost:10086...
```

Inspect with `tmux attach -t model_servers`. Stop with `tmux kill-session -t model_servers`.

---

## 4. Mode A: RoboCasa

**Checkpoint:** `Xiaomi-Robotics-1-RoboCasa` · **robot type:** `robocasa_mg` · **actions:** 10 × 7

1. Create the simulator environment once (RoboCasa **v0.2**):

   ```bash
   conda create -n robocasa python=3.10 -y
   conda activate robocasa
   git clone https://github.com/ARISE-Initiative/robosuite && (cd robosuite && pip install -e .)
   git clone https://github.com/robocasa/robocasa --branch v0.2 && (cd robocasa && pip install -e .)
   pip install transformers==4.57.1 imageio[ffmpeg] scipy tyro tqdm
   python -m robocasa.scripts.setup_macros
   python -m robocasa.scripts.download_kitchen_assets
   ```

2. Start the server with `MODEL_PATH=$PWD/checkpoints/Xiaomi-Robotics-1-RoboCasa` (§3).

3. In a second terminal, run the client:

   ```bash
   conda activate robocasa
   bash scripts/launch_robocasa.sh <num_ports> ./eval_robocasa/eval_logs "$PWD/checkpoints/Xiaomi-Robotics-1-RoboCasa"
   ```

4. Results are merged into `eval_robocasa/eval_logs/merged_results.json`. The reference average is 74.2 %.

Full details: [`eval_robocasa/README.md`](../eval_robocasa/README.md).

---

## 5. Mode B: RoboCasa365

**Checkpoint:** `Xiaomi-Robotics-1-RoboCasa365` · **robot type:** `robocasa365` · **actions:** 16 × 12

1. Create the simulator environment `robocasa_365` by following the [official RoboCasa installation guide](https://robocasa.ai/docs/build/html/introduction/installation.html), then:

   ```bash
   pip install transformers==4.57.1 imageio[ffmpeg] tqdm scipy
   ```

2. Start the server with `MODEL_PATH=$PWD/checkpoints/Xiaomi-Robotics-1-RoboCasa365` (§3).

3. Smoke test (one task, one trial), then run the full evaluation:

   ```bash
   CONDA_ENV=robocasa_365 NUM_TRIALS=1 bash scripts/launch_robocasa365.sh \
       1 eval_results/robocasa365-smoke "$PWD/checkpoints/Xiaomi-Robotics-1-RoboCasa365" \
       --task-name CloseBlenderLid --horizon 20

   CONDA_ENV=robocasa_365 bash scripts/launch_robocasa365.sh \
       <num_ports> eval_results/robocasa365 "$PWD/checkpoints/Xiaomi-Robotics-1-RoboCasa365"
   ```

4. Results are written to `eval_results/robocasa365/<run-id>/summary.json`. The reference success rate is 57.28 %.

Full details: [`eval_robocasa365/README.md`](../eval_robocasa365/README.md).

---

## 6. Mode C: VLABench

**Checkpoint:** `Xiaomi-Robotics-1-VLABench` · **robot type:** `vlabench_choice` · **actions:** 10 × 7

1. Create the simulator environment `vlabench` by following [`eval_vlabench/README.md`](../eval_vlabench/README.md#2-vlabench-simulation-environment-vlabench).
   - **Intel machines:** the client only uses PyTorch on the CPU, so install the CPU wheel instead of `cu128` in that step:

     ```bash
     python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
     ```

2. Start the server with `MODEL_PATH=$PWD/checkpoints/Xiaomi-Robotics-1-VLABench` (§3).

3. Run the client:

   ```bash
   conda activate vlabench
   export VLABENCH_ROOT=/path/to/VLABench MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false
   bash scripts/launch_vlabench.sh <num_ports> ./eval_vlabench/eval_logs "$PWD/checkpoints/Xiaomi-Robotics-1-VLABench"
   ```

4. Per-track results are merged by `eval_vlabench/merge_results.py` into `./eval_vlabench/eval_logs`.

---

## 7. Post-trained checkpoints (`xr1/` stack)

The `xr1/` stack has two parts:

- **Inference server** (`xr1/mibot/server/deploy.py`, started by `xr1/scripts/deploy.sh`) — **works on both CUDA and XPU**. It auto-selects `cuda:0` / `xpu:0` and Flash Attention 2 / SDPA the same way as the main server.

  ```bash
  cd xr1
  bash scripts/deploy.sh outputs/project_xiaomi-robotics-1/posttrain <num_ports> <num_gpus>
  ```

  On Intel XPU the `xr1` model imports `liger-kernel`, which needs the Intel Triton backend. If `pip install -e .` pulled in the plain `triton` wheel, replace it:

  ```bash
  pip uninstall -y triton && pip install --force-reinstall --no-deps triton-xpu --index-url https://download.pytorch.org/whl/xpu
  ```

- **Post-training** (`xr1/scripts/train.sh`) uses DeepSpeed and is **NVIDIA CUDA only**.

Full details: [`xr1/README.md`](../xr1/README.md).

---

## 8. Troubleshooting

| Symptom | Cause / fix |
| :--- | :--- |
| `ONEAPI_DEVICE_SELECTOR parsing error. Backend is required but missing from "xpu:0"` | Use `level_zero:<gpu>`, not `xpu:<gpu>` (`scripts/deploy.sh` already does this). |
| `flash_attn` import error on Intel | You're on the CUDA recipe; use §1b. XPU never uses `flash-attn`. |
| `torch.xpu.is_available()` is `False` | The Intel GPU driver / Level Zero runtime is missing, or the user isn't in the `render` group (`sudo usermod -aG render $USER`, then log in again). |
| The first request is slower than later ones | Expected on XPU: the first request compiles GPU kernels. With torch 2.14.0+xpu on an Arc B390, a request (including client-side preprocessing) took about 9–10 s. |
| `Robot type ... is missing from the checkpoint` | The server/client checkpoint doesn't match the benchmark (see §4–§6 for the right pairing). |
| `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` during a full simulator run | The simulator renders camera images with EGL/OpenGL on the same Intel GPU that the model uses via Level Zero/SYCL. On some Arc GPUs the two runtimes contend and the driver resets the device. The model itself is stable (60+ consecutive forwards pass). Fix: run the simulator's rendering on the CPU (install OSMesa and set `MUJOCO_GL=osmesa`), or use a Data Center GPU Max where EGL + Level Zero coexist. |
