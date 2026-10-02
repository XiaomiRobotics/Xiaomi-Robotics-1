<div align="center">

  # Xiaomi-Robotics-1

  **Breaking the Data Barrier: Scaling Robot Policy Models with Embodiment-Free Pre-training**

  [![Paper](https://img.shields.io/badge/📄-Paper-red)](https://arxiv.org/abs/2607.15330)
  [![Project Page](https://img.shields.io/badge/🌐-Project_Page-blue)](https://robotics.xiaomi.com/xiaomi-robotics-1.html)
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/collections/XiaomiRobotics/xiaomi-robotics-1)
  [![ModelScope Collection](https://img.shields.io/badge/ModelScope-Collection-624aff.svg?logo=modelscope&logoColor=white)](https://www.modelscope.cn/collections/XiaomiRobotics/Xiaomi-Robotics-1)
  [![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
  [![Papers with Code: SOTA on RoboCasa](https://paperswithcode.co/api/v1/papers/2607.15330/leaderboard-badge.svg?eval=18089&live=1)](https://paperswithcode.co/api/v1/papers/2607.15330/leaderboard-badge-link?eval=18089)

</div>

---

## 💡 About Xiaomi-Robotics-1

**Xiaomi-Robotics-1** is a robot foundation model trained on **over 100K hours of real-world manipulation trajectories**. It is a **Vision-Language-Action (VLA)** model engineered for out-of-the-box mobile manipulation in unseen environments and efficient adaptation to new tasks.

XR-1 follows a two-stage training paradigm inspired by large language models — **pre-training** for breadth, followed by **post-training** for alignment. It showcases that pre-training scaling behavior reliably transfers through post-training to real-world robot performance, with **no signs of saturation**.

XR-1 couples a pre-trained VLM (**Qwen3-VL**) with a Diffusion-Transformer (DiT) via a **Mixture-of-Transformers (MoT)** — the DiT matches the VLM in layer count but uses a smaller hidden size for faster inference.

### Key Features:

* **🧠 Strong Generalization**: Pre-trained on 100K+ hours of embodiment-free UMI trajectories spanning 1,700+ scenarios across household, commercial, industrial, and outdoor environments, enabling robust handling of complex, unseen tasks.
* **🚀 Real-Time Ready**: Optimized with asynchronous execution to minimize inference latency.
* **🛠️ Flexible Deployment**: Built on the Hugging Face `transformers` ecosystem (pinned to `4.57.1`) and optimized for consumer GPUs.
* **⚡ Data-Efficient Adaptation**: Learns entirely new complex tasks from just a few hours of demonstrations per task.

### Two-Stage Training Paradigm

**Stage 1 — Pre-training (breadth).** Uses embodiment-free UMI data to learn generalizable representations for action generation. A VLM-powered auto-labeling pipeline automatically segments long trajectories into fixed-length clips and generates descriptions of the corresponding state transitions. Trained on these annotated clips, the model learns to generate actions that bring each scene from its current state to the described target state. By making large-scale annotation practical, this approach eliminates the need for prohibitively expensive manual labeling and breaks the robot-data scarcity bottleneck that has historically constrained policy-model scaling.

**Stage 2 — Post-training (alignment).** Aligns the pre-trained model along two axes with cross-embodiment data, including in-house real-robot data, filtered open-sourced robot data, and high-quality manually-annotated UMI data:
- *Embodiment alignment* — maps general action-generation ability onto actual robots via cross-embodiment data.
- *Instruction alignment* — shifts from acting on scene-transition descriptions to understanding natural-language instructions and executing them directly.

After post-training, XR-1 can perform mobile manipulation in unseen environments with unseen objects, out of the box. The model, pre-trained on over **100K hours of UMI trajectories** and subsequently post-trained on over **10K hours of cross-embodiment data**, has been [open-sourced on Hugging Face](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B).


## 📅 Updates

- **[2026-08-03]** 💻 Released code and checkpoints for post-training, inference, and benchmark evaluation.
- **[2026-07-16]** 📄 Released technical report ([arXiv:2607.15330](https://arxiv.org/abs/2607.15330)).

---


## 🏆 Benchmark

We evaluate **Xiaomi-Robotics-1** on four standard simulation benchmarks: **RoboCasa**, **RoboCasa365**, **VLABench**, and **RoboDojo**. XR-1 achieves state-of-the-art results on all four. As of July 15, 2026, it ranks first on both the RoboCasa365 and RoboDojo leaderboards.

| Benchmark | XR-1 | 2nd Best | Relative Gain |
| :--- | :---: | :---: | :---: |
| RoboCasa | 74.5% | 72.6% | +2.6% |
| RoboCasa365 | 57.4% | 46.6% | +23.2% |
| VLABench | 59.1% | 53.2% | +11.1% |
| RoboDojo | 13.93% | 8.80% | +58.3% |

For each setting, we provide the corresponding fine-tuned checkpoint and a guide for running the evaluation:

| Benchmark | Checkpoint | Evaluation Guide |
| :--- | :--- | :--- |
| RoboCasa | [Xiaomi-Robotics-1-RoboCasa](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa) | [`eval_robocasa/README.md`](eval_robocasa/README.md) |
| RoboCasa365 | [Xiaomi-Robotics-1-RoboCasa365](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365) | [`eval_robocasa365/README.md`](eval_robocasa365/README.md) |
| VLABench | [Xiaomi-Robotics-1-VLABench](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-VLABench) | [`eval_vlabench/README.md`](eval_vlabench/README.md) |
| RoboDojo | — | [official docs](https://github.com/robodojo-benchmark/RoboDojo) |

### 🏠 Efficient Adaptation to New Real-World Tasks

XR-1 adapts to new complex tasks from minimal demonstration data, substantially outperforming the π0.5 baseline:

| Task | XR-1 (< 10h/task on avg.) | π0.5 (< 10h/task on avg.) | XR-1 (< 40h/task on avg.) | π0.5 (< 40h/task on avg.) |
| :--- | :---: | :---: | :---: | :---: |
| Phone Packing | 70% | 30% | 80% | 40% |
| Printer Refilling | 70% | 20% | 60% | 20% |
| Laundry Loading | 80% | 40% | 100% | 50% |
| Box Packing | 80% | 70% | 100% | 100% |
| **Overall** | **75%** | **40%** | **85%** | **53%** |

Each cell shows success rate, higher is better. XR-1 = Xiaomi-Robotics-1.


## 🛠️ Post-Training

Want to fine-tune XR-1 on your own data? The post-training and deployment code is available in [`xr1/`](xr1/). See **[`xr1/README.md`](xr1/README.md)** for details on environment setup, checkpoint conversion, data formats, training, and deployment.

> **Intel GPUs (XPU)** are supported for inference, deployment, and checkpoint conversion; see **[`docs/CUDA_XPU_GUIDE.md`](docs/CUDA_XPU_GUIDE.md)** for setup. Post-training (DeepSpeed) is currently validated on CUDA; XPU post-training is tracked as a follow-up.

For discussions, scan the QR code below to join our WeChat group.

<img src="xr1/assets/wechat_group.jpg" alt="Post-training reproduction WeChat group QR code" width="220">


## 📚 Citation

If you find this project useful, please consider citing:

```bibtex
@article{team2026xiaomi,
  title={Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories},
  author={Team, Xiaomi Robotics and Guo, Jun and Jin, Piaopiao and Li, Jason and Li, Peiyan and Li, Yingyan and Liu, Futeng and Peng, Wanli and Qin, Optimus and Su, Yifei and others},
  journal={arXiv preprint arXiv:2607.15330},
  year={2026}
}
```


## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
