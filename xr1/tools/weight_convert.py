# Copyright (C) 2026 Xiaomi Corporation.
import argparse
from pathlib import Path

import torch
from transformers import AutoModel


def convert(model_path, output_dir, output_filename):
    attn_implementation = "flash_attention_2"
    if not torch.cuda.is_available() and torch.xpu.is_available():
        # Intel XPU: flash-attn is CUDA-only, so use PyTorch SDPA.
        attn_implementation = "sdpa"
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16,
    ).cpu()
    state_dict = {f"model.{name}": tensor for name, tensor in model.state_dict().items()}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    torch.save({"module": state_dict}, output_path)
    print(f"Saved checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="pretrained_ckpt")
    parser.add_argument("--output_filename", default="model_states.pt")
    args = parser.parse_args()
    convert(args.model_path, args.output_dir, args.output_filename)


if __name__ == "__main__":
    main()
