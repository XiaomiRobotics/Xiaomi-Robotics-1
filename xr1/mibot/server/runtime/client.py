# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

from io import BytesIO
import socket
import struct

import numpy as np
import torch
from transformers import AutoProcessor

from mibot.utils.io import compose_state, recover_action, resize_image, split_action

MAX_PAYLOAD_BYTES = 128 * 1024 * 1024


class Client:
    def __init__(self, host: str = "localhost", port: int = 10086) -> None:
        self.socket = socket.create_connection((host, port))
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        self.processor.tokenizer.padding_side = "right"

    @staticmethod
    def _recv_all(sock, length):
        data = b""
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                raise ConnectionError("connection closed while receiving response")
            data += packet
        return data

    def _send(self, payload):
        arrays = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                arrays[key] = value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
            else:
                arrays[key] = np.asarray(value)
        buffer = BytesIO()
        np.savez(buffer, **arrays)
        data = buffer.getvalue()
        self.socket.sendall(struct.pack(">I", len(data)) + data)

    def _recv(self):
        size = struct.unpack(">I", self._recv_all(self.socket, 4))[0]
        if size <= 0 or size > MAX_PAYLOAD_BYTES:
            raise ValueError(f"payload length {size} outside allowed range (1..{MAX_PAYLOAD_BYTES})")
        with np.load(BytesIO(self._recv_all(self.socket, size)), allow_pickle=False) as payload:
            return torch.from_numpy(payload["action"].copy())

    @staticmethod
    def _messages(instruction, ego_obs, left_wrist_obs, right_wrist_obs):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The following observations are captured from multiple views.\n# Ego View\n",
                    },
                    {"type": "image", "image": ego_obs},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": left_wrist_obs},
                    {"type": "text", "text": "\n# Right-Wrist View\n"},
                    {"type": "image", "image": right_wrist_obs},
                    {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    @staticmethod
    def _crop_bbox(image, bbox):
        if bbox is None:
            return image
        top, left, height, width = (int(value) for value in bbox)
        return image.crop((left, top, left + width, top + height))

    def __call__(
        self,
        robot_state,
        ego_obs,
        left_wrist_obs,
        right_wrist_obs,
        instruction,
        action_prefix=None,
        crop_bboxes=None,
    ):
        crop_bboxes = crop_bboxes or (None, None, None)
        images = (ego_obs, left_wrist_obs, right_wrist_obs)
        ego_obs, left_wrist_obs, right_wrist_obs = [
            resize_image(self._crop_bbox(image, bbox), factor=32, max_pixels=160000)
            for image, bbox in zip(images, crop_bboxes)
        ]

        payload = self.processor.apply_chat_template(
            [self._messages(instruction, ego_obs, left_wrist_obs, right_wrist_obs)],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )
        payload["state"] = torch.from_numpy(
            compose_state(
                left_gripper=np.asarray(robot_state["left_gripper_pos"], dtype=np.float32),
                left_joint=np.asarray(robot_state["left_arm_joint"], dtype=np.float32),
                right_gripper=np.asarray(robot_state["right_gripper_pos"], dtype=np.float32),
                right_joint=np.asarray(robot_state["right_arm_joint"], dtype=np.float32),
            )
        )[None]
        if action_prefix is not None:
            action_prefix = np.asarray(action_prefix, dtype=np.float32)
            action = np.zeros((30, 60), dtype=np.float32)
            action[: len(action_prefix)] = action_prefix
            payload["action"] = torch.from_numpy(action)[None]
            payload["prefix_length"] = len(action_prefix)

        self._send(payload)
        action = self._recv()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        if action.shape == (1, 30, 60):
            action = action[0]
        if action.shape != (30, 60):
            raise ValueError(f"expected server output shape (30, 60), got {action.shape}")

        return {
            "raw_action": action,
            "action_components": split_action(action),
            "action_targets": recover_action(action, robot_state),
        }

    def close(self) -> None:
        self.socket.close()
