# Copyright (C) 2026 Xiaomi Corporation.
import math
import os
import time
from io import BytesIO
import socket
import struct

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from transformers import AutoProcessor

torch.set_printoptions(3, sci_mode=False)

MAX_PAYLOAD_BYTES = 128 * 1024 * 1024


class Client:
    def __init__(self, host="localhost", port=10086, model_path=None):
        self.host = host
        self.port = port
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False) if model_path else None
        self._connect_with_retry(max_retries=None, retry_interval=1)
        print(f"Client connected to server at {self.host}:{self.port}.")

    def _connect_with_retry(self, max_retries=None, retry_interval=1):
        """Connect with retry logic. max_retries=None implies infinite."""
        retry_count = 0
        while True:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                return
            except (ConnectionRefusedError, socket.error) as e:
                retry_count += 1
                time.sleep(retry_interval)
                if max_retries is not None and retry_count >= max_retries:
                    raise ConnectionError(f"Failed to connect to {self.host}:{self.port} after {retry_count} retries: {e}") from e

    def _send_with_length_prefix(self, data):
        arrays = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                arrays[key] = value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
            else:
                arrays[key] = np.asarray(value)
        buffer = BytesIO()
        np.savez(buffer, **arrays)
        serialized = buffer.getvalue()
        self.client_socket.sendall(struct.pack(">I", len(serialized)) + serialized)

    def _recv_all(self, length):
        data = b""
        while len(data) < length:
            packet = self.client_socket.recv(length - len(data))
            if not packet:
                raise ConnectionError("Connection closed while receiving response.")
            data += packet
        return data

    def _recv_with_length_prefix(self):
        len_data = self._recv_all(4)
        data_len = struct.unpack(">I", len_data)[0]
        if data_len <= 0 or data_len > MAX_PAYLOAD_BYTES:
            raise ValueError(f"payload length {data_len} outside allowed range (1..{MAX_PAYLOAD_BYTES})")
        data = self._recv_all(data_len)
        with np.load(BytesIO(data), allow_pickle=False) as payload:
            return torch.from_numpy(payload["actions"].copy())

    def __call__(self, **data):
        robot_type = data.get("task_id")
        self._send_with_length_prefix(data)
        actions = self._recv_with_length_prefix()
        action = self.processor.decode_action(actions, robot_type=robot_type)
        return action

    def close(self):
        self.client_socket.close()
        print("Client connection closed.")
