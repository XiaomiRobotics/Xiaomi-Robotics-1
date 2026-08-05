# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

from io import BytesIO
import socket
import struct
import time

import numpy as np
import torch
from tqdm import tqdm

from mibot.utils.io import ACTION_EPS, denormalize_action

MAX_PAYLOAD_BYTES = 128 * 1024 * 1024


class Server:
    def __init__(self, host: str, port: int, model, mean, std, q01, q99, action_mask, device: str) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.device = device
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.q01 = q01.to(device)
        self.q99 = q99.to(device)
        self.action_mask = action_mask.to(device)

    @staticmethod
    def _recv_all(conn, length):
        data = b""
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _recv(self, conn):
        head = self._recv_all(conn, 4)
        if not head:
            return None
        size = struct.unpack(">I", head)[0]
        if size <= 0 or size > MAX_PAYLOAD_BYTES:
            raise ValueError(f"payload length {size} outside allowed range (1..{MAX_PAYLOAD_BYTES})")
        body = self._recv_all(conn, size)
        if body is None:
            return None
        with np.load(BytesIO(body), allow_pickle=False) as payload:
            return {
                key: payload[key].item() if payload[key].ndim == 0 else torch.from_numpy(payload[key].copy())
                for key in payload.files
            }

    @staticmethod
    def _send(conn, payload):
        payload = payload.detach().cpu()
        action = payload.float().numpy() if payload.dtype == torch.bfloat16 else payload.numpy()
        buffer = BytesIO()
        np.savez(buffer, action=action)
        data = buffer.getvalue()
        conn.sendall(struct.pack(">I", len(data)) + data)

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            print(f"Server running on {self.host}:{self.port}...")

            while True:
                conn, _ = server.accept()
                try:
                    with tqdm(desc="Processing Requests", unit=" req") as pbar:
                        while True:
                            request = self._recv(conn)
                            if request is None:
                                break

                            tic = time.time()
                            batch = {
                                key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
                                for key, value in request.items()
                            }
                            mask = self.action_mask.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1)

                            if "action" in batch:
                                batch["action"] = ((batch["action"] - self.mean) / (self.std + ACTION_EPS)) * mask
                            else:
                                batch["action"] = torch.zeros(
                                    (batch["input_ids"].shape[0], *self.mean.shape),
                                    device=self.device,
                                    dtype=torch.bfloat16,
                                )

                            batch["action_mask"] = mask
                            state = batch["state"]
                            valid = self.q99 > self.q01
                            normalized_state = torch.zeros_like(state)
                            normalized_state[..., valid[0]] = (
                                2.0
                                * (state[..., valid[0]] - self.q01[..., valid[0]])
                                / (self.q99[..., valid[0]] - self.q01[..., valid[0]] + ACTION_EPS)
                                - 1.0
                            )
                            batch["state"] = normalized_state.clamp(-1.0, 1.0)

                            action = self.model.generate(batch)
                            action = denormalize_action(action * mask, self.mean, self.std) * mask
                            self._send(conn, action.cpu())

                            pbar.update(1)
                            pbar.set_postfix({"avg_time": f"{(time.time() - tic) * 1000:.2f}ms"})
                except Exception as error:
                    print(f"Error handling connection: {error}")
                finally:
                    conn.close()


if __name__ == "__main__":
    raise SystemExit("Use `python mibot/server/deploy.py --model <dir>` to start the inference server.")
