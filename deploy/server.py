# Copyright (C) 2026 Xiaomi Corporation.
from io import BytesIO
import socket
import struct
import time
import argparse
import traceback

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel


class Server:
    def __init__(self, model_path, host, port):
        self.host = host
        self.port = port

        print("Loading model...")
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16).cuda().to(torch.bfloat16)
        print("Model loaded.")

    def _recv_all(self, conn, length):
        data = b""
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _recv(self, conn):
        data_len_bytes = self._recv_all(conn, 4)
        if not data_len_bytes:
            return None
        data_len = struct.unpack(">I", data_len_bytes)[0]
        data = self._recv_all(conn, data_len)
        if not data:
            return None
        # allow_pickle=False: never deserialize untrusted pickle over the socket.
        with np.load(BytesIO(data), allow_pickle=False) as payload:
            return {
                key: payload[key].item() if payload[key].ndim == 0 else torch.from_numpy(payload[key].copy())
                for key in payload.files
            }

    @staticmethod
    def _send(conn, actions):
        actions = actions.detach().cpu()
        array = actions.float().numpy() if actions.dtype == torch.bfloat16 else actions.numpy()
        buffer = BytesIO()
        np.savez(buffer, actions=array)
        data = buffer.getvalue()
        conn.sendall(struct.pack(">I", len(data)) + data)

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"Server running on {self.host}:{self.port}...")

            while True:
                conn, addr = server_socket.accept()

                try:
                    request_count = 0
                    with tqdm(desc="Processing Requests", unit=" req") as pbar:
                        while True:
                            input_data = self._recv(conn)
                            if input_data is None:
                                break

                            tic = time.time()

                            data = {
                                key: (
                                    value.to(device=self.model.device, dtype=self.model.dtype)
                                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                                    else value.to(device=self.model.device)
                                    if isinstance(value, torch.Tensor)
                                    else value
                                )
                                for key, value in input_data.items()
                            }

                            outputs = self.model(**data)
                            self._send(conn, outputs.actions)

                            toc = time.time()
                            request_count += 1
                            pbar.update(1)
                            pbar.set_postfix(
                                {
                                    "avg_time": f"{(toc - tic) * 1000:.2f}ms",
                                }
                            )

                except Exception as e:
                    print(f"Error handling connection: {e}")
                    traceback.print_exc()
                finally:
                    conn.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model dir.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Bind address. Prefer localhost; the wire protocol has no authentication.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10086,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    server = Server(model_path=args.model, host=args.host, port=args.port)
    server.serve()
