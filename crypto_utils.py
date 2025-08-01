import torch
import numpy as np
import hashlib

def xor_bytes(data: bytes, key: bytes) -> bytes:
    key = hashlib.sha256(key).digest()
    key_len = len(key)
    return bytes([b ^ key[i % key_len] for i, b in enumerate(data)])

def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.cpu().numpy().astype(np.float32).tobytes()

def bytes_to_tensor(byte_data: bytes, shape, dtype=torch.float32) -> torch.Tensor:
    array = np.frombuffer(byte_data, dtype=np.float32).reshape(shape)
    return torch.tensor(array, dtype=dtype)
