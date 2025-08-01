# weight_utils.py

import torch
from crypto_utils import xor_bytes, tensor_to_bytes, bytes_to_tensor

def encrypt_model_weights(model: torch.nn.Module, key: str):
    encrypted_weights = {}
    shapes = {}
    for name, param in model.state_dict().items():
        data_bytes = tensor_to_bytes(param)
        encrypted_bytes = xor_bytes(data_bytes, key.encode())
        encrypted_weights[name] = encrypted_bytes
        shapes[name] = param.shape
    return encrypted_weights, shapes

def decrypt_model_weights(model: torch.nn.Module, encrypted_weights: dict, shapes: dict, key: str):
    state_dict = {}
    for name, encrypted_bytes in encrypted_weights.items():
        decrypted_bytes = xor_bytes(encrypted_bytes, key.encode())
        tensor = bytes_to_tensor(decrypted_bytes, shapes[name])
        state_dict[name] = tensor
    model.load_state_dict(state_dict)
