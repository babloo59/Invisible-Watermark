import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch

def save_image(tensor: torch.Tensor, path: str):
    vutils.save_image(tensor, path)

def tensor_to_np(tensor: torch.Tensor):
    img = tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def compute_metrics(img1_t: torch.Tensor, img2_t: torch.Tensor):
    img1, img2 = tensor_to_np(img1_t), tensor_to_np(img2_t)
    return psnr(img1, img2, data_range=255), ssim(img1, img2, channel_axis=2, data_range=255)

def compute_array_metrics(a: np.ndarray, b: np.ndarray):
    """
    PSNR & SSIM for arbitrary 2-D/3-D tensors (e.g. conv weights).
    Handles tiny kernels and avoids float32 overflow.
    """
    # --- promote & normalise ---
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    max_abs = max(np.abs(a).max(), np.abs(b).max())
    if max_abs == 0:
        # both zero → identical
        return float('inf'), 1.0
    a /= max_abs
    b /= max_abs

    # --- PSNR ---
    mse = np.mean((a - b) ** 2)
    psnr_val = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)  # data_range = 1

    # --- SSIM ---
    h, w = a.shape[:2]
    win = max(3, min(7, h, w))          # odd & ≤ min dim
    if win % 2 == 0:
        win -= 1
    ssim_val = ssim(
        a, b,
        win_size=win,
        channel_axis=None if a.ndim == 2 else 2,
        data_range=1.0
    )
    return psnr_val, ssim_val