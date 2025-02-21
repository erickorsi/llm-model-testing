"""
GPU detection and device management.
"""
import torch

def get_device():
    """Returns the best available device (CUDA if available, otherwise CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device} ("
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )
    return device

DEVICE = get_device()
