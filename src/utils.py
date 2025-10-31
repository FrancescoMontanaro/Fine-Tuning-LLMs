import torch
import random
import numpy as np


def get_device() -> torch.device:
    """
    Get the available device (GPU, MPS, or CPU).
    
    Returns:
        torch.device: The available device.
    """
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    # Default to CPU
    return torch.device("cpu")


def set_seed(seed: int):
    """
    Set the random seed for reproducibility across various libraries.
    
    Args:
        seed (int): The seed value to set.
    """

    # Set seed for random, numpy, and torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set seed for all CUDA devices if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)