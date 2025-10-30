import torch


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