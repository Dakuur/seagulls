import torch

def get_device():
    """
    Get the device to be used for PyTorch operations.
    Returns 'cuda' if a GPU is available, otherwise returns 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")