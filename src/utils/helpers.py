"""Common helper functions."""

from pathlib import Path

import torch
from safetensors.torch import load_file, load_model, save_model
from torch import nn

from src.utils.logger import logger


def get_device() -> torch.device:
    """Get the available compute device.

    Returns:
        torch.device: CUDA if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def num_trainable_params(model: torch.nn.Module) -> int:
    """Calculate the total number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        int: Total number of trainable parameters.
    """

    if model is None:
        raise ValueError("Model cannot be None")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_params(model: torch.nn.Module) -> int:
    """Calculate the total number of parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        int: Total number of parameters.
    """

    if model is None:
        raise ValueError("Model cannot be None")
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: nn.Module, checkpoint_dir: str) -> str:
    """Save the model checkpoint to the specified directory.

    Args:
        model (nn.Module): The PyTorch model to save.
        checkpoint_dir (str): Directory path where the checkpoint file will be saved.

    Returns:
        str: The full path to the saved checkpoint file.
    """

    if not checkpoint_dir:
        raise ValueError("Invalid path")
    if model is None:
        raise ValueError("Model cannot be None")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/model.safetensors"
    save_model(model, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model: nn.Module, checkpoint_address: str) -> None:
    """Load the model checkpoint from the specified file.

    Args:
        model (nn.Module): The PyTorch model to load the checkpoint into.
        checkpoint_path (str): The full path to the checkpoint file.
    """

    checkpoint_path = Path(checkpoint_address)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    device = get_device()
    try:
        load_model(model, checkpoint_path, device=str(device))
    except (TypeError, KeyError):
        model.load_state_dict(
            load_file(checkpoint_path, map_location=device, weights_only=True)
        )
        model.to(device)
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
