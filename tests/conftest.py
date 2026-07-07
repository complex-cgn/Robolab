"""Shared pytest fixtures for Robolab tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml


@pytest.fixture()
def temp_checkpoint_dir() -> Generator[Path, Any, None]:
    """Provide a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def temp_config_path(temp_checkpoint_dir: Path) -> Generator[Path, Any, None]:
    """Provide a temporary YAML config with valid Robolab structure."""
    config_data = {
        "hyperparams": {
            "logging_level": "INFO",
            "checkpoint_dir": str(temp_checkpoint_dir),
            "random_seed": 42,
            "num_classes": 10,
            "early_stopping_patience": 10,
        },
        "trainparams": {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "num_epochs": 2,
            "weight_decay": 5e-3,
            "criterion": "CrossEntropyLoss",
            "optimizer": "AdamW",
            "dtype": "float32",
            "warmup_epochs": 1,
            "max_grad_norm": 1.0,
            "accumulation_steps": 1,
        },
        "testparams": {
            "dtype": "float32",
        },
        "dataset": {
            "name": "CIFAR10",
            "train_split": 0.9,
            "validation_split": 0.1,
            "test_split": 0.0,
        },
    }
    config_path = temp_checkpoint_dir / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    yield config_path


@pytest.fixture()
def small_model() -> torch.nn.Module:
    """Provide a minimal ResNet18 model for fast unit tests."""
    from src.models.model import ResNet18

    return ResNet18(num_classes=10)


@pytest.fixture()
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Provide a small dummy batch (batch_size=4)."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels


@pytest.fixture()
def device() -> torch.device:
    """Provide the compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
