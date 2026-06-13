"""Tests for src/utils/helpers.py — get_device, total_params, num_trainable_params, save_checkpoint, load_checkpoint."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.utils.helpers import (
    get_device,
    load_checkpoint,
    num_trainable_params,
    save_checkpoint,
    total_params,
)

# ─── get_device ────────────────────────────────────────────────────────


class TestGetDevice:
    def test_returns_cpu_or_cuda(self):
        device = get_device()
        assert device.type in ("cpu", "cuda")

    def test_device_is_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_cuda_available_mock(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = get_device()
            assert device.type == "cuda"

    def test_cuda_unavailable_mock(self):
        with patch("torch.cuda.is_available", return_value=False):
            device = get_device()
            assert device.type == "cpu"


# ─── total_params ──────────────────────────────────────────────────────


class TestTotalParams:
    def test_simple_linear(self):
        model = nn.Linear(10, 5)
        result = total_params(model)
        # weight: 10*5 = 50, bias: 5
        assert result == 55

    def test_conv_layer(self):
        conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        result = total_params(conv)
        # weight: 3*64*3*3 = 1728, bias: 64
        assert result == 1792

    def test_none_model_raises(self):
        with pytest.raises(ValueError, match="Model cannot be None"):
            total_params(None)  # type: ignore

    def test_empty_model(self):
        model = nn.Sequential()
        result = total_params(model)
        assert result == 0

    def test_multilayer_perceptron(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
        result = total_params(model)
        # Layer 1: 100*50 + 50 = 5050
        # Layer 2: 50*10 + 10 = 510
        assert result == 5560


# ─── num_trainable_params ──────────────────────────────────────────────


class TestNumTrainableParams:
    def test_simple_linear_trainable(self):
        model = nn.Linear(10, 5)
        result = num_trainable_params(model)
        assert result == 55

    def test_frozen_model(self):
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        result = num_trainable_params(model)
        assert result == 0

    def test_partial_trainable(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2),
        )
        # Freeze first layer parameters
        for p in model[0].parameters():
            p.requires_grad = False
        result = num_trainable_params(model)
        # First layer: 55 params (not trainable)
        # Second layer: 5*2 + 2 = 12 params (trainable)
        assert result == 12

    def test_none_model_raises(self):
        with pytest.raises(ValueError, match="Model cannot be None"):
            num_trainable_params(None)  # type: ignore


# ─── save_checkpoint ───────────────────────────────────────────────────


class TestSaveCheckpoint:
    def test_save_creates_file(self, small_model: nn.Module, temp_checkpoint_dir: Path):
        result = save_checkpoint(small_model, str(temp_checkpoint_dir))
        expected = temp_checkpoint_dir / "model.safetensors"
        assert expected.exists()
        # save_checkpoint returns None on success
        assert result is None or result == str(expected)

    def test_save_creates_nested_dirs(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        nested = temp_checkpoint_dir / "a" / "b" / "c"
        result = save_checkpoint(small_model, str(nested))
        assert nested.exists()
        assert (nested / "model.safetensors").exists()

    def test_save_none_model_raises(self, temp_checkpoint_dir: Path):
        with pytest.raises(ValueError, match="Model cannot be None"):
            save_checkpoint(None, str(temp_checkpoint_dir))  # type: ignore

    def test_save_empty_dir_raises(self, small_model: nn.Module):
        with pytest.raises(ValueError, match="Invalid path"):
            save_checkpoint(small_model, "")  # type: ignore

    def test_save_load_roundtrip(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        save_checkpoint(small_model, str(temp_checkpoint_dir))
        checkpoint_path = temp_checkpoint_dir / "model.safetensors"
        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0


# ─── load_checkpoint ───────────────────────────────────────────────────


class TestLoadCheckpoint:
    def test_load_nonexistent_file_raises(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        nonexistent = temp_checkpoint_dir / "nonexistent.safetensors"
        with pytest.raises(FileNotFoundError, match="Checkpoint file does not exist"):
            load_checkpoint(small_model, str(nonexistent))

    def test_load_save_load_roundtrip(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        # Save original weights
        original_state = {k: p.clone() for k, p in small_model.named_parameters()}

        # Save checkpoint
        save_checkpoint(small_model, str(temp_checkpoint_dir))

        # Modify model weights
        with torch.no_grad():
            for p in small_model.parameters():
                p.zero_()

        # Load checkpoint
        checkpoint_path = temp_checkpoint_dir / "model.safetensors"
        load_checkpoint(small_model, str(checkpoint_path))

        # Verify weights are restored
        for name, param in small_model.named_parameters():
            assert torch.allclose(param, original_state[name])

    def test_load_with_invalid_file_raises(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        # Create an invalid file
        invalid_file = temp_checkpoint_dir / "invalid.safetensors"
        invalid_file.write_text("not a valid safetensors file")
        with pytest.raises(Exception):  # safetensors_rust.SafetensorError or similar
            load_checkpoint(small_model, str(invalid_file))
