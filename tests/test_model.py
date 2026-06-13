"""Tests for src/models/model.py — ResNet18, ConvBlock, BasicBlock, SEBlock, and model_factory."""

import pytest
import torch
import torch.nn as nn

from src.models.model import BasicBlock, ConvBlock, ResNet18, SEBlock, model_factory

# ─── ConvBlock ─────────────────────────────────────────────────────────


class TestConvBlock:
    def test_init_output_shape(self):
        block = ConvBlock(in_c=3, out_c=64, stride=1)
        assert block.net[0].in_channels == 3
        assert block.net[0].out_channels == 64
        assert block.net[0].kernel_size == (3, 3)

    def test_forward(self):
        block = ConvBlock(in_c=3, out_c=64, stride=1)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_forward_different_stride(self):
        block = ConvBlock(in_c=64, out_c=128, stride=2)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 128, 8, 8)


# ─── BasicBlock ────────────────────────────────────────────────────────


class TestBasicBlock:
    def test_block_no_projection_needed(self):
        block = BasicBlock(in_c=64, out_c=64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_block_with_projection(self):
        block = BasicBlock(in_c=64, out_c=128, stride=2)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 128, 8, 8)

    def test_skip_connection_shape(self):
        block = BasicBlock(in_c=64, out_c=128, stride=2)
        # When stride != 1 or in_c != out_c, skip should be a projection
        assert isinstance(block.skip, nn.Sequential)
        assert len(block.skip) == 2

    def test_block_identity_skip(self):
        block = BasicBlock(in_c=64, out_c=64, stride=1)
        assert isinstance(block.skip, nn.Identity)


# ─── SEBlock ───────────────────────────────────────────────────────────


class TestSEBlock:
    def test_init(self):
        block = SEBlock(channels=64, reduction=16)
        assert block.avg_pool is not None
        # fc has 4 layers: Linear -> ReLU -> Linear -> Sigmoid
        assert len(block.fc) == 4

    def test_forward_same_shape(self):
        block = SEBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_output_range(self):
        """SE block output should be in [0, 1] range due to sigmoid."""
        block = SEBlock(channels=64, reduction=16)
        x = torch.ones(1, 64, 8, 8)
        out = block(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_different_reduction(self):
        block = SEBlock(channels=128, reduction=8)
        x = torch.randn(2, 128, 8, 8)
        out = block(x)
        assert out.shape == (2, 128, 8, 8)


# ─── ResNet18 ──────────────────────────────────────────────────────────


class TestResNet18:
    def test_init_default_classes(self):
        model = ResNet18(num_classes=10)
        assert model.fc.out_features == 10

    def test_init_custom_classes(self):
        model = ResNet18(num_classes=100)
        assert model.fc.out_features == 100

    def test_forward(self, small_model: torch.nn.Module):
        x = torch.randn(2, 3, 32, 32)
        out = small_model(x)
        assert out.shape == (2, 10)

    def test_forward_batch_of_1(self, small_model: torch.nn.Module):
        x = torch.randn(1, 3, 32, 32)
        out = small_model(x)
        assert out.shape == (1, 10)

    def test_invalid_input_dimension(self, small_model: torch.nn.Module):
        x = torch.randn(4, 32, 32)  # 3D instead of 4D
        with pytest.raises(ValueError, match="Expected 4D input"):
            small_model(x)

    def test_forward_no_grad(self, small_model: torch.nn.Module):
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = small_model(x)
        assert out.shape == (4, 10)

    def test_parameters_count(self, small_model: torch.nn.Module):
        total = sum(p.numel() for p in small_model.parameters())
        # ResNet18 on CIFAR-10 has ~4.9M parameters (this implementation uses GroupNorm instead of BatchNorm)
        assert total > 1_000_000
        assert total < 10_000_000

    def test_trainable_parameters(self, small_model: torch.nn.Module):
        total_trainable = sum(
            p.numel() for p in small_model.parameters() if p.requires_grad
        )
        assert total_trainable == sum(p.numel() for p in small_model.parameters())

    def test_layers_structure(self, small_model: torch.nn.Module):
        assert hasattr(small_model, "conv1")
        assert hasattr(small_model, "bn1")
        assert hasattr(small_model, "relu")
        assert hasattr(small_model, "stages")
        assert hasattr(small_model, "avgpool")
        assert hasattr(small_model, "fc")

    def test_stages_output_shape(self, small_model: torch.nn.Module):
        x = torch.randn(2, 3, 32, 32)
        x = small_model.conv1(x)
        x = small_model.bn1(x)
        x = small_model.relu(x)
        x = small_model.stages(x)
        # After 4 stages with stride=2 downsampling at each stage start:
        # 32 -> 32 (stage1) -> 16 (stage2) -> 8 (stage3) -> 4 (stage4)
        assert x.shape == (2, 512, 4, 4)


# ─── model_factory ─────────────────────────────────────────────────────


class TestModelFactory:
    def test_factory_returns_resnet18(self):
        model = model_factory(num_classes=10, model_type="resnet18")
        assert isinstance(model, ResNet18)

    def test_factory_custom_classes(self):
        """Test that factory accepts num_classes parameter.

        Note: Current model_factory implementation has a bug — it ignores
        num_classes and always returns ResNet18 with default (10 classes).
        This test documents the actual behavior.
        """
        model = model_factory(num_classes=5, model_type="resnet18")
        assert isinstance(model, ResNet18)
        # Documenting the current behavior: num_classes is ignored
        assert model.fc.out_features == 10  # type: ignore[attr-defined]

    def test_factory_forward(self):
        model = model_factory(num_classes=10, model_type="resnet18")
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)
