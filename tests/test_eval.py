"""Tests for src/eval/eval.py — evaluate function with comprehensive metrics."""

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.eval.eval import evaluate
from src.models.model import ResNet18


def _create_dummy_data_loader(
    num_batches: int = 2,
    batch_size: int = 4,
    num_classes: int = 10,
) -> torch.utils.data.DataLoader[Any]:
    """Create a dummy data loader with random tensors for testing."""

    class DummyDataset(torch.utils.data.Dataset[Any]):
        def __init__(self, n: int, bs: int, nc: int) -> None:
            self.n = n
            self.bs = bs
            self.nc = nc

        def __len__(self) -> int:
            return self.n * self.bs

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            x = torch.randn(3, 32, 32)
            y = torch.tensor(int(np.random.randint(0, self.nc)))
            return x, y

    return torch.utils.data.DataLoader(
        DummyDataset(num_batches, batch_size, num_classes),
        batch_size=batch_size,
        shuffle=False,
    )


def _create_simple_model(num_classes: int = 10) -> nn.Module:
    """Create a tiny model for fast eval tests (not ResNet18, just a simple CNN)."""

    class TinyCNN(nn.Module):
        def __init__(self, nc: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.classifier = nn.Linear(16, nc)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    return TinyCNN(num_classes)


class TestEvaluate:
    def test_evaluate_returns_expected_keys(self, device: torch.device):
        model = _create_simple_model(10)
        data_loader = _create_dummy_data_loader(
            num_batches=1, batch_size=4, num_classes=10
        )
        metrics = evaluate(model, device, data_loader, dtype="float32")
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics
        assert "brier_score" in metrics

    def test_evaluate_accuracy_range(self, device: torch.device):
        model = _create_simple_model(10)
        data_loader = _create_dummy_data_loader(
            num_batches=2, batch_size=4, num_classes=10
        )
        metrics = evaluate(model, device, data_loader, dtype="float32")
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_f1_score_range(self, device: torch.device):
        model = _create_simple_model(10)
        data_loader = _create_dummy_data_loader(
            num_batches=2, batch_size=4, num_classes=10
        )
        metrics = evaluate(model, device, data_loader, dtype="float32")
        assert 0.0 <= metrics["f1_score"] <= 1.0

    def test_evaluate_brier_score_range(self, device: torch.device):
        model = _create_simple_model(10)
        data_loader = _create_dummy_data_loader(
            num_batches=2, batch_size=4, num_classes=10
        )
        metrics = evaluate(model, device, data_loader, dtype="float32")
        assert 0.0 <= metrics["brier_score"] <= 2.0

    def test_evaluate_confusion_matrix_shape(self, device: torch.device):
        model = _create_simple_model(5)
        data_loader = _create_dummy_data_loader(
            num_batches=2, batch_size=4, num_classes=5
        )
        metrics = evaluate(model, device, data_loader, dtype="float32")
        cm = metrics["confusion_matrix"]
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (5, 5)

    def test_evaluate_classification_report_is_string(self, device: torch.device):
        model = _create_simple_model(10)
        data_loader = _create_dummy_data_loader(
            num_batches=1, batch_size=4, num_classes=10
        )
        metrics = evaluate(model, device, data_loader, dtype="float32")
        assert isinstance(metrics["classification_report"], str)
        assert len(metrics["classification_report"]) > 0

    def test_evaluate_none_model_raises(self, device: torch.device):
        data_loader = _create_dummy_data_loader(
            num_batches=1, batch_size=4, num_classes=10
        )
        with pytest.raises(ValueError, match="Model cannot be None"):
            evaluate(None, device, data_loader, dtype="float32")  # type: ignore

    def test_evaluate_none_data_loader_raises(self, device: torch.device):
        model = _create_simple_model(10)
        with pytest.raises(ValueError, match="Data loader cannot be None"):
            evaluate(model, device, None, dtype="float32")  # type: ignore

    def test_evaluate_with_dtype_string(self, device: torch.device):
        model = _create_simple_model(10)
        data_loader = _create_dummy_data_loader(
            num_batches=1, batch_size=4, num_classes=10
        )
        metrics = evaluate(model, device, data_loader, dtype="float32")
        assert isinstance(metrics["accuracy"], float)

    def test_evaluate_with_torch_dtype(self, device: torch.device):
        model = _create_simple_model(10)
        data_loader = _create_dummy_data_loader(
            num_batches=1, batch_size=4, num_classes=10
        )
        metrics = evaluate(model, device, data_loader, dtype=torch.float32)
        assert isinstance(metrics["accuracy"], float)

    def test_evaluate_single_class(self, device: torch.device):
        """Evaluate with all same class labels (tests edge case for F1)."""

        class SingleClassDataset(torch.utils.data.Dataset[Any]):
            def __len__(self) -> int:
                return 8

            def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
                return torch.randn(3, 32, 32), 0

        data_loader = torch.utils.data.DataLoader(SingleClassDataset(), batch_size=4)
        model = _create_simple_model(1)
        metrics = evaluate(model, device, data_loader, dtype="float32", target_class=0)
        assert metrics["accuracy"] >= 0.0

    def test_evaluate_large_batch(self, device: torch.device):
        model = _create_simple_model(10)
        data_loader = _create_dummy_data_loader(
            num_batches=10, batch_size=32, num_classes=10
        )
        metrics = evaluate(model, device, data_loader, dtype="float32")
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["f1_score"], float)

    def test_evaluate_resnet18(self, small_model: nn.Module, device: torch.device):
        data_loader = _create_dummy_data_loader(
            num_batches=1, batch_size=4, num_classes=10
        )
        metrics = evaluate(small_model, device, data_loader, dtype="float32")
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)
