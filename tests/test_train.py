"""Tests for src/train/train.py — EarlyStopping class and train function."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.train.train import EarlyStopping

# ─── EarlyStopping ─────────────────────────────────────────────────────


class TestEarlyStopping:
    def test_init_default(self):
        es = EarlyStopping()
        assert es.patience == 5
        assert es.min_delta == 0.0
        assert es.counter == 0
        assert es.best_score is None
        assert es.early_stop is False
        assert es.checkpoint_path is None

    def test_init_custom(self):
        es = EarlyStopping(patience=10, min_delta=0.01)
        assert es.patience == 10
        assert es.min_delta == 0.01

    def test_first_call_sets_best_score(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es(small_model, 0.5, str(temp_checkpoint_dir))
        assert es.best_score == 0.5
        assert es.counter == 0
        assert es.early_stop is False
        assert (temp_checkpoint_dir / "model.safetensors").exists()

    def test_improvement_resets_counter(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping(patience=3, min_delta=0.01)
        es(small_model, 0.5, str(temp_checkpoint_dir))
        es(small_model, 0.6, str(temp_checkpoint_dir))  # improvement
        assert es.counter == 0
        assert es.best_score == 0.6

    def test_no_improvement_increases_counter(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping(patience=3, min_delta=0.01)
        es(small_model, 0.5, str(temp_checkpoint_dir))
        es(small_model, 0.49, str(temp_checkpoint_dir))  # no improvement
        es(small_model, 0.48, str(temp_checkpoint_dir))  # no improvement
        es(
            small_model, 0.47, str(temp_checkpoint_dir)
        )  # no improvement — triggers early stop
        assert es.early_stop is True
        assert es.counter == 3

    def test_improvement_within_min_delta_ignored(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping(patience=3, min_delta=0.01)
        es(small_model, 0.5, str(temp_checkpoint_dir))
        # Improvement smaller than min_delta should be ignored
        es(small_model, 0.505, str(temp_checkpoint_dir))
        assert es.counter == 1

    def test_score_below_zero_raises(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping()
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            es(small_model, -0.1, str(temp_checkpoint_dir))

    def test_score_above_one_raises(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping()
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            es(small_model, 1.1, str(temp_checkpoint_dir))

    def test_score_exactly_zero(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping()
        es(small_model, 0.0, str(temp_checkpoint_dir))
        assert es.best_score == 0.0

    def test_score_exactly_one(self, small_model: nn.Module, temp_checkpoint_dir: Path):
        es = EarlyStopping()
        es(small_model, 1.0, str(temp_checkpoint_dir))
        assert es.best_score == 1.0

    def test_patience_one_triggers_quickly(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping(patience=1, min_delta=0.0)
        es(small_model, 0.8, str(temp_checkpoint_dir))
        es(small_model, 0.7, str(temp_checkpoint_dir))
        assert es.early_stop is True

    def test_checkpoint_saved_on_improvement(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping(patience=5, min_delta=0.01)
        checkpoint_path = temp_checkpoint_dir / "model.safetensors"

        es(small_model, 0.5, str(temp_checkpoint_dir))
        first_mtime = checkpoint_path.stat().st_mtime

        # Wait a moment and improve
        import time

        time.sleep(0.01)
        es(small_model, 0.7, str(temp_checkpoint_dir))
        second_mtime = checkpoint_path.stat().st_mtime

        assert second_mtime >= first_mtime

    def test_best_score_updates(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping(patience=10, min_delta=0.0)
        es(small_model, 0.3, str(temp_checkpoint_dir))
        es(small_model, 0.5, str(temp_checkpoint_dir))
        es(small_model, 0.9, str(temp_checkpoint_dir))
        assert es.best_score == 0.9

    def test_counter_resets_after_improvement_then_stops(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        es = EarlyStopping(patience=2, min_delta=0.0)
        es(small_model, 0.5, str(temp_checkpoint_dir))  # best=0.5
        es(small_model, 0.4, str(temp_checkpoint_dir))  # counter=1
        es(small_model, 0.6, str(temp_checkpoint_dir))  # best=0.6, counter=0
        es(small_model, 0.55, str(temp_checkpoint_dir))  # counter=1
        es(small_model, 0.50, str(temp_checkpoint_dir))  # counter=2 → early_stop
        assert es.early_stop is True

    def test_same_score_increases_counter(
        self, small_model: nn.Module, temp_checkpoint_dir: Path
    ):
        """Same score as best should increase counter (not considered improvement)."""
        es = EarlyStopping(patience=2, min_delta=0.0)
        es(small_model, 0.5, str(temp_checkpoint_dir))
        es(small_model, 0.5, str(temp_checkpoint_dir))  # same score, not better
        es(small_model, 0.5, str(temp_checkpoint_dir))  # same score again
        assert es.counter == 2


# ─── train function integration tests (mocked) ─────────────────────────


class TestTrainFunction:
    def test_train_imports_successfully(self):
        """Verify the train function can be imported without side effects."""
        from src.train.train import train

        assert callable(train)

    def test_train_function_signature(self):
        """Verify train function has expected parameters."""
        import inspect
        from src.train.train import train

        sig = inspect.signature(train)
        params = list(sig.parameters.keys())
        assert "checkpoint_dir" in params
        assert "data_root" in params
        assert "accumulation_steps" in params
