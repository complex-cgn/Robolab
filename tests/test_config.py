"""Tests for src/config.py — configuration loading, validation, and Pydantic models."""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from src.config import (
    Config,
    Hyperparameters,
    TrainingParams,
    TestingParams,
    DatasetConfig,
    LossCriterion,
)


# ─── LossCriterion Enum ────────────────────────────────────────────────


class TestLossCriterion:
    def test_enum_values(self):
        assert LossCriterion.CROSS_ENTROPY.value == "CrossEntropyLoss"
        assert LossCriterion.FOCAL.value == "FocalLoss"
        assert LossCriterion.LABEL_SMOOTHING.value == "LabelSmoothing"
        assert LossCriterion.MSE.value == "MSE"

    def test_enum_member_check(self):
        assert "CrossEntropyLoss" in [e.value for e in LossCriterion]
        assert len(list(LossCriterion)) == 4


# ─── DatasetConfig ─────────────────────────────────────────────────────


class TestDatasetConfig:
    def test_default_factory_raises(self):
        """DatasetConfig requires all fields, so default factory raises."""
        with pytest.raises(ValidationError):
            DatasetConfig()

    def test_valid_dataset(self):
        cfg = DatasetConfig(
            name="CIFAR10", train_split=0.9, validation_split=0.1, test_split=0.0
        )
        assert cfg.name == "CIFAR10"
        assert cfg.train_split == 0.9

    def test_invalid_train_split_below_zero(self):
        with pytest.raises(ValidationError):
            DatasetConfig(
                name="CIFAR10", train_split=-0.1, validation_split=0.1, test_split=0.0
            )

    def test_invalid_train_split_above_one(self):
        with pytest.raises(ValidationError):
            DatasetConfig(
                name="CIFAR10", train_split=1.1, validation_split=0.1, test_split=0.0
            )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            DatasetConfig(
                name="CIFAR10",
                train_split=0.9,
                validation_split=0.1,
                test_split=0.0,
                extra_field="bad",
            )


# ─── Hyperparameters ───────────────────────────────────────────────────


class TestHyperparameters:
    def test_default_factory_raises(self):
        """Hyperparameters requires all fields, so default factory raises."""
        with pytest.raises(ValidationError):
            Hyperparameters()

    def test_valid_hyperparams(self):
        hp = Hyperparameters(
            logging_level="INFO",
            checkpoint_dir="checkpoints",
            random_seed=42,
            num_classes=10,
            early_stopping_patience=10,
        )
        assert hp.logging_level == "INFO"
        assert hp.num_classes == 10
        assert hp.random_seed == 42

    def test_num_classes_below_one(self):
        with pytest.raises(ValidationError):
            Hyperparameters(
                logging_level="INFO",
                checkpoint_dir="checkpoints",
                random_seed=42,
                num_classes=0,
                early_stopping_patience=10,
            )

    def test_early_stopping_patience_below_one(self):
        with pytest.raises(ValidationError):
            Hyperparameters(
                logging_level="INFO",
                checkpoint_dir="checkpoints",
                random_seed=42,
                num_classes=10,
                early_stopping_patience=0,
            )


# ─── TrainingParams ────────────────────────────────────────────────────


class TestTrainingParams:
    def test_default_factory_raises(self):
        """TrainingParams requires all fields, so default factory raises."""
        with pytest.raises(ValidationError):
            TrainingParams()

    def test_valid_training_params(self):
        tp = TrainingParams(
            batch_size=32,
            learning_rate=1e-3,
            num_epochs=50,
            weight_decay=5e-4,
            criterion="CrossEntropyLoss",
            optimizer="AdamW",
            dtype="float32",
            warmup_epochs=5,
            max_grad_norm=1.0,
            accumulation_steps=1,
        )
        assert tp.batch_size == 32
        assert tp.learning_rate == 1e-3
        assert tp.criterion == "CrossEntropyLoss"
        assert tp.optimizer == "AdamW"
        assert tp.dtype == "float32"

    def test_invalid_criterion(self):
        with pytest.raises(ValidationError) as exc_info:
            TrainingParams(
                batch_size=32,
                learning_rate=1e-3,
                num_epochs=50,
                weight_decay=5e-4,
                criterion="NonExistentLoss",
                optimizer="AdamW",
                dtype="float32",
                warmup_epochs=5,
                max_grad_norm=1.0,
                accumulation_steps=1,
            )
        assert any("Criterion" in str(e) for e in exc_info.value.errors())

    def test_invalid_optimizer(self):
        with pytest.raises(ValidationError) as exc_info:
            TrainingParams(
                batch_size=32,
                learning_rate=1e-3,
                num_epochs=50,
                weight_decay=5e-4,
                criterion="CrossEntropyLoss",
                optimizer="FakeOptimizer",
                dtype="float32",
                warmup_epochs=5,
                max_grad_norm=1.0,
                accumulation_steps=1,
            )
        assert any("Optimizer" in str(e) for e in exc_info.value.errors())

    def test_invalid_dtype(self):
        with pytest.raises(ValidationError) as exc_info:
            TrainingParams(
                batch_size=32,
                learning_rate=1e-3,
                num_epochs=50,
                weight_decay=5e-4,
                criterion="CrossEntropyLoss",
                optimizer="AdamW",
                dtype="float999",
                warmup_epochs=5,
                max_grad_norm=1.0,
                accumulation_steps=1,
            )
        assert any("dtype" in str(e.get("loc", [])) for e in exc_info.value.errors())

    def test_weight_decay_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrainingParams(
                batch_size=32,
                learning_rate=1e-3,
                num_epochs=50,
                weight_decay=0.0,
                criterion="CrossEntropyLoss",
                optimizer="AdamW",
                dtype="float32",
                warmup_epochs=5,
                max_grad_norm=1.0,
                accumulation_steps=1,
            )

    def test_num_epochs_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            TrainingParams(
                batch_size=32,
                learning_rate=1e-3,
                num_epochs=0,
                weight_decay=5e-4,
                criterion="CrossEntropyLoss",
                optimizer="AdamW",
                dtype="float32",
                warmup_epochs=5,
                max_grad_norm=1.0,
                accumulation_steps=1,
            )

    def test_accumulation_steps_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            TrainingParams(
                batch_size=32,
                learning_rate=1e-3,
                num_epochs=50,
                weight_decay=5e-4,
                criterion="CrossEntropyLoss",
                optimizer="AdamW",
                dtype="float32",
                warmup_epochs=5,
                max_grad_norm=1.0,
                accumulation_steps=0,
            )

    def test_warmup_epochs_can_be_zero(self):
        tp = TrainingParams(
            batch_size=32,
            learning_rate=1e-3,
            num_epochs=50,
            weight_decay=5e-4,
            criterion="CrossEntropyLoss",
            optimizer="AdamW",
            dtype="float32",
            warmup_epochs=0,
            max_grad_norm=1.0,
            accumulation_steps=1,
        )
        assert tp.warmup_epochs == 0


# ─── TestingParams ─────────────────────────────────────────────────────


class TestTestingParams:
    def test_default_factory_raises(self):
        """TestingParams requires dtype field."""
        with pytest.raises(ValidationError):
            TestingParams()

    def test_valid_dtype(self):
        ttp = TestingParams(dtype="float32")
        assert ttp.dtype == "float32"

    def test_invalid_dtype(self):
        with pytest.raises(ValidationError):
            TestingParams(dtype="float999")


# ─── Config ────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_factory(self):
        """Config uses default_factory for all sub-models, but they may have missing required fields."""
        # The Config uses default_factory for sub-models, but Hyperparameters
        # requires all fields — so default factory will fail at instantiation time
        # if the YAML config doesn't provide valid values.
        # This test verifies that Config can be constructed with valid sub-models.
        hp = Hyperparameters(
            logging_level="INFO",
            checkpoint_dir="checkpoints",
            random_seed=42,
            num_classes=10,
            early_stopping_patience=10,
        )
        tp = TrainingParams(
            batch_size=32,
            learning_rate=1e-3,
            num_epochs=50,
            weight_decay=5e-4,
            criterion="CrossEntropyLoss",
            optimizer="AdamW",
            dtype="float32",
            warmup_epochs=5,
            max_grad_norm=1.0,
            accumulation_steps=1,
        )
        ttp = TestingParams(dtype="float32")
        ds = DatasetConfig(
            name="CIFAR10", train_split=0.9, validation_split=0.1, test_split=0.0
        )
        cfg = Config(hyperparams=hp, trainparams=tp, testparams=ttp, dataset=ds)
        assert isinstance(cfg.hyperparams, Hyperparameters)
        assert isinstance(cfg.trainparams, TrainingParams)
        assert isinstance(cfg.testparams, TestingParams)
        assert isinstance(cfg.dataset, DatasetConfig)

    def test_valid_config_construction(self):
        hp = Hyperparameters(
            logging_level="INFO",
            checkpoint_dir="checkpoints",
            random_seed=42,
            num_classes=10,
            early_stopping_patience=10,
        )
        tp = TrainingParams(
            batch_size=32,
            learning_rate=1e-3,
            num_epochs=50,
            weight_decay=5e-4,
            criterion="CrossEntropyLoss",
            optimizer="AdamW",
            dtype="float32",
            warmup_epochs=5,
            max_grad_norm=1.0,
            accumulation_steps=1,
        )
        ttp = TestingParams(dtype="float32")
        ds = DatasetConfig(
            name="CIFAR10", train_split=0.9, validation_split=0.1, test_split=0.0
        )
        cfg = Config(hyperparams=hp, trainparams=tp, testparams=ttp, dataset=ds)
        assert cfg.hyperparams.num_classes == 10
        assert cfg.dataset.name == "CIFAR10"

    def test_load_from_yaml(self, temp_config_path: Path):
        cfg = Config.load_from_yaml(str(temp_config_path))
        assert cfg.hyperparams.num_classes == 10
        assert cfg.hyperparams.logging_level == "INFO"
        assert cfg.trainparams.batch_size == 32
        assert cfg.trainparams.learning_rate == pytest.approx(1e-3)
        assert cfg.dataset.name == "CIFAR10"
        assert cfg.dataset.train_split == 0.9

    def test_load_from_yaml_invalid_config(self, temp_checkpoint_dir: Path):
        bad_config = temp_checkpoint_dir / "bad_config.yaml"
        with open(bad_config, "w") as f:
            yaml.dump(
                {
                    "hyperparams": {
                        "logging_level": "INFO",
                        "checkpoint_dir": str(temp_checkpoint_dir),
                        "random_seed": 42,
                        "num_classes": -5,  # invalid: ge=1
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
                    "testparams": {"dtype": "float32"},
                    "dataset": {
                        "name": "CIFAR10",
                        "train_split": 0.9,
                        "validation_split": 0.1,
                        "test_split": 0.0,
                    },
                },
                f,
            )
        with pytest.raises(ValueError):
            Config.load_from_yaml(str(bad_config))

    def test_load_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            Config.load_from_yaml("/nonexistent/path/config.yaml")

    def test_load_from_yaml_extra_field(self, temp_checkpoint_dir: Path):
        bad_config = temp_checkpoint_dir / "extra_field_config.yaml"
        with open(bad_config, "w") as f:
            yaml.dump(
                {
                    "hyperparams": {
                        "logging_level": "INFO",
                        "checkpoint_dir": str(temp_checkpoint_dir),
                        "random_seed": 42,
                        "num_classes": 10,
                        "early_stopping_patience": 10,
                        "extra_field": "should_fail",
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
                    "testparams": {"dtype": "float32"},
                    "dataset": {
                        "name": "CIFAR10",
                        "train_split": 0.9,
                        "validation_split": 0.1,
                        "test_split": 0.0,
                    },
                },
                f,
            )
        with pytest.raises(ValueError):
            Config.load_from_yaml(str(bad_config))

    def test_load_from_yaml_missing_required_field(self, temp_checkpoint_dir: Path):
        bad_config = temp_checkpoint_dir / "missing_field_config.yaml"
        with open(bad_config, "w") as f:
            yaml.dump(
                {
                    "hyperparams": {
                        "logging_level": "INFO",
                        "checkpoint_dir": str(temp_checkpoint_dir),
                        "random_seed": 42,
                        "num_classes": 10,
                        "early_stopping_patience": 10,
                    },
                    # trainparams is missing entirely
                    "testparams": {"dtype": "float32"},
                    "dataset": {
                        "name": "CIFAR10",
                        "train_split": 0.9,
                        "validation_split": 0.1,
                        "test_split": 0.0,
                    },
                },
                f,
            )
        with pytest.raises(ValueError):
            Config.load_from_yaml(str(bad_config))

    def test_config_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            Config(
                hyperparams=Hyperparameters(
                    logging_level="INFO",
                    checkpoint_dir="cp",
                    random_seed=42,
                    num_classes=10,
                    early_stopping_patience=10,
                ),
                trainparams=TrainingParams(
                    batch_size=32,
                    learning_rate=1e-3,
                    num_epochs=50,
                    weight_decay=5e-4,
                    criterion="CrossEntropyLoss",
                    optimizer="AdamW",
                    dtype="float32",
                    warmup_epochs=5,
                    max_grad_norm=1.0,
                    accumulation_steps=1,
                ),
                testparams=TestingParams(dtype="float32"),
                dataset=DatasetConfig(
                    name="CIFAR10",
                    train_split=0.9,
                    validation_split=0.1,
                    test_split=0.0,
                ),
                extra_field="bad",
            )
