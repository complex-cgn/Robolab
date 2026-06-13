"""Hyperparameters configuration for model training."""

import difflib
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from pydantic_core import ErrorDetails

from src.utils import logger


class LossCriterion(Enum):
    CROSS_ENTROPY = "CrossEntropyLoss"
    FOCAL = "FocalLoss"
    LABEL_SMOOTHING = "LabelSmoothing"
    MSE = "MSE"


class DatasetConfig(BaseModel):
    """Dataset configuration.

    Attributes:
        name: Name of the dataset (e.g., "CIFAR10").
        train_split: Proportion of the dataset to use for training (0.0 to 1.0).
        validation_split: Proportion of the dataset to use for validation (0.0 to 1.0).
        test_split: Proportion of the dataset to use for testing (0.0 to 1.0).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    train_split: float = Field(ge=0.0, le=1.0)
    validation_split: float = Field(ge=0.0, le=1.0)
    test_split: float = Field(ge=0.0, le=1.0)


class Hyperparameters(BaseModel):
    """Hyperparameters configuration.

    Attributes:
        logging_level: Logging level (e.g., "INFO", "DEBUG").
        checkpoint_dir: Directory path for saving model checkpoints.
        random_seed: Random seed for reproducibility, or None to disable.
        num_classes: Number of output classification classes.
        early_stopping_patience: Epochs to wait for validation improvement.
        model_type: Type of model to use (resnet18 or mythos).
    """

    model_config = ConfigDict(extra="forbid")

    logging_level: str
    checkpoint_dir: str
    random_seed: int | None
    num_classes: int = Field(ge=1)
    early_stopping_patience: int = Field(ge=1)


class TrainingParams(BaseModel):
    """Training hyperparameters configuration.

    Attributes:
        batch_size: Number of samples per gradient update.
        learning_rate: Step size for optimizer updates.
        num_epochs: Total number of training passes through the dataset.
        dtype: Data type for the model tensors.
        weight_decay: L2 regularization factor.
        warmup_epochs: Number of warmup epochs for learning rate scheduler.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: int
    weight_decay: float = Field(gt=0.0)
    warmup_epochs: int = Field(ge=0)
    max_grad_norm: float = Field(gt=0.0)
    learning_rate: float = Field(gt=0.0)
    num_epochs: int = Field(ge=1)
    criterion: str
    optimizer: str
    dtype: str
    accumulation_steps: int = Field(ge=1)

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, v: str) -> str:
        if not hasattr(torch.nn, v):
            raise ValueError(f"Criterion '{v}' is not a valid torch.nn module. ")
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        if not hasattr(torch.optim, v):
            raise ValueError(f"Optimizer '{v}' is not a valid torch.optim module. ")
        return v

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        if not hasattr(torch, v):
            raise ValueError(f"Invalid dtype '{v}' provided. ")
        return v


class TestingParams(BaseModel):
    """Testing/evaluation hyperparameters configuration.

    Attributes:
        dtype: Data type for the model tensors during evaluation.
    """

    model_config = ConfigDict(extra="forbid")

    dtype: str

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        if not hasattr(torch, v):
            raise ValueError(f"Invalid dtype '{v}' provided. ")
        return v


class Config(BaseModel):
    """Application configuration loaded from YAML.

    Attributes:
        hyperparams: Common hyperparameters shared across modules.
        trainparams: Training-specific hyperparameters.
        testparams: Testing/evaluation-specific hyperparameters.
        dataset: Dataset configuration (e.g., name, splits).

    Example:
        Load from YAML (global singleton):

        >>> cfg = Config.load_from_yaml()

        Load from dict (useful for tests):

        >>> cfg = Config(
        ...     hyperparams=Hyperparameters(num_classes=10),
        ...     trainparams=TrainingParams(),
        ...     testparams=TestingParams(),
        ...     dataset=DatasetConfig()
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    hyperparams: Hyperparameters = Field(default_factory=lambda: Hyperparameters())
    trainparams: TrainingParams = Field(default_factory=lambda: TrainingParams())
    testparams: TestingParams = Field(default_factory=lambda: TestingParams())
    dataset: DatasetConfig = Field(default_factory=lambda: DatasetConfig())

    @classmethod
    def load_from_yaml(cls, path: str | None = None) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file. Defaults to configs/config.yaml.

        Returns:
            Config instance loaded from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file cannot be parsed.
            ValidationError: If the configuration fails Pydantic validation.
        """
        default_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        resolved_path = path or default_path

        with open(resolved_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        try:
            return cls(**config_data)
        except ValidationError as e:
            for error in e.errors():
                cls._log_validation_error(error)
            raise ValueError(f"Configuration validation error: {e}") from e

    @classmethod
    def _log_validation_error(cls, error: ErrorDetails) -> None:
        loc_list = error.get("loc", [])
        location = " -> ".join(str(loc) for loc in loc_list)
        wrong_field = loc_list[-1] if loc_list else "unknown"
        error_type = error.get("type", "unknown")

        logger.error(f"Validation error at '{location}': {error['msg']}")

        handlers = {
            "extra_forbidden": lambda: cls._handle_extra_field(wrong_field, loc_list),
            "value_error": lambda: logger.error(
                f"Invalid value for '{wrong_field}'. Check the constraint."
            ),
            "type_error": lambda: logger.error(
                f"Type mismatch for '{wrong_field}'. Check the expected type."
            ),
            "missing": lambda: logger.error(
                f"Required field '{wrong_field}' is missing."
            ),
        }

        handler = handlers.get(error_type)
        if handler:
            handler()

    @classmethod
    def _handle_extra_field(cls, wrong_field: str, loc_list: list[Any]) -> None:
        logger.error(f"Unexpected field '{wrong_field}' found. Please remove it.")

        parent_model = cls._resolve_model(loc_list[:-1])
        if parent_model is None or not hasattr(parent_model, "model_fields"):
            return

        valid_fields = list(parent_model.model_fields.keys())
        suggestions = difflib.get_close_matches(
            wrong_field, valid_fields, n=3, cutoff=0.3
        )

        if suggestions:
            logger.error(f"Did you mean: {', '.join(repr(s) for s in suggestions)}?")
        else:
            logger.error(
                f"No close matches for '{wrong_field}'. "
                f"Valid fields in '{parent_model.__name__}': {valid_fields}"
            )

    @classmethod
    def _resolve_model(cls, path: list[Any]) -> type | None:
        """Traverse nested Pydantic models by following field annotations."""

        current = cls
        for loc in path:
            fields = getattr(current, "model_fields", {})
            if loc not in fields:
                return None

            field_type = fields[loc].annotation

            if hasattr(field_type, "__annotations__"):
                current = field_type
                continue

            args = getattr(field_type, "__args__", [])
            nested = next((a for a in args if hasattr(a, "__annotations__")), None)
            if nested:
                current = nested
            else:
                return None

        return current


cfg: Config = Config.load_from_yaml()
