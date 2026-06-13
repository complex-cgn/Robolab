"""Test module for evaluating ConvNet on CIFAR-10 test set.

This module loads a trained model checkpoint and evaluates it on the
full CIFAR-10 test set, logging overall and per-class metrics.
"""

from src.config import cfg
from src.data import test_loader
from src.eval import evaluate
from src.models import model_factory
from src.utils import (
    get_device,
    load_checkpoint,
    logger,
)

from typing import Any

def test(
    checkpoint_path: str = "checkpoints/model.ckpt",
    dtype: str = "float32",
) -> dict[str, Any]:
    """Evaluate the trained model on the full CIFAR-10 test set.

    Loads a model checkpoint from disk, runs inference over the test
    loader, and returns a dictionary of aggregated metrics.

    Args:
        checkpoint_path: File path to the saved model checkpoint.
        dtype: Data type string for tensor operations
            (e.g., ``"float32"``, ``"float16"``).

    Returns:
        dict: Evaluation metrics including overall accuracy, F1 score,
            confusion matrix, and classification report.
    """
    logger.info("Starting evaluation on the test set...")

    # Determine compute device (CUDA if available, otherwise CPU)
    device = get_device()

    # Instantiate model architecture and restore weights from checkpoint
    model = model_factory(num_classes=cfg.hyperparams.num_classes).to(device)
    load_checkpoint(
        model=model,
        checkpoint_address=checkpoint_path,
    )

    # Run comprehensive evaluation via the eval module
    metrics = evaluate(model, device, test_loader, dtype=dtype)

    # Aggregate evaluation results into a single metrics dictionary
    metrics: dict[str, Any] = {
        "overall_accuracy": metrics["accuracy"],
        "total_samples": len(test_loader.dataset),  # type: ignore[union-attr]
        "correct_predictions": int(metrics["accuracy"] * len(test_loader.dataset)),  # type: ignore[union-attr]
        "f1_score": metrics["f1_score"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "brier_score": metrics["brier_score"],
    }

    # Log key evaluation results to console and file
    logger.info(f"Total Test Samples: {metrics['total_samples']}")
    logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
    logger.info(f"Overall Test Accuracy: {metrics['overall_accuracy'] * 100:.2f} %")
    logger.info(f"F1 Score: {metrics['f1_score'] * 100:.4f} %")
    logger.info(f"Brier Score: {metrics['brier_score']:.4f}")

    return metrics


if __name__ == "__main__":
    test()
