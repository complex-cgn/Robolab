"""Model evaluation module with comprehensive metric computation.

This module provides functions to evaluate trained PyTorch models
on CIFAR-10 datasets, computing accuracy, F1 score, confusion
matrix, and classification reports using scikit-learn.
"""

from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.amp.autocast_mode import autocast


def evaluate(
    model: nn.Module | None,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader[Any] | None,
    dtype: str | torch.dtype,
    target_class: int = 1,
) -> dict[str, Any]:
    """Evaluate the trained model and return detailed metrics.

    Runs the model in inference mode over the provided data loader,
    collects predictions, and computes evaluation metrics.

    Args:
        model: The trained PyTorch model to evaluate.
        data_loader: The data loader for the test or validation set.
        dtype: Data type string for tensor operations
            (e.g., ``"float32"``, ``"float16"``).
        target_class: The class for which to compute the Brier score.

    Returns:
        Dictionary containing the following keys:

        - ``"accuracy"``: Overall classification accuracy.
        - ``"f1_score"``: Macro-averaged F1 score.
        - ``"confusion_matrix"``: Confusion matrix as a NumPy array.
        - ``"classification_report"``: Per-class precision/recall/F1 report.
    """

    # Set dtype
    resolved_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    # Validate inputs
    if model is None:
        raise ValueError("Model cannot be None")
    if data_loader is None:
        raise ValueError("Data loader cannot be None")

    # Disable gradient computation and set model to inference mode
    model.eval()
    model = model.to(device, dtype=resolved_dtype)

    # Accumulators for batched predictions and ground-truth labels
    all_preds_list: list[np.ndarray] = []
    all_labels_list: list[np.ndarray] = []
    all_probs_list: list[np.ndarray] = []

    # Determine if autocast should be enabled (only for CUDA devices)
    use_autocast = device.type == "cuda"

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if use_autocast:
                with autocast(device_type=device.type, dtype=resolved_dtype):
                    outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs.data, dim=1)

            all_preds_list.append(predicted.cpu().numpy())
            all_labels_list.append(labels.cpu().numpy())
            all_probs_list.append(probs.cpu().numpy())

    # Concatenate batch-wise predictions and labels into single arrays
    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    all_probs = np.concatenate(all_probs_list)

    if len(all_labels) != len(all_preds):
        raise ValueError("Labels and predictions length mismatch")

    # Compute evaluation metrics using scikit-learn
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    cr = cast(str, classification_report(all_labels, all_preds, zero_division=0))

    # Compute Brier score for the specified target class
    true_binary = (all_labels == target_class).astype(int)
    pred_confidence = all_probs[:, target_class]
    brier = brier_score_loss(true_binary, pred_confidence)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": cr,
        "brier_score": brier,
    }
