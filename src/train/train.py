"""Training script for ConvNet on CIFAR-10.

This module provides a complete training pipeline for training a ConvNet
classification model on the CIFAR-10 dataset. It includes:
    - Early stopping mechanism to prevent overfitting
    - Mixed precision (AMP) training for faster GPU training
    - Cosine annealing with warm restarts learning rate scheduling
    - Gradient clipping for training stability
    - TensorBoard logging for training metrics
    - Checkpoint saving and best model tracking

Usage:
    python -m robolab.train          # Run training with default config
    python -c "from robolab.train import train; train()"

Configuration:
    Training hyperparameters are loaded from robolab/configs/config.yaml
    via the centralized cfg object (cfg.trainparams, cfg.hyperparams).
"""

from typing import cast

import torch
import torch.amp as amp
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from src.config import cfg
from src.data import DatasetWrapper, train_loader, val_loader
from src.eval import evaluate
from src.models import model_factory
from src.utils import get_device, logger, save_checkpoint, total_params


class EarlyStopping:
    """Early stopping utility to prevent overfitting during training.

    Monitors validation accuracy and saves a checkpoint whenever the model
    achieves a new best score. Training is halted if no improvement is
    observed for a configurable number of epochs (patience).

    Attributes:
        patience: Number of epochs with no improvement before triggering stop.
        min_delta: Minimum absolute change in validation accuracy to qualify
            as a meaningful improvement (prevents stopping on negligible gains).
        counter: Internal counter tracking consecutive non-improving epochs.
        best_score: The highest validation accuracy seen so far.
        early_stop: Flag indicating whether early stopping has been triggered.
        checkpoint_path: Path to the last saved best-model checkpoint.

    Example:
        >>> early_stop = EarlyStopping(patience=10, min_delta=0.001)
        >>> for epoch in range(num_epochs):
        ...     val_acc = evaluate(model, val_loader)
        ...     early_stop(model, val_acc, "checkpoints")
        ...     if early_stop.early_stop:
        ...         break
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """Initialize the EarlyStopping utility.

        Args:
            patience: Number of epochs to wait for validation accuracy
                improvement before triggering early stopping. Default is 5.
            min_delta: Minimum change in validation accuracy to qualify as
                an improvement. Metrics changes smaller than this value are
                ignored. Default is 0.0 (any improvement counts).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = None

    def __call__(self, model: nn.Module, score: float, checkpoint_dir: str) -> None:
        """Evaluate the current validation score and manage checkpoint saving.

        Compares the provided validation score against the best seen so far.
        If the score represents a meaningful improvement (exceeding best_score
        by at least min_delta), the model checkpoint is saved and the
        non-improvement counter is reset. Otherwise, the counter is incremented.

        Args:
            model: The PyTorch model instance to save. A checkpoint will be
                saved only if the current score represents an improvement.
            score: The current epoch's validation accuracy (float between 0.0
                and 1.0). This is the metric used to determine improvement.
            checkpoint_dir: Directory path where model checkpoint files (.ckpt)
                will be saved. Created if it does not exist.

        Side Effects:
            - Saves model checkpoint to disk when improvement is detected.
            - Logs progress messages via the logger.
            - Updates internal state (best_score, counter, early_stop flags).
            - Sets self.early_stop = True when patience is exhausted.
        """

        if not (0.0 <= score <= 1.0):
            raise ValueError("Score must be between 0 and 1")

        if self.best_score is None:
            self.best_score = score
            self.checkpoint_path = save_checkpoint(
                model=model, checkpoint_dir=checkpoint_dir
            )
            logger.info(f"New best validation accuracy: {score * 100:.2f}%")

        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.checkpoint_path = save_checkpoint(
                model=model, checkpoint_dir=checkpoint_dir
            )
            self.counter = 0
            logger.info(f"New best validation accuracy: {score * 100:.2f}%")

        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered.")


def train(
    checkpoint_dir: str = "checkpoints",
    data_root: str = "./data",
    accumulation_steps: int = 4,
) -> None:
    """Train ConvNet on CIFAR-10 with the configured hyperparameters.

    This function orchestrates the full training lifecycle including:
        1. Setting random seeds for reproducibility
        2. Initializing the ConvNet model and computing parameter count
        3. Configuring AdamW optimizer with weight decay
        4. Setting up CosineAnnealingWarmRestarts learning rate scheduler
        5. Training with mixed precision (AMP) on CUDA, standard precision on CPU
        6. Performing periodic validation and logging to TensorBoard
        7. Saving checkpoints and triggering early stopping if validation
           accuracy plateaus

    Args:
        checkpoint_dir: Directory path where model checkpoints (.ckpt files)
            will be saved. Best checkpoints are overwritten as validation
            accuracy improves. Default is "checkpoints".
        data_root: Root directory path where the CIFAR-10 dataset will be
            downloaded to or loaded from. Default is "./data".

    Note:
        - Mixed precision (float16) is automatically enabled when a CUDA GPU
          is available, providing faster training with minimal accuracy impact.
        - The learning rate follows a cosine annealing schedule with warm
          restarts (T_0=10, T_mult=2), restarting every 10, 20, 40... epochs.
        - Gradient clipping is applied at max_grad_norm=1.0 for stability.
        - TensorBoard logs are written to "runs/convnet_cifar10/".
        - Training stops early if validation accuracy does not improve for
          cfg.hyperparams.early_stopping_patience consecutive epochs.

    Example:
        # Run training with default settings
        >>> train()

        # Custom checkpoint directory and data path
        >>> train(checkpoint_dir="/tmp/my_checkpoints", data_root="/tmp/cifar10_data")
    """

    # Create fresh early stopping instance for each training run
    early_stopping = EarlyStopping(patience=cfg.hyperparams.early_stopping_patience)
    device = get_device()

    # Set random seed for reproducibility
    if cfg.hyperparams.random_seed is not None:
        torch.manual_seed(cfg.hyperparams.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.hyperparams.random_seed)

    # Model
    torch.set_float32_matmul_precision("high")
    model = model_factory(num_classes=cfg.hyperparams.num_classes).to(device)
    logger.info(f"total_params: {total_params(model):,}")

    # Loss and optimizer
    if not hasattr(nn, cfg.trainparams.criterion):
        raise ValueError(f"Criterion not found: {cfg.trainparams.criterion}")
    if not hasattr(torch.optim, cfg.trainparams.optimizer):
        raise ValueError(f"Optimizer not found: {cfg.trainparams.optimizer}")

    criterion = getattr(nn, cfg.trainparams.criterion)()
    optimizer = getattr(torch.optim, cfg.trainparams.optimizer)(
        model.parameters(),
        lr=cfg.trainparams.learning_rate,
        weight_decay=cfg.trainparams.weight_decay,
    )

    # OneCycle scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.trainparams.learning_rate,
        total_steps=cfg.trainparams.num_epochs * len(train_loader),
    )

    # Mixed precision training setup
    if device.type == "cuda":
        scaler = amp.GradScaler("cuda")
    else:
        scaler = None

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/convnet_cifar10")

    total_step = len(train_loader)
    for epoch in range(cfg.trainparams.num_epochs):
        # Training loop
        model.train()

        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            with amp.autocast(device.type, dtype=getattr(torch, cfg.trainparams.dtype)):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient clipping — unscale gradients before clipping
            if (i + 1) % accumulation_steps == 0 or (i + 1) == total_step:
                max_grad_norm = 1.0
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                scheduler.step()

            if (i + 1) % 100 == 0:
                writer.add_scalar("Training Loss", loss.item(), epoch * total_step + i)
                logger.info(
                    f"Epoch [{epoch + 1}/{cfg.trainparams.num_epochs}], "
                    f"Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}"
                )

        # Validation
        model.eval()
        metrics = evaluate(model, device, val_loader, dtype=cfg.trainparams.dtype)

        val_accuracy = metrics["accuracy"]
        writer.add_scalar("Validation Accuracy", val_accuracy, epoch)
        # val_loader.dataset is typed as Dataset[DatasetWrapper] which lacks __len__
        # in Pylance's type stubs, but the actual DatasetWrapper subclass has it.
        val_total = len(cast(DatasetWrapper, val_loader.dataset))
        logger.info(
            f"Validation Accuracy of the models on the "
            f"{val_total} validation images: {val_accuracy:.2f} %"
        )

        # Check for early stopping
        early_stopping(model, val_accuracy, checkpoint_dir)
        if early_stopping.early_stop:
            break


if __name__ == "__main__":
    train()
