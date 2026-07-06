"""Training script for ConvNet on CIFAR-10.

This module prov loaded from robolab/configs
ct (cfg.trainparams, cfg.hypides a complete training pipeline for training a ConvNet
classification model on the CIFAR-10 dataset. It includes:
    - Early stopping mechanism to prevent overfitting
    - Mixed precision (AMP) training for faster GPU training
    - Cosine annealing with warm restarts learning rate scheduling
    - Gradient clipping for training stability
    - Rich console logging with p loaded from robolab/configs
ct (cfg.trainparams, cfg.hyprogress bars and epoch tables
    - Checkpoint saving and best model tracking

Usage:
    python -m robolab.train          # Run training with default config
    python -c "from robolab.train import train; train()"

Configuration:
    Training hyperparameters are loaded from robolab/configs/config.yaml
    via the centralized cfg object (cfg.trainparams, cfg.hyperparams).
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from src.config import cfg
from src.data import DatasetWrapper, train_loader, val_loader
from src.eval import evaluate
from src.models import model_factory
from src.utils import get_device, logger, save_checkpoint, total_params
from src.utils.logger import RichLogger


class Callback:
    trainer: Trainer | None = None

    def _attach(self, trainer):
        self.trainer = trainer

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_batch_start(self):
        pass

    def on_batch_end(self):
        pass


class Checkpoint(Callback):
    def on_epoch_end(self):
        pass


class EarlyStopping(Callback):
    def on_epoch_end(self):
        pass


class Logger(Callback):
    def on_epoch_end(self):
        pass


class OneCycle(Callback):
    def on_epoch_end(self):
        pass


@dataclass
class Terminal(Callback):
    progress: Progress = field(init=False)
    task_id: TaskID = field(init=False)

    def on_train_start(self):
        self.progress = Progress(
            SpinnerColumn(spinner="dots12"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_color="green", finished_color="red"),
            TimeRemainingColumn(),
        )
        total_steps = self.total_steps()

    def on_train_end(self):
        self.progress.stop()

    def on_batch_end(self):
        self.progress.update(self.task_id, advance=1)


class Tensorboard(Callback):
    def on_epoch_end(self):
        pass


@dataclass
class Trainer:
    model: nn.Module
    device: torch.device
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    train_loader: torch.utils.data.DataLoader[DatasetWrapper]
    val_loader: torch.utils.data.DataLoader[DatasetWrapper]
    num_epochs: int
    callbacks: list[Callback] = field(default_factory=list)

    total_steps: int = field(init=False)

    def fit(self):
        for callback in self.callbacks:
            callback.on_train_start(self)

        for _ in range(self.num_epochs):
            self.model.train()
            self._train_epoch()

        for callback in self.callbacks:
            callback.on_train_end(self)

    def _train_epoch(self):
        for callback in self.callbacks:
            callback.on_epoch_start(self)

        for images, labels in self.train_loader:
            self._train_batch(images, labels)

        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def _train_batch(self, images: torch.Tensor, labels: torch.Tensor):
        for callback in self.callbacks:
            callback.on_batch_start(self)

        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for callback in self.callbacks:
            callback.on_batch_end(self)


def _create_epoch_progress(
    console: Console, epoch: int, total_epochs: int, total_batches: int
) -> Progress:
    """Create a Rich Progress instance for training and validation tasks.

    Args:
        console: Rich Console instance for output.
        epoch: Current epoch number (1-based).
        total_epochs: Total number of epochs.
        total_batches: Number of training batches per epoch.

    Returns:
        Configured Rich Progress instance with training and validation tasks.
    """
    # Build the text column with epoch info — the dynamic task name
    # will be set by the active task description at runtime.
    epoch_text = f"  [bold cyan]Epoch {epoch}/{total_epochs}[/]"
    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn(epoch_text, justify="right"),
        BarColumn(bar_width=40, style="blue", complete_style="bold blue"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    return progress


def train(
    checkpoint_dir: str = "checkpoints",
    data_root: str = "./data",
    accumulation_steps: int = 4,
) -> None:

    rich_logger = RichLogger()
    total_epochs = cfg.trainparams.num_epochs
    early_stopping = EarlyStopping(patience=cfg.hyperparams.early_stopping_patience)

    device = get_device()

    # reproducibility
    if cfg.hyperparams.random_seed is not None:
        torch.manual_seed(cfg.hyperparams.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.hyperparams.random_seed)

    torch.set_float32_matmul_precision("high")

    # 🧠 MODEL (HER ZAMAN FP32)
    model = model_factory(num_classes=cfg.hyperparams.num_classes)
    model = model.to(device).float()

    rich_logger.console.print(
        f"[bold green]✓ Model initialized[/bold green] "
        f"[dim]Params:[/dim] [cyan]{total_params(model):,}[/cyan]"
    )

    # loss + optimizer (FP32 şart)
    criterion = getattr(nn, cfg.trainparams.criterion)()

    optimizer = getattr(torch.optim, cfg.trainparams.optimizer)(
        model.parameters(),
        lr=cfg.trainparams.learning_rate,
        weight_decay=cfg.trainparams.weight_decay,
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.trainparams.learning_rate,
        total_steps=cfg.trainparams.num_epochs * len(train_loader),
    )

    # 🧊 AMP SETUP (BF16 / FP16 SAFE MODE)
    use_cuda = device.type == "cuda"

    if use_cuda:
        bf16_supported = torch.cuda.is_bf16_supported()

        amp_dtype = torch.bfloat16 if bf16_supported else torch.float16

        rich_logger.console.print(
            f"[bold green]✓ Mixed precision enabled[/bold green] "
            f"[dim]{'BF16' if bf16_supported else 'FP16'}[/dim]"
        )

    else:
        amp_dtype = None
        rich_logger.console.print(f"[bold yellow]⚠ CPU mode (no AMP)[/bold yellow]")

    scaler = (
        torch.cuda.amp.GradScaler()
        if (use_cuda and amp_dtype == torch.float16)
        else None
    )

    writer = SummaryWriter(log_dir="runs/convnet_cifar10")

    total_step = len(train_loader)

    for epoch in range(total_epochs):
        epoch_progress = _create_epoch_progress(
            rich_logger.console, epoch + 1, total_epochs, total_step
        )

        model.train()
        optimizer.zero_grad()

        avg_train_loss = 0.0

        with epoch_progress as progress:
            task = progress.add_task("Training", total=total_step)

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # 🧊 ONLY COMPUTE MIXED PRECISION
                if use_cuda:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss = loss / accumulation_steps
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / accumulation_steps

                # 🧠 backward
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 🔥 step
                if (i + 1) % accumulation_steps == 0 or (i + 1) == total_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()

                avg_train_loss += loss.item() * accumulation_steps

                if (i + 1) % 100 == 0:
                    writer.add_scalar(
                        "Training Loss",
                        avg_train_loss / (i + 1),
                        epoch * total_step + i,
                    )

                progress.update(task, advance=1)

        avg_train_loss /= total_step

        # 🧪 validation
        model.eval()
        metrics = evaluate(model, device, val_loader, torch.bfloat16)

        val_acc = metrics["accuracy"] * 100
        val_f1 = metrics["f1_score"]

        writer.add_scalar("Validation Accuracy", val_acc / 100, epoch)
        writer.add_scalar("Validation F1", val_f1, epoch)

        rich_logger.console.print("✓ Validation complete")

        current_lr = optimizer.param_groups[0]["lr"]

        rich_logger.log_epoch_end(
            epoch_data=None,
            train_loss=avg_train_loss,
            val_accuracy=val_acc,
            val_f1=val_f1,
            val_brier=metrics.get("brier_score"),
            learning_rate=current_lr,
        )

        early_stopping(model, val_acc / 100, checkpoint_dir)

        if early_stopping.early_stop:
            break

    writer.close()
    rich_logger.log_training_summary()


if __name__ == "__main__":
    model = model_factory(cfg.hyperparams.num_classes).to(get_device())
    trainer = Trainer(
        model=model,
        device=get_device(),
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=cfg.trainparams.learning_rate,
        ),
        criterion=getattr(nn, cfg.trainparams.criterion)(),
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        callbacks=[Terminal()],
    )
    trainer.fit()
