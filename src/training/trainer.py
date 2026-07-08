from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.datasets import DatasetWrapper
from training.callbacks.base import Callback


@dataclass
class Trainer:
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    train_loader: torch.utils.data.DataLoader[DatasetWrapper]
    val_loader: torch.utils.data.DataLoader[DatasetWrapper]
    num_epochs: int
    callbacks: Sequence[Callback]

    model: nn.Module = field(init=False)
    device: torch.device = field(init=False)
    epoch: int = field(init=False)
    total_steps: int = field(init=False)

    def __post_init__(self):
        for cb in self.callbacks:
            cb.attach(self)

        self.total_steps = len(self.train_loader)

    def fit(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

        for callback in self.callbacks:
            callback.on_train_start(self)

        for self.epoch in range(self.num_epochs):
            model.train()
            self._train_epoch()

        for callback in self.callbacks:
            callback.on_train_end(self)

    def validate(self, model: nn.Module):
        pass

    def test(self, model: nn.Module):
        pass

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
