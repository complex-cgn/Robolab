from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.config import cfg
from src.data import DatasetWrapper, train_loader, val_loader
from src.models import model_factory
from src.train.callback import Callback
from src.train.callbacks import Terminal
from src.utils import get_device


@dataclass
class Trainer:
    model: nn.Module
    device: torch.device
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    train_loader: torch.utils.data.DataLoader[DatasetWrapper]
    val_loader: torch.utils.data.DataLoader[DatasetWrapper]
    num_epochs: int
    callbacks: Sequence[Callback]

    epoch: int = field(init=False, default=0)
    total_steps: int = field(init=False)

    def __post_init__(self):
        for cb in self.callbacks:
            cb.attach(self)

        self.total_steps = len(self.train_loader)

    def fit(self):
        for callback in self.callbacks:
            callback.on_train_start(self)

        for self.epoch in range(self.num_epochs):
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
