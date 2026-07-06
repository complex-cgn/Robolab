from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.train.train import Trainer


@dataclass
class Callback:
    _trainer: Trainer | None = None

    @property
    def trainer(self):
        if self._trainer is None:
            raise RuntimeError(
                "Callback is not attached to a trainer. Call `attach` method first."
            )
        return self._trainer

    def attach(self, trainer: Trainer):
        self._trainer = trainer

    def on_epoch_start(self, trainer: Trainer):
        pass

    def on_epoch_end(self, trainer: Trainer):
        pass

    def on_train_start(self, trainer: Trainer):
        pass

    def on_train_end(self, trainer: Trainer):
        pass

    def on_batch_start(self, trainer: Trainer):
        pass

    def on_batch_end(self, trainer: Trainer):
        pass
