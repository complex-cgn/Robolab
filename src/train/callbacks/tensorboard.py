from __future__ import annotations

from typing import TYPE_CHECKING

from src.train.callback import Callback

if TYPE_CHECKING:
    from src.train.train import Trainer


class Tensorboard(Callback):
    def on_epoch_end(self, trainer: "Trainer" | None = None):
        pass
