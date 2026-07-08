from __future__ import annotations

from typing import TYPE_CHECKING

from training.callbacks.base import Callback

if TYPE_CHECKING:
    from training.trainer import Trainer


class OneCycleLR(Callback):
    def on_epoch_end(self, trainer: "Trainer" | None = None):
        pass
