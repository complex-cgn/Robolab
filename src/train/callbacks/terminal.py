from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

from src.train.callback import Callback

if TYPE_CHECKING:
    from src.train.train import Trainer


@dataclass
class Terminal(Callback):
    progress: Progress | None = field(default=None, init=False)
    task_id: TaskID | None = field(default=None, init=False)

    def on_epoch_start(self, trainer: Trainer | None = None):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
        )
        total_steps = self.trainer.total_steps
        epoch = self.trainer.epoch
        self.task_id = self.progress.add_task(
            f"[cyan]Epoch {epoch} / {self.trainer.num_epochs}", total=total_steps
        )
        self.progress.start()

    def on_epoch_end(self, trainer: Trainer | None = None):
        if self.progress is not None:
            self.progress.stop()

    def on_batch_end(self, trainer: Trainer | None = None):
        if self.progress is not None and self.task_id is not None:
            self.progress.update(self.task_id, advance=1)
