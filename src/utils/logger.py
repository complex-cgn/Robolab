"""Rich logging utilities for the Robolab package.

Provides enhanced logging with Rich library support including:
- Progress bars for training and validation
- Formatted tables for epoch summaries
- Color-coded log messages
- Training duration tracking
"""

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.traceback import install

install(show_locals=True, width=120, extra_lines=2)


class EpochData:
    """Stores data for a single training epoch.

    Attributes:
        epoch: Epoch number (1-based).
        train_loss: Training loss for this epoch.
        train_loss_avg: Average training loss across batches.
        val_accuracy: Validation accuracy.
        val_f1: Validation F1 score.
        val_brier: Validation Brier score.
        learning_rate: Learning rate at this epoch.
        duration: Epoch duration in seconds.
        start_time: Epoch start timestamp.
    """

    def __init__(self, epoch: int):
        self.epoch = epoch
        self.train_loss: float = 0.0
        self.train_loss_avg: float = 0.0
        self.val_accuracy: float = 0.0
        self.val_f1: float = 0.0
        self.val_brier: Optional[float] = None
        self.learning_rate: float = 0.0
        self.duration: float = 0.0
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Mark the start of this epoch."""
        self.start_time = time.time()

    def end(self) -> None:
        """Mark the end of this epoch and calculate duration."""
        if self.start_time is not None:
            self.duration = time.time() - self.start_time


class EpochProgressTracker:
    """Manages the Rich progress bar for a single epoch.

    Attributes:
        console: Rich Console instance.
        progress: Rich Progress instance.
        task_names: List of task names to display.
        tasks: Mapping of task names to Rich task IDs.
    """

    def __init__(self, console: Console, task_names: list[str]):
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self.tasks = {}
        for name in task_names:
            self.tasks[name] = self.progress.add_task(name)

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any):
        self.progress.__exit__(*args, **kwargs)

    def update(self, task_name: str, advance: float = 1, **kwargs: Any) -> None:
        """Update a specific task."""
        self.progress.update(self.tasks[task_name], advance=advance, **kwargs)

    def update_progress(self, task_name: str, total: int, current: int) -> None:
        """Set absolute progress for a task."""
        self.progress.update(self.tasks[task_name], completed=current, total=total)

    def stop(self) -> None:
        """Stop the progress bar."""
        self.progress.stop()


class RichLogger:
    """Enhanced logger with Rich library support for training visualization.

    Provides:
    - Rich console output with color and formatting
    - Epoch-by-epoch summary tables
    - Training progress bars
    - Duration tracking
    - Final training summary table

    Attributes:
        console: Rich Console instance for stdout output.
        logger: Standard Python logger for file logging.
        epoch_summaries: List of all epoch data for final summary.
    """

    def __init__(self, name: str = "robolab"):
        self.console = Console()
        self.epoch_summaries: list[EpochData] = []
        self._logger: Optional[logging.Logger] = None
        self.name = name

    def _get_logger(self) -> logging.Logger:
        """Get or create the underlying logger."""
        if self._logger is None:
            current_dir = Path(__file__).resolve().parent
            log_dir = current_dir.parent.parent / "logs"
            log_file = log_dir / "robolab.log"
            log_dir.mkdir(parents=True, exist_ok=True)

            self._logger = logging.getLogger(self.name)
            self._logger.setLevel(logging.INFO)
            self._logger.handlers.clear()

            # Rich console handler
            rich_handler = RichHandler(
                rich_tracebacks=True,
                console=self.console,
                show_time=True,
                show_path=False,
            )
            rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
            self._logger.addHandler(rich_handler)

            # Rotating file handler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self._logger.addHandler(file_handler)

        return self._logger

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message with Rich formatting."""
        self._get_logger().info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self._get_logger().warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self._get_logger().error(message, **kwargs)

    def log_epoch_start(self, epoch: int, total_epochs: int) -> EpochData:
        """Log the start of an epoch with a progress bar.

        Args:
            epoch: Current epoch number (1-based).
            total_epochs: Total number of epochs.

        Returns:
            EpochData instance for tracking this epoch's metrics.
        """
        self.console.print()
        self.console.print(
            f"[bold cyan]{'=' * 60}",
        )
        self.console.print(
            f"[bold cyan]  Epoch {epoch}/{total_epochs}[/bold cyan] [dim]{self._format_duration(time.time())}[/dim]",
        )
        self.console.print(
            f"[bold cyan]{'=' * 60}",
        )

        epoch_data = EpochData(epoch)
        epoch_data.start()
        return epoch_data

    def log_epoch_end(
        self,
        epoch_data: EpochData,
        train_loss: float,
        val_accuracy: float,
        val_f1: float = 0.0,
        val_brier: Optional[float] = None,
        learning_rate: float = 0.0,
    ) -> None:
        """Log the end of an epoch with formatted metrics.

        Args:
            epoch_data: The EpochData instance for this epoch.
            train_loss: Final training loss value.
            val_accuracy: Validation accuracy (0-100).
            val_f1: Validation F1 score (0-1).
            val_brier: Validation Brier score (optional).
            learning_rate: Current learning rate.
        """
        epoch_data.train_loss = train_loss
        epoch_data.val_accuracy = val_accuracy / 100.0  # Store as 0-1
        epoch_data.val_f1 = val_f1
        epoch_data.val_brier = val_brier
        epoch_data.learning_rate = learning_rate
        epoch_data.end()

        self.epoch_summaries.append(epoch_data)

        # Create formatted row
        acc_color = (
            "green" if val_accuracy >= 80 else "yellow" if val_accuracy >= 60 else "red"
        )
        f1_color = "green" if val_f1 >= 0.8 else "yellow" if val_f1 >= 0.6 else "red"

        self.console.print(
            f"  [dim]Loss:[/] {train_loss:.4f}  "
            f"[{acc_color}]Acc: {val_accuracy:.2f}%[/]"
            f"  [{f1_color}]F1: {val_f1:.4f}[/]"
            f"  [dim]LR: {learning_rate:.6f}[/]"
            f"  [dim]{epoch_data.duration:.1f}s[/]"
        )
        

        



    def log_training_summary(self) -> None:
        """Log a comprehensive training summary table."""
        if not self.epoch_summaries:
            return

        self.console.print()
        self.console.print(
            "[bold magenta]╔══════════════════════════════════════════════════════════╗[/bold magenta]"
        )
        self.console.print(
            "[bold magenta]║           TRAINING SUMMARY                               ║[/bold magenta]"
        )
        self.console.print(
            "[bold magenta]╚══════════════════════════════════════════════════════════╝[/bold magenta]"
        )
        self.console.print()

        # Create the summary table
        table = Table(
            box=None,
            padding=(0, 1),
            expand=True,
            style="bold white on black",
        )

        # Column headers with icons
        table.add_column(
            "Epoch",
            justify="center",
            style="bold cyan",
            width=7,
        )
        table.add_column(
            "Loss",
            justify="center",
            style="bold yellow",
            width=10,
        )
        table.add_column(
            "Val Acc",
            justify="center",
            style="bold green",
            width=10,
        )
        table.add_column(
            "Val F1",
            justify="center",
            style="bold blue",
            width=10,
        )
        table.add_column(
            "LR",
            justify="center",
            style="bold magenta",
            width=12,
        )
        table.add_column(
            "Duration",
            justify="center",
            style="bold white",
            width=10,
        )
        table.add_column(
            "Cumulative",
            justify="center",
            style="bold dim",
            width=10,
        )

        # Find best accuracy
        best_epoch = max(self.epoch_summaries, key=lambda x: x.val_accuracy)

        for es in self.epoch_summaries:
            cum_time = sum(
                e.duration
                for e in self.epoch_summaries[: self.epoch_summaries.index(es) + 1]
            )

            # Highlight best epoch
            if es.epoch == best_epoch.epoch:
                acc_style = "bold green on green"
                f1_style = "bold blue on blue"
                epoch_style = "bold cyan on cyan"
            else:
                acc_style = "green"
                f1_style = "blue"
                epoch_style = "cyan"

            table.add_row(
                f"[{epoch_style}]#{es.epoch:03d}[/]",
                f"{es.train_loss:.4f}",
                f"[{acc_style}]{es.val_accuracy * 100:.2f}%[/]",
                f"[{f1_style}]{es.val_f1:.4f}[/]",
                f"{es.learning_rate:.6f}",
                f"{es.duration:.1f}s",
                f"{cum_time:.0f}s",
            )

        self.console.print(table)
        self.console.print()

        # Print best model info
        self.console.print(
            f"[bold green]✓ Best Model:[/bold green] Epoch #{best_epoch.epoch:03d} "
            f"with Validation Accuracy: [bold green]{best_epoch.val_accuracy * 100:.2f}%[/bold green]"
        )

        total_time = sum(e.duration for e in self.epoch_summaries)
        self.console.print(
            f"[bold cyan]✓ Total Training Time:[/bold cyan] [bold]{self._format_duration(total_time)}[/bold]"
        )
        self.console.print()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into a human-readable duration string.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted string like "00:05:23" for hours, minutes, seconds.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# Legacy compatibility — keep the old functions working
def setup_logger() -> logging.Logger:
    """Configure and return a logger for the Robolab package.

    Deprecated: Use RichLogger instead for enhanced logging.
    """
    return RichLogger()._get_logger()


logger = RichLogger()
