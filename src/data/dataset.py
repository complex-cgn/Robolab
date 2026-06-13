from collections.abc import Callable
from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from src.config import cfg

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class DatasetWrapper(torch.utils.data.Dataset[Any]):
    """Wrapper to apply transforms to subsets of the dataset."""

    def __init__(
        self,
        subset: Any,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not (0 <= index < len(self.subset)):
            raise IndexError("Index out of range")
        x, y = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)


class DataLoader:
    def __init__(
        self,
        batch_size: int = 128,
        data_dir: str = "./data",
        num_workers: int = 4,
        train_split: float = 0.9,
        validation_split: float = 0.1,
        test_split: float = 0.0,
    ) -> None:
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split

        if not (0 < train_split + validation_split <= 1.0):
            raise ValueError(
                f"Invalid split ratios: {train_split} + "
                f"{validation_split} must be between 0 and 1.0"
            )
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        self._prepare_transforms()
        self._prepare_datasets()

    def _prepare_transforms(self) -> None:
        """Prepare data augmentation and normalization transforms."""

        # Data augmentation and normalization for training
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            ]
        )

        # Normalization for validation and testing (no augmentation)
        self.eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            ]
        )

    def _prepare_datasets(self) -> None:
        """Load CIFAR-10 dataset and create train/validation/test splits."""

        full_train_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=None
        )

        val_size = int(self.validation_split * len(full_train_set))
        train_size = len(full_train_set) - val_size
        if train_size + val_size != len(full_train_set):
            raise ValueError("Split size mismatch")

        g = torch.Generator().manual_seed(42)

        train_subset, val_subset = random_split(
            full_train_set, [train_size, val_size], generator=g
        )

        self.train_dataset = DatasetWrapper(
            train_subset, transform=self.train_transforms
        )
        self.val_dataset = DatasetWrapper(val_subset, transform=self.eval_transforms)

        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=None,
        )
        self.test_dataset = DatasetWrapper(test_dataset, transform=self.eval_transforms)

    def get_train_loader(self) -> torch.utils.data.DataLoader[DatasetWrapper]:
        """Get the training data loader."""
        return torch.utils.data.DataLoader[DatasetWrapper](
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_val_loader(self) -> torch.utils.data.DataLoader[DatasetWrapper]:
        """Get the validation data loader."""
        return torch.utils.data.DataLoader[DatasetWrapper](
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self) -> torch.utils.data.DataLoader[DatasetWrapper]:
        """Get the test data loader."""
        return torch.utils.data.DataLoader[DatasetWrapper](
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


data_loader: DataLoader = DataLoader(
    batch_size=cfg.trainparams.batch_size,
    train_split=cfg.dataset.train_split,
    validation_split=cfg.dataset.validation_split,
    test_split=cfg.dataset.test_split,
)

train_loader: torch.utils.data.DataLoader[DatasetWrapper] = (
    data_loader.get_train_loader()
)
val_loader: torch.utils.data.DataLoader[DatasetWrapper] = data_loader.get_val_loader()
test_loader: torch.utils.data.DataLoader[DatasetWrapper] = data_loader.get_test_loader()
