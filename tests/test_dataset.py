"""Tests for src/data/dataset.py — DatasetWrapper, DataLoader class."""

import pytest
import torch
import torchvision

from src.data.dataset import DataLoader, DatasetWrapper

# ─── DatasetWrapper ────────────────────────────────────────────────────


class TestDatasetWrapper:
    def test_getitem_returns_tensor_and_int(self):
        """getitem should return (tensor, int label) tuple when ToTensor is used."""
        from torchvision.transforms import ToTensor

        full_dataset = torchvision.datasets.CIFAR10(
            root="/tmp", train=False, download=True
        )
        wrapper = DatasetWrapper(full_dataset, transform=ToTensor())
        x, y = wrapper[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, int)

    def test_getitem_transform_applied(self):
        """Transform should be applied to the image tensor."""
        from torchvision.transforms import ToTensor

        full_dataset = torchvision.datasets.CIFAR10(
            root="/tmp", train=False, download=True
        )
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x * 2.0),
            ]
        )
        wrapper = DatasetWrapper(full_dataset, transform=transform)
        x_wrap, _ = wrapper[0]
        assert isinstance(x_wrap, torch.Tensor)
        assert x_wrap.shape == (3, 32, 32)

    def test_getitem_no_transform(self):
        """Without ToTensor transform, output is PIL Image."""
        from PIL import Image

        full_dataset = torchvision.datasets.CIFAR10(
            root="/tmp", train=False, download=True
        )
        wrapper = DatasetWrapper(full_dataset, transform=None)
        x, y = wrapper[0]
        assert isinstance(x, Image.Image)
        assert isinstance(y, int)

    def test_len_matches_subset(self):
        full_dataset = torchvision.datasets.CIFAR10(
            root="/tmp", train=False, download=True
        )
        wrapper = DatasetWrapper(full_dataset)
        assert len(wrapper) == len(full_dataset)

    def test_index_out_of_range_raises(self):
        full_dataset = torchvision.datasets.CIFAR10(
            root="/tmp", train=False, download=True
        )
        wrapper = DatasetWrapper(full_dataset)
        with pytest.raises(IndexError, match="Index out of range"):
            wrapper[len(full_dataset)]

    def test_negative_index_raises(self):
        full_dataset = torchvision.datasets.CIFAR10(
            root="/tmp", train=False, download=True
        )
        wrapper = DatasetWrapper(full_dataset)
        with pytest.raises(IndexError, match="Index out of range"):
            wrapper[-1]

    def test_item_shape_with_totensor(self):
        from torchvision.transforms import ToTensor

        full_dataset = torchvision.datasets.CIFAR10(
            root="/tmp", train=False, download=True
        )
        wrapper = DatasetWrapper(full_dataset, transform=ToTensor())
        x, _ = wrapper[0]
        assert x.shape == (3, 32, 32)


# ─── DataLoader ────────────────────────────────────────────────────────


class TestDataLoader:
    def test_init_default_splits(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        assert dl.train_dataset is not None
        assert dl.val_dataset is not None
        assert dl.test_dataset is not None

    def test_init_custom_splits(self):
        dl = DataLoader(
            batch_size=16, train_split=0.8, validation_split=0.2, test_split=0.0
        )
        assert len(dl.train_dataset) == int(50000 * 0.8)
        assert len(dl.val_dataset) == int(50000 * 0.2)

    def test_init_invalid_split_sum_exceeds_one(self):
        with pytest.raises(ValueError, match="Invalid split ratios"):
            DataLoader(
                batch_size=32, train_split=0.7, validation_split=0.5, test_split=0.0
            )

    def test_init_zero_batch_size_raises(self):
        with pytest.raises(ValueError, match="Batch size must be positive"):
            DataLoader(batch_size=0)

    def test_init_negative_batch_size_raises(self):
        with pytest.raises(ValueError, match="Batch size must be positive"):
            DataLoader(batch_size=-1)

    def test_get_train_loader_returns_dataloader(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        loader = dl.get_train_loader()
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_get_val_loader_returns_dataloader(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        loader = dl.get_val_loader()
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_get_test_loader_returns_dataloader(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        loader = dl.get_test_loader()
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_train_loader_shuffle(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        loader = dl.get_train_loader()
        # Verify it's a PyTorch DataLoader with shuffle=True
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_val_loader_no_shuffle(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        loader = dl.get_val_loader()
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_train_loader_batch_output(self):
        dl = DataLoader(
            batch_size=8, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        loader = dl.get_train_loader()
        images, labels = next(iter(loader))
        assert images.shape[0] == 8
        assert images.shape == (8, 3, 32, 32)
        assert len(labels) == 8

    def test_train_dataset_size(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        expected_train = int(50000 * 0.9)
        assert len(dl.train_dataset) == expected_train

    def test_val_dataset_size(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        expected_val = int(50000 * 0.1)
        assert len(dl.val_dataset) == expected_val

    def test_test_dataset_size(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        # test dataset is full CIFAR-10 test set (10000 images)
        assert len(dl.test_dataset) == 10000

    def test_split_zero_validation(self):
        dl = DataLoader(
            batch_size=32, train_split=1.0, validation_split=0.0, test_split=0.0
        )
        assert len(dl.train_dataset) == 50000
        assert len(dl.val_dataset) == 0

    def test_iterate_train_loader(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        loader = dl.get_train_loader()
        images, labels = next(iter(loader))
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_iterate_val_loader(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        loader = dl.get_val_loader()
        images, labels = next(iter(loader))
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_iterate_test_loader(self):
        dl = DataLoader(
            batch_size=32, train_split=0.9, validation_split=0.1, test_split=0.0
        )
        loader = dl.get_test_loader()
        images, labels = next(iter(loader))
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_num_workers_default(self):
        dl = DataLoader(batch_size=32)
        assert dl.num_workers == 4

    def test_num_workers_custom(self):
        dl = DataLoader(batch_size=32, num_workers=2)
        assert dl.num_workers == 2

    def test_data_dir_default(self):
        dl = DataLoader(batch_size=32)
        assert dl.data_dir == "./data"

    def test_data_dir_custom(self):
        dl = DataLoader(batch_size=32, data_dir="/tmp/cifar10")
        assert dl.data_dir == "/tmp/cifar10"
