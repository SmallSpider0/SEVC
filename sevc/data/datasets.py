"""Canonical dataset factory migrated from the historical experiment trees."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetBundle:
    train: Any
    test: Any
    num_classes: int
    input_shape: tuple[int, ...]
    dataset_key: str


def load_dataset_bundle(
    dataset_key: str,
    data_dir: Path,
    *,
    download: bool = False,
    imagenet_classes: tuple[str, ...] | None = None,
    cifar100_image_size: int = 224,
) -> DatasetBundle:
    """Load one supported dataset without hidden absolute paths or global caches."""

    from torchvision import datasets, transforms

    key = dataset_key.lower()
    data_dir.mkdir(parents=True, exist_ok=True)
    if key == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return DatasetBundle(
            train=datasets.MNIST(data_dir, train=True, download=download, transform=transform),
            test=datasets.MNIST(data_dir, train=False, download=download, transform=transform),
            num_classes=10,
            input_shape=(1, 28, 28),
            dataset_key=key,
        )
    if key == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        return DatasetBundle(
            train=datasets.CIFAR10(
                data_dir, train=True, download=download, transform=train_transform
            ),
            test=datasets.CIFAR10(
                data_dir, train=False, download=download, transform=test_transform
            ),
            num_classes=10,
            input_shape=(3, 32, 32),
            dataset_key=key,
        )
    if key == "cifar100":
        if cifar100_image_size not in {32, 224}:
            raise ValueError("cifar100_image_size must be 32 or 224")
        resize = (
            []
            if cifar100_image_size == 32
            else [transforms.Resize(cifar100_image_size)]
        )
        train_transform = transforms.Compose(
            [*resize,
                transforms.RandomCrop(cifar100_image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [*resize,
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        return DatasetBundle(
            train=datasets.CIFAR100(
                data_dir, train=True, download=download, transform=train_transform
            ),
            test=datasets.CIFAR100(
                data_dir, train=False, download=download, transform=test_transform
            ),
            num_classes=100,
            input_shape=(3, cifar100_image_size, cifar100_image_size),
            dataset_key=key,
        )
    if key == "imagenet":
        if imagenet_classes is not None:
            raise NotImplementedError(
                "deterministic ImageNet class filtering requires a reviewed experiment adapter"
            )
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return DatasetBundle(
            train=datasets.ImageFolder(data_dir / "train", transform=train_transform),
            test=datasets.ImageFolder(data_dir / "val", transform=test_transform),
            num_classes=len(datasets.ImageFolder(data_dir / "train").classes),
            input_shape=(3, 224, 224),
            dataset_key=key,
        )
    raise ValueError(f"unsupported dataset key: {dataset_key}")
