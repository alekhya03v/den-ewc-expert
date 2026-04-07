# datasets.py

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def build_task_dataset(name, root="./data", train=True, download=True):
    transform = get_transform()

    name = name.lower()
    if name == "mnist":
        ds = datasets.MNIST(root=root, train=train, download=download, transform=transform)
    elif name == "fashionmnist":
        ds = datasets.FashionMNIST(root=root, train=train, download=download, transform=transform)
    elif name == "kmnist":
        ds = datasets.KMNIST(root=root, train=train, download=download, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return ds


def split_train_val(dataset, val_ratio=0.1, seed=42):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    return train_ds, val_ds


def get_task_loaders(task_names, batch_size=128, root="./data", num_workers=2):
    """
    Returns:
        tasks = [
            {
                "task_name": ...,
                "train_loader": ...,
                "val_loader": ...,
                "test_loader": ...
            },
            ...
        ]
    """
    tasks = []

    for task_name in task_names:
        train_full = build_task_dataset(task_name, root=root, train=True, download=True)
        test_ds = build_task_dataset(task_name, root=root, train=False, download=True)
        train_ds, val_ds = split_train_val(train_full, val_ratio=0.1)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        tasks.append({
            "task_name": task_name,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader
        })

    return tasks
