from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .data_loader import InMemoryDataLoader


KNOWN_DATASETS = ["CIFAR10", "FashionMNIST", "MNIST"]
ORIGINAL_SIZE = {
    "CIFAR10": (3, 32, 32),
    "FashionMNIST": (1, 28, 28),
    "MNIST": (1, 28, 28),
}
MEAN_STD = {
    "CIFAR10": {(3, 32, 32): (0.4736, 0.2516)},
    "FashionMNIST": {
        (1, 28, 28): (0.2859, 0.353),
        (1, 32, 32): (0.2189, 0.3318),
        (3, 32, 32): (0.2189, 0.3318),
    },
    "MNIST": {
        (1, 28, 28): (0.1305, 0.3081),
        (1, 32, 32): (0.1003, 0.2752),
        (3, 32, 32): (0.1003, 0.2752),
    },
}


def padding(in_sz: List[int], out_sz: List[int]) -> Tuple[int, int, int, int]:
    d_h, d_w = out_sz[-2] - in_sz[-2], out_sz[-1] - in_sz[-1]
    p_h1, p_w1 = d_h // 2, d_w // 2
    p_h2, p_w2 = d_h - p_h1, d_w - p_w1
    return p_h1, p_h2, p_w1, p_w2


def get_torch_loader(  # pylint: disable=C0330
    dataset_name: str,
    train: bool = True,
    in_size: List[int] = None,
    batch_size: int = 1,
    shuffle: bool = True,
    normalize: bool = True,
) -> DataLoader:

    transfs = []
    if in_size is not None:
        if in_size[-2:] != ORIGINAL_SIZE[dataset_name][-2:]:
            _padding = padding(ORIGINAL_SIZE[dataset_name], in_size)
            transfs.append(transforms.Pad(_padding))
        transfs.append(transforms.ToTensor())
        if in_size[0] != ORIGINAL_SIZE[dataset_name][0]:
            transfs.append(transforms.Lambda(lambda t: t.expand(in_size)))
    else:
        in_size = ORIGINAL_SIZE[dataset_name]
        transfs.append(transforms.ToTensor())

    if normalize:
        mean, std = MEAN_STD[dataset_name][in_size]
        transfs.append(transforms.Normalize((mean,), (std,)))

    dataset = getattr(datasets, dataset_name)(
        f"./.data/{dataset_name:s}",
        train=train,
        download=True,
        transform=transforms.Compose(transfs),
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def to_memory(  # pylint: disable=C0330
    data_loader: DataLoader,
    batch_size: int = 1,
    shuffle: bool = True,
    order_by_class: bool = False,
    shuffle_classes: bool = False,
    allow_mixed_batches: bool = True,
    limit: int = None,
    device=None,
) -> InMemoryDataLoader:

    all_data, all_target = [], []
    data_len = 0
    for data, target in data_loader:
        all_data.append(data)
        all_target.append(target)
        data_len += len(data)
        if limit and limit <= data_len:
            break

    data = torch.cat(tuple(all_data), dim=0)
    target = torch.cat(tuple(all_target), dim=0)

    if limit and limit < len(data):
        data = data[:limit]
        target = target[:limit]

    if device is not None:
        data = data.to(device)
        target = target.to(device)

    return InMemoryDataLoader(
        (data, target),
        allow_mixed_batches=allow_mixed_batches,
        batch_size=batch_size,
        order_by_class=order_by_class,
        shuffle=shuffle,
        shuffle_classes=shuffle_classes,
    )


def get_loader(  # pylint: disable=C0330
    dataset_name: str,
    train: bool = True,
    in_size: List[int] = None,
    batch_size: int = 1,
    shuffle: bool = True,
    normalize: bool = True,
    allow_mixed_batches: bool = True,
    order_by_class: bool = False,
    shuffle_classes: bool = False,
    device=None,
    limit=None,
):

    torch_loader = get_torch_loader(
        dataset_name,
        train=train,
        in_size=in_size,
        batch_size=128,
        shuffle=False,
        normalize=normalize,
    )

    return to_memory(
        torch_loader,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_classes=shuffle_classes,
        order_by_class=order_by_class,
        allow_mixed_batches=allow_mixed_batches,
        limit=limit,
        device=device,
    )
