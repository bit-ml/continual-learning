from typing import Iterator, List, Tuple
import torch
from torch import Tensor


class InMemoryDataLoader:
    def __init__(  # pylint: disable=C0330
        self,
        dataset: Tuple[Tensor, Tensor],
        allow_mixed_batches: bool = True,
        batch_size: int = 1,
        order_by_class: bool = False,
        shuffle: bool = True,
        shuffle_classes: bool = False,
        classes: List[int] = None,
    ) -> None:
        data, target = dataset
        if target.dtype != torch.long:
            raise ValueError("Expected long tensor, got " + str(target.dtype))

        self._shuffle = bool(shuffle)
        self._batch_size = int(batch_size)
        self._ordered_by_class = bool(order_by_class)
        self._allow_mixed_batches = bool(allow_mixed_batches)

        self._nclasses = target.max().item() + 1

        if classes is not None:
            selected_idx = torch.zeros(
                len(target), dtype=torch.uint8, device=target.device
            )
            for class_label in classes:
                selected_idx |= target == class_label
            data = data[selected_idx]
            target = target[selected_idx]

        self._nexamples = len(data)

        self._class_ends = []  # type: List[int]
        if order_by_class:
            new_idxs = torch.empty_like(target)

            if shuffle_classes:
                classes = torch.randperm(self._nclasses, device=target.device)
            else:
                classes = range(self._nclasses)

            start = 0
            for class_label in classes:
                idxs = (target == class_label).nonzero()
                if idxs.nelement() > 0:
                    idxs.squeeze_(1)
                    end = start + len(idxs)
                    new_idxs[start:end] = idxs
                    self._class_ends.append(end)
                    start = end
            data, target = data[new_idxs], target[new_idxs]

        self.data = data
        self.target = target

        self._crt_idx = None
        self._crt_class = None
        self._perm_idxs = None

    def __len__(self) -> int:
        return self._nexamples

    @property
    def nclasses(self) -> int:
        return self._nclasses

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value) -> None:
        if self._crt_idx is not None:
            raise ValueError("Shuffle should not be set while iterating.")
        self._shuffle = bool(value)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value) -> None:
        if value <= 0:
            raise ValueError("Expected positive value, got " + str(value))
        if self._crt_idx is not None:
            raise ValueError("Batch size should not be set while iterating.")
        self._batch_size = int(value)

    @property
    def ordered_by_class(self) -> bool:
        return self._ordered_by_class

    @property
    def allow_mixed_batches(self) -> bool:
        return self._allow_mixed_batches

    @allow_mixed_batches.setter
    def allow_mixed_batches(self, value) -> None:
        self._allow_mixed_batches = bool(value)

    def to(self, device) -> None:  # pylint: disable=C0103
        self.data = self.data.to(device)
        self.target = self.target.to(device)

    def reset(self) -> None:
        self._crt_idx = None
        self._crt_class = None
        self._perm_idxs = None  # type: Tensor

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        self._crt_idx = 0  # type: int
        if self._ordered_by_class:
            self._crt_class = 0  # type: int

        if self.shuffle:
            if self.ordered_by_class:
                self._perm_idxs = torch.empty_like(self.target)
                start = 0
                for end in self._class_ends:
                    torch.randperm(end - start, out=self._perm_idxs[start:end])
                    self._perm_idxs[start:end] += start
                    start = end
            else:
                self._perm_idxs = torch.randperm(
                    len(self.data), device=self.data.device
                )
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self._crt_idx >= self._nexamples:
            self.reset()
            raise StopIteration

        start_idx = self._crt_idx
        end_idx = self._crt_idx + self.batch_size
        end_idx = min(end_idx, self._nexamples)

        if self._ordered_by_class:
            if not self.allow_mixed_batches:
                end_idx = min(end_idx, self._class_ends[self._crt_class])
            while self._class_ends[self._crt_class] <= end_idx:
                self._crt_class += 1
                if self._crt_class == len(self._class_ends):
                    break

        self._crt_idx = end_idx

        if self.shuffle:
            data = self.data[self._perm_idxs[start_idx:end_idx]]
            target = self.target[self._perm_idxs[start_idx:end_idx]]
        else:
            data = self.data[start_idx:end_idx]
            target = self.target[start_idx:end_idx]

        return data, target
