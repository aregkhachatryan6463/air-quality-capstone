from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class SplitIndices:
    train_end: int
    val_end: int
    n: int


def chronological_split(n: int, train_ratio: float = 0.70, val_ratio: float = 0.15) -> SplitIndices:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return SplitIndices(train_end=train_end, val_end=val_end, n=n)


def walk_forward_slices(
    n: int,
    *,
    min_train_size: int,
    test_size: int,
    step: int,
) -> Iterator[tuple[slice, slice]]:
    if n <= min_train_size + test_size:
        return
    start = min_train_size
    while start + test_size <= n:
        yield slice(0, start), slice(start, start + test_size)
        start += step

