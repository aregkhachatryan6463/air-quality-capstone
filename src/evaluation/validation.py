from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal


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
    mode: Literal["expanding", "sliding"] = "expanding",
    train_window: int | None = None,
    max_folds: int | None = None,
) -> Iterator[tuple[slice, slice]]:
    """
    Expanding: train slice 0:train_end, test [train_end:train_end+test_size) with train_end from min_train_size.
    Sliding: train slice (train_end - train_window):train_end if train_window set; else like expanding.
    """
    if mode == "sliding" and (train_window is None or int(train_window) < min_train_size + 1):
        raise ValueError("sliding mode requires train_window >= min_train_size + 1")
    n_folds = 0
    if mode == "expanding":
        if n <= min_train_size + test_size:
            return
        start = min_train_size
        while start + test_size <= n:
            if max_folds is not None and n_folds >= int(max_folds):
                return
            yield slice(0, start), slice(start, start + test_size)
            n_folds += 1
            start += step
        return
    # sliding
    start = min_train_size
    tw = int(train_window)  # type: ignore
    while start + test_size <= n:
        if max_folds is not None and n_folds >= int(max_folds):
            return
        t0 = max(0, start - tw)
        yield slice(t0, start), slice(start, start + test_size)
        n_folds += 1
        start += step
