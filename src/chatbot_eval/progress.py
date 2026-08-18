from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def track(iterable: Iterable[T], *, enabled: bool, description: str, total: int | None = None) -> Iterator[T]:
    """Wrap an iterable with a progress bar while keeping production opt-out simple."""
    yield from tqdm(iterable, total=total, desc=description, disable=not enabled, unit="item")

