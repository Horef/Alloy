from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

# Always surface elapsed time so long-running evaluations report progress even
# when the remaining-time estimate is unavailable (e.g. concurrent iterators).
_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed}, eta {remaining}, {rate_fmt}]"
_BAR_FORMAT_NOTOTAL = "{desc}: {n_fmt} [elapsed {elapsed}, {rate_fmt}]"


def track(iterable: Iterable[T], *, enabled: bool, description: str, total: int | None = None) -> Iterator[T]:
    """Wrap an iterable with a progress bar (including elapsed time) while keeping production opt-out simple."""
    yield from tqdm(
        iterable,
        total=total,
        desc=description,
        disable=not enabled,
        unit="item",
        bar_format=_BAR_FORMAT if total is not None else _BAR_FORMAT_NOTOTAL,
    )

