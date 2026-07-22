"""Runtime-only DataLoader options for CUDA training and evaluation."""

from __future__ import annotations


def loader_options(
    num_workers: int,
    *,
    cuda: bool,
    persistent: bool = True,
    prefetch_factor: int = 2,
) -> dict[str, object]:
    """Return valid DataLoader options without changing the sample order."""
    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers cannot be negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    options: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": bool(cuda),
    }
    if workers:
        options.update({
            "persistent_workers": bool(persistent),
            "prefetch_factor": int(prefetch_factor),
        })
    return options
