from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from piezojet.electrostatic_a0_fold_adjudication import (
    _clear_runtime_caches,
    _replace_progress_checkpoint,
    _train_evaluation_due,
    _training_schedule_tail,
)


def _tokens(loader: DataLoader) -> list[int]:
    values: list[int] = []
    for batch in loader:
        values.extend(int(value) for value in batch.token.reshape(-1))
    return values


def test_schedule_tail_matches_historical_block_slicing() -> None:
    dataset = [
        Data(
            x=torch.zeros(1, 1),
            token=torch.tensor([index], dtype=torch.long),
            num_nodes=1,
        )
        for index in range(8)
    ]
    schedule = [7, 2, 4, 1, 6, 0, 5, 3, 1, 5, 7, 0]
    logical_batch = 4
    microbatch = 2
    start_update = 1

    historical: list[int] = []
    for block_start, block_end in ((1, 2), (2, 3)):
        block_indices = schedule[
            block_start * logical_batch : block_end * logical_batch
        ]
        historical.extend(_tokens(DataLoader(
            Subset(dataset, block_indices), batch_size=microbatch, shuffle=False
        )))

    persistent = DataLoader(
        Subset(
            dataset,
            _training_schedule_tail(schedule, start_update, logical_batch),
        ),
        batch_size=microbatch,
        shuffle=False,
    )
    assert _tokens(persistent) == historical


def test_train_evaluation_schedule_preserves_legacy_and_final() -> None:
    assert _train_evaluation_due(50, 1500, 0)
    assert _train_evaluation_due(250, 1500, 250)
    assert not _train_evaluation_due(300, 1500, 250)
    assert _train_evaluation_due(1500, 1500, 250)


def test_progress_checkpoint_is_exact_copy_or_link(tmp_path: Path) -> None:
    source = tmp_path / "update.pt"
    progress = tmp_path / "progress.pt"
    payload = {"model": {"weight": torch.arange(5)}, "optimizer": {"step": 7}}
    torch.save(payload, source)
    _replace_progress_checkpoint(source, progress)
    restored = torch.load(progress, map_location="cpu", weights_only=False)
    assert restored["optimizer"]["step"] == 7
    assert torch.equal(restored["model"]["weight"], payload["model"]["weight"])
    assert source.read_bytes() == progress.read_bytes()


def test_clear_runtime_caches_drops_unregistered_tensor_references() -> None:
    module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    module[0]._geometry_cache = torch.ones(1)
    module[1]._geometry_cache = {"cached": torch.ones(1)}
    _clear_runtime_caches(module)
    assert module[0]._geometry_cache is None
    assert module[1]._geometry_cache is None
