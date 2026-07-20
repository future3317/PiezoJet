"""Independent ASE oracle checks for the custom periodic-neighbor graph."""

import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list

from piezojet.data import _periodic_edges, load_gmtnet_records
from tests.data_paths import gmtnet_root


def _our_edge_set(frac: torch.Tensor, cell: torch.Tensor, cutoff: float, max_neighbors: int) -> set[tuple[int, int, int, int, int]]:
    edge_index, edge_shift = _periodic_edges(frac, cell, cutoff, max_neighbors)
    inverse_cell = torch.linalg.inv(cell)
    result = set()
    for source, target, shift_cartesian in zip(
        edge_index[0].tolist(), edge_index[1].tolist(), edge_shift
    ):
        shift_fractional = torch.round(shift_cartesian @ inverse_cell).to(torch.long)
        result.add((target, source, *shift_fractional.tolist()))
    return result


def _ase_edge_set(frac: torch.Tensor, cell: torch.Tensor, cutoff: float) -> set[tuple[int, int, int, int, int]]:
    centers, neighbors, shifts = primitive_neighbor_list(
        "ijS",
        np.ones(3, dtype=bool),
        cell.detach().cpu().numpy(),
        (frac @ cell).detach().cpu().numpy(),
        cutoff,
        self_interaction=False,
    )
    return {
        (int(center), int(neighbor), *map(int, shift))
        for center, neighbor, shift in zip(centers, neighbors, shifts)
    }


def test_custom_pbc_edges_match_ase_for_single_atom_all_image_shells():
    frac = torch.zeros(1, 3)
    cell = 3.0 * torch.eye(3)
    ours = _our_edge_set(frac, cell, cutoff=3.01, max_neighbors=16)
    oracle = _ase_edge_set(frac, cell, cutoff=3.01)
    assert ours == oracle
    assert len(ours) == 6


def test_neighbor_budget_never_splits_a_degenerate_cubic_shell():
    frac = torch.zeros(1, 3)
    cell = 3.0 * torch.eye(3)
    # A hard top-5 truncation would arbitrarily remove one of the six
    # symmetry-equivalent nearest periodic images.
    ours = _our_edge_set(frac, cell, cutoff=3.01, max_neighbors=5)
    oracle = _ase_edge_set(frac, cell, cutoff=3.01)
    assert ours == oracle
    assert len(ours) == 6


def test_strongly_skew_cell_enumerates_cancelling_large_lattice_shifts():
    frac = torch.zeros(1, 3)
    # 4*a-b=(0,-0.2,0) is inside the cutoff although both basis vectors are
    # much longer and the required integer shift lies outside [-1,1]^3.
    cell = torch.tensor(((1.0, 0.0, 0.0), (4.0, 0.2, 0.0), (0.0, 0.0, 5.0)))
    ours = _our_edge_set(frac, cell, cutoff=0.25, max_neighbors=100)
    oracle = _ase_edge_set(frac, cell, cutoff=0.25)
    assert ours == oracle
    assert (0, 0, 4, -1, 0) in ours
    assert (0, 0, -4, 1, 0) in ours


def test_custom_pbc_edges_are_all_valid_ase_neighbors_for_real_nonorthogonal_cells():
    records = load_gmtnet_records(gmtnet_root())
    # Deliberately mix a small, a many-atom, and a non-orthogonal cell rather
    # than relying only on a cubic toy.  The custom graph may truncate to K
    # nearest images, so every retained edge must be an ASE-valid edge.
    candidates = [records[0], max(records, key=lambda record: len(record["atoms"]["elements"]))]
    nonorthogonal = next(
        record
        for record in records
        if not np.allclose(
            np.asarray(record["atoms"]["lattice_mat"]),
            np.diag(np.diag(np.asarray(record["atoms"]["lattice_mat"]))),
        )
    )
    candidates.append(nonorthogonal)
    for record in candidates:
        frac = torch.tensor(record["atoms"]["coords"], dtype=torch.float32)
        cell = torch.tensor(record["atoms"]["lattice_mat"], dtype=torch.float32)
        ours = _our_edge_set(frac, cell, cutoff=5.0, max_neighbors=32)
        oracle = _ase_edge_set(frac, cell, cutoff=5.0)
        assert ours <= oracle
