"""Offline representability audit for the maintained bond-quadratic Hessian class.

This module is deliberately an *oracle*, never a model head or a training
target.  For a fixed periodic graph it finds the least-squares projection of a
true Cartesian force-constant matrix onto the generous class

    Phi = sum_e B_e^T K_e B_e,  K_e = K_e^T,

with an unconstrained symmetric 3x3 matrix for every directed graph edge.  It
therefore removes encoder, optimization, and the learned four-tensor-basis
restriction.  A large residual falsifies the bond-Laplacian model class on the
given graph; a small residual only says that learning/parameterization remains
the bottleneck.  It is not a pseudoinverse lift and is never used at inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
from .projectors import translation_projector
from .project_config import load_project_config
from .train import load_explicit_splits


def symmetric_edge_stiffness(components: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 6)`` Cartesian components to symmetric 3x3 matrices.

    Component order is ``(xx, yy, zz, yz, xz, xy)``; it is unrelated to the
    engineering-strain convention and is used only for a symmetric matrix
    basis in this offline linear projection.
    """
    if components.shape[-1] != 6:
        raise ValueError("symmetric stiffness components must have trailing size 6")
    result = components.new_zeros(*components.shape[:-1], 3, 3)
    result[..., 0, 0], result[..., 1, 1], result[..., 2, 2] = components.unbind(-1)[:3]
    result[..., 1, 2] = result[..., 2, 1] = components[..., 3]
    result[..., 0, 2] = result[..., 2, 0] = components[..., 4]
    result[..., 0, 1] = result[..., 1, 0] = components[..., 5]
    return result


def bond_laplacian_from_stiffness(
    atoms: int, edge_index: torch.Tensor, stiffness: torch.Tensor,
) -> torch.Tensor:
    """Assemble ``sum_e B_e^T K_e B_e`` in atom-major Cartesian layout."""
    if edge_index.shape[0] != 2 or edge_index.shape[1] != stiffness.shape[0]:
        raise ValueError("edge_index and stiffness must have one common edge dimension")
    if stiffness.shape[1:] != (3, 3):
        raise ValueError("stiffness must have shape (edges, 3, 3)")
    source, target = edge_index.to(dtype=torch.long)
    blocks = stiffness.new_zeros(atoms, atoms, 3, 3)
    blocks.index_put_((source, source), stiffness, accumulate=True)
    blocks.index_put_((target, target), stiffness, accumulate=True)
    blocks.index_put_((source, target), -stiffness, accumulate=True)
    blocks.index_put_((target, source), -stiffness, accumulate=True)
    blocks = 0.5 * (blocks + blocks.permute(1, 0, 3, 2))
    return blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)


def bond_laplacian_design(atoms: int, edge_index: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return linear design matrix mapping all edge stiffness components to Phi."""
    edge_count = int(edge_index.shape[1])
    unit = torch.eye(6, dtype=dtype, device=edge_index.device)
    columns = []
    for edge in range(edge_count):
        edge_stiffness = symmetric_edge_stiffness(unit)
        one_edge = edge_index[:, edge : edge + 1].expand(2, 6)
        # Each of the six columns is one stiffness component for this edge.
        matrices = [
            bond_laplacian_from_stiffness(atoms, one_edge[:, component : component + 1], edge_stiffness[component : component + 1])
            for component in range(6)
        ]
        columns.extend(matrix.reshape(-1) for matrix in matrices)
    return torch.stack(columns, dim=1) if columns else torch.empty(9 * atoms * atoms, 0, dtype=dtype)


def _unique_undirected_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    """Collapse reverse periodic graph edges, whose symmetric K sums are redundant."""
    pairs = torch.sort(edge_index.to(dtype=torch.long).transpose(0, 1), dim=1).values
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not len(pairs):
        return edge_index.new_empty((0, 2), dtype=torch.long)
    return torch.unique(pairs, dim=0)


def _bond_oracle_normal_system(target: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytically assemble the least-squares normal system for bond K values.

    Working in pair/block form is exactly equivalent to materializing the
    tall design matrix, but costs O((6P)^2) rather than O(N^2 P) storage.
    """
    atoms = target.shape[0] // 3
    pairs = _unique_undirected_pairs(edge_index)
    basis = symmetric_edge_stiffness(torch.eye(6, dtype=target.dtype, device=target.device))
    pair_count = int(pairs.shape[0])
    block_inner = torch.einsum("aij,bij->ab", basis, basis)
    gram = target.new_zeros(pair_count, 6, pair_count, 6)
    rhs = target.new_zeros(pair_count, 6)
    blocks = target.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
    for left, (i_tensor, j_tensor) in enumerate(pairs):
        i, j = int(i_tensor), int(j_tensor)
        rhs[left] = torch.einsum("aij,ij->a", basis, blocks[i, i] + blocks[j, j] - blocks[i, j] - blocks[j, i])
        for right, (k_tensor, l_tensor) in enumerate(pairs):
            k, l = int(k_tensor), int(l_tensor)
            shared_diagonals = int(i == k or i == l) + int(j == k or j == l)
            cross_blocks = 2 if left == right else 0
            gram[left, :, right, :] = (shared_diagonals + cross_blocks) * block_inner
    return pairs, gram.reshape(6 * pair_count, 6 * pair_count), rhs.reshape(-1)


def offdiagonal_block_skew_fraction(matrix: torch.Tensor, atoms: int, eps: float = 1e-12) -> float:
    """Fraction of force-constant norm in off-diagonal blocks not symmetric in Cartesian axes."""
    blocks = matrix.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
    mask = ~torch.eye(atoms, dtype=torch.bool, device=matrix.device)
    offdiag = blocks[mask]
    skew = 0.5 * (offdiag - offdiag.transpose(-1, -2))
    return float(torch.linalg.vector_norm(skew) / torch.linalg.vector_norm(matrix).clamp_min(eps))


def project_force_constants(target: torch.Tensor, edge_index: torch.Tensor) -> dict[str, Any]:
    """Project one true force-constant matrix onto the generous bond oracle."""
    if target.ndim != 2 or target.shape[0] != target.shape[1] or target.shape[0] % 3:
        raise ValueError("target must be a square Cartesian matrix of size 3N")
    atoms = target.shape[0] // 3
    target = 0.5 * (target + target.transpose(0, 1))
    projector, _ = translation_projector(atoms, target)
    target = projector @ target @ projector
    pairs, gram, rhs = _bond_oracle_normal_system(target, edge_index)
    solution = torch.linalg.lstsq(gram, rhs.unsqueeze(-1)).solution.squeeze(-1)
    stiffness = symmetric_edge_stiffness(solution.reshape(-1, 6))
    prediction = bond_laplacian_from_stiffness(atoms, pairs.transpose(0, 1), stiffness)
    residual = prediction - target
    target_norm = torch.linalg.vector_norm(target).clamp_min(torch.finfo(target.dtype).eps)
    return {
        "atoms": atoms,
        "edges": int(edge_index.shape[1]),
        "parameters": int(gram.shape[0]),
        "design_rank": int(torch.linalg.matrix_rank(gram)),
        "target_frobenius": float(target_norm),
        "relative_frobenius_error": float(torch.linalg.vector_norm(residual) / target_norm),
        "explained_frobenius_fraction": float(1.0 - torch.linalg.vector_norm(residual).square() / target_norm.square()),
        "offdiagonal_block_skew_fraction": offdiagonal_block_skew_fraction(target, atoms),
        "translation_residual_max_abs": float((prediction @ torch.ones(3 * atoms, 3, dtype=target.dtype, device=target.device)).abs().max()),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = torch.tensor([row["relative_frobenius_error"] for row in rows], dtype=torch.float64)
    explained = torch.tensor([row["explained_frobenius_fraction"] for row in rows], dtype=torch.float64)
    return {
        "materials": len(rows),
        "relative_frobenius_error": {
            "mean": float(values.mean()), "median": float(values.median()),
            "p90": float(torch.quantile(values, 0.9)), "max": float(values.max()),
        },
        "explained_frobenius_fraction": {
            "mean": float(explained.mean()), "median": float(explained.median()),
            "p10": float(torch.quantile(explained, 0.1)), "min": float(explained.min()),
        },
    }


def run_oracle(config: dict[str, Any], split_path: Path, output: Path, max_materials: int | None = None) -> dict[str, Any]:
    """Run on strict-complete *training* IDs only; no validation/test selection."""
    records = load_gmtnet_records(config["data_root"])
    known_ids = {str(record["JARVIS_ID"]) for record in records}
    split = load_explicit_splits(split_path, known_ids)
    ids = list(split["train"])
    if max_materials is not None:
        ids = ids[:max_materials]
    kwargs = {
        "processed_dir": config["processed_dir"],
        "cache_key": graph_cache_key(records, config["cutoff"], config["max_neighbors"]),
        "dfpt_dir": config["jarvis_dfpt_dir"],
        "strain_completion_dir": config["jarvis_strain_completion_dir"],
        "dfpt_total_consistency_absolute_tolerance": config["dfpt_total_consistency_absolute_tolerance_c_per_m2"],
        "dfpt_total_consistency_relative_tolerance": config["dfpt_total_consistency_relative_tolerance"],
    }
    dataset = PiezoDataset(records, ids, config["cutoff"], config["max_neighbors"], **kwargs)
    rows = []
    for index, graph in enumerate(dataset, start=1):
        atoms = int(graph.num_nodes)
        target = graph.dfpt_force_constants_flat.reshape(atoms, atoms, 3, 3)
        target = target.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms).to(torch.float64)
        result = project_force_constants(target, graph.edge_index)
        result["material_id"] = str(graph.material_id)
        rows.append(result)
        if index % 10 == 0 or index == len(dataset):
            print(f"oracle_progress={index}/{len(dataset)}", flush=True)
    payload = {
        "schema": 1,
        "diagnostic": "offline_generous_bond_laplacian_oracle",
        "split": str(split_path),
        "selection": "strict-complete training IDs in split order; no validation/test used",
        "interpretation": (
            "This is an offline linear least-squares projection with unrestricted symmetric K per graph edge. "
            "It is more expressive than the maintained four-basis learned head and is not used for prediction, training, or a Lambda lift. "
            "A large error is evidence against the graph bond-Laplacian class; a small error leaves parameterization/optimization unresolved."
        ),
        "summary": _summary(rows),
        "materials": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--splits-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-materials", type=int)
    args = parser.parse_args()
    config = load_project_config(args.config)
    splits_file = args.splits_file or Path(config["strict_completion_split_file"])
    print(json.dumps(run_oracle(config, splits_file, args.output, args.max_materials), indent=2))


if __name__ == "__main__":
    main()
