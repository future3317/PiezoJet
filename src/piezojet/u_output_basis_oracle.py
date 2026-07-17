"""Read-only same-ID oracle for the maintained direct-U output basis.

The oracle never predicts held-out data and is not an inference fallback.  It
loads a trained capacity checkpoint, freezes its encoder/tensor features, and
solves independent per-node coefficients in the exact tensor bases exposed by
the local or global U head.  This separates a deficient equivariant output
basis from deficient structure-to-coefficient conditioning.  An unrestricted
translation-free node lookup is reported as a pipeline sanity upper bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
from .model import AtomCoordinateResponsePotential
from .project_config import load_project_config
from .u_capacity_adjudication import UCapacityModel, _true_graph_tensors


def _head_bases(model: UCapacityModel, batch) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return every per-node tensor basis exposed by the selected U head.

    The oracle must track the actual candidate model class.  In particular,
    ``global_l3`` adds the independent STF edge-octupole basis that made the
    samples32 capacity gate pass; omitting it would measure the superseded
    global-l2 head rather than the maintained readout.
    """
    features = model.encoder(batch)
    identity = torch.eye(3, dtype=features.vector.dtype, device=features.vector.device)
    if model.architecture == "local":
        vector, quadrupole = features.vector, features.quadrupole
        spectral = None
    else:
        assert model.local_polar is not None and model.global_context is not None
        local_polar = model.local_polar(features)
        _, spectral_operator = model.global_context(
            batch, batch.batch, local_polar, return_operator=True
        )
        vector, quadrupole = model.head._nonlocal_features(features, batch.batch)
        vector = torch.einsum("rc,nci->nri", model.head.vector_rank_mix, vector)
        quadrupole = torch.einsum(
            "rc,ncij->nrij", model.head.quadrupole_rank_mix, quadrupole
        )
        spectral = spectral_operator[batch.batch]
    vector_identity = torch.einsum("nca,jk->ncajk", vector, identity)
    identity_vector = 0.5 * (
        torch.einsum("aj,nck->ncajk", identity, vector)
        + torch.einsum("ak,ncj->ncajk", identity, vector)
    )
    vector_quadrupole = torch.einsum("nca,ncjk->ncajk", vector, quadrupole)
    quadrupole_vector = 0.5 * (
        torch.einsum("ncaj,nck->ncajk", quadrupole, vector)
        + torch.einsum("ncak,ncj->ncajk", quadrupole, vector)
    )
    basis_families = [
        vector_identity,
        identity_vector,
        vector_quadrupole,
        quadrupole_vector,
    ]
    if model.architecture == "global_l3":
        # ``_octupole_features`` already returns one STF rank-three basis per
        # cross-rank channel and uses the same complete-shell graph as forward.
        basis_families.append(model.head._octupole_features(features, batch))
    bases = torch.stack(basis_families, dim=2).flatten(1, 2)
    return bases, spectral


def _independent_components(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten a polar x symmetric-rank-2 tensor to 18 components per node."""
    return torch.stack(
        (
            tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2],
            tensor[..., 1, 2], tensor[..., 0, 2], tensor[..., 0, 1],
        ),
        dim=-1,
    ).flatten(-2, -1)


def _design_matrix(bases: torch.Tensor) -> torch.Tensor:
    """Apply graph-mean subtraction to independent per-node basis weights."""
    nodes, basis_count, _ = bases.shape
    projector = torch.eye(nodes, dtype=bases.dtype, device=bases.device)
    projector = projector - torch.full_like(projector, 1.0 / nodes)
    design = torch.einsum("ij,jbk->ikjb", projector, bases)
    return design.reshape(nodes * 18, nodes * basis_count)


def _projection_metrics(design: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    design = design.to(dtype=torch.float64, device="cpu")
    target = target.to(dtype=torch.float64, device="cpu").reshape(-1)
    # SVD gives both a stable orthogonal projection and the numerical rank;
    # solving an underdetermined coefficient vector is unnecessary here.
    left, singular, _ = torch.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * torch.finfo(design.dtype).eps * singular.max()
    rank = int((singular > tolerance).sum())
    projected = left[:, :rank] @ (left[:, :rank].transpose(0, 1) @ target)
    target_norm = torch.linalg.vector_norm(target).clamp_min(1e-15)
    projected_norm = torch.linalg.vector_norm(projected)
    cosine = torch.dot(projected, target) / (projected_norm * target_norm).clamp_min(1e-15)
    return {
        "rows": int(design.shape[0]),
        "columns": int(design.shape[1]),
        "rank": rank,
        "relative_frobenius_residual": float(torch.linalg.vector_norm(projected - target) / target_norm),
        "maximum_cosine": float(cosine),
        "amplitude_ratio": float(projected_norm / target_norm),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"Fresh output required: {args.output}")
    config = load_project_config(args.config)
    ids = json.loads(args.material_ids_file.read_text(encoding="utf-8-sig"))
    if len(ids) != 32:
        raise ValueError("This oracle is restricted to the declared samples32 IDs")
    records = load_gmtnet_records(config["data_root"])
    dataset = PiezoDataset(
        records, [str(value) for value in ids], float(config["cutoff"]),
        int(config["max_neighbors"]), processed_dir=config["processed_dir"],
        cache_key=graph_cache_key(records, float(config["cutoff"]), int(config["max_neighbors"])),
        dfpt_dir=config["jarvis_dfpt_dir"],
        strain_completion_dir=config["jarvis_strain_completion_dir"],
        elastic_targets_path=config.get("elastic_targets_path"),
    )
    batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)))
    device = torch.device(args.device)
    batch = batch.to(device)
    response = AtomCoordinateResponsePotential(
        optical_regularization=float(config["optical_regularization"]),
        optical_stability_cutoff=float(config["optical_stability_cutoff"]),
        optical_solve_policy="regularized",
    ).to(device)
    truths = _true_graph_tensors(batch, response)
    model = UCapacityModel(config, args.architecture, False).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    bases, spectral = _head_bases(model, batch)
    rows: list[dict] = []
    for index, material_id in enumerate(ids):
        start, stop = int(batch.ptr[index]), int(batch.ptr[index + 1])
        graph_bases = _independent_components(bases[start:stop])
        if spectral is not None:
            graph_bases = torch.cat(
                (graph_bases, _independent_components(spectral[start:stop]).unsqueeze(1)),
                dim=1,
            )
        target = truths[index]["u_tensor"]
        target_components = _independent_components(target)
        basis_metrics = _projection_metrics(
            _design_matrix(graph_bases), target_components
        )
        nodes = stop - start
        node_projector = torch.eye(nodes, dtype=target.dtype, device=target.device)
        node_projector -= torch.full_like(node_projector, 1.0 / nodes)
        unrestricted = torch.kron(
            node_projector,
            torch.eye(18, dtype=target.dtype, device=target.device),
        )
        lookup_metrics = _projection_metrics(unrestricted, target_components)
        rows.append({
            "material_id": str(material_id),
            "atoms": nodes,
            "current_head_basis": basis_metrics,
            "unrestricted_translation_free_lookup": lookup_metrics,
        })
    def mean(path: str) -> float:
        return float(sum(row["current_head_basis"][path] for row in rows) / len(rows))
    summary = {
        "mean_relative_frobenius_residual": mean("relative_frobenius_residual"),
        "mean_maximum_cosine": mean("maximum_cosine"),
        "mean_amplitude_ratio": mean("amplitude_ratio"),
        "worst_relative_frobenius_residual": max(
            row["current_head_basis"]["relative_frobenius_residual"] for row in rows
        ),
        "lookup_worst_relative_frobenius_residual": max(
            row["unrestricted_translation_free_lookup"]["relative_frobenius_residual"] for row in rows
        ),
    }
    result = {
        "schema": 1,
        "diagnostic": "same_id_direct_u_output_basis_oracle",
        "selection": "declared strict-train samples32 only; frozen validation/test not read",
        "architecture": args.architecture,
        "checkpoint": str(args.checkpoint),
        "interpretation_boundary": "read-only basis/lookup upper bound; never a prediction path or production fallback",
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--material-ids-file", type=Path,
        default=Path("data/processed/capacity_probe_ids/samples32_ids.json"),
    )
    parser.add_argument(
        "--architecture", choices=("local", "global", "global_l3"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
