import torch

from piezojet.jarvis_dfpt import (
    DFPT_CACHE_SCHEMA,
    DFPTArchive,
    build_dfpt_provenance,
    source_born_to_internal,
    source_internal_strain_to_internal,
    tensor_sha256,
)
from piezojet.migrate_dfpt_conventions import migrate_payload


def test_vasp_born_axes_are_transposed_once_at_the_cache_boundary():
    source = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    internal = source_born_to_internal(source)
    assert torch.equal(internal, source.transpose(-1, -2))
    assert not torch.equal(internal, source)


def test_vasp_printed_force_derivative_has_the_internal_lambda_sign():
    # E(u, eta)=k u^2/2+c u eta.  OUTCAR's printed force derivative is
    # dF/deta=-c, precisely PiezoJet's Lambda in E=k u^2/2-u Lambda eta.
    k, c, eta = 5.0, 2.0, 3.0
    source_lambda = torch.tensor([[[-c, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    internal_lambda = source_internal_strain_to_internal(source_lambda)
    equilibrium = internal_lambda[0, 0, 0] * eta / k
    assert torch.isclose(equilibrium, torch.tensor(-c * eta / k))
    assert torch.isclose(k * equilibrium + c * eta, torch.tensor(0.0))


def test_schema_two_cache_migration_preserves_source_and_records_internal_convention():
    source_born = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    source_xi = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
    payload = {
        "schema": 2,
        "jid": "JVASP-toy",
        "born_charges": source_born,
        "internal_strain_tensors": source_xi,
        "force_constants": torch.eye(6).reshape(2, 3, 2, 3),
    }
    migrated = migrate_payload(payload)
    assert migrated["schema"] == DFPT_CACHE_SCHEMA
    assert torch.equal(migrated["born_charges_source"], source_born)
    assert torch.equal(migrated["born_charges"], source_born.transpose(-1, -2))
    assert torch.equal(migrated["internal_strain_tensors_source"], source_xi)
    assert torch.equal(migrated["internal_strain_tensors"], source_xi)
    assert torch.equal(migrated["force_constants"], payload["force_constants"])
    assert migrated["provenance"]["status"] == "legacy_migration_without_raw_archive"


def test_schema_four_provenance_hashes_raw_files_and_converted_tensors():
    payload = {
        "born_charges_source": torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
        "born_charges": torch.arange(9, dtype=torch.float32).reshape(1, 3, 3).transpose(-1, -2),
        "dynamical_eigenvalues": torch.ones(3),
        "dynamical_eigenvectors": torch.ones(3, 1, 3),
        "masses": torch.ones(1),
        "dynamical_matrix": torch.ones(1, 1, 3, 3),
        "force_constants": torch.ones(1, 1, 3, 3),
        "ionic_piezo_source": torch.ones(3, 6),
        "total_piezo_source": torch.ones(3, 6),
        "internal_strain_tensors_source": torch.ones(1, 3, 3),
        "internal_strain_tensors": torch.ones(1, 3, 3),
        "internal_strain_rounding_halfwidth_source": torch.full((1, 3, 3), 5e-7),
        "internal_strain_rounding_halfwidth": torch.full((1, 3, 3), 5e-7),
        "internal_strain_ions": torch.zeros(1, dtype=torch.long),
        "internal_strain_directions": torch.zeros(1, dtype=torch.long),
        "epsilon": {"epsilon": torch.eye(3)},
    }
    provenance = build_dfpt_provenance(
        archive=DFPTArchive("JVASP-toy", "JVASP-toy.zip", "https://example.invalid/toy.zip"),
        archive_bytes=b"archive",
        vasprun_bytes=b"xml",
        outcar_bytes=b"outcar",
        payload=payload,
    )
    assert provenance["source_archive"]["archive_sha256"] != provenance["source_archive"]["outcar_sha256"]
    assert provenance["force_constant_conversion"]["raw_fc_mass"] is False
    assert provenance["force_constant_conversion"]["converted_fc_mass"] is True
    assert provenance["tensor_sha256"]["born_charges"] == tensor_sha256(payload["born_charges"])
