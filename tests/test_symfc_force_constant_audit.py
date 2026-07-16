from piezojet.data import load_gmtnet_records
from piezojet.jarvis_dfpt import JarvisDFPTCache
from piezojet.project_config import load_project_config
from piezojet.symfc_force_constant_audit import symfc_project_force_constants


def test_symfc_projection_is_read_only_and_restores_acoustic_sum_rule():
    config = load_project_config("config.yaml")
    record = next(
        item for item in load_gmtnet_records(config["data_root"])
        if str(item["JARVIS_ID"]) == "JVASP-22529"
    )
    payload = JarvisDFPTCache(config["jarvis_dfpt_dir"]).load("JVASP-22529")
    assert payload is not None
    original = payload["force_constants"].clone()
    projected, audit = symfc_project_force_constants(record, original)
    assert projected.shape == tuple(original.shape)
    assert audit["symfc_acoustic_relative_residual"] < 1e-10
    # The diagnostic must never mutate the caller's cached target tensor.
    assert original.equal(payload["force_constants"])
