from types import SimpleNamespace

import torch

from piezojet.multiresponse import multiresponse_loss


def test_multiresponse_mask_skips_missing_task_without_nan():
    prediction = {"piezo": torch.ones(2, 1), "elastic": torch.ones(2, 1), "dielectric_electronic": torch.ones(2, 1), "dielectric_ionic": torch.ones(2, 1)}
    batch = SimpleNamespace(
        y_piezo=torch.zeros(2, 1), y_elastic=torch.zeros(2, 1), y_dielectric_e=torch.zeros(2, 1), y_dielectric_i=torch.zeros(2, 1),
        mask_piezo=torch.tensor([True, True]), mask_elastic=torch.tensor([True, False]), mask_dielectric_e=torch.tensor([False, False]), mask_dielectric_i=torch.tensor([True, False]),
    )
    loss, details = multiresponse_loss(prediction, batch, {name: torch.tensor(1.0) for name in prediction})
    assert torch.isfinite(loss)
    assert details["piezo_count"] == 2
    assert details["elastic_count"] == 1
    assert details["dielectric_electronic_count"] == 0
    assert details["dielectric_ionic_count"] == 1
