import torch

from piezojet.train import born_loss


def test_born_loss_is_mean_of_single_material_losses_not_atom_weighted() -> None:
    torch.manual_seed(4)
    prediction = torch.randn(5, 3, 3)
    target = torch.randn(5, 3, 3)
    batch = torch.tensor([0, 0, 1, 1, 1])
    mask = torch.ones(5, dtype=torch.bool)

    batched = born_loss(prediction, target, mask, batch)
    singles = []
    for graph in (0, 1):
        selected = batch == graph
        singles.append(born_loss(prediction[selected], target[selected], torch.ones(selected.sum(), dtype=torch.bool), torch.zeros(selected.sum(), dtype=torch.long)))

    assert torch.allclose(batched, torch.stack(singles).mean(), atol=1e-7, rtol=1e-7)
