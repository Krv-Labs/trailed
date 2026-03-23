import pytest
import torch
from torch_geometric.data import Data

import dect as dect_pkg


def _make_data(ect_type: str, num_nodes: int, num_features: int) -> Data:
    x = torch.randn(num_nodes, num_features, requires_grad=True)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(x=x, batch=batch)
    if ect_type in ["edges", "faces"]:
        edge_index = torch.stack(
            [torch.arange(num_nodes - 1), torch.arange(1, num_nodes)], dim=0
        )
        data.edge_index = edge_index
    if ect_type == "faces":
        face = torch.stack(
            [
                torch.arange(num_nodes - 2),
                torch.arange(1, num_nodes - 1),
                torch.arange(2, num_nodes),
            ],
            dim=0,
        )
        data.face = face
    return data


@pytest.mark.parametrize(
    "ect_type", ["points", "points_derivative", "edges", "faces"]
)
def test_forward_and_backward_are_finite(ect_type: str) -> None:
    torch.manual_seed(0)

    num_nodes = 10
    num_features = 3
    num_thetas = 8
    bump_steps = 16

    config = dect_pkg.EctConfig(
        num_thetas=num_thetas,
        bump_steps=bump_steps,
        ect_type=ect_type,
        num_features=num_features,
    )

    v = torch.randn(num_features, num_thetas)
    v /= v.pow(2).sum(axis=0).sqrt()
    layer = dect_pkg.EctLayer(config, V=v)

    data = _make_data(
        ect_type=ect_type, num_nodes=num_nodes, num_features=num_features
    )
    out = layer(data)

    assert out.shape == (1, bump_steps, num_thetas)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert data.x.grad is not None
    assert data.x.grad.shape == data.x.shape
    assert torch.isfinite(data.x.grad).all()
