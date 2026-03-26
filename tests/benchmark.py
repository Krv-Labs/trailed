import torch
import time
import pytest
from torch_geometric.data import Data

import dect as dect_pkg


def _make_data(ect_type: str, num_nodes: int, num_features: int) -> Data:
    x = torch.randn(num_nodes, num_features, requires_grad=True)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    data = Data(x=x, batch=batch)

    if ect_type in ["edges", "faces"]:
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        data.edge_index = edge_index
    if ect_type == "faces":
        face = torch.randint(0, num_nodes, (3, num_nodes))
        data.face = face

    return data


def _run_layer(
    ect_type: str,
    num_nodes: int,
    num_features: int,
    num_thetas: int,
    bump_steps: int,
) -> float:
    config = dect_pkg.EctConfig(
        num_thetas=num_thetas,
        resolution=bump_steps,
        ect_type=ect_type,
        ambient_dim=num_features,
    )

    v = torch.randn(num_features, num_thetas)
    v /= v.pow(2).sum(axis=0).sqrt()
    layer = dect_pkg.EctLayer(config, directions=v)

    data = _make_data(
        ect_type=ect_type, num_nodes=num_nodes, num_features=num_features
    )

    # Warmup before timing to avoid one-time overheads.
    layer(data).sum().backward()
    data.x.grad.zero_()

    start = time.perf_counter()
    for _ in range(3):
        out = layer(data)
        out.sum().backward()
        data.x.grad.zero_()
    end = time.perf_counter()

    return (end - start) / 3


@pytest.mark.parametrize("ect_type", ["points", "edges", "faces"])
def test_benchmark_smoke(ect_type: str) -> None:
    torch.manual_seed(0)
    avg_time = _run_layer(
        ect_type=ect_type,
        num_nodes=64,
        num_features=3,
        num_thetas=16,
        bump_steps=16,
    )
    assert avg_time > 0.0
