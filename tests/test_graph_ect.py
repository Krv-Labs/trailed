"""Tests for graph ECT computation."""

import numpy as np
import pytest
import trailed_rust
from trailed.tabular import (
    compute_ect_from_numpy,
    compute_ect_from_pandas,
    compute_ect_from_polars,
    DataFrameEctTransformer,
)


def test_graph_ect_v_minus_e():
    """Verify that graph ECT = nodes ECT - edges ECT."""
    n_points = 10
    n_dirs = 8
    resolution = 16
    n_edges = 5
    
    points = np.random.randn(n_points, 3).astype(np.float32)
    edge_index = np.array([
        np.random.randint(0, n_points, n_edges),
        np.random.randint(0, n_points, n_edges)
    ], dtype=np.int64)
    
    # Ensure edge_index is valid (no self-loops for simpler reasoning)
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    n_edges = edge_index.shape[1]
    
    num_thetas = 8
    resolution = 16
    
    # 1. Compute graph ECT
    ect_graph = compute_ect_from_numpy(
        points,
        edge_index=edge_index,
        num_thetas=num_thetas,
        resolution=resolution,
        parallel=False
    )
    
    # 2. Compute nodes ECT
    ect_nodes = compute_ect_from_numpy(
        points,
        num_thetas=num_thetas,
        resolution=resolution,
        parallel=False,
        # Reuse directions and lin for exact comparison
        seed=42 
    )
    
    # 3. Compute edges ECT manually
    # Node heights
    directions = np.asarray(trailed_rust.generate_uniform_directions(num_thetas, 3, 42))
    nh = points @ directions
    
    # Edge heights (max of endpoints)
    eh = np.zeros((n_edges, num_thetas), dtype=np.float32)
    for i in range(n_edges):
        u, w = edge_index[:, i]
        eh[i] = np.maximum(nh[u], nh[w])
        
    lin = trailed_rust.generate_lin(1.0, resolution)
    batch_e = np.zeros(n_edges, dtype=np.int64)
    ect_edges = trailed_rust.compute_ect_points_forward(eh, batch_e, lin, 1, 500.0)[0]
    
    # Verify: graph_ect approx nodes_ect - edges_ect
    # Using high scale to make it close to discrete indicators
    ect_graph_high_scale = compute_ect_from_numpy(
        points,
        edge_index=edge_index,
        num_thetas=num_thetas,
        resolution=resolution,
        scale=500.0,
        parallel=False
    )
    
    ect_nodes_high_scale = compute_ect_from_numpy(
        points,
        num_thetas=num_thetas,
        resolution=resolution,
        scale=500.0,
        parallel=False
    )
    
    expected = ect_nodes_high_scale - ect_edges
    
    assert np.allclose(ect_graph_high_scale, expected, atol=1e-3)


def test_pandas_graph_ect():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "x": [0.0, 1.0, 0.0],
        "y": [0.0, 0.0, 1.0],
    })
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    
    ect = compute_ect_from_pandas(
        df,
        coord_columns=["x", "y"],
        edge_index=edge_index,
        num_thetas=8,
        resolution=8
    )
    
    assert ect.shape == (8, 8)


def test_polars_graph_ect():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({
        "x": [0.0, 1.0, 0.0],
        "y": [0.0, 0.0, 1.0],
    })
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    
    ect = compute_ect_from_polars(
        df,
        coord_columns=["x", "y"],
        edge_index=edge_index,
        num_thetas=8,
        resolution=8
    )
    
    assert ect.shape == (8, 8)


def test_transformer_graph_ect():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "x": np.random.randn(20),
        "y": np.random.randn(20),
        "group": np.repeat([0, 1], 10)
    })
    edge_index = np.array([[0, 1, 10, 11], [1, 2, 11, 12]], dtype=np.int64)
    
    transformer = DataFrameEctTransformer(
        coord_columns=["x", "y"],
        group_column="group",
        edge_index=edge_index,
        num_thetas=8,
        resolution=8
    )
    
    ect = transformer.fit_transform(df)
    assert ect.shape == (2, 8, 8)
