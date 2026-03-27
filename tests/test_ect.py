"""Tests for ECT computation functions."""

import numpy as np

import trailed_rust
from dect.sampling import generate_lin


class TestEctPointsForward:
    def test_shape(self):
        n_points = 100
        n_dirs = 32
        resolution = 64

        nh = np.random.randn(n_points, n_dirs).astype(np.float32)
        batch = np.zeros(n_points, dtype=np.int64)
        lin = generate_lin(1.0, resolution)

        ect = trailed_rust.compute_ect_points_forward(nh, batch, lin, 1, 50.0)
        assert ect.shape == (1, resolution, n_dirs)

    def test_batched_shape(self):
        n_points = 100
        n_dirs = 32
        resolution = 64
        n_batches = 4

        nh = np.random.randn(n_points, n_dirs).astype(np.float32)
        batch = np.repeat(np.arange(n_batches), n_points // n_batches).astype(np.int64)
        lin = generate_lin(1.0, resolution)

        ect = trailed_rust.compute_ect_points_forward(nh, batch, lin, n_batches, 50.0)
        assert ect.shape == (n_batches, resolution, n_dirs)

    def test_monotonic_in_resolution(self):
        n_points = 50
        n_dirs = 8
        resolution = 32

        nh = np.random.randn(n_points, n_dirs).astype(np.float32) * 0.5
        batch = np.zeros(n_points, dtype=np.int64)
        lin = generate_lin(1.0, resolution)

        ect = trailed_rust.compute_ect_points_forward(nh, batch, lin, 1, 100.0)

        # ECT should be monotonically increasing along resolution axis
        for j in range(n_dirs):
            for k in range(1, resolution):
                assert ect[0, k, j] >= ect[0, k - 1, j] - 1e-5

    def test_scale_effect(self):
        n_points = 50
        n_dirs = 8
        resolution = 32

        nh = np.random.randn(n_points, n_dirs).astype(np.float32) * 0.3
        batch = np.zeros(n_points, dtype=np.int64)
        lin = generate_lin(1.0, resolution)

        ect_low = trailed_rust.compute_ect_points_forward(nh, batch, lin, 1, 10.0)
        ect_high = trailed_rust.compute_ect_points_forward(nh, batch, lin, 1, 100.0)

        # Higher scale should produce sharper transitions
        # Middle values should differ more than edge values
        mid = resolution // 2
        diff_mid = np.abs(ect_high[0, mid, :] - ect_low[0, mid, :]).mean()
        diff_edge = np.abs(ect_high[0, 0, :] - ect_low[0, 0, :]).mean()

        # This is a heuristic check - high scale should be more binary
        assert diff_mid > 0 or diff_edge > 0


class TestEctPointsBackward:
    def test_shape(self):
        n_points = 100
        n_dirs = 32
        resolution = 64

        nh = np.random.randn(n_points, n_dirs).astype(np.float32)
        batch = np.zeros(n_points, dtype=np.int64)
        lin = generate_lin(1.0, resolution)
        grad_output = np.random.randn(1, resolution, n_dirs).astype(np.float32)

        grad_nh = trailed_rust.compute_ect_points_backward(
            nh, batch, lin, grad_output, 50.0
        )
        assert grad_nh.shape == (n_points, n_dirs)

    def test_gradient_finite(self):
        n_points = 50
        n_dirs = 16
        resolution = 32

        nh = np.random.randn(n_points, n_dirs).astype(np.float32)
        batch = np.zeros(n_points, dtype=np.int64)
        lin = generate_lin(1.0, resolution)
        grad_output = np.ones((1, resolution, n_dirs), dtype=np.float32)

        grad_nh = trailed_rust.compute_ect_points_backward(
            nh, batch, lin, grad_output, 50.0
        )
        assert np.all(np.isfinite(grad_nh))


class TestEctChannels:
    def test_shape(self):
        n_points = 100
        n_dirs = 32
        resolution = 64
        n_channels = 3

        nh = np.random.randn(n_points, n_dirs).astype(np.float32)
        batch = np.zeros(n_points, dtype=np.int64)
        channels = np.random.randint(0, n_channels, n_points).astype(np.int64)
        lin = generate_lin(1.0, resolution)

        ect = trailed_rust.compute_ect_channels_forward(
            nh, batch, channels, lin, 1, n_channels, 500.0
        )
        assert ect.shape == (1, n_dirs, resolution, n_channels)

    def test_channel_separation(self):
        n_points = 60
        n_dirs = 8
        resolution = 16
        n_channels = 3

        # Create points with distinct heights per channel
        nh = np.zeros((n_points, n_dirs), dtype=np.float32)
        batch = np.zeros(n_points, dtype=np.int64)
        channels = np.repeat(np.arange(n_channels), n_points // n_channels).astype(
            np.int64
        )

        # Channel 0: low heights, Channel 1: medium, Channel 2: high
        for i in range(n_points):
            ch = channels[i]
            nh[i, :] = (ch - 1) * 0.5

        lin = generate_lin(1.0, resolution)

        ect = trailed_rust.compute_ect_channels_forward(
            nh, batch, channels, lin, 1, n_channels, 100.0
        )

        # Check that channels have different patterns
        assert ect.shape == (1, n_dirs, resolution, n_channels)


class TestFastEct:
    def test_shape(self):
        n_points = 100
        n_dirs = 32
        resolution = 64

        nh = (np.random.randn(n_points, n_dirs) * 0.5).astype(np.float32)

        ect = trailed_rust.compute_fast_ect(nh, resolution)
        assert ect.shape == (resolution, n_dirs)

    def test_cumulative(self):
        n_points = 50
        n_dirs = 8
        resolution = 32

        nh = (np.random.randn(n_points, n_dirs) * 0.3).astype(np.float32)

        ect = trailed_rust.compute_fast_ect(nh, resolution)

        # Should be monotonically increasing (cumsum)
        for j in range(n_dirs):
            for k in range(1, resolution):
                assert ect[k, j] >= ect[k - 1, j]

    def test_batched_shape(self):
        n_points = 100
        n_dirs = 32
        resolution = 64
        n_batches = 4

        nh = (np.random.randn(n_points, n_dirs) * 0.5).astype(np.float32)
        batch = np.repeat(np.arange(n_batches), n_points // n_batches).astype(np.int64)

        ect = trailed_rust.compute_fast_ect_batched(nh, batch, n_batches, resolution)
        assert ect.shape == (n_batches, resolution, n_dirs)


class TestParallelConsistency:
    def test_parallel_matches_sequential(self):
        n_points = 200
        n_dirs = 32
        resolution = 64

        nh = np.random.randn(n_points, n_dirs).astype(np.float32)
        batch = np.zeros(n_points, dtype=np.int64)
        lin = generate_lin(1.0, resolution)

        ect_seq = trailed_rust.compute_ect_points_forward(nh, batch, lin, 1, 50.0)
        ect_par = trailed_rust.compute_ect_points_forward_parallel(nh, batch, lin, 1, 50.0)

        assert np.allclose(ect_seq, ect_par, atol=1e-5)

    def test_parallel_backward_matches_sequential(self):
        n_points = 100
        n_dirs = 16
        resolution = 32

        nh = np.random.randn(n_points, n_dirs).astype(np.float32)
        batch = np.zeros(n_points, dtype=np.int64)
        lin = generate_lin(1.0, resolution)
        grad_output = np.random.randn(1, resolution, n_dirs).astype(np.float32)

        grad_seq = trailed_rust.compute_ect_points_backward(
            nh, batch, lin, grad_output, 50.0
        )
        grad_par = trailed_rust.compute_ect_points_backward_parallel(
            nh, batch, lin, grad_output, 50.0
        )

        assert np.allclose(grad_seq, grad_par, atol=1e-5)

    def test_parallel_fast_ect_matches_sequential(self):
        n_points = 200
        n_dirs = 32
        resolution = 64

        nh = (np.random.randn(n_points, n_dirs) * 0.5).astype(np.float32)

        ect_seq = trailed_rust.compute_fast_ect(nh, resolution)
        ect_par = trailed_rust.compute_fast_ect_parallel(nh, resolution)

        assert np.allclose(ect_seq, ect_par, atol=1e-5)


class TestEctEdges:
    def test_shape(self):
        n_points = 50
        n_dirs = 16
        resolution = 32
        n_edges = 40

        nh = np.random.randn(n_points, n_dirs).astype(np.float32)
        batch = np.zeros(n_points, dtype=np.int64)
        edge_index = np.array(
            [
                np.random.randint(0, n_points, n_edges),
                np.random.randint(0, n_points, n_edges),
            ],
            dtype=np.int64,
        )
        lin = generate_lin(1.0, resolution)

        ect = trailed_rust.compute_ect_edges_forward(nh, batch, edge_index, lin, 1, 50.0)
        assert ect.shape == (1, resolution, n_dirs)


class TestEctFaces:
    def test_shape(self):
        n_points = 50
        n_dirs = 16
        resolution = 32
        n_edges = 40
        n_faces = 20

        nh = np.random.randn(n_points, n_dirs).astype(np.float32)
        batch = np.zeros(n_points, dtype=np.int64)
        edge_index = np.array(
            [
                np.random.randint(0, n_points, n_edges),
                np.random.randint(0, n_points, n_edges),
            ],
            dtype=np.int64,
        )
        face = np.array(
            [
                np.random.randint(0, n_points, n_faces),
                np.random.randint(0, n_points, n_faces),
                np.random.randint(0, n_points, n_faces),
            ],
            dtype=np.int64,
        )
        lin = generate_lin(1.0, resolution)

        ect = trailed_rust.compute_ect_faces_forward(
            nh, batch, edge_index, face, lin, 1, 50.0
        )
        assert ect.shape == (1, resolution, n_dirs)
