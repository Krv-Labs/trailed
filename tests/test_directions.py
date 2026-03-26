"""Tests for direction generation functions."""

import numpy as np
import pytest

from dect.sampling import (
    generate_uniform_directions,
    generate_2d_directions,
    generate_multiview_directions,
    generate_spherical_grid_directions,
    generate_directions,
    normalize_directions,
    compute_node_heights,
    generate_lin,
)


class TestUniformDirections:
    def test_shape(self):
        v = generate_uniform_directions(64, 3)
        assert v.shape == (3, 64)

    def test_unit_norm(self):
        v = generate_uniform_directions(100, 3)
        norms = np.linalg.norm(v, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_different_dimensions(self):
        for d in [2, 3, 5, 10]:
            v = generate_uniform_directions(32, d)
            assert v.shape == (d, 32)
            norms = np.linalg.norm(v, axis=0)
            assert np.allclose(norms, 1.0, atol=1e-5)

    def test_reproducibility(self):
        v1 = generate_uniform_directions(64, 3, seed=42)
        v2 = generate_uniform_directions(64, 3, seed=42)
        assert np.allclose(v1, v2)

    def test_different_seeds(self):
        v1 = generate_uniform_directions(64, 3, seed=42)
        v2 = generate_uniform_directions(64, 3, seed=123)
        assert not np.allclose(v1, v2)


class Test2DDirections:
    def test_shape(self):
        v = generate_2d_directions(64)
        assert v.shape == (2, 64)

    def test_unit_norm(self):
        v = generate_2d_directions(100)
        norms = np.linalg.norm(v, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_coverage(self):
        v = generate_2d_directions(8)
        # Should cover the unit circle
        angles = np.arctan2(v[0], v[1])
        # Check that we have directions in different quadrants
        assert np.any(angles > 0)
        assert np.any(angles < 0)
        # Check that angles span a reasonable range
        angle_range = angles.max() - angles.min()
        assert angle_range > np.pi  # At least half the circle


class TestMultiviewDirections:
    def test_shape(self):
        v = generate_multiview_directions(64, 3)
        assert v.shape == (3, 64)

    def test_different_dimensions(self):
        for d in [3, 4, 5]:
            v = generate_multiview_directions(64, d)
            assert v.shape == (d, 64)


class TestSphericalGridDirections:
    def test_shape(self):
        v = generate_spherical_grid_directions(8, 16)
        assert v.shape == (3, 128)

    def test_unit_norm(self):
        v = generate_spherical_grid_directions(8, 16)
        norms = np.linalg.norm(v, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestGenerateDirections:
    def test_uniform_method(self):
        v = generate_directions(64, 3, method="uniform")
        assert v.shape == (3, 64)

    def test_structured_2d_method(self):
        v = generate_directions(64, 2, method="structured_2d")
        assert v.shape == (2, 64)

    def test_structured_2d_wrong_dim(self):
        with pytest.raises(ValueError, match="structured_2d requires"):
            generate_directions(64, 3, method="structured_2d")

    def test_multiview_method(self):
        v = generate_directions(64, 3, method="multiview")
        assert v.shape == (3, 64)

    def test_spherical_grid_method(self):
        v = generate_directions(64, 3, method="spherical_grid")
        assert v.shape[0] == 3

    def test_spherical_grid_wrong_dim(self):
        with pytest.raises(ValueError, match="spherical_grid requires"):
            generate_directions(64, 2, method="spherical_grid")

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown"):
            generate_directions(64, 3, method="unknown")


class TestNormalizeDirections:
    def test_normalize(self):
        v = np.random.randn(3, 10)
        v_norm = normalize_directions(v)
        norms = np.linalg.norm(v_norm, axis=0)
        assert np.allclose(norms, 1.0)

    def test_already_normalized(self):
        v = generate_uniform_directions(32, 3)
        v_norm = normalize_directions(v)
        assert np.allclose(v, v_norm)


class TestComputeNodeHeights:
    def test_shape(self):
        x = np.random.randn(100, 3).astype(np.float32)
        v = generate_uniform_directions(32, 3)
        nh = compute_node_heights(x, v)
        assert nh.shape == (100, 32)

    def test_values(self):
        x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        v = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        nh = compute_node_heights(x, v)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert np.allclose(nh, expected)


class TestGenerateLin:
    def test_shape(self):
        lin = generate_lin(1.0, 64)
        assert lin.shape == (64,)

    def test_range(self):
        lin = generate_lin(1.0, 64)
        assert np.isclose(lin[0], -1.0)
        assert np.isclose(lin[-1], 1.0)

    def test_different_radius(self):
        lin = generate_lin(2.5, 32)
        assert np.isclose(lin[0], -2.5)
        assert np.isclose(lin[-1], 2.5)
