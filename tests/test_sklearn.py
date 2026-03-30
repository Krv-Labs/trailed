"""Tests for sklearn-compatible transformers."""

import numpy as np
import pytest


class TestEctTransformer:
    @pytest.fixture
    def transformer(self):
        from trailed.plugins.sklearn import EctTransformer

        return EctTransformer(
            num_thetas=16,
            resolution=16,
            radius=1.0,
            scale=100.0,
            flatten=True,
        )

    def test_fit(self, transformer):
        X = np.random.randn(10, 50, 3).astype(np.float32)
        transformer.fit(X)

        assert transformer.directions_ is not None
        assert transformer.directions_.shape == (3, 16)
        assert transformer.ambient_dim_ == 3

    def test_transform(self, transformer):
        X = np.random.randn(10, 50, 3).astype(np.float32)
        transformer.fit(X)
        features = transformer.transform(X)

        assert features.shape == (10, 16 * 16)

    def test_fit_transform(self, transformer):
        X = np.random.randn(10, 50, 3).astype(np.float32)
        features = transformer.fit_transform(X)

        assert features.shape == (10, 16 * 16)

    def test_unflatten(self):
        from trailed.plugins.sklearn import EctTransformer

        transformer = EctTransformer(
            num_thetas=16,
            resolution=16,
            flatten=False,
        )

        X = np.random.randn(10, 50, 3).astype(np.float32)
        features = transformer.fit_transform(X)

        assert features.shape == (10, 16, 16)

    def test_normalized(self):
        from trailed.plugins.sklearn import EctTransformer

        transformer = EctTransformer(
            num_thetas=16,
            resolution=16,
            normalized=True,
            flatten=False,
        )

        X = np.random.randn(10, 50, 3).astype(np.float32)
        features = transformer.fit_transform(X)

        assert np.all(features <= 1.0 + 1e-5)
        assert np.all(features >= 0.0 - 1e-5)

    def test_different_dimensions(self):
        from trailed.plugins.sklearn import EctTransformer

        for d in [2, 3, 5]:
            transformer = EctTransformer(num_thetas=16, resolution=16)
            X = np.random.randn(5, 30, d).astype(np.float32)
            features = transformer.fit_transform(X)

            assert features.shape == (5, 16 * 16)

    def test_transform_without_fit(self, transformer):
        X = np.random.randn(10, 50, 3).astype(np.float32)

        with pytest.raises(RuntimeError, match="not fitted"):
            transformer.transform(X)

    def test_wrong_input_dim(self, transformer):
        X = np.random.randn(10, 50).astype(np.float32)  # 2D instead of 3D

        with pytest.raises(ValueError, match="3D array"):
            transformer.fit(X)

    def test_get_params(self, transformer):
        params = transformer.get_params()

        assert "num_thetas" in params
        assert "resolution" in params
        assert params["num_thetas"] == 16

    def test_set_params(self, transformer):
        transformer.set_params(num_thetas=32, resolution=64)

        assert transformer.num_thetas == 32
        assert transformer.resolution == 64
        assert transformer.directions_ is None  # Reset fitted state


class TestFastEctTransformer:
    @pytest.fixture
    def transformer(self):
        from trailed.plugins.sklearn import FastEctTransformer

        return FastEctTransformer(
            num_thetas=32,
            resolution=32,
            flatten=True,
        )

    def test_fit_transform(self, transformer):
        X = np.random.randn(10, 50, 3).astype(np.float32)
        features = transformer.fit_transform(X)

        assert features.shape == (10, 32 * 32)

    def test_unflatten(self):
        from trailed.plugins.sklearn import FastEctTransformer

        transformer = FastEctTransformer(
            num_thetas=32,
            resolution=32,
            flatten=False,
        )

        X = np.random.randn(10, 50, 3).astype(np.float32)
        features = transformer.fit_transform(X)

        assert features.shape == (10, 32, 32)


class TestEctChannelTransformer:
    @pytest.fixture
    def transformer(self):
        from trailed.plugins.sklearn import EctChannelTransformer

        return EctChannelTransformer(
            num_thetas=16,
            resolution=16,
            max_channels=3,
            flatten=True,
        )

    def test_fit_transform(self, transformer):
        X = np.random.randn(10, 50, 3).astype(np.float32)
        channels = np.random.randint(0, 3, size=(10, 50))

        features = transformer.fit_transform(X, channels=channels)

        # 16 thetas * 16 resolution * 3 channels = 768
        assert features.shape == (10, 16 * 16 * 3)

    def test_unflatten(self):
        from trailed.plugins.sklearn import EctChannelTransformer

        transformer = EctChannelTransformer(
            num_thetas=16,
            resolution=16,
            max_channels=3,
            flatten=False,
        )

        X = np.random.randn(10, 50, 3).astype(np.float32)
        channels = np.random.randint(0, 3, size=(10, 50))

        features = transformer.fit_transform(X, channels=channels)

        assert features.shape == (10, 16, 16, 3)

    def test_infer_channels(self):
        from trailed.plugins.sklearn import EctChannelTransformer

        transformer = EctChannelTransformer(
            num_thetas=16,
            resolution=16,
            max_channels=None,  # Infer from data
            flatten=False,
        )

        X = np.random.randn(10, 50, 3).astype(np.float32)
        channels = np.random.randint(0, 5, size=(10, 50))  # 5 channels

        features = transformer.fit_transform(X, channels=channels)

        assert transformer.n_channels_ == 5
        assert features.shape == (10, 16, 16, 5)
