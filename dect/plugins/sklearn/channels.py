"""
ECT transformer with channel support for categorical features.

This module provides a transformer that computes separate ECTs for each
categorical channel in the point cloud.
"""

from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

import dect_rust
from dect.sampling import generate_directions as _generate_directions_func


class EctChannelTransformer:
    """ECT transformer with channel support for categorical features.

    This transformer computes separate ECTs for each categorical channel
    in the point cloud, useful for molecules with different atom types
    or other categorically-labeled point clouds.

    Parameters
    ----------
    num_thetas : int, default=64
        Number of directions to sample.
    resolution : int, default=64
        Number of threshold steps.
    radius : float, default=1.0
        Radius of the threshold interval.
    scale : float, default=500.0
        Scale factor for sigmoid approximation.
    max_channels : int or None, default=None
        Maximum number of channels. If None, inferred from data.
    sampling_method : str, default="uniform"
        Method for generating directions.
    flatten : bool, default=True
        If True, flatten the ECT to a 1D feature vector.
    normalized : bool, default=False
        If True, normalize each ECT to [0, 1].
    seed : int, default=42
        Random seed for direction generation.

    Examples
    --------
    >>> from dect.plugins.sklearn import EctChannelTransformer
    >>> import numpy as np
    >>> # Point clouds with channel labels
    >>> X = np.random.randn(10, 50, 3).astype(np.float32)
    >>> channels = np.random.randint(0, 3, size=(10, 50))  # 3 channels
    >>> transformer = EctChannelTransformer(max_channels=3)
    >>> features = transformer.fit_transform(X, channels=channels)
    """

    def __init__(
        self,
        num_thetas: int = 64,
        resolution: int = 64,
        radius: float = 1.0,
        scale: float = 500.0,
        max_channels: Optional[int] = None,
        sampling_method: Literal[
            "uniform", "structured_2d", "multiview", "spherical_grid"
        ] = "uniform",
        flatten: bool = True,
        normalized: bool = False,
        seed: int = 42,
    ):
        self.num_thetas = num_thetas
        self.resolution = resolution
        self.radius = radius
        self.scale = scale
        self.max_channels = max_channels
        self.sampling_method = sampling_method
        self.flatten = flatten
        self.normalized = normalized
        self.seed = seed

        self.directions_: Optional[NDArray] = None
        self.ambient_dim_: Optional[int] = None
        self._lin: Optional[NDArray] = None
        self.n_channels_: Optional[int] = None

    def _generate_directions(self, ambient_dim: int) -> NDArray:
        """Generate direction vectors."""
        return _generate_directions_func(
            self.num_thetas, ambient_dim, self.sampling_method, self.seed
        )

    def fit(
        self,
        X: ArrayLike,
        y=None,
        channels: Optional[ArrayLike] = None,
    ) -> "EctChannelTransformer":
        """Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points, n_dims)
            Training point clouds.
        y : None
            Ignored.
        channels : array-like of shape (n_samples, n_points), optional
            Channel indices for each point.
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_samples, n_points, n_dims), got {X.ndim}D"
            )

        self.ambient_dim_ = X.shape[2]
        self.directions_ = self._generate_directions(self.ambient_dim_)
        self._lin = dect_rust.generate_lin(self.radius, self.resolution)

        if self.max_channels is not None:
            self.n_channels_ = self.max_channels
        elif channels is not None:
            channels = np.asarray(channels, dtype=np.int64)
            self.n_channels_ = int(np.max(channels)) + 1
        else:
            self.n_channels_ = 1

        return self

    def transform(
        self,
        X: ArrayLike,
        channels: Optional[ArrayLike] = None,
    ) -> NDArray:
        """Transform point clouds to ECT features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points, n_dims)
            Point clouds to transform.
        channels : array-like of shape (n_samples, n_points)
            Channel indices for each point.

        Returns
        -------
        features : ndarray
            ECT features with shape depending on flatten parameter.
        """
        if self.directions_ is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_samples, n_points, n_dims), got {X.ndim}D"
            )

        n_samples = X.shape[0]
        n_points = X.shape[1]

        if channels is None:
            channels = np.zeros((n_samples, n_points), dtype=np.int64)
        else:
            channels = np.asarray(channels, dtype=np.int64)

        results = []

        for i in range(n_samples):
            points = X[i]
            ch = channels[i]
            nh = points @ self.directions_
            batch = np.zeros(n_points, dtype=np.int64)

            ect = dect_rust.compute_ect_channels_forward(
                nh, batch, ch, self._lin, 1, self.n_channels_, self.scale
            )

            ect = ect[0]  # Remove batch dimension

            if self.normalized:
                ect = ect / (np.max(ect) + 1e-8)

            results.append(ect)

        ects = np.stack(results, axis=0)

        if self.flatten:
            return ects.reshape(n_samples, -1)

        return ects

    def fit_transform(
        self,
        X: ArrayLike,
        y=None,
        channels: Optional[ArrayLike] = None,
    ) -> NDArray:
        """Fit and transform in one step."""
        return self.fit(X, y, channels=channels).transform(X, channels=channels)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "num_thetas": self.num_thetas,
            "resolution": self.resolution,
            "radius": self.radius,
            "scale": self.scale,
            "max_channels": self.max_channels,
            "sampling_method": self.sampling_method,
            "flatten": self.flatten,
            "normalized": self.normalized,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "EctChannelTransformer":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)

        self.directions_ = None
        self.ambient_dim_ = None
        self._lin = None
        self.n_channels_ = None

        return self
