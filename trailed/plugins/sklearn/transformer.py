"""
Sklearn-compatible transformer for ECT computation.

This module provides the standard EctTransformer class for computing
differentiable ECT features from point clouds.
"""

from typing import Literal, Optional

import numpy as np
import trailed_rust
from numpy.typing import ArrayLike, NDArray

from trailed.sampling import generate_directions as _generate_directions_func


class EctTransformer:
    """Sklearn-compatible transformer for computing ECT features.

    This transformer computes the Euler Characteristic Transform for
    batches of point clouds, producing fixed-size feature vectors
    suitable for machine learning classifiers.

    Parameters
    ----------
    num_thetas : int, default=64
        Number of directions to sample.
    resolution : int, default=64
        Number of threshold steps.
    radius : float, default=1.0
        Radius of the threshold interval [-radius, radius].
    scale : float, default=500.0
        Scale factor for sigmoid approximation.
    sampling_method : str, default="uniform"
        Method for generating directions. One of "uniform",
        "structured_2d", "multiview", "spherical_grid".
    flatten : bool, default=True
        If True, flatten the ECT to a 1D feature vector.
    normalized : bool, default=False
        If True, normalize each ECT to [0, 1].
    parallel : bool, default=True
        If True, use parallel computation.
    seed : int, default=42
        Random seed for direction generation.

    Attributes
    ----------
    directions_ : ndarray of shape (ambient_dim, num_thetas)
        The direction vectors used for ECT computation.
    ambient_dim_ : int
        Inferred ambient dimension from training data.

    Examples
    --------
    >>> from trailed.plugins.sklearn import EctTransformer
    >>> import numpy as np
    >>> # Create sample point clouds: 10 samples, 50 points each, 3D
    >>> X = np.random.randn(10, 50, 3).astype(np.float32)
    >>> transformer = EctTransformer(num_thetas=32, resolution=32)
    >>> features = transformer.fit_transform(X)
    >>> features.shape
    (10, 1024)  # 32 * 32 = 1024 features per sample
    """

    def __init__(
        self,
        num_thetas: int = 64,
        resolution: int = 64,
        radius: float = 1.0,
        scale: float = 500.0,
        sampling_method: Literal[
            "uniform", "structured_2d", "multiview", "spherical_grid"
        ] = "uniform",
        flatten: bool = True,
        normalized: bool = False,
        parallel: bool = True,
        seed: int = 42,
    ):
        self.num_thetas = num_thetas
        self.resolution = resolution
        self.radius = radius
        self.scale = scale
        self.sampling_method = sampling_method
        self.flatten = flatten
        self.normalized = normalized
        self.parallel = parallel
        self.seed = seed

        self.directions_: Optional[NDArray] = None
        self.ambient_dim_: Optional[int] = None
        self._lin: Optional[NDArray] = None

    def _generate_directions(self, ambient_dim: int) -> NDArray:
        """Generate direction vectors."""
        return _generate_directions_func(
            self.num_thetas, ambient_dim, self.sampling_method, self.seed
        )

    def fit(self, X: ArrayLike, y=None) -> "EctTransformer":
        """Fit the transformer by generating directions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points, n_dims)
            Training point clouds.
        y : None
            Ignored.

        Returns
        -------
        self
            The fitted transformer.
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_samples, n_points, n_dims), got {X.ndim}D"
            )

        self.ambient_dim_ = X.shape[2]
        self.directions_ = self._generate_directions(self.ambient_dim_)
        self._lin = trailed_rust.generate_lin(self.radius, self.resolution)

        return self

    def transform(self, X: ArrayLike) -> NDArray:
        """Transform point clouds to ECT features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points, n_dims)
            Point clouds to transform.

        Returns
        -------
        features : ndarray
            ECT features. Shape is (n_samples, resolution * num_thetas) if
            flatten=True, else (n_samples, resolution, num_thetas).
        """
        if self.directions_ is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_samples, n_points, n_dims), got {X.ndim}D"
            )

        if X.shape[2] != self.ambient_dim_:
            raise ValueError(f"Expected {self.ambient_dim_}D points, got {X.shape[2]}D")

        n_samples = X.shape[0]
        n_points = X.shape[1]

        if self.parallel:
            # Batched computation in Rust across all samples.
            ects = trailed_rust.compute_ect_batch_parallel(
                X,
                self.directions_,
                self.radius,
                self.resolution,
                self.scale,
            )

            if self.normalized:
                # Normalize each sample independently.
                max_per_sample = np.max(ects, axis=(1, 2), keepdims=True)
                ects = ects / (max_per_sample + 1e-8)
        else:
            # Non-parallel path: preserve original per-sample behavior.
            results = []

            for i in range(n_samples):
                points = X[i]
                nh = points @ self.directions_
                batch = np.zeros(n_points, dtype=np.int64)

                ect = trailed_rust.compute_ect_points_forward(
                    nh, batch, self._lin, 1, self.scale
                )

                ect = ect[0]  # Remove batch dimension

                if self.normalized:
                    ect = ect / (np.max(ect) + 1e-8)

                results.append(ect)

            ects = np.stack(results, axis=0)

        if self.flatten:
            return ects.reshape(n_samples, -1)

        return ects

    def fit_transform(self, X: ArrayLike, y=None) -> NDArray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points, n_dims)
            Point clouds to fit and transform.
        y : None
            Ignored.

        Returns
        -------
        features : ndarray
            ECT features.
        """
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "num_thetas": self.num_thetas,
            "resolution": self.resolution,
            "radius": self.radius,
            "scale": self.scale,
            "sampling_method": self.sampling_method,
            "flatten": self.flatten,
            "normalized": self.normalized,
            "parallel": self.parallel,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "EctTransformer":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)

        # Reset fitted state if parameters changed
        self.directions_ = None
        self.ambient_dim_ = None
        self._lin = None

        return self
