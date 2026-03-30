"""
Fast (non-differentiable) ECT transformer using histogram binning.

This module provides a speed-optimized transformer for ECT computation
that uses bincount-based histograms instead of sigmoid smoothing.
"""

from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

import trailed_rust
from trailed.sampling import generate_directions as _generate_directions_func


class FastEctTransformer:
    """Fast (non-differentiable) ECT transformer using bincount.

    This transformer is optimized for speed using histogram-based ECT
    computation. It's faster than EctTransformer but produces slightly
    different results (discrete vs smooth approximation).

    Parameters
    ----------
    num_thetas : int, default=64
        Number of directions to sample.
    resolution : int, default=64
        Number of histogram bins.
    sampling_method : str, default="uniform"
        Method for generating directions.
    flatten : bool, default=True
        If True, flatten the ECT to a 1D feature vector.
    parallel : bool, default=True
        If True, use parallel computation.
    seed : int, default=42
        Random seed for direction generation.

    Examples
    --------
    >>> from trailed.plugins.sklearn import FastEctTransformer
    >>> import numpy as np
    >>> X = np.random.randn(100, 50, 3).astype(np.float32)
    >>> transformer = FastEctTransformer(num_thetas=64, resolution=64)
    >>> features = transformer.fit_transform(X)
    """

    def __init__(
        self,
        num_thetas: int = 64,
        resolution: int = 64,
        sampling_method: Literal[
            "uniform", "structured_2d", "multiview", "spherical_grid"
        ] = "uniform",
        flatten: bool = True,
        parallel: bool = True,
        seed: int = 42,
    ):
        self.num_thetas = num_thetas
        self.resolution = resolution
        self.sampling_method = sampling_method
        self.flatten = flatten
        self.parallel = parallel
        self.seed = seed

        self.directions_: Optional[NDArray] = None
        self.ambient_dim_: Optional[int] = None

    def _generate_directions(self, ambient_dim: int) -> NDArray:
        """Generate direction vectors."""
        return _generate_directions_func(
            self.num_thetas, ambient_dim, self.sampling_method, self.seed
        )

    def fit(self, X: ArrayLike, y=None) -> "FastEctTransformer":
        """Fit the transformer."""
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_samples, n_points, n_dims), got {X.ndim}D"
            )

        self.ambient_dim_ = X.shape[2]
        self.directions_ = self._generate_directions(self.ambient_dim_)

        return self

    def transform(self, X: ArrayLike) -> NDArray:
        """Transform point clouds to ECT features."""
        if self.directions_ is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_samples, n_points, n_dims), got {X.ndim}D"
            )

        n_samples = X.shape[0]

        results = []

        for i in range(n_samples):
            points = X[i]
            nh = points @ self.directions_

            if self.parallel:
                ect = trailed_rust.compute_fast_ect_parallel(nh, self.resolution)
            else:
                ect = trailed_rust.compute_fast_ect(nh, self.resolution)

            results.append(ect)

        ects = np.stack(results, axis=0)

        if self.flatten:
            return ects.reshape(n_samples, -1)

        return ects

    def fit_transform(self, X: ArrayLike, y=None) -> NDArray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "num_thetas": self.num_thetas,
            "resolution": self.resolution,
            "sampling_method": self.sampling_method,
            "flatten": self.flatten,
            "parallel": self.parallel,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "FastEctTransformer":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)

        self.directions_ = None
        self.ambient_dim_ = None

        return self
