"""
Scikit-learn compatible transformer for ECT computation.

This module provides sklearn-compatible transformers that can be used
in sklearn pipelines for feature extraction from point clouds.
"""

from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

import dect_rust


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
    >>> from dect.sklearn import EctTransformer
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
        sampling_method: Literal["uniform", "structured_2d", "multiview", "spherical_grid"] = "uniform",
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
        if self.sampling_method == "uniform":
            return dect_rust.generate_uniform_directions(
                self.num_thetas, ambient_dim, self.seed
            )
        elif self.sampling_method == "structured_2d":
            if ambient_dim != 2:
                raise ValueError("structured_2d requires 2D point clouds")
            return dect_rust.generate_2d_directions(self.num_thetas)
        elif self.sampling_method == "multiview":
            return dect_rust.generate_multiview_directions(
                self.num_thetas, ambient_dim
            )
        elif self.sampling_method == "spherical_grid":
            if ambient_dim != 3:
                raise ValueError("spherical_grid requires 3D point clouds")
            num_phis = int(np.sqrt(self.num_thetas * 2))
            num_t = self.num_thetas // num_phis
            return dect_rust.generate_spherical_grid_directions(num_t, num_phis)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
    
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
        self._lin = dect_rust.generate_lin(self.radius, self.resolution)
        
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
            raise ValueError(
                f"Expected {self.ambient_dim_}D points, got {X.shape[2]}D"
            )
        
        n_samples = X.shape[0]
        n_points = X.shape[1]
        
        results = []
        
        for i in range(n_samples):
            points = X[i]
            nh = points @ self.directions_
            batch = np.zeros(n_points, dtype=np.int64)
            
            if self.parallel:
                ect = dect_rust.compute_ect_points_forward_parallel(
                    nh, batch, self._lin, 1, self.scale
                )
            else:
                ect = dect_rust.compute_ect_points_forward(
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
    >>> from dect.sklearn import FastEctTransformer
    >>> import numpy as np
    >>> X = np.random.randn(100, 50, 3).astype(np.float32)
    >>> transformer = FastEctTransformer(num_thetas=64, resolution=64)
    >>> features = transformer.fit_transform(X)
    """
    
    def __init__(
        self,
        num_thetas: int = 64,
        resolution: int = 64,
        sampling_method: Literal["uniform", "structured_2d", "multiview", "spherical_grid"] = "uniform",
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
        if self.sampling_method == "uniform":
            return dect_rust.generate_uniform_directions(
                self.num_thetas, ambient_dim, self.seed
            )
        elif self.sampling_method == "structured_2d":
            if ambient_dim != 2:
                raise ValueError("structured_2d requires 2D point clouds")
            return dect_rust.generate_2d_directions(self.num_thetas)
        elif self.sampling_method == "multiview":
            return dect_rust.generate_multiview_directions(
                self.num_thetas, ambient_dim
            )
        elif self.sampling_method == "spherical_grid":
            if ambient_dim != 3:
                raise ValueError("spherical_grid requires 3D point clouds")
            num_phis = int(np.sqrt(self.num_thetas * 2))
            num_t = self.num_thetas // num_phis
            return dect_rust.generate_spherical_grid_directions(num_t, num_phis)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
    
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
        n_points = X.shape[1]
        
        results = []
        
        for i in range(n_samples):
            points = X[i]
            nh = points @ self.directions_
            
            if self.parallel:
                ect = dect_rust.compute_fast_ect_parallel(nh, self.resolution)
            else:
                ect = dect_rust.compute_fast_ect(nh, self.resolution)
            
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
    >>> from dect.sklearn import EctChannelTransformer
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
        sampling_method: Literal["uniform", "structured_2d", "multiview", "spherical_grid"] = "uniform",
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
        if self.sampling_method == "uniform":
            return dect_rust.generate_uniform_directions(
                self.num_thetas, ambient_dim, self.seed
            )
        elif self.sampling_method == "structured_2d":
            if ambient_dim != 2:
                raise ValueError("structured_2d requires 2D point clouds")
            return dect_rust.generate_2d_directions(self.num_thetas)
        elif self.sampling_method == "multiview":
            return dect_rust.generate_multiview_directions(
                self.num_thetas, ambient_dim
            )
        elif self.sampling_method == "spherical_grid":
            if ambient_dim != 3:
                raise ValueError("spherical_grid requires 3D point clouds")
            num_phis = int(np.sqrt(self.num_thetas * 2))
            num_t = self.num_thetas // num_phis
            return dect_rust.generate_spherical_grid_directions(num_t, num_phis)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
    
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
