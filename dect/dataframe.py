"""
DataFrame integration for ECT computation.

This module provides functions to compute ECT directly from pandas and
polars DataFrames containing point cloud data.
"""

from typing import Literal, Optional, Union, List

import numpy as np
from numpy.typing import NDArray

import dect_rust

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None


def _generate_directions(
    num_thetas: int,
    ambient_dim: int,
    method: str = "uniform",
    seed: int = 42,
) -> NDArray:
    """Generate direction vectors."""
    if method == "uniform":
        return dect_rust.generate_uniform_directions(num_thetas, ambient_dim, seed)
    elif method == "structured_2d":
        if ambient_dim != 2:
            raise ValueError("structured_2d requires ambient_dim=2")
        return dect_rust.generate_2d_directions(num_thetas)
    elif method == "multiview":
        return dect_rust.generate_multiview_directions(num_thetas, ambient_dim)
    elif method == "spherical_grid":
        if ambient_dim != 3:
            raise ValueError("spherical_grid requires ambient_dim=3")
        num_phis = int(np.sqrt(num_thetas * 2))
        num_t = num_thetas // num_phis
        return dect_rust.generate_spherical_grid_directions(num_t, num_phis)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def compute_ect_from_numpy(
    points: NDArray,
    group_ids: Optional[NDArray] = None,
    channel_ids: Optional[NDArray] = None,
    num_thetas: int = 64,
    resolution: int = 64,
    radius: float = 1.0,
    scale: float = 500.0,
    sampling_method: str = "uniform",
    seed: int = 42,
    normalized: bool = False,
    parallel: bool = True,
) -> NDArray:
    """Compute ECT from numpy arrays.
    
    Parameters
    ----------
    points : ndarray of shape (n_points, n_dims)
        Point coordinates.
    group_ids : ndarray of shape (n_points,), optional
        Group/batch indices for each point. Points with the same group_id
        belong to the same point cloud.
    channel_ids : ndarray of shape (n_points,), optional
        Channel indices for each point (e.g., atom types).
    num_thetas : int, default=64
        Number of directions.
    resolution : int, default=64
        Number of threshold steps.
    radius : float, default=1.0
        Radius of threshold interval.
    scale : float, default=500.0
        Scale factor for sigmoid.
    sampling_method : str, default="uniform"
        Method for generating directions.
    seed : int, default=42
        Random seed.
    normalized : bool, default=False
        Whether to normalize the ECT.
    parallel : bool, default=True
        Whether to use parallel computation.
    
    Returns
    -------
    ect : ndarray
        ECT features. Shape depends on inputs:
        - No groups, no channels: (resolution, num_thetas)
        - With groups, no channels: (n_groups, resolution, num_thetas)
        - With channels: (n_groups, num_thetas, resolution, n_channels)
    """
    points = np.asarray(points, dtype=np.float32)
    
    if points.ndim != 2:
        raise ValueError(f"Expected 2D points array, got {points.ndim}D")
    
    n_points, ambient_dim = points.shape
    
    # Generate directions
    directions = _generate_directions(num_thetas, ambient_dim, sampling_method, seed)
    
    # Compute node heights
    nh = points @ directions
    
    # Generate linear thresholds
    lin = dect_rust.generate_lin(radius, resolution)
    
    # Handle groups
    if group_ids is None:
        batch = np.zeros(n_points, dtype=np.int64)
        dim_size = 1
    else:
        batch = np.asarray(group_ids, dtype=np.int64)
        dim_size = int(batch.max()) + 1
    
    # Compute ECT
    if channel_ids is not None:
        channels = np.asarray(channel_ids, dtype=np.int64)
        max_channels = int(channels.max()) + 1
        
        if parallel:
            ect = dect_rust.compute_ect_channels_forward_parallel(
                nh, batch, channels, lin, dim_size, max_channels, scale
            )
        else:
            ect = dect_rust.compute_ect_channels_forward(
                nh, batch, channels, lin, dim_size, max_channels, scale
            )
    else:
        if parallel:
            ect = dect_rust.compute_ect_points_forward_parallel(
                nh, batch, lin, dim_size, scale
            )
        else:
            ect = dect_rust.compute_ect_points_forward(
                nh, batch, lin, dim_size, scale
            )
    
    # Normalize if requested
    if normalized:
        if channel_ids is not None:
            ect = ect / (np.max(ect, axis=(-1, -2), keepdims=True) + 1e-8)
        else:
            ect = ect / (np.max(ect, axis=(1, 2), keepdims=True) + 1e-8)
    
    # Remove batch dimension if single group
    if group_ids is None:
        ect = ect[0]
    
    return ect


def compute_ect_from_pandas(
    df: "pd.DataFrame",
    coord_columns: List[str],
    group_column: Optional[str] = None,
    channel_column: Optional[str] = None,
    num_thetas: int = 64,
    resolution: int = 64,
    radius: float = 1.0,
    scale: float = 500.0,
    sampling_method: str = "uniform",
    seed: int = 42,
    normalized: bool = False,
    parallel: bool = True,
) -> NDArray:
    """Compute ECT from a pandas DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing point cloud data.
    coord_columns : list of str
        Column names containing point coordinates (e.g., ["x", "y", "z"]).
    group_column : str, optional
        Column name for group/batch IDs.
    channel_column : str, optional
        Column name for channel IDs.
    num_thetas : int, default=64
        Number of directions.
    resolution : int, default=64
        Number of threshold steps.
    radius : float, default=1.0
        Radius of threshold interval.
    scale : float, default=500.0
        Scale factor for sigmoid.
    sampling_method : str, default="uniform"
        Method for generating directions.
    seed : int, default=42
        Random seed.
    normalized : bool, default=False
        Whether to normalize the ECT.
    parallel : bool, default=True
        Whether to use parallel computation.
    
    Returns
    -------
    ect : ndarray
        ECT features.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from dect.dataframe import compute_ect_from_pandas
    >>> df = pd.DataFrame({
    ...     "x": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7],
    ...     "y": [0.1, 0.3, 0.2, 0.4, 0.5, 0.6],
    ...     "z": [0.0, 0.1, 0.2, 0.1, 0.2, 0.3],
    ...     "molecule_id": [0, 0, 0, 1, 1, 1],
    ...     "atom_type": [0, 1, 0, 1, 1, 0],
    ... })
    >>> ect = compute_ect_from_pandas(
    ...     df,
    ...     coord_columns=["x", "y", "z"],
    ...     group_column="molecule_id",
    ...     channel_column="atom_type",
    ... )
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for this function. Install with: pip install pandas")
    
    # Extract coordinates
    points = df[coord_columns].values.astype(np.float32)
    
    # Extract group IDs
    group_ids = None
    if group_column is not None:
        group_ids = df[group_column].values
        # Convert to contiguous integer indices
        unique_groups = np.unique(group_ids)
        group_map = {g: i for i, g in enumerate(unique_groups)}
        group_ids = np.array([group_map[g] for g in group_ids], dtype=np.int64)
    
    # Extract channel IDs
    channel_ids = None
    if channel_column is not None:
        channel_ids = df[channel_column].values
        # Convert to contiguous integer indices
        unique_channels = np.unique(channel_ids)
        channel_map = {c: i for i, c in enumerate(unique_channels)}
        channel_ids = np.array([channel_map[c] for c in channel_ids], dtype=np.int64)
    
    return compute_ect_from_numpy(
        points=points,
        group_ids=group_ids,
        channel_ids=channel_ids,
        num_thetas=num_thetas,
        resolution=resolution,
        radius=radius,
        scale=scale,
        sampling_method=sampling_method,
        seed=seed,
        normalized=normalized,
        parallel=parallel,
    )


def compute_ect_from_polars(
    df: "pl.DataFrame",
    coord_columns: List[str],
    group_column: Optional[str] = None,
    channel_column: Optional[str] = None,
    num_thetas: int = 64,
    resolution: int = 64,
    radius: float = 1.0,
    scale: float = 500.0,
    sampling_method: str = "uniform",
    seed: int = 42,
    normalized: bool = False,
    parallel: bool = True,
) -> NDArray:
    """Compute ECT from a polars DataFrame.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing point cloud data.
    coord_columns : list of str
        Column names containing point coordinates.
    group_column : str, optional
        Column name for group/batch IDs.
    channel_column : str, optional
        Column name for channel IDs.
    num_thetas : int, default=64
        Number of directions.
    resolution : int, default=64
        Number of threshold steps.
    radius : float, default=1.0
        Radius of threshold interval.
    scale : float, default=500.0
        Scale factor for sigmoid.
    sampling_method : str, default="uniform"
        Method for generating directions.
    seed : int, default=42
        Random seed.
    normalized : bool, default=False
        Whether to normalize the ECT.
    parallel : bool, default=True
        Whether to use parallel computation.
    
    Returns
    -------
    ect : ndarray
        ECT features.
    
    Examples
    --------
    >>> import polars as pl
    >>> from dect.dataframe import compute_ect_from_polars
    >>> df = pl.DataFrame({
    ...     "x": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7],
    ...     "y": [0.1, 0.3, 0.2, 0.4, 0.5, 0.6],
    ...     "z": [0.0, 0.1, 0.2, 0.1, 0.2, 0.3],
    ...     "molecule_id": [0, 0, 0, 1, 1, 1],
    ... })
    >>> ect = compute_ect_from_polars(
    ...     df,
    ...     coord_columns=["x", "y", "z"],
    ...     group_column="molecule_id",
    ... )
    """
    if not HAS_POLARS:
        raise ImportError("polars is required for this function. Install with: pip install polars")
    
    # Extract coordinates
    points = df.select(coord_columns).to_numpy().astype(np.float32)
    
    # Extract group IDs
    group_ids = None
    if group_column is not None:
        group_series = df.get_column(group_column)
        unique_groups = group_series.unique().sort()
        group_map = {g: i for i, g in enumerate(unique_groups.to_list())}
        group_ids = np.array(
            [group_map[g] for g in group_series.to_list()],
            dtype=np.int64
        )
    
    # Extract channel IDs
    channel_ids = None
    if channel_column is not None:
        channel_series = df.get_column(channel_column)
        unique_channels = channel_series.unique().sort()
        channel_map = {c: i for i, c in enumerate(unique_channels.to_list())}
        channel_ids = np.array(
            [channel_map[c] for c in channel_series.to_list()],
            dtype=np.int64
        )
    
    return compute_ect_from_numpy(
        points=points,
        group_ids=group_ids,
        channel_ids=channel_ids,
        num_thetas=num_thetas,
        resolution=resolution,
        radius=radius,
        scale=scale,
        sampling_method=sampling_method,
        seed=seed,
        normalized=normalized,
        parallel=parallel,
    )


def compute_ect_from_dataframe(
    df: Union["pd.DataFrame", "pl.DataFrame"],
    coord_columns: List[str],
    group_column: Optional[str] = None,
    channel_column: Optional[str] = None,
    **kwargs,
) -> NDArray:
    """Compute ECT from a pandas or polars DataFrame.
    
    This is a convenience function that automatically detects the
    DataFrame type and calls the appropriate function.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame containing point cloud data.
    coord_columns : list of str
        Column names containing point coordinates.
    group_column : str, optional
        Column name for group/batch IDs.
    channel_column : str, optional
        Column name for channel IDs.
    **kwargs
        Additional arguments passed to compute_ect_from_numpy.
    
    Returns
    -------
    ect : ndarray
        ECT features.
    """
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        return compute_ect_from_pandas(
            df, coord_columns, group_column, channel_column, **kwargs
        )
    elif HAS_POLARS and isinstance(df, pl.DataFrame):
        return compute_ect_from_polars(
            df, coord_columns, group_column, channel_column, **kwargs
        )
    else:
        raise TypeError(
            f"Unsupported DataFrame type: {type(df)}. "
            "Expected pandas.DataFrame or polars.DataFrame."
        )


def ect_to_dataframe(
    ect: NDArray,
    group_ids: Optional[List] = None,
    as_polars: bool = False,
) -> Union["pd.DataFrame", "pl.DataFrame"]:
    """Convert ECT array to a DataFrame.
    
    Parameters
    ----------
    ect : ndarray
        ECT array of shape (n_groups, resolution, num_thetas) or
        (n_groups, num_thetas, resolution, n_channels).
    group_ids : list, optional
        Original group identifiers to use as index.
    as_polars : bool, default=False
        If True, return a polars DataFrame instead of pandas.
    
    Returns
    -------
    df : pd.DataFrame or pl.DataFrame
        DataFrame with flattened ECT features.
    """
    if ect.ndim == 2:
        # Single sample: (resolution, num_thetas)
        ect = ect.reshape(1, -1)
    elif ect.ndim == 3:
        # (n_groups, resolution, num_thetas)
        n_groups = ect.shape[0]
        ect = ect.reshape(n_groups, -1)
    elif ect.ndim == 4:
        # (n_groups, num_thetas, resolution, n_channels)
        n_groups = ect.shape[0]
        ect = ect.reshape(n_groups, -1)
    else:
        raise ValueError(f"Unexpected ECT shape: {ect.shape}")
    
    n_features = ect.shape[1]
    columns = [f"ect_{i}" for i in range(n_features)]
    
    if as_polars:
        if not HAS_POLARS:
            raise ImportError("polars is required. Install with: pip install polars")
        
        df = pl.DataFrame({col: ect[:, i] for i, col in enumerate(columns)})
        
        if group_ids is not None:
            df = df.with_columns(pl.Series("group_id", group_ids))
        
        return df
    else:
        if not HAS_PANDAS:
            raise ImportError("pandas is required. Install with: pip install pandas")
        
        df = pd.DataFrame(ect, columns=columns)
        
        if group_ids is not None:
            df.index = group_ids
            df.index.name = "group_id"
        
        return df


class DataFrameEctTransformer:
    """DataFrame-native ECT transformer.
    
    This class provides a consistent interface for computing ECT from
    DataFrames, supporting both pandas and polars.
    
    Parameters
    ----------
    coord_columns : list of str
        Column names containing point coordinates.
    group_column : str, optional
        Column name for group/batch IDs.
    channel_column : str, optional
        Column name for channel IDs.
    num_thetas : int, default=64
        Number of directions.
    resolution : int, default=64
        Number of threshold steps.
    radius : float, default=1.0
        Radius of threshold interval.
    scale : float, default=500.0
        Scale factor for sigmoid.
    sampling_method : str, default="uniform"
        Method for generating directions.
    seed : int, default=42
        Random seed.
    normalized : bool, default=False
        Whether to normalize the ECT.
    parallel : bool, default=True
        Whether to use parallel computation.
    output_format : str, default="numpy"
        Output format: "numpy", "pandas", or "polars".
    
    Examples
    --------
    >>> import pandas as pd
    >>> from dect.dataframe import DataFrameEctTransformer
    >>> df = pd.DataFrame({
    ...     "x": np.random.randn(100),
    ...     "y": np.random.randn(100),
    ...     "z": np.random.randn(100),
    ...     "group": np.repeat(range(10), 10),
    ... })
    >>> transformer = DataFrameEctTransformer(
    ...     coord_columns=["x", "y", "z"],
    ...     group_column="group",
    ...     num_thetas=32,
    ...     resolution=32,
    ... )
    >>> ect = transformer.transform(df)
    """
    
    def __init__(
        self,
        coord_columns: List[str],
        group_column: Optional[str] = None,
        channel_column: Optional[str] = None,
        num_thetas: int = 64,
        resolution: int = 64,
        radius: float = 1.0,
        scale: float = 500.0,
        sampling_method: str = "uniform",
        seed: int = 42,
        normalized: bool = False,
        parallel: bool = True,
        output_format: Literal["numpy", "pandas", "polars"] = "numpy",
    ):
        self.coord_columns = coord_columns
        self.group_column = group_column
        self.channel_column = channel_column
        self.num_thetas = num_thetas
        self.resolution = resolution
        self.radius = radius
        self.scale = scale
        self.sampling_method = sampling_method
        self.seed = seed
        self.normalized = normalized
        self.parallel = parallel
        self.output_format = output_format
        
        self.directions_: Optional[NDArray] = None
        self._lin: Optional[NDArray] = None
    
    def fit(self, df: Union["pd.DataFrame", "pl.DataFrame"]) -> "DataFrameEctTransformer":
        """Fit the transformer by generating directions.
        
        Parameters
        ----------
        df : DataFrame
            Sample DataFrame to infer dimensions from.
        
        Returns
        -------
        self
        """
        ambient_dim = len(self.coord_columns)
        self.directions_ = _generate_directions(
            self.num_thetas, ambient_dim, self.sampling_method, self.seed
        )
        self._lin = dect_rust.generate_lin(self.radius, self.resolution)
        return self
    
    def transform(
        self,
        df: Union["pd.DataFrame", "pl.DataFrame"],
    ) -> Union[NDArray, "pd.DataFrame", "pl.DataFrame"]:
        """Transform DataFrame to ECT features.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame containing point cloud data.
        
        Returns
        -------
        result : ndarray or DataFrame
            ECT features in the specified output format.
        """
        if self.directions_ is None:
            self.fit(df)
        
        ect = compute_ect_from_dataframe(
            df,
            coord_columns=self.coord_columns,
            group_column=self.group_column,
            channel_column=self.channel_column,
            num_thetas=self.num_thetas,
            resolution=self.resolution,
            radius=self.radius,
            scale=self.scale,
            sampling_method=self.sampling_method,
            seed=self.seed,
            normalized=self.normalized,
            parallel=self.parallel,
        )
        
        if self.output_format == "numpy":
            return ect
        elif self.output_format == "pandas":
            return ect_to_dataframe(ect, as_polars=False)
        elif self.output_format == "polars":
            return ect_to_dataframe(ect, as_polars=True)
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")
    
    def fit_transform(
        self,
        df: Union["pd.DataFrame", "pl.DataFrame"],
    ) -> Union[NDArray, "pd.DataFrame", "pl.DataFrame"]:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
