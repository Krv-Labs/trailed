"""
Polars DataFrame integration for ECT computation.

This module provides functions to compute ECT directly from polars DataFrames.
"""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .core import compute_ect_from_numpy

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None


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
    >>> from dect.tabular import compute_ect_from_polars
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
