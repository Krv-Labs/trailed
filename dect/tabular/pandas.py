"""
Pandas DataFrame integration for ECT computation.

This module provides functions to compute ECT directly from pandas DataFrames.
"""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .core import compute_ect_from_numpy

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


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
    directions: Optional[NDArray] = None,
    lin: Optional[NDArray] = None,
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
    >>> from dect.tabular import compute_ect_from_pandas
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
        raise ImportError(
            "pandas is required for this function. Install with: pip install pandas"
        )

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
        directions=directions,
        lin=lin,
    )
