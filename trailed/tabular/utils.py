"""
Utility functions for DataFrame ECT operations.

This module provides helper functions for working with ECT results
and generic DataFrame dispatching.
"""

from typing import List, Optional, Union

from numpy.typing import NDArray

from .pandas import HAS_PANDAS, pd, compute_ect_from_pandas
from .polars import HAS_POLARS, pl, compute_ect_from_polars


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
