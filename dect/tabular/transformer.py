"""
DataFrame-native ECT transformer.

This module provides a transformer class for computing ECT from DataFrames
with a consistent interface supporting both pandas and polars.
"""

from typing import List, Literal, Optional, Union

import trailed_rust
from numpy.typing import NDArray

from dect.sampling import generate_directions as _generate_directions

from .utils import compute_ect_from_dataframe, ect_to_dataframe

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


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
    >>> from dect.tabular import DataFrameEctTransformer
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

    def fit(
        self, df: Union["pd.DataFrame", "pl.DataFrame"]
    ) -> "DataFrameEctTransformer":
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
        self._lin = trailed_rust.generate_lin(self.radius, self.resolution)
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
            directions=self.directions_,
            lin=self._lin,
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
