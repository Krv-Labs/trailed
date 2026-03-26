"""
Tabular data integration for ECT computation.

This subpackage provides DataFrame-based interfaces for computing ECT
from pandas and polars DataFrames.
"""

from .core import compute_ect_from_numpy
from .pandas import compute_ect_from_pandas
from .polars import compute_ect_from_polars
from .utils import compute_ect_from_dataframe, ect_to_dataframe
from .transformer import DataFrameEctTransformer

__all__ = [
    "compute_ect_from_numpy",
    "compute_ect_from_pandas",
    "compute_ect_from_polars",
    "compute_ect_from_dataframe",
    "ect_to_dataframe",
    "DataFrameEctTransformer",
]
