"""
DECT - Differentiable Euler Characteristic Transform

A fast Rust-based implementation of the ECT with Python bindings
for point clouds, graphs, and meshes.
"""

from .torch_layer import (
    EctConfig,
    EctLayer,
    FastEctLayer,
    EctPointsFunction,
    EctPointsDerivativeFunction,
    EctEdgesFunction,
    EctFacesFunction,
    EctChannelsFunction,
    compute_ect,
    generate_directions,
)

__all__ = [
    # Configuration
    "EctConfig",
    # PyTorch layers
    "EctLayer",
    "FastEctLayer",
    # Autograd functions
    "EctPointsFunction",
    "EctPointsDerivativeFunction",
    "EctEdgesFunction",
    "EctFacesFunction",
    "EctChannelsFunction",
    # Functional interface
    "compute_ect",
    "generate_directions",
]

try:
    from .sklearn import (
        EctTransformer,
        FastEctTransformer,
        EctChannelTransformer,
    )
    __all__.extend([
        "EctTransformer",
        "FastEctTransformer",
        "EctChannelTransformer",
    ])
except ImportError:
    pass

try:
    from .dataframe import (
        compute_ect_from_numpy,
        compute_ect_from_pandas,
        compute_ect_from_polars,
        compute_ect_from_dataframe,
        ect_to_dataframe,
        DataFrameEctTransformer,
    )
    __all__.extend([
        "compute_ect_from_numpy",
        "compute_ect_from_pandas",
        "compute_ect_from_polars",
        "compute_ect_from_dataframe",
        "ect_to_dataframe",
        "DataFrameEctTransformer",
    ])
except ImportError:
    pass
