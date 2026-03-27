"""
DECT - Differentiable Euler Characteristic Transform

A fast Rust-based implementation of the ECT with Python bindings
for point clouds, graphs, and meshes.
"""

# ruff: noqa: F401

# Import from submodules for backward compatibility
from .sampling import (
    compute_node_heights,
    generate_2d_directions,
    generate_directions,
    generate_lin,
    generate_multiview_directions,
    generate_spherical_grid_directions,
    generate_uniform_directions,
    normalize_directions,
)

__all__ = [
    # Sampling/directions
    "generate_directions",
    "generate_uniform_directions",
    "generate_2d_directions",
    "generate_multiview_directions",
    "generate_spherical_grid_directions",
    "normalize_directions",
    "compute_node_heights",
    "generate_lin",
]

# Conditionally import PyTorch components
try:
    from .plugins.torch import (
        EctChannelsFunction,
        EctConfig,
        EctEdgesFunction,
        EctFacesFunction,
        EctLayer,
        EctPointsDerivativeFunction,
        EctPointsFunction,
        FastEctLayer,
        compute_ect,
    )

    __all__.extend(
        [
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
        ]
    )
except ImportError:
    pass

# Conditionally import sklearn components
try:
    from .plugins.sklearn import (
        EctChannelTransformer,
        EctTransformer,
        FastEctTransformer,
    )

    __all__.extend(
        [
            "EctTransformer",
            "FastEctTransformer",
            "EctChannelTransformer",
        ]
    )
except ImportError:
    pass

# Conditionally import DataFrame components
try:
    from .tabular import (
        DataFrameEctTransformer,
        compute_ect_from_dataframe,
        compute_ect_from_numpy,
        compute_ect_from_pandas,
        compute_ect_from_polars,
        ect_to_dataframe,
    )

    __all__.extend(
        [
            "compute_ect_from_numpy",
            "compute_ect_from_pandas",
            "compute_ect_from_polars",
            "compute_ect_from_dataframe",
            "ect_to_dataframe",
            "DataFrameEctTransformer",
        ]
    )
except ImportError:
    pass
