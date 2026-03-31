"""
TRAILED - Topological Regularization and Integrity Learning for EHR Data

A fast Rust-backed implementation of the Euler Characteristic Transform (ECT)
for sklearn pipelines and tabular/DataFrame workflows, focused on EHR data.

For PyTorch neural network use cases, use the upstream aidos-lab/dect package:
    pip install dect @ git+https://github.com/aidos-lab/DECT/
"""

# ruff: noqa: F401

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
