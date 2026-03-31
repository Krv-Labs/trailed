"""
Plugin integrations for TRAILED.

This submodule provides optional integrations with scikit-learn.

For PyTorch use cases, use the upstream aidos-lab/dect package:
    pip install dect @ git+https://github.com/aidos-lab/DECT/
"""

# ruff: noqa: F401

__all__ = []

# Conditionally export sklearn classes
try:
    from .sklearn import (
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
