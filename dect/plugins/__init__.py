"""
Plugin integrations for DECT.

This submodule provides optional integrations with popular ML frameworks:
- sklearn: Scikit-learn compatible transformers
- torch: PyTorch layers and autograd functions
"""

__all__ = []

# Conditionally export sklearn classes
try:
    from .sklearn import (
        EctTransformer,
        FastEctTransformer,
        EctChannelTransformer,
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

# Conditionally export torch classes
try:
    from .torch import (
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

    __all__.extend(
        [
            "EctConfig",
            "EctLayer",
            "FastEctLayer",
            "EctPointsFunction",
            "EctPointsDerivativeFunction",
            "EctEdgesFunction",
            "EctFacesFunction",
            "EctChannelsFunction",
            "compute_ect",
            "generate_directions",
        ]
    )
except ImportError:
    pass
