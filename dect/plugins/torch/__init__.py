"""
PyTorch layers for computing the Euler Characteristic Transform.

This subpackage provides nn.Module implementations that wrap the Rust ECT
computation backend, supporting both fixed and learnable directions.
"""

from .config import EctConfig, generate_directions
from .autograd import (
    EctPointsFunction,
    EctPointsDerivativeFunction,
    EctEdgesFunction,
    EctFacesFunction,
    EctChannelsFunction,
)
from .layers import EctLayer, FastEctLayer
from .functional import compute_ect

__all__ = [
    # Configuration
    "EctConfig",
    "generate_directions",
    # Autograd functions
    "EctPointsFunction",
    "EctPointsDerivativeFunction",
    "EctEdgesFunction",
    "EctFacesFunction",
    "EctChannelsFunction",
    # Layers
    "EctLayer",
    "FastEctLayer",
    # Functional API
    "compute_ect",
]
