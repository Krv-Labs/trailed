"""
Configuration and direction generation for PyTorch ECT layers.

This module provides the EctConfig dataclass and direction generation
utilities used by ECT layers.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dect.sampling import generate_directions as _generate_directions_func


@dataclass
class EctConfig:
    """Configuration for ECT computation.

    Attributes:
        num_thetas: Number of directions to sample.
        resolution: Number of threshold steps (bump_steps).
        radius: Radius of the threshold interval [-radius, radius].
        scale: Scale factor for sigmoid approximation (higher = sharper).
        ambient_dim: Dimension of the input point clouds.
        ect_type: Type of ECT computation ("points", "points_derivative",
                  "edges", "faces").
        sampling_method: Method for generating directions ("uniform",
                        "structured_2d", "multiview", "spherical_grid").
        normalized: Whether to normalize the ECT output to [0, 1].
        seed: Random seed for direction generation.
        device: Device for computation ("cpu" or "cuda").
    """

    num_thetas: int = 32
    resolution: int = 32
    radius: float = 1.1
    scale: float = 500.0
    ambient_dim: int = 3
    ect_type: Literal["points", "points_derivative", "edges", "faces"] = "points"
    sampling_method: Literal[
        "uniform", "structured_2d", "multiview", "spherical_grid"
    ] = "uniform"
    normalized: bool = False
    seed: int = 42
    device: str = "cpu"


def generate_directions(
    num_thetas: int,
    ambient_dim: int,
    method: str = "uniform",
    seed: int = 42,
) -> np.ndarray:
    """Generate direction vectors using the Rust backend.

    Args:
        num_thetas: Number of directions to generate.
        ambient_dim: Dimension of the ambient space.
        method: Sampling method ("uniform", "structured_2d", "multiview",
                "spherical_grid").
        seed: Random seed for reproducibility.

    Returns:
        Direction vectors of shape [ambient_dim, num_thetas].
    """
    return _generate_directions_func(num_thetas, ambient_dim, method, seed)
