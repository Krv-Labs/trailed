"""
Direction generation utilities for ECT computation.

This module provides Python wrappers for the Rust direction generation
functions, with additional convenience features.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

import trailed_rust


def generate_uniform_directions(
    num_thetas: int,
    ambient_dim: int,
    seed: int = 42,
) -> NDArray:
    """Generate randomly sampled directions from a sphere.

    Samples points from a standard normal distribution and projects them
    onto the unit sphere to obtain uniformly distributed directions.

    Parameters
    ----------
    num_thetas : int
        Number of directions to generate.
    ambient_dim : int
        Dimension of the ambient space.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    directions : ndarray of shape (ambient_dim, num_thetas)
        Unit vectors representing directions.

    Examples
    --------
    >>> from dect.sampling import generate_uniform_directions
    >>> v = generate_uniform_directions(64, 3)
    >>> v.shape
    (3, 64)
    >>> np.allclose(np.linalg.norm(v, axis=0), 1.0)
    True
    """
    return trailed_rust.generate_uniform_directions(num_thetas, ambient_dim, seed)


def generate_2d_directions(num_thetas: int) -> NDArray:
    """Generate structured directions along the 2D unit circle.

    Divides the interval [0, 2*pi) into equal parts and returns the
    corresponding points on the unit circle.

    Parameters
    ----------
    num_thetas : int
        Number of directions to generate.

    Returns
    -------
    directions : ndarray of shape (2, num_thetas)
        Unit vectors representing directions.

    Examples
    --------
    >>> from dect.sampling import generate_2d_directions
    >>> v = generate_2d_directions(8)
    >>> v.shape
    (2, 8)
    """
    return trailed_rust.generate_2d_directions(num_thetas)


def generate_multiview_directions(
    num_thetas: int,
    ambient_dim: int,
) -> NDArray:
    """Generate structured directions in multiple 2D planes.

    Generates directions by embedding the 2D unit circle in the d-dimensional
    space along each pair of coordinate axes. This produces (d choose 2)
    sets of structured directions.

    Parameters
    ----------
    num_thetas : int
        Total number of directions to generate.
    ambient_dim : int
        Dimension of the ambient space.

    Returns
    -------
    directions : ndarray of shape (ambient_dim, num_thetas)
        Unit vectors representing directions.

    Examples
    --------
    >>> from dect.sampling import generate_multiview_directions
    >>> v = generate_multiview_directions(64, 3)
    >>> v.shape
    (3, 64)
    """
    return trailed_rust.generate_multiview_directions(num_thetas, ambient_dim)


def generate_spherical_grid_directions(
    num_thetas: int,
    num_phis: int,
) -> NDArray:
    """Generate directions on a spherical grid (3D only).

    Creates a grid of directions on the unit sphere using latitude-longitude
    parameterization. The polar angle theta ranges from 0 to pi, and the
    azimuthal angle phi ranges from 0 to 2*pi.

    Parameters
    ----------
    num_thetas : int
        Number of polar angle samples.
    num_phis : int
        Number of azimuthal angle samples.

    Returns
    -------
    directions : ndarray of shape (3, num_thetas * num_phis)
        Unit vectors representing directions.

    Examples
    --------
    >>> from dect.sampling import generate_spherical_grid_directions
    >>> v = generate_spherical_grid_directions(8, 16)
    >>> v.shape
    (3, 128)
    """
    return trailed_rust.generate_spherical_grid_directions(num_thetas, num_phis)


def generate_directions(
    num_thetas: int,
    ambient_dim: int,
    method: Literal[
        "uniform", "structured_2d", "multiview", "spherical_grid"
    ] = "uniform",
    seed: int = 42,
) -> NDArray:
    """Generate direction vectors using the specified method.

    This is a convenience function that dispatches to the appropriate
    direction generation function based on the method parameter.

    Parameters
    ----------
    num_thetas : int
        Number of directions to generate. For spherical_grid, this is
        used to estimate the grid dimensions.
    ambient_dim : int
        Dimension of the ambient space.
    method : str, default="uniform"
        Direction sampling method:
        - "uniform": Random sampling on unit sphere
        - "structured_2d": Evenly spaced on 2D circle (requires ambient_dim=2)
        - "multiview": Structured sampling in coordinate planes
        - "spherical_grid": Lat/lon grid on sphere (requires ambient_dim=3)
    seed : int, default=42
        Random seed (only used for "uniform" method).

    Returns
    -------
    directions : ndarray of shape (ambient_dim, num_directions)
        Unit vectors representing directions.

    Examples
    --------
    >>> from dect.sampling import generate_directions
    >>> v = generate_directions(64, 3, method="uniform")
    >>> v.shape
    (3, 64)
    >>> v = generate_directions(64, 2, method="structured_2d")
    >>> v.shape
    (2, 64)
    """
    if method == "uniform":
        return generate_uniform_directions(num_thetas, ambient_dim, seed)
    elif method == "structured_2d":
        if ambient_dim != 2:
            raise ValueError("structured_2d requires ambient_dim=2")
        return generate_2d_directions(num_thetas)
    elif method == "multiview":
        return generate_multiview_directions(num_thetas, ambient_dim)
    elif method == "spherical_grid":
        if ambient_dim != 3:
            raise ValueError("spherical_grid requires ambient_dim=3")
        num_phis = int(np.sqrt(num_thetas * 2))
        num_t = max(1, num_thetas // num_phis)
        return generate_spherical_grid_directions(num_t, num_phis)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def normalize_directions(v: NDArray) -> NDArray:
    """Normalize direction vectors to unit length.

    Parameters
    ----------
    v : ndarray of shape (d, n)
        Direction vectors.

    Returns
    -------
    normalized : ndarray of shape (d, n)
        Unit vectors.
    """
    norms = np.linalg.norm(v, axis=0, keepdims=True)
    return v / np.maximum(norms, 1e-10)


def compute_node_heights(
    x: NDArray,
    v: NDArray,
) -> NDArray:
    """Compute node heights (projections onto directions).

    Parameters
    ----------
    x : ndarray of shape (n_points, d)
        Point coordinates.
    v : ndarray of shape (d, n_directions)
        Direction vectors.

    Returns
    -------
    heights : ndarray of shape (n_points, n_directions)
        Projection of each point onto each direction.
    """
    return trailed_rust.compute_node_heights(
        x.astype(np.float32),
        v.astype(np.float32),
    )


def generate_lin(radius: float, resolution: int) -> NDArray:
    """Generate linear threshold values.

    Parameters
    ----------
    radius : float
        Radius of the interval [-radius, radius].
    resolution : int
        Number of threshold steps.

    Returns
    -------
    lin : ndarray of shape (resolution,)
        Threshold values.
    """
    return trailed_rust.generate_lin(radius, resolution)
