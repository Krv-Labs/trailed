"""
Core ECT computation from numpy arrays.

This module provides the fundamental numpy-based ECT computation
that other tabular integrations build upon.
"""

from typing import Optional

import numpy as np
import trailed_rust
from numpy.typing import NDArray

from dect.sampling import generate_directions as _generate_directions


def compute_ect_from_numpy(
    points: NDArray,
    group_ids: Optional[NDArray] = None,
    channel_ids: Optional[NDArray] = None,
    num_thetas: int = 64,
    resolution: int = 64,
    radius: float = 1.0,
    scale: float = 500.0,
    sampling_method: str = "uniform",
    seed: int = 42,
    normalized: bool = False,
    parallel: bool = True,
    directions: Optional[NDArray] = None,
    lin: Optional[NDArray] = None,
) -> NDArray:
    """Compute ECT from numpy arrays.

    Parameters
    ----------
    points : ndarray of shape (n_points, n_dims)
        Point coordinates.
    group_ids : ndarray of shape (n_points,), optional
        Group/batch indices for each point. Points with the same group_id
        belong to the same point cloud.
    channel_ids : ndarray of shape (n_points,), optional
        Channel indices for each point (e.g., atom types).
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

    Returns
    -------
    ect : ndarray
        ECT features. Shape depends on inputs:
        - No groups, no channels: (resolution, num_thetas)
        - With groups, no channels: (n_groups, resolution, num_thetas)
        - With channels: (n_groups, num_thetas, resolution, n_channels)
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2:
        raise ValueError(f"Expected 2D points array, got {points.ndim}D")

    n_points, ambient_dim = points.shape

    # Generate or reuse directions
    if directions is None:
        directions = _generate_directions(
            num_thetas, ambient_dim, sampling_method, seed
        )
    else:
        directions = np.asarray(directions, dtype=np.float32)
        if directions.shape[0] != ambient_dim:
            raise ValueError(
                f"Expected directions with first dim {ambient_dim}, got {directions.shape[0]}"
            )

    # Compute node heights
    nh = points @ directions

    # Generate or reuse linear thresholds
    if lin is None:
        lin = trailed_rust.generate_lin(radius, resolution)
    else:
        lin = np.asarray(lin, dtype=np.float32)

    # Handle groups
    if group_ids is None:
        batch = np.zeros(n_points, dtype=np.int64)
        dim_size = 1
    else:
        batch = np.asarray(group_ids, dtype=np.int64)
        dim_size = int(batch.max()) + 1

    # Compute ECT
    if channel_ids is not None:
        channels = np.asarray(channel_ids, dtype=np.int64)
        max_channels = int(channels.max()) + 1

        if parallel:
            ect = trailed_rust.compute_ect_channels_forward_parallel(
                nh, batch, channels, lin, dim_size, max_channels, scale
            )
        else:
            ect = trailed_rust.compute_ect_channels_forward(
                nh, batch, channels, lin, dim_size, max_channels, scale
            )
    else:
        if parallel:
            ect = trailed_rust.compute_ect_points_forward_parallel(
                nh, batch, lin, dim_size, scale
            )
        else:
            ect = trailed_rust.compute_ect_points_forward(
                nh, batch, lin, dim_size, scale
            )

    # Normalize if requested
    if normalized:
        if channel_ids is not None:
            ect = ect / (np.max(ect, axis=(-1, -2), keepdims=True) + 1e-8)
        else:
            ect = ect / (np.max(ect, axis=(1, 2), keepdims=True) + 1e-8)

    # Remove batch dimension if single group
    if group_ids is None:
        ect = ect[0]

    return ect
