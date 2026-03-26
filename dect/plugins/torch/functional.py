"""
Functional interface for ECT computation in PyTorch.

This module provides a stateless functional API for computing ECT
without creating layer objects.
"""

from typing import Optional

import torch
from torch import Tensor

from .autograd import EctPointsFunction


def compute_ect(
    x: Tensor,
    v: Tensor,
    radius: float = 1.0,
    resolution: int = 64,
    scale: float = 500.0,
    batch: Optional[Tensor] = None,
    normalized: bool = False,
) -> Tensor:
    """Functional interface for computing ECT.
    
    Args:
        x: Point coordinates [num_points, ambient_dim]
        v: Direction vectors [ambient_dim, num_thetas]
        radius: Radius of threshold interval.
        resolution: Number of threshold steps.
        scale: Scale factor for sigmoid.
        batch: Batch indices [num_points].
        normalized: Whether to normalize output.
    
    Returns:
        ECT tensor of shape [batch_size, resolution, num_thetas]
    """
    nh = (x @ v).contiguous()
    lin = torch.linspace(-radius, radius, resolution, device=x.device).contiguous()
    
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    batch = batch.contiguous()
    
    dim_size = batch.max().item() + 1
    
    ect = EctPointsFunction.apply(nh, batch, lin, dim_size, scale)
    
    if normalized:
        ect = ect / torch.amax(ect, dim=(1, 2), keepdim=True).clamp(min=1e-8)
    
    return ect
