"""
PyTorch nn.Module layers for ECT computation.

This module provides neural network layers that wrap the Rust ECT backend
for integration into PyTorch models.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import trailed_rust

from .config import EctConfig, generate_directions
from .autograd import (
    EctPointsFunction,
    EctPointsDerivativeFunction,
    EctEdgesFunction,
    EctFacesFunction,
    EctChannelsFunction,
)


class EctLayer(nn.Module):
    """Neural network layer for computing the Euler Characteristic Transform.

    This layer computes the ECT of input point clouds or graphs using a
    Rust backend for performance. Supports both fixed and learnable directions.

    Args:
        config: EctConfig instance with computation parameters.
        directions: Optional tensor of directions [ambient_dim, num_thetas].
                   If None, directions are generated using config.sampling_method.
        learnable: If True, directions become learnable parameters.

    Example:
        >>> config = EctConfig(num_thetas=64, resolution=64, ambient_dim=3)
        >>> layer = EctLayer(config)
        >>> # For torch_geometric data:
        >>> ect = layer(data)
        >>> # For raw tensors:
        >>> ect = layer.forward_raw(x, batch)
    """

    def __init__(
        self,
        config: EctConfig,
        directions: Optional[Tensor] = None,
        learnable: bool = False,
    ):
        super().__init__()
        self.config = config
        self.learnable = learnable

        # Generate linear threshold values
        lin = torch.linspace(-config.radius, config.radius, config.resolution)
        self.register_buffer("lin", lin)

        # Initialize directions
        if directions is not None:
            v = directions
        else:
            v_np = generate_directions(
                config.num_thetas,
                config.ambient_dim,
                config.sampling_method,
                config.seed,
            )
            v = torch.from_numpy(v_np).float()

        if learnable:
            self.v = nn.Parameter(v)
        else:
            self.register_buffer("v", v)

    def _normalize_directions(self) -> Tensor:
        """Normalize direction vectors to unit length."""
        v = self.v
        norms = v.pow(2).sum(dim=0, keepdim=True).sqrt().clamp(min=1e-8)
        return v / norms

    def forward(self, data) -> Tensor:
        """Compute ECT from torch_geometric Data object.

        Args:
            data: torch_geometric Data object with attributes:
                  - x: Node coordinates [num_nodes, ambient_dim]
                  - batch: Batch indices [num_nodes]
                  - edge_index: Edge indices [2, num_edges] (for edges/faces)
                  - face: Face indices [3, num_faces] (for faces)

        Returns:
            ECT tensor of shape [batch_size, resolution, num_thetas]
        """
        v = self._normalize_directions()
        nh = (data.x @ v).contiguous()
        batch = data.batch.contiguous()
        lin = self.lin.contiguous()
        dim_size = data.batch.max().item() + 1
        scale = self.config.scale

        if self.config.ect_type == "points":
            ect = EctPointsFunction.apply(nh, batch, lin, dim_size, scale)
        elif self.config.ect_type == "points_derivative":
            ect = EctPointsDerivativeFunction.apply(nh, batch, lin, dim_size, scale)
        elif self.config.ect_type == "edges":
            edge_index = data.edge_index.contiguous()
            ect = EctEdgesFunction.apply(nh, batch, edge_index, lin, dim_size, scale)
        elif self.config.ect_type == "faces":
            edge_index = data.edge_index.contiguous()
            face = data.face.contiguous()
            ect = EctFacesFunction.apply(
                nh, batch, edge_index, face, lin, dim_size, scale
            )
        else:
            raise ValueError(f"Unknown ect_type: {self.config.ect_type}")

        if self.config.normalized:
            ect = ect / torch.amax(ect, dim=(1, 2), keepdim=True).clamp(min=1e-8)

        return ect

    def forward_raw(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        channels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute ECT from raw tensors.

        Args:
            x: Point coordinates [num_points, ambient_dim]
            batch: Batch indices [num_points]. If None, assumes single batch.
            channels: Channel indices [num_points]. If provided, computes
                     per-channel ECT.

        Returns:
            ECT tensor:
            - Without channels: [batch_size, resolution, num_thetas]
            - With channels: [batch_size, num_thetas, resolution, num_channels]
        """
        v = self._normalize_directions()
        nh = (x @ v).contiguous()

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        batch = batch.contiguous()

        lin = self.lin.contiguous()
        dim_size = batch.max().item() + 1
        scale = self.config.scale

        if channels is not None:
            channels = channels.contiguous()
            max_channels = channels.max().item() + 1
            ect = EctChannelsFunction.apply(
                nh, batch, channels, lin, dim_size, max_channels, scale
            )
        else:
            ect = EctPointsFunction.apply(nh, batch, lin, dim_size, scale)

        if self.config.normalized:
            if channels is not None:
                ect = ect / torch.amax(ect, dim=(-1, -2), keepdim=True).clamp(min=1e-8)
            else:
                ect = ect / torch.amax(ect, dim=(1, 2), keepdim=True).clamp(min=1e-8)

        return ect


class FastEctLayer(nn.Module):
    """Fast (non-differentiable) ECT layer using bincount.

    This layer is optimized for inference speed but does not support
    backpropagation. Use for feature extraction in non-trainable pipelines.

    Args:
        config: EctConfig instance with computation parameters.
        directions: Optional tensor of directions.
    """

    def __init__(
        self,
        config: EctConfig,
        directions: Optional[Tensor] = None,
    ):
        super().__init__()
        self.config = config

        if directions is not None:
            v = directions
        else:
            v_np = generate_directions(
                config.num_thetas,
                config.ambient_dim,
                config.sampling_method,
                config.seed,
            )
            v = torch.from_numpy(v_np).float()

        self.register_buffer("v", v)

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """Compute fast ECT.

        Args:
            x: Point coordinates [num_points, ambient_dim]
            batch: Batch indices [num_points]

        Returns:
            ECT tensor of shape [batch_size, resolution, num_thetas]
        """
        nh = (x @ self.v).contiguous()
        nh_np = nh.detach().cpu().numpy()

        if batch is None:
            out_np = trailed_rust.compute_fast_ect(nh_np, self.config.resolution)
            out = torch.from_numpy(out_np).to(x.device)
            return out.unsqueeze(0)
        else:
            batch_np = batch.detach().cpu().numpy()
            dim_size = batch.max().item() + 1
            out_np = trailed_rust.compute_fast_ect_batched(
                nh_np, batch_np, dim_size, self.config.resolution
            )
            return torch.from_numpy(out_np).to(x.device)
