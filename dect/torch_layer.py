"""
PyTorch layers for computing the Euler Characteristic Transform.

This module provides nn.Module implementations that wrap the Rust ECT
computation backend, supporting both fixed and learnable directions.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import dect_rust


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
    sampling_method: Literal["uniform", "structured_2d", "multiview", "spherical_grid"] = "uniform"
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
    if method == "uniform":
        return dect_rust.generate_uniform_directions(num_thetas, ambient_dim, seed)
    elif method == "structured_2d":
        if ambient_dim != 2:
            raise ValueError("structured_2d requires ambient_dim=2")
        return dect_rust.generate_2d_directions(num_thetas)
    elif method == "multiview":
        return dect_rust.generate_multiview_directions(num_thetas, ambient_dim)
    elif method == "spherical_grid":
        if ambient_dim != 3:
            raise ValueError("spherical_grid requires ambient_dim=3")
        num_phis = int(np.sqrt(num_thetas * 2))
        num_t = num_thetas // num_phis
        return dect_rust.generate_spherical_grid_directions(num_t, num_phis)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


class EctPointsFunction(torch.autograd.Function):
    """Autograd function for differentiable ECT on point clouds."""
    
    @staticmethod
    def forward(ctx, nh, batch, lin, dim_size, scale):
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        
        out_np = dect_rust.compute_ect_points_forward(
            nh_np, batch_np, lin_np, dim_size, scale
        )
        
        ctx.save_for_backward(nh, batch, lin)
        ctx.scale = scale
        
        return torch.from_numpy(out_np).to(nh.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, lin = ctx.saved_tensors
        scale = ctx.scale
        
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        grad_output_np = grad_output.detach().cpu().numpy()
        
        grad_nh_np = dect_rust.compute_ect_points_backward(
            nh_np, batch_np, lin_np, grad_output_np, scale
        )
        
        return torch.from_numpy(grad_nh_np).to(nh.device), None, None, None, None


class EctPointsDerivativeFunction(torch.autograd.Function):
    """Autograd function for derivative ECT on point clouds."""
    
    @staticmethod
    def forward(ctx, nh, batch, lin, dim_size, scale):
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        
        out_np = dect_rust.compute_ect_points_derivative_forward(
            nh_np, batch_np, lin_np, dim_size, scale
        )
        
        ctx.save_for_backward(nh, batch, lin)
        ctx.scale = scale
        
        return torch.from_numpy(out_np).to(nh.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, lin = ctx.saved_tensors
        scale = ctx.scale
        
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        grad_output_np = grad_output.detach().cpu().numpy()
        
        grad_nh_np = dect_rust.compute_ect_points_derivative_backward(
            nh_np, batch_np, lin_np, grad_output_np, scale
        )
        
        return torch.from_numpy(grad_nh_np).to(nh.device), None, None, None, None


class EctEdgesFunction(torch.autograd.Function):
    """Autograd function for ECT on graphs (edges)."""
    
    @staticmethod
    def forward(ctx, nh, batch, edge_index, lin, dim_size, scale):
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        
        out_np = dect_rust.compute_ect_edges_forward(
            nh_np, batch_np, edge_index_np, lin_np, dim_size, scale
        )
        
        ctx.save_for_backward(nh, batch, edge_index, lin)
        ctx.scale = scale
        
        return torch.from_numpy(out_np).to(nh.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, edge_index, lin = ctx.saved_tensors
        scale = ctx.scale
        
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        grad_output_np = grad_output.detach().cpu().numpy()
        
        grad_nh_np = dect_rust.compute_ect_edges_backward(
            nh_np, batch_np, edge_index_np, lin_np, grad_output_np, scale
        )
        
        return (
            torch.from_numpy(grad_nh_np).to(nh.device),
            None, None, None, None, None
        )


class EctFacesFunction(torch.autograd.Function):
    """Autograd function for ECT on meshes (faces)."""
    
    @staticmethod
    def forward(ctx, nh, batch, edge_index, face, lin, dim_size, scale):
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()
        face_np = face.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        
        out_np = dect_rust.compute_ect_faces_forward(
            nh_np, batch_np, edge_index_np, face_np, lin_np, dim_size, scale
        )
        
        ctx.save_for_backward(nh, batch, edge_index, face, lin)
        ctx.scale = scale
        
        return torch.from_numpy(out_np).to(nh.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, edge_index, face, lin = ctx.saved_tensors
        scale = ctx.scale
        
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()
        face_np = face.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        grad_output_np = grad_output.detach().cpu().numpy()
        
        grad_nh_np = dect_rust.compute_ect_faces_backward(
            nh_np, batch_np, edge_index_np, face_np, lin_np, grad_output_np, scale
        )
        
        return (
            torch.from_numpy(grad_nh_np).to(nh.device),
            None, None, None, None, None, None
        )


class EctChannelsFunction(torch.autograd.Function):
    """Autograd function for ECT with channel support."""
    
    @staticmethod
    def forward(ctx, nh, batch, channels, lin, dim_size, max_channels, scale):
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        channels_np = channels.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        
        out_np = dect_rust.compute_ect_channels_forward(
            nh_np, batch_np, channels_np, lin_np, dim_size, max_channels, scale
        )
        
        ctx.save_for_backward(nh, batch, channels, lin)
        ctx.scale = scale
        
        return torch.from_numpy(out_np).to(nh.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, channels, lin = ctx.saved_tensors
        scale = ctx.scale
        
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        channels_np = channels.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        grad_output_np = grad_output.detach().cpu().numpy()
        
        grad_nh_np = dect_rust.compute_ect_channels_backward(
            nh_np, batch_np, channels_np, lin_np, grad_output_np, scale
        )
        
        return (
            torch.from_numpy(grad_nh_np).to(nh.device),
            None, None, None, None, None, None
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
            out_np = dect_rust.compute_fast_ect(nh_np, self.config.resolution)
            out = torch.from_numpy(out_np).to(x.device)
            return out.unsqueeze(0)
        else:
            batch_np = batch.detach().cpu().numpy()
            dim_size = batch.max().item() + 1
            out_np = dect_rust.compute_fast_ect_batched(
                nh_np, batch_np, dim_size, self.config.resolution
            )
            return torch.from_numpy(out_np).to(x.device)


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
