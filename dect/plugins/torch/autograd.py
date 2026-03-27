"""
PyTorch autograd functions for differentiable ECT computation.

This module provides custom autograd.Function implementations that wrap
the Rust ECT backend for differentiable computation.
"""

import numpy as np
import torch
import trailed_rust


class EctPointsFunction(torch.autograd.Function):
    """Autograd function for differentiable ECT on point clouds."""

    @staticmethod
    def forward(ctx, nh, batch, lin, dim_size, scale):
        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)

        out_np = trailed_rust.compute_ect_points_forward(
            nh_np, batch_np, lin_np, dim_size, scale
        )

        ctx.save_for_backward(nh, batch, lin)
        ctx.scale = scale

        return torch.from_numpy(out_np).to(nh.device)

    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, lin = ctx.saved_tensors
        scale = ctx.scale

        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)
        grad_output_np = (
            grad_output.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        grad_nh_np = trailed_rust.compute_ect_points_backward(
            nh_np, batch_np, lin_np, grad_output_np, scale
        )

        return torch.from_numpy(grad_nh_np).to(nh.device), None, None, None, None


class EctPointsDerivativeFunction(torch.autograd.Function):
    """Autograd function for derivative ECT on point clouds."""

    @staticmethod
    def forward(ctx, nh, batch, lin, dim_size, scale):
        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)

        out_np = trailed_rust.compute_ect_points_derivative_forward(
            nh_np, batch_np, lin_np, dim_size, scale
        )

        ctx.save_for_backward(nh, batch, lin)
        ctx.scale = scale

        return torch.from_numpy(out_np).to(nh.device)

    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, lin = ctx.saved_tensors
        scale = ctx.scale

        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)
        grad_output_np = (
            grad_output.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        grad_nh_np = trailed_rust.compute_ect_points_derivative_backward(
            nh_np, batch_np, lin_np, grad_output_np, scale
        )

        return torch.from_numpy(grad_nh_np).to(nh.device), None, None, None, None


class EctEdgesFunction(torch.autograd.Function):
    """Autograd function for ECT on graphs (edges)."""

    @staticmethod
    def forward(ctx, nh, batch, edge_index, lin, dim_size, scale):
        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        edge_index_np = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)

        out_np = trailed_rust.compute_ect_edges_forward(
            nh_np, batch_np, edge_index_np, lin_np, dim_size, scale
        )

        ctx.save_for_backward(nh, batch, edge_index, lin)
        ctx.scale = scale

        return torch.from_numpy(out_np).to(nh.device)

    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, edge_index, lin = ctx.saved_tensors
        scale = ctx.scale

        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        edge_index_np = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)
        grad_output_np = (
            grad_output.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        grad_nh_np = trailed_rust.compute_ect_edges_backward(
            nh_np, batch_np, edge_index_np, lin_np, grad_output_np, scale
        )

        return (
            torch.from_numpy(grad_nh_np).to(nh.device),
            None,
            None,
            None,
            None,
            None,
        )


class EctFacesFunction(torch.autograd.Function):
    """Autograd function for ECT on meshes (faces)."""

    @staticmethod
    def forward(ctx, nh, batch, edge_index, face, lin, dim_size, scale):
        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        edge_index_np = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        face_np = face.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)

        out_np = trailed_rust.compute_ect_faces_forward(
            nh_np, batch_np, edge_index_np, face_np, lin_np, dim_size, scale
        )

        ctx.save_for_backward(nh, batch, edge_index, face, lin)
        ctx.scale = scale

        return torch.from_numpy(out_np).to(nh.device)

    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, edge_index, face, lin = ctx.saved_tensors
        scale = ctx.scale

        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        edge_index_np = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        face_np = face.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)
        grad_output_np = (
            grad_output.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        grad_nh_np = trailed_rust.compute_ect_faces_backward(
            nh_np, batch_np, edge_index_np, face_np, lin_np, grad_output_np, scale
        )

        return (
            torch.from_numpy(grad_nh_np).to(nh.device),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class EctChannelsFunction(torch.autograd.Function):
    """Autograd function for ECT with channel support."""

    @staticmethod
    def forward(ctx, nh, batch, channels, lin, dim_size, max_channels, scale):
        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        channels_np = channels.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)

        out_np = trailed_rust.compute_ect_channels_forward(
            nh_np, batch_np, channels_np, lin_np, dim_size, max_channels, scale
        )

        ctx.save_for_backward(nh, batch, channels, lin)
        ctx.scale = scale

        return torch.from_numpy(out_np).to(nh.device)

    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, channels, lin = ctx.saved_tensors
        scale = ctx.scale

        nh_np = nh.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_np = batch.detach().cpu().numpy().astype(np.int64, copy=False)
        channels_np = channels.detach().cpu().numpy().astype(np.int64, copy=False)
        lin_np = lin.detach().cpu().numpy().flatten().astype(np.float32, copy=False)
        grad_output_np = (
            grad_output.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        grad_nh_np = trailed_rust.compute_ect_channels_backward(
            nh_np, batch_np, channels_np, lin_np, grad_output_np, scale
        )

        return (
            torch.from_numpy(grad_nh_np).to(nh.device),
            None,
            None,
            None,
            None,
            None,
            None,
        )
