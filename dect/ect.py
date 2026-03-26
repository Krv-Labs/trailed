from dataclasses import dataclass

import torch
import torch.nn as nn

import dect


@dataclass(frozen=True)
class EctConfig:
    num_thetas: int = 32
    bump_steps: int = 32
    R: float = 1.1
    ect_type: str = "points"
    device: str = "cpu"
    num_features: int = 3
    normalized: bool = False


class EctPointsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nh, batch, lin, dim_size):
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        out_np = dect.compute_ect_points_forward(nh_np, batch_np, lin_np, dim_size)
        ctx.save_for_backward(nh, batch, lin)
        return torch.from_numpy(out_np).to(nh.device)

    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, lin = ctx.saved_tensors
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        grad_output_np = grad_output.detach().cpu().numpy()
        grad_nh_np = dect.compute_ect_points_backward(
            nh_np, batch_np, lin_np, grad_output_np
        )
        return torch.from_numpy(grad_nh_np).to(nh.device), None, None, None


class EctPointsDerivativeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nh, batch, lin, dim_size):
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        out_np = dect.compute_ect_points_derivative_forward(
            nh_np, batch_np, lin_np, dim_size
        )
        ctx.save_for_backward(nh, batch, lin)
        return torch.from_numpy(out_np).to(nh.device)

    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, lin = ctx.saved_tensors
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        grad_output_np = grad_output.detach().cpu().numpy()
        grad_nh_np = dect.compute_ect_points_derivative_backward(
            nh_np, batch_np, lin_np, grad_output_np
        )
        return torch.from_numpy(grad_nh_np).to(nh.device), None, None, None


class EctEdgesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nh, batch, edge_index, lin, dim_size):
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        out_np = dect.compute_ect_edges_forward(
            nh_np, batch_np, edge_index_np, lin_np, dim_size
        )
        ctx.save_for_backward(nh, batch, edge_index, lin)
        return torch.from_numpy(out_np).to(nh.device)

    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, edge_index, lin = ctx.saved_tensors
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        grad_output_np = grad_output.detach().cpu().numpy()
        grad_nh_np = dect.compute_ect_edges_backward(
            nh_np, batch_np, edge_index_np, lin_np, grad_output_np
        )
        return (
            torch.from_numpy(grad_nh_np).to(nh.device),
            None,
            None,
            None,
            None,
        )


class EctFacesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nh, batch, edge_index, face, lin, dim_size):
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()
        face_np = face.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        out_np = dect.compute_ect_faces_forward(
            nh_np, batch_np, edge_index_np, face_np, lin_np, dim_size
        )
        ctx.save_for_backward(nh, batch, edge_index, face, lin)
        return torch.from_numpy(out_np).to(nh.device)

    @staticmethod
    def backward(ctx, grad_output):
        nh, batch, edge_index, face, lin = ctx.saved_tensors
        nh_np = nh.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()
        face_np = face.detach().cpu().numpy()
        lin_np = lin.detach().cpu().numpy().flatten()
        grad_output_np = grad_output.detach().cpu().numpy()
        grad_nh_np = dect.compute_ect_faces_backward(
            nh_np, batch_np, edge_index_np, face_np, lin_np, grad_output_np
        )
        return (
            torch.from_numpy(grad_nh_np).to(nh.device),
            None,
            None,
            None,
            None,
            None,
        )


class EctLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config: EctConfig, V=None):
        super().__init__()
        self.config = config
        self.lin = torch.linspace(-config.R, config.R, config.bump_steps).to(
            config.device
        )

        if torch.is_tensor(V):
            self.v = V
        else:
            self.v = (
                torch.rand(size=(config.num_features, config.num_thetas)) - 0.5
            ).T.to(config.device)
            self.v /= self.v.pow(2).sum(axis=1).sqrt().unsqueeze(1)
            self.v = self.v.T

    def forward(self, data):
        nh = (data.x @ self.v).contiguous()
        batch = data.batch.contiguous()
        lin = self.lin.contiguous()
        dim_size = data.batch.max().item() + 1

        if self.config.ect_type == "points":
            ect = EctPointsFunction.apply(nh, batch, lin, dim_size)
        elif self.config.ect_type == "points_derivative":
            ect = EctPointsDerivativeFunction.apply(nh, batch, lin, dim_size)
        elif self.config.ect_type == "edges":
            edge_index = data.edge_index.contiguous()
            ect = EctEdgesFunction.apply(nh, batch, edge_index, lin, dim_size)
        elif self.config.ect_type == "faces":
            edge_index = data.edge_index.contiguous()
            face = data.face.contiguous()
            ect = EctFacesFunction.apply(nh, batch, edge_index, face, lin, dim_size)
        else:
            raise ValueError(f"Unknown ect_type: {self.config.ect_type}")

        if self.config.normalized:
            return ect / torch.amax(ect, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        else:
            return ect
