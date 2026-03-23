from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised in sklearn-only environments
    torch = None
    nn = None

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except Exception:  # pragma: no cover - fallback when sklearn is unavailable
    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class TransformerMixin:  # type: ignore[no-redef]
        pass

try:
    import dect_rust as _dect_backend
except ImportError:
    import dect as _dect_backend


@dataclass(frozen=True)
class EctConfig:
    num_thetas: int = 32
    bump_steps: int = 32
    R: float = 1.1
    ect_type: str = "points"
    device: str = "cpu"
    num_features: int = 3
    normalized: bool = False


@dataclass(frozen=True)
class SklearnEctConfig:
    num_thetas: int = 32
    num_features: int = 3
    direction_mode: str = "random"
    directions: np.ndarray | None = None
    random_state: int | None = None

    def build_directions(self) -> np.ndarray:
        if self.direction_mode == "fixed":
            if self.directions is None:
                raise ValueError(
                    "directions must be provided when direction_mode='fixed'"
                )
            directions = np.asarray(self.directions, dtype=np.float32)
        elif self.direction_mode == "deterministic":
            rng = np.random.default_rng(0)
            directions = rng.random((self.num_features, self.num_thetas)) - 0.5
        elif self.direction_mode == "random":
            rng = np.random.default_rng(self.random_state)
            directions = rng.random((self.num_features, self.num_thetas)) - 0.5
        else:
            raise ValueError(
                f"Unknown direction_mode: {self.direction_mode}. "
                "Expected one of ['random', 'deterministic', 'fixed']"
            )

        if directions.shape != (self.num_features, self.num_thetas):
            raise ValueError(
                f"directions must have shape ({self.num_features}, "
                f"{self.num_thetas}), got {directions.shape}"
            )

        norms = np.linalg.norm(directions, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        return (directions / norms).astype(np.float32, copy=False)


class EctTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ect_type: str = "points",
        num_thetas: int = 32,
        bump_steps: int = 32,
        R: float = 1.1,
        num_features: int = 3,
        normalized: bool = False,
        direction_mode: str = "random",
        directions: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        self.ect_type = ect_type
        self.num_thetas = num_thetas
        self.bump_steps = bump_steps
        self.R = R
        self.num_features = num_features
        self.normalized = normalized
        self.direction_mode = direction_mode
        self.directions = directions
        self.random_state = random_state

    def fit(self, X, y=None):
        self._lin_ = np.linspace(-self.R, self.R, self.bump_steps).astype(
            np.float32
        )
        cfg = SklearnEctConfig(
            num_thetas=self.num_thetas,
            num_features=self.num_features,
            direction_mode=self.direction_mode,
            directions=self.directions,
            random_state=self.random_state,
        )
        self._v_ = cfg.build_directions()
        return self

    def transform(self, X):
        if not hasattr(self, "_v_"):
            raise RuntimeError("EctTransformer must be fitted before transform.")

        if self._is_graph_data(X):
            samples = [X]
        else:
            samples = list(X)
            if not samples:
                return np.empty(
                    (0, self.bump_steps * self.num_thetas), dtype=np.float32
                )
            invalid = [idx for idx, sample in enumerate(samples) if not self._is_graph_data(sample)]
            if invalid:
                raise ValueError(
                    "All samples passed to transform must be graph-like "
                    f"objects with an 'x' attribute. Invalid indices: {invalid}"
                )
        outputs = [self._transform_graph_sample(sample) for sample in samples]
        return np.vstack(outputs)

    @staticmethod
    def _is_graph_data(sample: Any) -> bool:
        return hasattr(sample, "x")

    @staticmethod
    def _to_numpy(array_like: Any, dtype: np.dtype) -> np.ndarray:
        if torch is not None and torch.is_tensor(array_like):
            return array_like.detach().cpu().numpy().astype(dtype, copy=False)
        return np.asarray(array_like, dtype=dtype)

    def _transform_graph_sample(self, data: Any) -> np.ndarray:
        x_np = self._to_numpy(data.x, np.float32)
        if x_np.ndim != 2 or x_np.shape[1] != self.num_features:
            raise ValueError(
                f"Expected x with shape (n_nodes, {self.num_features}), got {x_np.shape}"
            )

        if hasattr(data, "batch") and data.batch is not None:
            batch_np = self._to_numpy(data.batch, np.int64)
        else:
            batch_np = np.zeros(x_np.shape[0], dtype=np.int64)

        nh_np = (x_np @ self._v_).astype(np.float32, copy=False)
        dim_size = int(batch_np.max()) + 1 if batch_np.size else 1

        if self.ect_type == "points":
            ect = _dect_backend.compute_ect_points_forward(
                nh_np, batch_np, self._lin_, dim_size
            )
        elif self.ect_type == "points_derivative":
            ect = _dect_backend.compute_ect_points_derivative_forward(
                nh_np, batch_np, self._lin_, dim_size
            )
        elif self.ect_type == "edges":
            edge_index = self._to_numpy(data.edge_index, np.int64)
            ect = _dect_backend.compute_ect_edges_forward(
                nh_np, batch_np, edge_index, self._lin_, dim_size
            )
        elif self.ect_type == "faces":
            edge_index = self._to_numpy(data.edge_index, np.int64)
            face = self._to_numpy(data.face, np.int64)
            ect = _dect_backend.compute_ect_faces_forward(
                nh_np, batch_np, edge_index, face, self._lin_, dim_size
            )
        else:
            raise ValueError(f"Unknown ect_type: {self.ect_type}")

        ect = np.asarray(ect, dtype=np.float32)
        if self.normalized:
            denom = np.amax(ect, axis=(1, 2), keepdims=True)
            denom = np.where(denom == 0.0, 1.0, denom)
            ect = ect / denom

        return ect.reshape((ect.shape[0], -1))


if torch is not None:
    class EctPointsFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, nh, batch, lin, dim_size):
            nh_np = nh.detach().cpu().numpy()
            batch_np = batch.detach().cpu().numpy()
            lin_np = lin.detach().cpu().numpy().flatten()
            out_np = _dect_backend.compute_ect_points_forward(
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
            grad_nh_np = _dect_backend.compute_ect_points_backward(
                nh_np, batch_np, lin_np, grad_output_np
            )
            return torch.from_numpy(grad_nh_np).to(nh.device), None, None, None

    class EctPointsDerivativeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, nh, batch, lin, dim_size):
            nh_np = nh.detach().cpu().numpy()
            batch_np = batch.detach().cpu().numpy()
            lin_np = lin.detach().cpu().numpy().flatten()
            out_np = _dect_backend.compute_ect_points_derivative_forward(
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
            grad_nh_np = _dect_backend.compute_ect_points_derivative_backward(
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
            out_np = _dect_backend.compute_ect_edges_forward(
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
            grad_nh_np = _dect_backend.compute_ect_edges_backward(
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
            out_np = _dect_backend.compute_ect_faces_forward(
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
            grad_nh_np = _dect_backend.compute_ect_faces_backward(
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
                    torch.rand(size=(config.num_features, config.num_thetas))
                    - 0.5
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
                ect = EctPointsDerivativeFunction.apply(
                    nh, batch, lin, dim_size
                )
            elif self.config.ect_type == "edges":
                edge_index = data.edge_index.contiguous()
                ect = EctEdgesFunction.apply(
                    nh, batch, edge_index, lin, dim_size
                )
            elif self.config.ect_type == "faces":
                edge_index = data.edge_index.contiguous()
                face = data.face.contiguous()
                ect = EctFacesFunction.apply(
                    nh, batch, edge_index, face, lin, dim_size
                )
            else:
                raise ValueError(f"Unknown ect_type: {self.config.ect_type}")

            if self.config.normalized:
                return ect / torch.amax(ect, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
            else:
                return ect
else:
    class EctLayer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required to use EctLayer. Install torch to use this API."
            )
