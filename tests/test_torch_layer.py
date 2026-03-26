"""Tests for PyTorch layer implementations."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dect.torch_layer import (
    EctConfig,
    EctLayer,
    FastEctLayer,
    compute_ect,
    generate_directions,
)


class TestEctConfig:
    def test_defaults(self):
        config = EctConfig()
        assert config.num_thetas == 32
        assert config.resolution == 32
        assert config.radius == 1.1
        assert config.scale == 500.0
        assert config.ambient_dim == 3
        assert config.ect_type == "points"
    
    def test_custom_values(self):
        config = EctConfig(
            num_thetas=64,
            resolution=128,
            radius=2.0,
            scale=1000.0,
            ambient_dim=2,
            ect_type="edges",
        )
        assert config.num_thetas == 64
        assert config.resolution == 128


class TestGenerateDirections:
    def test_uniform(self):
        v = generate_directions(64, 3, method="uniform")
        assert v.shape == (3, 64)
    
    def test_structured_2d(self):
        v = generate_directions(32, 2, method="structured_2d")
        assert v.shape == (2, 32)
    
    def test_multiview(self):
        v = generate_directions(64, 3, method="multiview")
        assert v.shape == (3, 64)


class TestEctLayer:
    @pytest.fixture
    def config(self):
        return EctConfig(
            num_thetas=16,
            resolution=16,
            radius=1.0,
            scale=100.0,
            ambient_dim=3,
            ect_type="points",
        )
    
    @pytest.fixture
    def layer(self, config):
        return EctLayer(config)
    
    def test_initialization(self, layer, config):
        assert layer.v.shape == (config.ambient_dim, config.num_thetas)
        assert layer.lin.shape == (config.resolution,)
    
    def test_forward_raw_single(self, layer):
        x = torch.randn(50, 3)
        ect = layer.forward_raw(x)
        assert ect.shape == (1, 16, 16)
    
    def test_forward_raw_batched(self, layer):
        x = torch.randn(100, 3)
        batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        ect = layer.forward_raw(x, batch)
        assert ect.shape == (2, 16, 16)
    
    def test_forward_raw_with_channels(self, layer):
        x = torch.randn(100, 3)
        batch = torch.zeros(100).long()
        channels = torch.randint(0, 3, (100,)).long()
        ect = layer.forward_raw(x, batch, channels)
        assert ect.shape == (1, 16, 16, 3)
    
    def test_backward(self, layer):
        x = torch.randn(50, 3, requires_grad=True)
        ect = layer.forward_raw(x)
        loss = ect.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()
    
    def test_learnable_directions(self, config):
        layer = EctLayer(config, learnable=True)
        assert layer.v.requires_grad
        
        x = torch.randn(50, 3)
        ect = layer.forward_raw(x)
        loss = ect.sum()
        loss.backward()
        
        assert layer.v.grad is not None
    
    def test_normalized(self, config):
        config.normalized = True
        layer = EctLayer(config)
        
        x = torch.randn(50, 3)
        ect = layer.forward_raw(x)
        
        assert ect.max() <= 1.0 + 1e-5
        assert ect.min() >= 0.0 - 1e-5


class TestFastEctLayer:
    @pytest.fixture
    def config(self):
        return EctConfig(
            num_thetas=32,
            resolution=32,
            ambient_dim=3,
        )
    
    @pytest.fixture
    def layer(self, config):
        return FastEctLayer(config)
    
    def test_forward_single(self, layer):
        x = torch.randn(50, 3)
        ect = layer(x)
        assert ect.shape == (1, 32, 32)
    
    def test_forward_batched(self, layer):
        x = torch.randn(100, 3)
        batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        ect = layer(x, batch)
        assert ect.shape == (2, 32, 32)
    
    def test_no_gradient(self, layer):
        x = torch.randn(50, 3, requires_grad=True)
        ect = layer(x)
        
        # FastEctLayer doesn't support gradients
        with pytest.raises(RuntimeError):
            ect.sum().backward()


class TestComputeEct:
    def test_basic(self):
        x = torch.randn(50, 3)
        v = torch.randn(3, 16)
        v = v / v.norm(dim=0, keepdim=True)
        
        ect = compute_ect(x, v, radius=1.0, resolution=32, scale=100.0)
        assert ect.shape == (1, 32, 16)
    
    def test_batched(self):
        x = torch.randn(100, 3)
        v = torch.randn(3, 16)
        v = v / v.norm(dim=0, keepdim=True)
        batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        
        ect = compute_ect(x, v, batch=batch)
        assert ect.shape == (2, 64, 16)
    
    def test_gradient(self):
        x = torch.randn(50, 3, requires_grad=True)
        v = torch.randn(3, 16)
        v = v / v.norm(dim=0, keepdim=True)
        
        ect = compute_ect(x, v)
        loss = ect.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestTorchGeometricCompatibility:
    """Tests for compatibility with torch_geometric Data objects."""
    
    def test_forward_with_data_object(self):
        pytest.importorskip("torch_geometric")
        from torch_geometric.data import Data
        
        config = EctConfig(
            num_thetas=16,
            resolution=16,
            ambient_dim=3,
            ect_type="points",
        )
        layer = EctLayer(config)
        
        data = Data(
            x=torch.randn(50, 3),
            batch=torch.zeros(50, dtype=torch.long),
        )
        
        ect = layer(data)
        assert ect.shape == (1, 16, 16)
    
    def test_forward_edges(self):
        pytest.importorskip("torch_geometric")
        from torch_geometric.data import Data
        
        config = EctConfig(
            num_thetas=16,
            resolution=16,
            ambient_dim=3,
            ect_type="edges",
        )
        layer = EctLayer(config)
        
        num_nodes = 50
        edge_index = torch.stack([
            torch.arange(num_nodes - 1),
            torch.arange(1, num_nodes),
        ])
        
        data = Data(
            x=torch.randn(num_nodes, 3),
            edge_index=edge_index,
            batch=torch.zeros(num_nodes, dtype=torch.long),
        )
        
        ect = layer(data)
        assert ect.shape == (1, 16, 16)
    
    def test_forward_faces(self):
        pytest.importorskip("torch_geometric")
        from torch_geometric.data import Data
        
        config = EctConfig(
            num_thetas=16,
            resolution=16,
            ambient_dim=3,
            ect_type="faces",
        )
        layer = EctLayer(config)
        
        num_nodes = 50
        edge_index = torch.stack([
            torch.arange(num_nodes - 1),
            torch.arange(1, num_nodes),
        ])
        face = torch.stack([
            torch.arange(num_nodes - 2),
            torch.arange(1, num_nodes - 1),
            torch.arange(2, num_nodes),
        ])
        
        data = Data(
            x=torch.randn(num_nodes, 3),
            edge_index=edge_index,
            face=face,
            batch=torch.zeros(num_nodes, dtype=torch.long),
        )
        
        ect = layer(data)
        assert ect.shape == (1, 16, 16)
