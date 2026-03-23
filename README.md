# DECT (Differentiable Euler Characteristic Transform) - Rust Port

This is a Rust implementation of the Differentiable Euler Characteristic Transform (DECT), providing a fast and memory-efficient alternative to the original Python/PyTorch implementation.

## Features

- **Rust Backend**: Core computations (Points, Edges, Faces) are implemented in Rust using `ndarray` for performance.
- **Differentiable**: Custom `torch.autograd.Function` wrappers provide seamless integration with PyTorch's automatic differentiation.
- **Pythonic Interface**: Maintains the same API as the original implementation.

## Installation

### Prerequisites

- Rust (Cargo)
- Python 3.10+
- PyTorch and Torch Geometric

### Build from Source

From the `DECT/` directory:

```bash
# Create/update the virtual environment and install runtime deps
uv sync

# Build and install the local package in editable mode
uv pip install -e .

# (Optional) install test dependencies, including the upstream dect package
uv sync --group tests

# Or use maturin directly for development
maturin develop
```

## Testing & Verification

To verify that the Rust implementation matches the original Python version:

```bash
# Install test dependencies
uv sync --group tests

# Run all tests (includes compare and benchmark smoke tests)
uv run pytest -q tests

# Run only comparison tests
uv run pytest -q tests/compare.py

# Run only benchmark smoke tests
uv run pytest -q tests/benchmark.py
```

The pytest configuration is defined in `pyproject.toml` and explicitly includes
`tests/compare.py` and `tests/benchmark.py` for test discovery.

## Usage

```python
import torch
from dect import EctConfig, EctLayer
from torch_geometric.data import Data

config = EctConfig(ect_type="faces")
layer = EctLayer(config)

# Example data
x = torch.randn(10, 3, requires_grad=True)
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
face = torch.tensor([[0], [1], [2]], dtype=torch.long)
batch = torch.zeros(10, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, face=face, batch=batch)
ect = layer(data)
ect.sum().backward()
print(x.grad)
```
