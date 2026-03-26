# DECT

A high-performance Rust implementation of the Differentiable Euler Characteristic Transform (DECT), exposed through ergonomic Python bindings.

## Features

- Fast Rust core for DECT computation.
- Native Python API for NumPy workflows.
- Optional integrations for scikit-learn, PyTorch, and dataframe libraries.

## Installation

```bash
uv pip install -e .

# Optional extras
pip install dect-rust[sklearn]      # scikit-learn transformers
pip install dect-rust[torch]        # PyTorch layers
pip install dect-rust[dataframe]    # pandas + polars
pip install dect-rust[all]          # all optional dependencies
```

## Quick Start

```python
# NumPy
from dect import compute_ect_from_numpy

ect = compute_ect_from_numpy(points, num_thetas=32, resolution=32)

# Scikit-learn
from dect import EctTransformer

transformer = EctTransformer(num_thetas=32, resolution=32)
features = transformer.fit_transform(X)  # X: (n_samples, n_points, n_dims)

# PyTorch
from dect import EctConfig, EctLayer

layer = EctLayer(EctConfig(num_thetas=32, bump_steps=32))
ect = layer(data)  # torch_geometric Data
```

## Running Tests

```bash
uv sync --group tests
uv run pytest
```

## Acknowledgment

This project builds on the original [dect](https://github.com/aidos-lab/dect) implementation and accompanying research.

```bibtex
@inproceedings{Roell24a,
  title         = {Differentiable Euler Characteristic Transforms for Shape Classification},
  author        = {Ernst R{\"o}ell and Bastian Rieck},
  year          = 2024,
  booktitle     = {International Conference on Learning Representations},
  eprint        = {2310.07630},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  repository    = {https://github.com/aidos-lab/dect-evaluation},
  url           = {https://openreview.net/forum?id=MO632iPq3I},
}
```