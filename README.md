# TRAILED: Topological Regularization and Integrity Learning for EHR Data

A high-performance Rust implementation of the Euler Characteristic Transform (ECT), exposed through ergonomic Python bindings. TRAILED provides the ECT foundation for topological analysis of structured data — including patient trajectories, point clouds, graphs, and simplicial complexes.

## Overview

The Euler Characteristic Transform (ECT) is a provably injective topological descriptor — it encodes the shape of a dataset losslessly, without relying on coordinates or distance metrics. TRAILED implements a differentiable ECT, enabling it to be used as both a feature extractor and a training-time regularizer in deep learning pipelines.

This library is the open-source ECT core. It is designed to be embedded into larger systems that require topologically-aware representations of structured or sequential data.

## Features

- Fast Rust core for ECT computation.
- Differentiable: supports forward and backward passes for use as a loss or layer.
- Native Python API for NumPy workflows.
- Optional integrations for scikit-learn, PyTorch, and dataframe libraries.

## Installation

```bash
uv pip install -e .

# Optional extras
pip install trailed[sklearn]      # scikit-learn transformers
pip install trailed[torch]        # PyTorch layers
pip install trailed[dataframe]    # pandas + polars
pip install trailed[all]          # all optional dependencies
```

## Quick Start

```python
# NumPy
from trailed import compute_ect_from_numpy

ect = compute_ect_from_numpy(points, num_thetas=32, resolution=32)

# Scikit-learn
from trailed import EctTransformer

transformer = EctTransformer(num_thetas=32, resolution=32)
features = transformer.fit_transform(X)  # X: (n_samples, n_points, n_dims)

# PyTorch
from trailed import EctConfig, EctLayer

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
