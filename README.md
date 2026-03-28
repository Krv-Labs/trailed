# TRAILED: Topological Regularization and Integrity Learning for EHR Data

> **Note**: TRAILED is under active development. The current release provides the foundational ECT (Euler Characteristic Transform) implementation. Healthcare-specific methods for density-aware descriptors, patient manifold construction, and clinical fidelity metrics are in progress.

A topological representation learning library for Electronic Health Record (EHR) data, with applications in synthetic data generation, patient trajectory analysis, and clinical fidelity assessment.

## Overview

TRAILED provides topological methods for longitudinal clinical data analysis. At its core is the **Euler Characteristic Transform (ECT)** — a provably injective topological descriptor that encodes shape information without relying on coordinates or distance metrics. The differentiable implementation enables use as both a feature extractor and a training-time regularizer in deep learning pipelines.

This library is designed for:

- **Synthetic EHR validation**: Detect mode collapse and pathological interpolation in generated data
- **Patient trajectory analysis**: Characterize clinical pathways in topological latent spaces  
- **Representation learning**: Extract topological features from longitudinal health records

## Key Concepts

### Topological Descriptors for Clinical Data

The ECT captures higher-order structure that coordinate-based metrics miss. For EHR applications, this means detecting:

- **Underrepresented phenotypes**: Rare but clinically significant patient subgroups that generators fail to cover
- **Impossible state transitions**: Synthetic trajectories that pass through biologically implausible regions

### Patient Manifold

TRAILED conceptualizes EHR data as points on a patient manifold, where:

- Viable clinical pathways form connected regions
- "No-Go" zones represent impossible or implausible states
- Topological features capture trajectory structure across time

## Features

- **Fast Rust core** for ECT computation
- **Differentiable**: Full gradient support for end-to-end learning
- **Native Python API** for NumPy workflows
- **Optional integrations** for scikit-learn, PyTorch, and dataframe libraries

## Installation

We recommend installing with [uv](https://docs.astral.sh/uv/) for fast, reliable dependency resolution:

```bash
uv pip install trailed

# Optional extras
uv pip install trailed[sklearn]      # scikit-learn transformers
uv pip install trailed[torch]        # PyTorch layers
uv pip install trailed[dataframe]    # pandas + polars
uv pip install trailed[all]          # all optional dependencies
```

Or with pip:

```bash
pip install trailed
```

For development:

```bash
git clone https://github.com/Krv-Analytics/trailed.git
cd trailed
uv sync --extra dev --extra docs
```

## Quick Start

### NumPy

```python
from trailed import compute_ect_from_numpy

# Compute ECT descriptor for patient embeddings
ect = compute_ect_from_numpy(patient_embeddings, num_thetas=32, resolution=32)
```

### scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from trailed.sklearn import ECTTransformer

pipe = Pipeline([
    ("ect", ECTTransformer(num_thetas=32, resolution=32)),
    ("clf", SVC()),
])

# X: (n_patients, n_timepoints, n_features)
pipe.fit(X_train, y_train)
```

### PyTorch

```python
from trailed.torch import EctConfig, EctLayer

layer = EctLayer(EctConfig(num_thetas=32, resolution=32))
ect = layer(data)  # Differentiable — gradients flow through
```

## Applications

### Synthetic Data Fidelity

Use ECT distances to compare real and synthetic EHR distributions:

```python
from trailed import compute_ect_from_numpy
import numpy as np

# Compare topological structure of real vs synthetic cohorts
real_ect = compute_ect_from_numpy(real_embeddings, num_thetas=64, resolution=64)
synthetic_ect = compute_ect_from_numpy(synthetic_embeddings, num_thetas=64, resolution=64)

fidelity_score = np.linalg.norm(real_ect - synthetic_ect)
```

### Training Regularization

Integrate ECT as a differentiable loss term to regularize generative models:

```python
from trailed.torch import EctLayer, EctConfig

ect_layer = EctLayer(EctConfig(num_thetas=32, resolution=32))

# In training loop
real_ect = ect_layer(real_batch)
generated_ect = ect_layer(generated_batch)
topological_loss = torch.nn.functional.mse_loss(generated_ect, real_ect)

total_loss = reconstruction_loss + lambda_topo * topological_loss
```

## Running Tests

```bash
uv sync --group tests
uv run pytest
```

## Documentation

Full documentation is available at the [TRAILED docs](https://krv-analytics.github.io/trailed/).

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

## License

MIT
