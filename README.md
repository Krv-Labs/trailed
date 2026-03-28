# TRAILED: Topological Regularization and Integrity Learning for EHR Data

> **Note**: TRAILED is under active development. The current release provides the foundational ECT implementation. Healthcare-specific methods are in progress.

Topological representation learning for Electronic Health Record (EHR) data. Built on the differentiable Euler Characteristic Transform (ECT).

## Installation

```bash
uv pip install trailed

# Optional extras
uv pip install trailed[sklearn]      # scikit-learn transformers
uv pip install trailed[torch]        # PyTorch layers
uv pip install trailed[all]          # all dependencies
```

## Quick Start

```python
from trailed import compute_ect_from_numpy

# Compute topological descriptor
ect = compute_ect_from_numpy(patient_embeddings, num_thetas=32, resolution=32)
```

```python
# PyTorch — differentiable for use as training regularizer
from trailed.torch import EctLayer, EctConfig

layer = EctLayer(EctConfig(num_thetas=32, resolution=32))
ect = layer(data)  # gradients flow through
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
