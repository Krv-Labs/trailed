# TRAILED: Topological Regularization and Integrity Learning for EHR Data

> **Note**: TRAILED is under active development. The current release provides the foundational ECT implementation. Healthcare-specific methods are in progress.

Topological representation learning for Electronic Health Record (EHR) data. Built on the differentiable Euler Characteristic Transform (ECT).

## Installation

```bash
uv pip install trailed

# Optional extras
uv pip install trailed[sklearn]      # scikit-learn transformers
uv pip install trailed[dataframe]    # pandas/polars support
uv pip install trailed[all]          # all dependencies
```

## Quick Start

```python
from trailed import compute_ect_from_numpy

ect = compute_ect_from_numpy(points, num_thetas=32, resolution=32)
```

```python
import polars as pl
from trailed.tabular import compute_ect_from_polars

df = pl.DataFrame({"x": [0.1, 0.2, 0.3], "y": [0.1, 0.3, 0.2], "group": [0, 0, 1]})
ect = compute_ect_from_polars(df, coord_columns=["x", "y"], group_column="group")
```

```python
import pandas as pd
from trailed.tabular import compute_ect_from_pandas

df = pd.DataFrame({"x": [0.1, 0.2, 0.3], "y": [0.1, 0.3, 0.2], "group": [0, 0, 1]})
ect = compute_ect_from_pandas(df, coord_columns=["x", "y"], group_column="group")
```

## Documentation

Full documentation: [krv-analytics.github.io/trailed](https://krv-analytics.github.io/trailed/)

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
