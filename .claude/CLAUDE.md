# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

TRAILED (Topological Regularization and Integrity Learning for EHR Data) is an open-source, high-performance implementation of the Euler Characteristic Transform (ECT). It serves as the ECT foundation layer for topological analysis of structured data — the core primitive for topologically-aware representation learning.

## Commands

```bash
# Install runtime dependencies and build the Rust extension
uv sync
uv pip install -e .

# Rebuild the Rust extension after changes to src/lib.rs
maturin develop

# Install test dependencies
uv sync --group tests

# Run all tests
uv run pytest tests/
```

## Architecture

This is a **Rust-backed Python package** built with [maturin](https://github.com/PyO3/maturin) (PyO3 + numpy-rust). It reimplements the [aidos-lab/DECT](https://github.com/aidos-lab/DECT) library with a fast Rust core.

### Two-layer structure

**Rust core (`src/lib.rs` → compiled to `dect_rust` Python module)**
- Implements `compute_ecc_forward_loop` / `compute_ecc_backward_loop` as shared Rust functions (not exposed to Python directly) that implement the sigmoid-based ECC summation.
- Exposes 8 `#[pyfunction]`s covering forward+backward for: `points`, `points_derivative`, `edges`, `faces`.
- Edge ECT = points − max(edge endpoints). Face ECT = points − edges + max(face vertices). Backward pass tracks `argmax` to route gradients correctly.

**Python layer (`trailed/` package)**
- `sampling/`: direction generation and utility functions (wrapping Rust implementations)
- `tabular/`: DataFrame-based ECT computation for pandas/polars
- `plugins/sklearn/`: scikit-learn transformers for ECT-based feature engineering

### Key numeric details
- `ecc_factor = 50.0` for all standard functions; `100.0` for `points_derivative`.
- Output shape is `(dim_size, bump_steps, num_thetas)` — i.e., `(num_graphs, B, T)`.
- `lin` is a linspace over `[-R, R]` with `bump_steps` points.
