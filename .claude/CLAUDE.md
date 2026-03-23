# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install runtime dependencies and build the Rust extension
uv sync
uv pip install -e .

# Rebuild the Rust extension after changes to src/lib.rs
maturin develop

# Install test dependencies (includes the upstream 'dect' package for comparison)
uv sync --group tests

# Run all tests (use -s to see the benchmark table printed to stdout)
uv run pytest -s tests/test_dect.py

# Run a single test
uv run pytest -s tests/test_dect.py::test_compare_with_original
uv run pytest -s tests/test_dect.py::test_compare_with_original[faces]
```

## Architecture

This is a **Rust-backed Python package** built with [maturin](https://github.com/PyO3/maturin) (PyO3 + numpy-rust). It reimplements the [aidos-lab/DECT](https://github.com/aidos-lab/DECT) library with a fast Rust core.

### Two-layer structure

**Rust core (`src/lib.rs` → compiled to `dect_rust` Python module)**
- Implements `compute_ecc_forward_loop` / `compute_ecc_backward_loop` as shared Rust functions (not exposed to Python directly) that implement the sigmoid-based ECC summation.
- Exposes 8 `#[pyfunction]`s covering forward+backward for: `points`, `points_derivative`, `edges`, `faces`.
- Edge ECT = points − max(edge endpoints). Face ECT = points − edges + max(face vertices). Backward pass tracks `argmax` to route gradients correctly.

**Python layer (`dect/ect.py` → importable as `dect`)**
- `EctConfig`: frozen dataclass with fields `num_thetas`, `bump_steps`, `R`, `ect_type`, `device`, `num_features`, `normalized`.
- `EctLayer(nn.Module)`: projects node features via `data.x @ V` (shape `[N, num_thetas]`), then calls the appropriate `torch.autograd.Function` subclass, which converts tensors to numpy, calls the Rust functions, and converts back.
- Four `torch.autograd.Function` subclasses: `EctPointsFunction`, `EctPointsDerivativeFunction`, `EctEdgesFunction`, `EctFacesFunction`.

### Import disambiguation

`conftest.py` ensures `import dect` resolves to the **local** `dect/` package (this repo) rather than the upstream pip-installed `dect` package. The test file imports both as `dect_rust_pkg` (local) and `dect_orig_mod` (upstream) for comparison.

### Key numeric details
- `ecc_factor = 50.0` for all standard functions; `100.0` for `points_derivative`.
- Output shape is `(dim_size, bump_steps, num_thetas)` — i.e., `(num_graphs, B, T)`.
- `lin` is a linspace over `[-R, R]` with `bump_steps` points.
