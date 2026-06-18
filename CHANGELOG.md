# Changelog

All notable changes to the TRAILED project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.2] - 2026-06-18

### Added
- **Graph ECT Support:** Added the ability to compute full V - E Euler Characteristic Transforms for graph data by exposing `edge_index` across numpy, pandas, and polars computation interfaces (`compute_ect_from_numpy`, `compute_ect_from_pandas`, and `compute_ect_from_polars`).
- **Graph ECT Normalization Verification Test:** Added `test_graph_ect_normalization` inside `tests/test_graph_ect.py` to assert that normalized graph ECTs with negative values are appropriately bounded and scaled.

### Fixed
- **Principled ECT Normalization:** Upgraded the `normalized=True` functionality across `EctTransformer`, `EctChannelTransformer`, and core tabular APIs to use **Max Absolute Value** normalization (`np.max(np.abs(ect))`) instead of standard maximum division. This correctly bounds graph ECT values within `[-1, 1]` while remaining 100% backwards-compatible with standard non-negative point cloud ECTs (mapping them to `[0, 1]`).

---

## [0.1.1] - 2026-04-01

### Changed
- **Package Renaming:** Rebranded and renamed the package from `dect` to `trailed` across all files, directories, imports, and tests.
- **Upstream Integration (PyTorch):** Removed the local PyTorch/Torch plugin layer to avoid redundant tensor conversions; pointed deep-learning users to the upstream package `aidos-lab/dect` instead.
- **Enhanced Documentation:** Created extensive Sphinx-based documentation including custom CSS layouts, logo branding, quickstart tutorials, programmatic integration guidelines, API references, and testing standards.
- **Streamlined Configuration:** Simplified `README.md` to target the core `trailed` package usage and cleaned up test configuration files like `tests/conftest.py`.

### Improved
- **Benchmarking Capabilities:** Refactored benchmarks to support aggregating across multiple random seeds, returning cleaner and more robust statistical performance metrics.

---

## [0.1.0] - 2026-03-27

### Added
- **Rust Core Backend (`trailed_rust`):** Ported the core Euler Characteristic Curve (ECC) / Euler Characteristic Transform (ECT) computations into highly optimized Rust for blazing-fast computations.
- **Direction Sampling:** Introduced modular direction sampling strategies (uniform, 2D structured, multiview, and spherical grid) to project point clouds/graphs into configurable vector spaces.
- **Scikit-Learn Transformers:** Implemented Scikit-Learn compatible API classes (`EctTransformer`, `FastEctTransformer`, and `EctChannelTransformer`) for direct pipeline integration.
- **Tabular Integrations:** Provided Pandas and Polars DataFrame wrapper interfaces for smooth tabular workflows.
- **Rayon Parallelization:** Implemented multi-threaded Rayon parallelization in the Rust backend for high-volume batch processing of directions.
- **CI Pipelines:** Built GitHub Actions CI workflows to automate unit testing and formatting validation for Python 3.12 and 3.13.
- **Maintainability Guidelines:** Added a file-size enforcement unit test (`test_file_size.py`) ensuring Python source files stay under the 150-500 line sweet spot for optimal editor parsing and AI assistance.
