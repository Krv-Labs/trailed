.. _benchmarks:

==========
Benchmarks
==========

TRAILED provides a Rust-accelerated backend with native Polars integration, eliminating
the DataFrame-to-Tensor conversion overhead required by the upstream
`aidos-lab/dect <https://github.com/aidos-lab/DECT>`_ package.

Architecture Comparison
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - trailed
     - upstream dect
   * - Pipeline
     - Polars → Rust (direct)
     - Polars → NumPy → Torch → ECTLayer
   * - Dependencies
     - Polars only
     - Polars + NumPy + PyTorch
   * - Integration
     - Native ``compute_ect_from_polars``
     - Manual conversion glue code

When to Use Which
-----------------

- **trailed**: Batch ECT computation on tabular data (EHR, time series), sklearn pipelines,
  inference workloads where PyTorch is not needed.
- **upstream dect**: PyTorch training loops requiring gradient flow through the ECT layer,
  graph neural network integration via PyTorch Geometric.

Running Benchmarks
------------------

The benchmark suite compares end-to-end wall-clock time on simulated EHR data
(patient visits with lab values and vitals).

**Setup:**

.. code-block:: bash

   # Install test dependencies (includes trailed, dect, torch)
   uv sync --group tests

**Run benchmarks:**

.. code-block:: bash

   # Benchmark comparisons only (requires upstream dect + torch)
   uv run pytest -v -s tests/compare_polars_ehr.py::TestPolarsEHRBenchmark

   # All tests: correctness + benchmarks
   uv run pytest -v -s tests/compare_polars_ehr.py

   # Or run directly as a script
   uv run python tests/compare_polars_ehr.py

Results
-------

Each scenario is run over **10 random seeds** with 5 timed iterations per seed.
Results show mean ± std wall-clock time in milliseconds.

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18

   * - Scenario
     - trailed (ms)
     - upstream (ms)
     - Speedup
   * - Small (1K pts, 32×32)
     - 0.70 ± 0.05
     - 0.75 ± 0.03
     - 1.06x ± 0.05x
   * - Medium (10K pts, 64×64)
     - 8.19 ± 0.18
     - 12.83 ± 0.62
     - 1.57x ± 0.06x
   * - Large (100K pts, 64×64)
     - 75.27 ± 1.74
     - 120.01 ± 0.98
     - 1.60x ± 0.04x
   * - Labs only (10K pts, 5-D)
     - 2.89 ± 0.07
     - 4.27 ± 0.11
     - 1.48x ± 0.05x
   * - High-res (5K pts, 128×128)
     - 15.17 ± 1.69
     - 34.44 ± 1.70
     - 2.29x ± 0.23x

The speedup increases with resolution (high-res: **2.3x**) and data size (large: **1.6x**),
as trailed's Rust backend avoids Python/NumPy intermediate allocations and
Polars-to-Torch conversion overhead.

Example Output
--------------

.. code-block:: text

   ================================================================================
     Large cohort  — 500 patients × 200 visits, 10-D, 64×64
     (averaged over 10 random seeds)
   ================================================================================
   Metric                         trailed (Rust)           dect (upstream)
   --------------------------------------------------------------------------------
   Wall-clock (ms)                75.27 ± 1.74             120.01 ± 0.98
   --------------------------------------------------------------------------------
     Speedup: 1.60x ± 0.04x (trailed faster)

     Output shape: trailed=(500, 64, 64), upstream=(500, 64, 64)

Correctness Validation
----------------------

The benchmark suite includes correctness tests that verify:

- **Monotonicity**: ECT values are non-decreasing along the threshold axis
- **Consistency**: Polars and NumPy paths produce identical results
- **Correlation**: trailed and upstream outputs have >0.99 correlation when using identical directions

These tests run automatically with the full test suite:

.. code-block:: bash

   uv run pytest -v -s tests/compare_polars_ehr.py
