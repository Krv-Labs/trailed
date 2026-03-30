"""
Polars EHR time series comparison: trailed Rust backend vs. upstream dect package.

Demonstrates the natural Polars → ECT workflow on data that resembles a clinical
EHR extract: one row per visit, one column per feature (lab value or vital sign),
with a patient_id grouping column.

The upstream dect package (aidos-lab/DECT) has no Polars integration, so the
upstream benchmark measures the full realistic workflow a user would write:
    Polars DataFrame → numpy → torch → ECTLayer → output tensor

Run with:
    uv run pytest -s tests/compare_polars_ehr.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

polars = pytest.importorskip("polars")

from trailed.tabular import compute_ect_from_polars, compute_ect_from_numpy


# ---------------------------------------------------------------------------
# Load upstream dect (may not be installed)
# ---------------------------------------------------------------------------


def _load_upstream():
    """Load upstream dect package, working around local shadowing."""
    site_pkg = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )

    saved_modules = {k: v for k, v in sys.modules.items() if k.startswith("dect")}
    for key in list(sys.modules.keys()):
        if key.startswith("dect"):
            del sys.modules[key]

    saved_path = sys.path.copy()
    sys.path = [str(site_pkg)] + [
        p for p in sys.path if "trailed" not in p or "site-packages" in p
    ]

    try:
        from dect.nn import ECTConfig, ECTLayer
        from dect.directions import generate_uniform_directions

        return ECTConfig, ECTLayer, generate_uniform_directions
    except ImportError:
        return None, None, None
    finally:
        sys.path = saved_path
        for key in list(sys.modules.keys()):
            if key.startswith("dect"):
                del sys.modules[key]
        sys.modules.update(saved_modules)


UpstreamECTConfig, UpstreamECTLayer, upstream_generate_directions = _load_upstream()
HAS_UPSTREAM = UpstreamECTConfig is not None

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# EHR data generation
# ---------------------------------------------------------------------------

# Realistic column names for a clinical extract
_LAB_COLUMNS = ["hemoglobin", "wbc", "creatinine", "sodium", "potassium"]
_VITAL_COLUMNS = ["heart_rate", "sbp", "dbp", "temperature", "spo2"]
_ALL_FEATURES = _LAB_COLUMNS + _VITAL_COLUMNS  # 10-dimensional feature space


def generate_ehr_polars(
    n_patients: int = 100,
    visits_per_patient: int = 50,
    seed: int = 42,
) -> polars.DataFrame:
    """
    Simulate an EHR extract as a Polars DataFrame.

    Each row represents one clinical visit.  Features are normalized so that
    every column lives roughly in [-1, 1] — the same pre-processing you would
    apply before feeding into a topological layer.

    Schema
    ------
    patient_id  : str   (e.g. "P0042")
    visit_idx   : int   (0-based visit index within patient)
    hemoglobin, wbc, creatinine, sodium, potassium  : float32  (lab values)
    heart_rate, sbp, dbp, temperature, spo2         : float32  (vitals)
    """
    rng = np.random.default_rng(seed)
    n_features = len(_ALL_FEATURES)

    rows = []
    for patient_idx in range(n_patients):
        # Each patient has a personal "health baseline" + a temporal trajectory.
        baseline = rng.standard_normal(n_features).astype(np.float32) * 0.3
        noise = (
            rng.standard_normal((visits_per_patient, n_features)).astype(np.float32)
            * 0.1
        )
        trajectory = np.cumsum(noise, axis=0)
        visits = baseline + trajectory  # (visits_per_patient, n_features)

        # Normalize to [-1, 1] per-patient (simulates per-patient z-score + clip)
        col_std = visits.std(axis=0) + 1e-8
        visits = (visits - visits.mean(axis=0)) / col_std
        visits = np.clip(visits, -3.0, 3.0) / 3.0

        for visit_idx in range(visits_per_patient):
            row = {"patient_id": f"P{patient_idx:04d}", "visit_idx": visit_idx}
            for col, val in zip(_ALL_FEATURES, visits[visit_idx]):
                row[col] = float(val)
            rows.append(row)

    schema = {
        "patient_id": polars.Utf8,
        "visit_idx": polars.Int32,
        **{col: polars.Float32 for col in _ALL_FEATURES},
    }
    return polars.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    output_shape: tuple
    output_mean: float
    output_std: float


def _run_trailed(
    df: polars.DataFrame,
    coord_columns: list[str],
    group_column: str,
    num_thetas: int,
    resolution: int,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> BenchmarkResult:
    """Benchmark compute_ect_from_polars (vectorised Rust path, no torch needed)."""
    for _ in range(n_warmup):
        compute_ect_from_polars(
            df,
            coord_columns=coord_columns,
            group_column=group_column,
            num_thetas=num_thetas,
            resolution=resolution,
        )

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = compute_ect_from_polars(
            df,
            coord_columns=coord_columns,
            group_column=group_column,
            num_thetas=num_thetas,
            resolution=resolution,
        )
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    return BenchmarkResult(
        name="trailed (Polars → Rust)",
        time_ms=avg_ms,
        output_shape=tuple(out.shape),
        output_mean=float(out.mean()),
        output_std=float(out.std()),
    )


def _polars_to_torch(
    df: polars.DataFrame,
    coord_columns: list[str],
    group_column: str,
) -> tuple:
    """
    Extract (x, batch) torch tensors from a Polars DataFrame.

    This is the glue code a user must write when using the upstream dect
    package, which has no DataFrame integration.
    """
    points = torch.from_numpy(
        df.select(coord_columns).to_numpy().astype(np.float32)
    )

    patient_series = df.get_column(group_column)
    unique_patients = patient_series.unique().sort().to_list()
    group_map = {g: i for i, g in enumerate(unique_patients)}
    batch = torch.tensor(
        [group_map[g] for g in patient_series.to_list()], dtype=torch.long
    )
    return points, batch


def _run_upstream(
    df: polars.DataFrame,
    coord_columns: list[str],
    group_column: str,
    num_thetas: int,
    resolution: int,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> BenchmarkResult:
    """
    Benchmark upstream dect package, including the Polars→torch conversion.

    The conversion is included because it is mandatory overhead — upstream has
    no DataFrame API and every real-world run starts from a DataFrame.
    """
    ambient_dim = len(coord_columns)
    v = upstream_generate_directions(num_thetas, ambient_dim, seed=42, device="cpu")
    config = UpstreamECTConfig(
        resolution=resolution,
        scale=500.0,
        radius=1.1,
        ect_type="points",
        normalized=False,
        fixed=True,
    )
    layer = UpstreamECTLayer(config, v=v)

    class MockBatch:
        def __init__(self, x, batch):
            self.x = x
            self.batch = batch

    # Warmup
    for _ in range(n_warmup):
        x, batch = _polars_to_torch(df, coord_columns, group_column)
        layer(MockBatch(x, batch))

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        # Measure from Polars DataFrame — the realistic starting point
        x, batch = _polars_to_torch(df, coord_columns, group_column)
        out = layer(MockBatch(x, batch))
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    out_np = out.detach().numpy()
    return BenchmarkResult(
        name="dect upstream (Polars → torch → ECTLayer)",
        time_ms=avg_ms,
        output_shape=tuple(out_np.shape),
        output_mean=float(out_np.mean()),
        output_std=float(out_np.std()),
    )


def _print_comparison(
    trailed: BenchmarkResult,
    upstream: BenchmarkResult,
    title: str,
) -> None:
    w = 76
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")
    print(f"{'Metric':<30} {'trailed (Rust)':<22} {'dect (upstream)':<22}")
    print(f"{'-' * w}")
    print(
        f"{'Total wall-clock (ms)':<30} {trailed.time_ms:<22.2f} {upstream.time_ms:<22.2f}"
    )
    print(f"{'-' * w}")
    speedup = upstream.time_ms / trailed.time_ms
    if speedup >= 1:
        print(f"  trailed is {speedup:.2f}x faster")
    else:
        print(f"  upstream is {1/speedup:.2f}x faster")
    print(
        f"\n  Output shape : trailed={trailed.output_shape}, upstream={upstream.output_shape}"
    )
    print(
        f"  Output mean  : trailed={trailed.output_mean:.4f}, upstream={upstream.output_mean:.4f}"
    )
    print(
        f"  Output std   : trailed={trailed.output_std:.4f}, upstream={upstream.output_std:.4f}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPolarsEHRWorkflow:
    """Correctness tests for the Polars → ECT workflow."""

    @pytest.fixture
    def small_df(self):
        return generate_ehr_polars(n_patients=10, visits_per_patient=30)

    def test_smoke(self, small_df):
        """ECT runs end-to-end on an EHR Polars DataFrame."""
        ect = compute_ect_from_polars(
            small_df,
            coord_columns=_ALL_FEATURES,
            group_column="patient_id",
            num_thetas=16,
            resolution=16,
        )
        assert ect.shape == (10, 16, 16), f"Unexpected shape: {ect.shape}"
        assert np.isfinite(ect).all(), "ECT contains non-finite values"

    def test_monotonicity(self, small_df):
        """ECT values should be non-decreasing along the threshold axis."""
        ect = compute_ect_from_polars(
            small_df,
            coord_columns=_ALL_FEATURES,
            group_column="patient_id",
            num_thetas=16,
            resolution=32,
        )
        diff = np.diff(ect, axis=1)
        assert (diff >= -1e-4).all(), "ECT is not monotonic along threshold axis"

    def test_polars_matches_numpy(self, small_df):
        """Polars path and direct numpy path produce identical results."""
        num_thetas, resolution = 16, 16

        ect_polars = compute_ect_from_polars(
            small_df,
            coord_columns=_ALL_FEATURES,
            group_column="patient_id",
            num_thetas=num_thetas,
            resolution=resolution,
            seed=0,
        )

        points = small_df.select(_ALL_FEATURES).to_numpy().astype(np.float32)
        patient_series = small_df.get_column("patient_id")
        unique_patients = patient_series.unique().sort()
        group_map = {g: i for i, g in enumerate(unique_patients.to_list())}
        group_ids = np.array(
            [group_map[g] for g in patient_series.to_list()], dtype=np.int64
        )
        ect_numpy = compute_ect_from_numpy(
            points,
            group_ids=group_ids,
            num_thetas=num_thetas,
            resolution=resolution,
            seed=0,
        )

        np.testing.assert_allclose(
            ect_polars, ect_numpy, rtol=1e-5, atol=1e-6,
            err_msg="Polars and numpy paths diverged",
        )

    def test_output_shape_subset_features(self, small_df):
        """Works with a subset of features (e.g., labs only)."""
        ect = compute_ect_from_polars(
            small_df,
            coord_columns=_LAB_COLUMNS,
            group_column="patient_id",
            num_thetas=8,
            resolution=8,
        )
        assert ect.shape == (10, 8, 8)

    def test_single_patient(self):
        """Single-patient DataFrame returns a (1, R, T) array."""
        df = generate_ehr_polars(n_patients=1, visits_per_patient=40)
        ect = compute_ect_from_polars(
            df,
            coord_columns=_ALL_FEATURES,
            group_column="patient_id",
            num_thetas=16,
            resolution=16,
        )
        assert ect.shape == (1, 16, 16)

    def test_patient_fingerprints_differ(self, small_df):
        """Different patients should produce different ECT fingerprints."""
        ect = compute_ect_from_polars(
            small_df,
            coord_columns=_ALL_FEATURES,
            group_column="patient_id",
            num_thetas=16,
            resolution=16,
        )
        diff = np.abs(ect[0] - ect[1]).max()
        assert diff > 1e-3, "Patient ECTs are suspiciously identical"

    @pytest.mark.skipif(
        not (HAS_UPSTREAM and HAS_TORCH),
        reason="upstream dect + torch not installed",
    )
    def test_output_correlation_with_upstream(self):
        """
        Trailed and upstream outputs are highly correlated when given the same
        directions, confirming they compute the same quantity.
        """
        df = generate_ehr_polars(n_patients=5, visits_per_patient=20)
        num_thetas, resolution = 16, 16
        ambient_dim = len(_ALL_FEATURES)

        v_np = upstream_generate_directions(
            num_thetas, ambient_dim, seed=42, device="cpu"
        )
        v_torch = v_np if isinstance(v_np, torch.Tensor) else torch.from_numpy(v_np)

        ect_trailed = compute_ect_from_polars(
            df,
            coord_columns=_ALL_FEATURES,
            group_column="patient_id",
            num_thetas=num_thetas,
            resolution=resolution,
            directions=v_np.numpy() if hasattr(v_np, "numpy") else np.array(v_np),
            seed=42,
        )

        x, batch = _polars_to_torch(df, _ALL_FEATURES, "patient_id")
        config = UpstreamECTConfig(
            resolution=resolution,
            scale=500.0,
            radius=1.1,
            ect_type="points",
        )
        layer = UpstreamECTLayer(config, v=v_torch)

        class MockBatch:
            def __init__(self, x, batch):
                self.x = x
                self.batch = batch

        ect_upstream = layer(MockBatch(x, batch)).detach().numpy()

        flat_t = ect_trailed.flatten()
        flat_u = ect_upstream.flatten()
        correlation = np.corrcoef(flat_t, flat_u)[0, 1]

        print(f"\nOutput correlation (same directions): {correlation:.4f}")
        print(
            f"trailed shape: {ect_trailed.shape}, upstream shape: {ect_upstream.shape}"
        )
        assert correlation > 0.99, f"Output correlation too low: {correlation:.4f}"


_UPSTREAM_SKIP = pytest.mark.skipif(
    not (HAS_UPSTREAM and HAS_TORCH),
    reason="upstream dect + torch not installed",
)


class TestPolarsEHRBenchmark:
    """
    Wall-clock comparisons: trailed (Polars → Rust) vs. upstream dect package.

    The upstream timing includes the mandatory Polars → torch tensor conversion,
    because that overhead is real and unavoidable — the upstream package has no
    DataFrame integration.
    """

    @_UPSTREAM_SKIP
    def test_benchmark_small(self):
        """Small cohort: 20 patients × 50 visits (1 000 points)."""
        df = generate_ehr_polars(n_patients=20, visits_per_patient=50)

        trailed = _run_trailed(df, _ALL_FEATURES, "patient_id", num_thetas=32, resolution=32)
        upstream = _run_upstream(df, _ALL_FEATURES, "patient_id", num_thetas=32, resolution=32)

        _print_comparison(trailed, upstream, "Small cohort  — 20 patients × 50 visits, 10-D, 32×32")

    @_UPSTREAM_SKIP
    def test_benchmark_medium(self):
        """Medium cohort: 100 patients × 100 visits (10 000 points)."""
        df = generate_ehr_polars(n_patients=100, visits_per_patient=100)

        trailed = _run_trailed(df, _ALL_FEATURES, "patient_id", num_thetas=64, resolution=64)
        upstream = _run_upstream(df, _ALL_FEATURES, "patient_id", num_thetas=64, resolution=64)

        _print_comparison(trailed, upstream, "Medium cohort — 100 patients × 100 visits, 10-D, 64×64")

    @_UPSTREAM_SKIP
    def test_benchmark_large(self):
        """Large cohort: 500 patients × 200 visits (100 000 points)."""
        df = generate_ehr_polars(n_patients=500, visits_per_patient=200)

        trailed = _run_trailed(df, _ALL_FEATURES, "patient_id", num_thetas=64, resolution=64)
        upstream = _run_upstream(df, _ALL_FEATURES, "patient_id", num_thetas=64, resolution=64)

        _print_comparison(trailed, upstream, "Large cohort  — 500 patients × 200 visits, 10-D, 64×64")

    @_UPSTREAM_SKIP
    def test_benchmark_labs_only(self):
        """5-D lab-only feature set."""
        df = generate_ehr_polars(n_patients=100, visits_per_patient=100)

        trailed = _run_trailed(df, _LAB_COLUMNS, "patient_id", num_thetas=32, resolution=32)
        upstream = _run_upstream(df, _LAB_COLUMNS, "patient_id", num_thetas=32, resolution=32)

        _print_comparison(trailed, upstream, "Labs only     — 100 patients × 100 visits,  5-D, 32×32")

    @_UPSTREAM_SKIP
    def test_benchmark_high_resolution(self):
        """High-resolution ECT on a mid-sized cohort."""
        df = generate_ehr_polars(n_patients=50, visits_per_patient=100)

        trailed = _run_trailed(df, _ALL_FEATURES, "patient_id", num_thetas=128, resolution=128)
        upstream = _run_upstream(df, _ALL_FEATURES, "patient_id", num_thetas=128, resolution=128)

        _print_comparison(trailed, upstream, "High-res      — 50 patients × 100 visits, 10-D, 128×128")


def main():
    """Entry point for `trailed-benchmark` CLI and `uv run benchmark`."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
