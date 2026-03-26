use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
            PyArray1, PyArray2, PyArray3, PyArray4, IntoPyArray, PyUntypedArrayMethods};
use ndarray::{Array1, Array2, Array3, Array4};

mod directions;
mod ect;
mod parallel;

// ============================================================================
// Direction Generation Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (num_thetas, d, seed))]
fn generate_uniform_directions<'py>(
    py: Python<'py>,
    num_thetas: usize,
    d: usize,
    seed: u64,
) -> Bound<'py, PyArray2<f32>> {
    let v = directions::generate_uniform_directions(num_thetas, d, seed);
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (num_thetas))]
fn generate_2d_directions<'py>(
    py: Python<'py>,
    num_thetas: usize,
) -> Bound<'py, PyArray2<f32>> {
    let v = directions::generate_2d_directions(num_thetas);
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (num_thetas, d))]
fn generate_multiview_directions<'py>(
    py: Python<'py>,
    num_thetas: usize,
    d: usize,
) -> Bound<'py, PyArray2<f32>> {
    let v = directions::generate_multiview_directions(num_thetas, d);
    v.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (num_thetas, num_phis))]
fn generate_spherical_grid_directions<'py>(
    py: Python<'py>,
    num_thetas: usize,
    num_phis: usize,
) -> Bound<'py, PyArray2<f32>> {
    let v = directions::generate_spherical_grid_directions(num_thetas, num_phis);
    v.into_pyarray(py)
}

// ============================================================================
// ECT Computation Functions - Points
// ============================================================================

#[pyfunction]
#[pyo3(signature = (nh, batch, lin, dim_size, scale=50.0))]
fn compute_ect_points_forward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
    scale: f32,
) -> Bound<'py, PyArray3<f32>> {
    let out = ect::compute_ect_points_forward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &lin.as_array().to_owned(),
        dim_size,
        scale,
    );
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, lin, grad_output, scale=50.0))]
fn compute_ect_points_backward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray3<f32>,
    scale: f32,
) -> Bound<'py, PyArray2<f32>> {
    let grad_nh = ect::compute_ect_points_backward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &lin.as_array().to_owned(),
        &grad_output.as_array().to_owned(),
        scale,
    );
    grad_nh.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, lin, dim_size, scale=100.0))]
fn compute_ect_points_derivative_forward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
    scale: f32,
) -> Bound<'py, PyArray3<f32>> {
    let out = ect::compute_ect_points_derivative_forward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &lin.as_array().to_owned(),
        dim_size,
        scale,
    );
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, lin, grad_output, scale=100.0))]
fn compute_ect_points_derivative_backward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray3<f32>,
    scale: f32,
) -> Bound<'py, PyArray2<f32>> {
    let grad_nh = ect::compute_ect_points_derivative_backward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &lin.as_array().to_owned(),
        &grad_output.as_array().to_owned(),
        scale,
    );
    grad_nh.into_pyarray(py)
}

// ============================================================================
// ECT Computation Functions - Edges
// ============================================================================

#[pyfunction]
#[pyo3(signature = (nh, batch, edge_index, lin, dim_size, scale=50.0))]
fn compute_ect_edges_forward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    edge_index: PyReadonlyArray2<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
    scale: f32,
) -> Bound<'py, PyArray3<f32>> {
    let out = ect::compute_ect_edges_forward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &edge_index.as_array().to_owned(),
        &lin.as_array().to_owned(),
        dim_size,
        scale,
    );
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, edge_index, lin, grad_output, scale=50.0))]
fn compute_ect_edges_backward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    edge_index: PyReadonlyArray2<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray3<f32>,
    scale: f32,
) -> Bound<'py, PyArray2<f32>> {
    let grad_nh = ect::compute_ect_edges_backward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &edge_index.as_array().to_owned(),
        &lin.as_array().to_owned(),
        &grad_output.as_array().to_owned(),
        scale,
    );
    grad_nh.into_pyarray(py)
}

// ============================================================================
// ECT Computation Functions - Faces
// ============================================================================

#[pyfunction]
#[pyo3(signature = (nh, batch, edge_index, face, lin, dim_size, scale=50.0))]
fn compute_ect_faces_forward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    edge_index: PyReadonlyArray2<i64>,
    face: PyReadonlyArray2<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
    scale: f32,
) -> Bound<'py, PyArray3<f32>> {
    let out = ect::compute_ect_faces_forward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &edge_index.as_array().to_owned(),
        &face.as_array().to_owned(),
        &lin.as_array().to_owned(),
        dim_size,
        scale,
    );
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, edge_index, face, lin, grad_output, scale=50.0))]
fn compute_ect_faces_backward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    edge_index: PyReadonlyArray2<i64>,
    face: PyReadonlyArray2<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray3<f32>,
    scale: f32,
) -> Bound<'py, PyArray2<f32>> {
    let grad_nh = ect::compute_ect_faces_backward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &edge_index.as_array().to_owned(),
        &face.as_array().to_owned(),
        &lin.as_array().to_owned(),
        &grad_output.as_array().to_owned(),
        scale,
    );
    grad_nh.into_pyarray(py)
}

// ============================================================================
// ECT Computation Functions - Channels
// ============================================================================

#[pyfunction]
#[pyo3(signature = (nh, batch, channels, lin, dim_size, max_channels, scale=500.0))]
fn compute_ect_channels_forward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    channels: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
    max_channels: usize,
    scale: f32,
) -> Bound<'py, PyArray4<f32>> {
    let out = ect::compute_ect_channels_forward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &channels.as_array().to_owned(),
        &lin.as_array().to_owned(),
        dim_size,
        max_channels,
        scale,
    );
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, channels, lin, grad_output, scale=500.0))]
fn compute_ect_channels_backward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    channels: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray4<f32>,
    scale: f32,
) -> Bound<'py, PyArray2<f32>> {
    let grad_nh = ect::compute_ect_channels_backward(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &channels.as_array().to_owned(),
        &lin.as_array().to_owned(),
        &grad_output.as_array().to_owned(),
        scale,
    );
    grad_nh.into_pyarray(py)
}

// ============================================================================
// Fast ECT (Non-differentiable)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (nh, resolution))]
fn compute_fast_ect<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    resolution: usize,
) -> Bound<'py, PyArray2<f32>> {
    let out = ect::compute_fast_ect(&nh.as_array().to_owned(), resolution);
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, dim_size, resolution))]
fn compute_fast_ect_batched<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    dim_size: usize,
    resolution: usize,
) -> Bound<'py, PyArray3<f32>> {
    let out = ect::compute_fast_ect_batched(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        dim_size,
        resolution,
    );
    out.into_pyarray(py)
}

// ============================================================================
// Parallel ECT Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (nh, batch, lin, dim_size, scale=50.0))]
fn compute_ect_points_forward_parallel<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
    scale: f32,
) -> Bound<'py, PyArray3<f32>> {
    let out = parallel::compute_ecc_forward_parallel(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &lin.as_array().to_owned(),
        dim_size,
        scale,
    );
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, lin, grad_output, scale=50.0))]
fn compute_ect_points_backward_parallel<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray3<f32>,
    scale: f32,
) -> Bound<'py, PyArray2<f32>> {
    let grad_nh = parallel::compute_ecc_backward_parallel(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &lin.as_array().to_owned(),
        &grad_output.as_array().to_owned(),
        scale,
    );
    grad_nh.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, resolution))]
fn compute_fast_ect_parallel<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    resolution: usize,
) -> Bound<'py, PyArray2<f32>> {
    let out = parallel::compute_fast_ect_parallel(&nh.as_array().to_owned(), resolution);
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, dim_size, resolution))]
fn compute_fast_ect_batched_parallel<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    dim_size: usize,
    resolution: usize,
) -> Bound<'py, PyArray3<f32>> {
    let out = parallel::compute_fast_ect_batched_parallel(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        dim_size,
        resolution,
    );
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (nh, batch, channels, lin, dim_size, max_channels, scale=500.0))]
fn compute_ect_channels_forward_parallel<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    channels: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
    max_channels: usize,
    scale: f32,
) -> Bound<'py, PyArray4<f32>> {
    let out = parallel::compute_ect_channels_forward_parallel(
        &nh.as_array().to_owned(),
        &batch.as_array().to_owned(),
        &channels.as_array().to_owned(),
        &lin.as_array().to_owned(),
        dim_size,
        max_channels,
        scale,
    );
    out.into_pyarray(py)
}

// ============================================================================
// Utility Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (radius, resolution))]
fn generate_lin<'py>(
    py: Python<'py>,
    radius: f32,
    resolution: usize,
) -> Bound<'py, PyArray1<f32>> {
    let lin = ect::generate_lin(radius, resolution);
    lin.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (x, v))]
fn compute_node_heights<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let nh = ect::compute_node_heights(&x.as_array().to_owned(), &v.as_array().to_owned());
    nh.into_pyarray(py)
}

// ============================================================================
// Python Module Definition
// ============================================================================

#[pymodule]
fn dect_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Direction generation
    m.add_function(wrap_pyfunction!(generate_uniform_directions, m)?)?;
    m.add_function(wrap_pyfunction!(generate_2d_directions, m)?)?;
    m.add_function(wrap_pyfunction!(generate_multiview_directions, m)?)?;
    m.add_function(wrap_pyfunction!(generate_spherical_grid_directions, m)?)?;
    
    // ECT - Points
    m.add_function(wrap_pyfunction!(compute_ect_points_forward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_points_backward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_points_derivative_forward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_points_derivative_backward, m)?)?;
    
    // ECT - Edges
    m.add_function(wrap_pyfunction!(compute_ect_edges_forward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_edges_backward, m)?)?;
    
    // ECT - Faces
    m.add_function(wrap_pyfunction!(compute_ect_faces_forward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_faces_backward, m)?)?;
    
    // ECT - Channels
    m.add_function(wrap_pyfunction!(compute_ect_channels_forward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_channels_backward, m)?)?;
    
    // Fast ECT
    m.add_function(wrap_pyfunction!(compute_fast_ect, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fast_ect_batched, m)?)?;
    
    // Parallel versions
    m.add_function(wrap_pyfunction!(compute_ect_points_forward_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_points_backward_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fast_ect_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fast_ect_batched_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_channels_forward_parallel, m)?)?;
    
    // Utilities
    m.add_function(wrap_pyfunction!(generate_lin, m)?)?;
    m.add_function(wrap_pyfunction!(compute_node_heights, m)?)?;
    
    Ok(())
}
