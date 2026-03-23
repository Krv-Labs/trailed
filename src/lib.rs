use pyo3::prelude::*;
use numpy::{PyReadonlyArrayDyn, PyReadonlyArray3, PyReadonlyArray2, PyReadonlyArray1, PyArrayDyn, PyArray2, PyArray3, IntoPyArray, PyUntypedArrayMethods};
use ndarray::{Array1, Array2, Array3};

pub fn compute_ecc_forward_loop(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    dim_size: usize,
    ecc_factor: f32,
) -> Array3<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];
    let b = lin.len();
    
    let mut out = Array3::<f32>::zeros((dim_size, b, t));
    
    for i in 0..n {
        let graph_idx = batch[i] as usize;
        for j in 0..t {
            let nh_val = nh[[i, j]];
            for k in 0..b {
                let lin_val = lin[k];
                let exp_val = (-ecc_factor * (lin_val - nh_val)).exp();
                let ecc_val = 1.0 / (1.0 + exp_val);
                out[[graph_idx, k, j]] += ecc_val;
            }
        }
    }
    out
}

pub fn compute_ecc_backward_loop(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    grad_output: &Array3<f32>,
    ecc_factor: f32,
) -> Array2<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];
    let b = lin.len();
    
    let mut grad_nh = Array2::<f32>::zeros((n, t));
    
    for i in 0..n {
        let graph_idx = batch[i] as usize;
        for j in 0..t {
            let nh_val = nh[[i, j]];
            let mut grad_sum = 0.0;
            for k in 0..b {
                let lin_val = lin[k];
                let exp_val = (-ecc_factor * (lin_val - nh_val)).exp();
                let s = 1.0 / (1.0 + exp_val);
                
                let decc_dnh = s * (1.0 - s) * (-ecc_factor);
                let dout = grad_output[[graph_idx, k, j]];
                grad_sum += dout * decc_dnh;
            }
            grad_nh[[i, j]] += grad_sum;
        }
    }
    grad_nh
}

// Points
#[pyfunction]
fn compute_ect_points_forward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
) -> Bound<'py, PyArray3<f32>> {
    let out = compute_ecc_forward_loop(&nh.as_array().to_owned(), &batch.as_array().to_owned(), &lin.as_array().to_owned(), dim_size, 50.0);
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn compute_ect_points_backward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray3<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let grad_nh = compute_ecc_backward_loop(&nh.as_array().to_owned(), &batch.as_array().to_owned(), &lin.as_array().to_owned(), &grad_output.as_array().to_owned(), 50.0);
    grad_nh.into_pyarray_bound(py)
}

// Points Derivative
#[pyfunction]
fn compute_ect_points_derivative_forward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
) -> Bound<'py, PyArray3<f32>> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];
    let b = lin.as_array().len();
    let nh_arr = nh.as_array();
    let batch_arr = batch.as_array();
    let lin_arr = lin.as_array();
    
    let mut out = Array3::<f32>::zeros((dim_size, b, t));
    for i in 0..n {
        let graph_idx = batch_arr[i] as usize;
        for j in 0..t {
            let nh_val = nh_arr[[i, j]];
            for k in 0..b {
                let lin_val = lin_arr[k];
                let exp_val = (-100.0 * (lin_val - nh_val)).exp();
                let s = 1.0 / (1.0 + exp_val);
                let ecc_val = s * (1.0 - s);
                out[[graph_idx, k, j]] += ecc_val;
            }
        }
    }
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn compute_ect_points_derivative_backward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray3<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];
    let b = lin.as_array().len();
    let nh_arr = nh.as_array();
    let batch_arr = batch.as_array();
    let lin_arr = lin.as_array();
    let grad_output_arr = grad_output.as_array();
    
    let mut grad_nh = Array2::<f32>::zeros((n, t));
    for i in 0..n {
        let graph_idx = batch_arr[i] as usize;
        for j in 0..t {
            let nh_val = nh_arr[[i, j]];
            let mut grad_sum = 0.0;
            for k in 0..b {
                let lin_val = lin_arr[k];
                let exp_val = (-100.0 * (lin_val - nh_val)).exp();
                let s = 1.0 / (1.0 + exp_val);
                let ds_dnh = s * (1.0 - s) * (-100.0);
                let decc_dnh = (1.0 - 2.0 * s) * ds_dnh;
                let dout = grad_output_arr[[graph_idx, k, j]];
                grad_sum += dout * decc_dnh;
            }
            grad_nh[[i, j]] += grad_sum;
        }
    }
    grad_nh.into_pyarray_bound(py)
}

// Edges
#[pyfunction]
fn compute_ect_edges_forward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    edge_index: PyReadonlyArray2<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
) -> Bound<'py, PyArray3<f32>> {
    let nh_arr = nh.as_array().to_owned();
    let batch_arr = batch.as_array().to_owned();
    let edge_index_arr = edge_index.as_array().to_owned();
    let lin_arr = lin.as_array().to_owned();
    
    let t = nh_arr.shape()[1];
    let e = edge_index_arr.shape()[1];
    
    let term1 = compute_ecc_forward_loop(&nh_arr, &batch_arr, &lin_arr, dim_size, 50.0);
    
    let mut eh = Array2::<f32>::zeros((e, t));
    let mut batch_e = Array1::<i64>::zeros(e);
    
    for i in 0..e {
        let u = edge_index_arr[[0, i]] as usize;
        let w = edge_index_arr[[1, i]] as usize;
        batch_e[i] = batch_arr[u];
        
        for j in 0..t {
            let val_u = nh_arr[[u, j]];
            let val_w = nh_arr[[w, j]];
            eh[[i, j]] = if val_u > val_w { val_u } else { val_w };
        }
    }
    
    let term2 = compute_ecc_forward_loop(&eh, &batch_e, &lin_arr, dim_size, 50.0);
    let out = term1 - term2;
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn compute_ect_edges_backward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    edge_index: PyReadonlyArray2<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray3<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let nh_arr = nh.as_array().to_owned();
    let batch_arr = batch.as_array().to_owned();
    let edge_index_arr = edge_index.as_array().to_owned();
    let lin_arr = lin.as_array().to_owned();
    let grad_output_arr = grad_output.as_array().to_owned();
    
    let t = nh_arr.shape()[1];
    let e = edge_index_arr.shape()[1];
    
    let mut grad_nh = compute_ecc_backward_loop(&nh_arr, &batch_arr, &lin_arr, &grad_output_arr, 50.0);
    
    let mut eh = Array2::<f32>::zeros((e, t));
    let mut argmax_e = Array2::<usize>::zeros((e, t));
    let mut batch_e = Array1::<i64>::zeros(e);
    
    for i in 0..e {
        let u = edge_index_arr[[0, i]] as usize;
        let w = edge_index_arr[[1, i]] as usize;
        batch_e[i] = batch_arr[u];
        
        for j in 0..t {
            let val_u = nh_arr[[u, j]];
            let val_w = nh_arr[[w, j]];
            if val_u > val_w {
                eh[[i, j]] = val_u;
                argmax_e[[i, j]] = u;
            } else {
                eh[[i, j]] = val_w;
                argmax_e[[i, j]] = w;
            }
        }
    }
    
    let neg_grad_output = -grad_output_arr;
    let grad_eh = compute_ecc_backward_loop(&eh, &batch_e, &lin_arr, &neg_grad_output, 50.0);
    
    for i in 0..e {
        for j in 0..t {
            let winning_node = argmax_e[[i, j]];
            grad_nh[[winning_node, j]] += grad_eh[[i, j]];
        }
    }
    
    grad_nh.into_pyarray_bound(py)
}

// Faces
#[pyfunction]
fn compute_ect_faces_forward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    edge_index: PyReadonlyArray2<i64>,
    face: PyReadonlyArray2<i64>,
    lin: PyReadonlyArray1<f32>,
    dim_size: usize,
) -> Bound<'py, PyArray3<f32>> {
    let nh_arr = nh.as_array().to_owned();
    let batch_arr = batch.as_array().to_owned();
    let edge_index_arr = edge_index.as_array().to_owned();
    let face_arr = face.as_array().to_owned();
    let lin_arr = lin.as_array().to_owned();
    
    let t = nh_arr.shape()[1];
    let e = edge_index_arr.shape()[1];
    let f_count = face_arr.shape()[1];
    
    let term1 = compute_ecc_forward_loop(&nh_arr, &batch_arr, &lin_arr, dim_size, 50.0);
    
    let mut eh = Array2::<f32>::zeros((e, t));
    let mut batch_e = Array1::<i64>::zeros(e);
    for i in 0..e {
        let u = edge_index_arr[[0, i]] as usize;
        let w = edge_index_arr[[1, i]] as usize;
        batch_e[i] = batch_arr[u];
        for j in 0..t {
            let val_u = nh_arr[[u, j]];
            let val_w = nh_arr[[w, j]];
            eh[[i, j]] = if val_u > val_w { val_u } else { val_w };
        }
    }
    let term2 = compute_ecc_forward_loop(&eh, &batch_e, &lin_arr, dim_size, 50.0);
    
    let mut fh = Array2::<f32>::zeros((f_count, t));
    let mut batch_f = Array1::<i64>::zeros(f_count);
    for i in 0..f_count {
        let u = face_arr[[0, i]] as usize;
        let v_node = face_arr[[1, i]] as usize;
        let w = face_arr[[2, i]] as usize;
        batch_f[i] = batch_arr[u];
        for j in 0..t {
            let val_u = nh_arr[[u, j]];
            let val_v = nh_arr[[v_node, j]];
            let val_w = nh_arr[[w, j]];
            let mut max_val = val_u;
            if val_v > max_val { max_val = val_v; }
            if val_w > max_val { max_val = val_w; }
            fh[[i, j]] = max_val;
        }
    }
    let term3 = compute_ecc_forward_loop(&fh, &batch_f, &lin_arr, dim_size, 50.0);
    
    let out = term1 - term2 + term3;
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn compute_ect_faces_backward<'py>(
    py: Python<'py>,
    nh: PyReadonlyArray2<f32>,
    batch: PyReadonlyArray1<i64>,
    edge_index: PyReadonlyArray2<i64>,
    face: PyReadonlyArray2<i64>,
    lin: PyReadonlyArray1<f32>,
    grad_output: PyReadonlyArray3<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let nh_arr = nh.as_array().to_owned();
    let batch_arr = batch.as_array().to_owned();
    let edge_index_arr = edge_index.as_array().to_owned();
    let face_arr = face.as_array().to_owned();
    let lin_arr = lin.as_array().to_owned();
    let grad_output_arr = grad_output.as_array().to_owned();
    
    let t = nh_arr.shape()[1];
    let e = edge_index_arr.shape()[1];
    let f_count = face_arr.shape()[1];
    
    let mut grad_nh = compute_ecc_backward_loop(&nh_arr, &batch_arr, &lin_arr, &grad_output_arr, 50.0);
    
    let mut eh = Array2::<f32>::zeros((e, t));
    let mut argmax_e = Array2::<usize>::zeros((e, t));
    let mut batch_e = Array1::<i64>::zeros(e);
    for i in 0..e {
        let u = edge_index_arr[[0, i]] as usize;
        let w = edge_index_arr[[1, i]] as usize;
        batch_e[i] = batch_arr[u];
        for j in 0..t {
            let val_u = nh_arr[[u, j]];
            let val_w = nh_arr[[w, j]];
            if val_u > val_w {
                eh[[i, j]] = val_u;
                argmax_e[[i, j]] = u;
            } else {
                eh[[i, j]] = val_w;
                argmax_e[[i, j]] = w;
            }
        }
    }
    let neg_grad_output = -grad_output_arr.clone();
    let grad_eh = compute_ecc_backward_loop(&eh, &batch_e, &lin_arr, &neg_grad_output, 50.0);
    for i in 0..e {
        for j in 0..t {
            let winning_node = argmax_e[[i, j]];
            grad_nh[[winning_node, j]] += grad_eh[[i, j]];
        }
    }
    
    let mut fh = Array2::<f32>::zeros((f_count, t));
    let mut argmax_f = Array2::<usize>::zeros((f_count, t));
    let mut batch_f = Array1::<i64>::zeros(f_count);
    for i in 0..f_count {
        let u = face_arr[[0, i]] as usize;
        let v_node = face_arr[[1, i]] as usize;
        let w = face_arr[[2, i]] as usize;
        batch_f[i] = batch_arr[u];
        for j in 0..t {
            let val_u = nh_arr[[u, j]];
            let val_v = nh_arr[[v_node, j]];
            let val_w = nh_arr[[w, j]];
            
            let mut max_val = val_u;
            let mut max_node = u;
            
            if val_v > max_val {
                max_val = val_v;
                max_node = v_node;
            }
            if val_w > max_val {
                max_val = val_w;
                max_node = w;
            }
            
            fh[[i, j]] = max_val;
            argmax_f[[i, j]] = max_node;
        }
    }
    
    let grad_fh = compute_ecc_backward_loop(&fh, &batch_f, &lin_arr, &grad_output_arr, 50.0);
    for i in 0..f_count {
        for j in 0..t {
            let winning_node = argmax_f[[i, j]];
            grad_nh[[winning_node, j]] += grad_fh[[i, j]];
        }
    }
    
    grad_nh.into_pyarray_bound(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn dect_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_ect_points_forward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_points_backward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_points_derivative_forward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_points_derivative_backward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_edges_forward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_edges_backward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_faces_forward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ect_faces_backward, m)?)?;
    Ok(())
}
