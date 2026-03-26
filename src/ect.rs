use ndarray::{Array1, Array2, Array3, Array4, Axis};

/// Compute the sigmoid function: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the Euler Characteristic Curve (ECC) for points.
///
/// This is the core forward computation that computes the smooth ECT
/// using a sigmoid approximation of the indicator function.
///
/// # Arguments
/// * `nh` - Node heights of shape [num_points, num_directions]
/// * `batch` - Batch indices of shape [num_points]
/// * `lin` - Linear threshold values of shape [resolution]
/// * `dim_size` - Number of batches
/// * `scale` - Scale factor for sigmoid (higher = sharper approximation)
///
/// # Returns
/// ECT of shape [dim_size, resolution, num_directions]
pub fn compute_ecc_forward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    dim_size: usize,
    scale: f32,
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
                let ecc_val = sigmoid(scale * (lin_val - nh_val));
                out[[graph_idx, k, j]] += ecc_val;
            }
        }
    }
    out
}

/// Compute the backward pass for ECC.
///
/// # Arguments
/// * `nh` - Node heights of shape [num_points, num_directions]
/// * `batch` - Batch indices of shape [num_points]
/// * `lin` - Linear threshold values of shape [resolution]
/// * `grad_output` - Gradient from upstream of shape [dim_size, resolution, num_directions]
/// * `scale` - Scale factor for sigmoid
///
/// # Returns
/// Gradient w.r.t. node heights of shape [num_points, num_directions]
pub fn compute_ecc_backward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    grad_output: &Array3<f32>,
    scale: f32,
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
                let s = sigmoid(scale * (lin_val - nh_val));
                let decc_dnh = s * (1.0 - s) * (-scale);
                let dout = grad_output[[graph_idx, k, j]];
                grad_sum += dout * decc_dnh;
            }
            grad_nh[[i, j]] += grad_sum;
        }
    }
    grad_nh
}

/// Compute ECT for point clouds (forward pass).
///
/// # Arguments
/// * `nh` - Node heights of shape [num_points, num_directions]
/// * `batch` - Batch indices of shape [num_points]
/// * `lin` - Linear threshold values of shape [resolution]
/// * `dim_size` - Number of batches
/// * `scale` - Scale factor for sigmoid
///
/// # Returns
/// ECT of shape [dim_size, resolution, num_directions]
pub fn compute_ect_points_forward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    dim_size: usize,
    scale: f32,
) -> Array3<f32> {
    compute_ecc_forward(nh, batch, lin, dim_size, scale)
}

/// Compute ECT for point clouds (backward pass).
pub fn compute_ect_points_backward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    grad_output: &Array3<f32>,
    scale: f32,
) -> Array2<f32> {
    compute_ecc_backward(nh, batch, lin, grad_output, scale)
}

/// Compute derivative ECT for point clouds (forward pass).
///
/// This computes the derivative of the ECT, which uses the derivative
/// of the sigmoid function: s(x) * (1 - s(x)).
pub fn compute_ect_points_derivative_forward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    dim_size: usize,
    scale: f32,
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
                let s = sigmoid(scale * (lin_val - nh_val));
                let ecc_val = s * (1.0 - s);
                out[[graph_idx, k, j]] += ecc_val;
            }
        }
    }
    out
}

/// Compute derivative ECT for point clouds (backward pass).
pub fn compute_ect_points_derivative_backward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    grad_output: &Array3<f32>,
    scale: f32,
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
                let s = sigmoid(scale * (lin_val - nh_val));
                let ds_dnh = s * (1.0 - s) * (-scale);
                let decc_dnh = (1.0 - 2.0 * s) * ds_dnh;
                let dout = grad_output[[graph_idx, k, j]];
                grad_sum += dout * decc_dnh;
            }
            grad_nh[[i, j]] += grad_sum;
        }
    }
    grad_nh
}

/// Compute ECT with channel support (forward pass).
///
/// Allows for categorical channels within the point cloud to be separated
/// into different ECTs.
///
/// # Arguments
/// * `nh` - Node heights of shape [num_points, num_directions]
/// * `batch` - Batch indices of shape [num_points]
/// * `channels` - Channel indices of shape [num_points]
/// * `lin` - Linear threshold values of shape [resolution]
/// * `dim_size` - Number of batches
/// * `max_channels` - Maximum number of channels
/// * `scale` - Scale factor for sigmoid
///
/// # Returns
/// ECT of shape [dim_size, num_directions, resolution, max_channels]
pub fn compute_ect_channels_forward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    channels: &Array1<i64>,
    lin: &Array1<f32>,
    dim_size: usize,
    max_channels: usize,
    scale: f32,
) -> Array4<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];
    let r = lin.len();

    let mut out = Array4::<f32>::zeros((dim_size, t, r, max_channels));

    for i in 0..n {
        let graph_idx = batch[i] as usize;
        let ch = channels[i] as usize;
        for j in 0..t {
            let nh_val = nh[[i, j]];
            for k in 0..r {
                let lin_val = lin[k];
                let ecc_val = sigmoid(scale * (lin_val - nh_val));
                out[[graph_idx, j, k, ch]] += ecc_val;
            }
        }
    }
    out
}

/// Compute ECT with channel support (backward pass).
pub fn compute_ect_channels_backward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    channels: &Array1<i64>,
    lin: &Array1<f32>,
    grad_output: &Array4<f32>,
    scale: f32,
) -> Array2<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];
    let r = lin.len();

    let mut grad_nh = Array2::<f32>::zeros((n, t));

    for i in 0..n {
        let graph_idx = batch[i] as usize;
        let ch = channels[i] as usize;
        for j in 0..t {
            let nh_val = nh[[i, j]];
            let mut grad_sum = 0.0;
            for k in 0..r {
                let lin_val = lin[k];
                let s = sigmoid(scale * (lin_val - nh_val));
                let decc_dnh = s * (1.0 - s) * (-scale);
                let dout = grad_output[[graph_idx, j, k, ch]];
                grad_sum += dout * decc_dnh;
            }
            grad_nh[[i, j]] += grad_sum;
        }
    }
    grad_nh
}

/// Compute fast ECT using bincount (non-differentiable).
///
/// This is a fast implementation for inference that discretizes
/// heights into bins and counts occurrences.
///
/// # Arguments
/// * `nh` - Node heights of shape [num_points, num_directions]
/// * `resolution` - Number of bins
///
/// # Returns
/// ECT of shape [resolution, num_directions]
pub fn compute_fast_ect(nh: &Array2<f32>, resolution: usize) -> Array2<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];

    let mut out = Array2::<f32>::zeros((resolution, t));

    for i in 0..n {
        for j in 0..t {
            let nh_val = nh[[i, j]];
            let bin = (((nh_val + 1.0) * (resolution as f32 / 2.0)) as usize)
                .min(resolution - 1);
            out[[bin, j]] += 1.0;
        }
    }

    // Cumulative sum along resolution axis
    for j in 0..t {
        for k in 1..resolution {
            out[[k, j]] += out[[k - 1, j]];
        }
    }

    out
}

/// Compute fast ECT with batch support (non-differentiable).
///
/// # Arguments
/// * `nh` - Node heights of shape [num_points, num_directions]
/// * `batch` - Batch indices of shape [num_points]
/// * `dim_size` - Number of batches
/// * `resolution` - Number of bins
///
/// # Returns
/// ECT of shape [dim_size, resolution, num_directions]
pub fn compute_fast_ect_batched(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    dim_size: usize,
    resolution: usize,
) -> Array3<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];

    let mut out = Array3::<f32>::zeros((dim_size, resolution, t));

    for i in 0..n {
        let graph_idx = batch[i] as usize;
        for j in 0..t {
            let nh_val = nh[[i, j]];
            let bin = (((nh_val + 1.0) * (resolution as f32 / 2.0)) as usize)
                .min(resolution - 1);
            out[[graph_idx, bin, j]] += 1.0;
        }
    }

    // Cumulative sum along resolution axis
    for b in 0..dim_size {
        for j in 0..t {
            for k in 1..resolution {
                out[[b, k, j]] += out[[b, k - 1, j]];
            }
        }
    }

    out
}

/// Compute ECT for edges (forward pass).
///
/// Computes ECT = ECT(vertices) - ECT(edges)
/// using Euler characteristic formula.
pub fn compute_ect_edges_forward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    edge_index: &Array2<i64>,
    lin: &Array1<f32>,
    dim_size: usize,
    scale: f32,
) -> Array3<f32> {
    let t = nh.shape()[1];
    let e = edge_index.shape()[1];

    // Term 1: vertices contribution
    let term1 = compute_ecc_forward(nh, batch, lin, dim_size, scale);

    // Compute edge heights (max of endpoint heights)
    let mut eh = Array2::<f32>::zeros((e, t));
    let mut batch_e = Array1::<i64>::zeros(e);

    for i in 0..e {
        let u = edge_index[[0, i]] as usize;
        let w = edge_index[[1, i]] as usize;
        batch_e[i] = batch[u];

        for j in 0..t {
            let val_u = nh[[u, j]];
            let val_w = nh[[w, j]];
            eh[[i, j]] = val_u.max(val_w);
        }
    }

    // Term 2: edges contribution (subtracted)
    let term2 = compute_ecc_forward(&eh, &batch_e, lin, dim_size, scale);

    // ECT = vertices - edges
    term1 - term2
}

/// Compute ECT for edges (backward pass).
pub fn compute_ect_edges_backward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    edge_index: &Array2<i64>,
    lin: &Array1<f32>,
    grad_output: &Array3<f32>,
    scale: f32,
) -> Array2<f32> {
    let t = nh.shape()[1];
    let e = edge_index.shape()[1];

    // Gradient from vertices
    let mut grad_nh = compute_ecc_backward(nh, batch, lin, grad_output, scale);

    // Compute edge heights and argmax
    let mut eh = Array2::<f32>::zeros((e, t));
    let mut argmax_e = Array2::<usize>::zeros((e, t));
    let mut batch_e = Array1::<i64>::zeros(e);

    for i in 0..e {
        let u = edge_index[[0, i]] as usize;
        let w = edge_index[[1, i]] as usize;
        batch_e[i] = batch[u];

        for j in 0..t {
            let val_u = nh[[u, j]];
            let val_w = nh[[w, j]];
            if val_u > val_w {
                eh[[i, j]] = val_u;
                argmax_e[[i, j]] = u;
            } else {
                eh[[i, j]] = val_w;
                argmax_e[[i, j]] = w;
            }
        }
    }

    // Gradient from edges (negated because subtracted in forward)
    let neg_grad_output = -grad_output.clone();
    let grad_eh = compute_ecc_backward(&eh, &batch_e, lin, &neg_grad_output, scale);

    // Distribute edge gradients to winning nodes
    for i in 0..e {
        for j in 0..t {
            let winning_node = argmax_e[[i, j]];
            grad_nh[[winning_node, j]] += grad_eh[[i, j]];
        }
    }

    grad_nh
}

/// Compute ECT for faces/meshes (forward pass).
///
/// Computes ECT = ECT(vertices) - ECT(edges) + ECT(faces)
/// using Euler characteristic formula.
pub fn compute_ect_faces_forward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    edge_index: &Array2<i64>,
    face: &Array2<i64>,
    lin: &Array1<f32>,
    dim_size: usize,
    scale: f32,
) -> Array3<f32> {
    let t = nh.shape()[1];
    let e = edge_index.shape()[1];
    let f_count = face.shape()[1];

    // Term 1: vertices
    let term1 = compute_ecc_forward(nh, batch, lin, dim_size, scale);

    // Compute edge heights
    let mut eh = Array2::<f32>::zeros((e, t));
    let mut batch_e = Array1::<i64>::zeros(e);
    for i in 0..e {
        let u = edge_index[[0, i]] as usize;
        let w = edge_index[[1, i]] as usize;
        batch_e[i] = batch[u];
        for j in 0..t {
            eh[[i, j]] = nh[[u, j]].max(nh[[w, j]]);
        }
    }
    let term2 = compute_ecc_forward(&eh, &batch_e, lin, dim_size, scale);

    // Compute face heights
    let mut fh = Array2::<f32>::zeros((f_count, t));
    let mut batch_f = Array1::<i64>::zeros(f_count);
    for i in 0..f_count {
        let u = face[[0, i]] as usize;
        let v = face[[1, i]] as usize;
        let w = face[[2, i]] as usize;
        batch_f[i] = batch[u];
        for j in 0..t {
            fh[[i, j]] = nh[[u, j]].max(nh[[v, j]]).max(nh[[w, j]]);
        }
    }
    let term3 = compute_ecc_forward(&fh, &batch_f, lin, dim_size, scale);

    // ECT = vertices - edges + faces
    term1 - term2 + term3
}

/// Compute ECT for faces/meshes (backward pass).
pub fn compute_ect_faces_backward(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    edge_index: &Array2<i64>,
    face: &Array2<i64>,
    lin: &Array1<f32>,
    grad_output: &Array3<f32>,
    scale: f32,
) -> Array2<f32> {
    let t = nh.shape()[1];
    let e = edge_index.shape()[1];
    let f_count = face.shape()[1];

    // Gradient from vertices
    let mut grad_nh = compute_ecc_backward(nh, batch, lin, grad_output, scale);

    // Edge gradients
    let mut eh = Array2::<f32>::zeros((e, t));
    let mut argmax_e = Array2::<usize>::zeros((e, t));
    let mut batch_e = Array1::<i64>::zeros(e);
    for i in 0..e {
        let u = edge_index[[0, i]] as usize;
        let w = edge_index[[1, i]] as usize;
        batch_e[i] = batch[u];
        for j in 0..t {
            let val_u = nh[[u, j]];
            let val_w = nh[[w, j]];
            if val_u > val_w {
                eh[[i, j]] = val_u;
                argmax_e[[i, j]] = u;
            } else {
                eh[[i, j]] = val_w;
                argmax_e[[i, j]] = w;
            }
        }
    }
    let neg_grad_output = -grad_output.clone();
    let grad_eh = compute_ecc_backward(&eh, &batch_e, lin, &neg_grad_output, scale);
    for i in 0..e {
        for j in 0..t {
            grad_nh[[argmax_e[[i, j]], j]] += grad_eh[[i, j]];
        }
    }

    // Face gradients
    let mut fh = Array2::<f32>::zeros((f_count, t));
    let mut argmax_f = Array2::<usize>::zeros((f_count, t));
    let mut batch_f = Array1::<i64>::zeros(f_count);
    for i in 0..f_count {
        let u = face[[0, i]] as usize;
        let v = face[[1, i]] as usize;
        let w = face[[2, i]] as usize;
        batch_f[i] = batch[u];
        for j in 0..t {
            let val_u = nh[[u, j]];
            let val_v = nh[[v, j]];
            let val_w = nh[[w, j]];

            let mut max_val = val_u;
            let mut max_node = u;
            if val_v > max_val {
                max_val = val_v;
                max_node = v;
            }
            if val_w > max_val {
                max_val = val_w;
                max_node = w;
            }
            fh[[i, j]] = max_val;
            argmax_f[[i, j]] = max_node;
        }
    }
    let grad_fh = compute_ecc_backward(&fh, &batch_f, lin, grad_output, scale);
    for i in 0..f_count {
        for j in 0..t {
            grad_nh[[argmax_f[[i, j]], j]] += grad_fh[[i, j]];
        }
    }

    grad_nh
}

/// Compute node heights from coordinates and directions.
///
/// # Arguments
/// * `x` - Point coordinates of shape [num_points, dim]
/// * `v` - Direction vectors of shape [dim, num_directions]
///
/// # Returns
/// Node heights of shape [num_points, num_directions]
pub fn compute_node_heights(x: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
    x.dot(v)
}

/// Generate linear threshold values.
///
/// # Arguments
/// * `radius` - Radius of interval [-radius, radius]
/// * `resolution` - Number of steps
///
/// # Returns
/// Array of threshold values of shape [resolution]
pub fn generate_lin(radius: f32, resolution: usize) -> Array1<f32> {
    let mut lin = Array1::<f32>::zeros(resolution);
    for i in 0..resolution {
        lin[i] = -radius + (2.0 * radius * i as f32) / (resolution as f32 - 1.0);
    }
    lin
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_ect_points_shape() {
        let nh = Array2::<f32>::zeros((10, 8));
        let batch = Array1::<i64>::zeros(10);
        let lin = generate_lin(1.0, 16);
        
        let ect = compute_ect_points_forward(&nh, &batch, &lin, 1, 50.0);
        assert_eq!(ect.shape(), &[1, 16, 8]);
    }

    #[test]
    fn test_fast_ect_shape() {
        let nh = Array2::<f32>::from_shape_fn((10, 8), |(i, j)| {
            ((i + j) as f32 / 20.0) * 2.0 - 1.0
        });
        
        let ect = compute_fast_ect(&nh, 16);
        assert_eq!(ect.shape(), &[16, 8]);
    }

    #[test]
    fn test_node_heights() {
        let x = Array2::<f32>::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let v = Array2::<f32>::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        
        let nh = compute_node_heights(&x, &v);
        assert_eq!(nh.shape(), &[3, 2]);
        assert!((nh[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((nh[[0, 1]] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_lin() {
        let lin = generate_lin(1.0, 5);
        assert_eq!(lin.len(), 5);
        assert!((lin[0] - (-1.0)).abs() < 1e-6);
        assert!((lin[4] - 1.0).abs() < 1e-6);
    }
}
