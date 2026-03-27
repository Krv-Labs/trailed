use ndarray::{Array1, Array2, Array3, Array4, s};
use rayon::prelude::*;

/// Compute the sigmoid function: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Parallel computation of ECT for point clouds.
///
/// Parallelizes over directions for better cache locality when
/// the number of directions is larger than the number of points.
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
pub fn compute_ecc_forward_parallel(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    dim_size: usize,
    scale: f32,
) -> Array3<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];
    let r = lin.len();

    // Create output slices for each direction
    let results: Vec<Array2<f32>> = (0..t)
        .into_par_iter()
        .map(|j| {
            let mut out_slice = Array2::<f32>::zeros((dim_size, r));
            
            for i in 0..n {
                let graph_idx = batch[i] as usize;
                let nh_val = nh[[i, j]];
                
                for k in 0..r {
                    let lin_val = lin[k];
                    let ecc_val = sigmoid(scale * (lin_val - nh_val));
                    out_slice[[graph_idx, k]] += ecc_val;
                }
            }
            out_slice
        })
        .collect();

    // Combine results into 3D array
    let mut out = Array3::<f32>::zeros((dim_size, r, t));
    for (j, slice) in results.into_iter().enumerate() {
        for b in 0..dim_size {
            for k in 0..r {
                out[[b, k, j]] = slice[[b, k]];
            }
        }
    }
    out
}

/// Parallel computation of ECT backward pass.
pub fn compute_ecc_backward_parallel(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    lin: &Array1<f32>,
    grad_output: &Array3<f32>,
    scale: f32,
) -> Array2<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];
    let r = lin.len();

    // Compute gradients for each direction in parallel
    let grad_cols: Vec<Array1<f32>> = (0..t)
        .into_par_iter()
        .map(|j| {
            let mut grad_col = Array1::<f32>::zeros(n);
            
            for i in 0..n {
                let graph_idx = batch[i] as usize;
                let nh_val = nh[[i, j]];
                let mut grad_sum = 0.0;
                
                for k in 0..r {
                    let lin_val = lin[k];
                    let s = sigmoid(scale * (lin_val - nh_val));
                    let decc_dnh = s * (1.0 - s) * (-scale);
                    let dout = grad_output[[graph_idx, k, j]];
                    grad_sum += dout * decc_dnh;
                }
                grad_col[i] = grad_sum;
            }
            grad_col
        })
        .collect();

    // Combine into 2D array
    let mut grad_nh = Array2::<f32>::zeros((n, t));
    for (j, col) in grad_cols.into_iter().enumerate() {
        for i in 0..n {
            grad_nh[[i, j]] = col[i];
        }
    }
    grad_nh
}

/// Parallel computation of fast ECT (non-differentiable).
///
/// Parallelizes the bincount operation over directions.
pub fn compute_fast_ect_parallel(
    nh: &Array2<f32>,
    resolution: usize,
) -> Array2<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];

    if resolution == 0 {
        return Array2::<f32>::zeros((0, t));
    }

    let cols: Vec<Array1<f32>> = (0..t)
        .into_par_iter()
        .map(|j| {
            let mut col = Array1::<f32>::zeros(resolution);
            
            // Bincount
            for i in 0..n {
                let nh_val = nh[[i, j]];
                let bin = (((nh_val + 1.0) * (resolution as f32 / 2.0)) as usize)
                    .min(resolution - 1);
                col[bin] += 1.0;
            }
            
            // Cumulative sum
            for k in 1..resolution {
                col[k] += col[k - 1];
            }
            col
        })
        .collect();

    // Combine into 2D array
    let mut out = Array2::<f32>::zeros((resolution, t));
    for (j, col) in cols.into_iter().enumerate() {
        for k in 0..resolution {
            out[[k, j]] = col[k];
        }
    }
    out
}

/// Parallel computation of fast ECT with batch support.
pub fn compute_fast_ect_batched_parallel(
    nh: &Array2<f32>,
    batch: &Array1<i64>,
    dim_size: usize,
    resolution: usize,
) -> Array3<f32> {
    let n = nh.shape()[0];
    let t = nh.shape()[1];

    if resolution == 0 {
        return Array3::<f32>::zeros((dim_size, 0, t));
    }

    let slices: Vec<Array2<f32>> = (0..t)
        .into_par_iter()
        .map(|j| {
            let mut slice = Array2::<f32>::zeros((dim_size, resolution));
            
            // Bincount per batch
            for i in 0..n {
                let graph_idx = batch[i] as usize;
                let nh_val = nh[[i, j]];
                let bin = (((nh_val + 1.0) * (resolution as f32 / 2.0)) as usize)
                    .min(resolution - 1);
                slice[[graph_idx, bin]] += 1.0;
            }
            
            // Cumulative sum per batch
            for b in 0..dim_size {
                for k in 1..resolution {
                    slice[[b, k]] += slice[[b, k - 1]];
                }
            }
            slice
        })
        .collect();

    // Combine into 3D array
    let mut out = Array3::<f32>::zeros((dim_size, resolution, t));
    for (j, slice) in slices.into_iter().enumerate() {
        for b in 0..dim_size {
            for k in 0..resolution {
                out[[b, k, j]] = slice[[b, k]];
            }
        }
    }
    out
}

/// Parallel computation of ECT with channels.
pub fn compute_ect_channels_forward_parallel(
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

    // Parallelize over directions
    let slices: Vec<Array3<f32>> = (0..t)
        .into_par_iter()
        .map(|j| {
            let mut slice = Array3::<f32>::zeros((dim_size, r, max_channels));
            
            for i in 0..n {
                let graph_idx = batch[i] as usize;
                let ch = channels[i] as usize;
                let nh_val = nh[[i, j]];
                
                for k in 0..r {
                    let lin_val = lin[k];
                    let ecc_val = sigmoid(scale * (lin_val - nh_val));
                    slice[[graph_idx, k, ch]] += ecc_val;
                }
            }
            slice
        })
        .collect();

    // Combine into 4D array [dim_size, t, r, max_channels]
    let mut out = Array4::<f32>::zeros((dim_size, t, r, max_channels));
    for (j, slice) in slices.into_iter().enumerate() {
        for b in 0..dim_size {
            for k in 0..r {
                for c in 0..max_channels {
                    out[[b, j, k, c]] = slice[[b, k, c]];
                }
            }
        }
    }
    out
}

/// Batch matrix multiplication in parallel.
///
/// Computes X @ V for each sample in a batch.
///
/// # Arguments
/// * `x` - Input tensor of shape [batch_size, num_points, dim]
/// * `v` - Direction matrix of shape [dim, num_directions]
///
/// # Returns
/// Node heights of shape [batch_size, num_points, num_directions]
pub fn batch_matmul_parallel(
    x: &Array3<f32>,
    v: &Array2<f32>,
) -> Array3<f32> {
    let batch_size = x.shape()[0];
    let num_points = x.shape()[1];
    let num_dirs = v.shape()[1];

    let results: Vec<Array2<f32>> = (0..batch_size)
        .into_par_iter()
        .map(|b| {
            let x_slice = x.slice(s![b, .., ..]);
            x_slice.dot(v)
        })
        .collect();

    let mut out = Array3::<f32>::zeros((batch_size, num_points, num_dirs));
    for (b, mat) in results.into_iter().enumerate() {
        for i in 0..num_points {
            for j in 0..num_dirs {
                out[[b, i, j]] = mat[[i, j]];
            }
        }
    }
    out
}

/// Compute ECT for a batch of point clouds in parallel.
///
/// Each point cloud is processed independently.
///
/// # Arguments
/// * `x` - Point clouds of shape [batch_size, num_points, dim]
/// * `v` - Direction vectors of shape [dim, num_directions]
/// * `radius` - Radius of interval
/// * `resolution` - Number of threshold steps
/// * `scale` - Scale factor for sigmoid
///
/// # Returns
/// ECT of shape [batch_size, resolution, num_directions]
pub fn compute_ect_batch_parallel(
    x: &Array3<f32>,
    v: &Array2<f32>,
    radius: f32,
    resolution: usize,
    scale: f32,
) -> Array3<f32> {
    let batch_size = x.shape()[0];
    let num_dirs = v.shape()[1];

    if resolution == 0 {
        return Array3::<f32>::zeros((batch_size, 0, num_dirs));
    }

    // Generate linear thresholds
    let lin: Array1<f32> = if resolution == 1 {
        Array1::from_vec(vec![0.0])
    } else {
        Array1::from_iter(
            (0..resolution).map(|i| {
                -radius + (2.0 * radius * i as f32) / (resolution as f32 - 1.0)
            }),
        )
    };

    let results: Vec<Array2<f32>> = (0..batch_size)
        .into_par_iter()
        .map(|b| {
            let x_slice = x.slice(s![b, .., ..]);
            let nh = x_slice.dot(v);
            
            let n = nh.shape()[0];
            let t = nh.shape()[1];
            let mut out = Array2::<f32>::zeros((resolution, t));
            
            for i in 0..n {
                for j in 0..t {
                    let nh_val = nh[[i, j]];
                    for k in 0..resolution {
                        let lin_val = lin[k];
                        let ecc_val = sigmoid(scale * (lin_val - nh_val));
                        out[[k, j]] += ecc_val;
                    }
                }
            }
            out
        })
        .collect();

    let mut out = Array3::<f32>::zeros((batch_size, resolution, num_dirs));
    for (b, mat) in results.into_iter().enumerate() {
        for k in 0..resolution {
            for j in 0..num_dirs {
                out[[b, k, j]] = mat[[k, j]];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ect::{compute_ecc_forward, generate_lin};

    #[test]
    fn test_parallel_matches_sequential() {
        let nh = Array2::<f32>::from_shape_fn((100, 32), |(i, j)| {
            ((i * j) as f32 / 1000.0) * 2.0 - 1.0
        });
        let batch = Array1::<i64>::from_iter((0..100).map(|i| (i / 25) as i64));
        let lin = generate_lin(1.0, 64);
        
        let seq = compute_ecc_forward(&nh, &batch, &lin, 4, 50.0);
        let par = compute_ecc_forward_parallel(&nh, &batch, &lin, 4, 50.0);
        
        for b in 0..4 {
            for k in 0..64 {
                for j in 0..32 {
                    assert!(
                        (seq[[b, k, j]] - par[[b, k, j]]).abs() < 1e-5,
                        "Mismatch at [{}, {}, {}]: {} vs {}",
                        b, k, j, seq[[b, k, j]], par[[b, k, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_batch_matmul_parallel() {
        let x = Array3::<f32>::from_shape_fn((4, 10, 3), |(b, i, j)| {
            (b * 100 + i * 10 + j) as f32 / 1000.0
        });
        let v = Array2::<f32>::from_shape_fn((3, 8), |(i, j)| {
            (i * 8 + j) as f32 / 24.0
        });
        
        let nh = batch_matmul_parallel(&x, &v);
        assert_eq!(nh.shape(), &[4, 10, 8]);
    }

    #[test]
    fn test_compute_ect_batch_parallel() {
        let x = Array3::<f32>::from_shape_fn((4, 10, 3), |(b, i, j)| {
            ((b * 100 + i * 10 + j) as f32 / 1000.0) * 2.0 - 1.0
        });
        let v = Array2::<f32>::from_shape_fn((3, 8), |(i, j)| {
            let val = (i * 8 + j) as f32 / 24.0;
            val / (val * val + 0.1).sqrt()
        });
        
        let ect = compute_ect_batch_parallel(&x, &v, 1.0, 16, 50.0);
        assert_eq!(ect.shape(), &[4, 16, 8]);
    }
}
