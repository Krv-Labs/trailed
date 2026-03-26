use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::f32::consts::PI;

/// Generate randomly sampled directions from a sphere in d dimensions.
///
/// A standard normal is sampled and projected onto the unit sphere to
/// yield a randomly sampled set of points on the unit sphere.
///
/// # Arguments
/// * `num_thetas` - The number of directions to generate
/// * `d` - The dimension of the ambient space
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Array of shape [d, num_thetas] containing unit vectors
pub fn generate_uniform_directions(num_thetas: usize, d: usize, seed: u64) -> Array2<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = StandardNormal;

    let mut v = Array2::<f32>::zeros((d, num_thetas));

    for j in 0..num_thetas {
        let mut norm_sq = 0.0f32;
        for i in 0..d {
            let val: f32 = normal.sample(&mut rng);
            v[[i, j]] = val;
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-10 {
            for i in 0..d {
                v[[i, j]] /= norm;
            }
        }
    }

    v
}

/// Generate structured directions in 2D along the unit circle.
///
/// The interval [0, 2*pi] is divided into a regular grid and the
/// corresponding angles on the unit circle are calculated.
///
/// # Arguments
/// * `num_thetas` - The number of directions to generate
///
/// # Returns
/// Array of shape [2, num_thetas] containing unit vectors
pub fn generate_2d_directions(num_thetas: usize) -> Array2<f32> {
    let mut v = Array2::<f32>::zeros((2, num_thetas));

    for i in 0..num_thetas {
        let theta = 2.0 * PI * (i as f32) / (num_thetas as f32);
        v[[0, i]] = theta.sin();
        v[[1, i]] = theta.cos();
    }

    v
}

/// Generate multiple sets of structured directions in n dimensions.
///
/// Generates sets of directions by embedding the 2D unit circle in d
/// dimensions and sampling this unit circle in a structured fashion.
/// This generates (d choose 2) structured directions organized in channels.
///
/// # Arguments
/// * `num_thetas` - Total number of directions to generate
/// * `d` - The dimension of the ambient space
///
/// # Returns
/// Array of shape [d, num_thetas] containing unit vectors
pub fn generate_multiview_directions(num_thetas: usize, d: usize) -> Array2<f32> {
    let idx_pairs: Vec<(usize, usize)> = (0..d)
        .flat_map(|i| (i + 1..d).map(move |j| (i, j)))
        .collect();

    let num_pairs = idx_pairs.len();
    if num_pairs == 0 {
        return Array2::<f32>::zeros((d, num_thetas));
    }

    let num_directions_per_circle = num_thetas / num_pairs;
    let remainder = num_thetas % num_pairs;

    let mut v = Array2::<f32>::zeros((d, num_thetas));
    let mut col_offset = 0;

    for (idx, &(i1, i2)) in idx_pairs.iter().enumerate() {
        let num_t = if idx == 0 {
            num_directions_per_circle + remainder
        } else {
            num_directions_per_circle
        };

        for t in 0..num_t {
            let theta = 2.0 * PI * (t as f32) / ((num_directions_per_circle + remainder) as f32);
            v[[i1, col_offset + t]] = theta.sin();
            v[[i2, col_offset + t]] = theta.cos();
        }

        col_offset += num_t;
    }

    v
}

/// Generate a smooth spherical grid of directions on the unit sphere in 3D.
///
/// The directions are parameterized by theta (polar angle, [0, pi]) and
/// phi (azimuthal angle, [0, 2*pi)), and returned as unit vectors.
///
/// # Arguments
/// * `num_thetas` - Number of theta samples (from 0 to pi)
/// * `num_phis` - Number of phi samples (from 0 to 2*pi)
///
/// # Returns
/// Array of shape [3, num_thetas * num_phis] containing unit vectors
pub fn generate_spherical_grid_directions(num_thetas: usize, num_phis: usize) -> Array2<f32> {
    let total_dirs = num_thetas * num_phis;
    let mut v = Array2::<f32>::zeros((3, total_dirs));

    let mut col = 0;
    for phi_idx in 0..num_phis {
        let phi = 2.0 * PI * (phi_idx as f32) / (num_phis as f32);

        for theta_idx in 0..num_thetas {
            let theta = PI * ((theta_idx + 1) as f32) / ((num_thetas + 2) as f32);

            let sin_theta = theta.sin();
            v[[0, col]] = sin_theta * phi.cos();
            v[[1, col]] = sin_theta * phi.sin();
            v[[2, col]] = theta.cos();

            col += 1;
        }
    }

    v
}

/// Normalize a set of direction vectors to unit length.
///
/// # Arguments
/// * `v` - Array of shape [d, num_directions]
///
/// # Returns
/// Array of shape [d, num_directions] with normalized vectors
pub fn normalize_directions(v: &Array2<f32>) -> Array2<f32> {
    let d = v.shape()[0];
    let num_dirs = v.shape()[1];
    let mut result = v.clone();

    for j in 0..num_dirs {
        let mut norm_sq = 0.0f32;
        for i in 0..d {
            norm_sq += v[[i, j]] * v[[i, j]];
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-10 {
            for i in 0..d {
                result[[i, j]] /= norm;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_directions_shape() {
        let v = generate_uniform_directions(64, 3, 42);
        assert_eq!(v.shape(), &[3, 64]);
    }

    #[test]
    fn test_uniform_directions_unit_norm() {
        let v = generate_uniform_directions(64, 3, 42);
        for j in 0..64 {
            let norm_sq: f32 = (0..3).map(|i| v[[i, j]] * v[[i, j]]).sum();
            assert!((norm_sq - 1.0).abs() < 1e-6, "Direction {} has norm {}", j, norm_sq.sqrt());
        }
    }

    #[test]
    fn test_2d_directions_shape() {
        let v = generate_2d_directions(32);
        assert_eq!(v.shape(), &[2, 32]);
    }

    #[test]
    fn test_2d_directions_unit_norm() {
        let v = generate_2d_directions(32);
        for j in 0..32 {
            let norm_sq: f32 = v[[0, j]] * v[[0, j]] + v[[1, j]] * v[[1, j]];
            assert!((norm_sq - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_multiview_directions_shape() {
        let v = generate_multiview_directions(64, 3);
        assert_eq!(v.shape(), &[3, 64]);
    }

    #[test]
    fn test_spherical_grid_shape() {
        let v = generate_spherical_grid_directions(8, 16);
        assert_eq!(v.shape(), &[3, 128]);
    }

    #[test]
    fn test_spherical_grid_unit_norm() {
        let v = generate_spherical_grid_directions(8, 16);
        for j in 0..128 {
            let norm_sq: f32 = (0..3).map(|i| v[[i, j]] * v[[i, j]]).sum();
            assert!((norm_sq - 1.0).abs() < 1e-6);
        }
    }
}
