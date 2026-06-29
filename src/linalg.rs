use sprs::CsMat;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1.0e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CgResult {
    pub values: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Error, PartialEq)]
pub enum LinalgError {
    #[error("matrix must be square, got {rows} rows and {cols} columns")]
    NonSquareMatrix { rows: usize, cols: usize },
    #[error("rhs length {rhs_len} does not match matrix dimension {matrix_dim}")]
    DimensionMismatch { matrix_dim: usize, rhs_len: usize },
    #[error("CG broke down because the search direction is orthogonal to A*p")]
    Breakdown,
    #[error("CG did not converge after {iterations} iterations; residual norm is {residual_norm}")]
    NonConverged {
        iterations: usize,
        residual_norm: f64,
    },
}

pub fn conjugate_gradient(
    matrix: &CsMat<f64>,
    rhs: &[f64],
    options: SolverOptions,
) -> Result<CgResult, LinalgError> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    if rows != cols {
        return Err(LinalgError::NonSquareMatrix { rows, cols });
    }
    if rhs.len() != rows {
        return Err(LinalgError::DimensionMismatch {
            matrix_dim: rows,
            rhs_len: rhs.len(),
        });
    }

    let mut values = vec![0.0; rhs.len()];
    let mut residual = rhs.to_vec();
    let mut direction = residual.clone();
    let mut residual_squared = dot(&residual, &residual);
    let initial_residual = residual_squared.sqrt();

    if initial_residual <= options.tolerance {
        return Ok(CgResult {
            values,
            iterations: 0,
            residual_norm: initial_residual,
        });
    }

    for iterations in 1..=options.max_iterations {
        let matrix_direction = mul_csr_vec(matrix, &direction);
        let denominator = dot(&direction, &matrix_direction);
        if denominator.abs() <= f64::EPSILON {
            return Err(LinalgError::Breakdown);
        }

        let alpha = residual_squared / denominator;
        axpy(alpha, &direction, &mut values);
        axpy(-alpha, &matrix_direction, &mut residual);

        let next_residual_squared = dot(&residual, &residual);
        let residual_norm = next_residual_squared.sqrt();
        if residual_norm <= options.tolerance {
            return Ok(CgResult {
                values,
                iterations,
                residual_norm,
            });
        }

        let beta = next_residual_squared / residual_squared;
        for (direction_value, residual_value) in direction.iter_mut().zip(&residual) {
            *direction_value = residual_value + beta * *direction_value;
        }
        residual_squared = next_residual_squared;
    }

    Err(LinalgError::NonConverged {
        iterations: options.max_iterations,
        residual_norm: residual_squared.sqrt(),
    })
}

pub fn mul_csr_vec(matrix: &CsMat<f64>, vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.rows()];
    for (row_index, row) in matrix.outer_iterator().enumerate() {
        for (col_index, value) in row.iter() {
            result[row_index] += value * vector[col_index];
        }
    }
    result
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (y_value, x_value) in y.iter_mut().zip(x) {
        *y_value += alpha * x_value;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
