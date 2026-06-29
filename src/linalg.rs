use sprs::{CsMat, TriMat};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverBackend {
    ConjugateGradient,
    Gmres,
    DenseDirect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreconditionerKind {
    None,
    Jacobi,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearSolverOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub backend: LinearSolverBackend,
    pub preconditioner: PreconditionerKind,
    pub record_residual_history: bool,
}

impl Default for LinearSolverOptions {
    fn default() -> Self {
        let solver = SolverOptions::default();
        Self {
            max_iterations: solver.max_iterations,
            tolerance: solver.tolerance,
            backend: LinearSolverBackend::ConjugateGradient,
            preconditioner: PreconditionerKind::None,
            record_residual_history: false,
        }
    }
}

impl From<SolverOptions> for LinearSolverOptions {
    fn from(value: SolverOptions) -> Self {
        Self {
            max_iterations: value.max_iterations,
            tolerance: value.tolerance,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CgResult {
    pub values: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverDiagnostics {
    pub backend: LinearSolverBackend,
    pub preconditioner: PreconditionerKind,
    pub converged: bool,
    pub iterations: usize,
    pub initial_residual_norm: f64,
    pub residual_norm: f64,
    pub residual_history: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearSolverResult {
    pub values: Vec<f64>,
    pub diagnostics: SolverDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearSolverOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub linear_solver: LinearSolverOptions,
}

impl Default for NonlinearSolverOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1.0e-10,
            linear_solver: LinearSolverOptions::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearSolverDiagnostics {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
    pub residual_history: Vec<f64>,
    pub linear_iterations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearSolverResult {
    pub values: Vec<f64>,
    pub diagnostics: NonlinearSolverDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientSolverOptions {
    pub time_step: f64,
    pub steps: usize,
    pub theta: f64,
    pub linear_solver: LinearSolverOptions,
}

impl Default for TransientSolverOptions {
    fn default() -> Self {
        Self {
            time_step: 1.0,
            steps: 1,
            theta: 1.0,
            linear_solver: LinearSolverOptions::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientStepResult {
    pub time: f64,
    pub values: Vec<f64>,
    pub linear_diagnostics: SolverDiagnostics,
}

pub trait LinearSolver {
    fn solve(&self, matrix: &CsMat<f64>, rhs: &[f64]) -> Result<LinearSolverResult, LinalgError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfiguredLinearSolver {
    pub options: LinearSolverOptions,
}

impl ConfiguredLinearSolver {
    pub const fn new(options: LinearSolverOptions) -> Self {
        Self { options }
    }
}

impl LinearSolver for ConfiguredLinearSolver {
    fn solve(&self, matrix: &CsMat<f64>, rhs: &[f64]) -> Result<LinearSolverResult, LinalgError> {
        solve_linear_system(matrix, rhs, self.options.clone())
    }
}

pub trait NonlinearSystem {
    fn dimension(&self) -> usize;
    fn residual(&self, values: &[f64]) -> Vec<f64>;
    fn jacobian(&self, values: &[f64]) -> CsMat<f64>;
}

#[derive(Debug, Error, PartialEq)]
pub enum LinalgError {
    #[error("matrix must be square, got {rows} rows and {cols} columns")]
    NonSquareMatrix { rows: usize, cols: usize },
    #[error("rhs length {rhs_len} does not match matrix dimension {matrix_dim}")]
    DimensionMismatch { matrix_dim: usize, rhs_len: usize },
    #[error("CG broke down because the search direction is orthogonal to A*p")]
    Breakdown,
    #[error("GMRES broke down at Arnoldi step {iteration}")]
    GmresBreakdown { iteration: usize },
    #[error("CG did not converge after {iterations} iterations; residual norm is {residual_norm}")]
    NonConverged {
        iterations: usize,
        residual_norm: f64,
    },
    #[error("matrix is singular or nearly singular at pivot {pivot}")]
    SingularMatrix { pivot: usize },
    #[error("preconditioner cannot be built because diagonal entry {index} is {value}")]
    InvalidPreconditionerDiagonal { index: usize, value: f64 },
    #[error(
        "initial guess length {initial_len} does not match nonlinear system dimension {system_dim}"
    )]
    NonlinearDimensionMismatch {
        system_dim: usize,
        initial_len: usize,
    },
    #[error(
        "nonlinear solve did not converge after {iterations} iterations; residual norm is {residual_norm}"
    )]
    NonlinearNonConverged {
        iterations: usize,
        residual_norm: f64,
    },
    #[error("time step must be positive and finite, got {0}")]
    InvalidTimeStep(f64),
    #[error("theta must be finite and in [0, 1], got {0}")]
    InvalidTheta(f64),
    #[error("initial state length {initial_len} does not match matrix dimension {matrix_dim}")]
    TransientDimensionMismatch {
        matrix_dim: usize,
        initial_len: usize,
    },
}

pub fn conjugate_gradient(
    matrix: &CsMat<f64>,
    rhs: &[f64],
    options: SolverOptions,
) -> Result<CgResult, LinalgError> {
    let result = solve_linear_system(matrix, rhs, LinearSolverOptions::from(options))?;
    Ok(CgResult {
        values: result.values,
        iterations: result.diagnostics.iterations,
        residual_norm: result.diagnostics.residual_norm,
    })
}

pub fn solve_linear_system(
    matrix: &CsMat<f64>,
    rhs: &[f64],
    options: LinearSolverOptions,
) -> Result<LinearSolverResult, LinalgError> {
    validate_linear_system(matrix, rhs)?;
    match options.backend {
        LinearSolverBackend::ConjugateGradient => {
            conjugate_gradient_with_options(matrix, rhs, options)
        }
        LinearSolverBackend::Gmres => gmres_with_options(matrix, rhs, options),
        LinearSolverBackend::DenseDirect => dense_direct_solve(matrix, rhs, options),
    }
}

pub fn newton_solve<S: NonlinearSystem>(
    system: &S,
    initial_values: Vec<f64>,
    options: NonlinearSolverOptions,
) -> Result<NonlinearSolverResult, LinalgError> {
    if initial_values.len() != system.dimension() {
        return Err(LinalgError::NonlinearDimensionMismatch {
            system_dim: system.dimension(),
            initial_len: initial_values.len(),
        });
    }

    let mut values = initial_values;
    let mut residual = system.residual(&values);
    let mut residual_norm = norm(&residual);
    let mut residual_history = vec![residual_norm];
    let mut linear_iterations = 0;

    if residual_norm <= options.tolerance {
        return Ok(NonlinearSolverResult {
            values,
            diagnostics: NonlinearSolverDiagnostics {
                converged: true,
                iterations: 0,
                residual_norm,
                residual_history,
                linear_iterations,
            },
        });
    }

    for iterations in 1..=options.max_iterations {
        let jacobian = system.jacobian(&values);
        let rhs: Vec<_> = residual.iter().map(|value| -value).collect();
        let step = solve_linear_system(&jacobian, &rhs, options.linear_solver.clone())?;
        linear_iterations += step.diagnostics.iterations;
        axpy(1.0, &step.values, &mut values);

        residual = system.residual(&values);
        residual_norm = norm(&residual);
        residual_history.push(residual_norm);
        if residual_norm <= options.tolerance {
            return Ok(NonlinearSolverResult {
                values,
                diagnostics: NonlinearSolverDiagnostics {
                    converged: true,
                    iterations,
                    residual_norm,
                    residual_history,
                    linear_iterations,
                },
            });
        }
    }

    Err(LinalgError::NonlinearNonConverged {
        iterations: options.max_iterations,
        residual_norm,
    })
}

pub fn solve_linear_transient<F>(
    mass: &CsMat<f64>,
    stiffness: &CsMat<f64>,
    initial_values: Vec<f64>,
    source: F,
    options: TransientSolverOptions,
) -> Result<Vec<TransientStepResult>, LinalgError>
where
    F: Fn(f64) -> Vec<f64>,
{
    validate_transient_inputs(mass, stiffness, initial_values.len(), &options)?;

    let dt = options.time_step;
    let lhs = add_scaled_matrices(mass, stiffness, 1.0, options.theta * dt);
    let rhs_matrix = add_scaled_matrices(mass, stiffness, 1.0, -(1.0 - options.theta) * dt);
    let mut values = initial_values;
    let mut steps = Vec::with_capacity(options.steps);

    for step_index in 1..=options.steps {
        let previous_time = (step_index - 1) as f64 * dt;
        let time = step_index as f64 * dt;
        let previous_source = source(previous_time);
        let next_source = source(time);
        if previous_source.len() != values.len() || next_source.len() != values.len() {
            return Err(LinalgError::DimensionMismatch {
                matrix_dim: values.len(),
                rhs_len: previous_source.len().max(next_source.len()),
            });
        }

        let mut rhs = mul_csr_vec(&rhs_matrix, &values);
        for ((rhs_value, previous), next) in rhs.iter_mut().zip(previous_source).zip(next_source) {
            *rhs_value += dt * ((1.0 - options.theta) * previous + options.theta * next);
        }

        let result = solve_linear_system(&lhs, &rhs, options.linear_solver.clone())?;
        values = result.values.clone();
        steps.push(TransientStepResult {
            time,
            values: values.clone(),
            linear_diagnostics: result.diagnostics,
        });
    }

    Ok(steps)
}

fn conjugate_gradient_with_options(
    matrix: &CsMat<f64>,
    rhs: &[f64],
    options: LinearSolverOptions,
) -> Result<LinearSolverResult, LinalgError> {
    let mut values = vec![0.0; rhs.len()];
    let mut residual = rhs.to_vec();
    let preconditioner = build_preconditioner(matrix, options.preconditioner)?;
    let mut preconditioned_residual = apply_preconditioner(&preconditioner, &residual);
    let mut direction = preconditioned_residual.clone();
    let mut residual_preconditioned = dot(&residual, &preconditioned_residual);
    let initial_residual = norm(&residual);
    let mut residual_history = if options.record_residual_history {
        vec![initial_residual]
    } else {
        Vec::new()
    };

    if initial_residual <= options.tolerance {
        return Ok(LinearSolverResult {
            values,
            diagnostics: SolverDiagnostics {
                backend: LinearSolverBackend::ConjugateGradient,
                preconditioner: options.preconditioner,
                converged: true,
                iterations: 0,
                initial_residual_norm: initial_residual,
                residual_norm: initial_residual,
                residual_history,
            },
        });
    }

    for iterations in 1..=options.max_iterations {
        let matrix_direction = mul_csr_vec(matrix, &direction);
        let denominator = dot(&direction, &matrix_direction);
        if denominator.abs() <= f64::EPSILON {
            return Err(LinalgError::Breakdown);
        }

        let alpha = residual_preconditioned / denominator;
        axpy(alpha, &direction, &mut values);
        axpy(-alpha, &matrix_direction, &mut residual);

        let residual_norm = norm(&residual);
        if options.record_residual_history {
            residual_history.push(residual_norm);
        }
        if residual_norm <= options.tolerance {
            return Ok(LinearSolverResult {
                values,
                diagnostics: SolverDiagnostics {
                    backend: LinearSolverBackend::ConjugateGradient,
                    preconditioner: options.preconditioner,
                    converged: true,
                    iterations,
                    initial_residual_norm: initial_residual,
                    residual_norm,
                    residual_history,
                },
            });
        }

        preconditioned_residual = apply_preconditioner(&preconditioner, &residual);
        let next_residual_preconditioned = dot(&residual, &preconditioned_residual);
        let beta = next_residual_preconditioned / residual_preconditioned;
        for (direction_value, residual_value) in direction.iter_mut().zip(&preconditioned_residual)
        {
            *direction_value = residual_value + beta * *direction_value;
        }
        residual_preconditioned = next_residual_preconditioned;
    }

    Err(LinalgError::NonConverged {
        iterations: options.max_iterations,
        residual_norm: norm(&residual),
    })
}

fn dense_direct_solve(
    matrix: &CsMat<f64>,
    rhs: &[f64],
    options: LinearSolverOptions,
) -> Result<LinearSolverResult, LinalgError> {
    let initial_residual = norm(rhs);
    let values = gaussian_elimination(to_dense(matrix), rhs)?;
    let residual = residual_norm(matrix, &values, rhs);
    Ok(LinearSolverResult {
        values,
        diagnostics: SolverDiagnostics {
            backend: LinearSolverBackend::DenseDirect,
            preconditioner: PreconditionerKind::None,
            converged: residual <= options.tolerance.max(1.0e-12),
            iterations: 1,
            initial_residual_norm: initial_residual,
            residual_norm: residual,
            residual_history: if options.record_residual_history {
                vec![initial_residual, residual]
            } else {
                Vec::new()
            },
        },
    })
}

fn gmres_with_options(
    matrix: &CsMat<f64>,
    rhs: &[f64],
    options: LinearSolverOptions,
) -> Result<LinearSolverResult, LinalgError> {
    let initial_residual = norm(rhs);
    let mut residual_history = if options.record_residual_history {
        vec![initial_residual]
    } else {
        Vec::new()
    };

    if initial_residual <= options.tolerance {
        return Ok(LinearSolverResult {
            values: vec![0.0; rhs.len()],
            diagnostics: SolverDiagnostics {
                backend: LinearSolverBackend::Gmres,
                preconditioner: options.preconditioner,
                converged: true,
                iterations: 0,
                initial_residual_norm: initial_residual,
                residual_norm: initial_residual,
                residual_history,
            },
        });
    }

    let preconditioner = build_preconditioner(matrix, options.preconditioner)?;
    let max_iterations = options.max_iterations.min(rhs.len());
    let mut basis = vec![apply_preconditioner(&preconditioner, rhs)];
    normalize(&mut basis[0]);
    let mut hessenberg = vec![vec![0.0; max_iterations]; max_iterations + 1];
    let mut best_values = vec![0.0; rhs.len()];
    let mut best_residual = initial_residual;

    for iteration in 0..max_iterations {
        let mut arnoldi_vector =
            apply_preconditioner(&preconditioner, &mul_csr_vec(matrix, &basis[iteration]));

        for column in 0..=iteration {
            hessenberg[column][iteration] = dot(&arnoldi_vector, &basis[column]);
            axpy(
                -hessenberg[column][iteration],
                &basis[column],
                &mut arnoldi_vector,
            );
        }

        hessenberg[iteration + 1][iteration] = norm(&arnoldi_vector);
        let broke_down = hessenberg[iteration + 1][iteration] <= f64::EPSILON;
        if !broke_down && iteration + 1 < max_iterations {
            let mut next_basis = arnoldi_vector;
            scale(1.0 / hessenberg[iteration + 1][iteration], &mut next_basis);
            basis.push(next_basis);
        }

        let coefficients = solve_gmres_least_squares(&hessenberg, initial_residual, iteration + 1)?;
        let mut values = vec![0.0; rhs.len()];
        for (coefficient, basis_vector) in coefficients.iter().zip(&basis) {
            axpy(*coefficient, basis_vector, &mut values);
        }

        let residual = residual_norm(matrix, &values, rhs);
        best_residual = residual;
        best_values = values;
        if options.record_residual_history {
            residual_history.push(residual);
        }
        if residual <= options.tolerance {
            return Ok(LinearSolverResult {
                values: best_values,
                diagnostics: SolverDiagnostics {
                    backend: LinearSolverBackend::Gmres,
                    preconditioner: options.preconditioner,
                    converged: true,
                    iterations: iteration + 1,
                    initial_residual_norm: initial_residual,
                    residual_norm: residual,
                    residual_history,
                },
            });
        }
        if broke_down {
            return Err(LinalgError::GmresBreakdown {
                iteration: iteration + 1,
            });
        }
    }

    Err(LinalgError::NonConverged {
        iterations: max_iterations,
        residual_norm: best_residual,
    })
}

fn solve_gmres_least_squares(
    hessenberg: &[Vec<f64>],
    beta: f64,
    iteration_count: usize,
) -> Result<Vec<f64>, LinalgError> {
    let mut normal_matrix = vec![vec![0.0; iteration_count]; iteration_count];
    let mut normal_rhs = vec![0.0; iteration_count];

    for (row, hessenberg_row) in hessenberg.iter().enumerate().take(iteration_count + 1) {
        for col in 0..iteration_count {
            normal_rhs[col] += hessenberg_row[col] * if row == 0 { beta } else { 0.0 };
            for inner_col in 0..iteration_count {
                normal_matrix[col][inner_col] += hessenberg_row[col] * hessenberg_row[inner_col];
            }
        }
    }

    gaussian_elimination(normal_matrix, &normal_rhs)
}

fn gaussian_elimination(mut matrix: Vec<Vec<f64>>, rhs: &[f64]) -> Result<Vec<f64>, LinalgError> {
    let n = rhs.len();
    let mut values = rhs.to_vec();

    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_value = matrix[pivot][pivot].abs();
        for (row, row_values) in matrix.iter().enumerate().skip(pivot + 1) {
            let value = row_values[pivot].abs();
            if value > pivot_value {
                pivot_row = row;
                pivot_value = value;
            }
        }
        if pivot_value <= f64::EPSILON {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        if pivot_row != pivot {
            matrix.swap(pivot, pivot_row);
            values.swap(pivot, pivot_row);
        }

        for row in (pivot + 1)..n {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            matrix[row][pivot] = 0.0;
            let pivot_tail: Vec<_> = matrix[pivot].iter().copied().skip(pivot + 1).collect();
            for (value, pivot_value) in matrix[row].iter_mut().skip(pivot + 1).zip(pivot_tail) {
                *value -= factor * pivot_value;
            }
            values[row] -= factor * values[pivot];
        }
    }

    for row in (0..n).rev() {
        let mut sum = values[row];
        for (col, value) in matrix[row].iter().enumerate().skip(row + 1) {
            sum -= value * values[col];
        }
        values[row] = sum / matrix[row][row];
    }

    Ok(values)
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

fn validate_linear_system(matrix: &CsMat<f64>, rhs: &[f64]) -> Result<(), LinalgError> {
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
    Ok(())
}

fn validate_transient_inputs(
    mass: &CsMat<f64>,
    stiffness: &CsMat<f64>,
    initial_len: usize,
    options: &TransientSolverOptions,
) -> Result<(), LinalgError> {
    validate_linear_system(mass, &vec![0.0; mass.rows()])?;
    validate_linear_system(stiffness, &vec![0.0; stiffness.rows()])?;
    if mass.rows() != stiffness.rows() {
        return Err(LinalgError::DimensionMismatch {
            matrix_dim: mass.rows(),
            rhs_len: stiffness.rows(),
        });
    }
    if initial_len != mass.rows() {
        return Err(LinalgError::TransientDimensionMismatch {
            matrix_dim: mass.rows(),
            initial_len,
        });
    }
    if !options.time_step.is_finite() || options.time_step <= 0.0 {
        return Err(LinalgError::InvalidTimeStep(options.time_step));
    }
    if !options.theta.is_finite() || options.theta < 0.0 || options.theta > 1.0 {
        return Err(LinalgError::InvalidTheta(options.theta));
    }
    Ok(())
}

fn add_scaled_matrices(
    first: &CsMat<f64>,
    second: &CsMat<f64>,
    first_scale: f64,
    second_scale: f64,
) -> CsMat<f64> {
    let mut triplets = TriMat::new((first.rows(), first.cols()));
    for (row_index, row) in first.outer_iterator().enumerate() {
        for (col_index, value) in row.iter() {
            triplets.add_triplet(row_index, col_index, first_scale * value);
        }
    }
    for (row_index, row) in second.outer_iterator().enumerate() {
        for (col_index, value) in row.iter() {
            triplets.add_triplet(row_index, col_index, second_scale * value);
        }
    }
    triplets.to_csr()
}

fn build_preconditioner(
    matrix: &CsMat<f64>,
    kind: PreconditionerKind,
) -> Result<Vec<f64>, LinalgError> {
    match kind {
        PreconditionerKind::None => Ok(vec![1.0; matrix.rows()]),
        PreconditionerKind::Jacobi => {
            let mut inverse_diagonal = vec![0.0; matrix.rows()];
            for (index, value) in inverse_diagonal.iter_mut().enumerate() {
                let diagonal = matrix.get(index, index).copied().unwrap_or(0.0);
                if diagonal.abs() <= f64::EPSILON || !diagonal.is_finite() {
                    return Err(LinalgError::InvalidPreconditionerDiagonal {
                        index,
                        value: diagonal,
                    });
                }
                *value = 1.0 / diagonal;
            }
            Ok(inverse_diagonal)
        }
    }
}

fn apply_preconditioner(inverse_diagonal: &[f64], residual: &[f64]) -> Vec<f64> {
    inverse_diagonal
        .iter()
        .zip(residual)
        .map(|(inverse, residual)| inverse * residual)
        .collect()
}

fn to_dense(matrix: &CsMat<f64>) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; matrix.cols()]; matrix.rows()];
    for (row_index, row) in matrix.outer_iterator().enumerate() {
        for (col_index, value) in row.iter() {
            dense[row_index][col_index] += value;
        }
    }
    dense
}

fn residual_norm(matrix: &CsMat<f64>, values: &[f64], rhs: &[f64]) -> f64 {
    let matrix_values = mul_csr_vec(matrix, values);
    matrix_values
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| {
            let residual = expected - actual;
            residual * residual
        })
        .sum::<f64>()
        .sqrt()
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (y_value, x_value) in y.iter_mut().zip(x) {
        *y_value += alpha * x_value;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn normalize(values: &mut [f64]) {
    let vector_norm = norm(values);
    if vector_norm > 0.0 {
        scale(1.0 / vector_norm, values);
    }
}

fn scale(alpha: f64, values: &mut [f64]) {
    for value in values {
        *value *= alpha;
    }
}
