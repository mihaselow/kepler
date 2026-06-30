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
pub struct MatrixDiagnostics {
    pub sparsity: SparsityStats,
    pub symmetry: SymmetryDiagnostics,
    pub diagonal: DiagonalDiagnostics,
    pub spd_heuristics: SpdHeuristics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparsityStats {
    pub rows: usize,
    pub cols: usize,
    pub nonzeros: usize,
    pub density: f64,
    pub average_nonzeros_per_row: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SymmetryDiagnostics {
    pub is_square: bool,
    pub structurally_symmetric: bool,
    pub numerically_symmetric: bool,
    pub max_abs_asymmetry: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiagonalDiagnostics {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub all_positive: bool,
    pub zero_count: usize,
    pub non_finite_count: usize,
    pub min_diagonal_dominance_margin: Option<f64>,
    pub weakly_diagonally_dominant: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdHeuristics {
    pub positive_diagonal: bool,
    pub numerically_symmetric: bool,
    pub weakly_diagonally_dominant: bool,
    pub likely_spd: bool,
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

#[derive(Debug, Clone, PartialEq)]
pub struct NewmarkSolverOptions {
    pub time_step: f64,
    pub steps: usize,
    pub gamma: f64,
    pub beta: f64,
    pub linear_solver: LinearSolverOptions,
}

impl Default for NewmarkSolverOptions {
    fn default() -> Self {
        Self {
            time_step: 1.0,
            steps: 1,
            gamma: 0.5,
            beta: 0.25,
            linear_solver: LinearSolverOptions::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewmarkStepResult {
    pub time: f64,
    pub displacements: Vec<f64>,
    pub velocities: Vec<f64>,
    pub accelerations: Vec<f64>,
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
    #[error("Newmark gamma must be positive and finite, got {0}")]
    InvalidNewmarkGamma(f64),
    #[error("Newmark beta must be positive and finite, got {0}")]
    InvalidNewmarkBeta(f64),
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

pub fn analyze_matrix(matrix: &CsMat<f64>, symmetry_tolerance: f64) -> MatrixDiagnostics {
    let sparsity = sparsity_stats(matrix);
    let symmetry = symmetry_diagnostics(matrix, symmetry_tolerance);
    let diagonal = diagonal_diagnostics(matrix);
    let spd_heuristics = SpdHeuristics {
        positive_diagonal: diagonal.all_positive,
        numerically_symmetric: symmetry.numerically_symmetric,
        weakly_diagonally_dominant: diagonal.weakly_diagonally_dominant,
        likely_spd: symmetry.is_square
            && symmetry.numerically_symmetric
            && diagonal.all_positive
            && diagonal.non_finite_count == 0,
    };

    MatrixDiagnostics {
        sparsity,
        symmetry,
        diagonal,
        spd_heuristics,
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

pub fn solve_newmark_transient<F>(
    mass: &CsMat<f64>,
    damping: Option<&CsMat<f64>>,
    stiffness: &CsMat<f64>,
    initial_displacements: Vec<f64>,
    initial_velocities: Vec<f64>,
    source: F,
    options: NewmarkSolverOptions,
) -> Result<Vec<NewmarkStepResult>, LinalgError>
where
    F: Fn(f64) -> Vec<f64>,
{
    validate_newmark_inputs(
        mass,
        damping,
        stiffness,
        initial_displacements.len(),
        initial_velocities.len(),
        &options,
    )?;

    let dt = options.time_step;
    let beta = options.beta;
    let gamma = options.gamma;
    let damping_matrix = damping.cloned();
    let mut displacements = initial_displacements;
    let mut velocities = initial_velocities;
    let initial_source = source(0.0);
    if initial_source.len() != displacements.len() {
        return Err(LinalgError::DimensionMismatch {
            matrix_dim: displacements.len(),
            rhs_len: initial_source.len(),
        });
    }

    let mut acceleration_rhs = initial_source;
    subtract_assign(
        &mut acceleration_rhs,
        &mul_csr_vec(stiffness, &displacements),
    );
    if let Some(damping) = damping_matrix.as_ref() {
        subtract_assign(&mut acceleration_rhs, &mul_csr_vec(damping, &velocities));
    }
    let initial_acceleration =
        solve_linear_system(mass, &acceleration_rhs, options.linear_solver.clone())?;
    let mut accelerations = initial_acceleration.values;

    let mass_scale = 1.0 / (beta * dt * dt);
    let damping_scale = gamma / (beta * dt);
    let lhs = match damping_matrix.as_ref() {
        Some(damping) => {
            add_three_scaled_matrices(stiffness, damping, mass, 1.0, damping_scale, mass_scale)
        }
        None => add_scaled_matrices(stiffness, mass, 1.0, mass_scale),
    };
    let mut steps = Vec::with_capacity(options.steps);

    for step_index in 1..=options.steps {
        let time = step_index as f64 * dt;
        let next_source = source(time);
        if next_source.len() != displacements.len() {
            return Err(LinalgError::DimensionMismatch {
                matrix_dim: displacements.len(),
                rhs_len: next_source.len(),
            });
        }

        let predicted_displacements =
            predict_displacements(&displacements, &velocities, &accelerations, dt, beta);
        let predicted_velocities = predict_velocities(&velocities, &accelerations, dt, gamma);
        let mut rhs = next_source;
        axpy(
            mass_scale,
            &mul_csr_vec(mass, &predicted_displacements),
            &mut rhs,
        );
        if let Some(damping) = damping_matrix.as_ref() {
            let damping_values = mul_csr_vec(damping, &predicted_displacements);
            axpy(damping_scale, &damping_values, &mut rhs);
            subtract_assign(&mut rhs, &mul_csr_vec(damping, &predicted_velocities));
        }

        let result = solve_linear_system(&lhs, &rhs, options.linear_solver.clone())?;
        let next_displacements = result.values;
        let next_accelerations: Vec<_> = next_displacements
            .iter()
            .zip(&predicted_displacements)
            .map(|(next, predicted)| (next - predicted) * mass_scale)
            .collect();
        let mut next_velocities = predicted_velocities;
        axpy(gamma * dt, &next_accelerations, &mut next_velocities);

        displacements = next_displacements;
        velocities = next_velocities;
        accelerations = next_accelerations;
        steps.push(NewmarkStepResult {
            time,
            displacements: displacements.clone(),
            velocities: velocities.clone(),
            accelerations: accelerations.clone(),
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

fn sparsity_stats(matrix: &CsMat<f64>) -> SparsityStats {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let nonzeros = matrix.nnz();
    let entry_count = rows.saturating_mul(cols);
    SparsityStats {
        rows,
        cols,
        nonzeros,
        density: if entry_count == 0 {
            0.0
        } else {
            nonzeros as f64 / entry_count as f64
        },
        average_nonzeros_per_row: if rows == 0 {
            0.0
        } else {
            nonzeros as f64 / rows as f64
        },
    }
}

fn symmetry_diagnostics(matrix: &CsMat<f64>, tolerance: f64) -> SymmetryDiagnostics {
    let is_square = matrix.rows() == matrix.cols();
    if !is_square {
        return SymmetryDiagnostics {
            is_square,
            structurally_symmetric: false,
            numerically_symmetric: false,
            max_abs_asymmetry: f64::INFINITY,
        };
    }

    let mut structurally_symmetric = true;
    let mut max_abs_asymmetry: f64 = 0.0;
    for row_index in 0..matrix.rows() {
        for col_index in 0..matrix.cols() {
            let value = matrix.get(row_index, col_index).copied().unwrap_or(0.0);
            let transpose_value = matrix.get(col_index, row_index).copied().unwrap_or(0.0);
            if (value == 0.0) != (transpose_value == 0.0) {
                structurally_symmetric = false;
            }
            max_abs_asymmetry = max_abs_asymmetry.max((value - transpose_value).abs());
        }
    }

    SymmetryDiagnostics {
        is_square,
        structurally_symmetric,
        numerically_symmetric: max_abs_asymmetry <= tolerance,
        max_abs_asymmetry,
    }
}

fn diagonal_diagnostics(matrix: &CsMat<f64>) -> DiagonalDiagnostics {
    let diagonal_count = matrix.rows().min(matrix.cols());
    let mut min = None;
    let mut max = None;
    let mut all_positive = matrix.rows() == matrix.cols() && diagonal_count > 0;
    let mut zero_count = 0;
    let mut non_finite_count = 0;
    let mut min_margin = None;
    let mut weakly_diagonally_dominant = matrix.rows() == matrix.cols() && diagonal_count > 0;

    for index in 0..diagonal_count {
        let diagonal = matrix.get(index, index).copied().unwrap_or(0.0);
        min = Some(min.map_or(diagonal, |value: f64| value.min(diagonal)));
        max = Some(max.map_or(diagonal, |value: f64| value.max(diagonal)));
        if diagonal == 0.0 {
            zero_count += 1;
        }
        if !diagonal.is_finite() {
            non_finite_count += 1;
        }
        if !diagonal.is_finite() || diagonal <= 0.0 {
            all_positive = false;
        }

        let off_diagonal_sum = matrix
            .outer_view(index)
            .map(|row| {
                row.iter()
                    .filter(|(col_index, _)| *col_index != index)
                    .map(|(_, value)| value.abs())
                    .sum::<f64>()
            })
            .unwrap_or(0.0);
        let margin = diagonal.abs() - off_diagonal_sum;
        min_margin = Some(min_margin.map_or(margin, |value: f64| value.min(margin)));
        if margin < -f64::EPSILON {
            weakly_diagonally_dominant = false;
        }
    }

    DiagonalDiagnostics {
        min,
        max,
        all_positive,
        zero_count,
        non_finite_count,
        min_diagonal_dominance_margin: min_margin,
        weakly_diagonally_dominant,
    }
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

fn validate_newmark_inputs(
    mass: &CsMat<f64>,
    damping: Option<&CsMat<f64>>,
    stiffness: &CsMat<f64>,
    displacement_len: usize,
    velocity_len: usize,
    options: &NewmarkSolverOptions,
) -> Result<(), LinalgError> {
    validate_linear_system(mass, &vec![0.0; mass.rows()])?;
    validate_linear_system(stiffness, &vec![0.0; stiffness.rows()])?;
    if mass.rows() != stiffness.rows() {
        return Err(LinalgError::DimensionMismatch {
            matrix_dim: mass.rows(),
            rhs_len: stiffness.rows(),
        });
    }
    if let Some(damping) = damping {
        validate_linear_system(damping, &vec![0.0; damping.rows()])?;
        if damping.rows() != mass.rows() {
            return Err(LinalgError::DimensionMismatch {
                matrix_dim: mass.rows(),
                rhs_len: damping.rows(),
            });
        }
    }
    if displacement_len != mass.rows() {
        return Err(LinalgError::TransientDimensionMismatch {
            matrix_dim: mass.rows(),
            initial_len: displacement_len,
        });
    }
    if velocity_len != mass.rows() {
        return Err(LinalgError::TransientDimensionMismatch {
            matrix_dim: mass.rows(),
            initial_len: velocity_len,
        });
    }
    if !options.time_step.is_finite() || options.time_step <= 0.0 {
        return Err(LinalgError::InvalidTimeStep(options.time_step));
    }
    if !options.gamma.is_finite() || options.gamma <= 0.0 {
        return Err(LinalgError::InvalidNewmarkGamma(options.gamma));
    }
    if !options.beta.is_finite() || options.beta <= 0.0 {
        return Err(LinalgError::InvalidNewmarkBeta(options.beta));
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

fn add_three_scaled_matrices(
    first: &CsMat<f64>,
    second: &CsMat<f64>,
    third: &CsMat<f64>,
    first_scale: f64,
    second_scale: f64,
    third_scale: f64,
) -> CsMat<f64> {
    let partial = add_scaled_matrices(first, second, first_scale, second_scale);
    add_scaled_matrices(&partial, third, 1.0, third_scale)
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

fn subtract_assign(values: &mut [f64], decrement: &[f64]) {
    for (value, decrement) in values.iter_mut().zip(decrement) {
        *value -= decrement;
    }
}

fn predict_displacements(
    displacements: &[f64],
    velocities: &[f64],
    accelerations: &[f64],
    dt: f64,
    beta: f64,
) -> Vec<f64> {
    displacements
        .iter()
        .zip(velocities)
        .zip(accelerations)
        .map(|((displacement, velocity), acceleration)| {
            displacement + dt * velocity + dt * dt * (0.5 - beta) * acceleration
        })
        .collect()
}

fn predict_velocities(velocities: &[f64], accelerations: &[f64], dt: f64, gamma: f64) -> Vec<f64> {
    velocities
        .iter()
        .zip(accelerations)
        .map(|(velocity, acceleration)| velocity + dt * (1.0 - gamma) * acceleration)
        .collect()
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
