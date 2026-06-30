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
    SparseLdl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreconditionerKind {
    None,
    Jacobi,
    Amg,
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
        LinearSolverBackend::SparseLdl => sparse_ldl_solve(matrix, rhs, options),
    }
}

pub fn solve_harmonic_response(
    mass: &CsMat<f64>,
    damping: Option<&CsMat<f64>>,
    stiffness: &CsMat<f64>,
    force_amplitude_real: &[f64],
    force_amplitude_imag: &[f64],
    omega: f64,
    solver_options: LinearSolverOptions,
) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    let n = stiffness.rows();
    if stiffness.cols() != n {
        return Err(LinalgError::NonSquareMatrix {
            rows: n,
            cols: stiffness.cols(),
        });
    }
    if mass.rows() != n || mass.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            matrix_dim: n,
            rhs_len: mass.rows(),
        });
    }
    if force_amplitude_real.len() != n || force_amplitude_imag.len() != n {
        return Err(LinalgError::DimensionMismatch {
            matrix_dim: n,
            rhs_len: force_amplitude_real.len(),
        });
    }

    let mut triplets = TriMat::new((2 * n, 2 * n));
    let omega_sq = omega * omega;

    for (r, row) in stiffness.outer_iterator().enumerate() {
        for (c, &val) in row.iter() {
            triplets.add_triplet(r, c, val);
            triplets.add_triplet(r + n, c + n, val);
        }
    }

    for (r, row) in mass.outer_iterator().enumerate() {
        for (c, &val) in row.iter() {
            triplets.add_triplet(r, c, -omega_sq * val);
            triplets.add_triplet(r + n, c + n, -omega_sq * val);
        }
    }

    if let Some(c_mat) = damping {
        if c_mat.rows() != n || c_mat.cols() != n {
            return Err(LinalgError::DimensionMismatch {
                matrix_dim: n,
                rhs_len: c_mat.rows(),
            });
        }
        for (r, row) in c_mat.outer_iterator().enumerate() {
            for (c, &val) in row.iter() {
                triplets.add_triplet(r, c + n, -omega * val);
                triplets.add_triplet(r + n, c, omega * val);
            }
        }
    }

    let a_matrix: CsMat<f64> = triplets.to_csr();

    let mut b = vec![0.0; 2 * n];
    b[0..n].copy_from_slice(force_amplitude_real);
    b[n..2 * n].copy_from_slice(force_amplitude_imag);

    let result = solve_linear_system(&a_matrix, &b, solver_options)?;

    let x_real = result.values[0..n].to_vec();
    let x_imag = result.values[n..2 * n].to_vec();

    Ok((x_real, x_imag))
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

/// Result of a generalized Lanczos eigenvalue solve for `K v = λ M v`.
#[derive(Debug, Clone, PartialEq)]
pub struct LanczosEigenResult {
    /// Converged eigenvalues λ in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors, each normalised so that `v^T M v = 1`.
    /// Stored in row-major order: `eigenvectors[i]` is the i-th mode vector.
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of Lanczos iterations performed.
    pub iterations: usize,
}

/// Solves the generalized eigenvalue problem `K v = λ M v` for the
/// `num_modes` smallest positive eigenvalues using a shift-invert Lanczos
/// iteration.
///
/// # Arguments
/// * `stiffness` - Symmetric positive semi-definite stiffness matrix K (n×n).
/// * `mass_diag` - Diagonal entries of the lumped mass matrix M (length n).
/// * `num_modes` - Number of modes requested.
/// * `shift` - Spectral shift σ.  The factored matrix (K − σ M) is built
///   once and reused every Krylov step.  A value of 0.0 works for problems
///   with no rigid-body modes; otherwise provide a small positive value such
///   as 1e-3 * (smallest expected eigenvalue) to avoid singularity.
/// * `max_iterations` - Maximum Lanczos steps (≥ num_modes; typical: 3–10×
///   num_modes).
/// * `tolerance` - Relative convergence tolerance for eigenvalues (e.g. 1e-8).
///
/// # Returns
/// A [`LanczosEigenResult`] containing converged eigenvalues and eigenvectors,
/// or a [`LinalgError`] if the factored system is singular or too few modes
/// converge.
pub fn solve_lanczos_modes(
    stiffness: &CsMat<f64>,
    mass_diag: &[f64],
    num_modes: usize,
    shift: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<LanczosEigenResult, LinalgError> {
    let n = stiffness.rows();
    if stiffness.cols() != n {
        return Err(LinalgError::NonSquareMatrix {
            rows: n,
            cols: stiffness.cols(),
        });
    }
    if mass_diag.len() != n {
        return Err(LinalgError::DimensionMismatch {
            matrix_dim: n,
            rhs_len: mass_diag.len(),
        });
    }
    if num_modes == 0 || num_modes > n {
        return Err(LinalgError::DimensionMismatch {
            matrix_dim: n,
            rhs_len: num_modes,
        });
    }

    // Build the shifted matrix A = K − σ M (stored as sparse CSR).
    let a_shifted = build_shifted_matrix(stiffness, mass_diag, shift);

    // Lanczos iteration: build a tridiagonal representation of (A)^{-1} M
    // in the M-inner-product space.  We apply (A)^{-1} M at each step via
    // a sparse LDL solve.
    let solver_opts = LinearSolverOptions {
        backend: LinearSolverBackend::SparseLdl,
        ..LinearSolverOptions::default()
    };

    let max_steps = max_iterations.max(num_modes + 10).min(n);

    // Lanczos vectors (M-orthonormal), stored as columns.
    let mut q_vecs: Vec<Vec<f64>> = Vec::with_capacity(max_steps + 1);
    // Tridiagonal matrix entries.
    let mut alpha_vec: Vec<f64> = Vec::with_capacity(max_steps);
    let mut beta_vec: Vec<f64> = Vec::with_capacity(max_steps);

    // Starting vector: random-like but reproducible (alternating ±1 scaled).
    let mut q0 = vec![0.0; n];
    for (i, val) in q0.iter_mut().enumerate() {
        *val = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    m_normalize(&mut q0, mass_diag);
    q_vecs.push(q0);

    let mut beta = 0.0f64;
    let mut prev_eigenvalues: Vec<f64> = vec![f64::INFINITY; num_modes];

    let mut final_iterations = max_steps;
    'outer: for step in 0..max_steps {
        // z = A^{-1} * M * q_j
        let q_j = &q_vecs[step];
        let mq = m_mult(q_j, mass_diag);
        let z_result = solve_linear_system(&a_shifted, &mq, solver_opts.clone())?;
        let mut z = z_result.values;

        // alpha_j = q_j^T M z
        let alpha = m_dot(&z, q_j, mass_diag);
        alpha_vec.push(alpha);

        // Subtract alpha * q_j + beta * q_{j-1}
        axpy(-alpha, q_j, &mut z);
        if step > 0 {
            axpy(-beta, &q_vecs[step - 1], &mut z);
        }

        // M-orthogonalise against previous vectors (full re-orthogonalisation
        // for numerical stability at modest extra cost).
        for prev in &q_vecs {
            let coeff = m_dot(&z, prev, mass_diag);
            axpy(-coeff, prev, &mut z);
        }

        beta = m_norm(&z, mass_diag);
        beta_vec.push(beta);

        if beta > 1e-14 {
            scale(1.0 / beta, &mut z);
        } else {
            // Lucky breakdown: Krylov space exhausted.
            final_iterations = step + 1;
            q_vecs.push(z);
            break;
        }
        q_vecs.push(z);

        // Check convergence by solving the small tridiagonal eigenproblem.
        if step + 1 >= num_modes {
            let t_eigs = tridiagonal_eigenvalues(&alpha_vec, &beta_vec[..beta_vec.len() - 1]);
            // Map back from shift-invert: λ = σ + 1/θ  (where θ is the
            // Ritz value of the shift-inverted operator).
            let ritz_eigs: Vec<f64> = t_eigs
                .iter()
                .map(|&theta| {
                    if theta.abs() > 1e-14 {
                        shift + 1.0 / theta
                    } else {
                        f64::INFINITY
                    }
                })
                .collect();

            // Sort ascending and take the smallest num_modes.
            let mut sorted = ritz_eigs.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let current = &sorted[..num_modes.min(sorted.len())];

            // Check relative change in all requested eigenvalues.
            let converged = current.iter().zip(&prev_eigenvalues).all(|(c, p)| {
                let denom = p.abs().max(1.0);
                (c - p).abs() / denom < tolerance
            });

            prev_eigenvalues.clear();
            prev_eigenvalues.extend_from_slice(current);

            if converged {
                final_iterations = step + 1;
                break 'outer;
            }
        }
    }

    // Solve the final tridiagonal eigenproblem and recover Ritz pairs.
    let t_dim = alpha_vec.len();
    let (t_eigenvalues, t_eigenvectors) =
        tridiagonal_eigen_with_vectors(&alpha_vec, &beta_vec[..t_dim.saturating_sub(1)]);

    // Convert Ritz values back through shift-invert.
    let mut pairs: Vec<(f64, Vec<f64>)> = t_eigenvalues
        .into_iter()
        .zip(t_eigenvectors)
        .filter_map(|(theta, t_vec)| {
            if theta.abs() > 1e-14 {
                let lambda = shift + 1.0 / theta;
                // Reconstruct the global Ritz vector from Lanczos basis.
                let mut ritz = vec![0.0; n];
                for (k, &coeff) in t_vec.iter().enumerate() {
                    if k < q_vecs.len() {
                        axpy(coeff, &q_vecs[k], &mut ritz);
                    }
                }
                Some((lambda, ritz))
            } else {
                None
            }
        })
        .collect();

    // Sort ascending by eigenvalue.
    pairs.sort_by(|(a, _), (b, _)| a.total_cmp(b));

    // Keep only the smallest num_modes with positive eigenvalues.
    let pairs: Vec<_> = pairs
        .into_iter()
        .filter(|(lambda, _)| *lambda >= 0.0)
        .take(num_modes)
        .collect();

    // M-normalise the Ritz vectors.
    let mut eigenvalues = Vec::with_capacity(pairs.len());
    let mut eigenvectors = Vec::with_capacity(pairs.len());
    for (lambda, mut v) in pairs {
        m_normalize(&mut v, mass_diag);
        eigenvalues.push(lambda);
        eigenvectors.push(v);
    }

    Ok(LanczosEigenResult {
        eigenvalues,
        eigenvectors,
        iterations: final_iterations,
    })
}

/// Builds the sparse shifted matrix A = K − σ * diag(mass_diag).
fn build_shifted_matrix(stiffness: &CsMat<f64>, mass_diag: &[f64], shift: f64) -> CsMat<f64> {
    let n = stiffness.rows();
    let mut triplets = TriMat::new((n, n));
    for (row, row_vec) in stiffness.outer_iterator().enumerate() {
        for (col, &val) in row_vec.iter() {
            triplets.add_triplet(row, col, val);
        }
    }
    // Subtract σ * M_ii from diagonal.
    for (i, &m_ii) in mass_diag.iter().enumerate() {
        triplets.add_triplet(i, i, -shift * m_ii);
    }
    triplets.to_csr()
}

/// M-weighted dot product: a^T * diag(m) * b.
fn m_dot(a: &[f64], b: &[f64], m: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .zip(m)
        .map(|((ai, bi), mi)| ai * bi * mi)
        .sum()
}

/// M-weighted norm: sqrt(v^T M v).
fn m_norm(v: &[f64], m: &[f64]) -> f64 {
    m_dot(v, v, m).sqrt()
}

/// Normalise v in the M-inner-product.
fn m_normalize(v: &mut [f64], m: &[f64]) {
    let n = m_norm(v, m);
    if n > 0.0 {
        for val in v.iter_mut() {
            *val /= n;
        }
    }
}

/// Multiply M (diagonal) by v: result[i] = m[i] * v[i].
fn m_mult(v: &[f64], m: &[f64]) -> Vec<f64> {
    v.iter().zip(m).map(|(vi, mi)| vi * mi).collect()
}

/// Solves the symmetric tridiagonal eigenvalue problem using the explicit
/// QR algorithm (Givens rotations), returning eigenvalues in ascending order.
/// `alpha` is the diagonal, `beta` is the sub-diagonal (length = alpha.len()-1).
fn tridiagonal_eigenvalues(alpha: &[f64], beta: &[f64]) -> Vec<f64> {
    let (eigenvalues, _) = tridiagonal_eigen_with_vectors(alpha, beta);
    eigenvalues
}

/// Computes eigenvalues and eigenvectors of a symmetric tridiagonal matrix
/// using the implicit QR shift algorithm.  Returns (eigenvalues, eigenvectors)
/// where each eigenvector is a column of the accumulated rotation matrix.
fn tridiagonal_eigen_with_vectors(alpha: &[f64], beta: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = alpha.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![alpha[0]], vec![vec![1.0]]);
    }

    let mut diag = alpha.to_vec();
    let mut off = beta.to_vec();
    off.resize(n.saturating_sub(1), 0.0);

    // Accumulate Givens rotations in Q (n×n identity → eigenvector matrix).
    let mut q = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        q[i][i] = 1.0;
    }

    // Simple Jacobi-based symmetric tridiagonal QR (Francis double-shift is
    // overkill for the small Lanczos tridiagonal, typically <200 rows).
    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest sub-diagonal.
        let mut p = 0;
        let mut max_off = 0.0f64;
        for i in 0..n - 1 {
            if off[i].abs() > max_off {
                max_off = off[i].abs();
                p = i;
            }
        }
        if max_off <= 1e-13 {
            break;
        }
        let q_idx = p + 1;

        // Compute Givens rotation to zero off[p].
        let d = (diag[q_idx] - diag[p]) / 2.0;
        let t = if d.abs() < 1e-300 {
            1.0
        } else {
            let r = (d * d + off[p] * off[p]).sqrt();
            if d >= 0.0 {
                off[p] / (d + r)
            } else {
                off[p] / (d - r)
            }
        };
        let cos = 1.0 / (1.0 + t * t).sqrt();
        let sin = t * cos;

        // Update tridiagonal.
        let pp = diag[p];
        let qq = diag[q_idx];
        let pq = off[p];
        diag[p] = cos * cos * pp - 2.0 * sin * cos * pq + sin * sin * qq;
        diag[q_idx] = sin * sin * pp + 2.0 * sin * cos * pq + cos * cos * qq;
        off[p] = 0.0;

        // Propagate rotation to adjacent off-diagonals.
        if p > 0 {
            let prev = off[p - 1];
            off[p - 1] = cos * prev - sin * 0.0; // q_{p-1,p+1} was 0
            let _ = prev; // avoid unused warning
            off[p - 1] = cos * off[p - 1];
        }
        if q_idx < n - 1 {
            let next = off[q_idx];
            off[q_idx] = cos * next;
        }

        // Accumulate rotation in Q.
        for row in &mut q {
            let row_p = row[p];
            let row_q = row[q_idx];
            row[p] = cos * row_p - sin * row_q;
            row[q_idx] = sin * row_p + cos * row_q;
        }
    }

    // Sort by eigenvalue.
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|i| (diag[i], q.iter().map(|row| row[i]).collect()))
        .collect();
    pairs.sort_by(|(a, _), (b, _)| a.total_cmp(b));

    let eigenvalues = pairs.iter().map(|(e, _)| *e).collect();
    let eigenvectors = pairs.into_iter().map(|(_, v)| v).collect();
    (eigenvalues, eigenvectors)
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
        let norm_dir = norm(&direction);
        let norm_mat_dir = norm(&matrix_direction);
        if denominator.abs() <= 1e-15 * norm_dir * norm_mat_dir {
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

fn sparse_ldl_solve(
    matrix: &CsMat<f64>,
    rhs: &[f64],
    options: LinearSolverOptions,
) -> Result<LinearSolverResult, LinalgError> {
    let initial_residual = norm(rhs);
    let ldl = sprs_ldl::Ldl::default();
    let system = ldl
        .numeric(matrix.view())
        .map_err(|_| LinalgError::SingularMatrix { pivot: 0 })?;
    let values = system.solve(rhs);
    let residual = residual_norm(matrix, &values, rhs);
    Ok(LinearSolverResult {
        values,
        diagnostics: SolverDiagnostics {
            backend: LinearSolverBackend::SparseLdl,
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

enum Preconditioner {
    None,
    Jacobi(Vec<f64>),
    Amg(AmgPreconditioner),
}

struct AmgPreconditioner {
    matrix: CsMat<f64>,
    p_mat: CsMat<f64>,
    pt_mat: CsMat<f64>,
    coarse_solver: Option<sprs_ldl::LdlNumeric<f64, usize>>,
    inverse_diagonal: Vec<f64>,
}

impl AmgPreconditioner {
    fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = r.len();
        let omega = 2.0 / 3.0;

        // 1. Presmoothing (1 step of damped Jacobi)
        let mut e = vec![0.0; n];
        for i in 0..n {
            e[i] = omega * self.inverse_diagonal[i] * r[i];
        }

        // 2. Residual computation: r_new = r - A * e
        let ae = mul_csr_vec(&self.matrix, &e);
        let mut r_new = vec![0.0; n];
        for i in 0..n {
            r_new[i] = r[i] - ae[i];
        }

        // 3. Restriction: r_c = P^T * r_new
        let r_c = mul_csr_vec(&self.pt_mat, &r_new);

        // 4. Coarse grid solve
        if let Some(ref solver) = self.coarse_solver {
            if r_c.len() > 0 {
                let e_c = solver.solve(&r_c);
                // 5. Prolongation: e = e + P * e_c
                let e_fine = mul_csr_vec(&self.p_mat, &e_c);
                for i in 0..n {
                    e[i] += e_fine[i];
                }
            }
        }

        // 6. Postsmoothing (1 step of damped Jacobi)
        let ae_post = mul_csr_vec(&self.matrix, &e);
        for i in 0..n {
            let res_i = r[i] - ae_post[i];
            e[i] += omega * self.inverse_diagonal[i] * res_i;
        }

        e
    }
}

fn build_preconditioner(
    matrix: &CsMat<f64>,
    kind: PreconditionerKind,
) -> Result<Preconditioner, LinalgError> {
    match kind {
        PreconditionerKind::None => Ok(Preconditioner::None),
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
            Ok(Preconditioner::Jacobi(inverse_diagonal))
        }
        PreconditionerKind::Amg => {
            let n_fine = matrix.rows();
            let mut inverse_diagonal = vec![0.0; n_fine];
            for index in 0..n_fine {
                let diagonal = matrix.get(index, index).copied().unwrap_or(0.0);
                inverse_diagonal[index] = if diagonal.abs() > 1e-14 {
                    1.0 / diagonal
                } else {
                    1.0
                };
            }

            let theta = 0.25;
            let mut node_to_aggregate = vec![None; n_fine];
            let mut num_aggregates = 0;

            for i in 0..n_fine {
                if node_to_aggregate[i].is_some() {
                    continue;
                }

                let agg_idx = num_aggregates;
                node_to_aggregate[i] = Some(agg_idx);

                let a_ii = matrix.get(i, i).copied().unwrap_or(0.0).abs();

                if let Some(row) = matrix.outer_view(i) {
                    for (col_idx, &val) in row.iter() {
                        if col_idx == i || node_to_aggregate[col_idx].is_some() {
                            continue;
                        }
                        let a_jj = matrix.get(col_idx, col_idx).copied().unwrap_or(0.0).abs();
                        let threshold = theta * (a_ii * a_jj).sqrt();
                        if val.abs() >= threshold {
                            node_to_aggregate[col_idx] = Some(agg_idx);
                        }
                    }
                }
                num_aggregates += 1;
            }

            let n_coarse = num_aggregates;

            if n_coarse == 0 || n_coarse >= n_fine {
                return Ok(Preconditioner::Amg(AmgPreconditioner {
                    matrix: matrix.clone(),
                    p_mat: CsMat::new_csc((n_fine, 0), vec![0; 1], vec![], vec![]),
                    pt_mat: CsMat::new_csc((0, n_fine), vec![0; 1], vec![], vec![]),
                    coarse_solver: None,
                    inverse_diagonal,
                }));
            }

            let mut p_tri = TriMat::new((n_fine, n_coarse));
            let mut pt_tri = TriMat::new((n_coarse, n_fine));
            for (fine_idx, &agg) in node_to_aggregate.iter().enumerate() {
                if let Some(agg_idx) = agg {
                    p_tri.add_triplet(fine_idx, agg_idx, 1.0);
                    pt_tri.add_triplet(agg_idx, fine_idx, 1.0);
                }
            }
            let p_mat = p_tri.to_csr();
            let pt_mat = pt_tri.to_csr();

            let ap = matrix * &p_mat;
            let a_coarse = &pt_mat * &ap;

            let ldl = sprs_ldl::Ldl::default();
            let coarse_solver = match ldl.numeric(a_coarse.view()) {
                Ok(solver) => Some(solver),
                Err(_) => None,
            };

            Ok(Preconditioner::Amg(AmgPreconditioner {
                matrix: matrix.clone(),
                p_mat,
                pt_mat,
                coarse_solver,
                inverse_diagonal,
            }))
        }
    }
}

fn apply_preconditioner(precond: &Preconditioner, residual: &[f64]) -> Vec<f64> {
    match precond {
        Preconditioner::None => residual.to_vec(),
        Preconditioner::Jacobi(inverse_diagonal) => inverse_diagonal
            .iter()
            .zip(residual)
            .map(|(inverse, residual)| inverse * residual)
            .collect(),
        Preconditioner::Amg(amg) => amg.apply(residual),
    }
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
