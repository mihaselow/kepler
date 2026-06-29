use kepler::{
    ConfiguredLinearSolver, LinalgError, LinearSolver, LinearSolverBackend, LinearSolverOptions,
    NonlinearSolverOptions, NonlinearSystem, PreconditionerKind, TransientSolverOptions,
    newton_solve, solve_linear_system, solve_linear_transient,
};
use sprs::TriMat;

#[test]
fn linear_solver_selects_dense_direct_backend() {
    let matrix = csr_matrix(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)]);
    let rhs = [1.0, 2.0];

    let result = solve_linear_system(
        &matrix,
        &rhs,
        LinearSolverOptions {
            backend: LinearSolverBackend::DenseDirect,
            record_residual_history: true,
            ..LinearSolverOptions::default()
        },
    )
    .unwrap();

    assert_close(result.values[0], 0.2);
    assert_close(result.values[1], 0.6);
    assert_eq!(result.diagnostics.backend, LinearSolverBackend::DenseDirect);
    assert!(result.diagnostics.converged);
    assert_eq!(result.diagnostics.residual_history.len(), 2);
}

#[test]
fn linear_solver_uses_jacobi_preconditioned_cg_with_diagnostics() {
    let matrix = csr_matrix(2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)]);
    let rhs = [1.0, 2.0];
    let solver = ConfiguredLinearSolver::new(LinearSolverOptions {
        preconditioner: PreconditionerKind::Jacobi,
        record_residual_history: true,
        ..LinearSolverOptions::default()
    });

    let result = solver.solve(&matrix, &rhs).unwrap();

    assert_close(result.values[0], 1.0 / 11.0);
    assert_close(result.values[1], 7.0 / 11.0);
    assert_eq!(
        result.diagnostics.preconditioner,
        PreconditionerKind::Jacobi
    );
    assert!(result.diagnostics.converged);
    assert!(!result.diagnostics.residual_history.is_empty());
}

#[test]
fn gmres_solves_nonsymmetric_system_with_diagnostics() {
    let matrix = csr_matrix(2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 1, 2.0)]);
    let rhs = [7.0, 4.0];

    let result = solve_linear_system(
        &matrix,
        &rhs,
        LinearSolverOptions {
            backend: LinearSolverBackend::Gmres,
            record_residual_history: true,
            ..LinearSolverOptions::default()
        },
    )
    .unwrap();

    assert_close(result.values[0], 5.0 / 3.0);
    assert_close(result.values[1], 2.0);
    assert_eq!(result.diagnostics.backend, LinearSolverBackend::Gmres);
    assert!(result.diagnostics.converged);
    assert!(!result.diagnostics.residual_history.is_empty());
}

#[test]
fn gmres_reports_non_convergence() {
    let matrix = csr_matrix(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)]);
    let rhs = [1.0, 2.0];

    let error = solve_linear_system(
        &matrix,
        &rhs,
        LinearSolverOptions {
            backend: LinearSolverBackend::Gmres,
            max_iterations: 1,
            tolerance: 1.0e-14,
            ..LinearSolverOptions::default()
        },
    )
    .unwrap_err();

    assert!(matches!(
        error,
        LinalgError::NonConverged {
            iterations: 1,
            residual_norm,
        } if residual_norm > 0.0
    ));
}

#[test]
fn jacobi_preconditioner_rejects_missing_diagonal() {
    let matrix = csr_matrix(2, &[(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0)]);
    let rhs = [1.0, 2.0];

    let error = solve_linear_system(
        &matrix,
        &rhs,
        LinearSolverOptions {
            preconditioner: PreconditionerKind::Jacobi,
            ..LinearSolverOptions::default()
        },
    )
    .unwrap_err();

    assert_eq!(
        error,
        LinalgError::InvalidPreconditionerDiagonal {
            index: 1,
            value: 0.0,
        }
    );
}

#[test]
fn newton_solve_converges_for_scalar_nonlinear_system() {
    let result = newton_solve(
        &SquareRootTwo,
        vec![1.0],
        NonlinearSolverOptions {
            linear_solver: LinearSolverOptions {
                backend: LinearSolverBackend::DenseDirect,
                ..LinearSolverOptions::default()
            },
            ..NonlinearSolverOptions::default()
        },
    )
    .unwrap();

    assert_close(result.values[0], 2.0_f64.sqrt());
    assert!(result.diagnostics.converged);
    assert!(result.diagnostics.iterations > 0);
}

#[test]
fn transient_theta_method_solves_linear_decay() {
    let mass = csr_matrix(1, &[(0, 0, 1.0)]);
    let stiffness = csr_matrix(1, &[(0, 0, 1.0)]);

    let steps = solve_linear_transient(
        &mass,
        &stiffness,
        vec![1.0],
        |_| vec![0.0],
        TransientSolverOptions {
            time_step: 1.0,
            steps: 2,
            theta: 1.0,
            linear_solver: LinearSolverOptions {
                backend: LinearSolverBackend::DenseDirect,
                ..LinearSolverOptions::default()
            },
        },
    )
    .unwrap();

    assert_eq!(steps.len(), 2);
    assert_close(steps[0].values[0], 0.5);
    assert_close(steps[1].values[0], 0.25);
    assert_close(steps[1].time, 2.0);
}

struct SquareRootTwo;

impl NonlinearSystem for SquareRootTwo {
    fn dimension(&self) -> usize {
        1
    }

    fn residual(&self, values: &[f64]) -> Vec<f64> {
        vec![values[0] * values[0] - 2.0]
    }

    fn jacobian(&self, values: &[f64]) -> sprs::CsMat<f64> {
        csr_matrix(1, &[(0, 0, 2.0 * values[0])])
    }
}

fn csr_matrix(size: usize, entries: &[(usize, usize, f64)]) -> sprs::CsMat<f64> {
    let mut triplets = TriMat::new((size, size));
    for &(row, col, value) in entries {
        triplets.add_triplet(row, col, value);
    }
    triplets.to_csr()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-10,
        "expected {actual} to be close to {expected}",
    );
}
