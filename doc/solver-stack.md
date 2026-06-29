# Solver Stack

Kepler exposes a richer solver stack in `kepler::linalg` while keeping the original `SolverOptions` and `conjugate_gradient` API available for existing FEM callers.

## Linear Solvers

Use `solve_linear_system` or `ConfiguredLinearSolver` with `LinearSolverOptions` for backend selection:

```rust
let result = solve_linear_system(
    &matrix,
    &rhs,
    LinearSolverOptions {
        backend: LinearSolverBackend::DenseDirect,
        record_residual_history: true,
        ..LinearSolverOptions::default()
    },
)?;
```

Supported linear backends:

- `LinearSolverBackend::ConjugateGradient`: iterative SPD solve.
- `LinearSolverBackend::DenseDirect`: dense Gaussian-elimination hook for small systems and tests.

Supported preconditioners:

- `PreconditionerKind::None`.
- `PreconditionerKind::Jacobi`, using the inverse matrix diagonal.

`SolverDiagnostics` reports backend, preconditioner, convergence status, iteration count, initial residual, final residual, and optional residual history.

## Compatibility API

Existing physics modules still accept:

```rust
SolverOptions {
    max_iterations: 10_000,
    tolerance: 1.0e-10,
}
```

Internally, `conjugate_gradient` maps those options into `LinearSolverOptions` with the CG backend and no preconditioner. This preserves the previous Poisson, heat, diffusion-reaction, electrostatics, and elasticity APIs.

## Physics Solver APIs

Physics modules also expose `_with_solver` variants for callers that need backend selection, preconditioning, and full diagnostics:

```rust
let result = solve_poisson_with_solver(
    &mesh,
    &problem,
    LinearSolverOptions {
        backend: LinearSolverBackend::DenseDirect,
        preconditioner: PreconditionerKind::None,
        record_residual_history: true,
        ..LinearSolverOptions::default()
    },
)?;

let residual = result.diagnostics.residual_norm;
```

Available diagnostic solve variants include:

- `solve_poisson_with_solver` and `solve_poisson_3d_with_solver`.
- `solve_steady_heat_with_solver` and `solve_steady_heat_3d_with_solver`.
- `solve_diffusion_reaction_with_solver` and `solve_diffusion_reaction_3d_with_solver`.
- `solve_electrostatics_with_solver` and `solve_electrostatics_3d_with_solver`.
- `solve_elasticity_with_solver` and `solve_elasticity_3d_with_solver`.

These functions return domain-specific solver result types that preserve field values or displacements while carrying `SolverDiagnostics`.

## Nonlinear Solves

`newton_solve` provides a Newton iteration over user-supplied nonlinear systems:

```rust
struct MySystem;

impl NonlinearSystem for MySystem {
    fn dimension(&self) -> usize { 1 }
    fn residual(&self, x: &[f64]) -> Vec<f64> { vec![x[0] * x[0] - 2.0] }
    fn jacobian(&self, x: &[f64]) -> CsMat<f64> {
        // Build sparse Jacobian here.
    }
}

let result = newton_solve(
    &MySystem,
    vec![1.0],
    NonlinearSolverOptions::default(),
)?;
```

`NonlinearSolverDiagnostics` reports nonlinear convergence, nonlinear iterations, final residual, residual history, and total linear iterations used by Newton corrections.

## Transient Solves

`solve_linear_transient` implements a theta-method stepper for linear systems:

```text
M du/dt + K u = f(t)
```

The step equation is:

```text
(M + theta dt K) u[n+1] =
(M - (1 - theta) dt K) u[n] + dt ((1 - theta) f[n] + theta f[n+1])
```

`theta = 1.0` gives backward Euler, `theta = 0.5` gives Crank-Nicolson, and `theta = 0.0` gives forward Euler in this matrix form. Each `TransientStepResult` includes time, state values, and linear solver diagnostics.

## Current Limits

The direct backend is intentionally dense and intended for small systems, verification fixtures, and as a future hook point for external sparse direct solvers. Nonlinear and transient solvers are generic linalg-level primitives; they are not yet wired into the file CLI, REST API, or physics-specific problem definitions.
