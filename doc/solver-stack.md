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
- `LinearSolverBackend::Gmres`: iterative nonsymmetric solve using full GMRES.
- `LinearSolverBackend::DenseDirect`: dense Gaussian-elimination hook for small systems and tests.

Supported preconditioners:

- `PreconditionerKind::None`.
- `PreconditionerKind::Jacobi`, using the inverse matrix diagonal.

`SolverDiagnostics` reports backend, preconditioner, convergence status, iteration count, initial residual, final residual, and optional residual history.

## Matrix Diagnostics

Use `analyze_matrix` to inspect sparse matrices before choosing a solver or debugging convergence:

```rust
let diagnostics = analyze_matrix(&matrix, 1.0e-12);

if diagnostics.spd_heuristics.likely_spd {
    // CG is a reasonable first backend to try.
}
```

The returned `MatrixDiagnostics` includes:

- `SparsityStats`: rows, columns, nonzeros, density, and average row occupancy.
- `SymmetryDiagnostics`: square/structural/numerical symmetry and maximum absolute asymmetry.
- `DiagonalDiagnostics`: diagonal min/max, zero count, non-finite count, positive diagonal flag, and diagonal-dominance margin.
- `SpdHeuristics`: a conservative hint based on square shape, numerical symmetry, finite positive diagonal entries, and diagonal health.

These checks are heuristics, not proofs. They are intended for solver selection, validation warnings, and failure explanations.

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

## File, CLI, And REST Exposure

Poisson `.params` files can request supported solver-stack settings:

```text
solver max_iterations 10000
solver tolerance 1e-10
solver backend conjugate_gradient
solver preconditioner none
solver record_residual_history false
```

The file-driven CLI applies these options via `solve_poisson_with_solver`. It writes the existing compact `.solution` format and prints backend, preconditioner, iteration count, residual norm, and residual history when recording is enabled.

The `/solve/poisson` REST endpoint accepts the same concepts in `solver_options`:

```json
{
  "solver_options": {
    "max_iterations": 10000,
    "tolerance": 1e-10,
    "backend": "gmres",
    "preconditioner": "none",
    "record_residual_history": true
  }
}
```

REST responses include a `diagnostics` object with backend, preconditioner, convergence status, initial residual, and residual history.

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
