# Verification And Quality Gates

This document is the verification manifest for Kepler. It lists the checks, fixtures, and commands that currently guard solver behavior and workflow compatibility.

## Required Local Gates

Run these before considering a roadmap sub-step complete:

```shell
cargo fmt
cargo test
cargo clippy --all-targets --all-features
cargo build --bin server
```

Focused checks may be run during development, but the full gate above should pass before a sub-step is marked complete.

## Test Suite Inventory

Core FEM and mesh tests:

- `tests/poisson.rs`
- `tests/poisson_3d.rs`
- `tests/mesh_topology.rs`
- `tests/conditions.rs`
- `tests/annotations.rs`

Physics verification tests:

- `tests/heat.rs`
- `tests/diffusion_reaction.rs`
- `tests/electrostatics.rs`
- `tests/elasticity.rs`
- `tests/elasticity_3d.rs`
- `tests/modal.rs`

Solver stack tests:

- `tests/solver_stack.rs`
- `tests/physics_solver_stack.rs`
- `tests/transient_coverage.rs`

I/O and workflow tests:

- `tests/file_io.rs`
- `tests/mesh_import_export.rs`
- `tests/project_workflow.rs`
- `tests/cli_project.rs`
- `tests/verification_manifest.rs`
- `src/bin/server.rs` unit tests for REST workflows

## Golden Fixtures

Current checked-in fixtures:

- `examples/data/square.mesh`
- `examples/data/square.params`
- `examples/data/square.project.json`

These fixtures cover the legacy mesh/params path and the v1 project workflow path.

## Current Coverage Map

- Poisson: 2D `Tri3`, 3D `Tet4`, Dirichlet handling, local stiffness/load references, CG non-convergence.
- Heat: steady 2D/3D and transient 2D theta integration.
- Diffusion-reaction: 2D/3D reaction matrices and transient 2D/3D theta integration.
- Electrostatics: 2D/3D steady/quasi-static scalar potential and formulation marker.
- Elasticity: 2D/3D stiffness symmetry, rigid translations, constrained solves, transient Newmark dynamics.
- Modal analysis: 2D/3D sorted modes, density validation, constrained model validation.
- Solver stack: CG, GMRES, dense direct, Jacobi preconditioning, diagnostics, Newton, theta transient, Newmark transient.
- Import/export: legacy mesh/params/solution, Gmsh physical groups, VTK scalar output.
- Project workflows: v1 project parsing/validation, CLI validation/inspection, REST validation/solve, async jobs, artifact upload/download.

## Known Gaps

- Manufactured-solution suites are not yet complete for every physics/dimension pair.
- Benchmarks are not yet implemented.
- CAD import workflow fixtures are not yet present.
- REST project jobs and artifacts are in-memory only.
- Verification currently relies on Rust test binaries rather than a CI configuration file.

Future verification sub-steps should reduce these gaps without weakening the required local gates above.
