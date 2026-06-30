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

Ignored benchmark-style checks can be run explicitly when performance smoke coverage is needed:

```shell
cargo test --test benchmarks -- --ignored --nocapture
cargo test --bin server -- --ignored --nocapture
```

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
- `tests/manufactured_scalar.rs`
- `tests/elasticity.rs`
- `tests/elasticity_3d.rs`
- `tests/modal.rs`
- `tests/structural_verification.rs`

Solver stack tests:

- `tests/solver_stack.rs`
- `tests/physics_solver_stack.rs`
- `tests/transient_coverage.rs`

I/O and workflow tests:

- `tests/file_io.rs`
- `tests/mesh_import_export.rs`
- `tests/project_workflow.rs`
- `tests/cli_project.rs`
- `tests/benchmarks.rs`
- `tests/verification_manifest.rs`
- `src/bin/server.rs` unit tests for REST workflows

## Golden Fixtures

Current checked-in fixtures:

- `examples/data/square.mesh`
- `examples/data/square.params`
- `examples/data/square.project.json`
- `examples/data/physical_groups_2d.msh`
- `examples/data/physical_groups_2d_temperature.vtk`
- `examples/data/two_node.solution`
- `examples/data/cli_project_inspect_summary.txt`
- `examples/data/rest_project_request.json`
- `examples/data/rest_project_validate_response.json`
- `examples/data/rest_project_solve_response.json`
- `examples/data/rest_bad_schema_error_response.json`
- `examples/data/rest_mesh_artifact_upload.json`

These fixtures cover the legacy mesh/params path, the v1 project workflow path, Gmsh import, VTK export, compact solution output, CLI project inspection, REST project validation/solve envelopes, REST error schema stability, and artifact uploads.

## Current Coverage Map

- Scalar manufactured solutions: 2D/3D Poisson, steady heat, diffusion-reaction, and electrostatics.
- Poisson: 2D `Tri3`, 3D `Tet4`, Dirichlet handling, local stiffness/load references, CG non-convergence.
- Heat: steady 2D/3D and transient 2D theta integration.
- Diffusion-reaction: 2D/3D reaction matrices and transient 2D/3D theta integration.
- Electrostatics: 2D/3D steady/quasi-static scalar potential and formulation marker.
- Elasticity: 2D/3D stiffness symmetry, rigid translations, constrained solves, affine displacement constraints, transient Newmark dynamics.
- Modal analysis: 2D/3D sorted modes, density validation, constrained model validation, one-DOF frequency references.
- Solver stack: CG, GMRES, dense direct, Jacobi preconditioning, diagnostics, Newton, theta transient, Newmark transient.
- Import/export: legacy mesh/params/solution, golden solution output, Gmsh physical groups, VTK scalar output.
- Project workflows: v1 project parsing/validation, CLI validation/inspection golden output, REST validation/solve golden envelopes, async jobs, artifact upload/download fixtures, and stable error schemas.
- Benchmarks: ignored benchmark-style tests cover Poisson assembly, Poisson solve, Gmsh import, VTK export, project parse/validate/adapt, and REST project validate/solve workflows.

## Known Gaps

- Manufactured-solution suites are not yet complete for non-affine structural stress recovery or modal benchmark problems.
- Benchmarks are lightweight smoke tests only; they do not yet provide statistical sampling, persisted baselines, or CI regression thresholds.
- CAD import workflow fixtures are not yet present beyond Gmsh mesh import.
- REST project jobs and artifacts are in-memory only.
- Verification currently relies on Rust test binaries rather than a CI configuration file.

Future verification sub-steps should reduce these gaps without weakening the required local gates above.
