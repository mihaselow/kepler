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

## Local Workflow Guidance

Use focused commands while developing a narrow change, then finish with the required local gates. Good focused checks include `cargo test --test <name>` for an affected integration test, `cargo test --bin server` for REST workflow changes, and `cargo test --doc` when examples in public docs change.

Run ignored benchmark-style checks when a change affects assembly loops, solver backend selection, import/export paths, project adaptation, or REST job orchestration. These checks are smoke benchmarks: they confirm the path still runs and prints timing data, but they are not pass/fail performance thresholds.

Golden fixture updates should be deliberate. When a `.mesh`, `.params`, `.solution`, `.msh`, `.vtk`, project JSON, CLI output, or REST response fixture changes, update the corresponding test and explain whether the fixture reflects a behavior change, schema change, or formatting-only change.

## CI Guidance

A CI job should run the required local gates in the same order shown above. The benchmark-style checks can be scheduled separately or run manually before performance-sensitive changes merge, because their timing output is machine dependent and not yet compared against persisted baselines.

The verification manifest is part of the quality gate. Any new integration test file, checked-in golden fixture, required command, or known verification gap should be added here in the same change that introduces it. `tests/verification_manifest.rs` guards that integration tests and required fixtures stay listed.

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
- `tests/contact.rs`
- `tests/nonlinear_plasticity.rs`
- `tests/thermal_struct.rs`
- `tests/modal.rs`
- `tests/structural_verification.rs`

Solver stack tests:

- `tests/solver_stack.rs`
- `tests/physics_solver_stack.rs`
- `tests/transient_coverage.rs`

I/O and workflow tests:

- `tests/file_io.rs`
- `tests/mesh_import_export.rs`
- `tests/cad_workflow.rs`
- `tests/cli_cad.rs`
- `tests/project_workflow.rs`
- `tests/cli_project.rs`
- `tests/abaqus_import.rs`
- `tests/abaqus_verification.rs`
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
- `examples/data/abaqus/block.inp`
- `examples/data/abaqus/uniaxial_patch.inp`
- `examples/data/abaqus/uniaxial_patch.verify.json`
- `examples/data/abaqus/cantilever.inp`
- `examples/data/abaqus/cantilever.verify.json`
- `examples/data/abaqus/README.md`
- `examples/solve_inp.rs`

These fixtures cover the legacy mesh/params path, the v1 project workflow path, Gmsh import, VTK export, compact solution output, CLI project inspection, REST project validation/solve envelopes, REST error schema stability, artifact uploads, and Abaqus INP verification benchmarks with analytical reference checks.

## Current Coverage Map

- Scalar manufactured solutions: 2D/3D Poisson, steady heat, diffusion-reaction, and electrostatics.
- Poisson: 2D `Tri3`, 3D `Tet4`, Dirichlet handling, local stiffness/load references, CG non-convergence.
- Heat: steady 2D/3D and transient 2D theta integration.
- Diffusion-reaction: 2D/3D reaction matrices and transient 2D/3D theta integration.
- Electrostatics: 2D/3D steady/quasi-static scalar potential and formulation marker.
- Elasticity: 2D/3D stiffness symmetry, rigid translations, constrained solves, affine displacement constraints, transient Newmark and HHT-alpha dynamics, explicit central-difference dynamics.
- Nonlinear: J2 plasticity block compression, nonlinear continuum Newton solves, frictionless contact.
- Modal analysis: 2D/3D sorted modes, density validation, constrained model validation, one-DOF frequency references.
- Solver stack: CG, GMRES, dense direct, Jacobi preconditioning, diagnostics, Newton, theta transient, Newmark transient, HHT-alpha transient.
- Import/export: legacy mesh/params/solution, golden solution output, Gmsh physical groups, VTK scalar output, Abaqus INP import, JSON/HDF5 result files, external CAD-to-Gmsh command planning, and CLI CAD plan/run workflows.
- Project workflows: v1 project parsing/validation, CLI validation/inspection golden output, REST validation/solve golden envelopes, async jobs, artifact upload/download fixtures, and stable error schemas.
- Benchmarks: ignored benchmark-style tests cover Poisson assembly, Poisson solve, Gmsh import, VTK export, project parse/validate/adapt, and REST project validate/solve workflows.

## Known Gaps

- Manufactured-solution suites are not yet complete for non-affine structural stress recovery or modal benchmark problems.
- Benchmarks are lightweight smoke tests only; they do not yet provide statistical sampling, persisted baselines, or CI regression thresholds.
- CAD import workflow support currently plans external Gmsh commands only; it does not execute meshers or include full CAD-to-result fixtures.
- REST project jobs and artifacts are in-memory only.
- CI guidance is documented here, but the repository does not yet include a concrete CI configuration file.

Future verification sub-steps should reduce these gaps without weakening the required local gates above.
