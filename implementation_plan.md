# Kepler MVP: 50% Commercial Solver Parity

## Goal

Extend Kepler to provide a minimum viable product that delivers approximately **50% of the solver functionality** offered by ANSYS MAPDL and LS-DYNA. This document assesses the gap, proposes a phased roadmap, and describes the architectural changes required.

* **Execution Directive**: Implementations will be split into small, manageable sub-steps. Upon completion of each sub-step, the agent will provide a corresponding extensive commit message and run `cargo clean` to reclaim disk space.

---

## Competitive Gap Analysis

### What MAPDL and LS-DYNA Provide (representative 50% target scope)

| Capability Area | MAPDL/LS-DYNA (representative) | Kepler Today | Gap |
| :--- | :--- | :--- | :--- |
| **Parallelism** | Shared-memory OpenMP, distributed MPI | None (serial) | ❌ Critical |
| **Element library** | 200+ element types (shells, beams, solid, contact) | Tri3, Tet4, Tri6, Tet10, Quad4/8, Hex8/20 (many unimplemented as FEM solvers) | ❌ Large |
| **Shell elements** | SHELL181, SHELL281 (mindlin, kirchhoff) | Only 2D Beam/Truss1D; no plate/shell | ❌ Critical |
| **Contact** | Frictionless, frictional, bonded (penalty, augmented Lagrangian) | MPC linear constraints only | ❌ Critical |
| **Material models** | Elastic, bilinear plasticity, hyperelastic, creep, failure | Isotropic linear elastic only | ❌ Critical |
| **Nonlinear static** | Full Newton-Raphson with arc-length, load stepping | Newton-Raphson for trusses only | ❌ Large |
| **Transient dynamics** | Newmark, HHT-alpha, central difference explicit | Newmark (implicit) only | ⚠️ Partial |
| **Thermal coupling** | Thermal-structural coupling (ANSYS multi-physics) | Separate FEM modules; no coupling | ❌ Large |
| **Modal/Dynamics** | Lanczos eigensolver, QRDAMP, response spectrum | Dense Jacobi only | ❌ Large |
| **Linear solver** | PARDISO, MUMPS, PCG with AMG | CG, GMRES, dense, sparse LDL, Jacobi precond | ⚠️ Partial |
| **Output** | RST, HDF5, VTK, Paraview, stress/strain tensors | VTK scalar, custom .solution | ❌ Large |
| **Input** | APDL macros, external meshes, .cdb, Abaqus .inp | Custom .mesh/.params, Gmsh .msh, REST JSON | ❌ Large |
| **Stress recovery** | Full tensor, von Mises, principal stresses, averaging | None (displacements only) | ❌ Critical |
| **CI/Parallelism** | Certified, validated, multi-threaded assembly | Single-threaded assembly loops | ❌ Critical |

> [!IMPORTANT]
> The most impactful deficiencies for real-world structural simulations are: **(1) parallel assembly/solve, (2) shell elements, (3) plasticity, (4) contact, (5) stress recovery, and (6) Lanczos eigensolver.** These six areas define the critical path.

---

## Architecture Overview After MVP

```
kepler (library crate)
├── linalg/        ← parallel assembly, AMG precond, Lanczos, direct UMFPACK hook
├── mesh/          ← Hex8/Hex20, shell surface extraction, node renumbering
├── fem/
│   ├── elasticity/    ← isotropic + orthotropic materials, stress recovery
│   ├── plasticity/    ← J2 von Mises return mapping (bilinear isotropic hardening)
│   ├── shell/         ← new: Mindlin-Reissner MITC4 shell element
│   ├── contact/       ← new: penalty & augmented Lagrangian node-to-segment
│   ├── thermal_struct/← new: thermomechanical coupling module
│   ├── modal/         ← replace Jacobi with Lanczos eigensolver
│   └── nonlinear/     ← generalize Newton to continuum elements + arc-length
├── io/
│   ├── abaqus.rs      ← new: .inp import
│   ├── hdf5.rs        ← new: HDF5 output
│   └── rst.rs         ← new: Kepler native result file (HDF5-backed)
└── parallel/      ← new: Rayon-based element loop partitioning
```

---

## Phase 1 — Parallel Assembly & Core Infrastructure (Months 1–3) [COMPLETED]

This phase provides the performance foundation that all other phases depend on.

> [!IMPORTANT]
> Without shared-memory parallelism, Kepler cannot handle industrial mesh sizes (>500k DOFs). This is the highest-priority single change.

### 1.1 [x] Rayon Parallel Element Assembly

**Add `rayon` to `Cargo.toml`** and restructure all element stiffness loops to use parallel iterators.

- **Strategy**: Each element computes its local stiffness matrix independently in a `par_iter()` loop, accumulating `TriMat` triplets into thread-local vectors that are merged after all elements complete.
- **Affected modules**: `fem/poisson.rs`, `fem/elasticity.rs`, `fem/heat.rs`, `fem/diffusion_reaction.rs`, `fem/modal.rs`.
- **Expected speedup**: ~6–12× on a 16-core workstation for assembly of meshes with >100k elements.
- **No OpenMP required**: Rust's `rayon` crate provides data-race-free work-stealing parallelism without the pitfalls of C/C++ OpenMP. This is idiomatic and safe.

```toml
[dependencies]
rayon = "1.10"
```

```rust
// Conceptual parallel assembly pattern:
let triplets: Vec<_> = elements
    .par_iter()
    .flat_map(|elem| {
        let k_local = elem.local_stiffness(&coords, &properties)?;
        assemble_triplets(elem.nodes(), k_local)
    })
    .collect();
```

#### Files to Modify
- [fem/poisson.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/poisson.rs)
- [fem/elasticity.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/elasticity.rs)
- [fem/heat.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/heat.rs)
- [fem/diffusion_reaction.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/diffusion_reaction.rs)
- [Cargo.toml](file:///Users/michaelhaselow/_devel/kepler/Cargo.toml)
- **[NEW]** `src/parallel/mod.rs` — assembly partitioner and triplet merging utilities.

### 1.2 [x] Algebraic Multigrid (AMG) Preconditioner

Replace Jacobi diagonal scaling with a proper AMG preconditioner as a plugin for the `PreconditionerKind` enum.

- **Approach**: Implement a simple smoothed-aggregation AMG.
- **Target**: Enable CG+AMG to solve 1M-DOF elasticity problems in seconds on a laptop.
- **Fallback**: Keep `PreconditionerKind::Jacobi` as default; add `PreconditionerKind::Amg` as a solver option.

#### Files to Modify
- [src/linalg.rs](file:///Users/michaelhaselow/_devel/kepler/src/linalg.rs) — add `PreconditionerKind::Amg`, solver dispatch.

### 1.3 [x] Lanczos Eigensolver for Modal Analysis

Replace the current dense Jacobi iteration (O(N³)) with a sparse Lanczos eigensolver for practical large-scale modal analysis.

- **Approach**: Implement a shift-invert Lanczos using the `sparse-ldl` backend.
- **Expected**: Modal analysis of >10k-DOF models in seconds instead of minutes.
- **API change**: Backward compatible — `solve_modal`/`solve_modal_3d` signature unchanged, implementation swapped.

#### Files to Modify
- [fem/modal.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/modal.rs)
- [src/linalg.rs](file:///Users/michaelhaselow/_devel/kepler/src/linalg.rs) — add `solve_lanczos_modes`.

### 1.4 [x] Full Stress/Strain Tensor Recovery

All current elasticity solvers return only nodal displacement vectors. Post-processing for engineering use requires stress and strain.

- **Add**: `StressTensor2D { sigma_xx, sigma_yy, sigma_xy, von_mises, principal }` and 3D equivalent.
- **Add**: `StrainTensor2D`, `StrainTensor3D`.
- **Compute**: Element-centroid Gauss-point stress from `B * u` using B-matrix functions; project to nodes using simple averaging.
- **Output**: Extend VTK export to write stress components as cell or point data fields.

#### Files to Modify
- [fem/elasticity.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/elasticity.rs) — add `StressTensor2D/3D`, `StrainTensor2D/3D`, `ElasticityResult` extension.
- [io/vtk.rs](file:///Users/michaelhaselow/_devel/kepler/src/io/vtk.rs) — add `VtkVectorField`, extend `format_vtk_legacy`.

---

## Phase 2 — Shell Elements & Advanced Element Library (Months 3–6)

Shell elements are the single largest gap blocking industrial structural simulation. Plate/shell-dominated structures (aerospace panels, automotive body, pressure vessels) require them.

### Phase 2A — 2D Quadrilateral (Quad4) & Quadratic Triangle (Tri6) Elasticity [COMPLETED]

- **Quad4 plane stress/strain**: Implement 4-node bilinear quadrilateral element routines using 2×2 Gauss quadrature.
- **Tri6 elasticity**: Extend 2D plane elasticity to support 6-node quadratic triangle elements.
- **Verification**: Patch tests under uniform tension and shear.

#### Files Modified/Created
- [fem/elasticity.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/elasticity.rs)
- [tests/elasticity.rs](file:///Users/michaelhaselow/_devel/kepler/tests/elasticity.rs)
- [mesh.rs](file:///Users/michaelhaselow/_devel/kepler/src/mesh.rs)

### Phase 2B — 3D Hexahedral (Hex8) Solid Element [COMPLETED]

- **Hex8 solid elasticity**: Implement 8-node trilinear solid element routines using 2×2×2 Gauss quadrature.
- **System assembly**: Dispatch Hex8 local stiffness inside `assemble_elasticity_3d_system`.
- **Verification**: Bending and tension of a structured Hex8 cantilever block.

#### Files Modified/Created
- [fem/elasticity.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/elasticity.rs)
- [tests/elasticity_3d.rs](file:///Users/michaelhaselow/_devel/kepler/tests/elasticity_3d.rs)
- [fem/modal.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/modal.rs)
- [tests/modal.rs](file:///Users/michaelhaselow/_devel/kepler/tests/modal.rs)

### Phase 2C — 3D Euler-Bernoulli Beam Element [COMPLETED]

- **3D Beam element**: Implement 2-node beam element (6 DOFs/node) stiffness matrix.
- **Coordinate transformation**: Apply 3D coordinate transformations for global assembly.
- **Verification**: Shear force and bending moment diagrams for 3D frames.

#### Files Modified/Created
- [fem/structural.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/structural.rs)
- [tests/structural_verification.rs](file:///Users/michaelhaselow/_devel/kepler/tests/structural_verification.rs)
- [lib.rs](file:///Users/michaelhaselow/_devel/kepler/src/lib.rs)

### Phase 2D — MITC4 Mindlin-Reissner Shell Element [COMPLETED]

- **MITC4 shell element**: Implement a 4-node flat shell element (6 DOFs/node) using Mixed Interpolation of Tensorial Components to prevent shear locking.
- **Verification**: Pins-and-cylinder, Scordelis-Lo roof shell benchmark tests.

#### Files Modified/Created
- [fem/structural.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/structural.rs)
- [lib.rs](file:///Users/michaelhaselow/_devel/kepler/src/lib.rs)

### Phase 2E — REST API & I/O Mapping [COMPLETED]

- **I/O extension**: Map Quad4, Tri6, Hex8, beam, and shell elements in the project schema.
- **REST Endpoints**: Expose solve handlers for the new element solver configurations.

#### Files Modified/Created
- [src/io/project.rs](file:///Users/michaelhaselow/_devel/kepler/src/io/project.rs)
- [src/bin/server.rs](file:///Users/michaelhaselow/_devel/kepler/src/bin/server.rs)
- [src/lib.rs](file:///Users/michaelhaselow/_devel/kepler/src/lib.rs)
- [src/main.rs](file:///Users/michaelhaselow/_devel/kepler/src/main.rs)
- [tests/project_workflow.rs](file:///Users/michaelhaselow/_devel/kepler/tests/project_workflow.rs)
- [examples/data/cli_project_inspect_summary.txt](file:///Users/michaelhaselow/_devel/kepler/examples/data/cli_project_inspect_summary.txt)

---

## Phase 3 — Nonlinear Material Models & Contact (Months 6–10)

This phase transforms Kepler from a linear-only solver to a genuinely nonlinear structural code, closing the most significant gap with MAPDL and LS-DYNA.

### Phase 3A — Material Model Trait & Gauss Point Infrastructure [COMPLETED]
- **Deliverables**: Create `MaterialModel` trait and `MaterialState` infrastructure.
- **Files Modified/Created**:
  - [src/fem/material/mod.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/material/mod.rs)
  - [src/fem/mod.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/mod.rs)

### Phase 3B — J2 Plasticity Bilinear Isotropic Hardening [COMPLETED]
- **Deliverables**: Implement von Mises radial return mapping constitutive equations, tracking plastic strains and consistent tangent operator.
- **Files Modified/Created**:
  - [src/fem/material/plasticity.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/material/plasticity.rs)
  - [src/lib.rs](file:///Users/michaelhaselow/_devel/kepler/src/lib.rs)

### Phase 3C — Arc-Length (Riks) Solver [COMPLETED]
- **Deliverables**: Implement path-following Riks arc-length constraint algorithm in linear solver tools to trace snap-through buckling.
- **Files Modified/Created**:
  - [src/linalg.rs](file:///Users/michaelhaselow/_devel/kepler/src/linalg.rs)
  - [src/fem/nonlinear.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/nonlinear.rs)

### Phase 3D — Generalized Newton-Raphson Driver [COMPLETED]
- **Deliverables**: Implement incremental load-stepping Newton-Raphson driver for 2D/3D continuum elements wrapping Gauss-point history tracking.
- **Files Modified/Created**:
  - [src/fem/nonlinear_continuum.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/nonlinear_continuum.rs)
  - [src/fem/mod.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/mod.rs)
  - [src/lib.rs](file:///Users/michaelhaselow/_devel/kepler/src/lib.rs)
  - [src/linalg.rs](file:///Users/michaelhaselow/_devel/kepler/src/linalg.rs)

### Phase 3E — Spatial Hashing Contact Detection [COMPLETED]
- **Deliverables**: Implement boundary facet extraction and spatial hashing contact search candidate algorithm.
- **Files Modified/Created**:
  - [src/fem/contact/search.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/contact/search.rs)
  - [src/fem/contact/mod.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/contact/mod.rs)
  - [src/fem/mod.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/mod.rs)
  - [src/lib.rs](file:///Users/michaelhaselow/_devel/kepler/src/lib.rs)

### Phase 3F — Frictionless Penalty Contact Enforcement [COMPLETED]
- **Deliverables**: Implement node-to-segment frictionless contact stiffness and force assembly via penalty method.
- **Files Modified/Created**:
  - [src/fem/contact/penalty.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/contact/penalty.rs)
  - [src/fem/contact/mod.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/contact/mod.rs)
  - [src/lib.rs](file:///Users/michaelhaselow/_devel/kepler/src/lib.rs)

### Phase 3H — Frictionless Augmented Lagrangian Contact [COMPLETED]

### Phase 3I — Nonlinear Continuum Newton Driver [COMPLETED]

### Phase 3J — Project Schema for Nonlinear & Contact [COMPLETED]
- **Deliverables**: `ProjectPhysics::NonlinearElasticity`, `ProjectPhysics::Contact`, `job_to_nonlinear_continuum`, `job_to_contact`.
- **Files Modified/Created**:
  - [src/io/project.rs](file:///Users/michaelhaselow/_devel/kepler/src/io/project.rs)
  - [src/lib.rs](file:///Users/michaelhaselow/_devel/kepler/src/lib.rs)

### Phase 3K — J2 Plasticity Verification [COMPLETED]
- **Deliverables**: J2 block compression integration test.
- **Files Modified/Created**:
  - [tests/nonlinear_plasticity.rs](file:///Users/michaelhaselow/_devel/kepler/tests/nonlinear_plasticity.rs)
  - [tests/contact.rs](file:///Users/michaelhaselow/_devel/kepler/tests/contact.rs)

---

## Phase 4 — Multi-Physics Coupling & Advanced Dynamics (Months 10–14)

### 4.1  Thermal-Structural Coupling [COMPLETED]

Implement loosely-coupled thermoelastic analysis:

1. Solve steady-state or transient heat problem → nodal temperatures `T(x)`.
2. Compute thermal strain `epsilon_thermal = alpha * (T - T_ref)` per element.
3. Apply as equivalent nodal forces in elasticity: `f_thermal = integral(B^T D epsilon_thermal dV)`.
4. Solve elasticity problem with combined mechanical + thermal loading.

- **Strongly coupled option**: Alternating staggered iterations until convergence.
- **API**: `ThermoElasticProblem { heat_problem, elasticity_problem, thermal_expansion_coeff, ref_temp }`.

#### New Files
- **[NEW]** `src/fem/thermal_struct.rs` — `ThermoElasticProblem`, `ThermoElasticResult`, `solve_thermoelastic`.

### 4.2  HHT-α Transient Integrator [COMPLETED]

Add the Hilber-Hughes-Taylor-α method as an alternative to Newmark for improved numerical damping without accuracy loss. This is the method used by MAPDL's transient structural solver.

#### Files Modified/Created
- [src/linalg.rs](file:///Users/michaelhaselow/_devel/kepler/src/linalg.rs) — `solve_hht_transient`, `HhtSolverOptions`.
- [fem/elasticity.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/elasticity.rs) — `solve_transient_elasticity_hht`.
- [tests/transient_coverage.rs](file:///Users/michaelhaselow/_devel/kepler/tests/transient_coverage.rs)

### 4.3  Central Difference Explicit Dynamics (LS-DYNA style) [COMPLETED]

For highly dynamic problems (crash, blast, impact), explicit time integration with a lumped mass matrix and conditionally stable step size is far more efficient than implicit Newmark.

#### Files Modified/Created
- [src/fem/explicit.rs](file:///Users/michaelhaselow/_devel/kepler/src/fem/explicit.rs) — `solve_explicit_dynamics`, central difference loop.

---

## Phase 5 — I/O Ecosystem & REST API Expansion (Months 14–18)

### 5.1  Abaqus `.inp` Import [COMPLETED]

#### Files Modified/Created
- [src/io/abaqus.rs](file:///Users/michaelhaselow/_devel/kepler/src/io/abaqus.rs) — `parse_abaqus_str`, `read_abaqus_file`.
- [tests/abaqus_import.rs](file:///Users/michaelhaselow/_devel/kepler/tests/abaqus_import.rs)

### 5.2  HDF5 / JSON Result Files [COMPLETED]

HDF5 is available behind the optional `hdf5` feature; JSON result export is the default portable format.

```toml
[features]
hdf5 = ["dep:hdf5"]
```

#### Files Modified/Created
- [src/io/result_format.rs](file:///Users/michaelhaselow/_devel/kepler/src/io/result_format.rs) — `write_json_result`, `write_hdf5_result`, `KeplerResultFile`.

### 5.3  REST API — General Physics Endpoints [COMPLETED]

Extend the REST server to expose all current physics solvers:

- `POST /solve/elasticity` — 2D/3D linear elastic analysis.
- `POST /solve/heat` — steady/transient heat transfer.
- `POST /solve/modal` — free vibration modes and frequencies.
- `POST /solve/diffusion` — diffusion-reaction problems.

#### Files Modified/Created
- [src/bin/server.rs](file:///Users/michaelhaselow/_devel/kepler/src/bin/server.rs)

### 5.4  Documentation & Verification [COMPLETED]

#### Files Modified/Created
- [doc/verification.md](file:///Users/michaelhaselow/_devel/kepler/doc/verification.md)
- [doc/rest-api.md](file:///Users/michaelhaselow/_devel/kepler/doc/rest-api.md)
- [tests/verification_manifest.rs](file:///Users/michaelhaselow/_devel/kepler/tests/verification_manifest.rs)

---

## Open Questions & Design Decisions

> [!NOTE]
> **Q1 — Parallelism model**: MPI and distributed-memory HPC capability is required. In addition to Rayon (shared-memory), we will incorporate MPI scaffolding in later phases.
> 
> **Q2 — Solver bindings**: Pure Rust implementations are preferred for solver backends to avoid external C-library dependency build complexities.
> 
> **Q3 — Plasticity scope**: Both J2 isotropic hardening and kinematic hardening, as well as extensible material plugins (UMAT-style), are in scope.
> 
> **Q4 — Contact priority**: Coulomb friction is deferred. The priority is frictionless penalty and augmented Lagrangian contact.
> 
> **Q5 — Explicit dynamics**: Explicit dynamics (LS-DYNA style crash/impact central difference integration) is in scope for the MVP.

---

## Summary Roadmap

| Phase | Duration | Key Deliverables | MAPDL/LS-DYNA Parity Gained |
| :--- | :--- | :--- | :--- |
| **1** — Parallel + Perf | 3 months | Rayon assembly, Lanczos modal, AMG precond, stress tensors | ~10% → ~25% |
| **2** — Element Library | 3 months | MITC4 shell, Hex8, Quad4, 3D beam, Tri6 elasticity | ~25% → ~35% |
| **3** — Nonlinear + Contact | 4 months | J2 plasticity, Riks arc-length, node-to-segment contact | ~35% → ~45% |
| **4** — Multi-physics | 4 months | Thermoelastic, HHT-α, explicit central difference | ~45% → ~50% |
| **5** — I/O + REST | 4 months | Abaqus INP, HDF5/XDMF, full REST API for all physics | +5% ecosystem reach |

**Total estimated effort to MVP: ~18 months** (1–2 engineers).
