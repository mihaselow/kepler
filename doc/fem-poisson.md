# Minimal FEM Solver

Kepler currently includes a minimal finite element solver for the scalar Poisson equation, suitable as the first bare solver foundation for steady heat diffusion or similar scalar field problems.

```text
-div(k grad u) = f
```

The implementation supports first-order triangular elements (`Tri3`) over a 2D mesh and first-order tetrahedral elements (`Tet4`) over a 3D topology.

## Supported Functionality

- 2D points and triangular connectivity through `Mesh`, `Point2`, and `Tri3`.
- Dimension-aware mesh core primitives through `PointD`, `Point3`, `Cell`, `ElementKind`, `Region`, and `MeshTopology`.
- Geometry annotation primitives through `EntitySelector`, `GeometryAnnotations`, `MaterialAssignment`, and `ParameterAssignment`.
- Shared condition primitives through `ConditionSet`, `Condition`, and `ConditionKind`.
- Constant positive scalar conductivity `k`.
- Source term callbacks evaluated at each element centroid.
- Dirichlet boundary conditions specified as `(node_id, value)` pairs.
- Sparse global stiffness assembly using `sprs`.
- Conjugate Gradient solve for symmetric positive definite systems.
- Solver diagnostics with iteration count and residual norm.
- File-driven solves from `.mesh` and `.params` inputs.
- `.solution` output with nodal values and diagnostics.
- 3D `Tet4` Poisson assembly and solve through `MeshTopology<3>`.
- Gmsh ASCII 2.x mesh import into `MeshTopology`.
- Legacy VTK unstructured-grid export for topology and point scalar fields.
- REST solves through the separate `server` binary.

The solver does not yet support assembled Neumann boundaries, spatially varying conductivity, preconditioning, `Quad4`/`Hex8` assembly, or higher-order elements. The file-driven CLI and REST endpoint still expose the original 2D `Tri3` solve path.

## Public API

Most users can work with the re-exported crate-level types:

```rust
use kepler::{
    Mesh, Point2, PoissonProblem, SolverOptions, Tri3, solve_poisson,
};
```

The main solve path is:

```rust
let mesh = Mesh::new(points, triangles)?;
let problem = PoissonProblem {
    conductivity: 1.0,
    source: |_, _| 1.0,
    dirichlet: vec![(0, 0.0), (1, 0.0)],
};
let result = solve_poisson(&mesh, &problem, SolverOptions::default())?;
```

`PoissonResult::values` contains one scalar value per mesh node. `iterations` and `residual_norm` report the Conjugate Gradient convergence behavior.

The 3D solve path uses `MeshTopology<3>` and `Tet4` volume cells:

```rust
let problem = PoissonProblem3D {
    conductivity: 1.0,
    source: |x, y, z| 1.0,
    dirichlet: vec![(1, 0.0), (2, 0.0), (3, 0.0)],
};
let result = solve_poisson_3d(&topology, &problem, SolverOptions::default())?;
```

`solve_poisson_3d` assembles `Tet4` cells and ignores lower-dimensional boundary cells currently stored in the topology. `Hex8` cells are validated by the mesh core but are not assembled by Poisson yet.

## Mesh Core

The current Poisson API keeps the original ergonomic 2D types:

```rust
let mesh = Mesh::new(points, triangles)?;
```

Underneath that compatibility layer, the mesh module now exposes platform-oriented primitives:

- `PointD<D>` for dimension-aware coordinates.
- `Point3` for 3D coordinates.
- `ElementKind` for planned element families such as `Line2`, `Tri3`, `Quad4`, `Tet4`, and `Hex8`.
- `Cell` for generic element connectivity.
- `Region` and `EntityDimension` for named topology/geometry targets.
- `MeshTopology<D>` for validating dimension-aware points, cells, and region assignments.

`Mesh::topology()` converts the current 2D triangle mesh into `MeshTopology<2>`. This keeps existing solver behavior stable while preparing future region-targeted loads, CAD imports, and 3D elements.

## Geometry Annotations

The annotation layer lets callers target existing mesh regions by ID or by name:

```rust
let annotations = GeometryAnnotations::new()
    .with_material(MaterialAssignment::new(
        0,
        "steel",
        EntitySelector::region_name("body"),
        vec![
            Parameter::scalar("young_modulus", 210.0e9, Some("Pa")),
            Parameter::scalar("poisson_ratio", 0.3, None::<String>),
        ],
    ))
    .with_parameter(ParameterAssignment::new(
        1,
        "mesh_size",
        EntitySelector::region_id(10),
        ParameterValue::Scalar(0.05),
        Some("m"),
    ));

let resolved = annotations.validate_for_topology(&topology)?;
```

This is a platform foundation for applying future loads, constraints, material models, and solver parameters to named geometry or mesh entities. The current Poisson solver still uses its existing node-based Dirichlet and scalar conductivity API; annotations are validated and resolved separately for now.

## Loads And Conditions

The shared condition layer lets callers describe region-targeted loads and constraints independently from a specific physics assembler:

```rust
let conditions = ConditionSet::new()
    .with_condition(Condition::new(
        0,
        "fixed temperature",
        EntitySelector::region_name("left"),
        ConditionKind::Dirichlet {
            field: "temperature".to_owned(),
            value: ParameterValue::Scalar(300.0),
        },
    ))
    .with_condition(Condition::new(
        1,
        "heat flux",
        EntitySelector::region_name("left"),
        ConditionKind::HeatFlux {
            value: 25.0,
            units: Some("W/m^2".to_owned()),
        },
    ));

let resolved = conditions.validate_for_topology(&topology)?;
```

Supported condition kinds currently include:

- `Dirichlet`
- `Neumann`
- `Robin`
- `PointLoad`
- `BodyLoad`
- `Traction`
- `Pressure`
- `HeatFlux`

Validation resolves region selectors, rejects duplicate condition IDs, rejects duplicate condition signatures on the same region, and checks that point, boundary, and domain loads target regions with compatible dimensions. These conditions are a platform model only at this stage; Poisson assembly does not yet consume `ConditionSet`.

## Mesh Requirements

`Mesh::new` validates the mesh before it can be solved:

- The mesh must contain at least one point.
- Triangle node indices must reference existing points.
- Triangle node indices must be unique within each element.
- Triangles must have non-zero area.

Triangle orientation may be clockwise or counter-clockwise. Element area is treated as positive, while basis gradients are computed from the signed area.

## Assembly Flow

For each triangle, the solver:

1. Computes area, centroid, and P1 basis gradients.
2. Builds the local stiffness matrix with `measure * k * dot(grad_i, grad_j)`.
3. Builds the local load vector using centroid quadrature.
4. Adds local contributions into sparse global triplets.
5. Converts triplets to CSR format.
6. Applies Dirichlet constraints by adjusting unconstrained rows and replacing constrained rows with identity rows.
7. Solves the constrained system with Conjugate Gradient.

## Boundary Conditions

Dirichlet conditions are provided as node/value pairs:

```rust
dirichlet: vec![(0, 0.0), (1, 0.0), (2, 1.0)]
```

The solver rejects boundary entries that reference missing nodes or specify the same node more than once.

## Solver Options

`SolverOptions::default()` uses:

- `max_iterations: 10_000`
- `tolerance: 1.0e-10`

The Conjugate Gradient implementation reports an error for dimension mismatches, non-square matrices, numerical breakdown, or non-convergence.

## Example

Run the square Poisson example:

```shell
cargo run --example poisson_square
```

The example creates a unit square with a center node, applies zero boundary values on the square boundary, and solves with unit source. The expected center value for the current mesh is approximately:

```text
u[4] = 0.083333
```

## File Input And Output

The binary can solve from disk files:

```shell
cargo run -- solve --mesh examples/data/square.mesh --params examples/data/square.params --output square.solution
```

### Mesh Files

Mesh files use the `.mesh` extension by convention and contain `nodes` and `triangles` sections:

```text
nodes
0 0.0 0.0
1 1.0 0.0
2 1.0 1.0
3 0.0 1.0
4 0.5 0.5

triangles
0 0 1 4
1 1 2 4
2 2 3 4
3 3 0 4
```

Node and triangle IDs must be contiguous and start at `0`. Triangle rows use the form `<triangle_id> <node_a> <node_b> <node_c>`.

### Parameter Files

Parameter files use the `.params` extension by convention:

```text
conductivity 1.0
source constant 1.0
solver max_iterations 10000
solver tolerance 1e-10

dirichlet
0 0.0
1 0.0
2 0.0
3 0.0
```

The first file-driven implementation supports only constant source terms:

```text
source constant <value>
```

### Solution Files

Solution files use the `.solution` extension by convention:

```text
# kepler solution
# iterations 1
# residual_norm 0
node value
0 0
1 0
2 0
3 0
4 0.08333333333333333
```

Each data row contains `<node_id> <value>` in node order.

### Gmsh Import

The I/O layer can import Gmsh ASCII 2.x `.msh` files:

```rust
let imported = parse_gmsh_str(input)?;
```

The importer currently supports:

- Gmsh ASCII 2.x format only.
- `Line2`, `Tri3`, `Quad4`, `Tet4`, and `Hex8` element types.
- `$PhysicalNames` preservation as `Region` values.
- Physical entity tags on elements mapped to internal region IDs.
- Automatic 2D import when all node `z` coordinates are approximately zero and no volume elements are present.
- 3D import when volume elements or nonzero `z` coordinates are present.

The return type is `ImportedMesh`, with `TwoD(MeshTopology<2>)` and `ThreeD(MeshTopology<3>)` variants. This importer is a platform feature; the current Poisson solve path still consumes the original 2D `Mesh` type.

### VTK Export

The I/O layer can write legacy VTK unstructured grids:

```rust
let output = format_vtk_legacy(
    &topology,
    &[VtkScalarField::new("temperature", values)],
)?;
```

VTK export currently supports topology points/cells and optional point scalar fields. It writes cell types for `Line2`, `Tri3`, `Quad4`, `Tet4`, and `Hex8`. Scalar fields must contain one value per topology point.

## Verification

Run the full test suite:

```shell
cargo test
```

Run Clippy across all targets:

```shell
cargo clippy --all-targets --all-features
```

The integration tests in `tests/poisson.rs` cover mesh validation, local stiffness properties, Dirichlet handling, the square solve reference value, and CG non-convergence reporting.
