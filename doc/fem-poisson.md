# Minimal FEM Solver

Kepler currently includes a minimal finite element solver for the 2D scalar Poisson equation, suitable as the first bare solver foundation for steady heat diffusion or similar scalar field problems.

```text
-div(k grad u) = f
```

The implementation uses first-order triangular elements (`Tri3`) over a 2D mesh.

## Supported Functionality

- 2D points and triangular connectivity through `Mesh`, `Point2`, and `Tri3`.
- Constant positive scalar conductivity `k`.
- Source term callback `f(x, y)` evaluated at each triangle centroid.
- Dirichlet boundary conditions specified as `(node_id, value)` pairs.
- Sparse global stiffness assembly using `sprs`.
- Conjugate Gradient solve for symmetric positive definite systems.
- Solver diagnostics with iteration count and residual norm.
- File-driven solves from `.mesh` and `.params` inputs.
- `.solution` output with nodal values and diagnostics.

The solver does not yet support Neumann boundaries, spatially varying conductivity, preconditioning, or higher-order elements.

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
2. Builds the local stiffness matrix with `area * k * dot(grad_i, grad_j)`.
3. Builds the local load vector with `f(centroid) * area / 3`.
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
