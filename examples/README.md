# Kepler Gmsh Solver & Evaluator Guide

This directory contains examples for loading, solving, and exporting finite element analyses using Gmsh (`.msh`) mesh files.

## Running the Solver Example
You can evaluate any of the generated 2D/3D meshes for multiple physics types (Poisson, Elasticity, Modal Analysis) using the `solve_gmsh` example. The solver automatically applies cantilever-like boundary conditions on the region named `"boundary_left"` and writes a VTK file of the results for visualization (e.g. in ParaView).

### 1. 2D Evaluation
*   **Poisson equation (scalar)**:
    ```bash
    cargo run --example solve_gmsh -- examples/data/square_tri3_fine.msh poisson examples/data/square_tri3_poisson.vtk
    ```
*   **Elasticity (plane stress bending)**:
    ```bash
    cargo run --example solve_gmsh -- examples/data/square_tri3_fine.msh elasticity examples/data/square_tri3_elasticity.vtk
    ```
*   **Modal analysis (vibration mode shapes)**:
    ```bash
    cargo run --example solve_gmsh -- examples/data/square_tri3_small.msh modal examples/data/square_tri3_modal.vtk
    ```

### 2. 3D Evaluation
*   **Poisson equation (scalar)**:
    ```bash
    cargo run --example solve_gmsh -- examples/data/cube_tet4_fine.msh poisson examples/data/cube_tet4_poisson.vtk
    ```
*   **Elasticity (3D cantilever beam)**:
    ```bash
    cargo run --example solve_gmsh -- examples/data/cube_tet4_fine.msh elasticity examples/data/cube_tet4_elasticity.vtk
    ```
*   **Modal analysis (3D vibration mode shapes)**:
    ```bash
    cargo run --example solve_gmsh -- examples/data/cube_tet4_small.msh modal examples/data/cube_tet4_modal.vtk
    ```

> [!NOTE]
> Kepler's dense Jacobi eigenvalue solver scales with $O(N^3)$. For modal analysis, it is recommended to use the `_small` versions of the meshes to ensure fast execution. For static solves (Poisson & Elasticity), the sparse iterative solvers handle the `_fine` versions (e.g. 95,000 cells) in seconds.
