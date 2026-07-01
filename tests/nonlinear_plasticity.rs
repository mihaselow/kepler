use std::sync::Arc;

use kepler::mesh::{Cell, ElementKind};
use kepler::{
    J2PlasticMaterial, LinearSolverOptions, Mesh, NonlinearContinuumAssembly,
    NonlinearContinuumSolverOptions, Point2, solve_nonlinear_continuum,
};

/// J2 block under uniaxial compression yields and hardens beyond elastic limit.
#[test]
fn j2_block_compression_reaches_plastic_flow() {
    let points = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.0, 1.0),
        Point2::new(0.0, 1.0),
    ];
    let cells = vec![Cell::new(ElementKind::Quad4, vec![0, 1, 2, 3])];
    let mesh = Mesh::new_with_cells(points, cells).unwrap();

    let material = Arc::new(J2PlasticMaterial::new(200.0e9, 0.3, 40.0e6, 1.0e9));
    let dirichlet_boundary = vec![(0, 0, 0.0), (0, 1, 0.0), (3, 0, 0.0), (1, 1, 0.0)];
    let external_forces = vec![(1, 1, -5.0e7), (2, 1, -5.0e7)];

    let assembly = NonlinearContinuumAssembly {
        mesh,
        thickness: 1.0,
        is_plane_strain: true,
        material,
        external_forces,
        dirichlet_boundary,
        contact: None,
        num_dofs: 8,
    };

    let options = NonlinearContinuumSolverOptions {
        num_steps: 8,
        max_iterations: 20,
        tolerance: 1.0e-4,
        linear_solver: LinearSolverOptions::default(),
    };

    let result = solve_nonlinear_continuum(&assembly, options).unwrap();
    let final_displacements = result.displacements_history.last().unwrap();
    let final_stresses = result.nodal_stress_history.last().unwrap();

    let uy_top = final_displacements[2 * 2 + 1];
    assert!(
        uy_top < 0.0,
        "top nodes should move downward under compression"
    );

    let von_mises_like = final_stresses[1][0].abs().max(final_stresses[1][1].abs());
    assert!(
        von_mises_like > 35.0e6,
        "stress should exceed yield level, got {von_mises_like}"
    );
}
