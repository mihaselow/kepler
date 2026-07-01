use kepler::{
    ContactProblem, ContactStaticAssembly, ContactStaticSolverOptions, ElasticityMaterial,
    ElasticityModel, LinearSolverBackend, LinearSolverOptions, Mesh, Point2, solve_contact_static,
};
use kepler::fem::contact::search::BoundarySegment;
use kepler::mesh::{Cell, ElementKind};

/// Slave block initially penetrating the master surface relaxes to gap-free equilibrium.
#[test]
fn penalty_contact_resolves_initial_penetration() {
    let points = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.0, 1.0),
        Point2::new(0.0, 1.0),
        Point2::new(0.0, 0.99),
        Point2::new(1.0, 0.99),
        Point2::new(1.0, 1.99),
        Point2::new(0.0, 1.99),
    ];

    let cells = vec![
        Cell::new(ElementKind::Quad4, vec![0, 1, 2, 3]),
        Cell::new(ElementKind::Quad4, vec![4, 5, 6, 7]),
    ];
    let mesh = Mesh::new_with_cells(points, cells).unwrap();

    let material = ElasticityMaterial {
        young_modulus: 1.0e7,
        poisson_ratio: 0.3,
        model: ElasticityModel::PlaneStrain,
    };

    let dirichlet_boundary = vec![
        (0, 0, 0.0),
        (0, 1, 0.0),
        (1, 0, 0.0),
        (1, 1, 0.0),
        (3, 0, 0.0),
        (4, 0, 0.0),
        (7, 0, 0.0),
    ];

    let contact = ContactProblem {
        master_segments: vec![BoundarySegment { nodes: [3, 2] }],
        slave_nodes: vec![4, 5],
        penalty: 1.0e8,
        use_augmented: false,
    };

    let assembly = ContactStaticAssembly {
        mesh,
        material,
        thickness: 1.0,
        external_forces: Vec::new(),
        dirichlet_boundary,
        contact,
        num_dofs: 16,
    };

    let options = ContactStaticSolverOptions {
        max_newton_iterations: 40,
        max_augmented_iterations: 0,
        tolerance: 1.0e-5,
        penetration_tolerance: 1.0e-6,
        linear_solver: LinearSolverOptions {
            backend: LinearSolverBackend::DenseDirect,
            ..LinearSolverOptions::default()
        },
    };

    let result = solve_contact_static(&assembly, options).unwrap();

    let final_y4 = 0.99 + result.displacements[4 * 2 + 1];
    let final_y5 = 0.99 + result.displacements[5 * 2 + 1];

    assert!(
        final_y4 >= 1.0 - 1.0e-3,
        "slave node 4 below master surface: final y = {final_y4}"
    );
    assert!(
        final_y5 >= 1.0 - 1.0e-3,
        "slave node 5 below master surface: final y = {final_y5}"
    );
    assert!(!result.contact_pairs.is_empty());
}
