use kepler::{
    Cell, ELECTROSTATIC_FORMULATION, ElectrostaticFormulation, ElectrostaticProblem,
    ElectrostaticProblem3D, ElementKind, Mesh, MeshTopology, Point2, PointD, SolverOptions, Tri3,
    solve_electrostatics, solve_electrostatics_3d,
};

const EPS: f64 = 1.0e-12;

#[test]
fn electrostatics_solves_two_dimensional_square_reference() {
    let mesh = square_with_center_mesh();
    let problem = ElectrostaticProblem {
        permittivity: 1.0,
        charge_density: |_, _| 1.0,
        prescribed_potentials: vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_electrostatics(&mesh, &problem, SolverOptions::default()).unwrap();

    for boundary_node in 0..4 {
        assert_close(result.potentials[boundary_node], 0.0);
    }
    assert_close(result.potentials[4], 1.0 / 12.0);
}

#[test]
fn electrostatics_solves_three_dimensional_tetrahedron_reference() {
    let mesh = unit_tetrahedron();
    let problem = ElectrostaticProblem3D {
        permittivity: 1.0,
        charge_density: |_, _, _| 1.0,
        prescribed_potentials: vec![(1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_electrostatics_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_close(result.potentials[0], 1.0 / 12.0);
    assert_close(result.potentials[1], 0.0);
    assert_close(result.potentials[2], 0.0);
    assert_close(result.potentials[3], 0.0);
}

#[test]
fn electrostatics_propagates_invalid_permittivity_errors() {
    let mesh = square_with_center_mesh();
    let problem = ElectrostaticProblem {
        permittivity: -1.0,
        charge_density: |_, _| 1.0,
        prescribed_potentials: vec![],
    };

    let error = solve_electrostatics(&mesh, &problem, SolverOptions::default()).unwrap_err();

    assert!(matches!(
        error,
        kepler::fem::poisson::PoissonError::InvalidConductivity(-1.0)
    ));
}

#[test]
fn electrostatics_declares_steady_quasi_static_formulation() {
    assert_eq!(
        ELECTROSTATIC_FORMULATION,
        ElectrostaticFormulation::SteadyQuasiStatic
    );
}

fn square_with_center_mesh() -> Mesh {
    Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.5, 0.5),
        ],
        vec![
            Tri3::new([0, 1, 4]),
            Tri3::new([1, 2, 4]),
            Tri3::new([2, 3, 4]),
            Tri3::new([3, 0, 4]),
        ],
    )
    .unwrap()
}

fn unit_tetrahedron() -> MeshTopology<3> {
    MeshTopology::<3>::new(
        vec![
            PointD::new([0.0, 0.0, 0.0]),
            PointD::new([1.0, 0.0, 0.0]),
            PointD::new([0.0, 1.0, 0.0]),
            PointD::new([0.0, 0.0, 1.0]),
        ],
        vec![Cell::new(ElementKind::Tet4, vec![0, 1, 2, 3])],
    )
    .unwrap()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS,
        "expected {actual} to be within {EPS} of {expected}",
    );
}
