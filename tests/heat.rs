use kepler::{
    Cell, ElementKind, Mesh, MeshTopology, Point2, PointD, SolverOptions, SteadyHeatProblem,
    SteadyHeatProblem3D, Tri3, solve_steady_heat, solve_steady_heat_3d,
};

const EPS: f64 = 1.0e-12;

#[test]
fn steady_heat_solves_two_dimensional_square_reference() {
    let mesh = square_with_center_mesh();
    let problem = SteadyHeatProblem {
        thermal_conductivity: 1.0,
        heat_generation: |_, _| 1.0,
        prescribed_temperatures: vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_steady_heat(&mesh, &problem, SolverOptions::default()).unwrap();

    for boundary_node in 0..4 {
        assert_close(result.temperatures[boundary_node], 0.0);
    }
    assert_close(result.temperatures[4], 1.0 / 12.0);
}

#[test]
fn steady_heat_solves_three_dimensional_tetrahedron_reference() {
    let mesh = unit_tetrahedron();
    let problem = SteadyHeatProblem3D {
        thermal_conductivity: 1.0,
        heat_generation: |_, _, _| 1.0,
        prescribed_temperatures: vec![(1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_steady_heat_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_close(result.temperatures[0], 1.0 / 12.0);
    assert_close(result.temperatures[1], 0.0);
    assert_close(result.temperatures[2], 0.0);
    assert_close(result.temperatures[3], 0.0);
}

#[test]
fn steady_heat_propagates_invalid_conductivity_errors() {
    let mesh = square_with_center_mesh();
    let problem = SteadyHeatProblem {
        thermal_conductivity: 0.0,
        heat_generation: |_, _| 1.0,
        prescribed_temperatures: vec![],
    };

    let error = solve_steady_heat(&mesh, &problem, SolverOptions::default()).unwrap_err();

    assert!(matches!(
        error,
        kepler::fem::poisson::PoissonError::InvalidConductivity(0.0)
    ));
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
