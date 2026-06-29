use kepler::{
    Cell, ElementKind, LinearSolverBackend, LinearSolverOptions, Mesh, MeshTopology, Point2,
    PointD, SolverOptions, SteadyHeatProblem, SteadyHeatProblem3D, TransientHeatError,
    TransientHeatProblem, TransientSolverOptions, Tri3, solve_steady_heat, solve_steady_heat_3d,
    solve_transient_heat,
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

#[test]
fn transient_heat_uses_solver_stack_theta_integrator() {
    let mesh = unit_triangle();
    let problem = TransientHeatProblem {
        thermal_conductivity: 1.0,
        volumetric_heat_capacity: 1.0,
        heat_generation: |_, _, _| 0.0,
        initial_temperatures: vec![0.0, 1.0, 0.0],
        prescribed_temperatures: vec![(0, 0.0), (2, 0.0)],
    };

    let result = solve_transient_heat(
        &mesh,
        &problem,
        TransientSolverOptions {
            time_step: 1.0,
            steps: 2,
            theta: 1.0,
            linear_solver: LinearSolverOptions {
                backend: LinearSolverBackend::DenseDirect,
                ..LinearSolverOptions::default()
            },
        },
    )
    .unwrap();

    assert_eq!(result.steps.len(), 2);
    assert_close(result.steps[0].time, 1.0);
    assert_close(result.steps[0].temperatures[0], 0.0);
    assert_close(result.steps[0].temperatures[1], 0.25);
    assert_close(result.steps[0].temperatures[2], 0.0);
    assert_close(result.steps[1].temperatures[1], 0.0625);
    assert_eq!(
        result.steps[0].diagnostics.backend,
        LinearSolverBackend::DenseDirect
    );
}

#[test]
fn transient_heat_rejects_invalid_initial_temperature_length() {
    let mesh = unit_triangle();
    let problem = TransientHeatProblem {
        thermal_conductivity: 1.0,
        volumetric_heat_capacity: 1.0,
        heat_generation: |_, _, _| 0.0,
        initial_temperatures: vec![0.0, 1.0],
        prescribed_temperatures: vec![],
    };

    let error =
        solve_transient_heat(&mesh, &problem, TransientSolverOptions::default()).unwrap_err();

    assert_eq!(
        error,
        TransientHeatError::InitialTemperatureLengthMismatch {
            node_count: 3,
            initial_len: 2,
        }
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

fn unit_triangle() -> Mesh {
    Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ],
        vec![Tri3::new([0, 1, 2])],
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
