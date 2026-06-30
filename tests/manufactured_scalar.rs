use kepler::{
    Cell, DiffusionReactionProblem, DiffusionReactionProblem3D, ElectricPotentialResult,
    ElectrostaticProblem, ElectrostaticProblem3D, ElementKind, Mesh, MeshTopology, Point2, PointD,
    PoissonProblem, PoissonProblem3D, SolverOptions, SteadyHeatProblem, SteadyHeatProblem3D,
    TemperatureResult, Tri3, solve_diffusion_reaction, solve_diffusion_reaction_3d,
    solve_electrostatics, solve_electrostatics_3d, solve_poisson, solve_poisson_3d,
    solve_steady_heat, solve_steady_heat_3d,
};

const EPS: f64 = 1.0e-10;

#[test]
fn poisson_matches_affine_manufactured_solution_in_2d() {
    let mesh = square_with_center_mesh();
    let problem = PoissonProblem {
        conductivity: 3.0,
        source: |_, _| 0.0,
        dirichlet: boundary_values_2d(&mesh, affine_2d),
    };

    let result = solve_poisson(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_scalar_field_matches_2d(&mesh, &result.values, affine_2d);
}

#[test]
fn poisson_matches_affine_manufactured_solution_in_3d() {
    let mesh = unit_tetrahedron();
    let problem = PoissonProblem3D {
        conductivity: 2.0,
        source: |_, _, _| 0.0,
        dirichlet: all_values_3d(&mesh, affine_3d),
    };

    let result = solve_poisson_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_scalar_field_matches_3d(&mesh, &result.values, affine_3d);
}

#[test]
fn steady_heat_matches_affine_manufactured_solution_in_2d() {
    let mesh = square_with_center_mesh();
    let problem = SteadyHeatProblem {
        thermal_conductivity: 4.0,
        heat_generation: |_, _| 0.0,
        prescribed_temperatures: boundary_values_2d(&mesh, affine_2d),
    };

    let result = solve_steady_heat(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_temperature_field_matches_2d(&mesh, &result, affine_2d);
}

#[test]
fn steady_heat_matches_affine_manufactured_solution_in_3d() {
    let mesh = unit_tetrahedron();
    let problem = SteadyHeatProblem3D {
        thermal_conductivity: 5.0,
        heat_generation: |_, _, _| 0.0,
        prescribed_temperatures: all_values_3d(&mesh, affine_3d),
    };

    let result = solve_steady_heat_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_temperature_field_matches_3d(&mesh, &result, affine_3d);
}

#[test]
fn diffusion_reaction_matches_constant_manufactured_solution_in_2d() {
    let mesh = square_with_center_mesh();
    let reaction_rate = 2.0;
    let expected = 2.5;
    let problem = DiffusionReactionProblem {
        diffusivity: 1.5,
        reaction_rate,
        source: move |_, _| reaction_rate * expected,
        dirichlet: boundary_constant_2d(expected),
    };

    let result = solve_diffusion_reaction(&mesh, &problem, SolverOptions::default()).unwrap();

    for value in result.values {
        assert_close(value, expected);
    }
}

#[test]
fn diffusion_reaction_matches_constant_manufactured_solution_in_3d() {
    let mesh = unit_tetrahedron();
    let reaction_rate = 3.0;
    let expected = 1.75;
    let problem = DiffusionReactionProblem3D {
        diffusivity: 2.0,
        reaction_rate,
        source: move |_, _, _| reaction_rate * expected,
        dirichlet: all_constant_3d(&mesh, expected),
    };

    let result = solve_diffusion_reaction_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    for value in result.values {
        assert_close(value, expected);
    }
}

#[test]
fn electrostatics_matches_affine_manufactured_solution_in_2d() {
    let mesh = square_with_center_mesh();
    let problem = ElectrostaticProblem {
        permittivity: 8.85,
        charge_density: |_, _| 0.0,
        prescribed_potentials: boundary_values_2d(&mesh, affine_2d),
    };

    let result = solve_electrostatics(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_potential_field_matches_2d(&mesh, &result, affine_2d);
}

#[test]
fn electrostatics_matches_affine_manufactured_solution_in_3d() {
    let mesh = unit_tetrahedron();
    let problem = ElectrostaticProblem3D {
        permittivity: 1.25,
        charge_density: |_, _, _| 0.0,
        prescribed_potentials: all_values_3d(&mesh, affine_3d),
    };

    let result = solve_electrostatics_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_potential_field_matches_3d(&mesh, &result, affine_3d);
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

fn affine_2d(x: f64, y: f64) -> f64 {
    1.0 + x + 2.0 * y
}

fn affine_3d(x: f64, y: f64, z: f64) -> f64 {
    1.0 + x + 2.0 * y - 0.5 * z
}

fn boundary_values_2d(mesh: &Mesh, exact: fn(f64, f64) -> f64) -> Vec<(usize, f64)> {
    (0..4)
        .map(|node| {
            let point = mesh.points()[node];
            (node, exact(point.x, point.y))
        })
        .collect()
}

fn boundary_constant_2d(value: f64) -> Vec<(usize, f64)> {
    (0..4).map(|node| (node, value)).collect()
}

fn all_values_3d(mesh: &MeshTopology<3>, exact: fn(f64, f64, f64) -> f64) -> Vec<(usize, f64)> {
    mesh.points()
        .iter()
        .enumerate()
        .map(|(node, point)| {
            (
                node,
                exact(point.coords[0], point.coords[1], point.coords[2]),
            )
        })
        .collect()
}

fn all_constant_3d(mesh: &MeshTopology<3>, value: f64) -> Vec<(usize, f64)> {
    (0..mesh.points().len()).map(|node| (node, value)).collect()
}

fn assert_scalar_field_matches_2d(mesh: &Mesh, values: &[f64], exact: fn(f64, f64) -> f64) {
    for (node, point) in mesh.points().iter().enumerate() {
        assert_close(values[node], exact(point.x, point.y));
    }
}

fn assert_scalar_field_matches_3d(
    mesh: &MeshTopology<3>,
    values: &[f64],
    exact: fn(f64, f64, f64) -> f64,
) {
    for (node, point) in mesh.points().iter().enumerate() {
        assert_close(
            values[node],
            exact(point.coords[0], point.coords[1], point.coords[2]),
        );
    }
}

fn assert_temperature_field_matches_2d(
    mesh: &Mesh,
    result: &TemperatureResult,
    exact: fn(f64, f64) -> f64,
) {
    assert_scalar_field_matches_2d(mesh, &result.temperatures, exact);
}

fn assert_temperature_field_matches_3d(
    mesh: &MeshTopology<3>,
    result: &TemperatureResult,
    exact: fn(f64, f64, f64) -> f64,
) {
    assert_scalar_field_matches_3d(mesh, &result.temperatures, exact);
}

fn assert_potential_field_matches_2d(
    mesh: &Mesh,
    result: &ElectricPotentialResult,
    exact: fn(f64, f64) -> f64,
) {
    assert_scalar_field_matches_2d(mesh, &result.potentials, exact);
}

fn assert_potential_field_matches_3d(
    mesh: &MeshTopology<3>,
    result: &ElectricPotentialResult,
    exact: fn(f64, f64, f64) -> f64,
) {
    assert_scalar_field_matches_3d(mesh, &result.potentials, exact);
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS,
        "expected {actual} to be within {EPS} of {expected}",
    );
}
