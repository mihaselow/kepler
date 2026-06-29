use kepler::{
    Cell, DiffusionReactionProblem, DiffusionReactionProblem3D, ElementKind, LinearSolverBackend,
    LinearSolverOptions, Mesh, MeshTopology, Point2, PointD, SolverOptions,
    TransientDiffusionReactionProblem, TransientDiffusionReactionProblem3D, TransientSolverOptions,
    Tri3,
    fem::diffusion_reaction::{
        assemble_diffusion_reaction_system, local_tet4_reaction, local_tri3_reaction,
    },
    solve_diffusion_reaction, solve_diffusion_reaction_3d, solve_transient_diffusion_reaction,
    solve_transient_diffusion_reaction_3d,
};

const EPS: f64 = 1.0e-12;

#[test]
fn triangle_reaction_matrix_is_consistent_mass_scaled_by_rate() {
    let mesh = unit_triangle();
    let reaction = local_tri3_reaction(&mesh, &mesh.triangles()[0], 6.0);
    let expected = [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]];

    for row in 0..3 {
        for col in 0..3 {
            assert_close(reaction[row][col], expected[row][col]);
        }
    }
}

#[test]
fn tetrahedron_reaction_matrix_is_consistent_mass_scaled_by_rate() {
    let mesh = unit_tetrahedron();
    let reaction = local_tet4_reaction(&mesh, [0, 1, 2, 3], 20.0);

    for (row, values) in reaction.iter().enumerate() {
        for (col, value) in values.iter().enumerate() {
            let expected = if row == col { 1.0 / 3.0 } else { 1.0 / 6.0 };
            assert_close(*value, expected);
        }
    }
}

#[test]
fn zero_reaction_matches_square_poisson_reference() {
    let mesh = square_with_center_mesh();
    let problem = DiffusionReactionProblem {
        diffusivity: 1.0,
        reaction_rate: 0.0,
        source: |_, _| 1.0,
        dirichlet: vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_diffusion_reaction(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_close(result.values[4], 1.0 / 12.0);
}

#[test]
fn reaction_term_reduces_positive_single_tetrahedron_solution() {
    let mesh = unit_tetrahedron();
    let problem = DiffusionReactionProblem3D {
        diffusivity: 1.0,
        reaction_rate: 2.0,
        source: |_, _, _| 1.0,
        dirichlet: vec![(1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_diffusion_reaction_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    assert!(result.values[0] > 0.0);
    assert!(result.values[0] < 1.0 / 12.0);
    assert_close(result.values[1], 0.0);
    assert_close(result.values[2], 0.0);
    assert_close(result.values[3], 0.0);
}

#[test]
fn diffusion_reaction_rejects_invalid_coefficients() {
    let mesh = unit_triangle();
    let problem = DiffusionReactionProblem {
        diffusivity: 1.0,
        reaction_rate: -1.0,
        source: |_, _| 0.0,
        dirichlet: vec![],
    };

    let error = assemble_diffusion_reaction_system(&mesh, &problem).unwrap_err();

    assert!(matches!(
        error,
        kepler::DiffusionReactionError::InvalidReactionRate(-1.0)
    ));
}

#[test]
fn transient_diffusion_reaction_solves_two_dimensional_decay() {
    let mesh = unit_triangle();
    let problem = TransientDiffusionReactionProblem {
        diffusivity: 1.0,
        reaction_rate: 0.0,
        storage_coefficient: 1.0,
        source: |_, _, _| 0.0,
        initial_values: vec![0.0, 1.0, 0.0],
        dirichlet: vec![(0, 0.0), (2, 0.0)],
    };

    let result = solve_transient_diffusion_reaction(
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
    assert_close(result.steps[0].values[0], 0.0);
    assert_close(result.steps[0].values[1], 0.25);
    assert_close(result.steps[0].values[2], 0.0);
    assert_close(result.steps[1].values[1], 0.0625);
    assert_eq!(
        result.steps[0].diagnostics.backend,
        LinearSolverBackend::DenseDirect
    );
}

#[test]
fn transient_diffusion_reaction_solves_three_dimensional_decay() {
    let mesh = unit_tetrahedron();
    let problem = TransientDiffusionReactionProblem3D {
        diffusivity: 1.0,
        reaction_rate: 0.0,
        storage_coefficient: 1.0,
        source: |_, _, _, _| 0.0,
        initial_values: vec![1.0, 0.0, 0.0, 0.0],
        dirichlet: vec![(1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_transient_diffusion_reaction_3d(
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

    assert_close(result.steps[0].values[0], 1.0 / 13.0);
    assert_close(result.steps[0].values[1], 0.0);
    assert_close(result.steps[1].values[0], 1.0 / 169.0);
}

#[test]
fn transient_diffusion_reaction_rejects_invalid_storage_coefficient() {
    let mesh = unit_triangle();
    let problem = TransientDiffusionReactionProblem {
        diffusivity: 1.0,
        reaction_rate: 0.0,
        storage_coefficient: 0.0,
        source: |_, _, _| 0.0,
        initial_values: vec![0.0, 1.0, 0.0],
        dirichlet: vec![],
    };

    let error =
        solve_transient_diffusion_reaction(&mesh, &problem, TransientSolverOptions::default())
            .unwrap_err();

    assert!(matches!(
        error,
        kepler::DiffusionReactionError::InvalidStorageCoefficient(0.0)
    ));
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
