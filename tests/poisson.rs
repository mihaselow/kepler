use kepler::{
    Mesh, MeshError, Point2, PoissonProblem, SolverOptions, Tri3,
    fem::poisson::{assemble_poisson_system, local_stiffness},
    linalg::{LinalgError, conjugate_gradient},
    solve_poisson,
};
use sprs::TriMat;

const EPS: f64 = 1.0e-12;

#[test]
fn mesh_rejects_invalid_triangle_indices() {
    let error = Mesh::new(
        vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
        vec![Tri3::new([0, 1, 2])],
    )
    .unwrap_err();

    assert_eq!(
        error,
        MeshError::InvalidNodeIndex {
            triangle_index: 0,
            node_id: 2,
            node_count: 2,
        }
    );
}

#[test]
fn mesh_rejects_degenerate_triangles() {
    let error = Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ],
        vec![Tri3::new([0, 1, 2])],
    )
    .unwrap_err();

    assert_eq!(error, MeshError::DegenerateTriangle { triangle_index: 0 });
}

#[test]
fn single_triangle_stiffness_is_symmetric_with_zero_row_sums() {
    let mesh = Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ],
        vec![Tri3::new([0, 1, 2])],
    )
    .unwrap();

    let stiffness = local_stiffness(&mesh, &mesh.triangles()[0], 1.0).unwrap();
    let expected = [[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]];

    for row in 0..3 {
        let row_sum: f64 = stiffness[row].iter().sum();
        assert_close(row_sum, 0.0);
        for col in 0..3 {
            assert_close(stiffness[row][col], stiffness[col][row]);
            assert_close(stiffness[row][col], expected[row][col]);
        }
    }
}

#[test]
fn dirichlet_conditions_constrain_rows_and_adjust_rhs() {
    let mesh = Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ],
        vec![Tri3::new([0, 1, 2])],
    )
    .unwrap();
    let problem = PoissonProblem {
        conductivity: 1.0,
        source: |_, _| 0.0,
        dirichlet: vec![(0, 2.0)],
    };

    let (matrix, rhs) = assemble_poisson_system(&mesh, &problem).unwrap();

    assert_close(rhs[0], 2.0);
    assert_close(rhs[1], 1.0);
    assert_close(rhs[2], 1.0);
    assert_close(matrix_value(&matrix, 0, 0), 1.0);
    assert_close(matrix_value(&matrix, 0, 1), 0.0);
    assert_close(matrix_value(&matrix, 1, 0), 0.0);
}

#[test]
fn square_poisson_problem_solves_known_center_value() {
    let mesh = square_with_center_mesh();
    let problem = PoissonProblem {
        conductivity: 1.0,
        source: |_, _| 1.0,
        dirichlet: vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_poisson(&mesh, &problem, SolverOptions::default()).unwrap();

    for boundary_node in 0..4 {
        assert_close(result.values[boundary_node], 0.0);
    }
    assert_close(result.values[4], 1.0 / 12.0);
}

#[test]
fn cg_reports_non_convergence() {
    let mut triplets = TriMat::new((1, 1));
    triplets.add_triplet(0, 0, 1.0);
    let matrix = triplets.to_csr();

    let error = conjugate_gradient(
        &matrix,
        &[1.0],
        SolverOptions {
            max_iterations: 0,
            tolerance: EPS,
        },
    )
    .unwrap_err();

    assert_eq!(
        error,
        LinalgError::NonConverged {
            iterations: 0,
            residual_norm: 1.0,
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

fn matrix_value(matrix: &sprs::CsMat<f64>, row: usize, col: usize) -> f64 {
    matrix.get(row, col).copied().unwrap_or(0.0)
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS,
        "expected {actual} to be within {EPS} of {expected}",
    );
}
