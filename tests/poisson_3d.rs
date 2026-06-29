use kepler::{
    Cell, ElementKind, MeshTopology, PointD, PoissonProblem3D, SolverOptions,
    fem::poisson::{assemble_poisson_3d_system, local_tet4_load, local_tet4_stiffness},
    solve_poisson_3d,
};

const EPS: f64 = 1.0e-12;

#[test]
fn unit_tetrahedron_stiffness_is_symmetric_with_zero_row_sums() {
    let mesh = unit_tetrahedron();
    let nodes = [0, 1, 2, 3];
    let stiffness = local_tet4_stiffness(&mesh, nodes, 1.0).unwrap();
    let expected = [
        [0.5, -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
        [-1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0],
        [-1.0 / 6.0, 0.0, 1.0 / 6.0, 0.0],
        [-1.0 / 6.0, 0.0, 0.0, 1.0 / 6.0],
    ];

    for row in 0..4 {
        let row_sum: f64 = stiffness[row].iter().sum();
        assert_close(row_sum, 0.0);
        for col in 0..4 {
            assert_close(stiffness[row][col], stiffness[col][row]);
            assert_close(stiffness[row][col], expected[row][col]);
        }
    }
}

#[test]
fn unit_tetrahedron_load_uses_centroid_quadrature() {
    let mesh = unit_tetrahedron();
    let load = local_tet4_load(&mesh, [0, 1, 2, 3], |x, y, z| x + y + z).unwrap();

    for value in load {
        assert_close(value, 1.0 / 32.0);
    }
}

#[test]
fn three_dimensional_assembly_ignores_boundary_surface_cells() {
    let mesh = MeshTopology::<3>::new(
        unit_points(),
        vec![
            Cell::new(ElementKind::Tri3, vec![0, 1, 2]),
            Cell::new(ElementKind::Tet4, vec![0, 1, 2, 3]),
        ],
    )
    .unwrap();
    let problem = PoissonProblem3D {
        conductivity: 1.0,
        source: |_, _, _| 0.0,
        dirichlet: vec![(0, 1.0), (1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let (matrix, rhs) = assemble_poisson_3d_system(&mesh, &problem).unwrap();

    assert_eq!(matrix.rows(), 4);
    assert_eq!(matrix.cols(), 4);
    assert_close(rhs[0], 1.0);
    assert_close(rhs[1], 0.0);
}

#[test]
fn three_dimensional_poisson_solves_single_tetrahedron_reference() {
    let mesh = unit_tetrahedron();
    let problem = PoissonProblem3D {
        conductivity: 1.0,
        source: |_, _, _| 1.0,
        dirichlet: vec![(1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_poisson_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_close(result.values[0], 1.0 / 12.0);
    assert_close(result.values[1], 0.0);
    assert_close(result.values[2], 0.0);
    assert_close(result.values[3], 0.0);
}

#[test]
fn three_dimensional_poisson_rejects_hex8_until_implemented() {
    let mesh = MeshTopology::<3>::new(
        vec![
            PointD::new([0.0, 0.0, 0.0]),
            PointD::new([1.0, 0.0, 0.0]),
            PointD::new([1.0, 1.0, 0.0]),
            PointD::new([0.0, 1.0, 0.0]),
            PointD::new([0.0, 0.0, 1.0]),
            PointD::new([1.0, 0.0, 1.0]),
            PointD::new([1.0, 1.0, 1.0]),
            PointD::new([0.0, 1.0, 1.0]),
        ],
        vec![Cell::new(ElementKind::Hex8, vec![0, 1, 2, 3, 4, 5, 6, 7])],
    )
    .unwrap();
    let problem = PoissonProblem3D {
        conductivity: 1.0,
        source: |_, _, _| 0.0,
        dirichlet: vec![],
    };

    let error = assemble_poisson_3d_system(&mesh, &problem).unwrap_err();

    assert!(matches!(
        error,
        kepler::fem::poisson::PoissonError::UnsupportedElementKind {
            cell_index: 0,
            kind: ElementKind::Hex8,
        }
    ));
}

fn unit_tetrahedron() -> MeshTopology<3> {
    MeshTopology::<3>::new(
        unit_points(),
        vec![Cell::new(ElementKind::Tet4, vec![0, 1, 2, 3])],
    )
    .unwrap()
}

fn unit_points() -> Vec<PointD<3>> {
    vec![
        PointD::new([0.0, 0.0, 0.0]),
        PointD::new([1.0, 0.0, 0.0]),
        PointD::new([0.0, 1.0, 0.0]),
        PointD::new([0.0, 0.0, 1.0]),
    ]
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS,
        "expected {actual} to be within {EPS} of {expected}",
    );
}
