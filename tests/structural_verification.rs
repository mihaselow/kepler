use std::f64::consts::PI;

use kepler::{
    Cell, DisplacementComponent, DisplacementComponent3D, DisplacementConstraint,
    DisplacementConstraint3D, ElasticityMaterial, ElasticityMaterial3D, ElasticityModel,
    ElasticityProblem, ElasticityProblem3D, ElementKind, Mesh, MeshTopology, ModalProblem,
    ModalProblem3D, Point2, PointD, SolverOptions, Tri3,
    fem::elasticity::{local_elasticity_stiffness, local_tet4_elasticity_stiffness},
    solve_elasticity, solve_elasticity_3d, solve_modal, solve_modal_3d,
};

const EPS: f64 = 1.0e-10;

#[test]
fn elasticity_2d_preserves_affine_displacement_constraints() {
    let mesh = unit_triangle();
    let problem = ElasticityProblem {
        material: material_2d(),
        thickness: 1.0,
        constraints: mesh
            .points()
            .iter()
            .enumerate()
            .flat_map(|(node, point)| {
                let [ux, uy] = affine_displacement_2d(point.x, point.y);
                [
                    DisplacementConstraint {
                        node,
                        component: DisplacementComponent::X,
                        value: ux,
                    },
                    DisplacementConstraint {
                        node,
                        component: DisplacementComponent::Y,
                        value: uy,
                    },
                ]
            })
            .collect(),
        forces: vec![],
    };

    let result = solve_elasticity(&mesh, &problem, SolverOptions::default()).unwrap();

    for (node, point) in mesh.points().iter().enumerate() {
        let expected = affine_displacement_2d(point.x, point.y);
        assert_close(result.displacements[node][0], expected[0]);
        assert_close(result.displacements[node][1], expected[1]);
    }
}

#[test]
fn elasticity_3d_preserves_affine_displacement_constraints() {
    let mesh = unit_tetrahedron();
    let problem = ElasticityProblem3D {
        material: material_3d(),
        constraints: mesh
            .points()
            .iter()
            .enumerate()
            .flat_map(|(node, point)| {
                let [ux, uy, uz] =
                    affine_displacement_3d(point.coords[0], point.coords[1], point.coords[2]);
                [
                    DisplacementConstraint3D {
                        node,
                        component: DisplacementComponent3D::X,
                        value: ux,
                    },
                    DisplacementConstraint3D {
                        node,
                        component: DisplacementComponent3D::Y,
                        value: uy,
                    },
                    DisplacementConstraint3D {
                        node,
                        component: DisplacementComponent3D::Z,
                        value: uz,
                    },
                ]
            })
            .collect(),
        forces: vec![],
    };

    let result = solve_elasticity_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    for (node, point) in mesh.points().iter().enumerate() {
        let expected = affine_displacement_3d(point.coords[0], point.coords[1], point.coords[2]);
        assert_close(result.displacements[node][0], expected[0]);
        assert_close(result.displacements[node][1], expected[1]);
        assert_close(result.displacements[node][2], expected[2]);
    }
}

#[test]
fn local_elasticity_matrices_have_rigid_translation_null_modes() {
    let mesh = unit_triangle();
    let stiffness =
        local_elasticity_stiffness(&mesh, &mesh.triangles()[0], material_2d(), 1.0).unwrap();

    assert_matrix_vector_close_2d(stiffness, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0; 6]);
    assert_matrix_vector_close_2d(stiffness, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [0.0; 6]);

    let mesh_3d = unit_tetrahedron();
    let stiffness_3d =
        local_tet4_elasticity_stiffness(&mesh_3d, [0, 1, 2, 3], material_3d()).unwrap();
    assert_matrix_vector_close_3d(
        stiffness_3d,
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0; 12],
    );
    assert_matrix_vector_close_3d(
        stiffness_3d,
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0; 12],
    );
    assert_matrix_vector_close_3d(
        stiffness_3d,
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0; 12],
    );
}

#[test]
fn modal_2d_matches_one_dof_frequency_reference() {
    let mesh = unit_triangle();
    let density = 2.0;
    let thickness = 1.0;
    let constraints = all_dofs_except_node_component_2d(1, DisplacementComponent::X);
    let stiffness =
        local_elasticity_stiffness(&mesh, &mesh.triangles()[0], material_2d(), thickness).unwrap();
    let active_stiffness = stiffness[2][2];
    let active_mass = density * thickness * 0.5 / 3.0;
    let expected_frequency = (active_stiffness / active_mass).sqrt() / (2.0 * PI);
    let problem = ModalProblem {
        elasticity: ElasticityProblem {
            material: material_2d(),
            thickness,
            constraints,
            forces: vec![],
        },
        density,
        mode_count: 1,
    };

    let result = solve_modal(&mesh, &problem).unwrap();

    assert_close(result.modes[0].frequency_hz, expected_frequency);
    assert!(result.modes[0].displacements[1][0].abs() > 0.0);
    assert_close(result.modes[0].displacements[1][1], 0.0);
}

#[test]
fn modal_3d_matches_one_dof_frequency_reference() {
    let mesh = unit_tetrahedron();
    let density = 24.0;
    let constraints = all_dofs_except_node_component_3d(1, DisplacementComponent3D::X);
    let stiffness = local_tet4_elasticity_stiffness(&mesh, [0, 1, 2, 3], material_3d()).unwrap();
    let active_stiffness = stiffness[3][3];
    let active_mass = density * (1.0 / 6.0) / 4.0;
    let expected_frequency = (active_stiffness / active_mass).sqrt() / (2.0 * PI);
    let problem = ModalProblem3D {
        elasticity: ElasticityProblem3D {
            material: material_3d(),
            constraints,
            forces: vec![],
        },
        density,
        mode_count: 1,
    };

    let result = solve_modal_3d(&mesh, &problem).unwrap();

    assert_close(result.modes[0].frequency_hz, expected_frequency);
    assert!(result.modes[0].displacements[1][0].abs() > 0.0);
    assert_close(result.modes[0].displacements[1][1], 0.0);
    assert_close(result.modes[0].displacements[1][2], 0.0);
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

fn material_2d() -> ElasticityMaterial {
    ElasticityMaterial {
        young_modulus: 100.0,
        poisson_ratio: 0.25,
        model: ElasticityModel::PlaneStress,
    }
}

fn material_3d() -> ElasticityMaterial3D {
    ElasticityMaterial3D {
        young_modulus: 100.0,
        poisson_ratio: 0.25,
    }
}

fn affine_displacement_2d(x: f64, y: f64) -> [f64; 2] {
    [0.1 + 0.2 * x - 0.05 * y, -0.3 + 0.15 * x + 0.1 * y]
}

fn affine_displacement_3d(x: f64, y: f64, z: f64) -> [f64; 3] {
    [
        0.1 + 0.2 * x - 0.05 * y + 0.03 * z,
        -0.3 + 0.15 * x + 0.1 * y - 0.02 * z,
        0.2 - 0.04 * x + 0.07 * y + 0.12 * z,
    ]
}

fn all_dofs_except_node_component_2d(
    active_node: usize,
    active_component: DisplacementComponent,
) -> Vec<DisplacementConstraint> {
    let mut constraints = Vec::new();
    for node in 0..3 {
        for component in [DisplacementComponent::X, DisplacementComponent::Y] {
            if node == active_node && component == active_component {
                continue;
            }
            constraints.push(DisplacementConstraint {
                node,
                component,
                value: 0.0,
            });
        }
    }
    constraints
}

fn all_dofs_except_node_component_3d(
    active_node: usize,
    active_component: DisplacementComponent3D,
) -> Vec<DisplacementConstraint3D> {
    let mut constraints = Vec::new();
    for node in 0..4 {
        for component in [
            DisplacementComponent3D::X,
            DisplacementComponent3D::Y,
            DisplacementComponent3D::Z,
        ] {
            if node == active_node && component == active_component {
                continue;
            }
            constraints.push(DisplacementConstraint3D {
                node,
                component,
                value: 0.0,
            });
        }
    }
    constraints
}

fn assert_matrix_vector_close_2d(matrix: [[f64; 6]; 6], vector: [f64; 6], expected: [f64; 6]) {
    for row in 0..6 {
        let actual: f64 = matrix[row]
            .iter()
            .zip(vector)
            .map(|(value, vector_value)| value * vector_value)
            .sum();
        assert_close(actual, expected[row]);
    }
}

fn assert_matrix_vector_close_3d(matrix: [[f64; 12]; 12], vector: [f64; 12], expected: [f64; 12]) {
    for row in 0..12 {
        let actual: f64 = matrix[row]
            .iter()
            .zip(vector)
            .map(|(value, vector_value)| value * vector_value)
            .sum();
        assert_close(actual, expected[row]);
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS,
        "expected {actual} to be within {EPS} of {expected}",
    );
}
