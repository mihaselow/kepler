use kepler::{
    DisplacementComponent, DisplacementConstraint, ElasticityError, ElasticityMaterial,
    ElasticityModel, ElasticityProblem, Mesh, NodalForce, Point2, SolverOptions, Tri3,
    fem::elasticity::{assemble_elasticity_system, local_elasticity_stiffness},
    solve_elasticity,
};

const EPS: f64 = 1.0e-10;

#[test]
fn triangle_elasticity_stiffness_is_symmetric_and_has_rigid_translation_modes() {
    let mesh = unit_triangle();
    let stiffness =
        local_elasticity_stiffness(&mesh, &mesh.triangles()[0], material(), 1.0).unwrap();

    for (row, row_values) in stiffness.iter().enumerate() {
        for (col, value) in row_values.iter().enumerate() {
            assert_close(*value, stiffness[col][row]);
        }
    }

    let x_translation = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_translation = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    assert_matrix_vector_close(stiffness, x_translation, [0.0; 6]);
    assert_matrix_vector_close(stiffness, y_translation, [0.0; 6]);
}

#[test]
fn displacement_constraints_replace_constrained_dofs() {
    let mesh = unit_triangle();
    let problem = ElasticityProblem {
        material: material(),
        thickness: 1.0,
        constraints: vec![DisplacementConstraint {
            node: 0,
            component: DisplacementComponent::X,
            value: 0.25,
        }],
        forces: vec![],
    };

    let (matrix, rhs) = assemble_elasticity_system(&mesh, &problem).unwrap();

    assert_close(rhs[0], 0.25);
    assert_close(matrix.get(0, 0).copied().unwrap_or(0.0), 1.0);
    assert_close(matrix.get(0, 1).copied().unwrap_or(0.0), 0.0);
    assert_close(matrix.get(1, 0).copied().unwrap_or(0.0), 0.0);
}

#[test]
fn elasticity_solves_constrained_triangle_with_nodal_force() {
    let mesh = unit_triangle();
    let problem = ElasticityProblem {
        material: material(),
        thickness: 1.0,
        constraints: vec![
            DisplacementConstraint {
                node: 0,
                component: DisplacementComponent::X,
                value: 0.0,
            },
            DisplacementConstraint {
                node: 0,
                component: DisplacementComponent::Y,
                value: 0.0,
            },
            DisplacementConstraint {
                node: 2,
                component: DisplacementComponent::X,
                value: 0.0,
            },
            DisplacementConstraint {
                node: 2,
                component: DisplacementComponent::Y,
                value: 0.0,
            },
        ],
        forces: vec![NodalForce {
            node: 1,
            fx: 1.0,
            fy: 0.0,
        }],
    };

    let result = solve_elasticity(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_close(result.displacements[0][0], 0.0);
    assert_close(result.displacements[0][1], 0.0);
    assert!(result.displacements[1][0].is_finite());
    assert!(result.displacements[1][0] > 0.0);
    assert_close(result.displacements[2][0], 0.0);
    assert_close(result.displacements[2][1], 0.0);
}

#[test]
fn elasticity_rejects_duplicate_displacement_constraints() {
    let mesh = unit_triangle();
    let problem = ElasticityProblem {
        material: material(),
        thickness: 1.0,
        constraints: vec![
            DisplacementConstraint {
                node: 0,
                component: DisplacementComponent::X,
                value: 0.0,
            },
            DisplacementConstraint {
                node: 0,
                component: DisplacementComponent::X,
                value: 1.0,
            },
        ],
        forces: vec![],
    };

    let error = assemble_elasticity_system(&mesh, &problem).unwrap_err();

    assert_eq!(
        error,
        ElasticityError::DuplicateConstraint {
            node_id: 0,
            component: DisplacementComponent::X,
        }
    );
}

#[test]
fn elasticity_rejects_invalid_material_values() {
    let mesh = unit_triangle();
    let problem = ElasticityProblem {
        material: ElasticityMaterial {
            young_modulus: -1.0,
            poisson_ratio: 0.3,
            model: ElasticityModel::PlaneStress,
        },
        thickness: 1.0,
        constraints: vec![],
        forces: vec![],
    };

    let error = assemble_elasticity_system(&mesh, &problem).unwrap_err();

    assert_eq!(error, ElasticityError::InvalidYoungModulus(-1.0));
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

fn material() -> ElasticityMaterial {
    ElasticityMaterial {
        young_modulus: 100.0,
        poisson_ratio: 0.25,
        model: ElasticityModel::PlaneStress,
    }
}

fn assert_matrix_vector_close(matrix: [[f64; 6]; 6], vector: [f64; 6], expected: [f64; 6]) {
    for row in 0..6 {
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
