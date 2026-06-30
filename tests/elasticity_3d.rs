use kepler::{
    Cell, DisplacementComponent3D, DisplacementConstraint3D, ElasticityError, ElasticityMaterial3D,
    ElasticityProblem3D, ElementKind, LinearSolverBackend, LinearSolverOptions, MeshTopology,
    NewmarkSolverOptions, NodalForce3D, PointD, SolverOptions, TransientElasticityProblem3D,
    fem::elasticity::{assemble_elasticity_3d_system, local_tet4_elasticity_stiffness},
    solve_elasticity_3d, solve_elasticity_3d_with_solver, solve_transient_elasticity_3d,
};

const EPS: f64 = 1.0e-10;

#[test]
fn tet4_elasticity_stiffness_is_symmetric_and_has_rigid_translation_modes() {
    let mesh = unit_tetrahedron();
    let stiffness = local_tet4_elasticity_stiffness(&mesh, [0, 1, 2, 3], material()).unwrap();

    for (row, row_values) in stiffness.iter().enumerate() {
        for (col, value) in row_values.iter().enumerate() {
            assert_close(*value, stiffness[col][row]);
        }
    }

    let x_translation = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let y_translation = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let z_translation = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    assert_matrix_vector_close(stiffness, x_translation, [0.0; 12]);
    assert_matrix_vector_close(stiffness, y_translation, [0.0; 12]);
    assert_matrix_vector_close(stiffness, z_translation, [0.0; 12]);
}

#[test]
fn elasticity_3d_solves_constrained_tetrahedron_with_nodal_force() {
    let mesh = unit_tetrahedron();
    let problem = ElasticityProblem3D {
        material: material(),
        constraints: fixed_nodes(&[0, 2, 3]),
        forces: vec![NodalForce3D {
            node: 1,
            fx: 1.0,
            fy: 0.0,
            fz: 0.0,
        }],
    };

    let result = solve_elasticity_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    assert_close(result.displacements[0][0], 0.0);
    assert_close(result.displacements[0][1], 0.0);
    assert_close(result.displacements[0][2], 0.0);
    assert!(result.displacements[1][0].is_finite());
    assert!(result.displacements[1][0] > 0.0);
    assert_close(result.displacements[2][0], 0.0);
    assert_close(result.displacements[3][2], 0.0);
}

#[test]
fn elasticity_3d_rejects_duplicate_constraints() {
    let mesh = unit_tetrahedron();
    let problem = ElasticityProblem3D {
        material: material(),
        constraints: vec![
            DisplacementConstraint3D {
                node: 0,
                component: DisplacementComponent3D::X,
                value: 0.0,
            },
            DisplacementConstraint3D {
                node: 0,
                component: DisplacementComponent3D::X,
                value: 1.0,
            },
        ],
        forces: vec![],
    };

    let error = assemble_elasticity_3d_system(&mesh, &problem).unwrap_err();

    assert_eq!(
        error,
        ElasticityError::DuplicateConstraint3D {
            node_id: 0,
            component: DisplacementComponent3D::X,
        }
    );
}

#[test]
fn elasticity_3d_solves_hex8_solid_element() {
    let mesh = MeshTopology::<3>::new(
        vec![
            PointD::new([0.0, 0.0, 0.0]), // 0
            PointD::new([1.0, 0.0, 0.0]), // 1
            PointD::new([1.0, 1.0, 0.0]), // 2
            PointD::new([0.0, 1.0, 0.0]), // 3
            PointD::new([0.0, 0.0, 1.0]), // 4
            PointD::new([1.0, 0.0, 1.0]), // 5
            PointD::new([1.0, 1.0, 1.0]), // 6
            PointD::new([0.0, 1.0, 1.0]), // 7
        ],
        vec![Cell::new(ElementKind::Hex8, vec![0, 1, 2, 3, 4, 5, 6, 7])],
    )
    .unwrap();

    let problem = ElasticityProblem3D {
        material: material(),
        constraints: vec![
            // Constrain X-face at x=0 to prevent rigid body motion and rotation
            DisplacementConstraint3D {
                node: 0,
                component: DisplacementComponent3D::X,
                value: 0.0,
            },
            DisplacementConstraint3D {
                node: 0,
                component: DisplacementComponent3D::Y,
                value: 0.0,
            },
            DisplacementConstraint3D {
                node: 0,
                component: DisplacementComponent3D::Z,
                value: 0.0,
            },
            DisplacementConstraint3D {
                node: 3,
                component: DisplacementComponent3D::X,
                value: 0.0,
            },
            DisplacementConstraint3D {
                node: 3,
                component: DisplacementComponent3D::Z,
                value: 0.0,
            },
            DisplacementConstraint3D {
                node: 4,
                component: DisplacementComponent3D::X,
                value: 0.0,
            },
            DisplacementConstraint3D {
                node: 4,
                component: DisplacementComponent3D::Y,
                value: 0.0,
            },
            DisplacementConstraint3D {
                node: 7,
                component: DisplacementComponent3D::X,
                value: 0.0,
            },
        ],
        forces: vec![
            NodalForce3D {
                node: 1,
                fx: 0.25,
                fy: 0.0,
                fz: 0.0,
            },
            NodalForce3D {
                node: 2,
                fx: 0.25,
                fy: 0.0,
                fz: 0.0,
            },
            NodalForce3D {
                node: 5,
                fx: 0.25,
                fy: 0.0,
                fz: 0.0,
            },
            NodalForce3D {
                node: 6,
                fx: 0.25,
                fy: 0.0,
                fz: 0.0,
            },
        ],
    };

    let result =
        solve_elasticity_3d_with_solver(&mesh, &problem, LinearSolverOptions::default()).unwrap();

    // Verify displacements at loaded face are positive in X
    assert!(result.displacements[1][0] > 0.0);
    assert!(result.displacements[2][0] > 0.0);
    assert!(result.displacements[5][0] > 0.0);
    assert!(result.displacements[6][0] > 0.0);

    // Verify stress recovery at centroid
    assert_eq!(result.element_stress.len(), 1);
    assert_eq!(result.element_strain.len(), 1);
    assert_eq!(result.nodal_stress.len(), 8);

    let stress = result.element_stress[0];
    assert!(stress.sigma_xx > 0.0);
    assert_close(stress.sigma_yy, 0.0);
    assert_close(stress.sigma_zz, 0.0);
    assert_close(stress.sigma_xy, 0.0);
    assert_close(stress.sigma_yz, 0.0);
    assert_close(stress.sigma_xz, 0.0);
    assert_close(stress.von_mises, stress.sigma_xx);

    let strain = result.element_strain[0];
    assert!(strain.eps_xx > 0.0);
    assert!(strain.eps_yy < 0.0); // Poisson contraction
    assert!(strain.eps_zz < 0.0); // Poisson contraction
    assert_close(strain.gamma_xy, 0.0);
    assert_close(strain.gamma_yz, 0.0);
    assert_close(strain.gamma_xz, 0.0);
}

#[test]
fn transient_elasticity_3d_solves_constrained_tetrahedron_dynamics() {
    let mesh = unit_tetrahedron();
    let mut constraints = fixed_nodes(&[0, 2, 3]);
    constraints.push(DisplacementConstraint3D {
        node: 1,
        component: DisplacementComponent3D::Y,
        value: 0.0,
    });
    constraints.push(DisplacementConstraint3D {
        node: 1,
        component: DisplacementComponent3D::Z,
        value: 0.0,
    });
    let problem = TransientElasticityProblem3D {
        material: material(),
        density: 24.0,
        constraints,
        forces: |_| {
            vec![NodalForce3D {
                node: 1,
                fx: 1.0,
                fy: 0.0,
                fz: 0.0,
            }]
        },
        initial_displacements: vec![[0.0, 0.0, 0.0]; 4],
        initial_velocities: vec![[0.0, 0.0, 0.0]; 4],
        rayleigh_alpha: None,
        rayleigh_beta: None,
    };

    let result = solve_transient_elasticity_3d(
        &mesh,
        &problem,
        NewmarkSolverOptions {
            time_step: 1.0,
            steps: 2,
            linear_solver: LinearSolverOptions {
                backend: LinearSolverBackend::DenseDirect,
                ..LinearSolverOptions::default()
            },
            ..NewmarkSolverOptions::default()
        },
    )
    .unwrap();

    assert_eq!(result.steps.len(), 2);
    assert_close(result.steps[0].displacements[0][0], 0.0);
    assert_close(result.steps[0].displacements[1][1], 0.0);
    assert!(result.steps[0].displacements[1][0] > 0.0);
    assert!(result.steps[1].displacements[1][0].is_finite());
    assert!(result.steps[1].velocities[1][0].is_finite());
    assert_eq!(
        result.steps[0].diagnostics.backend,
        LinearSolverBackend::DenseDirect
    );
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

fn fixed_nodes(nodes: &[usize]) -> Vec<DisplacementConstraint3D> {
    nodes
        .iter()
        .flat_map(|&node| {
            [
                DisplacementConstraint3D {
                    node,
                    component: DisplacementComponent3D::X,
                    value: 0.0,
                },
                DisplacementConstraint3D {
                    node,
                    component: DisplacementComponent3D::Y,
                    value: 0.0,
                },
                DisplacementConstraint3D {
                    node,
                    component: DisplacementComponent3D::Z,
                    value: 0.0,
                },
            ]
        })
        .collect()
}

fn material() -> ElasticityMaterial3D {
    ElasticityMaterial3D {
        young_modulus: 100.0,
        poisson_ratio: 0.25,
    }
}

fn assert_matrix_vector_close(matrix: [[f64; 12]; 12], vector: [f64; 12], expected: [f64; 12]) {
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
