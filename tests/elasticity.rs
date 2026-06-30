use kepler::{
    DisplacementComponent, DisplacementConstraint, ElasticityError, ElasticityMaterial,
    ElasticityModel, ElasticityProblem, LinearSolverBackend, LinearSolverOptions, Mesh,
    NewmarkSolverOptions, NodalForce, Point2, SolverOptions, TransientElasticityProblem, Tri3,
    fem::elasticity::{assemble_elasticity_system, local_elasticity_stiffness},
    solve_elasticity, solve_elasticity_with_solver, solve_transient_elasticity,
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

#[test]
fn transient_elasticity_solves_constrained_triangle_dynamics() {
    let mesh = unit_triangle();
    let problem = TransientElasticityProblem {
        material: material(),
        thickness: 1.0,
        density: 6.0,
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
                node: 1,
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
        forces: |_| {
            vec![NodalForce {
                node: 1,
                fx: 1.0,
                fy: 0.0,
            }]
        },
        initial_displacements: vec![[0.0, 0.0]; 3],
        initial_velocities: vec![[0.0, 0.0]; 3],
        rayleigh_alpha: None,
        rayleigh_beta: None,
    };

    let result = solve_transient_elasticity(
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

#[test]
fn transient_elasticity_rejects_invalid_density() {
    let mesh = unit_triangle();
    let problem = TransientElasticityProblem {
        material: material(),
        thickness: 1.0,
        density: 0.0,
        constraints: vec![],
        forces: |_| vec![],
        initial_displacements: vec![[0.0, 0.0]; 3],
        initial_velocities: vec![[0.0, 0.0]; 3],
        rayleigh_alpha: None,
        rayleigh_beta: None,
    };

    let error =
        solve_transient_elasticity(&mesh, &problem, NewmarkSolverOptions::default()).unwrap_err();

    assert_eq!(error, ElasticityError::InvalidDensity(0.0));
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

#[test]
fn test_rayleigh_damping_decays_vibration() {
    let mesh = unit_triangle();
    let mut initial_displacements = vec![[0.0, 0.0]; 3];
    initial_displacements[1] = [1.0, 0.0];

    let problem_undamped = TransientElasticityProblem {
        material: material(),
        thickness: 1.0,
        density: 1.0,
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
        ],
        forces: |_| vec![],
        initial_displacements: initial_displacements.clone(),
        initial_velocities: vec![[0.0, 0.0]; 3],
        rayleigh_alpha: None,
        rayleigh_beta: None,
    };

    let problem_damped = TransientElasticityProblem {
        material: material(),
        thickness: 1.0,
        density: 1.0,
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
        ],
        forces: |_| vec![],
        initial_displacements: initial_displacements.clone(),
        initial_velocities: vec![[0.0, 0.0]; 3],
        rayleigh_alpha: Some(2.0),
        rayleigh_beta: Some(0.05),
    };

    let options = NewmarkSolverOptions {
        time_step: 0.1,
        steps: 10,
        beta: 0.25,
        gamma: 0.5,
        linear_solver: LinearSolverOptions::default(),
    };

    let res_undamped =
        solve_transient_elasticity(&mesh, &problem_undamped, options.clone()).unwrap();
    let res_damped = solve_transient_elasticity(&mesh, &problem_damped, options).unwrap();

    let x_undamped_last = res_undamped.steps.last().unwrap().displacements[1][0];
    let x_damped_last = res_damped.steps.last().unwrap().displacements[1][0];

    assert!(x_damped_last.abs() < x_undamped_last.abs());
}

#[test]
fn elasticity_recovers_stress_and_strain_for_triangle() {
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
        ],
        forces: vec![NodalForce {
            node: 1,
            fx: 1.0,
            fy: 0.0,
        }],
    };

    let result =
        solve_elasticity_with_solver(&mesh, &problem, LinearSolverOptions::default()).unwrap();

    assert_eq!(result.element_stress.len(), 1);
    assert_eq!(result.element_strain.len(), 1);
    assert_eq!(result.nodal_stress.len(), 3);

    // Stress components should be physically reasonable for uniaxial tension.
    let stress = result.element_stress[0];
    assert!(stress.sigma_xx > 0.0);
    assert_close(stress.sigma_yy, 0.0);
    assert_close(stress.sigma_xy, 0.0);

    // von Mises stress should match sigma_xx for uniaxial tension.
    assert_close(stress.von_mises, stress.sigma_xx);

    // Strain components should also reflect uniaxial tension.
    let strain = result.element_strain[0];
    assert!(strain.eps_xx > 0.0);
    assert!(strain.eps_yy < 0.0); // Poisson contraction
    assert_close(strain.gamma_xy, 0.0);
}

#[test]
fn elasticity_solve_quad4_plane_stress_patch() {
    use kepler::{Cell, ElementKind};

    // Unit square: Quad4
    let points = vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 { x: 1.0, y: 0.0 },
        Point2 { x: 1.0, y: 1.0 },
        Point2 { x: 0.0, y: 1.0 },
    ];
    let cells = vec![Cell::new(ElementKind::Quad4, vec![0, 1, 2, 3])];
    let mesh = Mesh::new_with_cells(points, cells).unwrap();

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
                node: 3,
                component: DisplacementComponent::X,
                value: 0.0,
            },
        ],
        forces: vec![
            NodalForce {
                node: 1,
                fx: 0.5,
                fy: 0.0,
            },
            NodalForce {
                node: 2,
                fx: 0.5,
                fy: 0.0,
            },
        ],
    };

    let result =
        solve_elasticity_with_solver(&mesh, &problem, LinearSolverOptions::default()).unwrap();

    // Verify displacements are positive in X direction
    assert!(result.displacements[1][0] > 0.0);
    assert!(result.displacements[2][0] > 0.0);

    // Verify stress recovery
    assert_eq!(result.element_stress.len(), 1);
    let stress = result.element_stress[0];
    assert!(stress.sigma_xx > 0.0);
    assert_close(stress.sigma_yy, 0.0);
    assert_close(stress.sigma_xy, 0.0);
    assert_close(stress.von_mises, stress.sigma_xx);

    let strain = result.element_strain[0];
    assert!(strain.eps_xx > 0.0);
    assert!(strain.eps_yy < 0.0);
    assert_close(strain.gamma_xy, 0.0);
}

#[test]
fn elasticity_solve_tri6_plane_stress_patch() {
    use kepler::{Cell, ElementKind};

    // Quadratic triangle: Tri6
    let points = vec![
        Point2 { x: 0.0, y: 0.0 }, // 0
        Point2 { x: 1.0, y: 0.0 }, // 1
        Point2 { x: 0.0, y: 1.0 }, // 2
        Point2 { x: 0.5, y: 0.0 }, // 3
        Point2 { x: 0.5, y: 0.5 }, // 4
        Point2 { x: 0.0, y: 0.5 }, // 5
    ];
    let cells = vec![Cell::new(ElementKind::Tri6, vec![0, 1, 2, 3, 4, 5])];
    let mesh = Mesh::new_with_cells(points, cells).unwrap();

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
                node: 5,
                component: DisplacementComponent::X,
                value: 0.0,
            },
            DisplacementConstraint {
                node: 2,
                component: DisplacementComponent::X,
                value: 0.0,
            },
        ],
        forces: vec![NodalForce {
            node: 1,
            fx: 1.0,
            fy: 0.0,
        }],
    };

    let result =
        solve_elasticity_with_solver(&mesh, &problem, LinearSolverOptions::default()).unwrap();

    // Verify displacements
    assert!(result.displacements[1][0] > 0.0);

    // Verify stress recovery
    assert_eq!(result.element_stress.len(), 1);
    let stress = result.element_stress[0];
    assert!(stress.sigma_xx > 0.0);
    assert_close(stress.sigma_yy, 0.0);
    assert_close(stress.sigma_xy, 0.0);
    assert_close(stress.von_mises, stress.sigma_xx);

    let strain = result.element_strain[0];
    assert!(strain.eps_xx > 0.0);
    assert!(strain.eps_yy < 0.0);
    assert_close(strain.gamma_xy, 0.0);
}
