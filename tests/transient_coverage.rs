use kepler::{
    Cell, DisplacementComponent, DisplacementComponent3D, DisplacementConstraint,
    DisplacementConstraint3D, ELECTROSTATIC_FORMULATION, ElasticityMaterial, ElasticityMaterial3D,
    ElasticityModel, ElectrostaticFormulation, ElementKind, LinearSolverBackend,
    LinearSolverOptions, Mesh, MeshTopology, NewmarkSolverOptions, NodalForce, NodalForce3D,
    Point2, PointD, TransientDiffusionReactionProblem, TransientDiffusionReactionProblem3D,
    TransientElasticityProblem, TransientElasticityProblem3D, TransientHeatProblem,
    TransientSolverOptions, Tri3, solve_transient_diffusion_reaction,
    solve_transient_diffusion_reaction_3d, solve_transient_elasticity,
    solve_transient_elasticity_3d, solve_transient_heat,
};

#[test]
fn solver_stack_exercises_complete_transient_physics_coverage() {
    let mesh_2d = unit_triangle();
    let mesh_3d = unit_tetrahedron();
    let theta_options = TransientSolverOptions {
        time_step: 1.0,
        steps: 1,
        theta: 1.0,
        linear_solver: dense_direct_options(),
    };
    let newmark_options = NewmarkSolverOptions {
        time_step: 1.0,
        steps: 1,
        linear_solver: dense_direct_options(),
        ..NewmarkSolverOptions::default()
    };

    let heat = solve_transient_heat(
        &mesh_2d,
        &TransientHeatProblem {
            thermal_conductivity: 1.0,
            volumetric_heat_capacity: 1.0,
            heat_generation: |_, _, _| 0.0,
            initial_temperatures: vec![0.0, 1.0, 0.0],
            prescribed_temperatures: vec![(0, 0.0), (2, 0.0)],
        },
        theta_options.clone(),
    )
    .unwrap();
    assert_eq!(
        heat.steps[0].diagnostics.backend,
        LinearSolverBackend::DenseDirect
    );

    let diffusion = solve_transient_diffusion_reaction(
        &mesh_2d,
        &TransientDiffusionReactionProblem {
            diffusivity: 1.0,
            reaction_rate: 0.0,
            storage_coefficient: 1.0,
            source: |_, _, _| 0.0,
            initial_values: vec![0.0, 1.0, 0.0],
            dirichlet: vec![(0, 0.0), (2, 0.0)],
        },
        theta_options.clone(),
    )
    .unwrap();
    assert_eq!(
        diffusion.steps[0].diagnostics.backend,
        LinearSolverBackend::DenseDirect
    );

    let diffusion_3d = solve_transient_diffusion_reaction_3d(
        &mesh_3d,
        &TransientDiffusionReactionProblem3D {
            diffusivity: 1.0,
            reaction_rate: 0.0,
            storage_coefficient: 1.0,
            source: |_, _, _, _| 0.0,
            initial_values: vec![1.0, 0.0, 0.0, 0.0],
            dirichlet: vec![(1, 0.0), (2, 0.0), (3, 0.0)],
        },
        theta_options,
    )
    .unwrap();
    assert_eq!(
        diffusion_3d.steps[0].diagnostics.backend,
        LinearSolverBackend::DenseDirect
    );

    let elasticity = solve_transient_elasticity(
        &mesh_2d,
        &TransientElasticityProblem {
            material: material_2d(),
            thickness: 1.0,
            density: 6.0,
            constraints: fixed_nodes_2d(&[0, 2]),
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
        },
        newmark_options.clone(),
    )
    .unwrap();
    assert_eq!(
        elasticity.steps[0].diagnostics.backend,
        LinearSolverBackend::DenseDirect
    );

    let elasticity_3d = solve_transient_elasticity_3d(
        &mesh_3d,
        &TransientElasticityProblem3D {
            material: material_3d(),
            density: 24.0,
            constraints: fixed_nodes_3d(&[0, 2, 3]),
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
        },
        newmark_options,
    )
    .unwrap();
    assert_eq!(
        elasticity_3d.steps[0].diagnostics.backend,
        LinearSolverBackend::DenseDirect
    );

    assert_eq!(
        ELECTROSTATIC_FORMULATION,
        ElectrostaticFormulation::SteadyQuasiStatic
    );
}

fn dense_direct_options() -> LinearSolverOptions {
    LinearSolverOptions {
        backend: LinearSolverBackend::DenseDirect,
        ..LinearSolverOptions::default()
    }
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

fn fixed_nodes_2d(nodes: &[usize]) -> Vec<DisplacementConstraint> {
    nodes
        .iter()
        .flat_map(|&node| {
            [
                DisplacementConstraint {
                    node,
                    component: DisplacementComponent::X,
                    value: 0.0,
                },
                DisplacementConstraint {
                    node,
                    component: DisplacementComponent::Y,
                    value: 0.0,
                },
            ]
        })
        .collect()
}

fn fixed_nodes_3d(nodes: &[usize]) -> Vec<DisplacementConstraint3D> {
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
