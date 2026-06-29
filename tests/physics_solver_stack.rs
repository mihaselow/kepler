use kepler::{
    DiffusionReactionProblem, DisplacementComponent, DisplacementConstraint, ElasticityMaterial,
    ElasticityModel, ElasticityProblem, ElectrostaticProblem, LinearSolverBackend,
    LinearSolverOptions, Mesh, NodalForce, Point2, PoissonProblem, PreconditionerKind,
    SteadyHeatProblem, Tri3, solve_diffusion_reaction_with_solver, solve_elasticity_with_solver,
    solve_electrostatics_with_solver, solve_poisson_with_solver, solve_steady_heat_with_solver,
};

#[test]
fn poisson_exposes_backend_selection_and_diagnostics() {
    let mesh = square_mesh();
    let problem = scalar_problem();

    let result = solve_poisson_with_solver(
        &mesh,
        &problem,
        LinearSolverOptions {
            backend: LinearSolverBackend::DenseDirect,
            record_residual_history: true,
            ..LinearSolverOptions::default()
        },
    )
    .unwrap();

    assert_close(result.values[4], 1.0 / 12.0);
    assert_eq!(result.diagnostics.backend, LinearSolverBackend::DenseDirect);
    assert!(result.diagnostics.converged);
    assert_eq!(result.diagnostics.residual_history.len(), 2);
}

#[test]
fn heat_and_electrostatics_preserve_domain_results_with_solver_diagnostics() {
    let mesh = square_mesh();
    let solver = LinearSolverOptions {
        preconditioner: PreconditionerKind::Jacobi,
        record_residual_history: true,
        ..LinearSolverOptions::default()
    };
    let heat = SteadyHeatProblem {
        thermal_conductivity: 1.0,
        heat_generation: |_, _| 1.0,
        prescribed_temperatures: boundary_conditions(),
    };
    let electrostatics = ElectrostaticProblem {
        permittivity: 1.0,
        charge_density: |_, _| 1.0,
        prescribed_potentials: boundary_conditions(),
    };

    let heat_result = solve_steady_heat_with_solver(&mesh, &heat, solver.clone()).unwrap();
    let electrostatic_result =
        solve_electrostatics_with_solver(&mesh, &electrostatics, solver).unwrap();

    assert_close(heat_result.temperatures[4], 1.0 / 12.0);
    assert_close(electrostatic_result.potentials[4], 1.0 / 12.0);
    assert_eq!(
        heat_result.diagnostics.preconditioner,
        PreconditionerKind::Jacobi
    );
    assert_eq!(
        electrostatic_result.diagnostics.preconditioner,
        PreconditionerKind::Jacobi
    );
}

#[test]
fn diffusion_reaction_exposes_solver_stack_result() {
    let mesh = square_mesh();
    let problem = DiffusionReactionProblem {
        diffusivity: 1.0,
        reaction_rate: 0.0,
        source: |_, _| 1.0,
        dirichlet: boundary_conditions(),
    };

    let result = solve_diffusion_reaction_with_solver(
        &mesh,
        &problem,
        LinearSolverOptions {
            backend: LinearSolverBackend::DenseDirect,
            ..LinearSolverOptions::default()
        },
    )
    .unwrap();

    assert_close(result.values[4], 1.0 / 12.0);
    assert_eq!(result.diagnostics.backend, LinearSolverBackend::DenseDirect);
}

#[test]
fn elasticity_exposes_solver_stack_result() {
    let mesh = Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ],
        vec![Tri3::new([0, 1, 2])],
    )
    .unwrap();
    let problem = ElasticityProblem {
        material: ElasticityMaterial {
            young_modulus: 100.0,
            poisson_ratio: 0.25,
            model: ElasticityModel::PlaneStress,
        },
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

    let result = solve_elasticity_with_solver(
        &mesh,
        &problem,
        LinearSolverOptions {
            backend: LinearSolverBackend::DenseDirect,
            ..LinearSolverOptions::default()
        },
    )
    .unwrap();

    assert!(result.displacements[1][0] > 0.0);
    assert_eq!(result.diagnostics.backend, LinearSolverBackend::DenseDirect);
}

fn square_mesh() -> Mesh {
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

fn scalar_problem() -> PoissonProblem<impl Fn(f64, f64) -> f64> {
    PoissonProblem {
        conductivity: 1.0,
        source: |_, _| 1.0,
        dirichlet: boundary_conditions(),
    }
}

fn boundary_conditions() -> Vec<(usize, f64)> {
    vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-10,
        "expected {actual} to be close to {expected}",
    );
}
