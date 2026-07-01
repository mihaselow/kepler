use kepler::{
    DisplacementComponent, DisplacementConstraint, ElasticityMaterial, ElasticityModel,
    ElasticityProblem, Mesh, Point2, SolverOptions, SteadyHeatProblem, ThermoElasticProblem,
    ThermoElasticStaggerOptions, Tri3, solve_thermoelastic,
};

#[test]
fn thermoelastic_unrestrained_bar_thermal_expansion() {
    let mesh = horizontal_bar_mesh(2.0, 0.1);
    let delta_t = 100.0;
    let alpha = 12.0e-6;
    let young_modulus = 200.0e9;
    let length = 2.0;

    let result = solve_thermoelastic(
        &mesh,
        &bar_problem(
            &mesh,
            delta_t,
            alpha,
            young_modulus,
            unrestrained_constraints(),
        ),
        SolverOptions::default(),
        SolverOptions::default(),
        ThermoElasticStaggerOptions::default(),
    )
    .unwrap();

    let expected_tip = alpha * delta_t * length;
    assert!((result.displacements[1][0] - expected_tip).abs() < 1.0e-5);
    assert!((result.displacements[3][0] - expected_tip).abs() < 1.0e-5);
}

#[test]
fn thermoelastic_constrained_bar_thermal_stress() {
    let mesh = horizontal_bar_mesh(2.0, 0.1);
    let delta_t = 100.0;
    let alpha = 12.0e-6;
    let young_modulus = 200.0e9;
    let expected_stress = -young_modulus * alpha * delta_t;

    let result = solve_thermoelastic(
        &mesh,
        &bar_problem(
            &mesh,
            delta_t,
            alpha,
            young_modulus,
            constrained_constraints(),
        ),
        SolverOptions::default(),
        SolverOptions::default(),
        ThermoElasticStaggerOptions::default(),
    )
    .unwrap();

    for displacement in &result.displacements {
        assert!(displacement[0].abs() < 1.0e-4);
    }

    for stress in result.element_stress {
        assert!((stress.sigma_xx - expected_stress).abs() < 1.0e8);
        assert!(stress.sigma_yy.abs() < 1.0e-3);
        assert!(stress.sigma_xy.abs() < 1.0e-3);
    }
}

fn bar_problem(
    mesh: &Mesh,
    temperature: f64,
    alpha: f64,
    young_modulus: f64,
    constraints: Vec<DisplacementConstraint>,
) -> ThermoElasticProblem<impl Fn(f64, f64) -> f64> {
    ThermoElasticProblem {
        heat_problem: SteadyHeatProblem {
            thermal_conductivity: 1.0,
            heat_generation: |_, _| 0.0,
            prescribed_temperatures: (0..mesh.node_count())
                .map(|node| (node, temperature))
                .collect(),
        },
        elasticity_problem: ElasticityProblem {
            material: ElasticityMaterial {
                young_modulus,
                poisson_ratio: 0.3,
                model: ElasticityModel::PlaneStress,
            },
            thickness: 1.0,
            constraints,
            forces: vec![],
        },
        thermal_expansion_coeff: alpha,
        reference_temperature: 0.0,
    }
}

fn unrestrained_constraints() -> Vec<DisplacementConstraint> {
    vec![
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
    ]
}

fn constrained_constraints() -> Vec<DisplacementConstraint> {
    vec![
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
            component: DisplacementComponent::X,
            value: 0.0,
        },
        DisplacementConstraint {
            node: 2,
            component: DisplacementComponent::X,
            value: 0.0,
        },
        DisplacementConstraint {
            node: 3,
            component: DisplacementComponent::X,
            value: 0.0,
        },
    ]
}

fn horizontal_bar_mesh(length: f64, height: f64) -> Mesh {
    Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(length, 0.0),
            Point2::new(0.0, height),
            Point2::new(length, height),
        ],
        vec![Tri3::new([0, 1, 3]), Tri3::new([0, 3, 2])],
    )
    .unwrap()
}
