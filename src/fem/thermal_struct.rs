use thiserror::Error;

use crate::{
    fem::{
        elasticity::{
            DisplacementComponent, ElasticityError, ElasticityMaterial, ElasticityModel,
            ElasticityProblem, ElasticitySolverResult, NodalForce, StressTensor2D,
            solve_elasticity_with_solver,
        },
        heat::{SteadyHeatProblem, solve_steady_heat},
        poisson::PoissonError,
    },
    linalg::{LinearSolverOptions, SolverOptions},
    mesh::{ElementKind, Mesh, NodeId, Point3, Tri3},
};

/// Coupled steady-state thermoelastic problem (loosely / staggered).
pub struct ThermoElasticProblem<F> {
    pub heat_problem: SteadyHeatProblem<F>,
    pub elasticity_problem: ElasticityProblem,
    pub thermal_expansion_coeff: f64,
    pub reference_temperature: f64,
}

/// Options for staggered thermoelastic coupling iterations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermoElasticStaggerOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for ThermoElasticStaggerOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1,
            tolerance: 1.0e-10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermoElasticResult {
    pub temperatures: Vec<f64>,
    pub displacements: Vec<[f64; 2]>,
    pub element_stress: Vec<StressTensor2D>,
    pub heat_iterations: usize,
    pub heat_residual_norm: f64,
    pub elasticity_iterations: usize,
    pub elasticity_residual_norm: f64,
    pub stagger_iterations: usize,
}

#[derive(Debug, Error, PartialEq)]
pub enum ThermoElasticError {
    #[error("thermal expansion coefficient must be finite, got {0}")]
    InvalidThermalExpansionCoeff(f64),
    #[error("steady heat solve failed")]
    Heat(#[from] PoissonError),
    #[error("elasticity solve failed")]
    Elasticity(#[from] ElasticityError),
}

/// Staggered thermoelastic solve: heat → thermal strain → equivalent load → elasticity.
pub fn solve_thermoelastic<F>(
    mesh: &Mesh,
    problem: &ThermoElasticProblem<F>,
    heat_options: SolverOptions,
    elasticity_options: SolverOptions,
    stagger_options: ThermoElasticStaggerOptions,
) -> Result<ThermoElasticResult, ThermoElasticError>
where
    F: Fn(f64, f64) -> f64,
{
    if !problem.thermal_expansion_coeff.is_finite() {
        return Err(ThermoElasticError::InvalidThermalExpansionCoeff(
            problem.thermal_expansion_coeff,
        ));
    }

    let max_iterations = stagger_options.max_iterations.max(1);
    let heat_result = solve_steady_heat(mesh, &problem.heat_problem, heat_options)?;
    let mut elasticity_result: Option<ElasticitySolverResult> = None;
    let mut stagger_iterations = 0usize;
    let elasticity_solver = LinearSolverOptions::from(elasticity_options);

    for iteration in 0..max_iterations {
        stagger_iterations = iteration + 1;
        let thermal_rhs = assemble_thermal_load(
            mesh,
            &heat_result.temperatures,
            problem.elasticity_problem.material,
            problem.elasticity_problem.thickness,
            problem.thermal_expansion_coeff,
            problem.reference_temperature,
        )?;
        let combined_forces = merge_forces(
            &problem.elasticity_problem.forces,
            &thermal_rhs,
            mesh.node_count(),
        );
        let elasticity_problem = ElasticityProblem {
            material: problem.elasticity_problem.material,
            thickness: problem.elasticity_problem.thickness,
            constraints: problem.elasticity_problem.constraints.clone(),
            forces: combined_forces,
        };
        let next =
            solve_elasticity_with_solver(mesh, &elasticity_problem, elasticity_solver.clone())?;

        let displacement_change = elasticity_result
            .as_ref()
            .map(|previous| displacement_delta(&previous.displacements, &next.displacements))
            .unwrap_or(f64::INFINITY);
        elasticity_result = Some(next);

        if displacement_change <= stagger_options.tolerance {
            break;
        }
    }

    let elasticity_result = elasticity_result.expect("at least one stagger iteration");
    let element_stress = recover_thermoelastic_stress(
        mesh,
        problem.elasticity_problem.material,
        &heat_result.temperatures,
        &elasticity_result.displacements,
        problem.thermal_expansion_coeff,
        problem.reference_temperature,
    );

    Ok(ThermoElasticResult {
        temperatures: heat_result.temperatures,
        displacements: elasticity_result.displacements,
        element_stress,
        heat_iterations: heat_result.iterations,
        heat_residual_norm: heat_result.residual_norm,
        elasticity_iterations: elasticity_result.diagnostics.iterations,
        elasticity_residual_norm: elasticity_result.diagnostics.residual_norm,
        stagger_iterations,
    })
}

fn displacement_delta(previous: &[[f64; 2]], current: &[[f64; 2]]) -> f64 {
    previous
        .iter()
        .zip(current)
        .map(|(a, b)| {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            dx * dx + dy * dy
        })
        .sum::<f64>()
        .sqrt()
}

fn merge_forces(
    mechanical: &[NodalForce],
    thermal_rhs: &[f64],
    node_count: usize,
) -> Vec<NodalForce> {
    let mut fx = vec![0.0; node_count];
    let mut fy = vec![0.0; node_count];

    for force in mechanical {
        fx[force.node] += force.fx;
        fy[force.node] += force.fy;
    }
    for node in 0..node_count {
        fx[node] += thermal_rhs[dof_index(node, DisplacementComponent::X)];
        fy[node] += thermal_rhs[dof_index(node, DisplacementComponent::Y)];
    }

    let mut forces = mechanical.to_vec();
    for node in 0..node_count {
        let has_mechanical = mechanical.iter().any(|force| force.node == node);
        if has_mechanical {
            if let Some(force) = forces.iter_mut().find(|force| force.node == node) {
                force.fx = fx[node];
                force.fy = fy[node];
            }
        } else if fx[node] != 0.0 || fy[node] != 0.0 {
            forces.push(NodalForce {
                node,
                fx: fx[node],
                fy: fy[node],
            });
        }
    }
    forces
}

fn assemble_thermal_load(
    mesh: &Mesh,
    temperatures: &[f64],
    material: ElasticityMaterial,
    thickness: f64,
    alpha: f64,
    reference_temperature: f64,
) -> Result<Vec<f64>, ElasticityError> {
    let mut rhs = vec![0.0; mesh.node_count() * 2];
    let constitutive = constitutive_matrix(material);

    for cell in mesh.cells() {
        let local = match cell.kind {
            ElementKind::Tri3 => {
                let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2]];
                let triangle = Tri3::new(nodes);
                local_thermal_load_tri3(
                    mesh,
                    &triangle,
                    temperatures,
                    &constitutive,
                    thickness,
                    alpha,
                    reference_temperature,
                )?
            }
            ElementKind::Quad4 => local_thermal_load_quad4(
                mesh,
                &cell.nodes,
                temperatures,
                &constitutive,
                thickness,
                alpha,
                reference_temperature,
            )?,
            _ => continue,
        };

        let dofs = cell_dofs(&cell.nodes);
        for (local_index, &global_dof) in dofs.iter().enumerate() {
            rhs[global_dof] += local[local_index];
        }
    }

    Ok(rhs)
}

fn local_thermal_load_tri3(
    mesh: &Mesh,
    triangle: &Tri3,
    temperatures: &[f64],
    constitutive: &[[f64; 3]; 3],
    thickness: f64,
    alpha: f64,
    reference_temperature: f64,
) -> Result<Vec<f64>, ElasticityError> {
    let (area, gradients) = triangle_gradients(mesh, triangle)?;
    let delta_temperature =
        element_delta_temperature(triangle.nodes, temperatures, reference_temperature);
    let thermal_stress = thermal_stress_vector(constitutive, alpha, delta_temperature);
    let strain_displacement = strain_displacement_matrix(gradients);
    Ok(scatter_thermal_load(
        &strain_displacement,
        &thermal_stress,
        thickness * area,
    ))
}

fn local_thermal_load_quad4(
    mesh: &Mesh,
    nodes: &[NodeId],
    temperatures: &[f64],
    constitutive: &[[f64; 3]; 3],
    thickness: f64,
    alpha: f64,
    reference_temperature: f64,
) -> Result<Vec<f64>, ElasticityError> {
    let node_coords: Vec<Point3> = nodes
        .iter()
        .map(|&id| {
            let point = mesh.points()[id];
            Point3::new([point.x, point.y, 0.0])
        })
        .collect();
    let delta_temperature = nodes.iter().map(|&node| temperatures[node]).sum::<f64>()
        / nodes.len() as f64
        - reference_temperature;
    let thermal_stress = thermal_stress_vector(constitutive, alpha, delta_temperature);

    let xi = 0.0;
    let eta = 0.0;
    let dn_dxi = [
        -0.25 * (1.0 - eta),
        0.25 * (1.0 - eta),
        0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta),
    ];
    let dn_deta = [
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
        0.25 * (1.0 + xi),
        0.25 * (1.0 - xi),
    ];
    let mut jacobian = [[0.0; 2]; 2];
    for node in 0..4 {
        jacobian[0][0] += dn_dxi[node] * node_coords[node].coords[0];
        jacobian[0][1] += dn_dxi[node] * node_coords[node].coords[1];
        jacobian[1][0] += dn_deta[node] * node_coords[node].coords[0];
        jacobian[1][1] += dn_deta[node] * node_coords[node].coords[1];
    }
    let det_j = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
    if det_j.abs() <= f64::EPSILON {
        return Ok(vec![0.0; 8]);
    }
    let inv_det = 1.0 / det_j;
    let dndx = [
        inv_det * (jacobian[1][1] * dn_dxi[0] - jacobian[0][1] * dn_deta[0]),
        inv_det * (jacobian[1][1] * dn_dxi[1] - jacobian[0][1] * dn_deta[1]),
        inv_det * (jacobian[1][1] * dn_dxi[2] - jacobian[0][1] * dn_deta[2]),
        inv_det * (jacobian[1][1] * dn_dxi[3] - jacobian[0][1] * dn_deta[3]),
    ];
    let dndy = [
        inv_det * (-jacobian[1][0] * dn_dxi[0] + jacobian[0][0] * dn_deta[0]),
        inv_det * (-jacobian[1][0] * dn_dxi[1] + jacobian[0][0] * dn_deta[1]),
        inv_det * (-jacobian[1][0] * dn_dxi[2] + jacobian[0][0] * dn_deta[2]),
        inv_det * (-jacobian[1][0] * dn_dxi[3] + jacobian[0][0] * dn_deta[3]),
    ];
    let gradients = [
        [dndx[0], dndy[0]],
        [dndx[1], dndy[1]],
        [dndx[2], dndy[2]],
        [dndx[3], dndy[3]],
    ];
    let strain_displacement = strain_displacement_matrix_quad4(gradients);
    Ok(scatter_thermal_load(
        &strain_displacement,
        &thermal_stress,
        thickness * det_j,
    ))
}

fn scatter_thermal_load<const N: usize>(
    strain_displacement: &[[f64; N]; 3],
    thermal_stress: &[f64; 3],
    factor: f64,
) -> Vec<f64> {
    let mut local = vec![0.0; N];
    for row in 0..N {
        let mut value = 0.0;
        for strain_component in 0..3 {
            value += strain_displacement[strain_component][row] * thermal_stress[strain_component];
        }
        local[row] = factor * value;
    }
    local
}

fn element_delta_temperature(
    nodes: impl IntoIterator<Item = NodeId>,
    temperatures: &[f64],
    reference_temperature: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for node in nodes {
        sum += temperatures[node];
        count += 1;
    }
    sum / count as f64 - reference_temperature
}

fn thermal_stress_vector(
    constitutive: &[[f64; 3]; 3],
    alpha: f64,
    delta_temperature: f64,
) -> [f64; 3] {
    let thermal_strain = [alpha * delta_temperature, alpha * delta_temperature, 0.0];
    let mut stress = [0.0; 3];
    for row in 0..3 {
        for col in 0..3 {
            stress[row] += constitutive[row][col] * thermal_strain[col];
        }
    }
    stress
}

fn triangle_gradients(
    mesh: &Mesh,
    triangle: &Tri3,
) -> Result<(f64, [[f64; 2]; 3]), ElasticityError> {
    let [a, b, c] = triangle.nodes.map(|node| mesh.points()[node]);
    let twice_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    let area = 0.5 * twice_area.abs();
    if area <= f64::EPSILON {
        return Err(ElasticityError::UnsupportedElementKind {
            cell_index: 0,
            kind: ElementKind::Tri3,
        });
    }
    let gradients = [
        [(b.y - c.y) / twice_area, (c.x - b.x) / twice_area],
        [(c.y - a.y) / twice_area, (a.x - c.x) / twice_area],
        [(a.y - b.y) / twice_area, (b.x - a.x) / twice_area],
    ];
    Ok((area, gradients))
}

fn constitutive_matrix(material: ElasticityMaterial) -> [[f64; 3]; 3] {
    match material.model {
        ElasticityModel::PlaneStress => {
            let scale =
                material.young_modulus / (1.0 - material.poisson_ratio * material.poisson_ratio);
            [
                [scale, scale * material.poisson_ratio, 0.0],
                [scale * material.poisson_ratio, scale, 0.0],
                [0.0, 0.0, scale * (1.0 - material.poisson_ratio) / 2.0],
            ]
        }
        ElasticityModel::PlaneStrain => {
            let scale = material.young_modulus
                / ((1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio));
            [
                [
                    scale * (1.0 - material.poisson_ratio),
                    scale * material.poisson_ratio,
                    0.0,
                ],
                [
                    scale * material.poisson_ratio,
                    scale * (1.0 - material.poisson_ratio),
                    0.0,
                ],
                [0.0, 0.0, scale * (1.0 - 2.0 * material.poisson_ratio) / 2.0],
            ]
        }
    }
}

fn strain_displacement_matrix(gradients: [[f64; 2]; 3]) -> [[f64; 6]; 3] {
    [
        [
            gradients[0][0],
            0.0,
            gradients[1][0],
            0.0,
            gradients[2][0],
            0.0,
        ],
        [
            0.0,
            gradients[0][1],
            0.0,
            gradients[1][1],
            0.0,
            gradients[2][1],
        ],
        [
            gradients[0][1],
            gradients[0][0],
            gradients[1][1],
            gradients[1][0],
            gradients[2][1],
            gradients[2][0],
        ],
    ]
}

fn strain_displacement_matrix_quad4(gradients: [[f64; 2]; 4]) -> [[f64; 8]; 3] {
    [
        [
            gradients[0][0],
            0.0,
            gradients[1][0],
            0.0,
            gradients[2][0],
            0.0,
            gradients[3][0],
            0.0,
        ],
        [
            0.0,
            gradients[0][1],
            0.0,
            gradients[1][1],
            0.0,
            gradients[2][1],
            0.0,
            gradients[3][1],
        ],
        [
            gradients[0][1],
            gradients[0][0],
            gradients[1][1],
            gradients[1][0],
            gradients[2][1],
            gradients[2][0],
            gradients[3][1],
            gradients[3][0],
        ],
    ]
}

fn cell_dofs(nodes: &[NodeId]) -> Vec<usize> {
    let mut dofs = Vec::with_capacity(nodes.len() * 2);
    for &node in nodes {
        dofs.push(dof_index(node, DisplacementComponent::X));
        dofs.push(dof_index(node, DisplacementComponent::Y));
    }
    dofs
}

fn dof_index(node: NodeId, component: DisplacementComponent) -> usize {
    node * 2
        + match component {
            DisplacementComponent::X => 0,
            DisplacementComponent::Y => 1,
        }
}

fn recover_thermoelastic_stress(
    mesh: &Mesh,
    material: ElasticityMaterial,
    temperatures: &[f64],
    displacements: &[[f64; 2]],
    alpha: f64,
    reference_temperature: f64,
) -> Vec<StressTensor2D> {
    let constitutive = constitutive_matrix(material);
    mesh.cells()
        .iter()
        .map(|cell| {
            let nodes: Vec<NodeId> = cell.nodes.clone();
            let delta_temperature = element_delta_temperature(
                nodes.iter().copied(),
                temperatures,
                reference_temperature,
            );
            let thermal_strain = [alpha * delta_temperature, alpha * delta_temperature, 0.0];
            let mut strain = [0.0; 3];

            if cell.kind == ElementKind::Tri3 && nodes.len() == 3 {
                let triangle = Tri3::new([nodes[0], nodes[1], nodes[2]]);
                if let Ok((_, gradients)) = triangle_gradients(mesh, &triangle) {
                    let b = strain_displacement_matrix(gradients);
                    let mut local_displacements = [0.0; 6];
                    for (local_node, &node) in nodes.iter().enumerate() {
                        local_displacements[local_node * 2] = displacements[node][0];
                        local_displacements[local_node * 2 + 1] = displacements[node][1];
                    }
                    for strain_component in 0..3 {
                        for (dof, &value) in local_displacements.iter().enumerate() {
                            strain[strain_component] += b[strain_component][dof] * value;
                        }
                    }
                }
            }

            let mut stress = [0.0; 3];
            for row in 0..3 {
                for col in 0..3 {
                    stress[row] += constitutive[row][col] * (strain[col] - thermal_strain[col]);
                }
            }
            StressTensor2D::from_components(stress[0], stress[1], stress[2])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DisplacementConstraint, ElasticityModel, Point2, Tri3};

    #[test]
    fn thermoelastic_unrestrained_bar_thermal_expansion() {
        let mesh = horizontal_bar_mesh(2.0, 0.1);
        let delta_t = 100.0;
        let alpha = 12.0e-6;
        let young_modulus = 200.0e9;
        let length = 2.0;

        let problem = ThermoElasticProblem {
            heat_problem: uniform_temperature_heat_problem(&mesh, delta_t),
            elasticity_problem: ElasticityProblem {
                material: ElasticityMaterial {
                    young_modulus,
                    poisson_ratio: 0.3,
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
                ],
                forces: vec![],
            },
            thermal_expansion_coeff: alpha,
            reference_temperature: 0.0,
        };

        let result = solve_thermoelastic(
            &mesh,
            &problem,
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

        let problem = ThermoElasticProblem {
            heat_problem: uniform_temperature_heat_problem(&mesh, delta_t),
            elasticity_problem: ElasticityProblem {
                material: ElasticityMaterial {
                    young_modulus,
                    poisson_ratio: 0.3,
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
                ],
                forces: vec![],
            },
            thermal_expansion_coeff: alpha,
            reference_temperature: 0.0,
        };

        let result = solve_thermoelastic(
            &mesh,
            &problem,
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

    fn uniform_temperature_heat_problem(
        mesh: &Mesh,
        temperature: f64,
    ) -> SteadyHeatProblem<impl Fn(f64, f64) -> f64> {
        SteadyHeatProblem {
            thermal_conductivity: 1.0,
            heat_generation: |_, _| 0.0,
            prescribed_temperatures: (0..mesh.node_count())
                .map(|node| (node, temperature))
                .collect(),
        }
    }
}
