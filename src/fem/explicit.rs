use std::collections::BTreeMap;

use sprs::CsMat;
use thiserror::Error;

use crate::{
    fem::elasticity::{
        ElasticityError, ElasticityMaterial, ElasticityProblem, NodalForce,
        assemble_elasticity_stiffness_matrix, assemble_force_vector, flatten_displacements_2d,
        unflatten_displacements_2d,
    },
    mesh::Mesh,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitDynamicsOptions {
    pub time_step: f64,
    pub steps: usize,
    pub safety_factor: f64,
}

impl Default for ExplicitDynamicsOptions {
    fn default() -> Self {
        Self {
            time_step: 1.0e-4,
            steps: 10,
            safety_factor: 0.9,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitDynamicsProblem<F> {
    pub material: ElasticityMaterial,
    pub thickness: f64,
    pub density: f64,
    pub constraints: Vec<crate::DisplacementConstraint>,
    pub forces: F,
    pub initial_displacements: Vec<[f64; 2]>,
    pub initial_velocities: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitDynamicsStep {
    pub time: f64,
    pub displacements: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub accelerations: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitDynamicsResult {
    pub steps: Vec<ExplicitDynamicsStep>,
    pub critical_time_step: f64,
    pub used_time_step: f64,
}

#[derive(Debug, Error, PartialEq)]
pub enum ExplicitDynamicsError {
    #[error(transparent)]
    Elasticity(#[from] ElasticityError),
    #[error("time step must be positive and finite, got {0}")]
    InvalidTimeStep(f64),
    #[error("time step {time_step} exceeds critical value {critical_time_step}")]
    UnstableTimeStep {
        time_step: f64,
        critical_time_step: f64,
    },
    #[error("lumped mass entry at dof {dof} is not positive")]
    NonPositiveMass { dof: usize },
}

/// Estimates the critical central-difference time step from diagonal stiffness and lumped mass.
pub fn estimate_critical_time_step(stiffness: &CsMat<f64>, lumped_mass: &[f64]) -> f64 {
    let mut omega_max_sq: f64 = 0.0;
    for (dof, &mass) in lumped_mass.iter().enumerate() {
        if mass <= 0.0 {
            continue;
        }
        let stiffness_diag = stiffness.get(dof, dof).copied().unwrap_or(0.0).max(0.0);
        omega_max_sq = omega_max_sq.max(stiffness_diag / mass);
    }
    if omega_max_sq <= 0.0 {
        return f64::INFINITY;
    }
    2.0 / omega_max_sq.sqrt()
}

pub fn solve_explicit_dynamics<F>(
    mesh: &Mesh,
    problem: &ExplicitDynamicsProblem<F>,
    options: ExplicitDynamicsOptions,
) -> Result<ExplicitDynamicsResult, ExplicitDynamicsError>
where
    F: Fn(f64) -> Vec<NodalForce>,
{
    if !options.time_step.is_finite() || options.time_step <= 0.0 {
        return Err(ExplicitDynamicsError::InvalidTimeStep(options.time_step));
    }

    let elasticity = ElasticityProblem {
        material: problem.material,
        thickness: problem.thickness,
        constraints: problem.constraints.clone(),
        forces: vec![],
    };
    let stiffness =
        assemble_elasticity_stiffness_matrix(mesh, problem.material, problem.thickness)?;
    let lumped_mass = assemble_lumped_mass(mesh, problem.density, problem.thickness);
    let critical_time_step =
        estimate_critical_time_step(&stiffness, &lumped_mass) * options.safety_factor;
    if options.time_step > critical_time_step {
        return Err(ExplicitDynamicsError::UnstableTimeStep {
            time_step: options.time_step,
            critical_time_step,
        });
    }

    let dof_count = mesh.node_count() * 2;
    let fixed = fixed_dofs(&problem.constraints);
    let mut displacements = flatten_displacements_2d(&problem.initial_displacements);
    let mut velocities = flatten_displacements_2d(&problem.initial_velocities);
    apply_dirichlet(&mut displacements, &fixed);
    apply_dirichlet(&mut velocities, &fixed);

    let mut previous_displacements = displacements.clone();
    let mut accelerations = compute_accelerations(
        mesh,
        &stiffness,
        &lumped_mass,
        &displacements,
        &problem.forces,
        0.0,
        &fixed,
    )?;
    apply_dirichlet(&mut accelerations, &fixed);

    let dt = options.time_step;
    let dt_sq = dt * dt;
    let mut steps = Vec::with_capacity(options.steps);

    for step_index in 1..=options.steps {
        let time = step_index as f64 * dt;
        let mut next_displacements = vec![0.0; dof_count];
        let internal = mul_csr_vec(&stiffness, &displacements);
        let external = assemble_force_vector(mesh.node_count(), &(problem.forces)(time))?;

        for dof in 0..dof_count {
            if fixed.contains_key(&dof) {
                next_displacements[dof] = fixed[&dof];
                continue;
            }
            let mass = lumped_mass[dof];
            if mass <= 0.0 {
                return Err(ExplicitDynamicsError::NonPositiveMass { dof });
            }
            let residual = external[dof] - internal[dof];
            next_displacements[dof] =
                2.0 * displacements[dof] - previous_displacements[dof] + dt_sq * residual / mass;
        }

        let mut next_velocities = vec![0.0; dof_count];
        for dof in 0..dof_count {
            if fixed.contains_key(&dof) {
                next_velocities[dof] = 0.0;
            } else {
                next_velocities[dof] =
                    (next_displacements[dof] - previous_displacements[dof]) / (2.0 * dt);
            }
        }

        let next_accelerations = compute_accelerations(
            mesh,
            &stiffness,
            &lumped_mass,
            &next_displacements,
            &problem.forces,
            time,
            &fixed,
        )?;
        previous_displacements.clone_from(&displacements);
        displacements = next_displacements;
        velocities = next_velocities;
        accelerations = next_accelerations;

        steps.push(ExplicitDynamicsStep {
            time,
            displacements: unflatten_displacements_2d(&displacements),
            velocities: unflatten_displacements_2d(&velocities),
            accelerations: unflatten_displacements_2d(&accelerations),
        });
    }

    let _ = elasticity;
    Ok(ExplicitDynamicsResult {
        steps,
        critical_time_step,
        used_time_step: options.time_step,
    })
}

fn compute_accelerations<F>(
    mesh: &Mesh,
    stiffness: &CsMat<f64>,
    lumped_mass: &[f64],
    displacements: &[f64],
    forces: &F,
    time: f64,
    fixed: &BTreeMap<usize, f64>,
) -> Result<Vec<f64>, ExplicitDynamicsError>
where
    F: Fn(f64) -> Vec<NodalForce>,
{
    let external = assemble_force_vector(mesh.node_count(), &forces(time))?;
    let internal = mul_csr_vec(stiffness, displacements);
    let mut accelerations = vec![0.0; displacements.len()];
    for dof in 0..displacements.len() {
        if fixed.contains_key(&dof) {
            accelerations[dof] = 0.0;
            continue;
        }
        let mass = lumped_mass[dof];
        if mass <= 0.0 {
            return Err(ExplicitDynamicsError::NonPositiveMass { dof });
        }
        accelerations[dof] = (external[dof] - internal[dof]) / mass;
    }
    Ok(accelerations)
}

fn assemble_lumped_mass(mesh: &Mesh, density: f64, thickness: f64) -> Vec<f64> {
    let mut lumped = vec![0.0; mesh.node_count() * 2];
    for cell in mesh.cells() {
        if cell.kind != crate::mesh::ElementKind::Tri3 {
            continue;
        }
        let n0 = cell.nodes[0];
        let n1 = cell.nodes[1];
        let n2 = cell.nodes[2];
        let p0 = mesh.points()[n0];
        let p1 = mesh.points()[n1];
        let p2 = mesh.points()[n2];
        let area = 0.5 * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)).abs();
        let element_mass = density * thickness * area / 3.0;
        for node in [n0, n1, n2] {
            lumped[node * 2] += element_mass;
            lumped[node * 2 + 1] += element_mass;
        }
    }
    lumped
}

fn fixed_dofs(constraints: &[crate::DisplacementConstraint]) -> BTreeMap<usize, f64> {
    let mut fixed = BTreeMap::new();
    for constraint in constraints {
        let dof = crate::fem::elasticity::dof_index(constraint.node, constraint.component);
        fixed.insert(dof, constraint.value);
    }
    fixed
}

fn apply_dirichlet(values: &mut [f64], fixed: &BTreeMap<usize, f64>) {
    for (&dof, &value) in fixed {
        values[dof] = value;
    }
}

fn mul_csr_vec(matrix: &CsMat<f64>, vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.rows()];
    for (row_index, row) in matrix.outer_iterator().enumerate() {
        for (col_index, value) in row.iter() {
            result[row_index] += value * vector[col_index];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DisplacementComponent, DisplacementConstraint, ElasticityModel, Mesh, Point2, Tri3,
        linalg::norm,
    };

    #[test]
    fn explicit_dynamics_advances_free_vibration() {
        let mesh = Mesh::new(
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(0.0, 1.0),
            ],
            vec![Tri3::new([0, 1, 2])],
        )
        .unwrap();

        let problem = ExplicitDynamicsProblem {
            material: ElasticityMaterial {
                young_modulus: 100.0,
                poisson_ratio: 0.25,
                model: ElasticityModel::PlaneStress,
            },
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
                DisplacementConstraint {
                    node: 2,
                    component: DisplacementComponent::Y,
                    value: 0.0,
                },
            ],
            forces: |_| vec![],
            initial_displacements: vec![[0.0, 0.0], [0.01, 0.0], [0.0, 0.0]],
            initial_velocities: vec![[0.0, 0.0]; 3],
        };

        let stiffness =
            assemble_elasticity_stiffness_matrix(&mesh, problem.material, problem.thickness)
                .unwrap();
        let lumped = assemble_lumped_mass(&mesh, problem.density, problem.thickness);
        let critical = estimate_critical_time_step(&stiffness, &lumped);

        let result = solve_explicit_dynamics(
            &mesh,
            &problem,
            ExplicitDynamicsOptions {
                time_step: 0.25 * critical,
                steps: 5,
                safety_factor: 0.9,
            },
        )
        .unwrap();

        assert_eq!(result.steps.len(), 5);
        let final_ux = result.steps.last().unwrap().displacements[1][0];
        assert!(final_ux.is_finite());
        assert!(norm(&[final_ux]).abs() > 0.0);
    }
}
