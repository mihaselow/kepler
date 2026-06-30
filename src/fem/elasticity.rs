use std::collections::BTreeMap;

use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    fem::element::{Element, ElementError},
    linalg::{
        LinalgError, LinearSolverOptions, NewmarkSolverOptions, SolverDiagnostics, SolverOptions,
        solve_linear_system, solve_newmark_transient,
    },
    mesh::{ElementKind, Mesh, MeshTopology, NodeId, Point3, Tri3},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElasticityModel {
    PlaneStress,
    PlaneStrain,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticityMaterial {
    pub young_modulus: f64,
    pub poisson_ratio: f64,
    pub model: ElasticityModel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticityMaterial3D {
    pub young_modulus: f64,
    pub poisson_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DisplacementComponent {
    X,
    Y,
}

impl DisplacementComponent {
    fn offset(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DisplacementComponent3D {
    X,
    Y,
    Z,
}

impl DisplacementComponent3D {
    fn offset(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplacementConstraint {
    pub node: NodeId,
    pub component: DisplacementComponent,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplacementConstraint3D {
    pub node: NodeId,
    pub component: DisplacementComponent3D,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodalForce {
    pub node: NodeId,
    pub fx: f64,
    pub fy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodalForce3D {
    pub node: NodeId,
    pub fx: f64,
    pub fy: f64,
    pub fz: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticityProblem {
    pub material: ElasticityMaterial,
    pub thickness: f64,
    pub constraints: Vec<DisplacementConstraint>,
    pub forces: Vec<NodalForce>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticityProblem3D {
    pub material: ElasticityMaterial3D,
    pub constraints: Vec<DisplacementConstraint3D>,
    pub forces: Vec<NodalForce3D>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientElasticityProblem<F> {
    pub material: ElasticityMaterial,
    pub thickness: f64,
    pub density: f64,
    pub constraints: Vec<DisplacementConstraint>,
    pub forces: F,
    pub initial_displacements: Vec<[f64; 2]>,
    pub initial_velocities: Vec<[f64; 2]>,
    pub rayleigh_alpha: Option<f64>,
    pub rayleigh_beta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientElasticityProblem3D<F> {
    pub material: ElasticityMaterial3D,
    pub density: f64,
    pub constraints: Vec<DisplacementConstraint3D>,
    pub forces: F,
    pub initial_displacements: Vec<[f64; 3]>,
    pub initial_velocities: Vec<[f64; 3]>,
    pub rayleigh_alpha: Option<f64>,
    pub rayleigh_beta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticityResult {
    pub displacements: Vec<[f64; 2]>,
    pub iterations: usize,
    pub residual_norm: f64,
}

/// 2D stress tensor components at a single Gauss point or node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressTensor2D {
    pub sigma_xx: f64,
    pub sigma_yy: f64,
    pub sigma_xy: f64,
    /// von Mises equivalent stress.
    pub von_mises: f64,
    /// Maximum principal stress.
    pub principal_max: f64,
    /// Minimum principal stress.
    pub principal_min: f64,
}

impl StressTensor2D {
    pub fn from_components(sigma_xx: f64, sigma_yy: f64, sigma_xy: f64) -> Self {
        let vm =
            (sigma_xx.powi(2) - sigma_xx * sigma_yy + sigma_yy.powi(2) + 3.0 * sigma_xy.powi(2))
                .sqrt();
        let avg = (sigma_xx + sigma_yy) / 2.0;
        let r = (((sigma_xx - sigma_yy) / 2.0).powi(2) + sigma_xy.powi(2)).sqrt();
        Self {
            sigma_xx,
            sigma_yy,
            sigma_xy,
            von_mises: vm,
            principal_max: avg + r,
            principal_min: avg - r,
        }
    }
}

/// 2D strain tensor at a single point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrainTensor2D {
    pub eps_xx: f64,
    pub eps_yy: f64,
    /// Engineering shear strain (2 * eps_xy).
    pub gamma_xy: f64,
}

/// 3D stress tensor at a single point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressTensor3D {
    pub sigma_xx: f64,
    pub sigma_yy: f64,
    pub sigma_zz: f64,
    pub sigma_xy: f64,
    pub sigma_yz: f64,
    pub sigma_xz: f64,
    /// von Mises equivalent stress.
    pub von_mises: f64,
}

impl StressTensor3D {
    pub fn from_components(
        sigma_xx: f64,
        sigma_yy: f64,
        sigma_zz: f64,
        sigma_xy: f64,
        sigma_yz: f64,
        sigma_xz: f64,
    ) -> Self {
        let vm = (0.5
            * ((sigma_xx - sigma_yy).powi(2)
                + (sigma_yy - sigma_zz).powi(2)
                + (sigma_zz - sigma_xx).powi(2)
                + 6.0 * (sigma_xy.powi(2) + sigma_yz.powi(2) + sigma_xz.powi(2))))
        .sqrt();
        Self {
            sigma_xx,
            sigma_yy,
            sigma_zz,
            sigma_xy,
            sigma_yz,
            sigma_xz,
            von_mises: vm,
        }
    }
}

/// 3D strain tensor at a single point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrainTensor3D {
    pub eps_xx: f64,
    pub eps_yy: f64,
    pub eps_zz: f64,
    pub gamma_xy: f64,
    pub gamma_yz: f64,
    pub gamma_xz: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticitySolverResult {
    pub displacements: Vec<[f64; 2]>,
    pub diagnostics: SolverDiagnostics,
    /// Element-centroid stress tensors (one per Tri3 element).
    pub element_stress: Vec<StressTensor2D>,
    /// Element-centroid strain tensors (one per Tri3 element).
    pub element_strain: Vec<StrainTensor2D>,
    /// Node-averaged stress tensors (one per mesh node).
    pub nodal_stress: Vec<StressTensor2D>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticityResult3D {
    pub displacements: Vec<[f64; 3]>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticitySolverResult3D {
    pub displacements: Vec<[f64; 3]>,
    pub diagnostics: SolverDiagnostics,
    /// Element-centroid stress tensors (one per Tet4 element).
    pub element_stress: Vec<StressTensor3D>,
    /// Element-centroid strain tensors (one per Tet4 element).
    pub element_strain: Vec<StrainTensor3D>,
    /// Node-averaged stress tensors (one per mesh node).
    pub nodal_stress: Vec<StressTensor3D>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientElasticityStep {
    pub time: f64,
    pub displacements: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub accelerations: Vec<[f64; 2]>,
    pub diagnostics: SolverDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientElasticityResult {
    pub steps: Vec<TransientElasticityStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientElasticityStep3D {
    pub time: f64,
    pub displacements: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub accelerations: Vec<[f64; 3]>,
    pub diagnostics: SolverDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientElasticityResult3D {
    pub steps: Vec<TransientElasticityStep3D>,
}

impl From<ElasticitySolverResult> for ElasticityResult {
    fn from(value: ElasticitySolverResult) -> Self {
        Self {
            displacements: value.displacements,
            iterations: value.diagnostics.iterations,
            residual_norm: value.diagnostics.residual_norm,
        }
    }
}

impl From<ElasticitySolverResult3D> for ElasticityResult3D {
    fn from(value: ElasticitySolverResult3D) -> Self {
        Self {
            displacements: value.displacements,
            iterations: value.diagnostics.iterations,
            residual_norm: value.diagnostics.residual_norm,
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum ElasticityError {
    #[error("Young's modulus must be positive and finite, got {0}")]
    InvalidYoungModulus(f64),
    #[error("Poisson ratio must be finite and in (-1, 0.5), got {0}")]
    InvalidPoissonRatio(f64),
    #[error("thickness must be positive and finite, got {0}")]
    InvalidThickness(f64),
    #[error("density must be positive and finite, got {0}")]
    InvalidDensity(f64),
    #[error("initial state length {initial_len} does not match mesh node count {node_count}")]
    InitialStateLengthMismatch {
        node_count: usize,
        initial_len: usize,
    },
    #[error("constraint references node {node_id}, but mesh has {node_count} nodes")]
    ConstraintNodeOutOfBounds { node_id: NodeId, node_count: usize },
    #[error("force references node {node_id}, but mesh has {node_count} nodes")]
    ForceNodeOutOfBounds { node_id: NodeId, node_count: usize },
    #[error("constraint for node {node_id} component {component:?} was specified more than once")]
    DuplicateConstraint {
        node_id: NodeId,
        component: DisplacementComponent,
    },
    #[error(
        "3D constraint for node {node_id} component {component:?} was specified more than once"
    )]
    DuplicateConstraint3D {
        node_id: NodeId,
        component: DisplacementComponent3D,
    },
    #[error("cell {cell_index} has unsupported element kind {kind:?} for 3D elasticity")]
    UnsupportedElementKind {
        cell_index: usize,
        kind: ElementKind,
    },
    #[error("all displacement degrees of freedom are constrained")]
    NoActiveDegreesOfFreedom,
    #[error("linear solver failed")]
    LinearSolve(#[from] LinalgError),
}

pub fn solve_elasticity(
    mesh: &Mesh,
    problem: &ElasticityProblem,
    options: SolverOptions,
) -> Result<ElasticityResult, ElasticityError> {
    solve_elasticity_with_solver(mesh, problem, LinearSolverOptions::from(options))
        .map(ElasticityResult::from)
}

pub fn solve_elasticity_with_solver(
    mesh: &Mesh,
    problem: &ElasticityProblem,
    options: LinearSolverOptions,
) -> Result<ElasticitySolverResult, ElasticityError> {
    let (matrix, rhs) = assemble_elasticity_system(mesh, problem)?;
    let result = solve_linear_system(&matrix, &rhs, options)?;
    let displacements: Vec<[f64; 2]> = result
        .values
        .chunks_exact(2)
        .map(|values| [values[0], values[1]])
        .collect();

    let (element_stress, element_strain, nodal_stress) =
        recover_stress_2d(mesh, &problem.material, &displacements);

    Ok(ElasticitySolverResult {
        displacements,
        diagnostics: result.diagnostics,
        element_stress,
        element_strain,
        nodal_stress,
    })
}

pub fn solve_elasticity_3d(
    mesh: &MeshTopology<3>,
    problem: &ElasticityProblem3D,
    options: SolverOptions,
) -> Result<ElasticityResult3D, ElasticityError> {
    solve_elasticity_3d_with_solver(mesh, problem, LinearSolverOptions::from(options))
        .map(ElasticityResult3D::from)
}

pub fn solve_elasticity_3d_with_solver(
    mesh: &MeshTopology<3>,
    problem: &ElasticityProblem3D,
    options: LinearSolverOptions,
) -> Result<ElasticitySolverResult3D, ElasticityError> {
    let (matrix, rhs) = assemble_elasticity_3d_system(mesh, problem)?;
    let result = solve_linear_system(&matrix, &rhs, options)?;
    let displacements: Vec<[f64; 3]> = result
        .values
        .chunks_exact(3)
        .map(|values| [values[0], values[1], values[2]])
        .collect();

    let (element_stress, element_strain, nodal_stress) =
        recover_stress_3d(mesh, mesh.points(), &problem.material, &displacements);

    Ok(ElasticitySolverResult3D {
        displacements,
        diagnostics: result.diagnostics,
        element_stress,
        element_strain,
        nodal_stress,
    })
}

pub fn solve_transient_elasticity<F>(
    mesh: &Mesh,
    problem: &TransientElasticityProblem<F>,
    options: NewmarkSolverOptions,
) -> Result<TransientElasticityResult, ElasticityError>
where
    F: Fn(f64) -> Vec<NodalForce>,
{
    validate_material(problem.material)?;
    validate_thickness(problem.thickness)?;
    validate_density(problem.density)?;
    validate_initial_state_lengths(
        mesh.node_count(),
        problem.initial_displacements.len(),
        problem.initial_velocities.len(),
    )?;
    let constraints = validate_constraints(mesh.node_count(), &problem.constraints)?;
    let active_dofs = active_dofs(mesh.node_count() * 2, &constraints);
    if active_dofs.is_empty() {
        return Err(ElasticityError::NoActiveDegreesOfFreedom);
    }
    let active_map = active_dof_map(mesh.node_count() * 2, &active_dofs);
    let stiffness =
        assemble_elasticity_stiffness_matrix(mesh, problem.material, problem.thickness)?;
    let mass = assemble_lumped_elasticity_mass(mesh, problem.density, problem.thickness);
    let reduced_stiffness = reduce_matrix(&stiffness, &active_dofs, &active_map);
    let reduced_mass = reduce_matrix(&mass, &active_dofs, &active_map);
    let initial_displacements = reduce_vector(
        &flatten_displacements_2d(&problem.initial_displacements),
        &active_dofs,
    );
    let initial_velocities = reduce_vector(
        &flatten_displacements_2d(&problem.initial_velocities),
        &active_dofs,
    );
    let source_values = transient_elasticity_sources(
        mesh,
        problem,
        &stiffness,
        &active_dofs,
        &constraints,
        &options,
    )?;
    let time_step = options.time_step;

    let reduced_damping = if problem.rayleigh_alpha.is_some() || problem.rayleigh_beta.is_some() {
        let alpha = problem.rayleigh_alpha.unwrap_or(0.0);
        let beta = problem.rayleigh_beta.unwrap_or(0.0);
        let mut triplets = TriMat::new((reduced_mass.rows(), reduced_mass.cols()));
        for (row_index, row) in reduced_mass.outer_iterator().enumerate() {
            for (col_index, &value) in row.iter() {
                triplets.add_triplet(row_index, col_index, alpha * value);
            }
        }
        for (row_index, row) in reduced_stiffness.outer_iterator().enumerate() {
            for (col_index, &value) in row.iter() {
                triplets.add_triplet(row_index, col_index, beta * value);
            }
        }
        Some(triplets.to_csr())
    } else {
        None
    };

    let reduced_steps = solve_newmark_transient(
        &reduced_mass,
        reduced_damping.as_ref(),
        &reduced_stiffness,
        initial_displacements,
        initial_velocities,
        move |time| {
            let index = (time / time_step).round() as usize;
            source_values[index].clone()
        },
        options,
    )?;

    Ok(TransientElasticityResult {
        steps: reduced_steps
            .into_iter()
            .map(|step| TransientElasticityStep {
                time: step.time,
                displacements: unflatten_displacements_2d(&reconstruct_values(
                    mesh.node_count() * 2,
                    &active_dofs,
                    &constraints,
                    &step.displacements,
                )),
                velocities: unflatten_displacements_2d(&reconstruct_values(
                    mesh.node_count() * 2,
                    &active_dofs,
                    &BTreeMap::new(),
                    &step.velocities,
                )),
                accelerations: unflatten_displacements_2d(&reconstruct_values(
                    mesh.node_count() * 2,
                    &active_dofs,
                    &BTreeMap::new(),
                    &step.accelerations,
                )),
                diagnostics: step.linear_diagnostics,
            })
            .collect(),
    })
}

pub fn solve_transient_elasticity_3d<F>(
    mesh: &MeshTopology<3>,
    problem: &TransientElasticityProblem3D<F>,
    options: NewmarkSolverOptions,
) -> Result<TransientElasticityResult3D, ElasticityError>
where
    F: Fn(f64) -> Vec<NodalForce3D>,
{
    validate_material_3d(problem.material)?;
    validate_density(problem.density)?;
    validate_initial_state_lengths(
        mesh.points().len(),
        problem.initial_displacements.len(),
        problem.initial_velocities.len(),
    )?;
    let constraints = validate_constraints_3d(mesh.points().len(), &problem.constraints)?;
    let active_dofs = active_dofs(mesh.points().len() * 3, &constraints);
    if active_dofs.is_empty() {
        return Err(ElasticityError::NoActiveDegreesOfFreedom);
    }
    let active_map = active_dof_map(mesh.points().len() * 3, &active_dofs);
    let stiffness = assemble_elasticity_3d_stiffness_matrix(mesh, problem.material)?;
    let mass = assemble_lumped_elasticity_mass_3d(mesh, problem.density);
    let reduced_stiffness = reduce_matrix(&stiffness, &active_dofs, &active_map);
    let reduced_mass = reduce_matrix(&mass, &active_dofs, &active_map);
    let initial_displacements = reduce_vector(
        &flatten_displacements_3d(&problem.initial_displacements),
        &active_dofs,
    );
    let initial_velocities = reduce_vector(
        &flatten_displacements_3d(&problem.initial_velocities),
        &active_dofs,
    );
    let source_values = transient_elasticity_sources_3d(
        mesh,
        problem,
        &stiffness,
        &active_dofs,
        &constraints,
        &options,
    )?;
    let time_step = options.time_step;

    let reduced_damping = if problem.rayleigh_alpha.is_some() || problem.rayleigh_beta.is_some() {
        let alpha = problem.rayleigh_alpha.unwrap_or(0.0);
        let beta = problem.rayleigh_beta.unwrap_or(0.0);
        let mut triplets = TriMat::new((reduced_mass.rows(), reduced_mass.cols()));
        for (row_index, row) in reduced_mass.outer_iterator().enumerate() {
            for (col_index, &value) in row.iter() {
                triplets.add_triplet(row_index, col_index, alpha * value);
            }
        }
        for (row_index, row) in reduced_stiffness.outer_iterator().enumerate() {
            for (col_index, &value) in row.iter() {
                triplets.add_triplet(row_index, col_index, beta * value);
            }
        }
        Some(triplets.to_csr())
    } else {
        None
    };

    let reduced_steps = solve_newmark_transient(
        &reduced_mass,
        reduced_damping.as_ref(),
        &reduced_stiffness,
        initial_displacements,
        initial_velocities,
        move |time| {
            let index = (time / time_step).round() as usize;
            source_values[index].clone()
        },
        options,
    )?;

    Ok(TransientElasticityResult3D {
        steps: reduced_steps
            .into_iter()
            .map(|step| TransientElasticityStep3D {
                time: step.time,
                displacements: unflatten_displacements_3d(&reconstruct_values(
                    mesh.points().len() * 3,
                    &active_dofs,
                    &constraints,
                    &step.displacements,
                )),
                velocities: unflatten_displacements_3d(&reconstruct_values(
                    mesh.points().len() * 3,
                    &active_dofs,
                    &BTreeMap::new(),
                    &step.velocities,
                )),
                accelerations: unflatten_displacements_3d(&reconstruct_values(
                    mesh.points().len() * 3,
                    &active_dofs,
                    &BTreeMap::new(),
                    &step.accelerations,
                )),
                diagnostics: step.linear_diagnostics,
            })
            .collect(),
    })
}

pub fn assemble_elasticity_system(
    mesh: &Mesh,
    problem: &ElasticityProblem,
) -> Result<(CsMat<f64>, Vec<f64>), ElasticityError> {
    validate_material(problem.material)?;
    validate_thickness(problem.thickness)?;
    let constraints = validate_constraints(mesh.node_count(), &problem.constraints)?;
    let dof_count = mesh.node_count() * 2;
    let matrix = assemble_elasticity_stiffness_matrix(mesh, problem.material, problem.thickness)?;
    let mut rhs = vec![0.0; dof_count];

    for force in &problem.forces {
        if force.node >= mesh.node_count() {
            return Err(ElasticityError::ForceNodeOutOfBounds {
                node_id: force.node,
                node_count: mesh.node_count(),
            });
        }
        rhs[dof_index(force.node, DisplacementComponent::X)] += force.fx;
        rhs[dof_index(force.node, DisplacementComponent::Y)] += force.fy;
    }

    Ok(apply_constraints(matrix, rhs, &constraints))
}

pub fn assemble_elasticity_3d_system(
    mesh: &MeshTopology<3>,
    problem: &ElasticityProblem3D,
) -> Result<(CsMat<f64>, Vec<f64>), ElasticityError> {
    validate_material_3d(problem.material)?;
    let constraints = validate_constraints_3d(mesh.points().len(), &problem.constraints)?;
    let dof_count = mesh.points().len() * 3;
    let matrix = assemble_elasticity_3d_stiffness_matrix(mesh, problem.material)?;
    let mut rhs = vec![0.0; dof_count];

    for force in &problem.forces {
        if force.node >= mesh.points().len() {
            return Err(ElasticityError::ForceNodeOutOfBounds {
                node_id: force.node,
                node_count: mesh.points().len(),
            });
        }
        rhs[dof_index_3d(force.node, DisplacementComponent3D::X)] += force.fx;
        rhs[dof_index_3d(force.node, DisplacementComponent3D::Y)] += force.fy;
        rhs[dof_index_3d(force.node, DisplacementComponent3D::Z)] += force.fz;
    }

    Ok(apply_constraints(matrix, rhs, &constraints))
}

fn assemble_elasticity_stiffness_matrix(
    mesh: &Mesh,
    material: ElasticityMaterial,
    thickness: f64,
) -> Result<CsMat<f64>, ElasticityError> {
    use crate::parallel::{Triplet, merge_triplets};
    use rayon::prelude::*;

    let dof_count = mesh.node_count() * 2;
    let cells = mesh.cells().to_vec();
    let element_triplets = cells
        .par_iter()
        .map(|cell| {
            let node_coords: Vec<Point3> = cell
                .nodes
                .iter()
                .map(|&id| Point3::new([mesh.points()[id].x, mesh.points()[id].y, 0.0]))
                .collect();

            let mut properties = BTreeMap::new();
            properties.insert("young_modulus".to_string(), material.young_modulus);
            properties.insert("poisson_ratio".to_string(), material.poisson_ratio);
            properties.insert("thickness".to_string(), thickness);
            let model_val = match material.model {
                ElasticityModel::PlaneStress => 0.0,
                ElasticityModel::PlaneStrain => 1.0,
            };
            properties.insert("model".to_string(), model_val);

            let local_k = match cell.kind {
                ElementKind::Tri3 => {
                    let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2]];
                    let el = ElasticityTri3 {
                        nodes: &nodes,
                        thickness,
                    };
                    el.local_stiffness(&node_coords, &properties)
                }
                ElementKind::Quad4 => {
                    let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
                    let el = ElasticityQuad4 {
                        nodes: &nodes,
                        thickness,
                    };
                    el.local_stiffness(&node_coords, &properties)
                }
                ElementKind::Tri6 => {
                    let nodes = [
                        cell.nodes[0],
                        cell.nodes[1],
                        cell.nodes[2],
                        cell.nodes[3],
                        cell.nodes[4],
                        cell.nodes[5],
                    ];
                    let el = ElasticityTri6 {
                        nodes: &nodes,
                        thickness,
                    };
                    el.local_stiffness(&node_coords, &properties)
                }
                _ => {
                    return Err(ElasticityError::UnsupportedElementKind {
                        cell_index: 0,
                        kind: cell.kind,
                    });
                }
            }
            .map_err(|_| ElasticityError::UnsupportedElementKind {
                cell_index: 0,
                kind: cell.kind,
            })?;

            let mut dofs = Vec::with_capacity(cell.nodes.len() * 2);
            for &node in &cell.nodes {
                dofs.push(node * 2);
                dofs.push(node * 2 + 1);
            }

            let mut triplets = Vec::with_capacity(dofs.len() * dofs.len());
            for (local_row, global_row) in dofs.iter().copied().enumerate() {
                for (local_col, global_col) in dofs.iter().copied().enumerate() {
                    triplets.push(Triplet {
                        row: global_row,
                        col: global_col,
                        val: local_k[local_row][local_col],
                    });
                }
            }
            Ok(triplets)
        })
        .collect::<Result<Vec<_>, ElasticityError>>()?;

    let tri = merge_triplets(dof_count, element_triplets);
    Ok(tri.to_csr())
}

fn assemble_elasticity_3d_stiffness_matrix(
    mesh: &MeshTopology<3>,
    material: ElasticityMaterial3D,
) -> Result<CsMat<f64>, ElasticityError> {
    use crate::parallel::{Triplet, merge_triplets};
    use rayon::prelude::*;

    // Check for unsupported element kinds before launching parallel work.
    for (cell_index, cell) in mesh.cells().iter().enumerate() {
        match cell.kind {
            ElementKind::Hex20 | ElementKind::Tet10 => {
                return Err(ElasticityError::UnsupportedElementKind {
                    cell_index,
                    kind: cell.kind,
                });
            }
            _ => {}
        }
    }

    let dof_count = mesh.points().len() * 3;
    let cells: Vec<_> = mesh.cells().to_vec();
    let element_triplets = cells
        .par_iter()
        .filter_map(|cell| {
            let stiffness = match cell.kind {
                ElementKind::Tet4 => {
                    let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
                    local_tet4_elasticity_stiffness(mesh, nodes, material)
                        .map(|arr| arr.iter().map(|row| row.to_vec()).collect::<Vec<_>>())
                }
                ElementKind::Hex8 => {
                    let nodes = [
                        cell.nodes[0],
                        cell.nodes[1],
                        cell.nodes[2],
                        cell.nodes[3],
                        cell.nodes[4],
                        cell.nodes[5],
                        cell.nodes[6],
                        cell.nodes[7],
                    ];
                    let el = ElasticityHex8 { nodes: &nodes };
                    let node_coords: Vec<Point3> = nodes
                        .iter()
                        .map(|&node_id| mesh.points()[node_id])
                        .collect();
                    let mut properties = BTreeMap::new();
                    properties.insert("young_modulus".to_string(), material.young_modulus);
                    properties.insert("poisson_ratio".to_string(), material.poisson_ratio);
                    el.local_stiffness(&node_coords, &properties).map_err(|_| {
                        ElasticityError::UnsupportedElementKind {
                            cell_index: 0,
                            kind: cell.kind,
                        }
                    })
                }
                _ => return None,
            };

            let stiffness = match stiffness {
                Ok(s) => s,
                Err(e) => return Some(Err(e)),
            };

            let mut dofs = Vec::with_capacity(cell.nodes.len() * 3);
            for &node in &cell.nodes {
                dofs.push(node * 3);
                dofs.push(node * 3 + 1);
                dofs.push(node * 3 + 2);
            }

            let mut triplets = Vec::with_capacity(dofs.len() * dofs.len());
            for (local_row, global_row) in dofs.iter().copied().enumerate() {
                for (local_col, global_col) in dofs.iter().copied().enumerate() {
                    triplets.push(Triplet {
                        row: global_row,
                        col: global_col,
                        val: stiffness[local_row][local_col],
                    });
                }
            }
            Some(Ok(triplets))
        })
        .collect::<Result<Vec<_>, ElasticityError>>()?;

    let tri = merge_triplets(dof_count, element_triplets);
    Ok(tri.to_csr())
}

pub fn local_elasticity_stiffness(
    mesh: &Mesh,
    triangle: &Tri3,
    material: ElasticityMaterial,
    thickness: f64,
) -> Result<[[f64; 6]; 6], ElasticityError> {
    validate_material(material)?;
    validate_thickness(thickness)?;

    let el = ElasticityTri3 {
        nodes: &triangle.nodes,
        thickness,
    };
    let node_coords: Vec<Point3> = triangle
        .nodes
        .iter()
        .map(|&node_id| {
            let p2 = mesh.points()[node_id];
            Point3::new([p2.x, p2.y, 0.0])
        })
        .collect();

    let mut properties = BTreeMap::new();
    properties.insert("young_modulus".to_string(), material.young_modulus);
    properties.insert("poisson_ratio".to_string(), material.poisson_ratio);
    properties.insert("thickness".to_string(), thickness);
    let model_val = match material.model {
        ElasticityModel::PlaneStress => 0.0,
        ElasticityModel::PlaneStrain => 1.0,
    };
    properties.insert("model".to_string(), model_val);

    let stiffness_vec = el.local_stiffness(&node_coords, &properties).map_err(|_| {
        ElasticityError::UnsupportedElementKind {
            cell_index: 0,
            kind: ElementKind::Tri3,
        }
    })?;

    let mut stiffness = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            stiffness[r][c] = stiffness_vec[r][c];
        }
    }
    Ok(stiffness)
}

pub fn local_tet4_elasticity_stiffness(
    mesh: &MeshTopology<3>,
    nodes: [NodeId; 4],
    material: ElasticityMaterial3D,
) -> Result<[[f64; 12]; 12], ElasticityError> {
    validate_material_3d(material)?;

    let el = ElasticityTet4 { nodes: &nodes };
    let node_coords: Vec<Point3> = nodes
        .iter()
        .map(|&node_id| mesh.points()[node_id])
        .collect();

    let mut properties = BTreeMap::new();
    properties.insert("young_modulus".to_string(), material.young_modulus);
    properties.insert("poisson_ratio".to_string(), material.poisson_ratio);

    let stiffness_vec = el.local_stiffness(&node_coords, &properties).map_err(|_| {
        ElasticityError::UnsupportedElementKind {
            cell_index: 0,
            kind: ElementKind::Tet4,
        }
    })?;

    let mut stiffness = [[0.0; 12]; 12];
    for r in 0..12 {
        for c in 0..12 {
            stiffness[r][c] = stiffness_vec[r][c];
        }
    }
    Ok(stiffness)
}

pub struct ElasticityTri3<'a> {
    pub nodes: &'a [NodeId; 3],
    pub thickness: f64,
}

impl<'a> Element for ElasticityTri3<'a> {
    fn spatial_dimension(&self) -> usize {
        2
    }
    fn node_count(&self) -> usize {
        3
    }
    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }
    fn active_fields(&self) -> Vec<String> {
        vec!["ux".to_string(), "uy".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let young_modulus = *properties
            .get("young_modulus")
            .ok_or_else(|| ElementError::MissingProperty("young_modulus".to_string()))?;
        let poisson_ratio = *properties
            .get("poisson_ratio")
            .ok_or_else(|| ElementError::MissingProperty("poisson_ratio".to_string()))?;
        let thickness = *properties
            .get("thickness")
            .ok_or_else(|| ElementError::MissingProperty("thickness".to_string()))?;
        let model_val = *properties.get("model").unwrap_or(&0.0);
        let model = if model_val == 1.0 {
            ElasticityModel::PlaneStrain
        } else {
            ElasticityModel::PlaneStress
        };

        if node_coords.len() != 3 {
            return Err(ElementError::InvalidNodeCount {
                expected: 3,
                actual: node_coords.len(),
            });
        }

        let a = node_coords[0].coords;
        let b = node_coords[1].coords;
        let c = node_coords[2].coords;
        let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        let area = 0.5 * twice_area.abs();
        if area <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }

        let gradients = [
            [(b[1] - c[1]) / twice_area, (c[0] - b[0]) / twice_area],
            [(c[1] - a[1]) / twice_area, (a[0] - c[0]) / twice_area],
            [(a[1] - b[1]) / twice_area, (b[0] - a[0]) / twice_area],
        ];

        let constitutive = constitutive_matrix(ElasticityMaterial {
            young_modulus,
            poisson_ratio,
            model,
        });
        let strain_displacement = strain_displacement_matrix(gradients);

        let mut stiffness = vec![vec![0.0; 6]; 6];
        for row in 0..6 {
            for col in 0..6 {
                let mut value = 0.0;
                for alpha in 0..3 {
                    for beta in 0..3 {
                        value += strain_displacement[alpha][row]
                            * constitutive[alpha][beta]
                            * strain_displacement[beta][col];
                    }
                }
                stiffness[row][col] = thickness * area * value;
            }
        }
        Ok(stiffness)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 3 {
            return Err(ElementError::InvalidNodeCount {
                expected: 3,
                actual: node_coords.len(),
            });
        }
        let a = node_coords[0].coords;
        let b = node_coords[1].coords;
        let c = node_coords[2].coords;
        let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        let area = 0.5 * twice_area.abs();
        if area <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }
        let total_mass = density * self.thickness * area;
        let mut mass = vec![vec![0.0; 6]; 6];
        if lumped {
            let nodal_mass = total_mass / 3.0;
            for i in 0..3 {
                mass[2 * i][2 * i] = nodal_mass;
                mass[2 * i + 1][2 * i + 1] = nodal_mass;
            }
        } else {
            let val_diag = total_mass / 6.0;
            let val_off = total_mass / 12.0;
            for i in 0..3 {
                for j in 0..3 {
                    let factor = if i == j { val_diag } else { val_off };
                    mass[2 * i][2 * j] = factor;
                    mass[2 * i + 1][2 * j + 1] = factor;
                }
            }
        }
        Ok(mass)
    }
}

pub struct ElasticityQuad4<'a> {
    pub nodes: &'a [NodeId; 4],
    pub thickness: f64,
}

impl<'a> Element for ElasticityQuad4<'a> {
    fn spatial_dimension(&self) -> usize {
        2
    }
    fn node_count(&self) -> usize {
        4
    }
    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }
    fn active_fields(&self) -> Vec<String> {
        vec!["ux".to_string(), "uy".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let young_modulus = *properties
            .get("young_modulus")
            .ok_or_else(|| ElementError::MissingProperty("young_modulus".to_string()))?;
        let poisson_ratio = *properties
            .get("poisson_ratio")
            .ok_or_else(|| ElementError::MissingProperty("poisson_ratio".to_string()))?;
        let thickness = *properties
            .get("thickness")
            .ok_or_else(|| ElementError::MissingProperty("thickness".to_string()))?;
        let model_val = *properties.get("model").unwrap_or(&0.0);
        let model = if model_val == 1.0 {
            ElasticityModel::PlaneStrain
        } else {
            ElasticityModel::PlaneStress
        };

        if node_coords.len() != 4 {
            return Err(ElementError::InvalidNodeCount {
                expected: 4,
                actual: node_coords.len(),
            });
        }

        let constitutive = constitutive_matrix(ElasticityMaterial {
            young_modulus,
            poisson_ratio,
            model,
        });

        let g_pts = [-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()];
        let mut stiffness = vec![vec![0.0; 8]; 8];

        for &xi in &g_pts {
            for &eta in &g_pts {
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

                let mut j11 = 0.0;
                let mut j12 = 0.0;
                let mut j21 = 0.0;
                let mut j22 = 0.0;
                for a in 0..4 {
                    j11 += dn_dxi[a] * node_coords[a].coords[0];
                    j12 += dn_dxi[a] * node_coords[a].coords[1];
                    j21 += dn_deta[a] * node_coords[a].coords[0];
                    j22 += dn_deta[a] * node_coords[a].coords[1];
                }

                let det_j = j11 * j22 - j12 * j21;
                if det_j.abs() <= f64::EPSILON {
                    return Err(ElementError::DegenerateGeometry);
                }

                let inv_j11 = j22 / det_j;
                let inv_j12 = -j12 / det_j;
                let inv_j21 = -j21 / det_j;
                let inv_j22 = j11 / det_j;

                let mut dn_dx = [0.0; 4];
                let mut dn_dy = [0.0; 4];
                for a in 0..4 {
                    dn_dx[a] = inv_j11 * dn_dxi[a] + inv_j12 * dn_deta[a];
                    dn_dy[a] = inv_j21 * dn_dxi[a] + inv_j22 * dn_deta[a];
                }

                let mut b_mat = [[0.0; 8]; 3];
                for a in 0..4 {
                    b_mat[0][2 * a] = dn_dx[a];
                    b_mat[1][2 * a + 1] = dn_dy[a];
                    b_mat[2][2 * a] = dn_dy[a];
                    b_mat[2][2 * a + 1] = dn_dx[a];
                }

                let factor = det_j.abs() * thickness;
                for row in 0..8 {
                    for col in 0..8 {
                        let mut val = 0.0;
                        for alpha in 0..3 {
                            for beta in 0..3 {
                                val += b_mat[alpha][row]
                                    * constitutive[alpha][beta]
                                    * b_mat[beta][col];
                            }
                        }
                        stiffness[row][col] += val * factor;
                    }
                }
            }
        }

        Ok(stiffness)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 4 {
            return Err(ElementError::InvalidNodeCount {
                expected: 4,
                actual: node_coords.len(),
            });
        }

        let g_pts = [-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()];
        let mut mass = vec![vec![0.0; 8]; 8];

        if lumped {
            let mut total_mass = 0.0;
            for &xi in &g_pts {
                for &eta in &g_pts {
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
                    let mut j11 = 0.0;
                    let mut j12 = 0.0;
                    let mut j21 = 0.0;
                    let mut j22 = 0.0;
                    for a in 0..4 {
                        j11 += dn_dxi[a] * node_coords[a].coords[0];
                        j12 += dn_dxi[a] * node_coords[a].coords[1];
                        j21 += dn_deta[a] * node_coords[a].coords[0];
                        j22 += dn_deta[a] * node_coords[a].coords[1];
                    }
                    let det_j = j11 * j22 - j12 * j21;
                    total_mass += density * self.thickness * det_j.abs();
                }
            }
            let nodal_mass = total_mass / 4.0;
            for i in 0..4 {
                mass[2 * i][2 * i] = nodal_mass;
                mass[2 * i + 1][2 * i + 1] = nodal_mass;
            }
        } else {
            for &xi in &g_pts {
                for &eta in &g_pts {
                    let n = [
                        0.25 * (1.0 - xi) * (1.0 - eta),
                        0.25 * (1.0 + xi) * (1.0 - eta),
                        0.25 * (1.0 + xi) * (1.0 + eta),
                        0.25 * (1.0 - xi) * (1.0 + eta),
                    ];
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
                    let mut j11 = 0.0;
                    let mut j12 = 0.0;
                    let mut j21 = 0.0;
                    let mut j22 = 0.0;
                    for a in 0..4 {
                        j11 += dn_dxi[a] * node_coords[a].coords[0];
                        j12 += dn_dxi[a] * node_coords[a].coords[1];
                        j21 += dn_deta[a] * node_coords[a].coords[0];
                        j22 += dn_deta[a] * node_coords[a].coords[1];
                    }
                    let det_j = j11 * j22 - j12 * j21;
                    let factor = density * self.thickness * det_j.abs();
                    for i in 0..4 {
                        for j in 0..4 {
                            let term = n[i] * n[j] * factor;
                            mass[2 * i][2 * j] += term;
                            mass[2 * i + 1][2 * j + 1] += term;
                        }
                    }
                }
            }
        }

        Ok(mass)
    }
}

pub struct ElasticityTri6<'a> {
    pub nodes: &'a [NodeId; 6],
    pub thickness: f64,
}

impl<'a> Element for ElasticityTri6<'a> {
    fn spatial_dimension(&self) -> usize {
        2
    }
    fn node_count(&self) -> usize {
        6
    }
    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }
    fn active_fields(&self) -> Vec<String> {
        vec!["ux".to_string(), "uy".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let young_modulus = *properties
            .get("young_modulus")
            .ok_or_else(|| ElementError::MissingProperty("young_modulus".to_string()))?;
        let poisson_ratio = *properties
            .get("poisson_ratio")
            .ok_or_else(|| ElementError::MissingProperty("poisson_ratio".to_string()))?;
        let thickness = *properties
            .get("thickness")
            .ok_or_else(|| ElementError::MissingProperty("thickness".to_string()))?;
        let model_val = *properties.get("model").unwrap_or(&0.0);
        let model = if model_val == 1.0 {
            ElasticityModel::PlaneStrain
        } else {
            ElasticityModel::PlaneStress
        };

        if node_coords.len() != 6 {
            return Err(ElementError::InvalidNodeCount {
                expected: 6,
                actual: node_coords.len(),
            });
        }

        let constitutive = constitutive_matrix(ElasticityMaterial {
            young_modulus,
            poisson_ratio,
            model,
        });

        let pts = [
            (2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0),
            (1.0 / 6.0, 1.0 / 6.0),
        ];
        let w = 1.0 / 6.0;

        let mut stiffness = vec![vec![0.0; 12]; 12];

        for &(l1, l2) in &pts {
            let _l3 = 1.0 - l1 - l2;

            let dn_dl1 = [
                4.0 * l1 - 1.0,
                0.0,
                4.0 * l1 + 4.0 * l2 - 3.0,
                4.0 * l2,
                -4.0 * l2,
                4.0 * (1.0 - 2.0 * l1 - l2),
            ];
            let dn_dl2 = [
                0.0,
                4.0 * l2 - 1.0,
                4.0 * l1 + 4.0 * l2 - 3.0,
                4.0 * l1,
                4.0 * (1.0 - l1 - 2.0 * l2),
                -4.0 * l1,
            ];

            let mut j11 = 0.0;
            let mut j12 = 0.0;
            let mut j21 = 0.0;
            let mut j22 = 0.0;
            for a in 0..6 {
                j11 += dn_dl1[a] * node_coords[a].coords[0];
                j12 += dn_dl1[a] * node_coords[a].coords[1];
                j21 += dn_dl2[a] * node_coords[a].coords[0];
                j22 += dn_dl2[a] * node_coords[a].coords[1];
            }

            let det_j = j11 * j22 - j12 * j21;
            if det_j.abs() <= f64::EPSILON {
                return Err(ElementError::DegenerateGeometry);
            }

            let inv_j11 = j22 / det_j;
            let inv_j12 = -j12 / det_j;
            let inv_j21 = -j21 / det_j;
            let inv_j22 = j11 / det_j;

            let mut dn_dx = [0.0; 6];
            let mut dn_dy = [0.0; 6];
            for a in 0..6 {
                dn_dx[a] = inv_j11 * dn_dl1[a] + inv_j12 * dn_dl2[a];
                dn_dy[a] = inv_j21 * dn_dl1[a] + inv_j22 * dn_dl2[a];
            }

            let mut b_mat = [[0.0; 12]; 3];
            for a in 0..6 {
                b_mat[0][2 * a] = dn_dx[a];
                b_mat[1][2 * a + 1] = dn_dy[a];
                b_mat[2][2 * a] = dn_dy[a];
                b_mat[2][2 * a + 1] = dn_dx[a];
            }

            let factor = det_j.abs() * w * thickness;
            for row in 0..12 {
                for col in 0..12 {
                    let mut val = 0.0;
                    for alpha in 0..3 {
                        for beta in 0..3 {
                            val += b_mat[alpha][row] * constitutive[alpha][beta] * b_mat[beta][col];
                        }
                    }
                    stiffness[row][col] += val * factor;
                }
            }
        }

        Ok(stiffness)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 6 {
            return Err(ElementError::InvalidNodeCount {
                expected: 6,
                actual: node_coords.len(),
            });
        }

        let pts = [
            (2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0),
            (1.0 / 6.0, 1.0 / 6.0),
        ];
        let w = 1.0 / 6.0;

        let mut mass = vec![vec![0.0; 12]; 12];

        if lumped {
            let mut total_mass = 0.0;
            for &(l1, l2) in &pts {
                let dn_dl1 = [
                    4.0 * l1 - 1.0,
                    0.0,
                    4.0 * l1 + 4.0 * l2 - 3.0,
                    4.0 * l2,
                    -4.0 * l2,
                    4.0 * (1.0 - 2.0 * l1 - l2),
                ];
                let dn_dl2 = [
                    0.0,
                    4.0 * l2 - 1.0,
                    4.0 * l1 + 4.0 * l2 - 3.0,
                    4.0 * l1,
                    4.0 * (1.0 - l1 - 2.0 * l2),
                    -4.0 * l1,
                ];
                let mut j11 = 0.0;
                let mut j12 = 0.0;
                let mut j21 = 0.0;
                let mut j22 = 0.0;
                for a in 0..6 {
                    j11 += dn_dl1[a] * node_coords[a].coords[0];
                    j12 += dn_dl1[a] * node_coords[a].coords[1];
                    j21 += dn_dl2[a] * node_coords[a].coords[0];
                    j22 += dn_dl2[a] * node_coords[a].coords[1];
                }
                let det_j = j11 * j22 - j12 * j21;
                total_mass += density * self.thickness * det_j.abs() * w;
            }
            let nodal_mass = total_mass / 6.0;
            for i in 0..6 {
                mass[2 * i][2 * i] = nodal_mass;
                mass[2 * i + 1][2 * i + 1] = nodal_mass;
            }
        } else {
            for &(l1, l2) in &pts {
                let l3 = 1.0 - l1 - l2;
                let n = [
                    l1 * (2.0 * l1 - 1.0),
                    l2 * (2.0 * l2 - 1.0),
                    l3 * (2.0 * l3 - 1.0),
                    4.0 * l1 * l2,
                    4.0 * l2 * l3,
                    4.0 * l3 * l1,
                ];
                let dn_dl1 = [
                    4.0 * l1 - 1.0,
                    0.0,
                    4.0 * l1 + 4.0 * l2 - 3.0,
                    4.0 * l2,
                    -4.0 * l2,
                    4.0 * (1.0 - 2.0 * l1 - l2),
                ];
                let dn_dl2 = [
                    0.0,
                    4.0 * l2 - 1.0,
                    4.0 * l1 + 4.0 * l2 - 3.0,
                    4.0 * l1,
                    4.0 * (1.0 - l1 - 2.0 * l2),
                    -4.0 * l1,
                ];
                let mut j11 = 0.0;
                let mut j12 = 0.0;
                let mut j21 = 0.0;
                let mut j22 = 0.0;
                for a in 0..6 {
                    j11 += dn_dl1[a] * node_coords[a].coords[0];
                    j12 += dn_dl1[a] * node_coords[a].coords[1];
                    j21 += dn_dl2[a] * node_coords[a].coords[0];
                    j22 += dn_dl2[a] * node_coords[a].coords[1];
                }
                let det_j = j11 * j22 - j12 * j21;
                let factor = density * self.thickness * det_j.abs() * w;
                for i in 0..6 {
                    for j in 0..6 {
                        let term = n[i] * n[j] * factor;
                        mass[2 * i][2 * j] += term;
                        mass[2 * i + 1][2 * j + 1] += term;
                    }
                }
            }
        }

        Ok(mass)
    }
}

pub struct ElasticityTet4<'a> {
    pub nodes: &'a [NodeId; 4],
}

impl<'a> Element for ElasticityTet4<'a> {
    fn spatial_dimension(&self) -> usize {
        3
    }
    fn node_count(&self) -> usize {
        4
    }
    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }
    fn active_fields(&self) -> Vec<String> {
        vec!["ux".to_string(), "uy".to_string(), "uz".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let young_modulus = *properties
            .get("young_modulus")
            .ok_or_else(|| ElementError::MissingProperty("young_modulus".to_string()))?;
        let poisson_ratio = *properties
            .get("poisson_ratio")
            .ok_or_else(|| ElementError::MissingProperty("poisson_ratio".to_string()))?;

        if node_coords.len() != 4 {
            return Err(ElementError::InvalidNodeCount {
                expected: 4,
                actual: node_coords.len(),
            });
        }

        let [a, b, c, d] = [
            node_coords[0],
            node_coords[1],
            node_coords[2],
            node_coords[3],
        ];
        let jacobian = [
            [
                b.coords[0] - a.coords[0],
                c.coords[0] - a.coords[0],
                d.coords[0] - a.coords[0],
            ],
            [
                b.coords[1] - a.coords[1],
                c.coords[1] - a.coords[1],
                d.coords[1] - a.coords[1],
            ],
            [
                b.coords[2] - a.coords[2],
                c.coords[2] - a.coords[2],
                d.coords[2] - a.coords[2],
            ],
        ];
        let determinant = determinant_3(jacobian);
        let volume = determinant.abs() / 6.0;
        if volume <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }
        let inverse = inverse_3(jacobian, determinant);
        let gradients = [
            [
                -(inverse[0][0] + inverse[1][0] + inverse[2][0]),
                -(inverse[0][1] + inverse[1][1] + inverse[2][1]),
                -(inverse[0][2] + inverse[1][2] + inverse[2][2]),
            ],
            inverse[0],
            inverse[1],
            inverse[2],
        ];

        let constitutive = constitutive_matrix_3d(ElasticityMaterial3D {
            young_modulus,
            poisson_ratio,
        });
        let strain_displacement = strain_displacement_matrix_3d(gradients);

        let mut stiffness = vec![vec![0.0; 12]; 12];
        for row in 0..12 {
            for col in 0..12 {
                let mut value = 0.0;
                for alpha in 0..6 {
                    for beta in 0..6 {
                        value += strain_displacement[alpha][row]
                            * constitutive[alpha][beta]
                            * strain_displacement[beta][col];
                    }
                }
                stiffness[row][col] = volume * value;
            }
        }
        Ok(stiffness)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 4 {
            return Err(ElementError::InvalidNodeCount {
                expected: 4,
                actual: node_coords.len(),
            });
        }
        let [a, b, c, d] = [
            node_coords[0],
            node_coords[1],
            node_coords[2],
            node_coords[3],
        ];
        let jacobian = [
            [
                b.coords[0] - a.coords[0],
                c.coords[0] - a.coords[0],
                d.coords[0] - a.coords[0],
            ],
            [
                b.coords[1] - a.coords[1],
                c.coords[1] - a.coords[1],
                d.coords[1] - a.coords[1],
            ],
            [
                b.coords[2] - a.coords[2],
                c.coords[2] - a.coords[2],
                d.coords[2] - a.coords[2],
            ],
        ];
        let determinant = determinant_3(jacobian);
        let volume = determinant.abs() / 6.0;
        if volume <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }
        let total_mass = density * volume;
        let mut mass = vec![vec![0.0; 12]; 12];
        if lumped {
            let nodal_mass = total_mass / 4.0;
            for i in 0..4 {
                mass[3 * i][3 * i] = nodal_mass;
                mass[3 * i + 1][3 * i + 1] = nodal_mass;
                mass[3 * i + 2][3 * i + 2] = nodal_mass;
            }
        } else {
            let val_diag = total_mass / 10.0;
            let val_off = total_mass / 20.0;
            for i in 0..4 {
                for j in 0..4 {
                    let factor = if i == j { val_diag } else { val_off };
                    mass[3 * i][3 * j] = factor;
                    mass[3 * i + 1][3 * j + 1] = factor;
                    mass[3 * i + 2][3 * j + 2] = factor;
                }
            }
        }
        Ok(mass)
    }
}

pub struct ElasticityHex8<'a> {
    pub nodes: &'a [NodeId; 8],
}

impl<'a> Element for ElasticityHex8<'a> {
    fn spatial_dimension(&self) -> usize {
        3
    }
    fn node_count(&self) -> usize {
        8
    }
    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }
    fn active_fields(&self) -> Vec<String> {
        vec!["ux".to_string(), "uy".to_string(), "uz".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let young_modulus = *properties
            .get("young_modulus")
            .ok_or_else(|| ElementError::MissingProperty("young_modulus".to_string()))?;
        let poisson_ratio = *properties
            .get("poisson_ratio")
            .ok_or_else(|| ElementError::MissingProperty("poisson_ratio".to_string()))?;

        if node_coords.len() != 8 {
            return Err(ElementError::InvalidNodeCount {
                expected: 8,
                actual: node_coords.len(),
            });
        }

        let constitutive = constitutive_matrix_3d(ElasticityMaterial3D {
            young_modulus,
            poisson_ratio,
        });

        let g_pts = [-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()];
        let mut stiffness = vec![vec![0.0; 24]; 24];

        for &xi in &g_pts {
            for &eta in &g_pts {
                for &zeta in &g_pts {
                    let dn_dxi = [
                        -0.125 * (1.0 - eta) * (1.0 - zeta),
                        0.125 * (1.0 - eta) * (1.0 - zeta),
                        0.125 * (1.0 + eta) * (1.0 - zeta),
                        -0.125 * (1.0 + eta) * (1.0 - zeta),
                        -0.125 * (1.0 - eta) * (1.0 + zeta),
                        0.125 * (1.0 - eta) * (1.0 + zeta),
                        0.125 * (1.0 + eta) * (1.0 + zeta),
                        -0.125 * (1.0 + eta) * (1.0 + zeta),
                    ];
                    let dn_deta = [
                        -0.125 * (1.0 - xi) * (1.0 - zeta),
                        -0.125 * (1.0 + xi) * (1.0 - zeta),
                        0.125 * (1.0 + xi) * (1.0 - zeta),
                        0.125 * (1.0 - xi) * (1.0 - zeta),
                        -0.125 * (1.0 - xi) * (1.0 + zeta),
                        -0.125 * (1.0 + xi) * (1.0 + zeta),
                        0.125 * (1.0 + xi) * (1.0 + zeta),
                        0.125 * (1.0 - xi) * (1.0 + zeta),
                    ];
                    let dn_dzeta = [
                        -0.125 * (1.0 - xi) * (1.0 - eta),
                        -0.125 * (1.0 + xi) * (1.0 - eta),
                        -0.125 * (1.0 + xi) * (1.0 + eta),
                        -0.125 * (1.0 - xi) * (1.0 + eta),
                        0.125 * (1.0 - xi) * (1.0 - eta),
                        0.125 * (1.0 + xi) * (1.0 - eta),
                        0.125 * (1.0 + xi) * (1.0 + eta),
                        0.125 * (1.0 - xi) * (1.0 + eta),
                    ];

                    let mut j = [[0.0; 3]; 3];
                    for a in 0..8 {
                        j[0][0] += dn_dxi[a] * node_coords[a].coords[0];
                        j[0][1] += dn_dxi[a] * node_coords[a].coords[1];
                        j[0][2] += dn_dxi[a] * node_coords[a].coords[2];

                        j[1][0] += dn_deta[a] * node_coords[a].coords[0];
                        j[1][1] += dn_deta[a] * node_coords[a].coords[1];
                        j[1][2] += dn_deta[a] * node_coords[a].coords[2];

                        j[2][0] += dn_dzeta[a] * node_coords[a].coords[0];
                        j[2][1] += dn_dzeta[a] * node_coords[a].coords[1];
                        j[2][2] += dn_dzeta[a] * node_coords[a].coords[2];
                    }

                    let det_j = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
                        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
                        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);

                    if det_j.abs() <= f64::EPSILON {
                        return Err(ElementError::DegenerateGeometry);
                    }

                    let inv_j = [
                        [
                            (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / det_j,
                            -(j[0][1] * j[2][2] - j[0][2] * j[2][1]) / det_j,
                            (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det_j,
                        ],
                        [
                            -(j[1][0] * j[2][2] - j[1][2] * j[2][0]) / det_j,
                            (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det_j,
                            -(j[0][0] * j[1][2] - j[0][2] * j[1][0]) / det_j,
                        ],
                        [
                            (j[1][0] * j[2][1] - j[1][1] * j[2][0]) / det_j,
                            -(j[0][0] * j[2][1] - j[0][1] * j[2][0]) / det_j,
                            (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / det_j,
                        ],
                    ];

                    let mut dn_dx = [0.0; 8];
                    let mut dn_dy = [0.0; 8];
                    let mut dn_dz = [0.0; 8];
                    for a in 0..8 {
                        dn_dx[a] = inv_j[0][0] * dn_dxi[a]
                            + inv_j[0][1] * dn_deta[a]
                            + inv_j[0][2] * dn_dzeta[a];
                        dn_dy[a] = inv_j[1][0] * dn_dxi[a]
                            + inv_j[1][1] * dn_deta[a]
                            + inv_j[1][2] * dn_dzeta[a];
                        dn_dz[a] = inv_j[2][0] * dn_dxi[a]
                            + inv_j[2][1] * dn_deta[a]
                            + inv_j[2][2] * dn_dzeta[a];
                    }

                    let mut b_mat = [[0.0; 24]; 6];
                    for a in 0..8 {
                        let col = 3 * a;
                        b_mat[0][col] = dn_dx[a];
                        b_mat[1][col + 1] = dn_dy[a];
                        b_mat[2][col + 2] = dn_dz[a];

                        b_mat[3][col] = dn_dy[a];
                        b_mat[3][col + 1] = dn_dx[a];

                        b_mat[4][col + 1] = dn_dz[a];
                        b_mat[4][col + 2] = dn_dy[a];

                        b_mat[5][col] = dn_dz[a];
                        b_mat[5][col + 2] = dn_dx[a];
                    }

                    let factor = det_j.abs();
                    for row in 0..24 {
                        for col in 0..24 {
                            let mut val = 0.0;
                            for alpha in 0..6 {
                                for beta in 0..6 {
                                    val += b_mat[alpha][row]
                                        * constitutive[alpha][beta]
                                        * b_mat[beta][col];
                                }
                            }
                            stiffness[row][col] += val * factor;
                        }
                    }
                }
            }
        }

        Ok(stiffness)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 8 {
            return Err(ElementError::InvalidNodeCount {
                expected: 8,
                actual: node_coords.len(),
            });
        }

        let g_pts = [-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()];
        let mut mass = vec![vec![0.0; 24]; 24];

        if lumped {
            let mut total_mass = 0.0;
            for &xi in &g_pts {
                for &eta in &g_pts {
                    for &zeta in &g_pts {
                        let dn_dxi = [
                            -0.125 * (1.0 - eta) * (1.0 - zeta),
                            0.125 * (1.0 - eta) * (1.0 - zeta),
                            0.125 * (1.0 + eta) * (1.0 - zeta),
                            -0.125 * (1.0 + eta) * (1.0 - zeta),
                            -0.125 * (1.0 - eta) * (1.0 + zeta),
                            0.125 * (1.0 - eta) * (1.0 + zeta),
                            0.125 * (1.0 + eta) * (1.0 + zeta),
                            -0.125 * (1.0 + eta) * (1.0 + zeta),
                        ];
                        let dn_deta = [
                            -0.125 * (1.0 - xi) * (1.0 - zeta),
                            -0.125 * (1.0 + xi) * (1.0 - zeta),
                            0.125 * (1.0 + xi) * (1.0 - zeta),
                            0.125 * (1.0 - xi) * (1.0 - zeta),
                            -0.125 * (1.0 - xi) * (1.0 + zeta),
                            -0.125 * (1.0 + xi) * (1.0 + zeta),
                            0.125 * (1.0 + xi) * (1.0 + zeta),
                            0.125 * (1.0 - xi) * (1.0 + zeta),
                        ];
                        let dn_dzeta = [
                            -0.125 * (1.0 - xi) * (1.0 - eta),
                            -0.125 * (1.0 + xi) * (1.0 - eta),
                            -0.125 * (1.0 + xi) * (1.0 + eta),
                            -0.125 * (1.0 - xi) * (1.0 + eta),
                            0.125 * (1.0 - xi) * (1.0 - eta),
                            0.125 * (1.0 + xi) * (1.0 - eta),
                            0.125 * (1.0 + xi) * (1.0 + eta),
                            0.125 * (1.0 - xi) * (1.0 + eta),
                        ];
                        let mut j = [[0.0; 3]; 3];
                        for a in 0..8 {
                            j[0][0] += dn_dxi[a] * node_coords[a].coords[0];
                            j[0][1] += dn_dxi[a] * node_coords[a].coords[1];
                            j[0][2] += dn_dxi[a] * node_coords[a].coords[2];

                            j[1][0] += dn_deta[a] * node_coords[a].coords[0];
                            j[1][1] += dn_deta[a] * node_coords[a].coords[1];
                            j[1][2] += dn_deta[a] * node_coords[a].coords[2];

                            j[2][0] += dn_dzeta[a] * node_coords[a].coords[0];
                            j[2][1] += dn_dzeta[a] * node_coords[a].coords[1];
                            j[2][2] += dn_dzeta[a] * node_coords[a].coords[2];
                        }
                        let det_j = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
                            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
                            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
                        total_mass += density * det_j.abs();
                    }
                }
            }
            let nodal_mass = total_mass / 8.0;
            for i in 0..8 {
                mass[3 * i][3 * i] = nodal_mass;
                mass[3 * i + 1][3 * i + 1] = nodal_mass;
                mass[3 * i + 2][3 * i + 2] = nodal_mass;
            }
        } else {
            for &xi in &g_pts {
                for &eta in &g_pts {
                    for &zeta in &g_pts {
                        let n = [
                            0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta),
                            0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta),
                            0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta),
                            0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta),
                            0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zeta),
                            0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zeta),
                            0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zeta),
                            0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zeta),
                        ];
                        let dn_dxi = [
                            -0.125 * (1.0 - eta) * (1.0 - zeta),
                            0.125 * (1.0 - eta) * (1.0 - zeta),
                            0.125 * (1.0 + eta) * (1.0 - zeta),
                            -0.125 * (1.0 + eta) * (1.0 - zeta),
                            -0.125 * (1.0 - eta) * (1.0 + zeta),
                            0.125 * (1.0 - eta) * (1.0 + zeta),
                            0.125 * (1.0 + eta) * (1.0 + zeta),
                            -0.125 * (1.0 + eta) * (1.0 + zeta),
                        ];
                        let dn_deta = [
                            -0.125 * (1.0 - xi) * (1.0 - zeta),
                            -0.125 * (1.0 + xi) * (1.0 - zeta),
                            0.125 * (1.0 + xi) * (1.0 - zeta),
                            0.125 * (1.0 - xi) * (1.0 - zeta),
                            -0.125 * (1.0 - xi) * (1.0 + zeta),
                            -0.125 * (1.0 + xi) * (1.0 + zeta),
                            0.125 * (1.0 + xi) * (1.0 + zeta),
                            0.125 * (1.0 - xi) * (1.0 + zeta),
                        ];
                        let dn_dzeta = [
                            -0.125 * (1.0 - xi) * (1.0 - eta),
                            -0.125 * (1.0 + xi) * (1.0 - eta),
                            -0.125 * (1.0 + xi) * (1.0 + eta),
                            -0.125 * (1.0 - xi) * (1.0 + eta),
                            0.125 * (1.0 - xi) * (1.0 - eta),
                            0.125 * (1.0 + xi) * (1.0 - eta),
                            0.125 * (1.0 + xi) * (1.0 + eta),
                            0.125 * (1.0 - xi) * (1.0 + eta),
                        ];
                        let mut j = [[0.0; 3]; 3];
                        for a in 0..8 {
                            j[0][0] += dn_dxi[a] * node_coords[a].coords[0];
                            j[0][1] += dn_dxi[a] * node_coords[a].coords[1];
                            j[0][2] += dn_dxi[a] * node_coords[a].coords[2];

                            j[1][0] += dn_deta[a] * node_coords[a].coords[0];
                            j[1][1] += dn_deta[a] * node_coords[a].coords[1];
                            j[1][2] += dn_deta[a] * node_coords[a].coords[2];

                            j[2][0] += dn_dzeta[a] * node_coords[a].coords[0];
                            j[2][1] += dn_dzeta[a] * node_coords[a].coords[1];
                            j[2][2] += dn_dzeta[a] * node_coords[a].coords[2];
                        }
                        let det_j = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
                            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
                            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
                        let factor = density * det_j.abs();
                        for i in 0..8 {
                            for j in 0..8 {
                                let term = n[i] * n[j] * factor;
                                mass[3 * i][3 * j] += term;
                                mass[3 * i + 1][3 * j + 1] += term;
                                mass[3 * i + 2][3 * j + 2] += term;
                            }
                        }
                    }
                }
            }
        }

        Ok(mass)
    }
}

fn validate_material(material: ElasticityMaterial) -> Result<(), ElasticityError> {
    if !material.young_modulus.is_finite() || material.young_modulus <= 0.0 {
        return Err(ElasticityError::InvalidYoungModulus(material.young_modulus));
    }
    if !material.poisson_ratio.is_finite()
        || material.poisson_ratio <= -1.0
        || material.poisson_ratio >= 0.5
    {
        return Err(ElasticityError::InvalidPoissonRatio(material.poisson_ratio));
    }
    Ok(())
}

fn validate_material_3d(material: ElasticityMaterial3D) -> Result<(), ElasticityError> {
    if !material.young_modulus.is_finite() || material.young_modulus <= 0.0 {
        return Err(ElasticityError::InvalidYoungModulus(material.young_modulus));
    }
    if !material.poisson_ratio.is_finite()
        || material.poisson_ratio <= -1.0
        || material.poisson_ratio >= 0.5
    {
        return Err(ElasticityError::InvalidPoissonRatio(material.poisson_ratio));
    }
    Ok(())
}

fn validate_thickness(thickness: f64) -> Result<(), ElasticityError> {
    if thickness.is_finite() && thickness > 0.0 {
        Ok(())
    } else {
        Err(ElasticityError::InvalidThickness(thickness))
    }
}

fn validate_density(density: f64) -> Result<(), ElasticityError> {
    if density.is_finite() && density > 0.0 {
        Ok(())
    } else {
        Err(ElasticityError::InvalidDensity(density))
    }
}

fn validate_initial_state_lengths(
    node_count: usize,
    displacement_len: usize,
    velocity_len: usize,
) -> Result<(), ElasticityError> {
    if displacement_len != node_count {
        return Err(ElasticityError::InitialStateLengthMismatch {
            node_count,
            initial_len: displacement_len,
        });
    }
    if velocity_len != node_count {
        return Err(ElasticityError::InitialStateLengthMismatch {
            node_count,
            initial_len: velocity_len,
        });
    }
    Ok(())
}

fn validate_constraints(
    node_count: usize,
    constraints: &[DisplacementConstraint],
) -> Result<BTreeMap<usize, f64>, ElasticityError> {
    let mut constrained = BTreeMap::new();
    for constraint in constraints {
        if constraint.node >= node_count {
            return Err(ElasticityError::ConstraintNodeOutOfBounds {
                node_id: constraint.node,
                node_count,
            });
        }
        let dof = dof_index(constraint.node, constraint.component);
        if constrained.insert(dof, constraint.value).is_some() {
            return Err(ElasticityError::DuplicateConstraint {
                node_id: constraint.node,
                component: constraint.component,
            });
        }
    }
    Ok(constrained)
}

fn validate_constraints_3d(
    node_count: usize,
    constraints: &[DisplacementConstraint3D],
) -> Result<BTreeMap<usize, f64>, ElasticityError> {
    let mut constrained = BTreeMap::new();
    for constraint in constraints {
        if constraint.node >= node_count {
            return Err(ElasticityError::ConstraintNodeOutOfBounds {
                node_id: constraint.node,
                node_count,
            });
        }
        let dof = dof_index_3d(constraint.node, constraint.component);
        if constrained.insert(dof, constraint.value).is_some() {
            return Err(ElasticityError::DuplicateConstraint3D {
                node_id: constraint.node,
                component: constraint.component,
            });
        }
    }
    Ok(constrained)
}

fn apply_constraints(
    matrix: CsMat<f64>,
    rhs: Vec<f64>,
    constraints: &BTreeMap<usize, f64>,
) -> (CsMat<f64>, Vec<f64>) {
    if constraints.is_empty() {
        return (matrix, rhs);
    }

    let mut adjusted_rhs = rhs;
    let mut constrained_triplets = TriMat::new((matrix.rows(), matrix.cols()));

    for (row_index, row) in matrix.outer_iterator().enumerate() {
        if constraints.contains_key(&row_index) {
            continue;
        }

        for (col_index, value) in row.iter() {
            if let Some(boundary_value) = constraints.get(&col_index) {
                adjusted_rhs[row_index] -= *value * boundary_value;
            } else {
                constrained_triplets.add_triplet(row_index, col_index, *value);
            }
        }
    }

    for (&dof, &value) in constraints {
        adjusted_rhs[dof] = value;
        constrained_triplets.add_triplet(dof, dof, 1.0);
    }

    (constrained_triplets.to_csr(), adjusted_rhs)
}

fn active_dofs(dof_count: usize, constraints: &BTreeMap<usize, f64>) -> Vec<usize> {
    (0..dof_count)
        .filter(|dof| !constraints.contains_key(dof))
        .collect()
}

fn active_dof_map(dof_count: usize, active_dofs: &[usize]) -> Vec<Option<usize>> {
    let mut map = vec![None; dof_count];
    for (active_index, &dof) in active_dofs.iter().enumerate() {
        map[dof] = Some(active_index);
    }
    map
}

fn reduce_matrix(
    matrix: &CsMat<f64>,
    active_dofs: &[usize],
    active_map: &[Option<usize>],
) -> CsMat<f64> {
    let mut triplets = TriMat::new((active_dofs.len(), active_dofs.len()));
    for (row_index, row) in matrix.outer_iterator().enumerate() {
        let Some(active_row) = active_map[row_index] else {
            continue;
        };
        for (col_index, value) in row.iter() {
            if let Some(active_col) = active_map[col_index] {
                triplets.add_triplet(active_row, active_col, *value);
            }
        }
    }
    triplets.to_csr()
}

fn reduce_vector(values: &[f64], active_dofs: &[usize]) -> Vec<f64> {
    active_dofs.iter().map(|&dof| values[dof]).collect()
}

fn reduce_source(
    full_force: &[f64],
    stiffness: &CsMat<f64>,
    active_dofs: &[usize],
    constraints: &BTreeMap<usize, f64>,
) -> Vec<f64> {
    let mut reduced = Vec::with_capacity(active_dofs.len());
    for &dof in active_dofs {
        let mut value = full_force[dof];
        if let Some(row) = stiffness.outer_view(dof) {
            for (col_index, stiffness_value) in row.iter() {
                if let Some(boundary_value) = constraints.get(&col_index) {
                    value -= stiffness_value * boundary_value;
                }
            }
        }
        reduced.push(value);
    }
    reduced
}

fn reconstruct_values(
    dof_count: usize,
    active_dofs: &[usize],
    constraints: &BTreeMap<usize, f64>,
    active_values: &[f64],
) -> Vec<f64> {
    let mut values = vec![0.0; dof_count];
    for (&dof, &value) in constraints {
        values[dof] = value;
    }
    for (&dof, &value) in active_dofs.iter().zip(active_values) {
        values[dof] = value;
    }
    values
}

fn assemble_lumped_elasticity_mass(mesh: &Mesh, density: f64, thickness: f64) -> CsMat<f64> {
    let mut mass = vec![0.0; mesh.node_count() * 2];
    for cell in mesh.cells() {
        let node_coords: Vec<Point3> = cell
            .nodes
            .iter()
            .map(|&id| Point3::new([mesh.points()[id].x, mesh.points()[id].y, 0.0]))
            .collect();
        match cell.kind {
            ElementKind::Tri3 => {
                let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2]];
                let el = ElasticityTri3 {
                    nodes: &nodes,
                    thickness,
                };
                let local_m = el.local_mass(&node_coords, density, true).unwrap();
                for i in 0..3 {
                    let global_node = cell.nodes[i];
                    mass[global_node * 2] += local_m[2 * i][2 * i];
                    mass[global_node * 2 + 1] += local_m[2 * i + 1][2 * i + 1];
                }
            }
            ElementKind::Quad4 => {
                let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
                let el = ElasticityQuad4 {
                    nodes: &nodes,
                    thickness,
                };
                let local_m = el.local_mass(&node_coords, density, true).unwrap();
                for i in 0..4 {
                    let global_node = cell.nodes[i];
                    mass[global_node * 2] += local_m[2 * i][2 * i];
                    mass[global_node * 2 + 1] += local_m[2 * i + 1][2 * i + 1];
                }
            }
            ElementKind::Tri6 => {
                let nodes = [
                    cell.nodes[0],
                    cell.nodes[1],
                    cell.nodes[2],
                    cell.nodes[3],
                    cell.nodes[4],
                    cell.nodes[5],
                ];
                let el = ElasticityTri6 {
                    nodes: &nodes,
                    thickness,
                };
                let local_m = el.local_mass(&node_coords, density, true).unwrap();
                for i in 0..6 {
                    let global_node = cell.nodes[i];
                    mass[global_node * 2] += local_m[2 * i][2 * i];
                    mass[global_node * 2 + 1] += local_m[2 * i + 1][2 * i + 1];
                }
            }
            _ => panic!("Unsupported element kind in 2D mass assembly"),
        }
    }
    diagonal_matrix(mass)
}

fn assemble_lumped_elasticity_mass_3d(mesh: &MeshTopology<3>, density: f64) -> CsMat<f64> {
    let mut mass = vec![0.0; mesh.points().len() * 3];
    for cell in mesh.cells() {
        if cell.kind != ElementKind::Tet4 {
            continue;
        }
        let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
        let (volume, _, _) = tetrahedron_geometry(mesh, nodes);
        let nodal_mass = density * volume / 4.0;
        for node in nodes {
            mass[dof_index_3d(node, DisplacementComponent3D::X)] += nodal_mass;
            mass[dof_index_3d(node, DisplacementComponent3D::Y)] += nodal_mass;
            mass[dof_index_3d(node, DisplacementComponent3D::Z)] += nodal_mass;
        }
    }
    diagonal_matrix(mass)
}

fn diagonal_matrix(values: Vec<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::with_capacity((values.len(), values.len()), values.len());
    for (index, value) in values.into_iter().enumerate() {
        triplets.add_triplet(index, index, value);
    }
    triplets.to_csr()
}

fn transient_elasticity_sources<F>(
    mesh: &Mesh,
    problem: &TransientElasticityProblem<F>,
    stiffness: &CsMat<f64>,
    active_dofs: &[usize],
    constraints: &BTreeMap<usize, f64>,
    options: &NewmarkSolverOptions,
) -> Result<Vec<Vec<f64>>, ElasticityError>
where
    F: Fn(f64) -> Vec<NodalForce>,
{
    let mut sources = Vec::with_capacity(options.steps + 1);
    for step in 0..=options.steps {
        let time = step as f64 * options.time_step;
        let full_force = assemble_force_vector(mesh.node_count(), &(problem.forces)(time))?;
        sources.push(reduce_source(
            &full_force,
            stiffness,
            active_dofs,
            constraints,
        ));
    }
    Ok(sources)
}

fn transient_elasticity_sources_3d<F>(
    mesh: &MeshTopology<3>,
    problem: &TransientElasticityProblem3D<F>,
    stiffness: &CsMat<f64>,
    active_dofs: &[usize],
    constraints: &BTreeMap<usize, f64>,
    options: &NewmarkSolverOptions,
) -> Result<Vec<Vec<f64>>, ElasticityError>
where
    F: Fn(f64) -> Vec<NodalForce3D>,
{
    let mut sources = Vec::with_capacity(options.steps + 1);
    for step in 0..=options.steps {
        let time = step as f64 * options.time_step;
        let full_force = assemble_force_vector_3d(mesh.points().len(), &(problem.forces)(time))?;
        sources.push(reduce_source(
            &full_force,
            stiffness,
            active_dofs,
            constraints,
        ));
    }
    Ok(sources)
}

fn assemble_force_vector(
    node_count: usize,
    forces: &[NodalForce],
) -> Result<Vec<f64>, ElasticityError> {
    let mut rhs = vec![0.0; node_count * 2];
    for force in forces {
        if force.node >= node_count {
            return Err(ElasticityError::ForceNodeOutOfBounds {
                node_id: force.node,
                node_count,
            });
        }
        rhs[dof_index(force.node, DisplacementComponent::X)] += force.fx;
        rhs[dof_index(force.node, DisplacementComponent::Y)] += force.fy;
    }
    Ok(rhs)
}

fn assemble_force_vector_3d(
    node_count: usize,
    forces: &[NodalForce3D],
) -> Result<Vec<f64>, ElasticityError> {
    let mut rhs = vec![0.0; node_count * 3];
    for force in forces {
        if force.node >= node_count {
            return Err(ElasticityError::ForceNodeOutOfBounds {
                node_id: force.node,
                node_count,
            });
        }
        rhs[dof_index_3d(force.node, DisplacementComponent3D::X)] += force.fx;
        rhs[dof_index_3d(force.node, DisplacementComponent3D::Y)] += force.fy;
        rhs[dof_index_3d(force.node, DisplacementComponent3D::Z)] += force.fz;
    }
    Ok(rhs)
}

fn flatten_displacements_2d(values: &[[f64; 2]]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| [value[0], value[1]])
        .collect()
}

fn flatten_displacements_3d(values: &[[f64; 3]]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| [value[0], value[1], value[2]])
        .collect()
}

fn unflatten_displacements_2d(values: &[f64]) -> Vec<[f64; 2]> {
    values
        .chunks_exact(2)
        .map(|values| [values[0], values[1]])
        .collect()
}

fn unflatten_displacements_3d(values: &[f64]) -> Vec<[f64; 3]> {
    values
        .chunks_exact(3)
        .map(|values| [values[0], values[1], values[2]])
        .collect()
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

fn constitutive_matrix_3d(material: ElasticityMaterial3D) -> [[f64; 6]; 6] {
    let shear_modulus = material.young_modulus / (2.0 * (1.0 + material.poisson_ratio));
    let lambda = material.young_modulus * material.poisson_ratio
        / ((1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio));
    let normal = lambda + 2.0 * shear_modulus;

    [
        [normal, lambda, lambda, 0.0, 0.0, 0.0],
        [lambda, normal, lambda, 0.0, 0.0, 0.0],
        [lambda, lambda, normal, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, shear_modulus, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, shear_modulus, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, shear_modulus],
    ]
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

fn strain_displacement_matrix_3d(gradients: [[f64; 3]; 4]) -> [[f64; 12]; 6] {
    let mut b = [[0.0; 12]; 6];
    for (node, gradient) in gradients.iter().enumerate() {
        let base = node * 3;
        let [gx, gy, gz] = *gradient;
        b[0][base] = gx;
        b[1][base + 1] = gy;
        b[2][base + 2] = gz;
        b[3][base] = gy;
        b[3][base + 1] = gx;
        b[4][base + 1] = gz;
        b[4][base + 2] = gy;
        b[5][base] = gz;
        b[5][base + 2] = gx;
    }
    b
}

#[allow(dead_code)]
fn triangle_geometry(mesh: &Mesh, triangle: &Tri3) -> (f64, [[f64; 2]; 3]) {
    let [a, b, c] = triangle.nodes.map(|node| mesh.points()[node]);
    let twice_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    let area = 0.5 * twice_area.abs();
    let gradients = [
        [(b.y - c.y) / twice_area, (c.x - b.x) / twice_area],
        [(c.y - a.y) / twice_area, (a.x - c.x) / twice_area],
        [(a.y - b.y) / twice_area, (b.x - a.x) / twice_area],
    ];
    (area, gradients)
}

#[allow(dead_code)]
fn triangle_dofs(triangle: &Tri3) -> [usize; 6] {
    [
        dof_index(triangle.nodes[0], DisplacementComponent::X),
        dof_index(triangle.nodes[0], DisplacementComponent::Y),
        dof_index(triangle.nodes[1], DisplacementComponent::X),
        dof_index(triangle.nodes[1], DisplacementComponent::Y),
        dof_index(triangle.nodes[2], DisplacementComponent::X),
        dof_index(triangle.nodes[2], DisplacementComponent::Y),
    ]
}

fn dof_index(node: NodeId, component: DisplacementComponent) -> usize {
    node * 2 + component.offset()
}

#[allow(dead_code)]
fn tet4_dofs(nodes: [NodeId; 4]) -> [usize; 12] {
    [
        dof_index_3d(nodes[0], DisplacementComponent3D::X),
        dof_index_3d(nodes[0], DisplacementComponent3D::Y),
        dof_index_3d(nodes[0], DisplacementComponent3D::Z),
        dof_index_3d(nodes[1], DisplacementComponent3D::X),
        dof_index_3d(nodes[1], DisplacementComponent3D::Y),
        dof_index_3d(nodes[1], DisplacementComponent3D::Z),
        dof_index_3d(nodes[2], DisplacementComponent3D::X),
        dof_index_3d(nodes[2], DisplacementComponent3D::Y),
        dof_index_3d(nodes[2], DisplacementComponent3D::Z),
        dof_index_3d(nodes[3], DisplacementComponent3D::X),
        dof_index_3d(nodes[3], DisplacementComponent3D::Y),
        dof_index_3d(nodes[3], DisplacementComponent3D::Z),
    ]
}

fn dof_index_3d(node: NodeId, component: DisplacementComponent3D) -> usize {
    node * 3 + component.offset()
}

fn tetrahedron_geometry(
    mesh: &MeshTopology<3>,
    nodes: [NodeId; 4],
) -> (f64, [[f64; 3]; 4], Point3) {
    let [a, b, c, d] = nodes.map(|node| mesh.points()[node]);
    let jacobian = [
        [
            b.coords[0] - a.coords[0],
            c.coords[0] - a.coords[0],
            d.coords[0] - a.coords[0],
        ],
        [
            b.coords[1] - a.coords[1],
            c.coords[1] - a.coords[1],
            d.coords[1] - a.coords[1],
        ],
        [
            b.coords[2] - a.coords[2],
            c.coords[2] - a.coords[2],
            d.coords[2] - a.coords[2],
        ],
    ];
    let determinant = determinant_3(jacobian);
    let volume = determinant.abs() / 6.0;
    let inverse = inverse_3(jacobian, determinant);
    let gradients = [
        [
            -(inverse[0][0] + inverse[1][0] + inverse[2][0]),
            -(inverse[0][1] + inverse[1][1] + inverse[2][1]),
            -(inverse[0][2] + inverse[1][2] + inverse[2][2]),
        ],
        inverse[0],
        inverse[1],
        inverse[2],
    ];
    let centroid = Point3::new([
        (a.coords[0] + b.coords[0] + c.coords[0] + d.coords[0]) / 4.0,
        (a.coords[1] + b.coords[1] + c.coords[1] + d.coords[1]) / 4.0,
        (a.coords[2] + b.coords[2] + c.coords[2] + d.coords[2]) / 4.0,
    ]);

    (volume, gradients, centroid)
}

fn determinant_3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn inverse_3(matrix: [[f64; 3]; 3], determinant: f64) -> [[f64; 3]; 3] {
    let inv_det = 1.0 / determinant;
    [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det,
        ],
    ]
}

/// Recovers element-centroid stress and strain tensors for a 2D Tri3 mesh.
///
/// Uses the element B-matrix (strain-displacement) and material D-matrix
/// (constitutive) to compute `sigma = D * B * u` at each element centroid.
pub fn recover_stress_2d(
    mesh: &Mesh,
    material: &ElasticityMaterial,
    displacements: &[[f64; 2]],
) -> (
    Vec<StressTensor2D>,
    Vec<StrainTensor2D>,
    Vec<StressTensor2D>,
) {
    let d = constitutive_matrix(*material);
    let node_count = mesh.node_count();
    let cells = mesh.cells();
    let n_elem = cells.len();

    let mut element_stress = Vec::with_capacity(n_elem);
    let mut element_strain = Vec::with_capacity(n_elem);

    // Accumulators for nodal averaging.
    let mut nodal_stress_sum = vec![[0.0f64; 3]; node_count];
    let mut nodal_count = vec![0usize; node_count];

    for cell in cells {
        let (b_mat_rows, dof_indices) = match cell.kind {
            ElementKind::Tri3 => {
                let n0 = cell.nodes[0];
                let n1 = cell.nodes[1];
                let n2 = cell.nodes[2];
                let p0 = mesh.points()[n0];
                let p1 = mesh.points()[n1];
                let p2 = mesh.points()[n2];

                let twice_area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
                if twice_area.abs() <= f64::EPSILON {
                    (vec![vec![0.0; 6]; 3], cell.nodes.clone())
                } else {
                    let b1 = p1.y - p2.y;
                    let b2 = p2.y - p0.y;
                    let b3 = p0.y - p1.y;
                    let c1 = p2.x - p1.x;
                    let c2 = p0.x - p2.x;
                    let c3 = p1.x - p0.x;

                    let inv2a = 1.0 / twice_area;
                    let b_mat = vec![
                        vec![b1 * inv2a, 0.0, b2 * inv2a, 0.0, b3 * inv2a, 0.0],
                        vec![0.0, c1 * inv2a, 0.0, c2 * inv2a, 0.0, c3 * inv2a],
                        vec![
                            c1 * inv2a,
                            b1 * inv2a,
                            c2 * inv2a,
                            b2 * inv2a,
                            c3 * inv2a,
                            b3 * inv2a,
                        ],
                    ];
                    (b_mat, cell.nodes.clone())
                }
            }
            ElementKind::Quad4 => {
                let node_coords: Vec<Point3> = cell
                    .nodes
                    .iter()
                    .map(|&id| Point3::new([mesh.points()[id].x, mesh.points()[id].y, 0.0]))
                    .collect();
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

                let mut j11 = 0.0;
                let mut j12 = 0.0;
                let mut j21 = 0.0;
                let mut j22 = 0.0;
                for a in 0..4 {
                    j11 += dn_dxi[a] * node_coords[a].coords[0];
                    j12 += dn_dxi[a] * node_coords[a].coords[1];
                    j21 += dn_deta[a] * node_coords[a].coords[0];
                    j22 += dn_deta[a] * node_coords[a].coords[1];
                }

                let det_j = j11 * j22 - j12 * j21;
                if det_j.abs() <= f64::EPSILON {
                    (vec![vec![0.0; 8]; 3], cell.nodes.clone())
                } else {
                    let inv_j11 = j22 / det_j;
                    let inv_j12 = -j12 / det_j;
                    let inv_j21 = -j21 / det_j;
                    let inv_j22 = j11 / det_j;

                    let mut dn_dx = [0.0; 4];
                    let mut dn_dy = [0.0; 4];
                    for a in 0..4 {
                        dn_dx[a] = inv_j11 * dn_dxi[a] + inv_j12 * dn_deta[a];
                        dn_dy[a] = inv_j21 * dn_dxi[a] + inv_j22 * dn_deta[a];
                    }

                    let mut b_mat = vec![vec![0.0; 8]; 3];
                    for a in 0..4 {
                        b_mat[0][2 * a] = dn_dx[a];
                        b_mat[1][2 * a + 1] = dn_dy[a];
                        b_mat[2][2 * a] = dn_dy[a];
                        b_mat[2][2 * a + 1] = dn_dx[a];
                    }
                    (b_mat, cell.nodes.clone())
                }
            }
            ElementKind::Tri6 => {
                let node_coords: Vec<Point3> = cell
                    .nodes
                    .iter()
                    .map(|&id| Point3::new([mesh.points()[id].x, mesh.points()[id].y, 0.0]))
                    .collect();
                let l1 = 1.0 / 3.0;
                let l2 = 1.0 / 3.0;
                let dn_dl1 = [
                    4.0 * l1 - 1.0,
                    0.0,
                    4.0 * l1 + 4.0 * l2 - 3.0,
                    4.0 * l2,
                    -4.0 * l2,
                    4.0 * (1.0 - 2.0 * l1 - l2),
                ];
                let dn_dl2 = [
                    0.0,
                    4.0 * l2 - 1.0,
                    4.0 * l1 + 4.0 * l2 - 3.0,
                    4.0 * l1,
                    4.0 * (1.0 - l1 - 2.0 * l2),
                    -4.0 * l1,
                ];

                let mut j11 = 0.0;
                let mut j12 = 0.0;
                let mut j21 = 0.0;
                let mut j22 = 0.0;
                for a in 0..6 {
                    j11 += dn_dl1[a] * node_coords[a].coords[0];
                    j12 += dn_dl1[a] * node_coords[a].coords[1];
                    j21 += dn_dl2[a] * node_coords[a].coords[0];
                    j22 += dn_dl2[a] * node_coords[a].coords[1];
                }

                let det_j = j11 * j22 - j12 * j21;
                if det_j.abs() <= f64::EPSILON {
                    (vec![vec![0.0; 12]; 3], cell.nodes.clone())
                } else {
                    let inv_j11 = j22 / det_j;
                    let inv_j12 = -j12 / det_j;
                    let inv_j21 = -j21 / det_j;
                    let inv_j22 = j11 / det_j;

                    let mut dn_dx = [0.0; 6];
                    let mut dn_dy = [0.0; 6];
                    for a in 0..6 {
                        dn_dx[a] = inv_j11 * dn_dl1[a] + inv_j12 * dn_dl2[a];
                        dn_dy[a] = inv_j21 * dn_dl1[a] + inv_j22 * dn_dl2[a];
                    }

                    let mut b_mat = vec![vec![0.0; 12]; 3];
                    for a in 0..6 {
                        b_mat[0][2 * a] = dn_dx[a];
                        b_mat[1][2 * a + 1] = dn_dy[a];
                        b_mat[2][2 * a] = dn_dy[a];
                        b_mat[2][2 * a + 1] = dn_dx[a];
                    }
                    (b_mat, cell.nodes.clone())
                }
            }
            _ => panic!("Unsupported element kind in 2D stress recovery"),
        };

        let mut u_e = vec![0.0; dof_indices.len() * 2];
        for (i, &node) in dof_indices.iter().enumerate() {
            u_e[2 * i] = displacements[node][0];
            u_e[2 * i + 1] = displacements[node][1];
        }

        let mut eps = [0.0f64; 3];
        for row in 0..3 {
            for col in 0..u_e.len() {
                eps[row] += b_mat_rows[row][col] * u_e[col];
            }
        }

        let mut sigma = [0.0f64; 3];
        for (row, d_row) in d.iter().enumerate() {
            for (col, &d_val) in d_row.iter().enumerate() {
                sigma[row] += d_val * eps[col];
            }
        }

        element_strain.push(StrainTensor2D {
            eps_xx: eps[0],
            eps_yy: eps[1],
            gamma_xy: eps[2],
        });
        element_stress.push(StressTensor2D::from_components(
            sigma[0], sigma[1], sigma[2],
        ));

        for &node in &dof_indices {
            nodal_stress_sum[node][0] += sigma[0];
            nodal_stress_sum[node][1] += sigma[1];
            nodal_stress_sum[node][2] += sigma[2];
            nodal_count[node] += 1;
        }
    }

    let nodal_stress: Vec<StressTensor2D> = (0..node_count)
        .map(|node| {
            let count = nodal_count[node] as f64;
            if count > 0.0 {
                StressTensor2D::from_components(
                    nodal_stress_sum[node][0] / count,
                    nodal_stress_sum[node][1] / count,
                    nodal_stress_sum[node][2] / count,
                )
            } else {
                StressTensor2D::from_components(0.0, 0.0, 0.0)
            }
        })
        .collect();

    (element_stress, element_strain, nodal_stress)
}

/// Recovers element-centroid stress and strain tensors for a 3D Tet4 mesh.
///
/// Uses the element B-matrix (strain-displacement) and material D-matrix
/// (constitutive) to compute `sigma = D * B * u` at each element centroid.
pub fn recover_stress_3d(
    mesh: &MeshTopology<3>,
    node_coords: &[Point3],
    material: &ElasticityMaterial3D,
    displacements: &[[f64; 3]],
) -> (
    Vec<StressTensor3D>,
    Vec<StrainTensor3D>,
    Vec<StressTensor3D>,
) {
    let d_mat = constitutive_matrix_3d(*material);
    let node_count = node_coords.len();
    let cells = mesh.cells();

    let n_elem = cells.len();
    let mut element_stress = Vec::with_capacity(n_elem);
    let mut element_strain = Vec::with_capacity(n_elem);

    // Accumulators for nodal averaging.
    let mut nodal_stress_sum = vec![[0.0f64; 6]; node_count];
    let mut nodal_count = vec![0usize; node_count];

    for cell in cells {
        let (sigma, eps, nodes) = match cell.kind {
            ElementKind::Tet4 => {
                let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
                let a = node_coords[nodes[0]];
                let b = node_coords[nodes[1]];
                let c = node_coords[nodes[2]];
                let d_node = node_coords[nodes[3]];

                let jacobian = [
                    [
                        b.coords[0] - a.coords[0],
                        c.coords[0] - a.coords[0],
                        d_node.coords[0] - a.coords[0],
                    ],
                    [
                        b.coords[1] - a.coords[1],
                        c.coords[1] - a.coords[1],
                        d_node.coords[1] - a.coords[1],
                    ],
                    [
                        b.coords[2] - a.coords[2],
                        c.coords[2] - a.coords[2],
                        d_node.coords[2] - a.coords[2],
                    ],
                ];
                let determinant = determinant_3(jacobian);
                let volume = determinant.abs() / 6.0;

                if volume <= f64::EPSILON {
                    ([0.0; 6], [0.0; 6], cell.nodes.clone())
                } else {
                    let inverse = inverse_3(jacobian, determinant);
                    let gradients = [
                        [
                            -(inverse[0][0] + inverse[1][0] + inverse[2][0]),
                            -(inverse[0][1] + inverse[1][1] + inverse[2][1]),
                            -(inverse[0][2] + inverse[1][2] + inverse[2][2]),
                        ],
                        inverse[0],
                        inverse[1],
                        inverse[2],
                    ];
                    let b_mat = strain_displacement_matrix_3d(gradients);
                    let u_e = [
                        displacements[nodes[0]][0],
                        displacements[nodes[0]][1],
                        displacements[nodes[0]][2],
                        displacements[nodes[1]][0],
                        displacements[nodes[1]][1],
                        displacements[nodes[1]][2],
                        displacements[nodes[2]][0],
                        displacements[nodes[2]][1],
                        displacements[nodes[2]][2],
                        displacements[nodes[3]][0],
                        displacements[nodes[3]][1],
                        displacements[nodes[3]][2],
                    ];
                    let mut eps = [0.0f64; 6];
                    for (row, b_row) in b_mat.iter().enumerate() {
                        for (col, &b_val) in b_row.iter().enumerate() {
                            eps[row] += b_val * u_e[col];
                        }
                    }
                    let mut sigma = [0.0f64; 6];
                    for (row, d_row) in d_mat.iter().enumerate() {
                        for (col, &d_val) in d_row.iter().enumerate() {
                            sigma[row] += d_val * eps[col];
                        }
                    }
                    (sigma, eps, cell.nodes.clone())
                }
            }
            ElementKind::Hex8 => {
                let nodes = [
                    cell.nodes[0],
                    cell.nodes[1],
                    cell.nodes[2],
                    cell.nodes[3],
                    cell.nodes[4],
                    cell.nodes[5],
                    cell.nodes[6],
                    cell.nodes[7],
                ];
                let dn_dxi = [-0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.125];
                let dn_deta = [-0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125];
                let dn_dzeta = [-0.125, -0.125, -0.125, -0.125, 0.125, 0.125, 0.125, 0.125];

                let mut j = [[0.0; 3]; 3];
                for a in 0..8 {
                    let pt = node_coords[nodes[a]];
                    j[0][0] += dn_dxi[a] * pt.coords[0];
                    j[0][1] += dn_dxi[a] * pt.coords[1];
                    j[0][2] += dn_dxi[a] * pt.coords[2];

                    j[1][0] += dn_deta[a] * pt.coords[0];
                    j[1][1] += dn_deta[a] * pt.coords[1];
                    j[1][2] += dn_deta[a] * pt.coords[2];

                    j[2][0] += dn_dzeta[a] * pt.coords[0];
                    j[2][1] += dn_dzeta[a] * pt.coords[1];
                    j[2][2] += dn_dzeta[a] * pt.coords[2];
                }

                let det_j = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
                    - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
                    + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);

                if det_j.abs() <= f64::EPSILON {
                    ([0.0; 6], [0.0; 6], cell.nodes.clone())
                } else {
                    let inv_j = [
                        [
                            (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / det_j,
                            -(j[0][1] * j[2][2] - j[0][2] * j[2][1]) / det_j,
                            (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det_j,
                        ],
                        [
                            -(j[1][0] * j[2][2] - j[1][2] * j[2][0]) / det_j,
                            (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det_j,
                            -(j[0][0] * j[1][2] - j[0][2] * j[1][0]) / det_j,
                        ],
                        [
                            (j[1][0] * j[2][1] - j[1][1] * j[2][0]) / det_j,
                            -(j[0][0] * j[2][1] - j[0][1] * j[2][0]) / det_j,
                            (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / det_j,
                        ],
                    ];

                    let mut dn_dx = [0.0; 8];
                    let mut dn_dy = [0.0; 8];
                    let mut dn_dz = [0.0; 8];
                    for a in 0..8 {
                        dn_dx[a] = inv_j[0][0] * dn_dxi[a]
                            + inv_j[0][1] * dn_deta[a]
                            + inv_j[0][2] * dn_dzeta[a];
                        dn_dy[a] = inv_j[1][0] * dn_dxi[a]
                            + inv_j[1][1] * dn_deta[a]
                            + inv_j[1][2] * dn_dzeta[a];
                        dn_dz[a] = inv_j[2][0] * dn_dxi[a]
                            + inv_j[2][1] * dn_deta[a]
                            + inv_j[2][2] * dn_dzeta[a];
                    }

                    let mut b_mat = [[0.0; 24]; 6];
                    for a in 0..8 {
                        let col = 3 * a;
                        b_mat[0][col] = dn_dx[a];
                        b_mat[1][col + 1] = dn_dy[a];
                        b_mat[2][col + 2] = dn_dz[a];

                        b_mat[3][col] = dn_dy[a];
                        b_mat[3][col + 1] = dn_dx[a];

                        b_mat[4][col + 1] = dn_dz[a];
                        b_mat[4][col + 2] = dn_dy[a];

                        b_mat[5][col] = dn_dz[a];
                        b_mat[5][col + 2] = dn_dx[a];
                    }

                    let mut u_e = [0.0; 24];
                    for a in 0..8 {
                        u_e[3 * a] = displacements[nodes[a]][0];
                        u_e[3 * a + 1] = displacements[nodes[a]][1];
                        u_e[3 * a + 2] = displacements[nodes[a]][2];
                    }

                    let mut eps = [0.0f64; 6];
                    for (row, b_row) in b_mat.iter().enumerate() {
                        for (col, &b_val) in b_row.iter().enumerate() {
                            eps[row] += b_val * u_e[col];
                        }
                    }
                    let mut sigma = [0.0f64; 6];
                    for (row, d_row) in d_mat.iter().enumerate() {
                        for (col, &d_val) in d_row.iter().enumerate() {
                            sigma[row] += d_val * eps[col];
                        }
                    }
                    (sigma, eps, cell.nodes.clone())
                }
            }
            _ => ([0.0; 6], [0.0; 6], cell.nodes.clone()),
        };

        element_strain.push(StrainTensor3D {
            eps_xx: eps[0],
            eps_yy: eps[1],
            eps_zz: eps[2],
            gamma_xy: eps[3],
            gamma_yz: eps[4],
            gamma_xz: eps[5],
        });
        element_stress.push(StressTensor3D::from_components(
            sigma[0], sigma[1], sigma[2], sigma[3], sigma[4], sigma[5],
        ));

        // Accumulate for nodal averaging.
        for &node in &nodes {
            for k in 0..6 {
                nodal_stress_sum[node][k] += sigma[k];
            }
            nodal_count[node] += 1;
        }
    }

    // Produce nodal-averaged stress.
    let nodal_stress: Vec<StressTensor3D> = (0..node_count)
        .map(|node| {
            let count = nodal_count[node] as f64;
            if count > 0.0 {
                let s = nodal_stress_sum[node];
                StressTensor3D::from_components(
                    s[0] / count,
                    s[1] / count,
                    s[2] / count,
                    s[3] / count,
                    s[4] / count,
                    s[5] / count,
                )
            } else {
                StressTensor3D::from_components(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        })
        .collect();

    (element_stress, element_strain, nodal_stress)
}
