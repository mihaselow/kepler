use std::collections::BTreeMap;

use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    linalg::{LinalgError, SolverOptions, conjugate_gradient},
    mesh::{Mesh, NodeId, Tri3},
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplacementConstraint {
    pub node: NodeId,
    pub component: DisplacementComponent,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodalForce {
    pub node: NodeId,
    pub fx: f64,
    pub fy: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticityProblem {
    pub material: ElasticityMaterial,
    pub thickness: f64,
    pub constraints: Vec<DisplacementConstraint>,
    pub forces: Vec<NodalForce>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticityResult {
    pub displacements: Vec<[f64; 2]>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Error, PartialEq)]
pub enum ElasticityError {
    #[error("Young's modulus must be positive and finite, got {0}")]
    InvalidYoungModulus(f64),
    #[error("Poisson ratio must be finite and in (-1, 0.5), got {0}")]
    InvalidPoissonRatio(f64),
    #[error("thickness must be positive and finite, got {0}")]
    InvalidThickness(f64),
    #[error("constraint references node {node_id}, but mesh has {node_count} nodes")]
    ConstraintNodeOutOfBounds { node_id: NodeId, node_count: usize },
    #[error("force references node {node_id}, but mesh has {node_count} nodes")]
    ForceNodeOutOfBounds { node_id: NodeId, node_count: usize },
    #[error("constraint for node {node_id} component {component:?} was specified more than once")]
    DuplicateConstraint {
        node_id: NodeId,
        component: DisplacementComponent,
    },
    #[error("linear solver failed")]
    LinearSolve(#[from] LinalgError),
}

pub fn solve_elasticity(
    mesh: &Mesh,
    problem: &ElasticityProblem,
    options: SolverOptions,
) -> Result<ElasticityResult, ElasticityError> {
    let (matrix, rhs) = assemble_elasticity_system(mesh, problem)?;
    let result = conjugate_gradient(&matrix, &rhs, options)?;
    let displacements = result
        .values
        .chunks_exact(2)
        .map(|values| [values[0], values[1]])
        .collect();

    Ok(ElasticityResult {
        displacements,
        iterations: result.iterations,
        residual_norm: result.residual_norm,
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
    let mut triplets = TriMat::with_capacity((dof_count, dof_count), mesh.triangles().len() * 36);
    let mut rhs = vec![0.0; dof_count];

    for triangle in mesh.triangles() {
        let stiffness =
            local_elasticity_stiffness(mesh, triangle, problem.material, problem.thickness)?;
        let dofs = triangle_dofs(triangle);
        for (local_row, global_row) in dofs.iter().copied().enumerate() {
            for (local_col, global_col) in dofs.iter().copied().enumerate() {
                triplets.add_triplet(global_row, global_col, stiffness[local_row][local_col]);
            }
        }
    }

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

    let matrix = triplets.to_csr();
    Ok(apply_constraints(matrix, rhs, &constraints))
}

pub fn local_elasticity_stiffness(
    mesh: &Mesh,
    triangle: &Tri3,
    material: ElasticityMaterial,
    thickness: f64,
) -> Result<[[f64; 6]; 6], ElasticityError> {
    validate_material(material)?;
    validate_thickness(thickness)?;
    let (area, gradients) = triangle_geometry(mesh, triangle);
    let constitutive = constitutive_matrix(material);
    let b = strain_displacement_matrix(gradients);
    let mut stiffness = [[0.0; 6]; 6];

    for row in 0..6 {
        for col in 0..6 {
            let mut value = 0.0;
            for alpha in 0..3 {
                for beta in 0..3 {
                    value += b[alpha][row] * constitutive[alpha][beta] * b[beta][col];
                }
            }
            stiffness[row][col] = thickness * area * value;
        }
    }

    Ok(stiffness)
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

fn validate_thickness(thickness: f64) -> Result<(), ElasticityError> {
    if thickness.is_finite() && thickness > 0.0 {
        Ok(())
    } else {
        Err(ElasticityError::InvalidThickness(thickness))
    }
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
