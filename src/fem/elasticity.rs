use std::collections::BTreeMap;

use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    linalg::{LinalgError, SolverOptions, conjugate_gradient},
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
pub struct ElasticityResult {
    pub displacements: Vec<[f64; 2]>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticityResult3D {
    pub displacements: Vec<[f64; 3]>,
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

pub fn solve_elasticity_3d(
    mesh: &MeshTopology<3>,
    problem: &ElasticityProblem3D,
    options: SolverOptions,
) -> Result<ElasticityResult3D, ElasticityError> {
    let (matrix, rhs) = assemble_elasticity_3d_system(mesh, problem)?;
    let result = conjugate_gradient(&matrix, &rhs, options)?;
    let displacements = result
        .values
        .chunks_exact(3)
        .map(|values| [values[0], values[1], values[2]])
        .collect();

    Ok(ElasticityResult3D {
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

pub fn assemble_elasticity_3d_system(
    mesh: &MeshTopology<3>,
    problem: &ElasticityProblem3D,
) -> Result<(CsMat<f64>, Vec<f64>), ElasticityError> {
    validate_material_3d(problem.material)?;
    let constraints = validate_constraints_3d(mesh.points().len(), &problem.constraints)?;
    let dof_count = mesh.points().len() * 3;
    let tet_count = mesh
        .cells()
        .iter()
        .filter(|cell| cell.kind == ElementKind::Tet4)
        .count();
    let mut triplets = TriMat::with_capacity((dof_count, dof_count), tet_count * 144);
    let mut rhs = vec![0.0; dof_count];

    for (cell_index, cell) in mesh.cells().iter().enumerate() {
        match cell.kind {
            ElementKind::Tet4 => {
                let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
                let stiffness = local_tet4_elasticity_stiffness(mesh, nodes, problem.material)?;
                let dofs = tet4_dofs(nodes);
                for (local_row, global_row) in dofs.iter().copied().enumerate() {
                    for (local_col, global_col) in dofs.iter().copied().enumerate() {
                        triplets.add_triplet(
                            global_row,
                            global_col,
                            stiffness[local_row][local_col],
                        );
                    }
                }
            }
            ElementKind::Line2 | ElementKind::Tri3 | ElementKind::Quad4 => {}
            ElementKind::Hex8 => {
                return Err(ElasticityError::UnsupportedElementKind {
                    cell_index,
                    kind: cell.kind,
                });
            }
        }
    }

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

    Ok(apply_constraints(triplets.to_csr(), rhs, &constraints))
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

pub fn local_tet4_elasticity_stiffness(
    mesh: &MeshTopology<3>,
    nodes: [NodeId; 4],
    material: ElasticityMaterial3D,
) -> Result<[[f64; 12]; 12], ElasticityError> {
    validate_material_3d(material)?;
    let (volume, gradients, _) = tetrahedron_geometry(mesh, nodes);
    let constitutive = constitutive_matrix_3d(material);
    let b = strain_displacement_matrix_3d(gradients);
    let mut stiffness = [[0.0; 12]; 12];

    for row in 0..12 {
        for col in 0..12 {
            let mut value = 0.0;
            for alpha in 0..6 {
                for beta in 0..6 {
                    value += b[alpha][row] * constitutive[alpha][beta] * b[beta][col];
                }
            }
            stiffness[row][col] = volume * value;
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
