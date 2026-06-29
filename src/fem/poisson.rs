use std::collections::BTreeMap;

use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    linalg::{LinalgError, SolverOptions, conjugate_gradient},
    mesh::{ElementKind, Mesh, MeshTopology, NodeId, Point2, Point3, Tri3},
};

pub struct PoissonProblem<F> {
    pub conductivity: f64,
    pub source: F,
    pub dirichlet: Vec<(NodeId, f64)>,
}

pub struct PoissonProblem3D<F> {
    pub conductivity: f64,
    pub source: F,
    pub dirichlet: Vec<(NodeId, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoissonResult {
    pub values: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Error, PartialEq)]
pub enum PoissonError {
    #[error("conductivity must be positive and finite, got {0}")]
    InvalidConductivity(f64),
    #[error("Dirichlet boundary references node {node_id}, but mesh has {node_count} nodes")]
    BoundaryNodeOutOfBounds { node_id: NodeId, node_count: usize },
    #[error("Dirichlet boundary node {node_id} was specified more than once")]
    DuplicateBoundaryNode { node_id: NodeId },
    #[error("source evaluated to a non-finite value at ({x}, {y})")]
    NonFiniteSource { x: f64, y: f64 },
    #[error("source evaluated to a non-finite value at ({x}, {y}, {z})")]
    NonFiniteSource3D { x: f64, y: f64, z: f64 },
    #[error("cell {cell_index} has unsupported element kind {kind:?} for 3D Poisson")]
    UnsupportedElementKind {
        cell_index: usize,
        kind: ElementKind,
    },
    #[error("linear solver failed")]
    LinearSolve(#[from] LinalgError),
}

pub fn solve_poisson<F>(
    mesh: &Mesh,
    problem: &PoissonProblem<F>,
    options: SolverOptions,
) -> Result<PoissonResult, PoissonError>
where
    F: Fn(f64, f64) -> f64,
{
    let (matrix, rhs) = assemble_poisson_system(mesh, problem)?;
    let result = conjugate_gradient(&matrix, &rhs, options)?;

    Ok(PoissonResult {
        values: result.values,
        iterations: result.iterations,
        residual_norm: result.residual_norm,
    })
}

pub fn solve_poisson_3d<F>(
    mesh: &MeshTopology<3>,
    problem: &PoissonProblem3D<F>,
    options: SolverOptions,
) -> Result<PoissonResult, PoissonError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let (matrix, rhs) = assemble_poisson_3d_system(mesh, problem)?;
    let result = conjugate_gradient(&matrix, &rhs, options)?;

    Ok(PoissonResult {
        values: result.values,
        iterations: result.iterations,
        residual_norm: result.residual_norm,
    })
}

pub fn assemble_poisson_system<F>(
    mesh: &Mesh,
    problem: &PoissonProblem<F>,
) -> Result<(CsMat<f64>, Vec<f64>), PoissonError>
where
    F: Fn(f64, f64) -> f64,
{
    validate_conductivity(problem.conductivity)?;
    let dirichlet = validate_dirichlet(mesh.node_count(), &problem.dirichlet)?;

    let mut triplets = TriMat::with_capacity(
        (mesh.node_count(), mesh.node_count()),
        mesh.triangles().len() * 9 + dirichlet.len(),
    );
    let mut rhs = vec![0.0; mesh.node_count()];

    for triangle in mesh.triangles() {
        let stiffness = local_stiffness(mesh, triangle, problem.conductivity)?;
        let load = local_load(mesh, triangle, &problem.source)?;

        for (local_row, global_row) in triangle.nodes.iter().copied().enumerate() {
            rhs[global_row] += load[local_row];
            for (local_col, global_col) in triangle.nodes.iter().copied().enumerate() {
                triplets.add_triplet(global_row, global_col, stiffness[local_row][local_col]);
            }
        }
    }

    let matrix = triplets.to_csr();
    Ok(apply_dirichlet(matrix, rhs, &dirichlet))
}

pub fn assemble_poisson_3d_system<F>(
    mesh: &MeshTopology<3>,
    problem: &PoissonProblem3D<F>,
) -> Result<(CsMat<f64>, Vec<f64>), PoissonError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    validate_conductivity(problem.conductivity)?;
    let dirichlet = validate_dirichlet(mesh.points().len(), &problem.dirichlet)?;
    let tet_count = mesh
        .cells()
        .iter()
        .filter(|cell| cell.kind == ElementKind::Tet4)
        .count();
    let mut triplets =
        TriMat::with_capacity((mesh.points().len(), mesh.points().len()), tet_count * 16);
    let mut rhs = vec![0.0; mesh.points().len()];

    for (cell_index, cell) in mesh.cells().iter().enumerate() {
        match cell.kind {
            ElementKind::Tet4 => {
                let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
                let stiffness = local_tet4_stiffness(mesh, nodes, problem.conductivity)?;
                let load = local_tet4_load(mesh, nodes, &problem.source)?;

                for (local_row, global_row) in nodes.iter().copied().enumerate() {
                    rhs[global_row] += load[local_row];
                    for (local_col, global_col) in nodes.iter().copied().enumerate() {
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
                return Err(PoissonError::UnsupportedElementKind {
                    cell_index,
                    kind: cell.kind,
                });
            }
        }
    }

    let matrix = triplets.to_csr();
    Ok(apply_dirichlet(matrix, rhs, &dirichlet))
}

pub fn local_stiffness(
    mesh: &Mesh,
    triangle: &Tri3,
    conductivity: f64,
) -> Result<[[f64; 3]; 3], PoissonError> {
    validate_conductivity(conductivity)?;
    let (area, gradients, _) = triangle_geometry(mesh, triangle);
    let mut stiffness = [[0.0; 3]; 3];

    for row in 0..3 {
        for col in 0..3 {
            stiffness[row][col] = conductivity
                * area
                * (gradients[row][0] * gradients[col][0] + gradients[row][1] * gradients[col][1]);
        }
    }

    Ok(stiffness)
}

pub fn local_load<F>(mesh: &Mesh, triangle: &Tri3, source: F) -> Result<[f64; 3], PoissonError>
where
    F: Fn(f64, f64) -> f64,
{
    let (area, _, centroid) = triangle_geometry(mesh, triangle);
    let source_value = source(centroid.x, centroid.y);
    if !source_value.is_finite() {
        return Err(PoissonError::NonFiniteSource {
            x: centroid.x,
            y: centroid.y,
        });
    }

    Ok([source_value * area / 3.0; 3])
}

pub fn local_tet4_stiffness(
    mesh: &MeshTopology<3>,
    nodes: [NodeId; 4],
    conductivity: f64,
) -> Result<[[f64; 4]; 4], PoissonError> {
    validate_conductivity(conductivity)?;
    let (volume, gradients, _) = tetrahedron_geometry(mesh, nodes);
    let mut stiffness = [[0.0; 4]; 4];

    for row in 0..4 {
        for col in 0..4 {
            stiffness[row][col] = conductivity
                * volume
                * (gradients[row][0] * gradients[col][0]
                    + gradients[row][1] * gradients[col][1]
                    + gradients[row][2] * gradients[col][2]);
        }
    }

    Ok(stiffness)
}

pub fn local_tet4_load<F>(
    mesh: &MeshTopology<3>,
    nodes: [NodeId; 4],
    source: F,
) -> Result<[f64; 4], PoissonError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let (volume, _, centroid) = tetrahedron_geometry(mesh, nodes);
    let source_value = source(centroid.coords[0], centroid.coords[1], centroid.coords[2]);
    if !source_value.is_finite() {
        return Err(PoissonError::NonFiniteSource3D {
            x: centroid.coords[0],
            y: centroid.coords[1],
            z: centroid.coords[2],
        });
    }

    Ok([source_value * volume / 4.0; 4])
}

fn validate_conductivity(conductivity: f64) -> Result<(), PoissonError> {
    if conductivity.is_finite() && conductivity > 0.0 {
        Ok(())
    } else {
        Err(PoissonError::InvalidConductivity(conductivity))
    }
}

fn validate_dirichlet(
    node_count: usize,
    entries: &[(NodeId, f64)],
) -> Result<BTreeMap<NodeId, f64>, PoissonError> {
    let mut dirichlet = BTreeMap::new();

    for &(node_id, value) in entries {
        if node_id >= node_count {
            return Err(PoissonError::BoundaryNodeOutOfBounds {
                node_id,
                node_count,
            });
        }
        if dirichlet.insert(node_id, value).is_some() {
            return Err(PoissonError::DuplicateBoundaryNode { node_id });
        }
    }

    Ok(dirichlet)
}

fn apply_dirichlet(
    matrix: CsMat<f64>,
    rhs: Vec<f64>,
    dirichlet: &BTreeMap<NodeId, f64>,
) -> (CsMat<f64>, Vec<f64>) {
    if dirichlet.is_empty() {
        return (matrix, rhs);
    }

    let mut adjusted_rhs = rhs;
    let mut constrained_triplets = TriMat::new((matrix.rows(), matrix.cols()));

    for (row_index, row) in matrix.outer_iterator().enumerate() {
        if dirichlet.contains_key(&row_index) {
            continue;
        }

        for (col_index, value) in row.iter() {
            if let Some(boundary_value) = dirichlet.get(&col_index) {
                adjusted_rhs[row_index] -= *value * boundary_value;
            } else {
                constrained_triplets.add_triplet(row_index, col_index, *value);
            }
        }
    }

    for (&node_id, &value) in dirichlet {
        adjusted_rhs[node_id] = value;
        constrained_triplets.add_triplet(node_id, node_id, 1.0);
    }

    (constrained_triplets.to_csr(), adjusted_rhs)
}

fn triangle_geometry(mesh: &Mesh, triangle: &Tri3) -> (f64, [[f64; 2]; 3], Point2) {
    let [a, b, c] = triangle.nodes.map(|node| mesh.points()[node]);
    let twice_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    let area = 0.5 * twice_area.abs();

    let gradients = [
        [(b.y - c.y) / twice_area, (c.x - b.x) / twice_area],
        [(c.y - a.y) / twice_area, (a.x - c.x) / twice_area],
        [(a.y - b.y) / twice_area, (b.x - a.x) / twice_area],
    ];

    let centroid = Point2::new((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0);

    (area, gradients, centroid)
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
