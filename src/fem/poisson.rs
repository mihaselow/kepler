use std::collections::BTreeMap;

use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    linalg::{LinalgError, SolverOptions, conjugate_gradient},
    mesh::{Mesh, NodeId, Point2, Tri3},
};

pub struct PoissonProblem<F> {
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
