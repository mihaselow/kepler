use std::collections::BTreeMap;

use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    fem::poisson::{local_load, local_stiffness, local_tet4_load, local_tet4_stiffness},
    linalg::{LinalgError, SolverOptions, conjugate_gradient},
    mesh::{ElementKind, Mesh, MeshTopology, NodeId, Tri3},
};

pub struct DiffusionReactionProblem<F> {
    pub diffusivity: f64,
    pub reaction_rate: f64,
    pub source: F,
    pub dirichlet: Vec<(NodeId, f64)>,
}

pub struct DiffusionReactionProblem3D<F> {
    pub diffusivity: f64,
    pub reaction_rate: f64,
    pub source: F,
    pub dirichlet: Vec<(NodeId, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionReactionResult {
    pub values: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Error, PartialEq)]
pub enum DiffusionReactionError {
    #[error("diffusivity must be positive and finite, got {0}")]
    InvalidDiffusivity(f64),
    #[error("reaction rate must be non-negative and finite, got {0}")]
    InvalidReactionRate(f64),
    #[error("Dirichlet boundary references node {node_id}, but mesh has {node_count} nodes")]
    BoundaryNodeOutOfBounds { node_id: NodeId, node_count: usize },
    #[error("Dirichlet boundary node {node_id} was specified more than once")]
    DuplicateBoundaryNode { node_id: NodeId },
    #[error("source evaluated to a non-finite value at ({x}, {y})")]
    NonFiniteSource { x: f64, y: f64 },
    #[error("source evaluated to a non-finite value at ({x}, {y}, {z})")]
    NonFiniteSource3D { x: f64, y: f64, z: f64 },
    #[error("cell {cell_index} has unsupported element kind {kind:?} for 3D diffusion-reaction")]
    UnsupportedElementKind {
        cell_index: usize,
        kind: ElementKind,
    },
    #[error("linear solver failed")]
    LinearSolve(#[from] LinalgError),
}

pub fn solve_diffusion_reaction<F>(
    mesh: &Mesh,
    problem: &DiffusionReactionProblem<F>,
    options: SolverOptions,
) -> Result<DiffusionReactionResult, DiffusionReactionError>
where
    F: Fn(f64, f64) -> f64,
{
    let (matrix, rhs) = assemble_diffusion_reaction_system(mesh, problem)?;
    let result = conjugate_gradient(&matrix, &rhs, options)?;

    Ok(DiffusionReactionResult {
        values: result.values,
        iterations: result.iterations,
        residual_norm: result.residual_norm,
    })
}

pub fn solve_diffusion_reaction_3d<F>(
    mesh: &MeshTopology<3>,
    problem: &DiffusionReactionProblem3D<F>,
    options: SolverOptions,
) -> Result<DiffusionReactionResult, DiffusionReactionError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let (matrix, rhs) = assemble_diffusion_reaction_3d_system(mesh, problem)?;
    let result = conjugate_gradient(&matrix, &rhs, options)?;

    Ok(DiffusionReactionResult {
        values: result.values,
        iterations: result.iterations,
        residual_norm: result.residual_norm,
    })
}

pub fn assemble_diffusion_reaction_system<F>(
    mesh: &Mesh,
    problem: &DiffusionReactionProblem<F>,
) -> Result<(CsMat<f64>, Vec<f64>), DiffusionReactionError>
where
    F: Fn(f64, f64) -> f64,
{
    validate_coefficients(problem.diffusivity, problem.reaction_rate)?;
    let dirichlet = validate_dirichlet(mesh.node_count(), &problem.dirichlet)?;
    let mut triplets = TriMat::with_capacity(
        (mesh.node_count(), mesh.node_count()),
        mesh.triangles().len() * 9,
    );
    let mut rhs = vec![0.0; mesh.node_count()];

    for triangle in mesh.triangles() {
        let stiffness = local_stiffness(mesh, triangle, problem.diffusivity)
            .map_err(map_poisson_source_error)?;
        let reaction = local_tri3_reaction(mesh, triangle, problem.reaction_rate);
        let load = local_load(mesh, triangle, &problem.source).map_err(map_poisson_source_error)?;

        for (local_row, global_row) in triangle.nodes.iter().copied().enumerate() {
            rhs[global_row] += load[local_row];
            for (local_col, global_col) in triangle.nodes.iter().copied().enumerate() {
                triplets.add_triplet(
                    global_row,
                    global_col,
                    stiffness[local_row][local_col] + reaction[local_row][local_col],
                );
            }
        }
    }

    Ok(apply_dirichlet(triplets.to_csr(), rhs, &dirichlet))
}

pub fn assemble_diffusion_reaction_3d_system<F>(
    mesh: &MeshTopology<3>,
    problem: &DiffusionReactionProblem3D<F>,
) -> Result<(CsMat<f64>, Vec<f64>), DiffusionReactionError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    validate_coefficients(problem.diffusivity, problem.reaction_rate)?;
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
                let stiffness = local_tet4_stiffness(mesh, nodes, problem.diffusivity)
                    .map_err(map_poisson_source_error)?;
                let reaction = local_tet4_reaction(mesh, nodes, problem.reaction_rate);
                let load = local_tet4_load(mesh, nodes, &problem.source)
                    .map_err(map_poisson_source_error)?;

                for (local_row, global_row) in nodes.iter().copied().enumerate() {
                    rhs[global_row] += load[local_row];
                    for (local_col, global_col) in nodes.iter().copied().enumerate() {
                        triplets.add_triplet(
                            global_row,
                            global_col,
                            stiffness[local_row][local_col] + reaction[local_row][local_col],
                        );
                    }
                }
            }
            ElementKind::Line2 | ElementKind::Tri3 | ElementKind::Quad4 => {}
            ElementKind::Hex8 => {
                return Err(DiffusionReactionError::UnsupportedElementKind {
                    cell_index,
                    kind: cell.kind,
                });
            }
        }
    }

    Ok(apply_dirichlet(triplets.to_csr(), rhs, &dirichlet))
}

pub fn local_tri3_reaction(mesh: &Mesh, triangle: &Tri3, reaction_rate: f64) -> [[f64; 3]; 3] {
    let area = mesh.triangle_area(triangle);
    let scale = reaction_rate * area / 12.0;
    [
        [2.0 * scale, scale, scale],
        [scale, 2.0 * scale, scale],
        [scale, scale, 2.0 * scale],
    ]
}

pub fn local_tet4_reaction(
    mesh: &MeshTopology<3>,
    nodes: [NodeId; 4],
    reaction_rate: f64,
) -> [[f64; 4]; 4] {
    let volume = tet4_volume(mesh, nodes);
    let scale = reaction_rate * volume / 20.0;
    [
        [2.0 * scale, scale, scale, scale],
        [scale, 2.0 * scale, scale, scale],
        [scale, scale, 2.0 * scale, scale],
        [scale, scale, scale, 2.0 * scale],
    ]
}

fn validate_coefficients(
    diffusivity: f64,
    reaction_rate: f64,
) -> Result<(), DiffusionReactionError> {
    if !diffusivity.is_finite() || diffusivity <= 0.0 {
        return Err(DiffusionReactionError::InvalidDiffusivity(diffusivity));
    }
    if !reaction_rate.is_finite() || reaction_rate < 0.0 {
        return Err(DiffusionReactionError::InvalidReactionRate(reaction_rate));
    }
    Ok(())
}

fn validate_dirichlet(
    node_count: usize,
    entries: &[(NodeId, f64)],
) -> Result<BTreeMap<NodeId, f64>, DiffusionReactionError> {
    let mut dirichlet = BTreeMap::new();
    for &(node_id, value) in entries {
        if node_id >= node_count {
            return Err(DiffusionReactionError::BoundaryNodeOutOfBounds {
                node_id,
                node_count,
            });
        }
        if dirichlet.insert(node_id, value).is_some() {
            return Err(DiffusionReactionError::DuplicateBoundaryNode { node_id });
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

fn tet4_volume(mesh: &MeshTopology<3>, nodes: [NodeId; 4]) -> f64 {
    let [a, b, c, d] = nodes.map(|node| mesh.points()[node]);
    let ab = [
        b.coords[0] - a.coords[0],
        b.coords[1] - a.coords[1],
        b.coords[2] - a.coords[2],
    ];
    let ac = [
        c.coords[0] - a.coords[0],
        c.coords[1] - a.coords[1],
        c.coords[2] - a.coords[2],
    ];
    let ad = [
        d.coords[0] - a.coords[0],
        d.coords[1] - a.coords[1],
        d.coords[2] - a.coords[2],
    ];
    determinant_3(ab, ac, ad).abs() / 6.0
}

fn determinant_3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn map_poisson_source_error(error: crate::fem::poisson::PoissonError) -> DiffusionReactionError {
    match error {
        crate::fem::poisson::PoissonError::InvalidConductivity(value) => {
            DiffusionReactionError::InvalidDiffusivity(value)
        }
        crate::fem::poisson::PoissonError::NonFiniteSource { x, y } => {
            DiffusionReactionError::NonFiniteSource { x, y }
        }
        crate::fem::poisson::PoissonError::NonFiniteSource3D { x, y, z } => {
            DiffusionReactionError::NonFiniteSource3D { x, y, z }
        }
        crate::fem::poisson::PoissonError::LinearSolve(error) => {
            DiffusionReactionError::LinearSolve(error)
        }
        crate::fem::poisson::PoissonError::BoundaryNodeOutOfBounds {
            node_id,
            node_count,
        } => DiffusionReactionError::BoundaryNodeOutOfBounds {
            node_id,
            node_count,
        },
        crate::fem::poisson::PoissonError::DuplicateBoundaryNode { node_id } => {
            DiffusionReactionError::DuplicateBoundaryNode { node_id }
        }
        crate::fem::poisson::PoissonError::UnsupportedElementKind { cell_index, kind } => {
            DiffusionReactionError::UnsupportedElementKind { cell_index, kind }
        }
    }
}
