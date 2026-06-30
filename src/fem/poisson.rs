use std::collections::BTreeMap;

use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    fem::{
        dof::DOFManager,
        element::{Element, ElementError},
    },
    linalg::{
        LinalgError, LinearSolverOptions, SolverDiagnostics, SolverOptions, solve_linear_system,
    },
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

#[derive(Debug, Clone, PartialEq)]
pub struct PoissonSolverResult {
    pub values: Vec<f64>,
    pub diagnostics: SolverDiagnostics,
}

impl From<PoissonSolverResult> for PoissonResult {
    fn from(value: PoissonSolverResult) -> Self {
        Self {
            values: value.values,
            iterations: value.diagnostics.iterations,
            residual_norm: value.diagnostics.residual_norm,
        }
    }
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
    solve_poisson_with_solver(mesh, problem, LinearSolverOptions::from(options))
        .map(PoissonResult::from)
}

pub fn solve_poisson_with_solver<F>(
    mesh: &Mesh,
    problem: &PoissonProblem<F>,
    options: LinearSolverOptions,
) -> Result<PoissonSolverResult, PoissonError>
where
    F: Fn(f64, f64) -> f64,
{
    let (matrix, rhs) = assemble_poisson_system(mesh, problem)?;
    let result = solve_linear_system(&matrix, &rhs, options)?;

    Ok(PoissonSolverResult {
        values: result.values,
        diagnostics: result.diagnostics,
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
    solve_poisson_3d_with_solver(mesh, problem, LinearSolverOptions::from(options))
        .map(PoissonResult::from)
}

pub fn solve_poisson_3d_with_solver<F>(
    mesh: &MeshTopology<3>,
    problem: &PoissonProblem3D<F>,
    options: LinearSolverOptions,
) -> Result<PoissonSolverResult, PoissonError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let (matrix, rhs) = assemble_poisson_3d_system(mesh, problem)?;
    let result = solve_linear_system(&matrix, &rhs, options)?;

    Ok(PoissonSolverResult {
        values: result.values,
        diagnostics: result.diagnostics,
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

    let mut dof_manager = DOFManager::new();
    for node_id in 0..mesh.node_count() {
        dof_manager.register_dof(node_id, "u");
    }

    // Pre-evaluate the source in serial to avoid Sync constraints on F.
    let mut source_values = Vec::with_capacity(mesh.triangles().len());
    for triangle in mesh.triangles() {
        let (_, _, centroid) = triangle_geometry(mesh, triangle);
        let val = (problem.source)(centroid.x, centroid.y);
        if !val.is_finite() {
            return Err(PoissonError::NonFiniteSource {
                x: centroid.x,
                y: centroid.y,
            });
        }
        source_values.push(val);
    }

    use crate::parallel::Triplet;
    use rayon::prelude::*;

    let conductivity = problem.conductivity;
    let triangles = mesh.triangles();
    let element_contributions = triangles
        .par_iter()
        .zip(&source_values)
        .map(|(triangle, &source_val)| {
            let stiffness = local_stiffness(mesh, triangle, conductivity)?;
            let (area, _, _) = triangle_geometry(mesh, triangle);
            let load = [source_val * area / 3.0; 3];

            let mut elem_triplets = Vec::with_capacity(9);
            let mut elem_loads = Vec::with_capacity(3);
            for (local_row, global_row) in triangle.nodes.iter().copied().enumerate() {
                let eq_row = dof_manager.get_eq_index(global_row, "u").unwrap();
                elem_loads.push((eq_row, load[local_row]));
                for (local_col, global_col) in triangle.nodes.iter().copied().enumerate() {
                    let eq_col = dof_manager.get_eq_index(global_col, "u").unwrap();
                    elem_triplets.push(Triplet {
                        row: eq_row,
                        col: eq_col,
                        val: stiffness[local_row][local_col],
                    });
                }
            }
            Ok((elem_triplets, elem_loads))
        })
        .collect::<Result<Vec<_>, PoissonError>>()?;

    let mut triplets = TriMat::new((mesh.node_count(), mesh.node_count()));
    let mut rhs = vec![0.0; mesh.node_count()];
    for (elem_triplets, elem_loads) in element_contributions {
        for t in elem_triplets {
            triplets.add_triplet(t.row, t.col, t.val);
        }
        for (eq_row, val) in elem_loads {
            rhs[eq_row] += val;
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

    let mut dof_manager = DOFManager::new();
    for node_id in 0..mesh.points().len() {
        dof_manager.register_dof(node_id, "u");
    }

    // Check for unsupported elements and pre-evaluate source values in serial.
    let mut source_values = vec![0.0f64; mesh.cells().len()];
    for (cell_index, cell) in mesh.cells().iter().enumerate() {
        match cell.kind {
            ElementKind::Tet4 => {
                let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
                let (_, _, centroid) = tetrahedron_geometry(mesh, nodes);
                let val =
                    (problem.source)(centroid.coords[0], centroid.coords[1], centroid.coords[2]);
                if !val.is_finite() {
                    return Err(PoissonError::NonFiniteSource3D {
                        x: centroid.coords[0],
                        y: centroid.coords[1],
                        z: centroid.coords[2],
                    });
                }
                source_values[cell_index] = val;
            }
            ElementKind::Line2
            | ElementKind::Line3
            | ElementKind::Tri3
            | ElementKind::Tri6
            | ElementKind::Quad4
            | ElementKind::Quad8 => {}
            ElementKind::Hex8 | ElementKind::Hex20 | ElementKind::Tet10 => {
                return Err(PoissonError::UnsupportedElementKind {
                    cell_index,
                    kind: cell.kind,
                });
            }
        }
    }

    use crate::parallel::Triplet;
    use rayon::prelude::*;

    let conductivity = problem.conductivity;
    let cells = mesh.cells();
    let element_contributions = cells
        .par_iter()
        .enumerate()
        .filter_map(|(cell_index, cell)| {
            if cell.kind != ElementKind::Tet4 {
                return None;
            }
            let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
            let stiffness = match local_tet4_stiffness(mesh, nodes, conductivity) {
                Ok(s) => s,
                Err(e) => return Some(Err(e)),
            };
            let (volume, _, _) = tetrahedron_geometry(mesh, nodes);
            let source_val = source_values[cell_index];
            let load = [source_val * volume / 4.0; 4];

            let mut elem_triplets = Vec::with_capacity(16);
            let mut elem_loads = Vec::with_capacity(4);
            for (local_row, global_row) in nodes.iter().copied().enumerate() {
                let eq_row = dof_manager.get_eq_index(global_row, "u").unwrap();
                elem_loads.push((eq_row, load[local_row]));
                for (local_col, global_col) in nodes.iter().copied().enumerate() {
                    let eq_col = dof_manager.get_eq_index(global_col, "u").unwrap();
                    elem_triplets.push(Triplet {
                        row: eq_row,
                        col: eq_col,
                        val: stiffness[local_row][local_col],
                    });
                }
            }
            Some(Ok((elem_triplets, elem_loads)))
        })
        .collect::<Result<Vec<_>, PoissonError>>()?;

    let mut triplets = TriMat::new((mesh.points().len(), mesh.points().len()));
    let mut rhs = vec![0.0; mesh.points().len()];
    for (elem_triplets, elem_loads) in element_contributions {
        for t in elem_triplets {
            triplets.add_triplet(t.row, t.col, t.val);
        }
        for (eq_row, val) in elem_loads {
            rhs[eq_row] += val;
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

    let el = PoissonTri3 {
        nodes: &triangle.nodes,
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
    properties.insert("conductivity".to_string(), conductivity);

    let stiffness_vec = el.local_stiffness(&node_coords, &properties).map_err(|_| {
        PoissonError::UnsupportedElementKind {
            cell_index: 0,
            kind: ElementKind::Tri3,
        }
    })?;

    let mut stiffness = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            stiffness[r][c] = stiffness_vec[r][c];
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

    let el = PoissonTet4 { nodes: &nodes };
    let node_coords: Vec<Point3> = nodes
        .iter()
        .map(|&node_id| mesh.points()[node_id])
        .collect();

    let mut properties = BTreeMap::new();
    properties.insert("conductivity".to_string(), conductivity);

    let stiffness_vec = el.local_stiffness(&node_coords, &properties).map_err(|_| {
        PoissonError::UnsupportedElementKind {
            cell_index: 0,
            kind: ElementKind::Tet4,
        }
    })?;

    let mut stiffness = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            stiffness[r][c] = stiffness_vec[r][c];
        }
    }
    Ok(stiffness)
}

pub struct PoissonTri3<'a> {
    pub nodes: &'a [NodeId; 3],
}

impl<'a> Element for PoissonTri3<'a> {
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
        vec!["u".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let conductivity = *properties
            .get("conductivity")
            .ok_or_else(|| ElementError::MissingProperty("conductivity".to_string()))?;
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

        let mut stiffness = vec![vec![0.0; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                stiffness[row][col] = conductivity
                    * area
                    * (gradients[row][0] * gradients[col][0]
                        + gradients[row][1] * gradients[col][1]);
            }
        }
        Ok(stiffness)
    }

    #[allow(clippy::needless_range_loop)]
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
        let total_mass = density * area;
        let mut mass = vec![vec![0.0; 3]; 3];
        if lumped {
            for i in 0..3 {
                mass[i][i] = total_mass / 3.0;
            }
        } else {
            let val_diag = total_mass / 6.0;
            let val_off = total_mass / 12.0;
            for i in 0..3 {
                for j in 0..3 {
                    mass[i][j] = if i == j { val_diag } else { val_off };
                }
            }
        }
        Ok(mass)
    }
}

pub struct PoissonTet4<'a> {
    pub nodes: &'a [NodeId; 4],
}

impl<'a> Element for PoissonTet4<'a> {
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
        vec!["u".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let conductivity = *properties
            .get("conductivity")
            .ok_or_else(|| ElementError::MissingProperty("conductivity".to_string()))?;
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

        let mut stiffness = vec![vec![0.0; 4]; 4];
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

    #[allow(clippy::needless_range_loop)]
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
        let mut mass = vec![vec![0.0; 4]; 4];
        if lumped {
            for i in 0..4 {
                mass[i][i] = total_mass / 4.0;
            }
        } else {
            let val_diag = total_mass / 10.0;
            let val_off = total_mass / 20.0;
            for i in 0..4 {
                for j in 0..4 {
                    mass[i][j] = if i == j { val_diag } else { val_off };
                }
            }
        }
        Ok(mass)
    }
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
