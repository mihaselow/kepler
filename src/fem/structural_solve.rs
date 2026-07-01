use std::collections::BTreeMap;

use rayon::prelude::*;
use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    fem::{
        element::{Element, ElementError},
        structural::{Beam3D, ShellQuad4, ShellTri3},
    },
    linalg::{LinearSolverOptions, SolverOptions, solve_linear_system},
    mesh::{ElementKind, MeshTopology, NodeId, Point3},
};

pub const DOFS_PER_NODE: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StructuralComponent {
    Ux,
    Uy,
    Uz,
    ThetaX,
    ThetaY,
    ThetaZ,
}

impl StructuralComponent {
    pub fn offset(self) -> usize {
        match self {
            Self::Ux => 0,
            Self::Uy => 1,
            Self::Uz => 2,
            Self::ThetaX => 3,
            Self::ThetaY => 4,
            Self::ThetaZ => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralMaterial {
    pub young_modulus: f64,
    pub poisson_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamSection {
    pub area: f64,
    pub moment_y: f64,
    pub moment_z: f64,
    pub torsional_constant: f64,
    pub local_y_direction: [f64; 3],
}

impl Default for BeamSection {
    fn default() -> Self {
        Self {
            area: 0.01,
            moment_y: 1.0e-5,
            moment_z: 2.0e-5,
            torsional_constant: 3.0e-5,
            local_y_direction: [0.0, 1.0, 0.0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralConstraint {
    pub node: NodeId,
    pub component: StructuralComponent,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralForce {
    pub node: NodeId,
    pub component: StructuralComponent,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuralProblem {
    pub material: StructuralMaterial,
    pub beam_section: BeamSection,
    pub shell_thickness: f64,
    pub constraints: Vec<StructuralConstraint>,
    pub forces: Vec<StructuralForce>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuralResult {
    pub displacements: Vec<[f64; DOFS_PER_NODE]>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Error, PartialEq)]
pub enum StructuralError {
    #[error("structural mesh must contain at least one cell")]
    EmptyMesh,
    #[error("unsupported element kind {kind:?} at cell {cell_index}")]
    UnsupportedElementKind {
        cell_index: usize,
        kind: ElementKind,
    },
    #[error("element assembly failed at cell {cell_index}: {source}")]
    ElementAssembly {
        cell_index: usize,
        #[source]
        source: ElementError,
    },
    #[error("constraint node {node_id} is out of bounds (mesh has {node_count} nodes)")]
    ConstraintNodeOutOfBounds { node_id: NodeId, node_count: usize },
    #[error("duplicate constraint at node {node_id} for component {component:?}")]
    DuplicateConstraint {
        node_id: NodeId,
        component: StructuralComponent,
    },
    #[error("force node {node_id} is out of bounds (mesh has {node_count} nodes)")]
    ForceNodeOutOfBounds { node_id: NodeId, node_count: usize },
    #[error("linear solver error: {0}")]
    Linalg(#[from] crate::linalg::LinalgError),
}

pub fn dof_index_6(node: NodeId, component: StructuralComponent) -> usize {
    node * DOFS_PER_NODE + component.offset()
}

pub fn solve_structural(
    mesh: &MeshTopology<3>,
    problem: &StructuralProblem,
    options: SolverOptions,
) -> Result<StructuralResult, StructuralError> {
    solve_structural_with_solver(mesh, problem, LinearSolverOptions::from(options))
}

pub fn solve_structural_with_solver(
    mesh: &MeshTopology<3>,
    problem: &StructuralProblem,
    options: LinearSolverOptions,
) -> Result<StructuralResult, StructuralError> {
    let (matrix, rhs) = assemble_structural_system(mesh, problem)?;
    let result = solve_linear_system(&matrix, &rhs, options)?;
    let displacements: Vec<[f64; DOFS_PER_NODE]> = result
        .values
        .chunks_exact(DOFS_PER_NODE)
        .map(|values| {
            [
                values[0], values[1], values[2], values[3], values[4], values[5],
            ]
        })
        .collect();

    Ok(StructuralResult {
        displacements,
        iterations: result.diagnostics.iterations,
        residual_norm: result.diagnostics.residual_norm,
    })
}

pub fn assemble_structural_system(
    mesh: &MeshTopology<3>,
    problem: &StructuralProblem,
) -> Result<(CsMat<f64>, Vec<f64>), StructuralError> {
    if mesh.cells().is_empty() {
        return Err(StructuralError::EmptyMesh);
    }

    let node_count = mesh.points().len();
    let constraints = validate_constraints(node_count, &problem.constraints)?;
    let matrix = assemble_structural_stiffness_matrix(mesh, problem)?;
    let mut rhs = vec![0.0; node_count * DOFS_PER_NODE];
    for force in &problem.forces {
        if force.node >= node_count {
            return Err(StructuralError::ForceNodeOutOfBounds {
                node_id: force.node,
                node_count,
            });
        }
        let dof = dof_index_6(force.node, force.component);
        rhs[dof] += force.value;
    }
    Ok(apply_constraints(matrix, rhs, &constraints))
}

fn validate_constraints(
    node_count: usize,
    constraints: &[StructuralConstraint],
) -> Result<BTreeMap<usize, f64>, StructuralError> {
    let mut constrained = BTreeMap::new();
    for constraint in constraints {
        if constraint.node >= node_count {
            return Err(StructuralError::ConstraintNodeOutOfBounds {
                node_id: constraint.node,
                node_count,
            });
        }
        let dof = dof_index_6(constraint.node, constraint.component);
        if constrained.insert(dof, constraint.value).is_some() {
            return Err(StructuralError::DuplicateConstraint {
                node_id: constraint.node,
                component: constraint.component,
            });
        }
    }
    Ok(constrained)
}

fn material_properties(problem: &StructuralProblem) -> BTreeMap<String, f64> {
    let mut properties = BTreeMap::new();
    properties.insert("young_modulus".to_string(), problem.material.young_modulus);
    properties.insert("poisson_ratio".to_string(), problem.material.poisson_ratio);
    properties
}

fn assemble_structural_stiffness_matrix(
    mesh: &MeshTopology<3>,
    problem: &StructuralProblem,
) -> Result<CsMat<f64>, StructuralError> {
    use crate::parallel::{Triplet, merge_triplets};

    let dof_count = mesh.points().len() * DOFS_PER_NODE;
    let points = mesh.points();
    let properties = material_properties(problem);
    let cells: Vec<_> = mesh.cells().to_vec();

    let element_triplets = cells
        .par_iter()
        .enumerate()
        .map(|(cell_index, cell)| {
            let node_coords: Vec<Point3> = cell.nodes.iter().map(|&id| points[id]).collect();

            let local_k = match cell.kind {
                ElementKind::Line2 => {
                    let nodes = [cell.nodes[0], cell.nodes[1]];
                    let el = Beam3D {
                        nodes: &nodes,
                        area: problem.beam_section.area,
                        moment_y: problem.beam_section.moment_y,
                        moment_z: problem.beam_section.moment_z,
                        torsional_constant: problem.beam_section.torsional_constant,
                        local_y_direction: problem.beam_section.local_y_direction,
                    };
                    el.local_stiffness(&node_coords, &properties)
                }
                ElementKind::Tri3 => {
                    let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2]];
                    let el = ShellTri3 {
                        nodes: &nodes,
                        thickness: problem.shell_thickness,
                    };
                    el.local_stiffness(&node_coords, &properties)
                }
                ElementKind::Quad4 => {
                    let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
                    let el = ShellQuad4 {
                        nodes: &nodes,
                        thickness: problem.shell_thickness,
                    };
                    el.local_stiffness(&node_coords, &properties)
                }
                kind => {
                    return Err(StructuralError::UnsupportedElementKind { cell_index, kind });
                }
            }
            .map_err(|source| StructuralError::ElementAssembly { cell_index, source })?;

            let mut dofs = Vec::with_capacity(cell.nodes.len() * DOFS_PER_NODE);
            for &node in &cell.nodes {
                for component_offset in 0..DOFS_PER_NODE {
                    dofs.push(node * DOFS_PER_NODE + component_offset);
                }
            }

            let mut triplets = Vec::with_capacity(dofs.len() * dofs.len());
            for (local_row, &global_row) in dofs.iter().enumerate() {
                for (local_col, &global_col) in dofs.iter().enumerate() {
                    triplets.push(Triplet {
                        row: global_row,
                        col: global_col,
                        val: local_k[local_row][local_col],
                    });
                }
            }
            Ok(triplets)
        })
        .collect::<Result<Vec<_>, StructuralError>>()?;

    let tri = merge_triplets(dof_count, element_triplets);
    Ok(tri.to_csr())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Cell, ElementKind};

    fn axial_beam_mesh(length: f64) -> MeshTopology<3> {
        MeshTopology::new(
            vec![
                Point3::new([0.0, 0.0, 0.0]),
                Point3::new([length, 0.0, 0.0]),
            ],
            vec![Cell::new(ElementKind::Line2, vec![0, 1])],
        )
        .unwrap()
    }

    fn fixed_node_constraints(node: NodeId) -> Vec<StructuralConstraint> {
        [
            StructuralComponent::Ux,
            StructuralComponent::Uy,
            StructuralComponent::Uz,
            StructuralComponent::ThetaX,
            StructuralComponent::ThetaY,
            StructuralComponent::ThetaZ,
        ]
        .into_iter()
        .map(|component| StructuralConstraint {
            node,
            component,
            value: 0.0,
        })
        .collect()
    }

    #[test]
    fn solve_axial_beam_matches_analytical_stretch() {
        let length = 2.0;
        let mesh = axial_beam_mesh(length);
        let area = 0.01;
        let young_modulus = 200e9;
        let force = 1.0e6;

        let problem = StructuralProblem {
            material: StructuralMaterial {
                young_modulus,
                poisson_ratio: 0.3,
            },
            beam_section: BeamSection {
                area,
                ..BeamSection::default()
            },
            shell_thickness: 0.1,
            constraints: fixed_node_constraints(0),
            forces: vec![StructuralForce {
                node: 1,
                component: StructuralComponent::Ux,
                value: force,
            }],
        };

        let result = solve_structural(&mesh, &problem, SolverOptions::default()).unwrap();
        let expected_ux = force * length / (young_modulus * area);
        assert!((result.displacements[1][0] - expected_ux).abs() < 1.0e-9);
        assert!(result.displacements[1][1].abs() < 1.0e-9);
        assert!(result.displacements[1][2].abs() < 1.0e-9);
    }

    #[test]
    fn solve_square_shell_patch_is_finite_and_symmetric() {
        let mesh = MeshTopology::new(
            vec![
                Point3::new([0.0, 0.0, 0.0]),
                Point3::new([2.0, 0.0, 0.0]),
                Point3::new([2.0, 2.0, 0.0]),
                Point3::new([0.0, 2.0, 0.0]),
            ],
            vec![Cell::new(ElementKind::Quad4, vec![0, 1, 2, 3])],
        )
        .unwrap();

        let mut constraints = fixed_node_constraints(0);
        constraints.extend(fixed_node_constraints(3));

        let problem = StructuralProblem {
            material: StructuralMaterial {
                young_modulus: 200e9,
                poisson_ratio: 0.3,
            },
            beam_section: BeamSection::default(),
            shell_thickness: 0.1,
            constraints,
            forces: vec![StructuralForce {
                node: 1,
                component: StructuralComponent::Uz,
                value: 1000.0,
            }],
        };

        let result = solve_structural(&mesh, &problem, SolverOptions::default()).unwrap();
        assert!(result.displacements[1][2].is_finite());
        assert!(result.displacements[1][2] > 0.0);
    }
}
