use std::{collections::BTreeSet, f64::consts::PI};

use sprs::CsMat;
use thiserror::Error;

use crate::{
    fem::elasticity::{
        DisplacementComponent, DisplacementComponent3D, DisplacementConstraint,
        DisplacementConstraint3D, ElasticityError, ElasticityProblem, ElasticityProblem3D,
        assemble_elasticity_3d_system, assemble_elasticity_system,
    },
    mesh::{ElementKind, Mesh, MeshTopology, NodeId},
};

#[derive(Debug, Clone, PartialEq)]
pub struct ModalProblem {
    pub elasticity: ElasticityProblem,
    pub density: f64,
    pub mode_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModalProblem3D {
    pub elasticity: ElasticityProblem3D,
    pub density: f64,
    pub mode_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModeShape {
    pub frequency_hz: f64,
    pub angular_frequency: f64,
    pub displacements: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModeShape3D {
    pub frequency_hz: f64,
    pub angular_frequency: f64,
    pub displacements: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModalResult {
    pub modes: Vec<ModeShape>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModalResult3D {
    pub modes: Vec<ModeShape3D>,
}

#[derive(Debug, Error, PartialEq)]
pub enum ModalError {
    #[error("density must be positive and finite, got {0}")]
    InvalidDensity(f64),
    #[error("mode count must be greater than zero")]
    InvalidModeCount,
    #[error("no unconstrained degrees of freedom remain")]
    NoActiveDegreesOfFreedom,
    #[error("degree of freedom {dof} has non-positive lumped mass {mass}")]
    NonPositiveMass { dof: usize, mass: f64 },
    #[error("elasticity assembly failed")]
    Elasticity(#[from] ElasticityError),
}

pub fn solve_modal(mesh: &Mesh, problem: &ModalProblem) -> Result<ModalResult, ModalError> {
    validate_modal_inputs(problem.density, problem.mode_count)?;

    let (stiffness, _) = assemble_elasticity_system(mesh, &problem.elasticity)?;
    let mass = lumped_mass_2d(mesh, problem.density, problem.elasticity.thickness);
    let constrained = constrained_dofs_2d(&problem.elasticity.constraints);
    let eigenpairs = solve_reduced_modes(&stiffness, &mass, &constrained, problem.mode_count)?;
    let modes = eigenpairs
        .into_iter()
        .map(|pair| ModeShape {
            frequency_hz: pair.frequency_hz,
            angular_frequency: pair.angular_frequency,
            displacements: pair
                .dof_values
                .chunks_exact(2)
                .map(|values| [values[0], values[1]])
                .collect(),
        })
        .collect();

    Ok(ModalResult { modes })
}

pub fn solve_modal_3d(
    mesh: &MeshTopology<3>,
    problem: &ModalProblem3D,
) -> Result<ModalResult3D, ModalError> {
    validate_modal_inputs(problem.density, problem.mode_count)?;

    let (stiffness, _) = assemble_elasticity_3d_system(mesh, &problem.elasticity)?;
    let mass = lumped_mass_3d(mesh, problem.density);
    let constrained = constrained_dofs_3d(&problem.elasticity.constraints);
    let eigenpairs = solve_reduced_modes(&stiffness, &mass, &constrained, problem.mode_count)?;
    let modes = eigenpairs
        .into_iter()
        .map(|pair| ModeShape3D {
            frequency_hz: pair.frequency_hz,
            angular_frequency: pair.angular_frequency,
            displacements: pair
                .dof_values
                .chunks_exact(3)
                .map(|values| [values[0], values[1], values[2]])
                .collect(),
        })
        .collect();

    Ok(ModalResult3D { modes })
}

#[derive(Debug, Clone, PartialEq)]
struct Eigenpair {
    frequency_hz: f64,
    angular_frequency: f64,
    dof_values: Vec<f64>,
}

fn validate_modal_inputs(density: f64, mode_count: usize) -> Result<(), ModalError> {
    if !density.is_finite() || density <= 0.0 {
        return Err(ModalError::InvalidDensity(density));
    }
    if mode_count == 0 {
        return Err(ModalError::InvalidModeCount);
    }
    Ok(())
}

fn constrained_dofs_2d(constraints: &[DisplacementConstraint]) -> BTreeSet<usize> {
    constraints
        .iter()
        .map(|constraint| dof_index_2d(constraint.node, constraint.component))
        .collect()
}

fn constrained_dofs_3d(constraints: &[DisplacementConstraint3D]) -> BTreeSet<usize> {
    constraints
        .iter()
        .map(|constraint| dof_index_3d(constraint.node, constraint.component))
        .collect()
}

fn solve_reduced_modes(
    stiffness: &CsMat<f64>,
    mass: &[f64],
    constrained: &BTreeSet<usize>,
    requested_modes: usize,
) -> Result<Vec<Eigenpair>, ModalError> {
    let active_dofs: Vec<_> = (0..mass.len())
        .filter(|dof| !constrained.contains(dof))
        .collect();
    if active_dofs.is_empty() {
        return Err(ModalError::NoActiveDegreesOfFreedom);
    }

    for &dof in &active_dofs {
        if mass[dof] <= 0.0 || !mass[dof].is_finite() {
            return Err(ModalError::NonPositiveMass {
                dof,
                mass: mass[dof],
            });
        }
    }

    let mut reduced = vec![vec![0.0; active_dofs.len()]; active_dofs.len()];
    for (row_index, &row_dof) in active_dofs.iter().enumerate() {
        for (col_index, &col_dof) in active_dofs.iter().enumerate() {
            if let Some(value) = stiffness.get(row_dof, col_dof) {
                reduced[row_index][col_index] = *value / (mass[row_dof] * mass[col_dof]).sqrt();
            }
        }
    }

    let eigenpairs = jacobi_eigen_symmetric(reduced);
    let mode_count = requested_modes.min(eigenpairs.len());
    let mut modes = Vec::with_capacity(mode_count);
    for (eigenvalue, eigenvector) in eigenpairs.into_iter().take(mode_count) {
        let angular_frequency = eigenvalue.max(0.0).sqrt();
        let mut dof_values = vec![0.0; mass.len()];
        for (active_index, &dof) in active_dofs.iter().enumerate() {
            dof_values[dof] = eigenvector[active_index] / mass[dof].sqrt();
        }
        normalize(&mut dof_values);
        modes.push(Eigenpair {
            frequency_hz: angular_frequency / (2.0 * PI),
            angular_frequency,
            dof_values,
        });
    }

    Ok(modes)
}

fn lumped_mass_2d(mesh: &Mesh, density: f64, thickness: f64) -> Vec<f64> {
    let mut mass = vec![0.0; mesh.node_count() * 2];
    for triangle in mesh.triangles() {
        let [a, b, c] = triangle.nodes.map(|node| mesh.points()[node]);
        let twice_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        let nodal_mass = density * thickness * twice_area.abs() / 6.0;
        for node in triangle.nodes {
            mass[dof_index_2d(node, DisplacementComponent::X)] += nodal_mass;
            mass[dof_index_2d(node, DisplacementComponent::Y)] += nodal_mass;
        }
    }
    mass
}

fn lumped_mass_3d(mesh: &MeshTopology<3>, density: f64) -> Vec<f64> {
    let mut mass = vec![0.0; mesh.points().len() * 3];
    for cell in mesh.cells() {
        if cell.kind != ElementKind::Tet4 {
            continue;
        }
        let nodes = [cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]];
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
        let nodal_mass = density * determinant_3(jacobian).abs() / 24.0;
        for node in nodes {
            mass[dof_index_3d(node, DisplacementComponent3D::X)] += nodal_mass;
            mass[dof_index_3d(node, DisplacementComponent3D::Y)] += nodal_mass;
            mass[dof_index_3d(node, DisplacementComponent3D::Z)] += nodal_mass;
        }
    }
    mass
}

pub(crate) fn jacobi_eigen_symmetric(mut matrix: Vec<Vec<f64>>) -> Vec<(f64, Vec<f64>)> {
    let n = matrix.len();
    let mut vectors = identity(n);
    if n == 1 {
        return vec![(matrix[0][0], vec![1.0])];
    }

    for _ in 0..(100 * n * n) {
        let (p, q, max_value) = largest_off_diagonal(&matrix);
        if max_value <= 1.0e-12 {
            break;
        }

        let theta = 0.5 * (matrix[q][q] - matrix[p][p]).atan2(2.0 * matrix[p][q]);
        let cos = theta.cos();
        let sin = theta.sin();

        let mut row = 0;
        while row < n {
            if row != p && row != q {
                let row_p = matrix[row][p];
                let row_q = matrix[row][q];
                matrix[row][p] = cos * row_p - sin * row_q;
                matrix[p][row] = matrix[row][p];
                matrix[row][q] = sin * row_p + cos * row_q;
                matrix[q][row] = matrix[row][q];
            }
            row += 1;
        }

        let pp = matrix[p][p];
        let qq = matrix[q][q];
        let pq = matrix[p][q];
        matrix[p][p] = cos * cos * pp - 2.0 * sin * cos * pq + sin * sin * qq;
        matrix[q][q] = sin * sin * pp + 2.0 * sin * cos * pq + cos * cos * qq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for row in &mut vectors {
            let row_p = row[p];
            let row_q = row[q];
            row[p] = cos * row_p - sin * row_q;
            row[q] = sin * row_p + cos * row_q;
        }
    }

    let mut eigenpairs: Vec<_> = (0..n)
        .map(|index| {
            (
                matrix[index][index],
                vectors.iter().map(|row| row[index]).collect(),
            )
        })
        .collect();
    eigenpairs.sort_by(|(left, _), (right, _)| left.total_cmp(right));
    eigenpairs
}

fn largest_off_diagonal(matrix: &[Vec<f64>]) -> (usize, usize, f64) {
    let mut p = 0;
    let mut q = 1;
    let mut max_value = matrix[p][q].abs();
    for (row, row_values) in matrix.iter().enumerate() {
        for (col, item) in row_values.iter().enumerate().skip(row + 1) {
            let value = item.abs();
            if value > max_value {
                p = row;
                q = col;
                max_value = value;
            }
        }
    }
    (p, q, max_value)
}

fn identity(size: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; size]; size];
    for (index, row) in matrix.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    matrix
}

fn normalize(values: &mut [f64]) {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        for value in values {
            *value /= norm;
        }
    }
}

fn dof_index_2d(node: NodeId, component: DisplacementComponent) -> usize {
    let offset = match component {
        DisplacementComponent::X => 0,
        DisplacementComponent::Y => 1,
    };
    node * 2 + offset
}

fn dof_index_3d(node: NodeId, component: DisplacementComponent3D) -> usize {
    let offset = match component {
        DisplacementComponent3D::X => 0,
        DisplacementComponent3D::Y => 1,
        DisplacementComponent3D::Z => 2,
    };
    node * 3 + offset
}

fn determinant_3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}
