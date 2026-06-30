use std::{collections::BTreeSet, f64::consts::PI};

use sprs::CsMat;
use thiserror::Error;

use crate::fem::element::Element;
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

    let n_active = active_dofs.len();
    let mode_count = requested_modes.min(n_active);

    // For large systems, use the sparse shift-invert Lanczos eigensolver.
    // For small systems the dense Jacobi path has lower overhead.
    const LANCZOS_THRESHOLD: usize = 300;
    if n_active >= LANCZOS_THRESHOLD {
        return solve_reduced_modes_lanczos(stiffness, mass, &active_dofs, mode_count);
    }

    // --- Dense Jacobi path (small systems) ---
    let mut reduced = vec![vec![0.0; n_active]; n_active];
    for (row_index, &row_dof) in active_dofs.iter().enumerate() {
        for (col_index, &col_dof) in active_dofs.iter().enumerate() {
            if let Some(value) = stiffness.get(row_dof, col_dof) {
                reduced[row_index][col_index] = *value / (mass[row_dof] * mass[col_dof]).sqrt();
            }
        }
    }

    let eigenpairs = jacobi_eigen_symmetric(reduced);
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

/// Sparse Lanczos path for large modal problems.
///
/// Builds a reduced stiffness and mass from the active DOFs, then calls
/// `solve_lanczos_modes` with a small spectral shift to handle near-zero
/// rigid-body-mode eigenvalues gracefully.
fn solve_reduced_modes_lanczos(
    stiffness: &CsMat<f64>,
    mass: &[f64],
    active_dofs: &[usize],
    mode_count: usize,
) -> Result<Vec<Eigenpair>, ModalError> {
    use crate::linalg::solve_lanczos_modes;
    use sprs::TriMat;

    let n_active = active_dofs.len();

    // Map global DOF → reduced DOF index.
    let mut global_to_reduced = vec![usize::MAX; mass.len()];
    for (reduced_idx, &global_dof) in active_dofs.iter().enumerate() {
        global_to_reduced[global_dof] = reduced_idx;
    }

    // Extract reduced stiffness matrix.
    let mut k_tri = TriMat::new((n_active, n_active));
    for (row_global, row_vec) in stiffness.outer_iterator().enumerate() {
        let row_r = global_to_reduced[row_global];
        if row_r == usize::MAX {
            continue;
        }
        for (col_global, &val) in row_vec.iter() {
            let col_r = global_to_reduced[col_global];
            if col_r != usize::MAX {
                k_tri.add_triplet(row_r, col_r, val);
            }
        }
    }
    let k_reduced: sprs::CsMat<f64> = k_tri.to_csr();

    // Reduced lumped mass vector.
    let mass_reduced: Vec<f64> = active_dofs.iter().map(|&d| mass[d]).collect();

    // Small shift to regularise near-zero modes (e.g. from floating structure).
    // Estimate a shift as ~1e-5 * max diagonal of K to stay well below the
    // smallest physical mode.
    let max_k_diag = active_dofs
        .iter()
        .filter_map(|&d| stiffness.get(d, d).copied())
        .fold(0.0f64, f64::max);
    let shift = if max_k_diag > 0.0 {
        1e-5 * max_k_diag
    } else {
        0.0
    };

    let max_iters = (10 * mode_count).max(mode_count + 20).min(n_active);
    let result = solve_lanczos_modes(
        &k_reduced,
        &mass_reduced,
        mode_count,
        shift,
        max_iters,
        1e-8,
    )
    .map_err(|_| ModalError::NoActiveDegreesOfFreedom)?;

    let mut modes = Vec::with_capacity(result.eigenvalues.len());
    for (lambda, reduced_vec) in result.eigenvalues.into_iter().zip(result.eigenvectors) {
        let angular_frequency = lambda.max(0.0).sqrt();
        // Reconstruct global DOF vector (constrained DOFs remain zero).
        let mut dof_values = vec![0.0; mass.len()];
        for (reduced_idx, &global_dof) in active_dofs.iter().enumerate() {
            dof_values[global_dof] = reduced_vec[reduced_idx];
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
    use rayon::prelude::*;
    let triangles = mesh.triangles();
    let contributions: Vec<Vec<(usize, f64)>> = triangles
        .par_iter()
        .map(|triangle| {
            let [a, b, c] = triangle.nodes.map(|node| mesh.points()[node]);
            let twice_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
            let nodal_mass = density * thickness * twice_area.abs() / 6.0;
            let mut elem_mass = Vec::with_capacity(6);
            for node in triangle.nodes {
                elem_mass.push((dof_index_2d(node, DisplacementComponent::X), nodal_mass));
                elem_mass.push((dof_index_2d(node, DisplacementComponent::Y), nodal_mass));
            }
            elem_mass
        })
        .collect();

    let mut mass = vec![0.0; mesh.node_count() * 2];
    for elem_mass in contributions {
        for (dof, val) in elem_mass {
            mass[dof] += val;
        }
    }
    mass
}

fn lumped_mass_3d(mesh: &MeshTopology<3>, density: f64) -> Vec<f64> {
    use rayon::prelude::*;
    let cells = mesh.cells();
    let contributions: Vec<Vec<(usize, f64)>> = cells
        .par_iter()
        .filter_map(|cell| match cell.kind {
            ElementKind::Tet4 => {
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
                let mut elem_mass = Vec::with_capacity(12);
                for node in nodes {
                    elem_mass.push((dof_index_3d(node, DisplacementComponent3D::X), nodal_mass));
                    elem_mass.push((dof_index_3d(node, DisplacementComponent3D::Y), nodal_mass));
                    elem_mass.push((dof_index_3d(node, DisplacementComponent3D::Z), nodal_mass));
                }
                Some(elem_mass)
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
                let node_coords: Vec<_> = nodes.iter().map(|&n| mesh.points()[n]).collect();
                let el = crate::fem::elasticity::ElasticityHex8 { nodes: &nodes };
                let local_m = el.local_mass(&node_coords, density, true).ok()?;
                let mut elem_mass = Vec::with_capacity(24);
                for i in 0..8 {
                    let node = nodes[i];
                    elem_mass.push((
                        dof_index_3d(node, DisplacementComponent3D::X),
                        local_m[3 * i][3 * i],
                    ));
                    elem_mass.push((
                        dof_index_3d(node, DisplacementComponent3D::Y),
                        local_m[3 * i + 1][3 * i + 1],
                    ));
                    elem_mass.push((
                        dof_index_3d(node, DisplacementComponent3D::Z),
                        local_m[3 * i + 2][3 * i + 2],
                    ));
                }
                Some(elem_mass)
            }
            _ => None,
        })
        .collect();

    let mut mass = vec![0.0; mesh.points().len() * 3];
    for elem_mass in contributions {
        for (dof, val) in elem_mass {
            mass[dof] += val;
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
