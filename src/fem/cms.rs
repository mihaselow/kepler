#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

use crate::linalg::{LinalgError, LinearSolverBackend, LinearSolverOptions, solve_linear_system};
use sprs::{CsMat, TriMat};
use std::collections::BTreeMap;

pub struct CraigBamptonReduction {
    pub k_reduced: CsMat<f64>,
    pub m_reduced: CsMat<f64>,
    pub t_cb: CsMat<f64>,
    pub boundary_dofs: Vec<usize>,
    pub internal_dofs: Vec<usize>,
}

pub fn reduce_craig_bampton(
    mass: &CsMat<f64>,
    stiffness: &CsMat<f64>,
    boundary_dofs: &[usize],
    num_modes: usize,
) -> Result<CraigBamptonReduction, LinalgError> {
    let n = stiffness.rows();
    if stiffness.cols() != n {
        return Err(LinalgError::NonSquareMatrix {
            rows: n,
            cols: stiffness.cols(),
        });
    }
    if mass.rows() != n || mass.cols() != n {
        return Err(LinalgError::DimensionMismatch {
            matrix_dim: n,
            rhs_len: mass.rows(),
        });
    }

    // 1. Sort and deduplicate boundary DOFs
    let mut b_dofs = boundary_dofs.to_vec();
    b_dofs.sort();
    b_dofs.dedup();

    // 2. Identify internal DOFs
    let mut i_dofs = Vec::new();
    for dof in 0..n {
        if b_dofs.binary_search(&dof).is_err() {
            i_dofs.push(dof);
        }
    }

    let n_b = b_dofs.len();
    let n_i = i_dofs.len();

    // Extract submatrices
    let k_ib = extract_submatrix(stiffness, &i_dofs, &b_dofs);
    let k_ii = extract_submatrix(stiffness, &i_dofs, &i_dofs);
    let m_ii = extract_submatrix(mass, &i_dofs, &i_dofs);

    // If there are no internal degrees of freedom, reduction is trivial
    if n_i == 0 {
        return Ok(CraigBamptonReduction {
            k_reduced: stiffness.clone(),
            m_reduced: mass.clone(),
            t_cb: CsMat::eye(n),
            boundary_dofs: b_dofs,
            internal_dofs: i_dofs,
        });
    }

    // 3. Compute constraint modes: Psi_c = - K_ii^-1 * K_ib
    let mut psi_c = vec![vec![0.0; n_b]; n_i];
    let solver_options = LinearSolverOptions {
        backend: LinearSolverBackend::SparseLdl,
        ..LinearSolverOptions::default()
    };

    for j in 0..n_b {
        let mut k_ib_col = vec![0.0; n_i];
        for (r, row) in k_ib.outer_iterator().enumerate() {
            for (c, &val) in row.iter() {
                if c == j {
                    k_ib_col[r] = -val;
                }
            }
        }

        let sol = solve_linear_system(&k_ii, &k_ib_col, solver_options.clone())?;
        for r in 0..n_i {
            psi_c[r][j] = sol.values[r];
        }
    }

    // 4. Compute fixed-boundary normal modes
    let k_ii_dense = to_dense(&k_ii);
    let m_ii_dense = to_dense(&m_ii);

    let l = cholesky_dense(&m_ii_dense).ok_or(LinalgError::Breakdown)?;

    // Construct A_std = L^-1 * K_ii * L^-T
    let mut v = vec![vec![0.0; n_i]; n_i];
    for j in 0..n_i {
        let mut k_col = vec![0.0; n_i];
        for i in 0..n_i {
            k_col[i] = k_ii_dense[i][j];
        }
        let v_col = forward_substitute(&l, &k_col);
        for i in 0..n_i {
            v[i][j] = v_col[i];
        }
    }

    let mut a_std = vec![vec![0.0; n_i]; n_i];
    for i in 0..n_i {
        let w_row = forward_substitute(&l, &v[i]);
        for j in 0..n_i {
            a_std[i][j] = w_row[j];
        }
    }

    let eigenpairs = crate::fem::modal::jacobi_eigen_symmetric(a_std);
    let active_modes = num_modes.min(n_i);

    let mut phi_k = vec![vec![0.0; active_modes]; n_i];
    for (m_idx, (_eigenval, y)) in eigenpairs.into_iter().take(active_modes).enumerate() {
        let phi_col = backward_substitute_transpose(&l, &y);
        for r in 0..n_i {
            phi_k[r][m_idx] = phi_col[r];
        }
    }

    // 5. Construct partitioned T_cb matrix
    let total_reduced_dofs = n_b + active_modes;
    let mut triplets_t = TriMat::new((n, total_reduced_dofs));

    // Map global DOF to partitioned index: boundary first, then internal
    let mut dof_map = vec![0; n];
    for (partition_idx, &global_idx) in b_dofs.iter().enumerate() {
        dof_map[global_idx] = partition_idx;
    }
    for (partition_idx, &global_idx) in i_dofs.iter().enumerate() {
        dof_map[global_idx] = n_b + partition_idx;
    }

    // T_cb block (0..n_b, 0..n_b) is identity
    for r in 0..n_b {
        triplets_t.add_triplet(r, r, 1.0);
    }

    // T_cb block (n_b..n, 0..n_b) is Psi_c
    for r in 0..n_i {
        for c in 0..n_b {
            triplets_t.add_triplet(n_b + r, c, psi_c[r][c]);
        }
    }

    // T_cb block (n_b..n, n_b..n_b+m) is Phi_k
    for r in 0..n_i {
        for c in 0..active_modes {
            triplets_t.add_triplet(n_b + r, n_b + c, phi_k[r][c]);
        }
    }

    let t_cb_partitioned = triplets_t.to_csr();

    // Map K and M to partitioned coordinates
    let k_partitioned = partition_matrix(stiffness, &b_dofs, &i_dofs, &dof_map);
    let m_partitioned = partition_matrix(mass, &b_dofs, &i_dofs, &dof_map);

    // K_reduced = T_cb^T * K_partitioned * T_cb
    let t_cb_partitioned_t = t_cb_partitioned.transpose_view().to_csr();
    let temp_k = &k_partitioned * &t_cb_partitioned;
    let k_reduced = &t_cb_partitioned_t * &temp_k;

    // M_reduced = T_cb^T * M_partitioned * T_cb
    let temp_m = &m_partitioned * &t_cb_partitioned;
    let m_reduced = &t_cb_partitioned_t * &temp_m;

    // Transform transformation matrix T_cb back to global physical coordinates
    // u_physical = T_cb_global * u_reduced
    // T_cb_global = P^T * T_cb_partitioned
    let mut triplets_t_global = TriMat::new((n, total_reduced_dofs));
    for (partition_row, row) in t_cb_partitioned.outer_iterator().enumerate() {
        // Find which physical DOF has this partition index
        let physical_row = if partition_row < n_b {
            b_dofs[partition_row]
        } else {
            i_dofs[partition_row - n_b]
        };
        for (c, &val) in row.iter() {
            triplets_t_global.add_triplet(physical_row, c, val);
        }
    }
    let t_cb_global = triplets_t_global.to_csr();

    Ok(CraigBamptonReduction {
        k_reduced,
        m_reduced,
        t_cb: t_cb_global,
        boundary_dofs: b_dofs,
        internal_dofs: i_dofs,
    })
}

fn extract_submatrix(
    matrix: &CsMat<f64>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> CsMat<f64> {
    let mut triplets = TriMat::new((row_indices.len(), col_indices.len()));
    let mut col_map = BTreeMap::new();
    for (local_idx, &global_idx) in col_indices.iter().enumerate() {
        col_map.insert(global_idx, local_idx);
    }

    for (local_row_idx, &global_row_idx) in row_indices.iter().enumerate() {
        if let Some(row_view) = matrix.outer_view(global_row_idx) {
            for (global_col_idx, &val) in row_view.iter() {
                if let Some(&local_col_idx) = col_map.get(&global_col_idx) {
                    triplets.add_triplet(local_row_idx, local_col_idx, val);
                }
            }
        }
    }
    triplets.to_csr()
}

fn partition_matrix(
    matrix: &CsMat<f64>,
    _boundary_dofs: &[usize],
    _internal_dofs: &[usize],
    dof_map: &[usize],
) -> CsMat<f64> {
    let n = matrix.rows();
    let mut triplets = TriMat::new((n, n));

    for (r, row) in matrix.outer_iterator().enumerate() {
        let p_row = dof_map[r];
        for (c, &val) in row.iter() {
            let p_col = dof_map[c];
            triplets.add_triplet(p_row, p_col, val);
        }
    }
    triplets.to_csr()
}

fn to_dense(matrix: &CsMat<f64>) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; matrix.cols()]; matrix.rows()];
    for (row_index, row) in matrix.outer_iterator().enumerate() {
        for (col_index, &value) in row.iter() {
            dense[row_index][col_index] += value;
        }
    }
    dense
}

fn cholesky_dense(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let val = matrix[i][i] - sum;
                if val <= 0.0 {
                    return None;
                }
                l[i][j] = val.sqrt();
            } else {
                l[i][j] = (matrix[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(l)
}

fn forward_substitute(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

fn backward_substitute_transpose(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j][i] * x[j];
        }
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_craig_bampton_mass_spring() {
        // Simple 3-DOF mass spring system:
        // M = diag(2.0, 1.5, 1.0)
        // K = [[ 3.0, -2.0,  0.0],
        //      [-2.0,  5.0, -3.0],
        //      [ 0.0, -3.0,  3.0]]
        let mut triplets_m = TriMat::new((3, 3));
        triplets_m.add_triplet(0, 0, 2.0);
        triplets_m.add_triplet(1, 1, 1.5);
        triplets_m.add_triplet(2, 2, 1.0);
        let mass = triplets_m.to_csr();

        let mut triplets_k = TriMat::new((3, 3));
        triplets_k.add_triplet(0, 0, 3.0);
        triplets_k.add_triplet(0, 1, -2.0);
        triplets_k.add_triplet(1, 0, -2.0);
        triplets_k.add_triplet(1, 1, 5.0);
        triplets_k.add_triplet(1, 2, -3.0);
        triplets_k.add_triplet(2, 0, 0.0);
        triplets_k.add_triplet(2, 1, -3.0);
        triplets_k.add_triplet(2, 2, 3.0);
        let stiffness = triplets_k.to_csr();

        // Reduce designating DOF 0 as boundary, keeping 1 fixed-boundary mode
        let boundary = vec![0];
        let reduction = reduce_craig_bampton(&mass, &stiffness, &boundary, 1).unwrap();

        // Reduced system should have size 2 (1 boundary + 1 mode)
        assert_eq!(reduction.k_reduced.rows(), 2);
        assert_eq!(reduction.m_reduced.rows(), 2);
        assert_eq!(reduction.t_cb.rows(), 3);
        assert_eq!(reduction.t_cb.cols(), 2);

        // Check stiffness matrix symmetry
        assert!(
            (reduction.k_reduced.get(0, 1).unwrap() - reduction.k_reduced.get(1, 0).unwrap()).abs()
                < 1e-10
        );
    }
}
