use crate::fem::dof::DOFManager;
use crate::mesh::NodeId;
use sprs::{CsMat, TriMat};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct MPCTerm {
    pub node: NodeId,
    pub field: String,
    pub coefficient: f64,
}

impl MPCTerm {
    pub fn new(node: NodeId, field: &str, coefficient: f64) -> Self {
        Self {
            node,
            field: field.to_string(),
            coefficient,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MPCConstraint {
    pub terms: Vec<MPCTerm>,
    pub value: f64,
}

impl MPCConstraint {
    pub fn new(terms: Vec<MPCTerm>, value: f64) -> Self {
        Self { terms, value }
    }
}

/// Applies MPC constraints to a reduced linear system using the penalty method.
pub fn apply_mpc_penalty(
    reduced_stiffness: &CsMat<f64>,
    reduced_rhs: &[f64],
    active_dofs: &[usize],
    dof_manager: &DOFManager,
    mpc_constraints: &[MPCConstraint],
    dirichlet_constraints: &BTreeMap<usize, f64>,
    penalty: f64,
) -> (CsMat<f64>, Vec<f64>) {
    let n = reduced_stiffness.rows();
    let mut triplets = TriMat::new((n, n));
    for (row_index, row) in reduced_stiffness.outer_iterator().enumerate() {
        for (col_index, &value) in row.iter() {
            triplets.add_triplet(row_index, col_index, value);
        }
    }

    let mut rhs = reduced_rhs.to_vec();

    // Map from global DOF index to its position index in reduced system
    let mut active_map = BTreeMap::new();
    for (reduced_idx, &dof) in active_dofs.iter().enumerate() {
        active_map.insert(dof, reduced_idx);
    }

    for mpc in mpc_constraints {
        let mut active_terms = Vec::new();
        let mut rhs_val = mpc.value;

        for term in &mpc.terms {
            if let Some(global_dof) = dof_manager.get_eq_index(term.node, &term.field) {
                if let Some(&reduced_idx) = active_map.get(&global_dof) {
                    active_terms.push((reduced_idx, term.coefficient));
                } else if let Some(&fixed_val) = dirichlet_constraints.get(&global_dof) {
                    rhs_val -= term.coefficient * fixed_val;
                }
            }
        }

        // Add penalty: (K + alpha * a * a^T) u = F + alpha * rhs_val * a
        for &(r_idx, r_coeff) in &active_terms {
            rhs[r_idx] += penalty * rhs_val * r_coeff;
            for &(c_idx, c_coeff) in &active_terms {
                triplets.add_triplet(r_idx, c_idx, penalty * r_coeff * c_coeff);
            }
        }
    }

    (triplets.to_csr(), rhs)
}

/// Applies MPC constraints by augmenting the reduced linear system with Lagrange multipliers.
pub fn apply_mpc_lagrange(
    reduced_stiffness: &CsMat<f64>,
    reduced_rhs: &[f64],
    active_dofs: &[usize],
    dof_manager: &DOFManager,
    mpc_constraints: &[MPCConstraint],
    dirichlet_constraints: &BTreeMap<usize, f64>,
) -> (CsMat<f64>, Vec<f64>) {
    let n = reduced_stiffness.rows();
    let m = mpc_constraints.len();
    let total_size = n + m;

    let mut triplets = TriMat::new((total_size, total_size));
    for (row_index, row) in reduced_stiffness.outer_iterator().enumerate() {
        for (col_index, &value) in row.iter() {
            triplets.add_triplet(row_index, col_index, value);
        }
    }

    let mut rhs = vec![0.0; total_size];
    rhs[..n].copy_from_slice(reduced_rhs);

    // Map from global DOF index to its position index in reduced system
    let mut active_map = BTreeMap::new();
    for (reduced_idx, &dof) in active_dofs.iter().enumerate() {
        active_map.insert(dof, reduced_idx);
    }

    for (mpc_idx, mpc) in mpc_constraints.iter().enumerate() {
        let mut rhs_val = mpc.value;
        let lam_idx = n + mpc_idx;

        for term in &mpc.terms {
            if let Some(global_dof) = dof_manager.get_eq_index(term.node, &term.field) {
                if let Some(&reduced_idx) = active_map.get(&global_dof) {
                    triplets.add_triplet(lam_idx, reduced_idx, term.coefficient);
                    triplets.add_triplet(reduced_idx, lam_idx, term.coefficient);
                } else if let Some(&fixed_val) = dirichlet_constraints.get(&global_dof) {
                    rhs_val -= term.coefficient * fixed_val;
                }
            }
        }
        rhs[lam_idx] = rhs_val;
    }

    (triplets.to_csr(), rhs)
}

/// Splits the solution vector of a Lagrange-augmented system into the active displacements and Lagrange multipliers.
pub fn split_lagrange_solution(solution: &[f64], num_active_dofs: usize) -> (Vec<f64>, Vec<f64>) {
    let active = solution[..num_active_dofs].to_vec();
    let multipliers = solution[num_active_dofs..].to_vec();
    (active, multipliers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{LinearSolverBackend, LinearSolverOptions, solve_linear_system};

    #[test]
    fn test_mpc_penalty_and_lagrange_methods() {
        // Construct the 3-node spring system:
        // Node 0, 1, 2. Field "u" on each.
        let mut dof_manager = DOFManager::new();
        let dof0 = dof_manager.register_dof(0, "u");
        let dof1 = dof_manager.register_dof(1, "u");
        let dof2 = dof_manager.register_dof(2, "u");

        assert_eq!(dof0, 0);
        assert_eq!(dof1, 1);
        assert_eq!(dof2, 2);

        // Global stiffness
        let mut triplets = TriMat::new((3, 3));
        triplets.add_triplet(0, 0, 100.0);
        triplets.add_triplet(0, 1, -100.0);
        triplets.add_triplet(1, 0, -100.0);
        triplets.add_triplet(1, 1, 200.0);
        triplets.add_triplet(1, 2, -100.0);
        triplets.add_triplet(2, 1, -100.0);
        triplets.add_triplet(2, 2, 100.0);
        let _k_global: CsMat<f64> = triplets.to_csr();

        // Dirichlet BC: u_0 = 0.0
        let mut dirichlet = BTreeMap::new();
        dirichlet.insert(0, 0.0);

        let active_dofs = vec![1, 2];
        let mut reduced_stiffness = TriMat::new((2, 2));
        reduced_stiffness.add_triplet(0, 0, 200.0);
        reduced_stiffness.add_triplet(0, 1, -100.0);
        reduced_stiffness.add_triplet(1, 0, -100.0);
        reduced_stiffness.add_triplet(1, 1, 100.0);
        let k_reduced: CsMat<f64> = reduced_stiffness.to_csr();

        // F = [0, 10]
        let f_reduced = vec![0.0, 10.0];

        // MPC: u_2 - u_1 = 0.05
        let mpc = MPCConstraint::new(
            vec![MPCTerm::new(2, "u", 1.0), MPCTerm::new(1, "u", -1.0)],
            0.05,
        );

        // Solve via Lagrange
        let (k_lag, f_lag) = apply_mpc_lagrange(
            &k_reduced,
            &f_reduced,
            &active_dofs,
            &dof_manager,
            std::slice::from_ref(&mpc),
            &dirichlet,
        );

        let options = LinearSolverOptions {
            backend: LinearSolverBackend::DenseDirect,
            ..LinearSolverOptions::default()
        };

        let res_lag = solve_linear_system(&k_lag, &f_lag, options.clone()).unwrap();
        let (u_active_lag, lam) = split_lagrange_solution(&res_lag.values, 2);

        // Analytical solution: u_1 = 0.1, u_2 = 0.15, lam = 5.0
        assert!((u_active_lag[0] - 0.1).abs() < 1e-9);
        assert!((u_active_lag[1] - 0.15).abs() < 1e-9);
        assert!((lam[0] - 5.0).abs() < 1e-9);

        // Solve via Penalty (penalty = 1e5)
        let (k_pen, f_pen) = apply_mpc_penalty(
            &k_reduced,
            &f_reduced,
            &active_dofs,
            &dof_manager,
            &[mpc],
            &dirichlet,
            1e5,
        );

        let res_pen = solve_linear_system(&k_pen, &f_pen, options).unwrap();
        let u_active_pen = &res_pen.values;

        assert!((u_active_pen[0] - 0.1).abs() < 1e-3);
        assert!((u_active_pen[1] - 0.15).abs() < 1e-3);
    }
}
