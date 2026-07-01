use std::sync::Arc;
use sprs::{CsMat, TriMat};

use crate::{
    fem::{
        contact::{
            penalty::assemble_penalty_contact,
            solve::{find_contact_pairs, ContactProblem},
        },
        material::{MaterialModel, MaterialState},
    },
    linalg::{LinalgError, LinearSolverOptions, axpy, norm, solve_linear_system},
    mesh::{ElementKind, Mesh},
};

pub struct NonlinearContinuumAssembly {
    pub mesh: Mesh,
    pub thickness: f64,
    pub is_plane_strain: bool,
    pub material: Arc<dyn MaterialModel>,
    pub external_forces: Vec<(usize, usize, f64)>, // (node, component, val)
    pub dirichlet_boundary: Vec<(usize, usize, f64)>, // (node, component, val)
    pub contact: Option<ContactProblem>,
    pub num_dofs: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearContinuumSolverOptions {
    pub num_steps: usize,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub linear_solver: LinearSolverOptions,
}

impl Default for NonlinearContinuumSolverOptions {
    fn default() -> Self {
        Self {
            num_steps: 10,
            max_iterations: 20,
            tolerance: 1.0e-5,
            linear_solver: LinearSolverOptions::default(),
        }
    }
}

pub struct NonlinearContinuumResult {
    pub displacements_history: Vec<Vec<f64>>,
    pub lambda_history: Vec<f64>,
    pub nodal_stress_history: Vec<Vec<[f64; 6]>>,
}

impl NonlinearContinuumAssembly {
    #[allow(clippy::needless_range_loop)]
    pub fn evaluate_system(
        &self,
        u: &[f64],
        states_old: &[Vec<MaterialState>],
    ) -> (Vec<f64>, CsMat<f64>, Vec<Vec<MaterialState>>) {
        let n_elem = self.mesh.cells().len();
        let mut f_int = vec![0.0; self.num_dofs];
        let mut triplets = TriMat::new((self.num_dofs, self.num_dofs));
        let mut states_new = Vec::with_capacity(n_elem);

        let mut bc_set = std::collections::BTreeSet::new();
        for &(node, comp, _) in &self.dirichlet_boundary {
            bc_set.insert(node * 2 + comp);
        }

        for (el_idx, cell) in self.mesh.cells().iter().enumerate() {
            match cell.kind {
                ElementKind::Tri3 => {
                    let n0 = cell.nodes[0];
                    let n1 = cell.nodes[1];
                    let n2 = cell.nodes[2];
                    let p0 = self.mesh.points()[n0];
                    let p1 = self.mesh.points()[n1];
                    let p2 = self.mesh.points()[n2];

                    let twice_area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
                    let area = 0.5 * twice_area.abs();

                    let b1 = p1.y - p2.y;
                    let b2 = p2.y - p0.y;
                    let b3 = p0.y - p1.y;
                    let c1 = p2.x - p1.x;
                    let c2 = p0.x - p2.x;
                    let c3 = p1.x - p0.x;

                    let inv2a = 1.0 / twice_area;
                    let b_mat = [
                        [b1 * inv2a, 0.0, b2 * inv2a, 0.0, b3 * inv2a, 0.0],
                        [0.0, c1 * inv2a, 0.0, c2 * inv2a, 0.0, c3 * inv2a],
                        [c1 * inv2a, b1 * inv2a, c2 * inv2a, b2 * inv2a, c3 * inv2a, b3 * inv2a],
                    ];

                    let u_e = [
                        u[n0 * 2],
                        u[n0 * 2 + 1],
                        u[n1 * 2],
                        u[n1 * 2 + 1],
                        u[n2 * 2],
                        u[n2 * 2 + 1],
                    ];

                    let mut eps_2d = [0.0; 3];
                    for i in 0..3 {
                        for j in 0..6 {
                            eps_2d[i] += b_mat[i][j] * u_e[j];
                        }
                    }

                    let mut strain_3d = [0.0; 6];
                    strain_3d[0] = eps_2d[0];
                    strain_3d[1] = eps_2d[1];
                    if self.is_plane_strain {
                        strain_3d[2] = 0.0;
                    } else {
                        let nu = 0.3;
                        strain_3d[2] = -nu / (1.0 - nu) * (eps_2d[0] + eps_2d[1]);
                    }
                    strain_3d[3] = eps_2d[2];

                    let state_old_g = &states_old[el_idx][0];
                    let (stress_3d, state_new_g, c_3d) = self.material.integrate(&strain_3d, state_old_g);
                    states_new.push(vec![state_new_g]);

                    let stress_2d = [stress_3d[0], stress_3d[1], stress_3d[3]];

                    let factor = area * self.thickness;
                    let mut f_local = [0.0; 6];
                    for i in 0..6 {
                        for alpha in 0..3 {
                            f_local[i] += b_mat[alpha][i] * stress_2d[alpha] * factor;
                        }
                    }

                    let global_dofs = [n0 * 2, n0 * 2 + 1, n1 * 2, n1 * 2 + 1, n2 * 2, n2 * 2 + 1];
                    for i in 0..6 {
                        f_int[global_dofs[i]] += f_local[i];
                    }

                    let c_2d = [
                        [c_3d[0][0], c_3d[0][1], c_3d[0][3]],
                        [c_3d[1][0], c_3d[1][1], c_3d[1][3]],
                        [c_3d[3][0], c_3d[3][1], c_3d[3][3]],
                    ];

                    let mut k_local = [[0.0; 6]; 6];
                    for i in 0..6 {
                        for j in 0..6 {
                            let mut val = 0.0;
                            for alpha in 0..3 {
                                for beta in 0..3 {
                                    val += b_mat[alpha][i] * c_2d[alpha][beta] * b_mat[beta][j] * factor;
                                }
                            }
                            k_local[i][j] = val;
                        }
                    }

                    for i in 0..6 {
                        let r = global_dofs[i];
                        if bc_set.contains(&r) {
                            continue;
                        }
                        for j in 0..6 {
                            let c = global_dofs[j];
                            if bc_set.contains(&c) {
                                continue;
                            }
                            triplets.add_triplet(r, c, k_local[i][j]);
                        }
                    }
                }
                ElementKind::Quad4 => {
                    let n0 = cell.nodes[0];
                    let n1 = cell.nodes[1];
                    let n2 = cell.nodes[2];
                    let n3 = cell.nodes[3];
                    let p0 = self.mesh.points()[n0];
                    let p1 = self.mesh.points()[n1];
                    let p2 = self.mesh.points()[n2];
                    let p3 = self.mesh.points()[n3];

                    let node_coords = [p0, p1, p2, p3];

                    let g_pts = [-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()];

                    let mut f_local = [0.0; 8];
                    let mut k_local = [[0.0; 8]; 8];
                    let mut el_states = Vec::with_capacity(4);

                    let u_e = [
                        u[n0 * 2],
                        u[n0 * 2 + 1],
                        u[n1 * 2],
                        u[n1 * 2 + 1],
                        u[n2 * 2],
                        u[n2 * 2 + 1],
                        u[n3 * 2],
                        u[n3 * 2 + 1],
                    ];

                    let mut gp_idx = 0;
                    for &xi in &g_pts {
                        for &eta in &g_pts {
                            let dn_dxi = [
                                -0.25 * (1.0 - eta),
                                0.25 * (1.0 - eta),
                                0.25 * (1.0 + eta),
                                -0.25 * (1.0 + eta),
                            ];
                            let dn_deta = [
                                -0.25 * (1.0 - xi),
                                -0.25 * (1.0 + xi),
                                0.25 * (1.0 + xi),
                                0.25 * (1.0 - xi),
                            ];

                            let mut j11 = 0.0;
                            let mut j12 = 0.0;
                            let mut j21 = 0.0;
                            let mut j22 = 0.0;
                            for a in 0..4 {
                                j11 += dn_dxi[a] * node_coords[a].x;
                                j12 += dn_dxi[a] * node_coords[a].y;
                                j21 += dn_deta[a] * node_coords[a].x;
                                j22 += dn_deta[a] * node_coords[a].y;
                            }

                            let det_j = j11 * j22 - j12 * j21;
                            let inv_j11 = j22 / det_j;
                            let inv_j12 = -j12 / det_j;
                            let inv_j21 = -j21 / det_j;
                            let inv_j22 = j11 / det_j;

                            let mut dn_dx = [0.0; 4];
                            let mut dn_dy = [0.0; 4];
                            for a in 0..4 {
                                dn_dx[a] = inv_j11 * dn_dxi[a] + inv_j12 * dn_deta[a];
                                dn_dy[a] = inv_j21 * dn_dxi[a] + inv_j22 * dn_deta[a];
                            }

                            let b_mat = [
                                [dn_dx[0], 0.0, dn_dx[1], 0.0, dn_dx[2], 0.0, dn_dx[3], 0.0],
                                [0.0, dn_dy[0], 0.0, dn_dy[1], 0.0, dn_dy[2], 0.0, dn_dy[3]],
                                [dn_dy[0], dn_dx[0], dn_dy[1], dn_dx[1], dn_dy[2], dn_dx[2], dn_dy[3], dn_dx[3]],
                            ];

                            let mut eps_2d = [0.0; 3];
                            for i in 0..3 {
                                for j in 0..8 {
                                    eps_2d[i] += b_mat[i][j] * u_e[j];
                                }
                            }

                            let mut strain_3d = [0.0; 6];
                            strain_3d[0] = eps_2d[0];
                            strain_3d[1] = eps_2d[1];
                            if self.is_plane_strain {
                                strain_3d[2] = 0.0;
                            } else {
                                let nu = 0.3;
                                strain_3d[2] = -nu / (1.0 - nu) * (eps_2d[0] + eps_2d[1]);
                            }
                            strain_3d[3] = eps_2d[2];

                            let state_old_g = &states_old[el_idx][gp_idx];
                            let (stress_3d, state_new_g, c_3d) = self.material.integrate(&strain_3d, state_old_g);
                            el_states.push(state_new_g);

                            let stress_2d = [stress_3d[0], stress_3d[1], stress_3d[3]];
                            let weight_factor = det_j.abs() * self.thickness;

                            for i in 0..8 {
                                for alpha in 0..3 {
                                    f_local[i] += b_mat[alpha][i] * stress_2d[alpha] * weight_factor;
                                }
                            }

                            let c_2d = [
                                [c_3d[0][0], c_3d[0][1], c_3d[0][3]],
                                [c_3d[1][0], c_3d[1][1], c_3d[1][3]],
                                [c_3d[3][0], c_3d[3][1], c_3d[3][3]],
                            ];

                            for i in 0..8 {
                                for j in 0..8 {
                                    let mut val = 0.0;
                                    for alpha in 0..3 {
                                        for beta in 0..3 {
                                            val += b_mat[alpha][i] * c_2d[alpha][beta] * b_mat[beta][j] * weight_factor;
                                        }
                                    }
                                    k_local[i][j] += val;
                                }
                            }

                            gp_idx += 1;
                        }
                    }
                    states_new.push(el_states);

                    let global_dofs = [
                        n0 * 2, n0 * 2 + 1,
                        n1 * 2, n1 * 2 + 1,
                        n2 * 2, n2 * 2 + 1,
                        n3 * 2, n3 * 2 + 1,
                    ];

                    for i in 0..8 {
                        f_int[global_dofs[i]] += f_local[i];
                    }

                    for i in 0..8 {
                        let r = global_dofs[i];
                        if bc_set.contains(&r) {
                            continue;
                        }
                        for j in 0..8 {
                            let c = global_dofs[j];
                            if bc_set.contains(&c) {
                                continue;
                            }
                            triplets.add_triplet(r, c, k_local[i][j]);
                        }
                    }
                }
                _ => panic!("Unsupported element type for nonlinear continuum assembly"),
            }
        }

        for &dof in &bc_set {
            triplets.add_triplet(dof, dof, 1.0);
        }

        (f_int, triplets.to_csr(), states_new)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn recover_nodal_stresses(&self, u: &[f64], states: &[Vec<MaterialState>]) -> Vec<[f64; 6]> {
        let node_count = self.mesh.node_count();
        let mut nodal_stress = vec![[0.0; 6]; node_count];
        let mut counts = vec![0.0; node_count];

        for (el_idx, cell) in self.mesh.cells().iter().enumerate() {
            match cell.kind {
                ElementKind::Tri3 => {
                    let n0 = cell.nodes[0];
                    let n1 = cell.nodes[1];
                    let n2 = cell.nodes[2];
                    let p0 = self.mesh.points()[n0];
                    let p1 = self.mesh.points()[n1];
                    let p2 = self.mesh.points()[n2];

                    let twice_area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
                    let b1 = p1.y - p2.y;
                    let b2 = p2.y - p0.y;
                    let b3 = p0.y - p1.y;
                    let c1 = p2.x - p1.x;
                    let c2 = p0.x - p2.x;
                    let c3 = p1.x - p0.x;

                    let inv2a = 1.0 / twice_area;
                    let b_mat = [
                        [b1 * inv2a, 0.0, b2 * inv2a, 0.0, b3 * inv2a, 0.0],
                        [0.0, c1 * inv2a, 0.0, c2 * inv2a, 0.0, c3 * inv2a],
                        [c1 * inv2a, b1 * inv2a, c2 * inv2a, b2 * inv2a, c3 * inv2a, b3 * inv2a],
                    ];

                    let u_e = [
                        u[n0 * 2],
                        u[n0 * 2 + 1],
                        u[n1 * 2],
                        u[n1 * 2 + 1],
                        u[n2 * 2],
                        u[n2 * 2 + 1],
                    ];

                    let mut eps_2d = [0.0; 3];
                    for i in 0..3 {
                        for j in 0..6 {
                            eps_2d[i] += b_mat[i][j] * u_e[j];
                        }
                    }

                    let mut strain_3d = [0.0; 6];
                    strain_3d[0] = eps_2d[0];
                    strain_3d[1] = eps_2d[1];
                    if self.is_plane_strain {
                        strain_3d[2] = 0.0;
                    } else {
                        let nu = 0.3;
                        strain_3d[2] = -nu / (1.0 - nu) * (eps_2d[0] + eps_2d[1]);
                    }
                    strain_3d[3] = eps_2d[2];

                    let state_old_g = &states[el_idx][0];
                    let (stress_3d, _, _) = self.material.integrate(&strain_3d, state_old_g);

                    for &n in &cell.nodes {
                        for i in 0..6 {
                            nodal_stress[n][i] += stress_3d[i];
                        }
                        counts[n] += 1.0;
                    }
                }
                ElementKind::Quad4 => {
                    let n0 = cell.nodes[0];
                    let n1 = cell.nodes[1];
                    let n2 = cell.nodes[2];
                    let n3 = cell.nodes[3];
                    let p0 = self.mesh.points()[n0];
                    let p1 = self.mesh.points()[n1];
                    let p2 = self.mesh.points()[n2];
                    let p3 = self.mesh.points()[n3];

                    let node_coords = [p0, p1, p2, p3];
                    let u_e = [
                        u[n0 * 2],
                        u[n0 * 2 + 1],
                        u[n1 * 2],
                        u[n1 * 2 + 1],
                        u[n2 * 2],
                        u[n2 * 2 + 1],
                        u[n3 * 2],
                        u[n3 * 2 + 1],
                    ];

                    let xi = 0.0;
                    let eta = 0.0;
                    let dn_dxi = [
                        -0.25 * (1.0 - eta),
                        0.25 * (1.0 - eta),
                        0.25 * (1.0 + eta),
                        -0.25 * (1.0 + eta),
                    ];
                    let dn_deta = [
                        -0.25 * (1.0 - xi),
                        -0.25 * (1.0 + xi),
                        0.25 * (1.0 + xi),
                        0.25 * (1.0 - xi),
                    ];

                    let mut j11 = 0.0;
                    let mut j12 = 0.0;
                    let mut j21 = 0.0;
                    let mut j22 = 0.0;
                    for a in 0..4 {
                        j11 += dn_dxi[a] * node_coords[a].x;
                        j12 += dn_dxi[a] * node_coords[a].y;
                        j21 += dn_deta[a] * node_coords[a].x;
                        j22 += dn_deta[a] * node_coords[a].y;
                    }

                    let det_j = j11 * j22 - j12 * j21;
                    let inv_j11 = j22 / det_j;
                    let inv_j12 = -j12 / det_j;
                    let inv_j21 = -j21 / det_j;
                    let inv_j22 = j11 / det_j;

                    let mut dn_dx = [0.0; 4];
                    let mut dn_dy = [0.0; 4];
                    for a in 0..4 {
                        dn_dx[a] = inv_j11 * dn_dxi[a] + inv_j12 * dn_deta[a];
                        dn_dy[a] = inv_j21 * dn_dxi[a] + inv_j22 * dn_deta[a];
                    }

                    let b_mat = [
                        [dn_dx[0], 0.0, dn_dx[1], 0.0, dn_dx[2], 0.0, dn_dx[3], 0.0],
                        [0.0, dn_dy[0], 0.0, dn_dy[1], 0.0, dn_dy[2], 0.0, dn_dy[3]],
                        [dn_dy[0], dn_dx[0], dn_dy[1], dn_dx[1], dn_dy[2], dn_dx[2], dn_dy[3], dn_dx[3]],
                    ];

                    let mut eps_2d = [0.0; 3];
                    for i in 0..3 {
                        for j in 0..8 {
                            eps_2d[i] += b_mat[i][j] * u_e[j];
                        }
                    }

                    let mut strain_3d = [0.0; 6];
                    strain_3d[0] = eps_2d[0];
                    strain_3d[1] = eps_2d[1];
                    if self.is_plane_strain {
                        strain_3d[2] = 0.0;
                    } else {
                        let nu = 0.3;
                        strain_3d[2] = -nu / (1.0 - nu) * (eps_2d[0] + eps_2d[1]);
                    }
                    strain_3d[3] = eps_2d[2];

                    let state_old_g = &states[el_idx][0];
                    let (stress_3d, _, _) = self.material.integrate(&strain_3d, state_old_g);

                    for &n in &cell.nodes {
                        for i in 0..6 {
                            nodal_stress[n][i] += stress_3d[i];
                        }
                        counts[n] += 1.0;
                    }
                }
                _ => {}
            }
        }

        for n in 0..node_count {
            if counts[n] > 0.0 {
                for i in 0..6 {
                    nodal_stress[n][i] /= counts[n];
                }
            }
        }

        nodal_stress
    }
}

pub fn solve_nonlinear_continuum(
    assembly: &NonlinearContinuumAssembly,
    options: NonlinearContinuumSolverOptions,
) -> Result<NonlinearContinuumResult, LinalgError> {
    let n = assembly.num_dofs;
    let mut u = vec![0.0; n];

    let mut states_converged = Vec::new();
    for cell in assembly.mesh.cells() {
        let gps = match cell.kind {
            ElementKind::Tri3 => 1,
            ElementKind::Quad4 => 4,
            _ => panic!("Unsupported element type"),
        };
        states_converged.push(vec![MaterialState::default(); gps]);
    }

    let mut displacements_history = vec![u.clone()];
    let mut lambda_history = vec![0.0];
    let mut nodal_stress_history = vec![assembly.recover_nodal_stresses(&u, &states_converged)];

    let mut f_ext = vec![0.0; n];
    for &(node, comp, val) in &assembly.external_forces {
        f_ext[node * 2 + comp] += val;
    }

    for step in 1..=options.num_steps {
        let lambda = step as f64 / options.num_steps as f64;

        let mut converged = false;
        let mut u_iter = u.clone();
        let mut _states_iter = states_converged.clone();

        for _iter in 1..=options.max_iterations {
            let (mut f_int, mut kt, states_next) = assembly.evaluate_system(&u_iter, &states_converged);

            if let Some(contact) = &assembly.contact {
                let pairs = find_contact_pairs(
                    &assembly.mesh,
                    &contact.master_segments,
                    &contact.slave_nodes,
                    &u_iter,
                );
                let ref_coords: Vec<[f64; 2]> = assembly
                    .mesh
                    .points()
                    .iter()
                    .map(|p| [p.x, p.y])
                    .collect();
                let bc_set: std::collections::BTreeSet<usize> = assembly
                    .dirichlet_boundary
                    .iter()
                    .map(|&(node, comp, _)| node * 2 + comp)
                    .collect();
                let mut f_contact = vec![0.0; n];
                let mut contact_tri = TriMat::new((n, n));
                assemble_penalty_contact(
                    &ref_coords,
                    &u_iter,
                    &pairs,
                    contact.penalty,
                    &mut f_contact,
                    &mut contact_tri,
                    &bc_set,
                );
                for i in 0..n {
                    f_int[i] -= f_contact[i];
                }
                let k_contact: CsMat<f64> = contact_tri.to_csr();
                let mut merged = TriMat::new((n, n));
                for (&val, (i, j)) in kt.iter() {
                    merged.add_triplet(i, j, val);
                }
                for (&val, (i, j)) in k_contact.iter() {
                    merged.add_triplet(i, j, val);
                }
                kt = merged.to_csr();
            }

            let mut r = vec![0.0; n];
            for i in 0..n {
                r[i] = f_int[i] - lambda * f_ext[i];
            }

            for &(node, comp, val) in &assembly.dirichlet_boundary {
                let dof = node * 2 + comp;
                r[dof] = u_iter[dof] - lambda * val;
            }

            let r_norm = norm(&r);
            if r_norm <= options.tolerance {
                converged = true;
                u = u_iter;
                states_converged = states_next;
                break;
            }

            let neg_r: Vec<_> = r.iter().map(|&x| -x).collect();
            let du = solve_linear_system(&kt, &neg_r, options.linear_solver.clone())?.values;

            axpy(1.0, &du, &mut u_iter);
            _states_iter = states_next;
        }

        if !converged {
            return Err(LinalgError::NonlinearNonConverged {
                iterations: options.max_iterations,
                residual_norm: 0.0,
            });
        }

        displacements_history.push(u.clone());
        lambda_history.push(lambda);
        nodal_stress_history.push(assembly.recover_nodal_stresses(&u, &states_converged));
    }

    Ok(NonlinearContinuumResult {
        displacements_history,
        lambda_history,
        nodal_stress_history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::material::plasticity::J2PlasticMaterial;
    use crate::mesh::Point2;

    #[test]
    fn test_nonlinear_continuum_j2_uniaxial() {
        // Uniaxial tension test of a single Quad4 element under plane strain
        // Nodes: (0,0), (1,0), (1,1), (0,1)
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        let cells = vec![crate::mesh::Cell::new(
            ElementKind::Quad4,
            vec![0, 1, 2, 3],
        )];

        let mesh = Mesh::new_with_cells(points, cells).unwrap();
        let material = Arc::new(J2PlasticMaterial::new(200e9, 0.3, 250e6, 10e9));

        // Fixed on left (ux=0 at 0,3), fixed in y on bottom (uy=0 at 0,1)
        let dirichlet_boundary = vec![
            (0, 0, 0.0),
            (0, 1, 0.0),
            (3, 0, 0.0),
            (1, 1, 0.0),
        ];

        // Apply a tension force of 3e7 N at the right nodes (ux force at 1 and 2)
        // This will pull the element to yield stress (limit is 250 MPa)
        let external_forces = vec![
            (1, 0, 1.5e7),
            (2, 0, 1.5e7),
        ];

        let assembly = NonlinearContinuumAssembly {
            mesh,
            thickness: 1.0,
            is_plane_strain: true,
            material,
            external_forces,
            dirichlet_boundary,
            contact: None,
            num_dofs: 8,
        };

        let options = NonlinearContinuumSolverOptions {
            num_steps: 5,
            max_iterations: 15,
            tolerance: 1e-4,
            linear_solver: LinearSolverOptions::default(),
        };

        let result = solve_nonlinear_continuum(&assembly, options).unwrap();

        // The displacement history should show monotonically increasing tension
        assert_eq!(result.displacements_history.len(), 6);
        let final_ux = result.displacements_history.last().unwrap()[2]; // ux at node 1
        assert!(final_ux > 0.0);

        // Nodal stress xx at final step should be close to 3e7 Pa / Area
        let final_stresses = result.nodal_stress_history.last().unwrap();
        // Since it's uniaxial tension, stress_xx should be positive, stress_yy should be close to Poisson effect
        let sig_xx = final_stresses[1][0];
        assert!(sig_xx > 2e7);
    }
}
