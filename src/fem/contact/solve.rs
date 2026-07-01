use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    fem::{
        contact::{
            augmented::assemble_augmented_contact,
            penalty::{assemble_penalty_contact, ContactPair},
            search::{BoundarySegment, SpatialHashGrid2D},
        },
        elasticity::{assemble_elasticity_system, ElasticityError, ElasticityMaterial, ElasticityProblem},
    },
    linalg::{axpy, norm, solve_linear_system, LinalgError, LinearSolverOptions},
    mesh::Mesh,
};

#[derive(Debug, Clone)]
pub struct ContactProblem {
    pub master_segments: Vec<BoundarySegment>,
    pub slave_nodes: Vec<usize>,
    pub penalty: f64,
    pub use_augmented: bool,
}

#[derive(Debug, Clone)]
pub struct ContactStaticAssembly {
    pub mesh: Mesh,
    pub material: ElasticityMaterial,
    pub thickness: f64,
    pub external_forces: Vec<(usize, usize, f64)>,
    pub dirichlet_boundary: Vec<(usize, usize, f64)>,
    pub contact: ContactProblem,
    pub num_dofs: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContactStaticSolverOptions {
    pub max_newton_iterations: usize,
    pub max_augmented_iterations: usize,
    pub tolerance: f64,
    pub penetration_tolerance: f64,
    pub linear_solver: LinearSolverOptions,
}

impl Default for ContactStaticSolverOptions {
    fn default() -> Self {
        Self {
            max_newton_iterations: 25,
            max_augmented_iterations: 10,
            tolerance: 1.0e-6,
            penetration_tolerance: 1.0e-8,
            linear_solver: LinearSolverOptions::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContactStaticResult {
    pub displacements: Vec<f64>,
    pub contact_pairs: Vec<ContactPair>,
    pub max_penetration: f64,
    pub newton_iterations: usize,
    pub augmented_iterations: usize,
}

#[derive(Debug, Error, PartialEq)]
pub enum ContactSolveError {
    #[error(transparent)]
    Elasticity(#[from] ElasticityError),
    #[error(transparent)]
    Linalg(#[from] LinalgError),
}

fn node_coords(mesh: &Mesh) -> Vec<[f64; 2]> {
    mesh.points()
        .iter()
        .map(|p| [p.x, p.y])
        .collect()
}

fn deformed_position(node_coords: &[[f64; 2]], u: &[f64], node: usize) -> [f64; 2] {
    [
        node_coords[node][0] + u[node * 2],
        node_coords[node][1] + u[node * 2 + 1],
    ]
}

/// Signed normal gap for a node relative to a master segment (negative = penetration).
fn segment_normal_gap(x_c: [f64; 2], x_m1: [f64; 2], x_m2: [f64; 2]) -> Option<f64> {
    let dx = x_m2[0] - x_m1[0];
    let dy = x_m2[1] - x_m1[1];
    let len_sq = dx * dx + dy * dy;
    if len_sq <= 1e-14 {
        return None;
    }
    let len = len_sq.sqrt();

    let nx = -dy / len;
    let ny = dx / len;

    let rx = x_c[0] - x_m1[0];
    let ry = x_c[1] - x_m1[1];
    let xi = (rx * dx + ry * dy) / len_sq;

    if !(0.0..=1.0).contains(&xi) {
        return None;
    }

    Some(rx * nx + ry * ny)
}

fn estimate_grid_cell_size(mesh: &Mesh, segments: &[BoundarySegment]) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for seg in segments {
        for &node in &seg.nodes {
            let p = mesh.points()[node];
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
    }

    let extent = (max_x - min_x).max(max_y - min_y);
    (extent / 10.0).max(1e-6)
}

fn best_contact_segment(
    x_c: [f64; 2],
    ref_coords: &[[f64; 2]],
    u: &[f64],
    seg: BoundarySegment,
    current: Option<(f64, BoundarySegment)>,
) -> Option<(f64, BoundarySegment)> {
    let x_m1 = deformed_position(ref_coords, u, seg.nodes[0]);
    let x_m2 = deformed_position(ref_coords, u, seg.nodes[1]);
    let gap = segment_normal_gap(x_c, x_m1, x_m2)?;
    if current.as_ref().is_none_or(|(best_gap, _)| gap < *best_gap) {
        Some((gap, seg))
    } else {
        current
    }
}

/// Builds node-to-segment contact pairs for slave nodes against master segments.
pub fn find_contact_pairs(
    mesh: &Mesh,
    master_segments: &[BoundarySegment],
    slave_nodes: &[usize],
    u: &[f64],
) -> Vec<ContactPair> {
    if master_segments.is_empty() || slave_nodes.is_empty() {
        return Vec::new();
    }

    let ref_coords = node_coords(mesh);
    let cell_size = estimate_grid_cell_size(mesh, master_segments);
    let mut grid = SpatialHashGrid2D::new(cell_size);
    grid.insert_segments(mesh, master_segments);

    let mut pairs = Vec::new();
    for &slave in slave_nodes {
        let x_c = deformed_position(&ref_coords, u, slave);
        let candidates = grid.query_candidates(x_c[0], x_c[1]);

        let mut best: Option<(f64, BoundarySegment)> = None;
        for seg_idx in candidates {
            best = best_contact_segment(
                x_c,
                &ref_coords,
                u,
                master_segments[seg_idx],
                best,
            );
        }

        if best.is_none() {
            for &seg in master_segments {
                best = best_contact_segment(x_c, &ref_coords, u, seg, best);
            }
        }

        if let Some((_, seg)) = best {
            pairs.push(ContactPair {
                candidate_node: slave,
                segment_nodes: seg.nodes,
            });
        }
    }

    pairs
}

fn mat_vec_mul(matrix: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; matrix.rows()];
    for (&val, (i, j)) in matrix.iter() {
        y[i] += val * x[j];
    }
    y
}

fn dirichlet_bc_set(dirichlet_boundary: &[(usize, usize, f64)]) -> std::collections::BTreeSet<usize> {
    let mut bc_set = std::collections::BTreeSet::new();
    for &(node, comp, _) in dirichlet_boundary {
        bc_set.insert(node * 2 + comp);
    }
    bc_set
}

impl ContactStaticAssembly {
    fn elastic_stiffness(&self) -> Result<CsMat<f64>, ElasticityError> {
        let problem = ElasticityProblem {
            material: self.material,
            thickness: self.thickness,
            constraints: Vec::new(),
            forces: Vec::new(),
        };
        let (matrix, _) = assemble_elasticity_system(&self.mesh, &problem)?;
        Ok(matrix)
    }

    fn evaluate_system(
        &self,
        u: &[f64],
        pairs: &[ContactPair],
        k_elastic: &CsMat<f64>,
        multipliers: &[f64],
    ) -> (Vec<f64>, CsMat<f64>, f64) {
        let ref_coords = node_coords(&self.mesh);
        let bc_set = dirichlet_bc_set(&self.dirichlet_boundary);

        let f_elastic = mat_vec_mul(k_elastic, u);
        let mut f_contact = vec![0.0; self.num_dofs];
        let mut triplets = TriMat::new((self.num_dofs, self.num_dofs));

        for (&val, (i, j)) in k_elastic.iter() {
            if bc_set.contains(&i) || bc_set.contains(&j) {
                continue;
            }
            triplets.add_triplet(i, j, val);
        }

        let max_penetration = if self.contact.use_augmented {
            let (_, max_pen) = assemble_augmented_contact(
                &ref_coords,
                u,
                pairs,
                multipliers,
                self.contact.penalty,
                &mut f_contact,
                &mut triplets,
                &bc_set,
            );
            max_pen
        } else {
            assemble_penalty_contact(
                &ref_coords,
                u,
                pairs,
                self.contact.penalty,
                &mut f_contact,
                &mut triplets,
                &bc_set,
            );
            pairs
                .iter()
                .filter_map(|pair| {
                    let c = pair.candidate_node;
                    let m1 = pair.segment_nodes[0];
                    let m2 = pair.segment_nodes[1];
                    let x_c = deformed_position(&ref_coords, u, c);
                    let x_m1 = deformed_position(&ref_coords, u, m1);
                    let x_m2 = deformed_position(&ref_coords, u, m2);
                    segment_normal_gap(x_c, x_m1, x_m2).filter(|gap| *gap < 0.0).map(|gap| -gap)
                })
                .fold(0.0_f64, f64::max)
        };

        let mut f_residual = f_elastic;
        for i in 0..self.num_dofs {
            f_residual[i] -= f_contact[i];
        }

        for &dof in &bc_set {
            triplets.add_triplet(dof, dof, 1.0);
        }

        (f_residual, triplets.to_csr(), max_penetration)
    }

    fn residual_norm(&self, u: &[f64], f_int: &[f64], f_ext: &[f64]) -> f64 {
        let mut r = vec![0.0; self.num_dofs];
        for i in 0..self.num_dofs {
            r[i] = f_int[i] - f_ext[i];
        }
        for &(node, comp, val) in &self.dirichlet_boundary {
            let dof = node * 2 + comp;
            r[dof] = u[dof] - val;
        }
        norm(&r)
    }

    fn newton_solve(
        &self,
        k_elastic: &CsMat<f64>,
        pairs: &[ContactPair],
        multipliers: &[f64],
        u_init: &[f64],
        options: &ContactStaticSolverOptions,
    ) -> Result<(Vec<f64>, usize, f64), LinalgError> {
        let n = self.num_dofs;
        let mut u = u_init.to_vec();
        let mut f_ext = vec![0.0; n];
        for &(node, comp, val) in &self.external_forces {
            f_ext[node * 2 + comp] += val;
        }

        let mut last_residual = 0.0;
        for iter in 1..=options.max_newton_iterations {
            let (f_int, k_tangent, max_pen) =
                self.evaluate_system(&u, pairs, k_elastic, multipliers);

            last_residual = self.residual_norm(&u, &f_int, &f_ext);
            if last_residual <= options.tolerance {
                return Ok((u, iter, max_pen));
            }

            let mut r = vec![0.0; n];
            for i in 0..n {
                r[i] = f_int[i] - f_ext[i];
            }
            for &(node, comp, val) in &self.dirichlet_boundary {
                let dof = node * 2 + comp;
                r[dof] = u[dof] - val;
            }

            let neg_r: Vec<_> = r.iter().map(|&x| -x).collect();
            let du =
                solve_linear_system(&k_tangent, &neg_r, options.linear_solver.clone())?.values;
            axpy(1.0, &du, &mut u);
        }

        Err(LinalgError::NonlinearNonConverged {
            iterations: options.max_newton_iterations,
            residual_norm: last_residual,
        })
    }
}

/// Solves a 2D linear-elastic static problem with frictionless node-to-segment contact.
pub fn solve_contact_static(
    assembly: &ContactStaticAssembly,
    options: ContactStaticSolverOptions,
) -> Result<ContactStaticResult, ContactSolveError> {
    let k_elastic = assembly.elastic_stiffness()?;
    let mut u = vec![0.0; assembly.num_dofs];
    let mut multipliers = Vec::new();
    let mut augmented_iterations = 0usize;
    let mut newton_iterations = 0usize;
    let mut max_penetration = 0.0;

    if assembly.contact.use_augmented {
        for aug_iter in 1..=options.max_augmented_iterations {
            augmented_iterations = aug_iter;
            let pairs = find_contact_pairs(
                &assembly.mesh,
                &assembly.contact.master_segments,
                &assembly.contact.slave_nodes,
                &u,
            );
            multipliers.resize(pairs.len(), 0.0);

            let (u_new, newton_iters, _) = assembly.newton_solve(
                &k_elastic,
                &pairs,
                &multipliers,
                &u,
                &options,
            )?;
            u = u_new;
            newton_iterations = newton_iters;

            let pairs = find_contact_pairs(
                &assembly.mesh,
                &assembly.contact.master_segments,
                &assembly.contact.slave_nodes,
                &u,
            );
            multipliers.resize(pairs.len(), 0.0);

            let ref_coords = node_coords(&assembly.mesh);
            let bc_set = dirichlet_bc_set(&assembly.dirichlet_boundary);
            let mut f_dummy = vec![0.0; assembly.num_dofs];
            let mut triplets = TriMat::new((assembly.num_dofs, assembly.num_dofs));
            let (updated_multipliers, max_pen) = assemble_augmented_contact(
                &ref_coords,
                &u,
                &pairs,
                &multipliers,
                assembly.contact.penalty,
                &mut f_dummy,
                &mut triplets,
                &bc_set,
            );
            multipliers = updated_multipliers;
            max_penetration = max_pen;

            if max_penetration <= options.penetration_tolerance {
                break;
            }
        }
    } else {
        let n = assembly.num_dofs;
        let mut f_ext = vec![0.0; n];
        for &(node, comp, val) in &assembly.external_forces {
            f_ext[node * 2 + comp] += val;
        }

        let mut last_residual = 0.0;
        for iter in 1..=options.max_newton_iterations {
            let pairs = find_contact_pairs(
                &assembly.mesh,
                &assembly.contact.master_segments,
                &assembly.contact.slave_nodes,
                &u,
            );
            let (f_int, k_tangent, max_pen) =
                assembly.evaluate_system(&u, &pairs, &k_elastic, &multipliers);
            max_penetration = max_pen;

            last_residual = assembly.residual_norm(&u, &f_int, &f_ext);
            if last_residual <= options.tolerance {
                newton_iterations = iter;
                let contact_pairs = find_contact_pairs(
                    &assembly.mesh,
                    &assembly.contact.master_segments,
                    &assembly.contact.slave_nodes,
                    &u,
                );
                return Ok(ContactStaticResult {
                    displacements: u,
                    contact_pairs,
                    max_penetration,
                    newton_iterations,
                    augmented_iterations: 0,
                });
            }

            let mut r = vec![0.0; n];
            for i in 0..n {
                r[i] = f_int[i] - f_ext[i];
            }
            for &(node, comp, val) in &assembly.dirichlet_boundary {
                let dof = node * 2 + comp;
                r[dof] = u[dof] - val;
            }

            let neg_r: Vec<_> = r.iter().map(|&x| -x).collect();
            let du = solve_linear_system(&k_tangent, &neg_r, options.linear_solver.clone())?.values;
            axpy(1.0, &du, &mut u);
        }

        return Err(LinalgError::NonlinearNonConverged {
            iterations: options.max_newton_iterations,
            residual_norm: last_residual,
        }
        .into());
    }

    let contact_pairs = find_contact_pairs(
        &assembly.mesh,
        &assembly.contact.master_segments,
        &assembly.contact.slave_nodes,
        &u,
    );

    Ok(ContactStaticResult {
        displacements: u,
        contact_pairs,
        max_penetration,
        newton_iterations,
        augmented_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Cell, ElementKind, Point2};

    #[test]
    fn find_contact_pairs_selects_closest_master_segment() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 0.2),
        ];
        let cells = vec![Cell::new(ElementKind::Tri3, vec![0, 1, 2])];
        let mesh = Mesh::new_with_cells(points, cells).unwrap();

        let master_segments = vec![BoundarySegment {
            nodes: [1, 0],
        }];
        let pairs = find_contact_pairs(&mesh, &master_segments, &[2], &[0.0; 6]);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].candidate_node, 2);
        assert_eq!(pairs[0].segment_nodes, [1, 0]);
    }
}
