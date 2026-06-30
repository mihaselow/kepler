#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

use crate::linalg::NonlinearSystem;
use crate::mesh::Point3;
use sprs::{CsMat, TriMat};

#[derive(Debug, Clone)]
pub struct NonlinearTrussElement {
    pub nodes: [usize; 2],
    pub area: f64,
    pub young_modulus: f64,
}

impl NonlinearTrussElement {
    pub fn local_stiffness(&self, points: &[Point3], displacements: &[f64]) -> [[f64; 4]; 4] {
        let n1 = self.nodes[0];
        let n2 = self.nodes[1];

        let ref_x1 = points[n1].coords[0];
        let ref_y1 = points[n1].coords[1];
        let ref_x2 = points[n2].coords[0];
        let ref_y2 = points[n2].coords[1];

        let l0 = ((ref_x2 - ref_x1).powi(2) + (ref_y2 - ref_y1).powi(2)).sqrt();

        let u_x1 = displacements[n1 * 2];
        let u_y1 = displacements[n1 * 2 + 1];
        let u_x2 = displacements[n2 * 2];
        let u_y2 = displacements[n2 * 2 + 1];

        let dx = (ref_x2 + u_x2) - (ref_x1 + u_x1);
        let dy = (ref_y2 + u_y2) - (ref_y1 + u_y1);
        let l = (dx.powi(2) + dy.powi(2)).sqrt();

        let strain = (l.powi(2) - l0.powi(2)) / (2.0 * l0.powi(2));
        let stress = self.young_modulus * strain;
        let force = stress * self.area;

        // Tangent stiffness = K_elastic_nonlinear + K_geometric
        let mut k = [[0.0; 4]; 4];

        // 1. Elastic stiffness: (E * A / L_0^3) * d * d^T
        let factor_e = (self.young_modulus * self.area) / l0.powi(3);
        let d = [-dx, -dy, dx, dy];
        for i in 0..4 {
            for j in 0..4 {
                k[i][j] += factor_e * d[i] * d[j];
            }
        }

        // 2. Geometric stiffness: (N / L_0) * I_truss
        let factor_g = force / l0;
        let i_truss = [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
        ];
        for i in 0..4 {
            for j in 0..4 {
                k[i][j] += factor_g * i_truss[i][j];
            }
        }

        k
    }

    pub fn local_force(&self, points: &[Point3], displacements: &[f64]) -> [f64; 4] {
        let n1 = self.nodes[0];
        let n2 = self.nodes[1];

        let ref_x1 = points[n1].coords[0];
        let ref_y1 = points[n1].coords[1];
        let ref_x2 = points[n2].coords[0];
        let ref_y2 = points[n2].coords[1];

        let l0 = ((ref_x2 - ref_x1).powi(2) + (ref_y2 - ref_y1).powi(2)).sqrt();

        let u_x1 = displacements[n1 * 2];
        let u_y1 = displacements[n1 * 2 + 1];
        let u_x2 = displacements[n2 * 2];
        let u_y2 = displacements[n2 * 2 + 1];

        let dx = (ref_x2 + u_x2) - (ref_x1 + u_x1);
        let dy = (ref_y2 + u_y2) - (ref_y1 + u_y1);
        let l = (dx.powi(2) + dy.powi(2)).sqrt();

        let strain = (l.powi(2) - l0.powi(2)) / (2.0 * l0.powi(2));
        let stress = self.young_modulus * strain;
        let force = stress * self.area;

        let factor = force / l0;
        [-factor * dx, -factor * dy, factor * dx, factor * dy]
    }
}

pub struct NonlinearTrussAssembly {
    pub points: Vec<Point3>,
    pub elements: Vec<NonlinearTrussElement>,
    pub external_forces: Vec<(usize, usize, f64)>, // (node, component, value)
    pub dirichlet_boundary: Vec<(usize, usize, f64)>, // (node, component, value)
    pub num_dofs: usize,
}

impl NonlinearSystem for NonlinearTrussAssembly {
    fn dimension(&self) -> usize {
        self.num_dofs
    }

    fn residual(&self, values: &[f64]) -> Vec<f64> {
        let mut f_int = vec![0.0; self.num_dofs];

        for el in &self.elements {
            let f_local = el.local_force(&self.points, values);
            let n1 = el.nodes[0];
            let n2 = el.nodes[1];

            f_int[n1 * 2] += f_local[0];
            f_int[n1 * 2 + 1] += f_local[1];
            f_int[n2 * 2] += f_local[2];
            f_int[n2 * 2 + 1] += f_local[3];
        }

        let mut r = vec![0.0; self.num_dofs];

        // Add internal forces
        for i in 0..self.num_dofs {
            r[i] = f_int[i];
        }

        // Subtract external forces
        for &(node, comp, val) in &self.external_forces {
            r[node * 2 + comp] -= val;
        }

        // Enforce boundary conditions
        for &(node, comp, val) in &self.dirichlet_boundary {
            let dof = node * 2 + comp;
            r[dof] = values[dof] - val;
        }

        r
    }

    fn jacobian(&self, values: &[f64]) -> CsMat<f64> {
        let mut triplets = TriMat::new((self.num_dofs, self.num_dofs));
        let mut bc_set = std::collections::BTreeSet::new();
        for &(node, comp, _) in &self.dirichlet_boundary {
            bc_set.insert(node * 2 + comp);
        }

        for el in &self.elements {
            let k_local = el.local_stiffness(&self.points, values);
            let global_indices = [
                el.nodes[0] * 2,
                el.nodes[0] * 2 + 1,
                el.nodes[1] * 2,
                el.nodes[1] * 2 + 1,
            ];

            for i in 0..4 {
                let r = global_indices[i];
                if bc_set.contains(&r) {
                    continue;
                }
                for j in 0..4 {
                    let c = global_indices[j];
                    if bc_set.contains(&c) {
                        continue;
                    }
                    triplets.add_triplet(r, c, k_local[i][j]);
                }
            }
        }

        // Put 1.0 on diagonal of boundary conditions
        for &dof in &bc_set {
            triplets.add_triplet(dof, dof, 1.0);
        }

        triplets.to_csr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{NonlinearSolverOptions, newton_solve};

    #[test]
    fn test_von_mises_snap_through() {
        // von Mises Truss: 3 nodes
        // Node 0: fixed at (0, 0)
        // Node 2: fixed at (2, 0)
        // Node 1: apex at (1, 0.1)
        let points = vec![
            Point3::new([0.0, 0.0, 0.0]),
            Point3::new([1.0, 0.1, 0.0]),
            Point3::new([2.0, 0.0, 0.0]),
        ];

        let elements = vec![
            NonlinearTrussElement {
                nodes: [0, 1],
                area: 1e-4,
                young_modulus: 200e9,
            },
            NonlinearTrussElement {
                nodes: [1, 2],
                area: 1e-4,
                young_modulus: 200e9,
            },
        ];

        // Case 1: Displacement control (prescribe apex displacement of -0.2)
        let dirichlet_boundary = vec![
            (0, 0, 0.0),
            (0, 1, 0.0),
            (2, 0, 0.0),
            (2, 1, 0.0),
            (1, 1, -0.2),
        ];

        let assembly_disp = NonlinearTrussAssembly {
            points: points.clone(),
            elements: elements.clone(),
            external_forces: vec![],
            dirichlet_boundary,
            num_dofs: 6,
        };

        let initial_values = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let options = NonlinearSolverOptions::default();

        let result_disp = newton_solve(&assembly_disp, initial_values, options.clone()).unwrap();

        // Apex displacement should be exactly -0.2
        let u_y_apex_disp = result_disp.values[3];
        assert!((u_y_apex_disp - -0.2).abs() < 1e-10);

        // Case 2: Load control with a small force (should converge to a small pre-buckling displacement)
        let dirichlet_boundary_load = vec![(0, 0, 0.0), (0, 1, 0.0), (2, 0, 0.0), (2, 1, 0.0)];
        let external_forces = vec![(1, 1, -1000.0)];

        let assembly_load = NonlinearTrussAssembly {
            points,
            elements,
            external_forces,
            dirichlet_boundary: dirichlet_boundary_load,
            num_dofs: 6,
        };

        let initial_values = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut load_options = options;
        load_options.tolerance = 1.0e-6;
        let result_load = newton_solve(&assembly_load, initial_values, load_options).unwrap();

        // Displacement should be negative and small
        let u_y_apex_load = result_load.values[3];
        assert!(u_y_apex_load < 0.0);
        assert!(u_y_apex_load > -0.05);
    }
}
