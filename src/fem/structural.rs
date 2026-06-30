use crate::fem::element::{Element, ElementError};
use crate::mesh::{NodeId, Point3};
use std::collections::BTreeMap;

pub struct Truss<'a> {
    pub nodes: &'a [NodeId; 2],
    pub dim: usize,
    pub area: f64,
}

impl<'a> Element for Truss<'a> {
    fn spatial_dimension(&self) -> usize {
        self.dim
    }

    fn node_count(&self) -> usize {
        2
    }

    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }

    fn active_fields(&self) -> Vec<String> {
        if self.dim == 2 {
            vec!["ux".to_string(), "uy".to_string()]
        } else {
            vec!["ux".to_string(), "uy".to_string(), "uz".to_string()]
        }
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let young_modulus = *properties
            .get("young_modulus")
            .ok_or_else(|| ElementError::MissingProperty("young_modulus".to_string()))?;

        if node_coords.len() != 2 {
            return Err(ElementError::InvalidNodeCount {
                expected: 2,
                actual: node_coords.len(),
            });
        }

        let a = node_coords[0].coords;
        let b = node_coords[1].coords;

        let mut diff = [0.0; 3];
        let mut length_sq = 0.0;
        for i in 0..self.dim {
            diff[i] = b[i] - a[i];
            length_sq += diff[i].powi(2);
        }
        let length = length_sq.sqrt();
        if length <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }

        let mut n = vec![0.0; self.dim];
        for i in 0..self.dim {
            n[i] = diff[i] / length;
        }

        let k_factor = self.area * young_modulus / length;
        let size = 2 * self.dim;
        let mut stiffness = vec![vec![0.0; size]; size];

        for r_node in 0..2 {
            for c_node in 0..2 {
                let sign = if r_node == c_node { 1.0 } else { -1.0 };
                for r_dof in 0..self.dim {
                    for c_dof in 0..self.dim {
                        let r = r_node * self.dim + r_dof;
                        let c = c_node * self.dim + c_dof;
                        stiffness[r][c] = sign * k_factor * n[r_dof] * n[c_dof];
                    }
                }
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
        if node_coords.len() != 2 {
            return Err(ElementError::InvalidNodeCount {
                expected: 2,
                actual: node_coords.len(),
            });
        }

        let a = node_coords[0].coords;
        let b = node_coords[1].coords;

        let mut length_sq = 0.0;
        for i in 0..self.dim {
            length_sq += (b[i] - a[i]).powi(2);
        }
        let length = length_sq.sqrt();
        if length <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }

        let total_mass = density * self.area * length;
        let size = 2 * self.dim;
        let mut mass = vec![vec![0.0; size]; size];

        if lumped {
            let nodal_mass = total_mass / 2.0;
            for i in 0..size {
                mass[i][i] = nodal_mass;
            }
        } else {
            let val_diag = total_mass / 3.0;
            let val_off = total_mass / 6.0;
            for r_node in 0..2 {
                for c_node in 0..2 {
                    let factor = if r_node == c_node { val_diag } else { val_off };
                    for dof in 0..self.dim {
                        let r = r_node * self.dim + dof;
                        let c = c_node * self.dim + dof;
                        mass[r][c] = factor;
                    }
                }
            }
        }

        Ok(mass)
    }
}

pub struct Beam2D<'a> {
    pub nodes: &'a [NodeId; 2],
    pub area: f64,
    pub moment_of_inertia: f64,
    pub shear_factor: f64, // k_s. Set to 0.0 to enforce Euler-Bernoulli (no shear)
}

impl<'a> Element for Beam2D<'a> {
    fn spatial_dimension(&self) -> usize {
        2
    }

    fn node_count(&self) -> usize {
        2
    }

    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }

    fn active_fields(&self) -> Vec<String> {
        vec!["ux".to_string(), "uy".to_string(), "theta_z".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let young_modulus = *properties
            .get("young_modulus")
            .ok_or_else(|| ElementError::MissingProperty("young_modulus".to_string()))?;
        let poisson_ratio = *properties
            .get("poisson_ratio")
            .ok_or_else(|| ElementError::MissingProperty("poisson_ratio".to_string()))?;

        if node_coords.len() != 2 {
            return Err(ElementError::InvalidNodeCount {
                expected: 2,
                actual: node_coords.len(),
            });
        }

        let a = node_coords[0].coords;
        let b = node_coords[1].coords;

        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let length = (dx.powi(2) + dy.powi(2)).sqrt();
        if length <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }

        let c = dx / length;
        let s = dy / length;

        // Shear deformation parameter Phi
        let phi = if self.shear_factor > 0.0 {
            let shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio));
            (12.0 * young_modulus * self.moment_of_inertia)
                / (self.shear_factor * shear_modulus * self.area * length.powi(2))
        } else {
            0.0
        };

        let k_axial = self.area * young_modulus / length;
        let val_a =
            (12.0 * young_modulus * self.moment_of_inertia) / (length.powi(3) * (1.0 + phi));
        let val_b = (6.0 * young_modulus * self.moment_of_inertia) / (length.powi(2) * (1.0 + phi));
        let val_rot1 =
            ((4.0 + phi) * young_modulus * self.moment_of_inertia) / (length * (1.0 + phi));
        let val_rot2 =
            ((2.0 - phi) * young_modulus * self.moment_of_inertia) / (length * (1.0 + phi));

        let mut k_local = vec![vec![0.0; 6]; 6];
        k_local[0][0] = k_axial;
        k_local[0][3] = -k_axial;
        k_local[3][0] = -k_axial;
        k_local[3][3] = k_axial;

        k_local[1][1] = val_a;
        k_local[1][2] = val_b;
        k_local[1][4] = -val_a;
        k_local[1][5] = val_b;

        k_local[2][1] = val_b;
        k_local[2][2] = val_rot1;
        k_local[2][4] = -val_b;
        k_local[2][5] = val_rot2;

        k_local[4][1] = -val_a;
        k_local[4][2] = -val_b;
        k_local[4][4] = val_a;
        k_local[4][5] = -val_b;

        k_local[5][1] = val_b;
        k_local[5][2] = val_rot2;
        k_local[5][4] = -val_b;
        k_local[5][5] = val_rot1;

        let transform = vec![
            vec![c, s, 0.0, 0.0, 0.0, 0.0],
            vec![-s, c, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, c, s, 0.0],
            vec![0.0, 0.0, 0.0, -s, c, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        let trans_t = mat_transpose(&transform);
        let temp = mat_mul(&k_local, &transform);
        let k_global = mat_mul(&trans_t, &temp);

        Ok(k_global)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 2 {
            return Err(ElementError::InvalidNodeCount {
                expected: 2,
                actual: node_coords.len(),
            });
        }

        let a = node_coords[0].coords;
        let b = node_coords[1].coords;

        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let length = (dx.powi(2) + dy.powi(2)).sqrt();
        if length <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }

        let c = dx / length;
        let s = dy / length;

        let transform = vec![
            vec![c, s, 0.0, 0.0, 0.0, 0.0],
            vec![-s, c, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, c, s, 0.0],
            vec![0.0, 0.0, 0.0, -s, c, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        let trans_t = mat_transpose(&transform);

        let m_local = if lumped {
            let m_trans = density * self.area * length * 0.5;
            let m_rot = density * self.area * length.powi(3) / 24.0;
            let mut m = vec![vec![0.0; 6]; 6];
            m[0][0] = m_trans;
            m[1][1] = m_trans;
            m[2][2] = m_rot;
            m[3][3] = m_trans;
            m[4][4] = m_trans;
            m[5][5] = m_rot;
            m
        } else {
            let m_axial = density * self.area * length / 6.0;
            let m_b = density * self.area * length / 420.0;
            let mut m = vec![vec![0.0; 6]; 6];

            m[0][0] = 2.0 * m_axial;
            m[0][3] = m_axial;
            m[3][0] = m_axial;
            m[3][3] = 2.0 * m_axial;

            m[1][1] = 156.0 * m_b;
            m[1][2] = 22.0 * length * m_b;
            m[1][4] = 54.0 * m_b;
            m[1][5] = -13.0 * length * m_b;

            m[2][1] = 22.0 * length * m_b;
            m[2][2] = 4.0 * length.powi(2) * m_b;
            m[2][4] = 13.0 * length * m_b;
            m[2][5] = -3.0 * length.powi(2) * m_b;

            m[4][1] = 54.0 * m_b;
            m[4][2] = 13.0 * length * m_b;
            m[4][4] = 156.0 * m_b;
            m[4][5] = -22.0 * length * m_b;

            m[5][1] = -13.0 * length * m_b;
            m[5][2] = -3.0 * length.powi(2) * m_b;
            m[5][4] = -22.0 * length * m_b;
            m[5][5] = 4.0 * length.powi(2) * m_b;
            m
        };

        let temp = mat_mul(&m_local, &transform);
        let m_global = mat_mul(&trans_t, &temp);

        Ok(m_global)
    }
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    let inner = b.len();
    let mut c = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..inner {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = a[0].len();
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truss_element() {
        let nodes = [0, 1];
        let el = Truss {
            nodes: &nodes,
            dim: 2,
            area: 0.1,
        };

        let coords = [
            Point3::new([0.0, 0.0, 0.0]),
            Point3::new([3.0, 4.0, 0.0]), // length = 5.0, cos = 0.6, sin = 0.8
        ];

        let mut properties = BTreeMap::new();
        properties.insert("young_modulus".to_string(), 200e9);

        let k = el.local_stiffness(&coords, &properties).unwrap();
        assert_eq!(k.len(), 4);

        // K = (A*E/L) * n * n^T. A*E/L = 0.1 * 200e9 / 5.0 = 4e9.
        // n = [0.6, 0.8]. n*n^T = [[0.36, 0.48], [0.48, 0.64]].
        // K_00 = 4e9 * 0.36 = 1.44e9.
        assert!((k[0][0] - 1.44e9).abs() < 1.0);
        assert!((k[0][1] - 1.92e9).abs() < 1.0);
        assert!((k[0][2] - -1.44e9).abs() < 1.0);

        // Mass matrix (lumped) - density = 7800. Volume = A * L = 0.1 * 5.0 = 0.5.
        // Mass = 7800 * 0.5 = 3900 kg. Lumped = 1950 kg per node.
        let m = el.local_mass(&coords, 7800.0, true).unwrap();
        assert_eq!(m[0][0], 1950.0);
        assert_eq!(m[1][1], 1950.0);
        assert_eq!(m[2][2], 1950.0);
        assert_eq!(m[3][3], 1950.0);
    }

    #[test]
    fn test_beam_2d_element() {
        let nodes = [0, 1];
        let el = Beam2D {
            nodes: &nodes,
            area: 0.01,
            moment_of_inertia: 1e-5,
            shear_factor: 0.0, // Euler-Bernoulli
        };

        // Horizontal beam of length 2.0
        let coords = [Point3::new([0.0, 0.0, 0.0]), Point3::new([2.0, 0.0, 0.0])];

        let mut properties = BTreeMap::new();
        properties.insert("young_modulus".to_string(), 200e9);
        properties.insert("poisson_ratio".to_string(), 0.3);

        let k = el.local_stiffness(&coords, &properties).unwrap();
        assert_eq!(k.len(), 6);

        // Axial: AE/L = 0.01 * 200e9 / 2.0 = 1e9
        assert!((k[0][0] - 1e9).abs() < 1.0);
        assert!((k[0][3] - -1e9).abs() < 1.0);

        // Bending: 12*E*I/L^3 = 12 * 200e9 * 1e-5 / 8.0 = 3e6
        assert!((k[1][1] - 3e6).abs() < 1.0);
    }
}
