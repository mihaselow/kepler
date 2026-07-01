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

pub struct Beam3D<'a> {
    pub nodes: &'a [NodeId; 2],
    pub area: f64,
    pub moment_y: f64,
    pub moment_z: f64,
    pub torsional_constant: f64,
    pub local_y_direction: [f64; 3],
}

impl<'a> Element for Beam3D<'a> {
    fn spatial_dimension(&self) -> usize {
        3
    }

    fn node_count(&self) -> usize {
        2
    }

    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }

    fn active_fields(&self) -> Vec<String> {
        vec![
            "ux".to_string(),
            "uy".to_string(),
            "uz".to_string(),
            "theta_x".to_string(),
            "theta_y".to_string(),
            "theta_z".to_string(),
        ]
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
        let dz = b[2] - a[2];
        let length = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();
        if length <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }

        let ex = [dx / length, dy / length, dz / length];
        let v_aux = self.local_y_direction;

        let mut vz = [
            ex[1] * v_aux[2] - ex[2] * v_aux[1],
            ex[2] * v_aux[0] - ex[0] * v_aux[2],
            ex[0] * v_aux[1] - ex[1] * v_aux[0],
        ];
        let mut len_z = (vz[0].powi(2) + vz[1].powi(2) + vz[2].powi(2)).sqrt();
        if len_z <= 1e-6 {
            let alt_aux = if ex[0].abs() > 0.9 {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            vz = [
                ex[1] * alt_aux[2] - ex[2] * alt_aux[1],
                ex[2] * alt_aux[0] - ex[0] * alt_aux[2],
                ex[0] * alt_aux[1] - ex[1] * alt_aux[0],
            ];
            len_z = (vz[0].powi(2) + vz[1].powi(2) + vz[2].powi(2)).sqrt();
        }
        let ez = [vz[0] / len_z, vz[1] / len_z, vz[2] / len_z];
        let ey = [
            ez[1] * ex[2] - ez[2] * ex[1],
            ez[2] * ex[0] - ez[0] * ex[2],
            ez[0] * ex[1] - ez[1] * ex[0],
        ];

        let shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio));

        let k_axial = self.area * young_modulus / length;
        let k_torsion = self.torsional_constant * shear_modulus / length;

        let val_az = 12.0 * young_modulus * self.moment_z / length.powi(3);
        let val_bz = 6.0 * young_modulus * self.moment_z / length.powi(2);
        let val_rot1z = 4.0 * young_modulus * self.moment_z / length;
        let val_rot2z = 2.0 * young_modulus * self.moment_z / length;

        let val_ay = 12.0 * young_modulus * self.moment_y / length.powi(3);
        let val_by = 6.0 * young_modulus * self.moment_y / length.powi(2);
        let val_rot1y = 4.0 * young_modulus * self.moment_y / length;
        let val_rot2y = 2.0 * young_modulus * self.moment_y / length;

        let mut k_local = vec![vec![0.0; 12]; 12];

        // Axial
        k_local[0][0] = k_axial;
        k_local[0][6] = -k_axial;
        k_local[6][0] = -k_axial;
        k_local[6][6] = k_axial;

        // Torsion
        k_local[3][3] = k_torsion;
        k_local[3][9] = -k_torsion;
        k_local[9][3] = -k_torsion;
        k_local[9][9] = k_torsion;

        // Bending z (in x-y plane)
        k_local[1][1] = val_az;
        k_local[1][5] = val_bz;
        k_local[1][7] = -val_az;
        k_local[1][11] = val_bz;

        k_local[5][1] = val_bz;
        k_local[5][5] = val_rot1z;
        k_local[5][7] = -val_bz;
        k_local[5][11] = val_rot2z;

        k_local[7][1] = -val_az;
        k_local[7][5] = -val_bz;
        k_local[7][7] = val_az;
        k_local[7][11] = -val_bz;

        k_local[11][1] = val_bz;
        k_local[11][5] = val_rot2z;
        k_local[11][7] = -val_bz;
        k_local[11][11] = val_rot1z;

        // Bending y (in x-z plane)
        k_local[2][2] = val_ay;
        k_local[2][4] = -val_by;
        k_local[2][8] = -val_ay;
        k_local[2][10] = -val_by;

        k_local[4][2] = -val_by;
        k_local[4][4] = val_rot1y;
        k_local[4][8] = val_by;
        k_local[4][10] = val_rot2y;

        k_local[8][2] = -val_ay;
        k_local[8][4] = val_by;
        k_local[8][8] = val_ay;
        k_local[8][10] = val_by;

        k_local[10][2] = -val_by;
        k_local[10][4] = val_rot2y;
        k_local[10][8] = val_by;
        k_local[10][10] = val_rot1y;

        let transform = vec![
            vec![
                ex[0], ex[1], ex[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                ey[0], ey[1], ey[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                ez[0], ez[1], ez[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, ex[0], ex[1], ex[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, ey[0], ey[1], ey[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, ez[0], ez[1], ez[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ex[0], ex[1], ex[2], 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ey[0], ey[1], ey[2], 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ez[0], ez[1], ez[2], 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ex[0], ex[1], ex[2],
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ey[0], ey[1], ey[2],
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ez[0], ez[1], ez[2],
            ],
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
        let dz = b[2] - a[2];
        let length = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();
        if length <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }

        let ex = [dx / length, dy / length, dz / length];
        let v_aux = self.local_y_direction;

        let mut vz = [
            ex[1] * v_aux[2] - ex[2] * v_aux[1],
            ex[2] * v_aux[0] - ex[0] * v_aux[2],
            ex[0] * v_aux[1] - ex[1] * v_aux[0],
        ];
        let mut len_z = (vz[0].powi(2) + vz[1].powi(2) + vz[2].powi(2)).sqrt();
        if len_z <= 1e-6 {
            let alt_aux = if ex[0].abs() > 0.9 {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            vz = [
                ex[1] * alt_aux[2] - ex[2] * alt_aux[1],
                ex[2] * alt_aux[0] - ex[0] * alt_aux[2],
                ex[0] * alt_aux[1] - ex[1] * alt_aux[0],
            ];
            len_z = (vz[0].powi(2) + vz[1].powi(2) + vz[2].powi(2)).sqrt();
        }
        let ez = [vz[0] / len_z, vz[1] / len_z, vz[2] / len_z];
        let ey = [
            ez[1] * ex[2] - ez[2] * ex[1],
            ez[2] * ex[0] - ez[0] * ex[2],
            ez[0] * ex[1] - ez[1] * ex[0],
        ];

        let transform = vec![
            vec![
                ex[0], ex[1], ex[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                ey[0], ey[1], ey[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                ez[0], ez[1], ez[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, ex[0], ex[1], ex[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, ey[0], ey[1], ey[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, ez[0], ez[1], ez[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ex[0], ex[1], ex[2], 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ey[0], ey[1], ey[2], 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ez[0], ez[1], ez[2], 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ex[0], ex[1], ex[2],
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ey[0], ey[1], ey[2],
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ez[0], ez[1], ez[2],
            ],
        ];

        let trans_t = mat_transpose(&transform);

        let m_local = if lumped {
            let m_trans = density * self.area * length * 0.5;
            let m_rot_x = density * (self.moment_y + self.moment_z) * length * 0.5;
            let m_rot_yz = density * self.area * length.powi(3) / 24.0;
            let mut m = vec![vec![0.0; 12]; 12];
            for i in 0..2 {
                let offset = 6 * i;
                m[offset][offset] = m_trans;
                m[offset + 1][offset + 1] = m_trans;
                m[offset + 2][offset + 2] = m_trans;
                m[offset + 3][offset + 3] = m_rot_x;
                m[offset + 4][offset + 4] = m_rot_yz;
                m[offset + 5][offset + 5] = m_rot_yz;
            }
            m
        } else {
            let mut m = vec![vec![0.0; 12]; 12];
            let m_axial = density * self.area * length / 6.0;
            let m_torsion = density * (self.moment_y + self.moment_z) * length / 6.0;
            let m_bend_z = density * self.area * length / 420.0;
            let m_bend_y = density * self.area * length / 420.0;

            // Axial
            m[0][0] = 2.0 * m_axial;
            m[0][6] = m_axial;
            m[6][0] = m_axial;
            m[6][6] = 2.0 * m_axial;

            // Torsion
            m[3][3] = 2.0 * m_torsion;
            m[3][9] = m_torsion;
            m[9][3] = m_torsion;
            m[9][9] = 2.0 * m_torsion;

            // Bending z (in x-y plane: DOFs uy (1), theta_z (5))
            m[1][1] = 156.0 * m_bend_z;
            m[1][5] = 22.0 * length * m_bend_z;
            m[1][7] = 54.0 * m_bend_z;
            m[1][11] = -13.0 * length * m_bend_z;

            m[5][1] = 22.0 * length * m_bend_z;
            m[5][5] = 4.0 * length.powi(2) * m_bend_z;
            m[5][7] = 13.0 * length * m_bend_z;
            m[5][11] = -3.0 * length.powi(2) * m_bend_z;

            m[7][1] = 54.0 * m_bend_z;
            m[7][5] = 13.0 * length * m_bend_z;
            m[7][7] = 156.0 * m_bend_z;
            m[7][11] = -22.0 * length * m_bend_z;

            m[11][1] = -13.0 * length * m_bend_z;
            m[11][5] = -3.0 * length.powi(2) * m_bend_z;
            m[11][7] = -22.0 * length * m_bend_z;
            m[11][11] = 4.0 * length.powi(2) * m_bend_z;

            // Bending y (in x-z plane: DOFs uz (2), theta_y (4))
            m[2][2] = 156.0 * m_bend_y;
            m[2][4] = -22.0 * length * m_bend_y;
            m[2][8] = 54.0 * m_bend_y;
            m[2][10] = 13.0 * length * m_bend_y;

            m[4][2] = -22.0 * length * m_bend_y;
            m[4][4] = 4.0 * length.powi(2) * m_bend_y;
            m[4][8] = -13.0 * length * m_bend_y;
            m[4][10] = -3.0 * length.powi(2) * m_bend_y;

            m[8][2] = 54.0 * m_bend_y;
            m[8][4] = -13.0 * length * m_bend_y;
            m[8][8] = 156.0 * m_bend_y;
            m[8][10] = 22.0 * length * m_bend_y;

            m[10][2] = 13.0 * length * m_bend_y;
            m[10][4] = -3.0 * length.powi(2) * m_bend_y;
            m[10][8] = 22.0 * length * m_bend_y;
            m[10][10] = 4.0 * length.powi(2) * m_bend_y;

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

// Tests are placed at the end of the file.

pub struct ShellTri3<'a> {
    pub nodes: &'a [NodeId; 3],
    pub thickness: f64,
}

impl<'a> Element for ShellTri3<'a> {
    fn spatial_dimension(&self) -> usize {
        3
    }

    fn node_count(&self) -> usize {
        3
    }

    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }

    fn active_fields(&self) -> Vec<String> {
        vec![
            "ux".to_string(),
            "uy".to_string(),
            "uz".to_string(),
            "theta_x".to_string(),
            "theta_y".to_string(),
            "theta_z".to_string(),
        ]
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

        if node_coords.len() != 3 {
            return Err(ElementError::InvalidNodeCount {
                expected: 3,
                actual: node_coords.len(),
            });
        }

        // 1. Establish local coordinate system (e1, e2, e3)
        let p0 = node_coords[0].coords;
        let p1 = node_coords[1].coords;
        let p2 = node_coords[2].coords;

        let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let len_v1 = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
        if len_v1 <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }
        let e1 = [v1[0] / len_v1, v1[1] / len_v1, v1[2] / len_v1];

        // Normal vector
        let v_n = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];
        let len_vn = (v_n[0].powi(2) + v_n[1].powi(2) + v_n[2].powi(2)).sqrt();
        if len_vn <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }
        let e3 = [v_n[0] / len_vn, v_n[1] / len_vn, v_n[2] / len_vn];

        // e2 = e3 x e1
        let e2 = [
            e3[1] * e1[2] - e3[2] * e1[1],
            e3[2] * e1[0] - e3[0] * e1[2],
            e3[0] * e1[1] - e3[1] * e1[0],
        ];

        // Local coordinates of nodes: origin at node 0
        let x1 = v1[0] * e1[0] + v1[1] * e1[1] + v1[2] * e1[2];
        let x2 = v2[0] * e1[0] + v2[1] * e1[1] + v2[2] * e1[2];
        let y2 = v2[0] * e2[0] + v2[1] * e2[1] + v2[2] * e2[2];

        let area = 0.5 * x1 * y2.abs();
        if area <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }

        // Shape function derivatives in local coordinates
        let gx0 = -y2 / (2.0 * area);
        let gy0 = (x2 - x1) / (2.0 * area);

        let gx1 = y2 / (2.0 * area);
        let gy1 = -x2 / (2.0 * area);

        let gx2 = 0.0;
        let gy2 = x1 / (2.0 * area);

        // --- membrane part (plane stress, 6x6) ---
        let mut k_membrane = vec![vec![0.0; 6]; 6];
        let d_factor = young_modulus / (1.0 - poisson_ratio.powi(2));
        let dm = [
            [d_factor, d_factor * poisson_ratio, 0.0],
            [d_factor * poisson_ratio, d_factor, 0.0],
            [0.0, 0.0, d_factor * 0.5 * (1.0 - poisson_ratio)],
        ];

        // B_m matrix (3x6) for membrane
        let bm = [
            [gx0, 0.0, gx1, 0.0, gx2, 0.0],
            [0.0, gy0, 0.0, gy1, 0.0, gy2],
            [gy0, gx0, gy1, gx1, gy2, gx2],
        ];

        let factor_m = area * self.thickness;
        for r in 0..6 {
            for c in 0..6 {
                let mut db = [0.0; 3];
                db[0] = dm[0][0] * bm[0][c] + dm[0][1] * bm[1][c];
                db[1] = dm[1][0] * bm[0][c] + dm[1][1] * bm[1][c];
                db[2] = dm[2][2] * bm[2][c];

                k_membrane[r][c] =
                    factor_m * (bm[0][r] * db[0] + bm[1][r] * db[1] + bm[2][r] * db[2]);
            }
        }

        // --- bending part (Mindlin, 9x9) ---
        let mut k_bend = vec![vec![0.0; 9]; 9];
        let db_factor =
            young_modulus * self.thickness.powi(3) / (12.0 * (1.0 - poisson_ratio.powi(2)));
        let db_mat = [
            [db_factor, db_factor * poisson_ratio, 0.0],
            [db_factor * poisson_ratio, db_factor, 0.0],
            [0.0, 0.0, db_factor * 0.5 * (1.0 - poisson_ratio)],
        ];

        // B_b matrix (3x9)
        let bb = [
            [0.0, -gx0, 0.0, 0.0, -gx1, 0.0, 0.0, -gx2, 0.0],
            [0.0, 0.0, -gy0, 0.0, 0.0, -gy1, 0.0, 0.0, -gy2],
            [0.0, -gy0, -gx0, 0.0, -gy1, -gx1, 0.0, -gy2, -gx2],
        ];

        for r in 0..9 {
            for c in 0..9 {
                let mut db = [0.0; 3];
                db[0] = db_mat[0][0] * bb[0][c] + db_mat[0][1] * bb[1][c];
                db[1] = db_mat[1][0] * bb[0][c] + db_mat[1][1] * bb[1][c];
                db[2] = db_mat[2][2] * bb[2][c];

                k_bend[r][c] += area * (bb[0][r] * db[0] + bb[1][r] * db[1] + bb[2][r] * db[2]);
            }
        }

        // --- shear part (Mindlin shear locking correction, 9x9) ---
        let shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio));
        let k_shear_factor = 5.0 / 6.0;
        let ds = shear_modulus * self.thickness * k_shear_factor;

        // B_s matrix (2x9) evaluated at centroid (shape = 1/3)
        let bs = [
            [
                gx0,
                -1.0 / 3.0,
                0.0,
                gx1,
                -1.0 / 3.0,
                0.0,
                gx2,
                -1.0 / 3.0,
                0.0,
            ],
            [
                gy0,
                0.0,
                -1.0 / 3.0,
                gy1,
                0.0,
                -1.0 / 3.0,
                gy2,
                0.0,
                -1.0 / 3.0,
            ],
        ];

        for r in 0..9 {
            for c in 0..9 {
                k_bend[r][c] += area * ds * (bs[0][r] * bs[0][c] + bs[1][r] * bs[1][c]);
            }
        }

        // --- Assemble local 18x18 matrix ---
        let mut k_local = vec![vec![0.0; 18]; 18];

        let mem_dofs = [0, 1, 6, 7, 12, 13];
        for r in 0..6 {
            for c in 0..6 {
                k_local[mem_dofs[r]][mem_dofs[c]] = k_membrane[r][c];
            }
        }

        let bend_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16];
        for r in 0..9 {
            for c in 0..9 {
                k_local[bend_dofs[r]][bend_dofs[c]] = k_bend[r][c];
            }
        }

        let drill_stiff = 1e-4 * shear_modulus * self.thickness * area;
        let drill_dofs = [5, 11, 17];
        for dof in drill_dofs {
            k_local[dof][dof] = drill_stiff;
        }

        // --- Transform to global 18x18 matrix ---
        let r_mat = [
            [e1[0], e2[0], e3[0]],
            [e1[1], e2[1], e3[1]],
            [e1[2], e2[2], e3[2]],
        ];

        let mut k_global = vec![vec![0.0; 18]; 18];
        for n1 in 0..3 {
            for n2 in 0..3 {
                let mut local_block = [[0.0; 6]; 6];
                for r in 0..6 {
                    for c in 0..6 {
                        local_block[r][c] = k_local[n1 * 6 + r][n2 * 6 + c];
                    }
                }

                let mut global_block = [[0.0; 6]; 6];
                for sub_r in 0..2 {
                    for sub_c in 0..2 {
                        let mut local_sub = [[0.0; 3]; 3];
                        for r in 0..3 {
                            for c in 0..3 {
                                local_sub[r][c] = local_block[sub_r * 3 + r][sub_c * 3 + c];
                            }
                        }

                        let mut temp = [[0.0; 3]; 3];
                        for r in 0..3 {
                            for c in 0..3 {
                                let mut sum = 0.0;
                                for k in 0..3 {
                                    sum += r_mat[r][k] * local_sub[k][c];
                                }
                                temp[r][c] = sum;
                            }
                        }

                        for r in 0..3 {
                            for c in 0..3 {
                                let mut sum = 0.0;
                                for k in 0..3 {
                                    sum += temp[r][k] * r_mat[c][k];
                                }
                                global_block[sub_r * 3 + r][sub_c * 3 + c] = sum;
                            }
                        }
                    }
                }

                for r in 0..6 {
                    for c in 0..6 {
                        k_global[n1 * 6 + r][n2 * 6 + c] = global_block[r][c];
                    }
                }
            }
        }

        Ok(k_global)
    }

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

        let p0 = node_coords[0].coords;
        let p1 = node_coords[1].coords;
        let p2 = node_coords[2].coords;

        let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let len_v1 = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
        let e1 = [v1[0] / len_v1, v1[1] / len_v1, v1[2] / len_v1];

        let v_n = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];
        let len_vn = (v_n[0].powi(2) + v_n[1].powi(2) + v_n[2].powi(2)).sqrt();
        let e3 = [v_n[0] / len_vn, v_n[1] / len_vn, v_n[2] / len_vn];
        let e2 = [
            e3[1] * e1[2] - e3[2] * e1[1],
            e3[2] * e1[0] - e3[0] * e1[2],
            e3[0] * e1[1] - e3[1] * e1[0],
        ];

        let x1 = v1[0] * e1[0] + v1[1] * e1[1] + v1[2] * e1[2];
        let y2 = v2[0] * e2[0] + v2[1] * e2[1] + v2[2] * e2[2];
        let area = 0.5 * x1 * y2.abs();

        let total_mass = density * self.thickness * area;

        let mut mass = vec![vec![0.0; 18]; 18];
        if lumped {
            let nodal_trans_mass = total_mass / 3.0;
            let nodal_rot_mass = nodal_trans_mass * self.thickness.powi(2) / 12.0;

            for n in 0..3 {
                let base = n * 6;
                mass[base][base] = nodal_trans_mass;
                mass[base + 1][base + 1] = nodal_trans_mass;
                mass[base + 2][base + 2] = nodal_trans_mass;
                mass[base + 3][base + 3] = nodal_rot_mass;
                mass[base + 4][base + 4] = nodal_rot_mass;
                mass[base + 5][base + 5] = nodal_rot_mass;
            }
        } else {
            let trans_factor = total_mass / 12.0;
            for i in 0..3 {
                for j in 0..3 {
                    let base_i = i * 6;
                    let base_j = j * 6;
                    let diag = if i == j { 2.0 } else { 1.0 };
                    mass[base_i][base_j] = diag * trans_factor;
                    mass[base_i + 1][base_j + 1] = diag * trans_factor;
                    mass[base_i + 2][base_j + 2] = diag * trans_factor;
                }
            }
        }
        Ok(mass)
    }
}

pub struct ShellQuad4<'a> {
    pub nodes: &'a [NodeId; 4],
    pub thickness: f64,
}

impl<'a> Element for ShellQuad4<'a> {
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
        vec![
            "ux".to_string(),
            "uy".to_string(),
            "uz".to_string(),
            "theta_x".to_string(),
            "theta_y".to_string(),
            "theta_z".to_string(),
        ]
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

        if node_coords.len() != 4 {
            return Err(ElementError::InvalidNodeCount {
                expected: 4,
                actual: node_coords.len(),
            });
        }

        let p0 = node_coords[0].coords;
        let p1 = node_coords[1].coords;
        let p2 = node_coords[2].coords;
        let p3 = node_coords[3].coords;

        // e1 connects midpoint of 0-3 and midpoint of 1-2
        let mid12 = [
            0.5 * (p1[0] + p2[0]),
            0.5 * (p1[1] + p2[1]),
            0.5 * (p1[2] + p2[2]),
        ];
        let mid03 = [
            0.5 * (p0[0] + p3[0]),
            0.5 * (p0[1] + p3[1]),
            0.5 * (p0[2] + p3[2]),
        ];
        let v1 = [
            mid12[0] - mid03[0],
            mid12[1] - mid03[1],
            mid12[2] - mid03[2],
        ];
        let len_v1 = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
        if len_v1 <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }
        let e1 = [v1[0] / len_v1, v1[1] / len_v1, v1[2] / len_v1];

        // Normal from diagonals: d13 x d24
        let d13 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let d24 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];
        let vn = [
            d13[1] * d24[2] - d13[2] * d24[1],
            d13[2] * d24[0] - d13[0] * d24[2],
            d13[0] * d24[1] - d13[1] * d24[0],
        ];
        let len_vn = (vn[0].powi(2) + vn[1].powi(2) + vn[2].powi(2)).sqrt();
        if len_vn <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }
        let e3 = [vn[0] / len_vn, vn[1] / len_vn, vn[2] / len_vn];
        let e2 = [
            e3[1] * e1[2] - e3[2] * e1[1],
            e3[2] * e1[0] - e3[0] * e1[2],
            e3[0] * e1[1] - e3[1] * e1[0],
        ];

        // Local coordinates
        let mut x = [0.0; 4];
        let mut y = [0.0; 4];
        for i in 0..4 {
            let dx = node_coords[i].coords[0] - p0[0];
            let dy = node_coords[i].coords[1] - p0[1];
            let dz = node_coords[i].coords[2] - p0[2];
            x[i] = dx * e1[0] + dy * e1[1] + dz * e1[2];
            y[i] = dx * e2[0] + dy * e2[1] + dz * e2[2];
        }

        let area = 0.5 * ((x[2] - x[0]) * (y[3] - y[1]) - (x[3] - x[1]) * (y[2] - y[0])).abs();
        if area <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }

        let shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio));

        // Midpoint difference terms for tying point coordinates
        let dx_10 = x[1] - x[0];
        let dy_10 = y[1] - y[0];
        let dx_21 = x[2] - x[1];
        let dy_21 = y[2] - y[1];
        let dx_32 = x[3] - x[2];
        let dy_32 = y[3] - y[2];
        let dx_03 = x[0] - x[3];
        let dy_03 = y[0] - y[3];

        let gp = [-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()];
        let mut k_membrane = vec![vec![0.0; 8]; 8];
        let mut k_bending = vec![vec![0.0; 12]; 12];

        let dm_factor = young_modulus * self.thickness / (1.0 - poisson_ratio.powi(2));
        let dm = [
            [dm_factor, dm_factor * poisson_ratio, 0.0],
            [dm_factor * poisson_ratio, dm_factor, 0.0],
            [0.0, 0.0, dm_factor * 0.5 * (1.0 - poisson_ratio)],
        ];

        let db_factor =
            young_modulus * self.thickness.powi(3) / (12.0 * (1.0 - poisson_ratio.powi(2)));
        let db = [
            [db_factor, db_factor * poisson_ratio, 0.0],
            [db_factor * poisson_ratio, db_factor, 0.0],
            [0.0, 0.0, db_factor * 0.5 * (1.0 - poisson_ratio)],
        ];

        let ds = shear_modulus * self.thickness * (5.0 / 6.0);

        for &xi in &gp {
            for &eta in &gp {
                let _n_val = [
                    0.25 * (1.0 - xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 + eta),
                    0.25 * (1.0 - xi) * (1.0 + eta),
                ];
                let d_n_dxi = [
                    -0.25 * (1.0 - eta),
                    0.25 * (1.0 - eta),
                    0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta),
                ];
                let d_n_deta = [
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                    0.25 * (1.0 + xi),
                    0.25 * (1.0 - xi),
                ];

                let mut dx_dxi = 0.0;
                let mut dy_dxi = 0.0;
                let mut dx_deta = 0.0;
                let mut dy_deta = 0.0;
                for i in 0..4 {
                    dx_dxi += d_n_dxi[i] * x[i];
                    dy_dxi += d_n_dxi[i] * y[i];
                    dx_deta += d_n_deta[i] * x[i];
                    dy_deta += d_n_deta[i] * y[i];
                }
                let detj = dx_dxi * dy_deta - dy_dxi * dx_deta;
                let detj_abs = detj.abs();
                if detj_abs <= f64::EPSILON {
                    return Err(ElementError::DegenerateGeometry);
                }

                let inv_j = [
                    [dy_deta / detj, -dy_dxi / detj],
                    [-dx_deta / detj, dx_dxi / detj],
                ];

                let mut nx = [0.0; 4];
                let mut ny = [0.0; 4];
                for i in 0..4 {
                    nx[i] = inv_j[0][0] * d_n_dxi[i] + inv_j[0][1] * d_n_deta[i];
                    ny[i] = inv_j[1][0] * d_n_dxi[i] + inv_j[1][1] * d_n_deta[i];
                }

                // --- 1. Membrane ---
                let mut bm = [[0.0; 8]; 3];
                for i in 0..4 {
                    bm[0][2 * i] = nx[i];
                    bm[1][2 * i + 1] = ny[i];
                    bm[2][2 * i] = ny[i];
                    bm[2][2 * i + 1] = nx[i];
                }

                let d_w = detj_abs;
                for r in 0..8 {
                    for c in 0..8 {
                        let mut db_val = [0.0; 3];
                        db_val[0] = dm[0][0] * bm[0][c] + dm[0][1] * bm[1][c];
                        db_val[1] = dm[1][0] * bm[0][c] + dm[1][1] * bm[1][c];
                        db_val[2] = dm[2][2] * bm[2][c];
                        k_membrane[r][c] += d_w
                            * (bm[0][r] * db_val[0] + bm[1][r] * db_val[1] + bm[2][r] * db_val[2]);
                    }
                }

                // --- 2. Bending ---
                let mut bb = [[0.0; 12]; 3];
                for i in 0..4 {
                    bb[0][3 * i + 2] = -nx[i];
                    bb[1][3 * i + 1] = ny[i];
                    bb[2][3 * i + 1] = nx[i];
                    bb[2][3 * i + 2] = -ny[i];
                }

                for r in 0..12 {
                    for c in 0..12 {
                        let mut db_val = [0.0; 3];
                        db_val[0] = db[0][0] * bb[0][c] + db[0][1] * bb[1][c];
                        db_val[1] = db[1][0] * bb[0][c] + db[1][1] * bb[1][c];
                        db_val[2] = db[2][2] * bb[2][c];
                        k_bending[r][c] += d_w
                            * (bb[0][r] * db_val[0] + bb[1][r] * db_val[1] + bb[2][r] * db_val[2]);
                    }
                }

                // --- 3. Transverse Shear (MITC4) ---
                let mut ha = [0.0; 12];
                ha[0] = -0.5;
                ha[1] = 0.25 * dy_10;
                ha[2] = -0.25 * dx_10;
                ha[3] = 0.5;
                ha[4] = 0.25 * dy_10;
                ha[5] = -0.25 * dx_10;

                let mut hb = [0.0; 12];
                hb[3] = -0.5;
                hb[4] = 0.25 * dy_21;
                hb[5] = -0.25 * dx_21;
                hb[6] = 0.5;
                hb[7] = 0.25 * dy_21;
                hb[8] = -0.25 * dx_21;

                let mut hc = [0.0; 12];
                hc[9] = -0.5;
                hc[10] = 0.25 * dy_32;
                hc[11] = -0.25 * dx_32;
                hc[6] = 0.5;
                hc[7] = 0.25 * dy_32;
                hc[8] = -0.25 * dx_32;

                let mut hd = [0.0; 12];
                hd[0] = -0.5;
                hd[1] = 0.25 * dy_03;
                hd[2] = -0.25 * dx_03;
                hd[9] = 0.5;
                hd[10] = 0.25 * dy_03;
                hd[11] = -0.25 * dx_03;

                let mut hs = [[0.0; 12]; 2];
                for i in 0..12 {
                    hs[0][i] = 0.5 * (1.0 - eta) * ha[i] + 0.5 * (1.0 + eta) * hc[i];
                    hs[1][i] = 0.5 * (1.0 - xi) * hd[i] + 0.5 * (1.0 + xi) * hb[i];
                }

                let mut bs = [[0.0; 12]; 2];
                for i in 0..12 {
                    bs[0][i] = inv_j[0][0] * hs[0][i] + inv_j[1][0] * hs[1][i];
                    bs[1][i] = inv_j[0][1] * hs[0][i] + inv_j[1][1] * hs[1][i];
                }

                for r in 0..12 {
                    for c in 0..12 {
                        k_bending[r][c] += d_w * ds * (bs[0][r] * bs[0][c] + bs[1][r] * bs[1][c]);
                    }
                }
            }
        }

        let drill_stiff = 1e-4 * shear_modulus * self.thickness * area;

        let mut k_local = vec![vec![0.0; 24]; 24];
        let mem_dofs = [0, 1, 6, 7, 12, 13, 18, 19];
        for r in 0..8 {
            for c in 0..8 {
                k_local[mem_dofs[r]][mem_dofs[c]] = k_membrane[r][c];
            }
        }

        let bend_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22];
        for r in 0..12 {
            for c in 0..12 {
                k_local[bend_dofs[r]][bend_dofs[c]] = k_bending[r][c];
            }
        }

        let drill_dofs = [5, 11, 17, 23];
        for dof in drill_dofs {
            k_local[dof][dof] = drill_stiff;
        }

        let r_mat = [
            [e1[0], e2[0], e3[0]],
            [e1[1], e2[1], e3[1]],
            [e1[2], e2[2], e3[2]],
        ];

        let mut k_global = vec![vec![0.0; 24]; 24];
        for n1 in 0..4 {
            for n2 in 0..4 {
                let mut local_block = [[0.0; 6]; 6];
                for r in 0..6 {
                    for c in 0..6 {
                        local_block[r][c] = k_local[n1 * 6 + r][n2 * 6 + c];
                    }
                }

                let mut global_block = [[0.0; 6]; 6];
                for sub_r in 0..2 {
                    for sub_c in 0..2 {
                        let mut local_sub = [[0.0; 3]; 3];
                        for r in 0..3 {
                            for c in 0..3 {
                                local_sub[r][c] = local_block[sub_r * 3 + r][sub_c * 3 + c];
                            }
                        }

                        let mut temp = [[0.0; 3]; 3];
                        for r in 0..3 {
                            for c in 0..3 {
                                let mut sum = 0.0;
                                for k in 0..3 {
                                    sum += r_mat[r][k] * local_sub[k][c];
                                }
                                temp[r][c] = sum;
                            }
                        }

                        for r in 0..3 {
                            for c in 0..3 {
                                let mut sum = 0.0;
                                for k in 0..3 {
                                    sum += temp[r][k] * r_mat[c][k];
                                }
                                global_block[sub_r * 3 + r][sub_c * 3 + c] = sum;
                            }
                        }
                    }
                }

                for r in 0..6 {
                    for c in 0..6 {
                        k_global[n1 * 6 + r][n2 * 6 + c] = global_block[r][c];
                    }
                }
            }
        }

        Ok(k_global)
    }

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

        let p0 = node_coords[0].coords;
        let p1 = node_coords[1].coords;
        let p2 = node_coords[2].coords;
        let p3 = node_coords[3].coords;

        let mid12 = [
            0.5 * (p1[0] + p2[0]),
            0.5 * (p1[1] + p2[1]),
            0.5 * (p1[2] + p2[2]),
        ];
        let mid03 = [
            0.5 * (p0[0] + p3[0]),
            0.5 * (p0[1] + p3[1]),
            0.5 * (p0[2] + p3[2]),
        ];
        let v1 = [
            mid12[0] - mid03[0],
            mid12[1] - mid03[1],
            mid12[2] - mid03[2],
        ];
        let len_v1 = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
        if len_v1 <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }
        let e1 = [v1[0] / len_v1, v1[1] / len_v1, v1[2] / len_v1];

        let d13 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let d24 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];
        let vn = [
            d13[1] * d24[2] - d13[2] * d24[1],
            d13[2] * d24[0] - d13[0] * d24[2],
            d13[0] * d24[1] - d13[1] * d24[0],
        ];
        let len_vn = (vn[0].powi(2) + vn[1].powi(2) + vn[2].powi(2)).sqrt();
        if len_vn <= f64::EPSILON {
            return Err(ElementError::DegenerateGeometry);
        }
        let e3 = [vn[0] / len_vn, vn[1] / len_vn, vn[2] / len_vn];
        let e2 = [
            e3[1] * e1[2] - e3[2] * e1[1],
            e3[2] * e1[0] - e3[0] * e1[2],
            e3[0] * e1[1] - e3[1] * e1[0],
        ];

        let mut x = [0.0; 4];
        let mut y = [0.0; 4];
        for i in 0..4 {
            let dx = node_coords[i].coords[0] - p0[0];
            let dy = node_coords[i].coords[1] - p0[1];
            let dz = node_coords[i].coords[2] - p0[2];
            x[i] = dx * e1[0] + dy * e1[1] + dz * e1[2];
            y[i] = dx * e2[0] + dy * e2[1] + dz * e2[2];
        }

        let area = 0.5 * ((x[2] - x[0]) * (y[3] - y[1]) - (x[3] - x[1]) * (y[2] - y[0])).abs();
        let total_mass = density * self.thickness * area;

        let mut mass = vec![vec![0.0; 24]; 24];
        if lumped {
            let nodal_trans_mass = total_mass / 4.0;
            let nodal_rot_mass = nodal_trans_mass * self.thickness.powi(2) / 12.0;

            for n in 0..4 {
                let base = n * 6;
                mass[base][base] = nodal_trans_mass;
                mass[base + 1][base + 1] = nodal_trans_mass;
                mass[base + 2][base + 2] = nodal_trans_mass;
                mass[base + 3][base + 3] = nodal_rot_mass;
                mass[base + 4][base + 4] = nodal_rot_mass;
                mass[base + 5][base + 5] = nodal_rot_mass;
            }
        } else {
            let gp = [-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()];
            let density_factor = density * self.thickness;
            let mut m_local = vec![vec![0.0; 24]; 24];

            for &xi in &gp {
                for &eta in &gp {
                    let n_val = [
                        0.25 * (1.0 - xi) * (1.0 - eta),
                        0.25 * (1.0 + xi) * (1.0 - eta),
                        0.25 * (1.0 + xi) * (1.0 + eta),
                        0.25 * (1.0 - xi) * (1.0 + eta),
                    ];
                    let d_n_dxi = [
                        -0.25 * (1.0 - eta),
                        0.25 * (1.0 - eta),
                        0.25 * (1.0 + eta),
                        -0.25 * (1.0 + eta),
                    ];
                    let d_n_deta = [
                        -0.25 * (1.0 - xi),
                        -0.25 * (1.0 + xi),
                        0.25 * (1.0 + xi),
                        0.25 * (1.0 - xi),
                    ];
                    let mut dx_dxi = 0.0;
                    let mut dy_dxi = 0.0;
                    let mut dx_deta = 0.0;
                    let mut dy_deta = 0.0;
                    for i in 0..4 {
                        dx_dxi += d_n_dxi[i] * x[i];
                        dy_dxi += d_n_dxi[i] * y[i];
                        dx_deta += d_n_deta[i] * x[i];
                        dy_deta += d_n_deta[i] * y[i];
                    }
                    let detj = dx_dxi * dy_deta - dy_dxi * dx_deta;
                    let factor = density_factor * detj.abs();

                    for i in 0..4 {
                        for j in 0..4 {
                            let val = n_val[i] * n_val[j] * factor;
                            m_local[i * 6][j * 6] += val;
                            m_local[i * 6 + 1][j * 6 + 1] += val;
                            m_local[i * 6 + 2][j * 6 + 2] += val;

                            let val_rot = val * self.thickness.powi(2) / 12.0;
                            m_local[i * 6 + 3][j * 6 + 3] += val_rot;
                            m_local[i * 6 + 4][j * 6 + 4] += val_rot;
                            m_local[i * 6 + 5][j * 6 + 5] += val_rot;
                        }
                    }
                }
            }

            let r_mat = [
                [e1[0], e2[0], e3[0]],
                [e1[1], e2[1], e3[1]],
                [e1[2], e2[2], e3[2]],
            ];

            for n1 in 0..4 {
                for n2 in 0..4 {
                    let mut local_block = [[0.0; 6]; 6];
                    for r in 0..6 {
                        for c in 0..6 {
                            local_block[r][c] = m_local[n1 * 6 + r][n2 * 6 + c];
                        }
                    }

                    let mut global_block = [[0.0; 6]; 6];
                    for sub_r in 0..2 {
                        for sub_c in 0..2 {
                            let mut local_sub = [[0.0; 3]; 3];
                            for r in 0..3 {
                                for c in 0..3 {
                                    local_sub[r][c] = local_block[sub_r * 3 + r][sub_c * 3 + c];
                                }
                            }

                            let mut temp = [[0.0; 3]; 3];
                            for r in 0..3 {
                                for c in 0..3 {
                                    let mut sum = 0.0;
                                    for k in 0..3 {
                                        sum += r_mat[r][k] * local_sub[k][c];
                                    }
                                    temp[r][c] = sum;
                                }
                            }

                            for r in 0..3 {
                                for c in 0..3 {
                                    let mut sum = 0.0;
                                    for k in 0..3 {
                                        sum += temp[r][k] * r_mat[c][k];
                                    }
                                    global_block[sub_r * 3 + r][sub_c * 3 + c] = sum;
                                }
                            }
                        }
                    }

                    for r in 0..6 {
                        for c in 0..6 {
                            mass[n1 * 6 + r][n2 * 6 + c] = global_block[r][c];
                        }
                    }
                }
            }
        }

        Ok(mass)
    }
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

    #[test]
    fn test_shell_tri3_element() {
        let nodes = [0, 1, 2];
        let el = ShellTri3 {
            nodes: &nodes,
            thickness: 0.1,
        };

        // Triangular shell in xy-plane
        let coords = [
            Point3::new([0.0, 0.0, 0.0]),
            Point3::new([1.0, 0.0, 0.0]),
            Point3::new([0.0, 1.0, 0.0]),
        ];

        let mut properties = BTreeMap::new();
        properties.insert("young_modulus".to_string(), 200e9);
        properties.insert("poisson_ratio".to_string(), 0.3);

        let k = el.local_stiffness(&coords, &properties).unwrap();
        assert_eq!(k.len(), 18);

        // Check stiffness matrix symmetry
        for (r, row) in k.iter().enumerate().take(18) {
            for (c, &val) in row.iter().enumerate().take(18) {
                assert!(
                    (val - k[c][r]).abs() < 1e-3,
                    "Symmetry failed at ({}, {})",
                    r,
                    c
                );
            }
        }
    }

    #[test]
    fn test_shell_quad4_element() {
        let nodes = [0, 1, 2, 3];
        let el = ShellQuad4 {
            nodes: &nodes,
            thickness: 0.1,
        };

        // Square shell of size 2.0 x 2.0 in xy-plane
        let coords = [
            Point3::new([0.0, 0.0, 0.0]),
            Point3::new([2.0, 0.0, 0.0]),
            Point3::new([2.0, 2.0, 0.0]),
            Point3::new([0.0, 2.0, 0.0]),
        ];

        let mut properties = BTreeMap::new();
        properties.insert("young_modulus".to_string(), 200e9);
        properties.insert("poisson_ratio".to_string(), 0.3);

        let k = el.local_stiffness(&coords, &properties).unwrap();
        assert_eq!(k.len(), 24);

        // Check symmetry of 24x24 matrix
        for (r, row) in k.iter().enumerate().take(24) {
            for (c, &k_rc) in row.iter().enumerate().take(24) {
                assert!(
                    (k_rc - k[c][r]).abs() < 1e-2,
                    "Symmetry failed at ({}, {}): k[r][c]={}, k[c][r]={}",
                    r,
                    c,
                    k_rc,
                    k[c][r]
                );
            }
        }

        // Check mass matrix
        let m = el.local_mass(&coords, 7800.0, true).unwrap();
        assert_eq!(m.len(), 24);
        // Total mass = 7800 * 0.1 * 4.0 = 3120 kg. Lumped mass = 780 kg per node.
        assert!((m[0][0] - 780.0).abs() < 1.0);
    }
}
