#![allow(clippy::needless_range_loop)]

use crate::fem::element::{Element, ElementError};
use crate::fem::quadrature::QuadratureRule;
use crate::mesh::{NodeId, Point3};
use std::collections::BTreeMap;

pub struct PoissonTri6<'a> {
    pub nodes: &'a [NodeId; 6],
}

impl<'a> Element for PoissonTri6<'a> {
    fn spatial_dimension(&self) -> usize {
        2
    }

    fn node_count(&self) -> usize {
        6
    }

    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }

    fn active_fields(&self) -> Vec<String> {
        vec!["u".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let conductivity = *properties.get("conductivity").ok_or_else(|| {
            ElementError::MissingProperty("conductivity".to_string())
        })?;

        if node_coords.len() != 6 {
            return Err(ElementError::InvalidNodeCount {
                expected: 6,
                actual: node_coords.len(),
            });
        }

        let mut stiffness = vec![vec![0.0; 6]; 6];
        let rule = QuadratureRule::triangle(2);

        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let (_, local_derivs) = tri6_shape_and_derivs(xi, eta);

            let mut jacobian = [[0.0; 2]; 2];
            for j in 0..2 {
                for k in 0..2 {
                    let mut val = 0.0;
                    for i in 0..6 {
                        val += local_derivs[i][j] * node_coords[i].coords[k];
                    }
                    jacobian[j][k] = val;
                }
            }

            let det_j = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
            if det_j <= f64::EPSILON {
                return Err(ElementError::DegenerateGeometry);
            }

            let inv_j = [
                [jacobian[1][1] / det_j, -jacobian[0][1] / det_j],
                [-jacobian[1][0] / det_j, jacobian[0][0] / det_j],
            ];

            let mut global_derivs = [[0.0; 2]; 6];
            for i in 0..6 {
                for k in 0..2 {
                    global_derivs[i][k] = inv_j[k][0] * local_derivs[i][0]
                        + inv_j[k][1] * local_derivs[i][1];
                }
            }

            let jacobian_det = 2.0 * det_j;

            for i in 0..6 {
                for j in 0..6 {
                    let grad_dot = global_derivs[i][0] * global_derivs[j][0]
                        + global_derivs[i][1] * global_derivs[j][1];
                    stiffness[i][j] += gp.weight * jacobian_det * conductivity * grad_dot;
                }
            }
        }

        Ok(stiffness)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 6 {
            return Err(ElementError::InvalidNodeCount {
                expected: 6,
                actual: node_coords.len(),
            });
        }

        let mut mass = vec![vec![0.0; 6]; 6];
        let rule = QuadratureRule::triangle(2);

        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let (shape, local_derivs) = tri6_shape_and_derivs(xi, eta);

            let mut jacobian = [[0.0; 2]; 2];
            for j in 0..2 {
                for k in 0..2 {
                    let mut val = 0.0;
                    for i in 0..6 {
                        val += local_derivs[i][j] * node_coords[i].coords[k];
                    }
                    jacobian[j][k] = val;
                }
            }

            let det_j = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
            if det_j <= f64::EPSILON {
                return Err(ElementError::DegenerateGeometry);
            }
            let jacobian_det = 2.0 * det_j;

            for i in 0..6 {
                for j in 0..6 {
                    mass[i][j] += gp.weight * jacobian_det * density * shape[i] * shape[j];
                }
            }
        }

        if lumped {
            let mut diag_sums = [0.0; 6];
            for i in 0..6 {
                let mut row_sum = 0.0;
                for j in 0..6 {
                    row_sum += mass[i][j];
                }
                diag_sums[i] = row_sum;
            }
            let mut lumped_mass = vec![vec![0.0; 6]; 6];
            for i in 0..6 {
                lumped_mass[i][i] = diag_sums[i];
            }
            Ok(lumped_mass)
        } else {
            Ok(mass)
        }
    }
}

pub struct PoissonTet10<'a> {
    pub nodes: &'a [NodeId; 10],
}

impl<'a> Element for PoissonTet10<'a> {
    fn spatial_dimension(&self) -> usize {
        3
    }

    fn node_count(&self) -> usize {
        10
    }

    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }

    fn active_fields(&self) -> Vec<String> {
        vec!["u".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let conductivity = *properties.get("conductivity").ok_or_else(|| {
            ElementError::MissingProperty("conductivity".to_string())
        })?;

        if node_coords.len() != 10 {
            return Err(ElementError::InvalidNodeCount {
                expected: 10,
                actual: node_coords.len(),
            });
        }

        let mut stiffness = vec![vec![0.0; 10]; 10];
        let rule = QuadratureRule::tetrahedron(2);

        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let zeta = gp.coords[2];
            let (_, local_derivs) = tet10_shape_and_derivs(xi, eta, zeta);

            let mut jacobian = [[0.0; 3]; 3];
            for j in 0..3 {
                for k in 0..3 {
                    let mut val = 0.0;
                    for i in 0..10 {
                        val += local_derivs[i][j] * node_coords[i].coords[k];
                    }
                    jacobian[j][k] = val;
                }
            }

            let det_j = determinant_3(jacobian);
            if det_j.abs() <= f64::EPSILON {
                return Err(ElementError::DegenerateGeometry);
            }

            let inv_j = inverse_3(jacobian, det_j);

            let mut global_derivs = [[0.0; 3]; 10];
            for i in 0..10 {
                for k in 0..3 {
                    global_derivs[i][k] = inv_j[k][0] * local_derivs[i][0]
                        + inv_j[k][1] * local_derivs[i][1]
                        + inv_j[k][2] * local_derivs[i][2];
                }
            }

            let jacobian_det = 6.0 * det_j.abs();

            for i in 0..10 {
                for j in 0..10 {
                    let grad_dot = global_derivs[i][0] * global_derivs[j][0]
                        + global_derivs[i][1] * global_derivs[j][1]
                        + global_derivs[i][2] * global_derivs[j][2];
                    stiffness[i][j] += gp.weight * jacobian_det * conductivity * grad_dot;
                }
            }
        }

        Ok(stiffness)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 10 {
            return Err(ElementError::InvalidNodeCount {
                expected: 10,
                actual: node_coords.len(),
            });
        }

        let mut mass = vec![vec![0.0; 10]; 10];
        let rule = QuadratureRule::tetrahedron(2);

        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let zeta = gp.coords[2];
            let (shape, local_derivs) = tet10_shape_and_derivs(xi, eta, zeta);

            let mut jacobian = [[0.0; 3]; 3];
            for j in 0..3 {
                for k in 0..3 {
                    let mut val = 0.0;
                    for i in 0..10 {
                        val += local_derivs[i][j] * node_coords[i].coords[k];
                    }
                    jacobian[j][k] = val;
                }
            }

            let det_j = determinant_3(jacobian);
            if det_j.abs() <= f64::EPSILON {
                return Err(ElementError::DegenerateGeometry);
            }
            let jacobian_det = 6.0 * det_j.abs();

            for i in 0..10 {
                for j in 0..10 {
                    mass[i][j] += gp.weight * jacobian_det * density * shape[i] * shape[j];
                }
            }
        }

        if lumped {
            let mut diag_sums = [0.0; 10];
            for i in 0..10 {
                let mut row_sum = 0.0;
                for j in 0..10 {
                    row_sum += mass[i][j];
                }
                diag_sums[i] = row_sum;
            }
            let mut lumped_mass = vec![vec![0.0; 10]; 10];
            for i in 0..10 {
                lumped_mass[i][i] = diag_sums[i];
            }
            Ok(lumped_mass)
        } else {
            Ok(mass)
        }
    }
}

pub struct ElasticityTri6<'a> {
    pub nodes: &'a [NodeId; 6],
}

impl<'a> Element for ElasticityTri6<'a> {
    fn spatial_dimension(&self) -> usize {
        2
    }

    fn node_count(&self) -> usize {
        6
    }

    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }

    fn active_fields(&self) -> Vec<String> {
        vec!["ux".to_string(), "uy".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let young_modulus = *properties.get("young_modulus").ok_or_else(|| {
            ElementError::MissingProperty("young_modulus".to_string())
        })?;
        let poisson_ratio = *properties.get("poisson_ratio").ok_or_else(|| {
            ElementError::MissingProperty("poisson_ratio".to_string())
        })?;
        let thickness = *properties.get("thickness").unwrap_or(&1.0);
        let model_val = *properties.get("model").unwrap_or(&0.0);

        if node_coords.len() != 6 {
            return Err(ElementError::InvalidNodeCount {
                expected: 6,
                actual: node_coords.len(),
            });
        }

        let d = if model_val == 1.0 {
            let factor = young_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
            [
                [factor * (1.0 - poisson_ratio), factor * poisson_ratio, 0.0],
                [factor * poisson_ratio, factor * (1.0 - poisson_ratio), 0.0],
                [0.0, 0.0, factor * 0.5 * (1.0 - 2.0 * poisson_ratio)],
            ]
        } else {
            let factor = young_modulus / (1.0 - poisson_ratio.powi(2));
            [
                [factor, factor * poisson_ratio, 0.0],
                [factor * poisson_ratio, factor, 0.0],
                [0.0, 0.0, factor * 0.5 * (1.0 - poisson_ratio)],
            ]
        };

        let mut stiffness = vec![vec![0.0; 12]; 12];
        let rule = QuadratureRule::triangle(2);

        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let (_, local_derivs) = tri6_shape_and_derivs(xi, eta);

            let mut jacobian = [[0.0; 2]; 2];
            for j in 0..2 {
                for k in 0..2 {
                    let mut val = 0.0;
                    for i in 0..6 {
                        val += local_derivs[i][j] * node_coords[i].coords[k];
                    }
                    jacobian[j][k] = val;
                }
            }

            let det_j = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
            if det_j <= f64::EPSILON {
                return Err(ElementError::DegenerateGeometry);
            }

            let inv_j = [
                [jacobian[1][1] / det_j, -jacobian[0][1] / det_j],
                [-jacobian[1][0] / det_j, jacobian[0][0] / det_j],
            ];

            let mut global_derivs = [[0.0; 2]; 6];
            for i in 0..6 {
                for k in 0..2 {
                    global_derivs[i][k] = inv_j[k][0] * local_derivs[i][0]
                        + inv_j[k][1] * local_derivs[i][1];
                }
            }

            let mut b = [[0.0; 12]; 3];
            for i in 0..6 {
                let base = i * 2;
                let gx = global_derivs[i][0];
                let gy = global_derivs[i][1];
                b[0][base] = gx;
                b[1][base + 1] = gy;
                b[2][base] = gy;
                b[2][base + 1] = gx;
            }

            let jacobian_det = 2.0 * det_j;
            let factor = gp.weight * jacobian_det * thickness;

            for r in 0..12 {
                let mut db = [0.0; 3];
                for c in 0..12 {
                    db[0] = d[0][0] * b[0][c] + d[0][1] * b[1][c];
                    db[1] = d[1][0] * b[0][c] + d[1][1] * b[1][c];
                    db[2] = d[2][2] * b[2][c];

                    let val = b[0][r] * db[0] + b[1][r] * db[1] + b[2][r] * db[2];
                    stiffness[r][c] += factor * val;
                }
            }
        }

        Ok(stiffness)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 6 {
            return Err(ElementError::InvalidNodeCount {
                expected: 6,
                actual: node_coords.len(),
            });
        }

        let mut mass = vec![vec![0.0; 12]; 12];
        let rule = QuadratureRule::triangle(2);

        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let (shape, local_derivs) = tri6_shape_and_derivs(xi, eta);

            let mut jacobian = [[0.0; 2]; 2];
            for j in 0..2 {
                for k in 0..2 {
                    let mut val = 0.0;
                    for i in 0..6 {
                        val += local_derivs[i][j] * node_coords[i].coords[k];
                    }
                    jacobian[j][k] = val;
                }
            }

            let det_j = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
            if det_j <= f64::EPSILON {
                return Err(ElementError::DegenerateGeometry);
            }
            let jacobian_det = 2.0 * det_j;
            let factor = gp.weight * jacobian_det * density;

            for i in 0..6 {
                for j in 0..6 {
                    let m_val = factor * shape[i] * shape[j];
                    mass[i * 2][j * 2] += m_val;
                    mass[i * 2 + 1][j * 2 + 1] += m_val;
                }
            }
        }

        if lumped {
            let mut diag_sums = [0.0; 12];
            for i in 0..12 {
                let mut row_sum = 0.0;
                for j in 0..12 {
                    row_sum += mass[i][j];
                }
                diag_sums[i] = row_sum;
            }
            let mut lumped_mass = vec![vec![0.0; 12]; 12];
            for i in 0..12 {
                lumped_mass[i][i] = diag_sums[i];
            }
            Ok(lumped_mass)
        } else {
            Ok(mass)
        }
    }
}

pub struct ElasticityTet10<'a> {
    pub nodes: &'a [NodeId; 10],
}

impl<'a> Element for ElasticityTet10<'a> {
    fn spatial_dimension(&self) -> usize {
        3
    }

    fn node_count(&self) -> usize {
        10
    }

    fn nodes(&self) -> &[NodeId] {
        self.nodes
    }

    fn active_fields(&self) -> Vec<String> {
        vec!["ux".to_string(), "uy".to_string(), "uz".to_string()]
    }

    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        let young_modulus = *properties.get("young_modulus").ok_or_else(|| {
            ElementError::MissingProperty("young_modulus".to_string())
        })?;
        let poisson_ratio = *properties.get("poisson_ratio").ok_or_else(|| {
            ElementError::MissingProperty("poisson_ratio".to_string())
        })?;

        if node_coords.len() != 10 {
            return Err(ElementError::InvalidNodeCount {
                expected: 10,
                actual: node_coords.len(),
            });
        }

        let mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
        let lambda = (young_modulus * poisson_ratio) / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));

        let d = [
            [lambda + 2.0 * mu, lambda, lambda, 0.0, 0.0, 0.0],
            [lambda, lambda + 2.0 * mu, lambda, 0.0, 0.0, 0.0],
            [lambda, lambda, lambda + 2.0 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ];

        let mut stiffness = vec![vec![0.0; 30]; 30];
        let rule = QuadratureRule::tetrahedron(2);

        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let zeta = gp.coords[2];
            let (_, local_derivs) = tet10_shape_and_derivs(xi, eta, zeta);

            let mut jacobian = [[0.0; 3]; 3];
            for j in 0..3 {
                for k in 0..3 {
                    let mut val = 0.0;
                    for i in 0..10 {
                        val += local_derivs[i][j] * node_coords[i].coords[k];
                    }
                    jacobian[j][k] = val;
                }
            }

            let det_j = determinant_3(jacobian);
            if det_j.abs() <= f64::EPSILON {
                return Err(ElementError::DegenerateGeometry);
            }

            let inv_j = inverse_3(jacobian, det_j);

            let mut global_derivs = [[0.0; 3]; 10];
            for i in 0..10 {
                for k in 0..3 {
                    global_derivs[i][k] = inv_j[k][0] * local_derivs[i][0]
                        + inv_j[k][1] * local_derivs[i][1]
                        + inv_j[k][2] * local_derivs[i][2];
                }
            }

            let mut b = [[0.0; 30]; 6];
            for i in 0..10 {
                let base = i * 3;
                let gx = global_derivs[i][0];
                let gy = global_derivs[i][1];
                let gz = global_derivs[i][2];

                b[0][base] = gx;
                b[1][base + 1] = gy;
                b[2][base + 2] = gz;

                b[3][base] = gy;
                b[3][base + 1] = gx;

                b[4][base + 1] = gz;
                b[4][base + 2] = gy;

                b[5][base] = gz;
                b[5][base + 2] = gx;
            }

            let jacobian_det = 6.0 * det_j.abs();
            let factor = gp.weight * jacobian_det;

            for r in 0..30 {
                let mut db = [0.0; 6];
                for c in 0..30 {
                    db[0] = d[0][0] * b[0][c] + d[0][1] * b[1][c] + d[0][2] * b[2][c];
                    db[1] = d[1][0] * b[0][c] + d[1][1] * b[1][c] + d[1][2] * b[2][c];
                    db[2] = d[2][0] * b[0][c] + d[2][1] * b[1][c] + d[2][2] * b[2][c];
                    db[3] = d[3][3] * b[3][c];
                    db[4] = d[4][4] * b[4][c];
                    db[5] = d[5][5] * b[5][c];

                    let val = b[0][r] * db[0]
                        + b[1][r] * db[1]
                        + b[2][r] * db[2]
                        + b[3][r] * db[3]
                        + b[4][r] * db[4]
                        + b[5][r] * db[5];
                    stiffness[r][c] += factor * val;
                }
            }
        }

        Ok(stiffness)
    }

    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError> {
        if node_coords.len() != 10 {
            return Err(ElementError::InvalidNodeCount {
                expected: 10,
                actual: node_coords.len(),
            });
        }

        let mut mass = vec![vec![0.0; 30]; 30];
        let rule = QuadratureRule::tetrahedron(2);

        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let zeta = gp.coords[2];
            let (shape, local_derivs) = tet10_shape_and_derivs(xi, eta, zeta);

            let mut jacobian = [[0.0; 3]; 3];
            for j in 0..3 {
                for k in 0..3 {
                    let mut val = 0.0;
                    for i in 0..10 {
                        val += local_derivs[i][j] * node_coords[i].coords[k];
                    }
                    jacobian[j][k] = val;
                }
            }

            let det_j = determinant_3(jacobian);
            if det_j.abs() <= f64::EPSILON {
                return Err(ElementError::DegenerateGeometry);
            }
            let jacobian_det = 6.0 * det_j.abs();
            let factor = gp.weight * jacobian_det * density;

            for i in 0..10 {
                for j in 0..10 {
                    let m_val = factor * shape[i] * shape[j];
                    mass[i * 3][j * 3] += m_val;
                    mass[i * 3 + 1][j * 3 + 1] += m_val;
                    mass[i * 3 + 2][j * 3 + 2] += m_val;
                }
            }
        }

        if lumped {
            let mut diag_sums = [0.0; 30];
            for i in 0..30 {
                let mut row_sum = 0.0;
                for j in 0..30 {
                    row_sum += mass[i][j];
                }
                diag_sums[i] = row_sum;
            }
            let mut lumped_mass = vec![vec![0.0; 30]; 30];
            for i in 0..30 {
                lumped_mass[i][i] = diag_sums[i];
            }
            Ok(lumped_mass)
        } else {
            Ok(mass)
        }
    }
}

fn tri6_shape_and_derivs(xi: f64, eta: f64) -> (Vec<f64>, [[f64; 2]; 6]) {
    let l1 = 1.0 - xi - eta;
    let l2 = xi;
    let l3 = eta;

    let shape = vec![
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
    ];

    let mut derivs = [[0.0; 2]; 6];
    let d_l = |node_idx: usize, l_idx: usize| -> f64 {
        match (node_idx, l_idx) {
            (0, 0) => 4.0 * l1 - 1.0,
            (1, 1) => 4.0 * l2 - 1.0,
            (2, 2) => 4.0 * l3 - 1.0,
            (3, 0) => 4.0 * l2,
            (3, 1) => 4.0 * l1,
            (4, 1) => 4.0 * l3,
            (4, 2) => 4.0 * l2,
            (5, 2) => 4.0 * l1,
            (5, 0) => 4.0 * l3,
            _ => 0.0,
        }
    };

    for (i, deriv) in derivs.iter_mut().enumerate() {
        deriv[0] = -d_l(i, 0) + d_l(i, 1);
        deriv[1] = -d_l(i, 0) + d_l(i, 2);
    }

    (shape, derivs)
}

fn tet10_shape_and_derivs(xi: f64, eta: f64, zeta: f64) -> (Vec<f64>, [[f64; 3]; 10]) {
    let l1 = 1.0 - xi - eta - zeta;
    let l2 = xi;
    let l3 = eta;
    let l4 = zeta;

    let shape = vec![
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        l4 * (2.0 * l4 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
        4.0 * l1 * l4,
        4.0 * l2 * l4,
        4.0 * l3 * l4,
    ];

    let mut derivs = [[0.0; 3]; 10];
    let d_l = |node_idx: usize, l_idx: usize| -> f64 {
        match (node_idx, l_idx) {
            (0, 0) => 4.0 * l1 - 1.0,
            (1, 1) => 4.0 * l2 - 1.0,
            (2, 2) => 4.0 * l3 - 1.0,
            (3, 3) => 4.0 * l4 - 1.0,
            (4, 0) => 4.0 * l2,
            (4, 1) => 4.0 * l1,
            (5, 1) => 4.0 * l3,
            (5, 2) => 4.0 * l2,
            (6, 2) => 4.0 * l1,
            (6, 0) => 4.0 * l3,
            (7, 0) => 4.0 * l4,
            (7, 3) => 4.0 * l1,
            (8, 1) => 4.0 * l4,
            (8, 3) => 4.0 * l2,
            (9, 2) => 4.0 * l4,
            (9, 3) => 4.0 * l3,
            _ => 0.0,
        }
    };

    for (i, deriv) in derivs.iter_mut().enumerate() {
        deriv[0] = -d_l(i, 0) + d_l(i, 1);
        deriv[1] = -d_l(i, 0) + d_l(i, 2);
        deriv[2] = -d_l(i, 0) + d_l(i, 3);
    }

    (shape, derivs)
}

fn determinant_3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn inverse_3(matrix: [[f64; 3]; 3], determinant: f64) -> [[f64; 3]; 3] {
    let inv_det = 1.0 / determinant;
    [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri6_shape_partition_of_unity() {
        let rule = QuadratureRule::triangle(2);
        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let (shape, _) = tri6_shape_and_derivs(xi, eta);
            let sum: f64 = shape.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_tet10_shape_partition_of_unity() {
        let rule = QuadratureRule::tetrahedron(2);
        for gp in &rule.points {
            let xi = gp.coords[0];
            let eta = gp.coords[1];
            let zeta = gp.coords[2];
            let (shape, _) = tet10_shape_and_derivs(xi, eta, zeta);
            let sum: f64 = shape.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }
}
