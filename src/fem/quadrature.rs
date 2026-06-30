#[derive(Debug, Clone, PartialEq)]
pub struct GaussPoint {
    pub coords: Vec<f64>,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuadratureRule {
    pub points: Vec<GaussPoint>,
}

impl QuadratureRule {
    /// Returns 1D Gauss-Legendre quadrature rule on the reference domain [-1, 1].
    pub fn line(order: usize) -> Self {
        match order {
            1 => Self {
                points: vec![GaussPoint {
                    coords: vec![0.0],
                    weight: 2.0,
                }],
            },
            2 => Self {
                points: vec![
                    GaussPoint {
                        coords: vec![-1.0 / 3.0f64.sqrt()],
                        weight: 1.0,
                    },
                    GaussPoint {
                        coords: vec![1.0 / 3.0f64.sqrt()],
                        weight: 1.0,
                    },
                ],
            },
            _ => panic!("1D quadrature order {} not implemented", order),
        }
    }

    /// Returns quadrature rule on the 2D reference triangle with vertices (0,0), (1,0), (0,1).
    pub fn triangle(order: usize) -> Self {
        match order {
            1 => Self {
                // Centroid rule, exact for linear polynomials. Area of reference triangle is 0.5.
                points: vec![GaussPoint {
                    coords: vec![1.0 / 3.0, 1.0 / 3.0],
                    weight: 0.5,
                }],
            },
            2 => Self {
                // Three-point midpoint rule.
                points: vec![
                    GaussPoint {
                        coords: vec![1.0 / 6.0, 1.0 / 6.0],
                        weight: 1.0 / 6.0,
                    },
                    GaussPoint {
                        coords: vec![2.0 / 3.0, 1.0 / 6.0],
                        weight: 1.0 / 6.0,
                    },
                    GaussPoint {
                        coords: vec![1.0 / 6.0, 2.0 / 3.0],
                        weight: 1.0 / 6.0,
                    },
                ],
            },
            _ => panic!("Triangle quadrature order {} not implemented", order),
        }
    }

    /// Returns quadrature rule on the 3D reference tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    pub fn tetrahedron(order: usize) -> Self {
        match order {
            1 => Self {
                // Centroid rule, exact for linear polynomials. Volume of reference tetrahedron is 1/6.
                points: vec![GaussPoint {
                    coords: vec![1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
                    weight: 1.0 / 6.0,
                }],
            },
            2 => {
                // Four-point rule.
                let a = 0.5854101966249685;
                let b = 0.1381966011250105;
                let w = 1.0 / 24.0;
                Self {
                    points: vec![
                        GaussPoint {
                            coords: vec![a, b, b],
                            weight: w,
                        },
                        GaussPoint {
                            coords: vec![b, a, b],
                            weight: w,
                        },
                        GaussPoint {
                            coords: vec![b, b, a],
                            weight: w,
                        },
                        GaussPoint {
                            coords: vec![b, b, b],
                            weight: w,
                        },
                    ],
                }
            }
            _ => panic!("Tetrahedron quadrature order {} not implemented", order),
        }
    }
}

use crate::fem::element::ElementError;
use crate::mesh::Point3;

/// Integrates a distributed force/traction over a 1D Line2 boundary element in 2D or 3D.
/// `coords` contains the global coordinates of the 2 segment nodes.
/// `force_dim` is the number of components of the force vector (e.g. 2 for 2D elasticity).
/// `traction` is a callback returning the force vector at any Point3.
/// Returns a flat vector of size `2 * force_dim` representing the nodal forces.
pub fn integrate_line_boundary<F>(
    coords: &[Point3; 2],
    order: usize,
    force_dim: usize,
    traction: F,
) -> Result<Vec<f64>, ElementError>
where
    F: Fn(&Point3) -> Vec<f64>,
{
    let a = coords[0].coords;
    let b = coords[1].coords;
    let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
    if length <= f64::EPSILON {
        return Err(ElementError::DegenerateGeometry);
    }

    let rule = QuadratureRule::line(order);
    let mut nodal_forces = vec![0.0; 2 * force_dim];

    for gp in &rule.points {
        let xi = gp.coords[0];
        let shape = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];

        // Interpolate spatial coordinates
        let mut x = [0.0; 3];
        for i in 0..3 {
            x[i] = shape[0] * a[i] + shape[1] * b[i];
        }
        let p = Point3::new(x);

        let t_val = traction(&p);
        if t_val.len() != force_dim {
            return Err(ElementError::DimensionMismatch {
                expected: force_dim,
                actual: t_val.len(),
            });
        }

        let jacobian_det = length * 0.5;

        for node in 0..2 {
            for dof in 0..force_dim {
                nodal_forces[node * force_dim + dof] +=
                    gp.weight * shape[node] * t_val[dof] * jacobian_det;
            }
        }
    }

    Ok(nodal_forces)
}

/// Integrates a distributed force/traction over a 2D Tri3 boundary element in 3D.
/// `coords` contains the global 3D coordinates of the 3 face nodes.
/// `force_dim` is the number of components of the force vector (e.g. 3 for 3D elasticity).
/// `traction` is a callback returning the force vector at any Point3.
/// Returns a flat vector of size `3 * force_dim` representing the nodal forces.
pub fn integrate_triangle_boundary<F>(
    coords: &[Point3; 3],
    order: usize,
    force_dim: usize,
    traction: F,
) -> Result<Vec<f64>, ElementError>
where
    F: Fn(&Point3) -> Vec<f64>,
{
    let a = coords[0].coords;
    let b = coords[1].coords;
    let c = coords[2].coords;

    // Twice the area of the 3D triangle via cross product of (b - a) and (c - a)
    let v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    let twice_area = (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
    let area = 0.5 * twice_area;
    if area <= f64::EPSILON {
        return Err(ElementError::DegenerateGeometry);
    }

    let rule = QuadratureRule::triangle(order);
    let mut nodal_forces = vec![0.0; 3 * force_dim];

    for gp in &rule.points {
        let xi = gp.coords[0];
        let eta = gp.coords[1];
        let shape = [1.0 - xi - eta, xi, eta];

        // Interpolate spatial coordinates
        let mut x = [0.0; 3];
        for i in 0..3 {
            x[i] = shape[0] * a[i] + shape[1] * b[i] + shape[2] * c[i];
        }
        let p = Point3::new(x);

        let t_val = traction(&p);
        if t_val.len() != force_dim {
            return Err(ElementError::DimensionMismatch {
                expected: force_dim,
                actual: t_val.len(),
            });
        }

        let jacobian_det = 2.0 * area;

        for node in 0..3 {
            for dof in 0..force_dim {
                nodal_forces[node * force_dim + dof] +=
                    gp.weight * shape[node] * t_val[dof] * jacobian_det;
            }
        }
    }

    Ok(nodal_forces)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadrature_weights() {
        // Line total weight should be 2.0
        assert!(
            (QuadratureRule::line(1)
                .points
                .iter()
                .map(|p| p.weight)
                .sum::<f64>()
                - 2.0)
                .abs()
                < 1e-12
        );
        assert!(
            (QuadratureRule::line(2)
                .points
                .iter()
                .map(|p| p.weight)
                .sum::<f64>()
                - 2.0)
                .abs()
                < 1e-12
        );

        // Triangle total weight should be 0.5 (reference triangle area)
        assert!(
            (QuadratureRule::triangle(1)
                .points
                .iter()
                .map(|p| p.weight)
                .sum::<f64>()
                - 0.5)
                .abs()
                < 1e-12
        );
        assert!(
            (QuadratureRule::triangle(2)
                .points
                .iter()
                .map(|p| p.weight)
                .sum::<f64>()
                - 0.5)
                .abs()
                < 1e-12
        );

        // Tetrahedron total weight should be 1.0/6.0 (reference tetrahedron volume)
        assert!(
            (QuadratureRule::tetrahedron(1)
                .points
                .iter()
                .map(|p| p.weight)
                .sum::<f64>()
                - 1.0 / 6.0)
                .abs()
                < 1e-12
        );
        assert!(
            (QuadratureRule::tetrahedron(2)
                .points
                .iter()
                .map(|p| p.weight)
                .sum::<f64>()
                - 1.0 / 6.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn test_boundary_integration_line() {
        // Line segment from (0,0,0) to (2,0,0) - length = 2.0
        let coords = [Point3::new([0.0, 0.0, 0.0]), Point3::new([2.0, 0.0, 0.0])];

        // Constant traction [5.0, -3.0] in 2D
        let traction = |_p: &Point3| vec![5.0, -3.0];

        let forces = integrate_line_boundary(&coords, 2, 2, traction).unwrap();
        // Total force: traction * length = [10.0, -6.0].
        // Distributed equally between the two nodes: [5.0, -3.0, 5.0, -3.0].
        assert_eq!(forces.len(), 4);
        assert!((forces[0] - 5.0).abs() < 1e-12);
        assert!((forces[1] - -3.0).abs() < 1e-12);
        assert!((forces[2] - 5.0).abs() < 1e-12);
        assert!((forces[3] - -3.0).abs() < 1e-12);
    }

    #[test]
    fn test_boundary_integration_triangle() {
        // Reference triangle face in 3D: (0,0,0), (1,0,0), (0,1,0) - area = 0.5
        let coords = [
            Point3::new([0.0, 0.0, 0.0]),
            Point3::new([1.0, 0.0, 0.0]),
            Point3::new([0.0, 1.0, 0.0]),
        ];

        // Constant traction [0.0, 0.0, -10.0]
        let traction = |_p: &Point3| vec![0.0, 0.0, -10.0];

        let forces = integrate_triangle_boundary(&coords, 2, 3, traction).unwrap();
        // Total force: traction * area = [0, 0, -5].
        // Distributed equally between three nodes: [0, 0, -5/3] at each node.
        assert_eq!(forces.len(), 9);
        for node in 0..3 {
            assert_eq!(forces[node * 3], 0.0);
            assert_eq!(forces[node * 3 + 1], 0.0);
            assert!((forces[node * 3 + 2] - -5.0 / 3.0).abs() < 1e-12);
        }
    }
}
