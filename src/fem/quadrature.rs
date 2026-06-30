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
}
