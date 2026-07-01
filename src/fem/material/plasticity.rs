//! von Mises (J2) plasticity return-mapping.

use super::{MaterialModel, MaterialState};

/// J2 plasticity with bilinear isotropic hardening.
#[derive(Debug, Clone)]
pub struct J2PlasticMaterial {
    pub young_modulus: f64,
    pub poisson_ratio: f64,
    pub yield_stress: f64,
    pub hardening_modulus: f64,
}

impl J2PlasticMaterial {
    pub fn new(
        young_modulus: f64,
        poisson_ratio: f64,
        yield_stress: f64,
        hardening_modulus: f64,
    ) -> Self {
        Self {
            young_modulus,
            poisson_ratio,
            yield_stress,
            hardening_modulus,
        }
    }
}

impl MaterialModel for J2PlasticMaterial {
    #[allow(clippy::needless_range_loop)]
    fn integrate(
        &self,
        strain: &[f64; 6],
        state_old: &MaterialState,
    ) -> ([f64; 6], MaterialState, [[f64; 6]; 6]) {
        let e = self.young_modulus;
        let nu = self.poisson_ratio;
        let mu = e / (2.0 * (1.0 + nu));
        let k_bulk = e / (3.0 * (1.0 - 2.0 * nu));

        // 1. Elastic Predictor
        let mut strain_trial = [0.0; 6];
        for i in 0..6 {
            strain_trial[i] = strain[i] - state_old.plastic_strain[i];
        }

        let tr_strain_trial = strain_trial[0] + strain_trial[1] + strain_trial[2];
        let mut s_trial = [0.0; 6];
        s_trial[0] = 2.0 * mu * (strain_trial[0] - tr_strain_trial / 3.0);
        s_trial[1] = 2.0 * mu * (strain_trial[1] - tr_strain_trial / 3.0);
        s_trial[2] = 2.0 * mu * (strain_trial[2] - tr_strain_trial / 3.0);
        s_trial[3] = mu * strain_trial[3];
        s_trial[4] = mu * strain_trial[4];
        s_trial[5] = mu * strain_trial[5];

        let s_norm = (s_trial[0].powi(2)
            + s_trial[1].powi(2)
            + s_trial[2].powi(2)
            + 2.0 * (s_trial[3].powi(2) + s_trial[4].powi(2) + s_trial[5].powi(2)))
        .sqrt();

        let sigma_eff_trial = (1.5f64).sqrt() * s_norm;
        let alpha = state_old.equivalent_plastic_strain;
        let yield_radius = self.yield_stress + self.hardening_modulus * alpha;
        let f_trial = sigma_eff_trial - yield_radius;

        let mut c_matrix = [[0.0; 6]; 6];
        // Volumetric part in c_matrix
        for i in 0..3 {
            for j in 0..3 {
                c_matrix[i][j] += k_bulk;
            }
        }

        if f_trial <= 0.0 {
            // Elastic step
            let mut stress = [0.0; 6];
            for i in 0..3 {
                stress[i] = k_bulk * tr_strain_trial + s_trial[i];
            }
            stress[3] = s_trial[3];
            stress[4] = s_trial[4];
            stress[5] = s_trial[5];

            // Elastic tangent
            for i in 0..3 {
                c_matrix[i][i] += 2.0 * mu * (2.0 / 3.0);
                for j in 0..3 {
                    if i != j {
                        c_matrix[i][j] += 2.0 * mu * (-1.0 / 3.0);
                    }
                }
            }
            for i in 3..6 {
                c_matrix[i][i] += mu;
            }

            (stress, state_old.clone(), c_matrix)
        } else {
            // Plastic step
            let h = self.hardening_modulus;
            let d_gamma = f_trial / (3.0 * mu + h);

            let s_factor = 1.0 - (3.0 * mu * d_gamma) / sigma_eff_trial;
            let mut s_new = [0.0; 6];
            for i in 0..6 {
                s_new[i] = s_factor * s_trial[i];
            }

            let mut stress = [0.0; 6];
            for i in 0..3 {
                stress[i] = k_bulk * tr_strain_trial + s_new[i];
            }
            stress[3] = s_new[3];
            stress[4] = s_new[4];
            stress[5] = s_new[5];

            let alpha_new = alpha + d_gamma;

            let mut n = [0.0; 6];
            if s_norm > 0.0 {
                for i in 0..6 {
                    n[i] = s_trial[i] / s_norm;
                }
            }

            let mut plastic_strain_new = state_old.plastic_strain;
            let factor_p = (1.5f64).sqrt() * d_gamma;
            plastic_strain_new[0] += factor_p * n[0];
            plastic_strain_new[1] += factor_p * n[1];
            plastic_strain_new[2] += factor_p * n[2];
            plastic_strain_new[3] += 2.0 * factor_p * n[3];
            plastic_strain_new[4] += 2.0 * factor_p * n[4];
            plastic_strain_new[5] += 2.0 * factor_p * n[5];

            let state_new = MaterialState {
                plastic_strain: plastic_strain_new,
                equivalent_plastic_strain: alpha_new,
            };

            // Algorithmic consistent tangent stiffness
            let beta = s_factor;
            let gamma_factor = beta - 1.0 + 1.0 / (1.0 + h / (3.0 * mu));

            // Volumetric + Deviatoric parts
            let beta_factor = 2.0 * mu * beta;
            for i in 0..3 {
                c_matrix[i][i] += beta_factor * (2.0 / 3.0);
                for j in 0..3 {
                    if i != j {
                        c_matrix[i][j] += beta_factor * (-1.0 / 3.0);
                    }
                }
            }
            for i in 3..6 {
                c_matrix[i][i] += mu * beta;
            }

            let normal_factor = 2.0 * mu * gamma_factor;
            for i in 0..6 {
                for j in 0..6 {
                    c_matrix[i][j] -= normal_factor * n[i] * n[j];
                }
            }

            (stress, state_new, c_matrix)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j2_elastic_response() {
        let material = J2PlasticMaterial::new(200e9, 0.3, 250e6, 10e9);
        let state = MaterialState::default();

        // Small strain -> purely elastic
        let strain = [1e-4, -0.3e-4, -0.3e-4, 0.0, 0.0, 0.0];
        let (stress, state_new, _c) = material.integrate(&strain, &state);

        assert_eq!(state_new.equivalent_plastic_strain, 0.0);
        // sigma_xx should be E * strain_xx
        assert!((stress[0] - 20e6).abs() < 1e-3);
    }

    #[test]
    fn test_j2_plastic_yielding_and_return() {
        let material = J2PlasticMaterial::new(200e9, 0.3, 250e6, 10e9);
        let state = MaterialState::default();

        // Large uniaxial strain to trigger yielding
        let strain = [3e-3, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (stress, state_new, _c) = material.integrate(&strain, &state);

        assert!(state_new.equivalent_plastic_strain > 0.0);

        // Compute von Mises of returned stress: should lie on the yield surface
        let tr = stress[0] + stress[1] + stress[2];
        let s = [
            stress[0] - tr / 3.0,
            stress[1] - tr / 3.0,
            stress[2] - tr / 3.0,
            stress[3],
            stress[4],
            stress[5],
        ];
        let s_norm = (s[0].powi(2)
            + s[1].powi(2)
            + s[2].powi(2)
            + 2.0 * (s[3].powi(2) + s[4].powi(2) + s[5].powi(2)))
        .sqrt();
        let vm = (1.5f64).sqrt() * s_norm;

        let expected_yield = material.yield_stress
            + material.hardening_modulus * state_new.equivalent_plastic_strain;
        assert!((vm - expected_yield).abs() < 1.0); // within 1 Pa accuracy
    }

    #[test]
    fn test_j2_consistent_tangent() {
        let material = J2PlasticMaterial::new(200e9, 0.3, 250e6, 10e9);
        let state = MaterialState::default();

        // Load to a plastic state first
        let strain_pre = [2e-3, -0.6e-3, -0.6e-3, 1e-3, 0.0, 0.0];
        let (_, state_p, _) = material.integrate(&strain_pre, &state);
        assert!(state_p.equivalent_plastic_strain > 0.0);

        // Perturb strain to check consistent tangent operator numerically
        let base_strain = [2.1e-3, -0.62e-3, -0.61e-3, 1.05e-3, 0.01e-3, -0.02e-3];
        let (_stress_base, _, c_analytical) = material.integrate(&base_strain, &state_p);

        let eps = 1.0e-6;
        let mut c_numerical = [[0.0; 6]; 6];

        for j in 0..6 {
            let mut perturbed_strain_plus = base_strain;
            perturbed_strain_plus[j] += eps;

            let mut perturbed_strain_minus = base_strain;
            perturbed_strain_minus[j] -= eps;

            let (stress_perturbed_plus, _, _) =
                material.integrate(&perturbed_strain_plus, &state_p);
            let (stress_perturbed_minus, _, _) =
                material.integrate(&perturbed_strain_minus, &state_p);

            for i in 0..6 {
                c_numerical[i][j] =
                    (stress_perturbed_plus[i] - stress_perturbed_minus[i]) / (2.0 * eps);
            }
        }

        // Compare analytical tangent matrix with numerical tangent matrix
        for i in 0..6 {
            for j in 0..6 {
                let diff = (c_analytical[i][j] - c_numerical[i][j]).abs();
                let ref_val = c_analytical[i][j].abs().max(1.0);
                assert!(
                    diff / ref_val < 1.0e-5,
                    "Consistent tangent mismatch at ({i}, {j}): analytical={:.6e}, numerical={:.6e}",
                    c_analytical[i][j],
                    c_numerical[i][j]
                );
            }
        }
    }
}
