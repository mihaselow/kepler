//! Constitutive material models and integration history tracking.

pub mod plasticity;

/// Local state/history variables tracked at each integration (Gauss) point.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialState {
    /// Plastic strain tensor components in Voigt notation:
    /// `[eps_xx^p, eps_yy^p, eps_zz^p, gamma_xy^p, gamma_yz^p, gamma_xz^p]`
    pub plastic_strain: [f64; 6],
    /// Accumulated equivalent plastic strain (hardening parameter `alpha` / `kappa`).
    pub equivalent_plastic_strain: f64,
}

impl Default for MaterialState {
    fn default() -> Self {
        Self {
            plastic_strain: [0.0; 6],
            equivalent_plastic_strain: 0.0,
        }
    }
}

/// A general constitutive model interface for 3D state integration.
pub trait MaterialModel: Send + Sync {
    /// Integrates the constitutive equations at a single Gauss point.
    ///
    /// Given the total strain tensor `strain` (in Voigt notation: `[eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz]`)
    /// and the history variables at the beginning of the load step `state_old`,
    /// computes the new stress tensor, updated history variables, and the 6x6 algorithmic consistent tangent stiffness matrix.
    fn integrate(
        &self,
        strain: &[f64; 6],
        state_old: &MaterialState,
    ) -> ([f64; 6], MaterialState, [[f64; 6]; 6]);

    /// Returns the default/initial state variables for this material model.
    fn initial_state(&self) -> MaterialState {
        MaterialState::default()
    }
}
