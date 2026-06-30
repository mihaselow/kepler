use crate::mesh::{NodeId, Point3};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ElementError {
    #[error("element geometry is degenerate (zero or negative measure)")]
    DegenerateGeometry,
    #[error("missing required material property: {0}")]
    MissingProperty(String),
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("invalid node count: expected {expected}, got {actual}")]
    InvalidNodeCount { expected: usize, actual: usize },
    #[error("other element error: {0}")]
    Other(String),
}

pub trait Element {
    /// Returns the spatial dimension of this element (e.g., 2 for Tri3, 3 for Tet4).
    fn spatial_dimension(&self) -> usize;

    /// Returns the number of nodes in this element.
    fn node_count(&self) -> usize;

    /// Returns the global node indices of this element.
    fn nodes(&self) -> &[NodeId];

    /// Returns the active fields/variables at each node (e.g. `vec!["u".to_string()]` for Poisson,
    /// `vec!["ux".to_string(), "uy".to_string()]` for 2D elasticity).
    fn active_fields(&self) -> Vec<String>;

    /// Computes the element stiffness matrix K_e.
    /// The returned matrix must have dimensions `(n_dofs, n_dofs)` where `n_dofs = node_count() * active_fields().len()`.
    fn local_stiffness(
        &self,
        node_coords: &[Point3],
        properties: &BTreeMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, ElementError>;

    /// Computes the element mass matrix M_e.
    fn local_mass(
        &self,
        node_coords: &[Point3],
        density: f64,
        lumped: bool,
    ) -> Result<Vec<Vec<f64>>, ElementError>;
}
