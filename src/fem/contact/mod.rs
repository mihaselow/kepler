pub mod search;
pub mod penalty;

pub use search::{BoundarySegment, SpatialHashGrid2D, extract_boundary_segments};
pub use penalty::{ContactPair, evaluate_penalty_contact, assemble_penalty_contact};
