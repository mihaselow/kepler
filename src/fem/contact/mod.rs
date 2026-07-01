pub mod search;
pub mod penalty;
pub mod augmented;

pub use search::{BoundarySegment, SpatialHashGrid2D, extract_boundary_segments};
pub use penalty::{ContactPair, evaluate_penalty_contact, assemble_penalty_contact};
pub use augmented::{evaluate_augmented_contact, assemble_augmented_contact};
