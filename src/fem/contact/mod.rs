pub mod augmented;
pub mod penalty;
pub mod search;
pub mod solve;

pub use augmented::{assemble_augmented_contact, evaluate_augmented_contact};
pub use penalty::{ContactPair, assemble_penalty_contact, evaluate_penalty_contact};
pub use search::{BoundarySegment, SpatialHashGrid2D, extract_boundary_segments};
pub use solve::{
    ContactProblem, ContactSolveError, ContactStaticAssembly, ContactStaticResult,
    ContactStaticSolverOptions, find_contact_pairs, solve_contact_static,
};
