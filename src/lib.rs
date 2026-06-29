pub mod fem;
pub mod linalg;
pub mod mesh;

pub use fem::poisson::{PoissonProblem, PoissonResult, solve_poisson};
pub use linalg::SolverOptions;
pub use mesh::{Mesh, MeshError, NodeId, Point2, Tri3};
