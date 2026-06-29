pub mod fem;
pub mod io;
pub mod linalg;
pub mod mesh;

pub use fem::poisson::{PoissonProblem, PoissonResult, solve_poisson};
pub use io::{
    FileIoError,
    mesh::{parse_mesh_str, read_mesh_file},
    params::{PoissonFileConfig, SourceConfig, parse_params_str, read_params_file},
    solution::{format_solution, write_solution_file},
};
pub use linalg::SolverOptions;
pub use mesh::{
    Cell, CellId, ElementKind, EntityDimension, FacetId, FieldId, MaterialId, Mesh, MeshError,
    MeshTopology, NodeId, Point2, Point3, PointD, Region, RegionId, Tri3,
};
