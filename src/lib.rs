pub mod annotation;
pub mod condition;
pub mod fem;
pub mod io;
pub mod linalg;
pub mod mesh;

pub use annotation::{
    AnnotationError, AnnotationId, EntitySelector, GeometryAnnotations, MaterialAssignment,
    Parameter, ParameterAssignment, ParameterId, ParameterValue, ResolvedAnnotations,
    ResolvedMaterialAssignment, ResolvedParameterAssignment,
};
pub use condition::{
    Condition, ConditionError, ConditionId, ConditionKind, ConditionSet, ConditionSignature,
    ResolvedCondition, ResolvedConditionSet,
};
pub use fem::elasticity::{
    DisplacementComponent, DisplacementConstraint, ElasticityError, ElasticityMaterial,
    ElasticityModel, ElasticityProblem, ElasticityResult, NodalForce, solve_elasticity,
};
pub use fem::heat::{
    SteadyHeatProblem, SteadyHeatProblem3D, TemperatureResult, solve_steady_heat,
    solve_steady_heat_3d,
};
pub use fem::poisson::{
    PoissonProblem, PoissonProblem3D, PoissonResult, solve_poisson, solve_poisson_3d,
};
pub use io::{
    FileIoError,
    gmsh::{ImportedMesh, parse_gmsh_str, read_gmsh_file},
    mesh::{parse_mesh_str, read_mesh_file},
    params::{PoissonFileConfig, SourceConfig, parse_params_str, read_params_file},
    solution::{format_solution, write_solution_file},
    vtk::{VtkScalarField, format_vtk_legacy, write_vtk_legacy_file},
};
pub use linalg::SolverOptions;
pub use mesh::{
    Cell, CellId, ElementKind, EntityDimension, FacetId, FieldId, MaterialId, Mesh, MeshError,
    MeshTopology, NodeId, Point2, Point3, PointD, Region, RegionId, Tri3,
};
