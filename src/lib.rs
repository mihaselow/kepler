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
pub use fem::diffusion_reaction::{
    DiffusionReactionError, DiffusionReactionProblem, DiffusionReactionProblem3D,
    DiffusionReactionResult, DiffusionReactionSolverResult, TransientDiffusionReactionProblem,
    TransientDiffusionReactionProblem3D, TransientDiffusionReactionResult,
    TransientDiffusionReactionStep, solve_diffusion_reaction, solve_diffusion_reaction_3d,
    solve_diffusion_reaction_3d_with_solver, solve_diffusion_reaction_with_solver,
    solve_transient_diffusion_reaction, solve_transient_diffusion_reaction_3d,
};
pub use fem::elasticity::{
    DisplacementComponent, DisplacementComponent3D, DisplacementConstraint,
    DisplacementConstraint3D, ElasticityError, ElasticityMaterial, ElasticityMaterial3D,
    ElasticityModel, ElasticityProblem, ElasticityProblem3D, ElasticityResult, ElasticityResult3D,
    ElasticitySolverResult, ElasticitySolverResult3D, NodalForce, NodalForce3D,
    TransientElasticityProblem, TransientElasticityProblem3D, TransientElasticityResult,
    TransientElasticityResult3D, TransientElasticityStep, TransientElasticityStep3D,
    solve_elasticity, solve_elasticity_3d, solve_elasticity_3d_with_solver,
    solve_elasticity_with_solver, solve_transient_elasticity, solve_transient_elasticity_3d,
};
pub use fem::electrostatics::{
    ELECTROSTATIC_FORMULATION, ElectricPotentialResult, ElectricPotentialSolverResult,
    ElectrostaticFormulation, ElectrostaticProblem, ElectrostaticProblem3D, solve_electrostatics,
    solve_electrostatics_3d, solve_electrostatics_3d_with_solver, solve_electrostatics_with_solver,
};
pub use fem::heat::{
    SteadyHeatProblem, SteadyHeatProblem3D, TemperatureResult, TemperatureSolverResult,
    TransientHeatError, TransientHeatProblem, TransientTemperatureResult, TransientTemperatureStep,
    solve_steady_heat, solve_steady_heat_3d, solve_steady_heat_3d_with_solver,
    solve_steady_heat_with_solver, solve_transient_heat,
};
pub use fem::modal::{
    ModalError, ModalProblem, ModalProblem3D, ModalResult, ModalResult3D, ModeShape, ModeShape3D,
    solve_modal, solve_modal_3d,
};
pub use fem::poisson::{
    PoissonProblem, PoissonProblem3D, PoissonResult, PoissonSolverResult, solve_poisson,
    solve_poisson_3d, solve_poisson_3d_with_solver, solve_poisson_with_solver,
};
pub use io::{
    FileIoError,
    cad::{
        CadFileFormat, CadLengthUnit, CadMeshOutputFormat, CadMeshingDimension, CadMeshingOptions,
        CadMeshingWorkflow, CadSource, CadWorkflowError, ExternalCommand, ExternalMesher,
        infer_cad_format, plan_cad_meshing_command, validate_cad_meshing_workflow,
    },
    gmsh::{ImportedMesh, parse_gmsh_str, read_gmsh_file},
    mesh::{parse_mesh_str, read_mesh_file},
    params::{PoissonFileConfig, SourceConfig, parse_params_str, read_params_file},
    project::{
        PROJECT_SCHEMA_VERSION, ProjectDirichlet, ProjectError, ProjectFile, ProjectJob,
        ProjectLinearSolverBackend, ProjectLinearSolverOptions, ProjectMesh, ProjectOutput,
        ProjectOutputFormat, ProjectPhysics, ProjectPoint2, ProjectPoissonProblem,
        ProjectPreconditionerKind, ProjectSource, ProjectTriangle, default_project_solver_options,
        format_project, job_to_poisson, parse_project_str, read_project_file, validate_job,
        validate_project,
    },
    solution::{format_solution, write_solution_file},
    vtk::{VtkScalarField, format_vtk_legacy, write_vtk_legacy_file},
};
pub use linalg::{
    ConfiguredLinearSolver, DiagonalDiagnostics, LinalgError, LinearSolver, LinearSolverBackend,
    LinearSolverOptions, MatrixDiagnostics, NewmarkSolverOptions, NewmarkStepResult,
    NonlinearSolverDiagnostics, NonlinearSolverOptions, NonlinearSolverResult, NonlinearSystem,
    PreconditionerKind, SolverDiagnostics, SolverOptions, SparsityStats, SpdHeuristics,
    SymmetryDiagnostics, TransientSolverOptions, TransientStepResult, analyze_matrix, newton_solve,
    solve_linear_system, solve_linear_transient, solve_newmark_transient,
};
pub use mesh::{
    Cell, CellId, ElementKind, EntityDimension, FacetId, FieldId, MaterialId, Mesh, MeshError,
    MeshTopology, NodeId, Point2, Point3, PointD, Region, RegionId, Tri3,
};
