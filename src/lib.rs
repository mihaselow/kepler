pub mod annotation;
pub mod condition;
pub mod fem;
pub mod io;
pub mod linalg;
pub mod mesh;
pub mod parallel;

pub use annotation::{
    AnnotationError, AnnotationId, EntitySelector, GeometryAnnotations, MaterialAssignment,
    Parameter, ParameterAssignment, ParameterId, ParameterValue, ResolvedAnnotations,
    ResolvedMaterialAssignment, ResolvedParameterAssignment,
};
pub use condition::{
    Condition, ConditionError, ConditionId, ConditionKind, ConditionSet, ConditionSignature,
    ResolvedCondition, ResolvedConditionSet,
};
pub use fem::cms::{CraigBamptonReduction, reduce_craig_bampton};
pub use fem::constraint::{
    MPCConstraint, MPCTerm, apply_mpc_lagrange, apply_mpc_penalty, split_lagrange_solution,
};
pub use fem::contact::{
    BoundarySegment, ContactPair, ContactProblem, ContactSolveError, ContactStaticAssembly,
    ContactStaticResult, ContactStaticSolverOptions, SpatialHashGrid2D, assemble_augmented_contact,
    assemble_penalty_contact, evaluate_augmented_contact, evaluate_penalty_contact,
    extract_boundary_segments, find_contact_pairs, solve_contact_static,
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
    solve_transient_elasticity_hht,
};
pub use fem::electrostatics::{
    ELECTROSTATIC_FORMULATION, ElectricPotentialResult, ElectricPotentialSolverResult,
    ElectrostaticFormulation, ElectrostaticProblem, ElectrostaticProblem3D, solve_electrostatics,
    solve_electrostatics_3d, solve_electrostatics_3d_with_solver, solve_electrostatics_with_solver,
};
pub use fem::explicit::{
    ExplicitDynamicsError, ExplicitDynamicsOptions, ExplicitDynamicsProblem,
    ExplicitDynamicsResult, ExplicitDynamicsStep, estimate_critical_time_step,
    solve_explicit_dynamics,
};
pub use fem::heat::{
    SteadyHeatProblem, SteadyHeatProblem3D, TemperatureResult, TemperatureSolverResult,
    TransientHeatError, TransientHeatProblem, TransientTemperatureResult, TransientTemperatureStep,
    solve_steady_heat, solve_steady_heat_3d, solve_steady_heat_3d_with_solver,
    solve_steady_heat_with_solver, solve_transient_heat,
};
pub use fem::material::plasticity::J2PlasticMaterial;
pub use fem::material::{MaterialModel, MaterialState};
pub use fem::modal::{
    ModalError, ModalProblem, ModalProblem3D, ModalResult, ModalResult3D, ModeShape, ModeShape3D,
    solve_modal, solve_modal_3d,
};
pub use fem::nonlinear::{NonlinearTrussAssembly, NonlinearTrussElement};
pub use fem::nonlinear_continuum::{
    NonlinearContinuumAssembly, NonlinearContinuumResult, NonlinearContinuumSolverOptions,
    solve_nonlinear_continuum,
};
pub use fem::poisson::{
    PoissonProblem, PoissonProblem3D, PoissonResult, PoissonSolverResult, solve_poisson,
    solve_poisson_3d, solve_poisson_3d_with_solver, solve_poisson_with_solver,
};
pub use fem::quadrature::{integrate_line_boundary, integrate_triangle_boundary};
pub use fem::structural::{Beam2D, Beam3D, ShellQuad4, ShellTri3, Truss};
pub use fem::structural_solve::{
    BeamSection, StructuralComponent, StructuralConstraint, StructuralError, StructuralForce,
    StructuralMaterial, StructuralProblem, StructuralResult, dof_index_6, solve_structural,
    solve_structural_with_solver,
};
pub use fem::thermal_struct::{
    ThermoElasticError, ThermoElasticProblem, ThermoElasticResult, ThermoElasticStaggerOptions,
    solve_thermoelastic,
};
pub use io::{
    FileIoError,
    abaqus::{
        AbaqusBoundary, AbaqusCload, AbaqusElement, AbaqusMaterial, AbaqusModel, AbaqusStep,
        AbaqusVerifyCase, AbaqusVerifyCheck, abaqus_to_annotations, abaqus_to_elasticity_problem,
        abaqus_to_mesh_2d, parse_abaqus_str, parse_abaqus_verify_str, read_abaqus_file,
        read_abaqus_verify_case, verify_elasticity_against_case,
    },
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
        format_project, job_to_contact, job_to_elasticity, job_to_elasticity_3d, job_to_modal,
        job_to_modal_3d, job_to_nonlinear_continuum, job_to_poisson, job_to_structural,
        parse_project_str, read_project_file, validate_job, validate_project,
    },
    result_format::{
        KeplerResultCell, KeplerResultFile, KeplerResultMesh, KeplerResultStep,
        RESULT_SCHEMA_VERSION, ResultIoError, read_json_result, write_hdf5_result,
        write_json_result, write_result_file,
    },
    solution::{format_solution, write_solution_file},
    vtk::{VtkScalarField, format_vtk_legacy, write_vtk_legacy_file},
};
pub use linalg::{
    ArcLengthSystem, ConfiguredLinearSolver, DiagonalDiagnostics, HhtSolverOptions, HhtStepResult,
    LanczosEigenResult, LinalgError, LinearSolver, LinearSolverBackend, LinearSolverOptions,
    MatrixDiagnostics, NewmarkSolverOptions, NewmarkStepResult, NonlinearSolverDiagnostics,
    NonlinearSolverOptions, NonlinearSolverResult, NonlinearSystem, PreconditionerKind, RiksResult,
    RiksSolverOptions, RiksStepResult, SolverDiagnostics, SolverOptions, SparsityStats,
    SpdHeuristics, SymmetryDiagnostics, TransientSolverOptions, TransientStepResult,
    analyze_matrix, axpy, newton_solve, norm, riks_solve, solve_harmonic_response,
    solve_hht_transient, solve_lanczos_modes, solve_linear_system, solve_linear_transient,
    solve_newmark_transient,
};
pub use mesh::{
    Cell, CellId, ElementKind, EntityDimension, FacetId, FieldId, MaterialId, Mesh, MeshError,
    MeshTopology, NodeId, Point2, Point3, PointD, Region, RegionId, Tri3,
};
