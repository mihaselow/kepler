use std::{collections::BTreeSet, fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    io::{
        FileIoError,
        params::{PoissonFileConfig, SourceConfig, validate_params_for_mesh},
    },
    linalg::{LinearSolverBackend, LinearSolverOptions, PreconditionerKind, SolverOptions},
    mesh::{Mesh, MeshError, Point2, Tri3},
};

pub const PROJECT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectFile {
    pub schema_version: u32,
    #[serde(default)]
    pub name: Option<String>,
    pub jobs: Vec<ProjectJob>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectJob {
    pub id: String,
    pub mesh: ProjectMesh,
    pub physics: ProjectPhysics,
    #[serde(default)]
    pub output: Option<ProjectOutput>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectMesh {
    pub points: Vec<ProjectPoint2>,
    pub triangles: Vec<ProjectTriangle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectPoint2 {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectTriangle {
    pub nodes: [usize; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProjectPhysics {
    Poisson(ProjectPoissonProblem),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectPoissonProblem {
    pub conductivity: f64,
    pub source: ProjectSource,
    #[serde(default)]
    pub dirichlet: Vec<ProjectDirichlet>,
    #[serde(default)]
    pub solver_options: ProjectLinearSolverOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProjectSource {
    Constant { value: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectDirichlet {
    pub node: usize,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectLinearSolverOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub backend: ProjectLinearSolverBackend,
    pub preconditioner: ProjectPreconditionerKind,
    pub record_residual_history: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectLinearSolverBackend {
    ConjugateGradient,
    Gmres,
    DenseDirect,
    SparseLdl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectPreconditionerKind {
    None,
    Jacobi,
    Amg,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectOutput {
    pub format: ProjectOutputFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectOutputFormat {
    Solution,
}

#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("failed to read project file {path}")]
    Read {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse project JSON")]
    Json(#[from] serde_json::Error),
    #[error("unsupported project schema version {version}; expected {expected}")]
    UnsupportedSchemaVersion { version: u32, expected: u32 },
    #[error("project must contain at least one job")]
    MissingJobs,
    #[error("job id must not be empty")]
    EmptyJobId,
    #[error("duplicate job id '{id}'")]
    DuplicateJobId { id: String },
    #[error("job '{job_id}' mesh validation failed")]
    Mesh {
        job_id: String,
        #[source]
        source: MeshError,
    },
    #[error("job '{job_id}' parameter validation failed")]
    Params {
        job_id: String,
        #[source]
        source: FileIoError,
    },
}

impl Default for ProjectFile {
    fn default() -> Self {
        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            name: None,
            jobs: Vec::new(),
        }
    }
}

impl Default for ProjectLinearSolverOptions {
    fn default() -> Self {
        Self::from(LinearSolverOptions::default())
    }
}

impl From<LinearSolverOptions> for ProjectLinearSolverOptions {
    fn from(value: LinearSolverOptions) -> Self {
        Self {
            max_iterations: value.max_iterations,
            tolerance: value.tolerance,
            backend: ProjectLinearSolverBackend::from(value.backend),
            preconditioner: ProjectPreconditionerKind::from(value.preconditioner),
            record_residual_history: value.record_residual_history,
        }
    }
}

impl From<ProjectLinearSolverOptions> for LinearSolverOptions {
    fn from(value: ProjectLinearSolverOptions) -> Self {
        Self {
            max_iterations: value.max_iterations,
            tolerance: value.tolerance,
            backend: LinearSolverBackend::from(value.backend),
            preconditioner: PreconditionerKind::from(value.preconditioner),
            record_residual_history: value.record_residual_history,
        }
    }
}

impl From<LinearSolverBackend> for ProjectLinearSolverBackend {
    fn from(value: LinearSolverBackend) -> Self {
        match value {
            LinearSolverBackend::ConjugateGradient => Self::ConjugateGradient,
            LinearSolverBackend::Gmres => Self::Gmres,
            LinearSolverBackend::DenseDirect => Self::DenseDirect,
            LinearSolverBackend::SparseLdl => Self::SparseLdl,
        }
    }
}

impl From<ProjectLinearSolverBackend> for LinearSolverBackend {
    fn from(value: ProjectLinearSolverBackend) -> Self {
        match value {
            ProjectLinearSolverBackend::ConjugateGradient => Self::ConjugateGradient,
            ProjectLinearSolverBackend::Gmres => Self::Gmres,
            ProjectLinearSolverBackend::DenseDirect => Self::DenseDirect,
            ProjectLinearSolverBackend::SparseLdl => Self::SparseLdl,
        }
    }
}

impl From<PreconditionerKind> for ProjectPreconditionerKind {
    fn from(value: PreconditionerKind) -> Self {
        match value {
            PreconditionerKind::None => Self::None,
            PreconditionerKind::Jacobi => Self::Jacobi,
            PreconditionerKind::Amg => Self::Amg,
        }
    }
}

impl From<ProjectPreconditionerKind> for PreconditionerKind {
    fn from(value: ProjectPreconditionerKind) -> Self {
        match value {
            ProjectPreconditionerKind::None => Self::None,
            ProjectPreconditionerKind::Jacobi => Self::Jacobi,
            ProjectPreconditionerKind::Amg => Self::Amg,
        }
    }
}

impl ProjectFile {
    pub fn from_legacy_poisson(
        job_id: impl Into<String>,
        mesh: &Mesh,
        config: PoissonFileConfig,
    ) -> Self {
        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            name: None,
            jobs: vec![ProjectJob {
                id: job_id.into(),
                mesh: ProjectMesh::from(mesh),
                physics: ProjectPhysics::Poisson(ProjectPoissonProblem::from(config)),
                output: Some(ProjectOutput {
                    format: ProjectOutputFormat::Solution,
                }),
            }],
        }
    }
}

impl From<&Mesh> for ProjectMesh {
    fn from(value: &Mesh) -> Self {
        Self {
            points: value
                .points()
                .iter()
                .map(|point| ProjectPoint2 {
                    x: point.x,
                    y: point.y,
                })
                .collect(),
            triangles: value
                .triangles()
                .iter()
                .map(|triangle| ProjectTriangle {
                    nodes: triangle.nodes,
                })
                .collect(),
        }
    }
}

impl TryFrom<ProjectMesh> for Mesh {
    type Error = MeshError;

    fn try_from(value: ProjectMesh) -> Result<Self, Self::Error> {
        Mesh::new(
            value
                .points
                .into_iter()
                .map(|point| Point2::new(point.x, point.y))
                .collect(),
            value
                .triangles
                .into_iter()
                .map(|triangle| Tri3::new(triangle.nodes))
                .collect(),
        )
    }
}

impl From<PoissonFileConfig> for ProjectPoissonProblem {
    fn from(value: PoissonFileConfig) -> Self {
        Self {
            conductivity: value.conductivity,
            source: ProjectSource::from(value.source),
            dirichlet: value
                .dirichlet
                .into_iter()
                .map(|(node, value)| ProjectDirichlet { node, value })
                .collect(),
            solver_options: ProjectLinearSolverOptions::from(value.solver_options),
        }
    }
}

impl From<ProjectPoissonProblem> for PoissonFileConfig {
    fn from(value: ProjectPoissonProblem) -> Self {
        Self {
            conductivity: value.conductivity,
            source: SourceConfig::from(value.source),
            dirichlet: value
                .dirichlet
                .into_iter()
                .map(|entry| (entry.node, entry.value))
                .collect(),
            solver_options: LinearSolverOptions::from(value.solver_options),
        }
    }
}

impl From<SourceConfig> for ProjectSource {
    fn from(value: SourceConfig) -> Self {
        match value {
            SourceConfig::Constant(value) => Self::Constant { value },
        }
    }
}

impl From<ProjectSource> for SourceConfig {
    fn from(value: ProjectSource) -> Self {
        match value {
            ProjectSource::Constant { value } => Self::Constant(value),
        }
    }
}

pub fn read_project_file(path: impl AsRef<Path>) -> Result<ProjectFile, ProjectError> {
    let path = path.as_ref();
    let input = fs::read_to_string(path).map_err(|source| ProjectError::Read {
        path: path.to_owned(),
        source,
    })?;
    parse_project_str(&input)
}

pub fn parse_project_str(input: &str) -> Result<ProjectFile, ProjectError> {
    let project = serde_json::from_str(input)?;
    validate_project(&project)?;
    Ok(project)
}

pub fn format_project(project: &ProjectFile) -> Result<String, ProjectError> {
    Ok(serde_json::to_string_pretty(project)?)
}

pub fn validate_project(project: &ProjectFile) -> Result<(), ProjectError> {
    if project.schema_version != PROJECT_SCHEMA_VERSION {
        return Err(ProjectError::UnsupportedSchemaVersion {
            version: project.schema_version,
            expected: PROJECT_SCHEMA_VERSION,
        });
    }
    if project.jobs.is_empty() {
        return Err(ProjectError::MissingJobs);
    }

    let mut job_ids = BTreeSet::new();
    for job in &project.jobs {
        if job.id.trim().is_empty() {
            return Err(ProjectError::EmptyJobId);
        }
        if !job_ids.insert(job.id.clone()) {
            return Err(ProjectError::DuplicateJobId { id: job.id.clone() });
        }
        validate_job(job)?;
    }
    Ok(())
}

pub fn validate_job(job: &ProjectJob) -> Result<(), ProjectError> {
    match &job.physics {
        ProjectPhysics::Poisson(problem) => {
            let mesh = Mesh::try_from(job.mesh.clone()).map_err(|source| ProjectError::Mesh {
                job_id: job.id.clone(),
                source,
            })?;
            let config = PoissonFileConfig::from(problem.clone());
            validate_params_for_mesh(&config, mesh.node_count()).map_err(|source| {
                ProjectError::Params {
                    job_id: job.id.clone(),
                    source,
                }
            })?;
        }
    }
    Ok(())
}

pub fn job_to_poisson(job: &ProjectJob) -> Result<(Mesh, PoissonFileConfig), ProjectError> {
    match &job.physics {
        ProjectPhysics::Poisson(problem) => {
            let mesh = Mesh::try_from(job.mesh.clone()).map_err(|source| ProjectError::Mesh {
                job_id: job.id.clone(),
                source,
            })?;
            let config = PoissonFileConfig::from(problem.clone());
            validate_params_for_mesh(&config, mesh.node_count()).map_err(|source| {
                ProjectError::Params {
                    job_id: job.id.clone(),
                    source,
                }
            })?;
            Ok((mesh, config))
        }
    }
}

pub fn default_project_solver_options() -> ProjectLinearSolverOptions {
    ProjectLinearSolverOptions::from(LinearSolverOptions::from(SolverOptions::default()))
}
