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
    #[serde(default)]
    pub points: Option<Vec<ProjectPoint2>>,
    #[serde(default)]
    pub points_3d: Option<Vec<ProjectPoint3>>,
    #[serde(default)]
    pub triangles: Vec<ProjectTriangle>,
    #[serde(default)]
    pub cells: Option<Vec<ProjectCell>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectPoint2 {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectPoint3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectTriangle {
    pub nodes: [usize; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectCell {
    pub kind: String,
    pub nodes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProjectPhysics {
    Poisson(ProjectPoissonProblem),
    Elasticity(ProjectElasticityProblem),
    #[serde(rename = "elasticity_3d")]
    Elasticity3d(ProjectElasticityProblem3D),
    Modal(ProjectModalProblem),
    #[serde(rename = "modal_3d")]
    Modal3d(ProjectModalProblem3D),
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
pub struct ProjectElasticityProblem {
    pub material: ProjectElasticityMaterial,
    pub thickness: f64,
    #[serde(default)]
    pub constraints: Vec<ProjectElasticityConstraint>,
    #[serde(default)]
    pub forces: Vec<ProjectElasticityForce>,
    #[serde(default)]
    pub solver_options: ProjectLinearSolverOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectElasticityMaterial {
    pub young_modulus: f64,
    pub poisson_ratio: f64,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectElasticityConstraint {
    pub node: usize,
    pub component: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectElasticityForce {
    pub node: usize,
    pub component: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectElasticityProblem3D {
    pub material: ProjectElasticityMaterial3D,
    #[serde(default)]
    pub constraints: Vec<ProjectElasticityConstraint3D>,
    #[serde(default)]
    pub forces: Vec<ProjectElasticityForce3D>,
    #[serde(default)]
    pub solver_options: ProjectLinearSolverOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectElasticityMaterial3D {
    pub young_modulus: f64,
    pub poisson_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectElasticityConstraint3D {
    pub node: usize,
    pub component: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectElasticityForce3D {
    pub node: usize,
    pub component: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectModalProblem {
    pub material: ProjectElasticityMaterial,
    pub thickness: f64,
    pub density: f64,
    pub requested_modes: usize,
    #[serde(default)]
    pub constraints: Vec<ProjectElasticityConstraint>,
    pub lumped: bool,
    #[serde(default)]
    pub solver_options: ProjectLinearSolverOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectModalProblem3D {
    pub material: ProjectElasticityMaterial3D,
    pub density: f64,
    pub requested_modes: usize,
    #[serde(default)]
    pub constraints: Vec<ProjectElasticityConstraint3D>,
    pub lumped: bool,
    #[serde(default)]
    pub solver_options: ProjectLinearSolverOptions,
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
    #[error("job '{job_id}' has invalid configuration: {message}")]
    InvalidConfiguration { job_id: String, message: String },
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
            points: Some(value
                .points()
                .iter()
                .map(|point| ProjectPoint2 {
                    x: point.x,
                    y: point.y,
                })
                .collect()),
            points_3d: None,
            triangles: value
                .triangles()
                .iter()
                .map(|triangle| ProjectTriangle {
                    nodes: triangle.nodes,
                })
                .collect(),
            cells: None,
        }
    }
}

impl TryFrom<ProjectMesh> for Mesh {
    type Error = MeshError;

    fn try_from(value: ProjectMesh) -> Result<Self, Self::Error> {
        let pts: Vec<Point2> = value
            .points
            .as_ref()
            .map(|pts| {
                pts.iter()
                    .map(|point| Point2::new(point.x, point.y))
                    .collect()
            })
            .unwrap_or_default();

        if let Some(cells_input) = value.cells {
            let cells = cells_input
                .into_iter()
                .map(|cell| {
                    let kind = match cell.kind.to_lowercase().as_str() {
                        "line2" | "beam" | "beam3d" | "truss" => crate::mesh::ElementKind::Line2,
                        "line3" => crate::mesh::ElementKind::Line3,
                        "tri3" | "shell" | "shell_tri3" => crate::mesh::ElementKind::Tri3,
                        "tri6" => crate::mesh::ElementKind::Tri6,
                        "quad4" | "shell_quad4" => crate::mesh::ElementKind::Quad4,
                        "quad8" => crate::mesh::ElementKind::Quad8,
                        "tet4" => crate::mesh::ElementKind::Tet4,
                        "tet10" => crate::mesh::ElementKind::Tet10,
                        "hex8" => crate::mesh::ElementKind::Hex8,
                        "hex20" => crate::mesh::ElementKind::Hex20,
                        _ => crate::mesh::ElementKind::Tri3,
                    };
                    crate::mesh::Cell::new(kind, cell.nodes)
                })
                .collect();
            Mesh::new_with_cells(pts, cells)
        } else {
            Mesh::new(
                pts,
                value
                    .triangles
                    .into_iter()
                    .map(|triangle| Tri3::new(triangle.nodes))
                    .collect(),
            )
        }
    }
}

impl TryFrom<ProjectMesh> for crate::MeshTopology<3> {
    type Error = MeshError;

    fn try_from(value: ProjectMesh) -> Result<Self, Self::Error> {
        let pts: Vec<crate::PointD<3>> = if let Some(pts_3d) = value.points_3d {
            pts_3d
                .into_iter()
                .map(|p| crate::PointD::new([p.x, p.y, p.z]))
                .collect()
        } else if let Some(pts_2d) = value.points {
            pts_2d
                .into_iter()
                .map(|p| crate::PointD::new([p.x, p.y, 0.0]))
                .collect()
        } else {
            vec![]
        };

        let cells: Vec<crate::mesh::Cell> = if let Some(cells_input) = value.cells {
            cells_input
                .into_iter()
                .map(|cell| {
                    let kind = match cell.kind.to_lowercase().as_str() {
                        "line2" | "beam" | "beam3d" | "truss" => crate::mesh::ElementKind::Line2,
                        "line3" => crate::mesh::ElementKind::Line3,
                        "tri3" | "shell" | "shell_tri3" => crate::mesh::ElementKind::Tri3,
                        "tri6" => crate::mesh::ElementKind::Tri6,
                        "quad4" | "shell_quad4" => crate::mesh::ElementKind::Quad4,
                        "quad8" => crate::mesh::ElementKind::Quad8,
                        "tet4" => crate::mesh::ElementKind::Tet4,
                        "tet10" => crate::mesh::ElementKind::Tet10,
                        "hex8" => crate::mesh::ElementKind::Hex8,
                        "hex20" => crate::mesh::ElementKind::Hex20,
                        _ => crate::mesh::ElementKind::Tet4,
                    };
                    crate::mesh::Cell::new(kind, cell.nodes)
                })
                .collect()
        } else {
            value
                .triangles
                .into_iter()
                .map(|t| crate::mesh::Cell::new(crate::mesh::ElementKind::Tri3, t.nodes.to_vec()))
                .collect()
        };

        crate::MeshTopology::<3>::new(pts, cells)
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
        ProjectPhysics::Elasticity(_) | ProjectPhysics::Modal(_) => {
            let _mesh = Mesh::try_from(job.mesh.clone()).map_err(|source| ProjectError::Mesh {
                job_id: job.id.clone(),
                source,
            })?;
        }
        ProjectPhysics::Elasticity3d(_) | ProjectPhysics::Modal3d(_) => {
            let _mesh = crate::MeshTopology::<3>::try_from(job.mesh.clone()).map_err(|source| ProjectError::Mesh {
                job_id: job.id.clone(),
                source,
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
        _ => Err(ProjectError::InvalidConfiguration {
            job_id: job.id.clone(),
            message: "expected poisson physics".to_string(),
        }),
    }
}

pub fn job_to_elasticity(job: &ProjectJob) -> Result<(Mesh, crate::fem::elasticity::ElasticityProblem, LinearSolverOptions), ProjectError> {
    match &job.physics {
        ProjectPhysics::Elasticity(problem) => {
            let mesh = Mesh::try_from(job.mesh.clone()).map_err(|source| ProjectError::Mesh {
                job_id: job.id.clone(),
                source,
            })?;
            let material = crate::ElasticityMaterial::from(problem.material.clone());
            let constraints: Vec<crate::DisplacementConstraint> = problem
                .constraints
                .iter()
                .map(|c| crate::DisplacementConstraint::from(c.clone()))
                .collect();
            let forces = project_forces_to_nodal_forces(problem.forces.clone());
            let problem_internal = crate::fem::elasticity::ElasticityProblem {
                material,
                thickness: problem.thickness,
                constraints,
                forces,
            };
            let solver_options = LinearSolverOptions::from(problem.solver_options.clone());
            Ok((mesh, problem_internal, solver_options))
        }
        _ => Err(ProjectError::InvalidConfiguration {
            job_id: job.id.clone(),
            message: "expected elasticity physics".to_string(),
        }),
    }
}

pub fn job_to_elasticity_3d(job: &ProjectJob) -> Result<(crate::MeshTopology<3>, crate::fem::elasticity::ElasticityProblem3D, LinearSolverOptions), ProjectError> {
    match &job.physics {
        ProjectPhysics::Elasticity3d(problem) => {
            let mesh = crate::MeshTopology::<3>::try_from(job.mesh.clone()).map_err(|source| ProjectError::Mesh {
                job_id: job.id.clone(),
                source,
            })?;
            let material = crate::ElasticityMaterial3D::from(problem.material.clone());
            let constraints: Vec<crate::DisplacementConstraint3D> = problem
                .constraints
                .iter()
                .map(|c| crate::DisplacementConstraint3D::from(c.clone()))
                .collect();
            let forces = project_forces_to_nodal_forces_3d(problem.forces.clone());
            let problem_internal = crate::fem::elasticity::ElasticityProblem3D {
                material,
                constraints,
                forces,
            };
            let solver_options = LinearSolverOptions::from(problem.solver_options.clone());
            Ok((mesh, problem_internal, solver_options))
        }
        _ => Err(ProjectError::InvalidConfiguration {
            job_id: job.id.clone(),
            message: "expected elasticity_3d physics".to_string(),
        }),
    }
}

pub fn job_to_modal(job: &ProjectJob) -> Result<(Mesh, crate::fem::modal::ModalProblem, LinearSolverOptions), ProjectError> {
    match &job.physics {
        ProjectPhysics::Modal(problem) => {
            let mesh = Mesh::try_from(job.mesh.clone()).map_err(|source| ProjectError::Mesh {
                job_id: job.id.clone(),
                source,
            })?;
            let material = crate::ElasticityMaterial::from(problem.material.clone());
            let constraints: Vec<crate::DisplacementConstraint> = problem
                .constraints
                .iter()
                .map(|c| crate::DisplacementConstraint::from(c.clone()))
                .collect();
            let elasticity = crate::fem::elasticity::ElasticityProblem {
                material,
                thickness: problem.thickness,
                constraints,
                forces: vec![],
            };
            let problem_internal = crate::fem::modal::ModalProblem {
                elasticity,
                density: problem.density,
                mode_count: problem.requested_modes,
            };
            let solver_options = LinearSolverOptions::from(problem.solver_options.clone());
            Ok((mesh, problem_internal, solver_options))
        }
        _ => Err(ProjectError::InvalidConfiguration {
            job_id: job.id.clone(),
            message: "expected modal physics".to_string(),
        }),
    }
}

pub fn job_to_modal_3d(job: &ProjectJob) -> Result<(crate::MeshTopology<3>, crate::fem::modal::ModalProblem3D, LinearSolverOptions), ProjectError> {
    match &job.physics {
        ProjectPhysics::Modal3d(problem) => {
            let mesh = crate::MeshTopology::<3>::try_from(job.mesh.clone()).map_err(|source| ProjectError::Mesh {
                job_id: job.id.clone(),
                source,
            })?;
            let material = crate::ElasticityMaterial3D::from(problem.material.clone());
            let constraints: Vec<crate::DisplacementConstraint3D> = problem
                .constraints
                .iter()
                .map(|c| crate::DisplacementConstraint3D::from(c.clone()))
                .collect();
            let elasticity = crate::fem::elasticity::ElasticityProblem3D {
                material,
                constraints,
                forces: vec![],
            };
            let problem_internal = crate::fem::modal::ModalProblem3D {
                elasticity,
                density: problem.density,
                mode_count: problem.requested_modes,
            };
            let solver_options = LinearSolverOptions::from(problem.solver_options.clone());
            Ok((mesh, problem_internal, solver_options))
        }
        _ => Err(ProjectError::InvalidConfiguration {
            job_id: job.id.clone(),
            message: "expected modal_3d physics".to_string(),
        }),
    }
}

pub fn project_forces_to_nodal_forces(forces: Vec<ProjectElasticityForce>) -> Vec<crate::fem::elasticity::NodalForce> {
    let mut map = std::collections::BTreeMap::new();
    for f in forces {
        let entry = map.entry(f.node).or_insert((0.0, 0.0));
        match f.component.to_lowercase().as_str() {
            "x" => entry.0 += f.value,
            "y" => entry.1 += f.value,
            _ => {}
        }
    }
    map.into_iter()
        .map(|(node, (fx, fy))| crate::fem::elasticity::NodalForce { node, fx, fy })
        .collect()
}

pub fn project_forces_to_nodal_forces_3d(forces: Vec<ProjectElasticityForce3D>) -> Vec<crate::fem::elasticity::NodalForce3D> {
    let mut map = std::collections::BTreeMap::new();
    for f in forces {
        let entry = map.entry(f.node).or_insert((0.0, 0.0, 0.0));
        match f.component.to_lowercase().as_str() {
            "x" => entry.0 += f.value,
            "y" => entry.1 += f.value,
            "z" => entry.2 += f.value,
            _ => {}
        }
    }
    map.into_iter()
        .map(|(node, (fx, fy, fz))| crate::fem::elasticity::NodalForce3D { node, fx, fy, fz })
        .collect()
}

impl From<ProjectElasticityMaterial> for crate::ElasticityMaterial {
    fn from(val: ProjectElasticityMaterial) -> Self {
        Self {
            young_modulus: val.young_modulus,
            poisson_ratio: val.poisson_ratio,
            model: match val.model.to_lowercase().as_str() {
                "plane_strain" => crate::ElasticityModel::PlaneStrain,
                _ => crate::ElasticityModel::PlaneStress,
            },
        }
    }
}

impl From<ProjectElasticityConstraint> for crate::DisplacementConstraint {
    fn from(val: ProjectElasticityConstraint) -> Self {
        Self {
            node: val.node,
            component: match val.component.to_lowercase().as_str() {
                "y" => crate::DisplacementComponent::Y,
                _ => crate::DisplacementComponent::X,
            },
            value: val.value,
        }
    }
}

impl From<ProjectElasticityConstraint3D> for crate::DisplacementConstraint3D {
    fn from(val: ProjectElasticityConstraint3D) -> Self {
        Self {
            node: val.node,
            component: match val.component.to_lowercase().as_str() {
                "y" => crate::DisplacementComponent3D::Y,
                "z" => crate::DisplacementComponent3D::Z,
                _ => crate::DisplacementComponent3D::X,
            },
            value: val.value,
        }
    }
}

impl From<ProjectElasticityMaterial3D> for crate::ElasticityMaterial3D {
    fn from(val: ProjectElasticityMaterial3D) -> Self {
        Self {
            young_modulus: val.young_modulus,
            poisson_ratio: val.poisson_ratio,
        }
    }
}

pub fn default_project_solver_options() -> ProjectLinearSolverOptions {
    ProjectLinearSolverOptions::from(LinearSolverOptions::from(SolverOptions::default()))
}
