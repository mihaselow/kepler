use std::{
    collections::BTreeMap,
    env,
    net::SocketAddr,
    process,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use kepler::{
    LinearSolverBackend, LinearSolverOptions, Mesh, Point2, PoissonProblem, PreconditionerKind,
    ProjectFile, ProjectPhysics, SolverDiagnostics, SolverOptions, Tri3, job_to_poisson,
    parse_mesh_str, parse_params_str, parse_project_str, solve_poisson_with_solver,
    validate_project,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = match parse_addr(env::args().skip(1)) {
        Ok(addr) => addr,
        Err(message) => {
            eprintln!("{message}");
            eprintln!();
            eprintln!("{}", usage());
            process::exit(2);
        }
    };

    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!(
        "kepler REST server listening on http://{}",
        listener.local_addr()?
    );
    axum::serve(listener, app()).await?;

    Ok(())
}

fn app() -> Router {
    app_with_state(AppState::default())
}

fn app_with_state(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/solve/poisson", post(solve_poisson_endpoint))
        .route("/projects/validate", post(validate_project_endpoint))
        .route("/projects/solve", post(solve_project_endpoint))
        .route("/projects/jobs", post(submit_project_job_endpoint))
        .route("/projects/jobs/{job_id}", get(project_job_status_endpoint))
        .route(
            "/projects/jobs/{job_id}/cancel",
            post(cancel_project_job_endpoint),
        )
        .route(
            "/projects/jobs/{job_id}/result",
            get(project_job_result_endpoint),
        )
        .route("/projects/artifacts", post(upload_artifact_endpoint))
        .route(
            "/projects/artifacts/{artifact_id}",
            get(download_artifact_endpoint),
        )
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn solve_poisson_endpoint(
    Json(request): Json<SolvePoissonRequest>,
) -> Result<Json<SolvePoissonResponse>, ApiError> {
    let mesh = Mesh::new(
        request
            .mesh
            .points
            .into_iter()
            .map(|point| Point2::new(point.x, point.y))
            .collect(),
        request
            .mesh
            .triangles
            .into_iter()
            .map(|triangle| Tri3::new(triangle.nodes))
            .collect(),
    )?;

    let source_constant = request.problem.source.constant;
    let problem = PoissonProblem {
        conductivity: request.problem.conductivity,
        source: move |_, _| source_constant,
        dirichlet: request
            .problem
            .dirichlet
            .into_iter()
            .map(|entry| (entry.node, entry.value))
            .collect(),
    };

    let options = request
        .solver_options
        .unwrap_or_default()
        .try_into()
        .map_err(ApiError::message)?;
    let result = solve_poisson_with_solver(&mesh, &problem, options)?;

    Ok(Json(SolvePoissonResponse {
        values: result.values,
        iterations: result.diagnostics.iterations,
        residual_norm: result.diagnostics.residual_norm,
        diagnostics: DiagnosticsResponse::from(result.diagnostics),
    }))
}

async fn validate_project_endpoint(
    Json(request): Json<ProjectEnvelopeRequest>,
) -> Result<Json<ProjectValidationResponse>, ApiError> {
    validate_project(&request.project)?;
    Ok(Json(ProjectValidationResponse {
        schema_version: request.project.schema_version,
        status: "valid",
        job_count: request.project.jobs.len(),
        jobs: request
            .project
            .jobs
            .iter()
            .map(ProjectJobSummaryResponse::from)
            .collect(),
    }))
}

async fn solve_project_endpoint(
    Json(request): Json<ProjectEnvelopeRequest>,
) -> Result<Json<ProjectSolveResponse>, ApiError> {
    solve_project(&request.project).map(Json)
}

async fn submit_project_job_endpoint(
    State(state): State<AppState>,
    Json(request): Json<ProjectEnvelopeRequest>,
) -> Result<Json<AsyncProjectSubmitResponse>, ApiError> {
    validate_project(&request.project)?;
    let job_id = state.next_job_id();
    state.insert_job(&job_id, request.project.clone());
    let worker_state = state.clone();
    let worker_job_id = job_id.clone();
    let project = request.project;

    tokio::spawn(async move {
        run_project_job(worker_state, worker_job_id, project);
    });

    Ok(Json(AsyncProjectSubmitResponse {
        job_id: job_id.clone(),
        status: "queued",
        status_url: format!("/projects/jobs/{job_id}"),
        result_url: format!("/projects/jobs/{job_id}/result"),
    }))
}

async fn project_job_status_endpoint(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<AsyncProjectStatusResponse>, ApiError> {
    let record = state
        .job_snapshot(&job_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown project job '{job_id}'")))?;
    Ok(Json(AsyncProjectStatusResponse::from_record(
        job_id, &record,
    )))
}

async fn cancel_project_job_endpoint(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<AsyncProjectStatusResponse>, ApiError> {
    let record = state
        .cancel_job(&job_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown project job '{job_id}'")))?;
    Ok(Json(AsyncProjectStatusResponse::from_record(
        job_id, &record,
    )))
}

async fn project_job_result_endpoint(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<AsyncProjectResultResponse>, ApiError> {
    let record = state
        .job_snapshot(&job_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown project job '{job_id}'")))?;
    Ok(Json(AsyncProjectResultResponse::from_record(
        job_id, &record,
    )))
}

async fn upload_artifact_endpoint(
    State(state): State<AppState>,
    Json(request): Json<ArtifactUploadRequest>,
) -> Result<Json<ArtifactUploadResponse>, ApiError> {
    if request.name.trim().is_empty() {
        return Err(ApiError::message(
            "artifact name must not be empty".to_owned(),
        ));
    }
    validate_artifact_content(request.kind, &request.content)?;
    let artifact_id = state.next_artifact_id();
    let record = ArtifactRecord {
        kind: request.kind,
        name: request.name,
        content: request.content,
    };
    let response = ArtifactUploadResponse::from_record(&artifact_id, &record);
    state.insert_artifact(&artifact_id, record);
    Ok(Json(response))
}

async fn download_artifact_endpoint(
    State(state): State<AppState>,
    Path(artifact_id): Path<String>,
) -> Result<Json<ArtifactDownloadResponse>, ApiError> {
    let record = state
        .artifact_snapshot(&artifact_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown artifact '{artifact_id}'")))?;
    Ok(Json(ArtifactDownloadResponse::from_record(
        artifact_id,
        &record,
    )))
}

fn parse_addr(args: impl IntoIterator<Item = String>) -> Result<SocketAddr, String> {
    let mut args = args.into_iter();
    let Some(first) = args.next() else {
        return Ok(default_addr());
    };

    if first == "--help" || first == "-h" {
        return Err(usage());
    }

    if first != "--addr" {
        return Err(format!("unknown option '{first}'"));
    }

    let addr = args
        .next()
        .ok_or_else(|| "missing value for --addr".to_owned())?;

    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument '{extra}'"));
    }

    addr.parse()
        .map_err(|_| format!("invalid socket address '{addr}'"))
}

fn default_addr() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 3000))
}

fn usage() -> String {
    "usage: server [--addr <ip:port>]".to_owned()
}

#[derive(Debug, Clone, Default)]
struct AppState {
    jobs: Arc<Mutex<BTreeMap<String, AsyncProjectJobRecord>>>,
    artifacts: Arc<Mutex<BTreeMap<String, ArtifactRecord>>>,
    next_job: Arc<AtomicU64>,
    next_artifact: Arc<AtomicU64>,
}

impl AppState {
    fn next_job_id(&self) -> String {
        let id = self.next_job.fetch_add(1, Ordering::Relaxed) + 1;
        format!("project-job-{id}")
    }

    fn next_artifact_id(&self) -> String {
        let id = self.next_artifact.fetch_add(1, Ordering::Relaxed) + 1;
        format!("artifact-{id}")
    }

    fn insert_job(&self, job_id: &str, project: ProjectFile) {
        let mut jobs = self.jobs.lock().expect("job store mutex poisoned");
        jobs.insert(
            job_id.to_owned(),
            AsyncProjectJobRecord {
                status: AsyncProjectJobStatus::Queued,
                project,
                logs: vec!["job accepted".to_owned()],
                result: None,
                error: None,
                cancellation_requested: false,
            },
        );
    }

    fn job_snapshot(&self, job_id: &str) -> Option<AsyncProjectJobRecord> {
        self.jobs
            .lock()
            .expect("job store mutex poisoned")
            .get(job_id)
            .cloned()
    }

    fn insert_artifact(&self, artifact_id: &str, record: ArtifactRecord) {
        self.artifacts
            .lock()
            .expect("artifact store mutex poisoned")
            .insert(artifact_id.to_owned(), record);
    }

    fn artifact_snapshot(&self, artifact_id: &str) -> Option<ArtifactRecord> {
        self.artifacts
            .lock()
            .expect("artifact store mutex poisoned")
            .get(artifact_id)
            .cloned()
    }

    fn cancel_job(&self, job_id: &str) -> Option<AsyncProjectJobRecord> {
        let mut jobs = self.jobs.lock().expect("job store mutex poisoned");
        let record = jobs.get_mut(job_id)?;
        record.cancellation_requested = true;
        match record.status {
            AsyncProjectJobStatus::Queued | AsyncProjectJobStatus::Running => {
                record.status = AsyncProjectJobStatus::Cancelled;
                record.logs.push("cancellation requested".to_owned());
            }
            AsyncProjectJobStatus::Completed
            | AsyncProjectJobStatus::Failed
            | AsyncProjectJobStatus::Cancelled => {
                record
                    .logs
                    .push("cancellation requested after terminal state".to_owned());
            }
        }
        Some(record.clone())
    }

    fn mark_running(&self, job_id: &str) -> bool {
        let mut jobs = self.jobs.lock().expect("job store mutex poisoned");
        let Some(record) = jobs.get_mut(job_id) else {
            return false;
        };
        if record.cancellation_requested || record.status == AsyncProjectJobStatus::Cancelled {
            return false;
        }
        record.status = AsyncProjectJobStatus::Running;
        record.logs.push("job started".to_owned());
        true
    }

    fn complete_job(&self, job_id: &str, result: Result<ProjectSolveResponse, String>) {
        let mut jobs = self.jobs.lock().expect("job store mutex poisoned");
        let Some(record) = jobs.get_mut(job_id) else {
            return;
        };
        if record.status == AsyncProjectJobStatus::Cancelled {
            record
                .logs
                .push("worker finished after cancellation".to_owned());
            return;
        }
        match result {
            Ok(result) => {
                record.status = AsyncProjectJobStatus::Completed;
                record.logs.push("job completed".to_owned());
                record.result = Some(result);
            }
            Err(error) => {
                record.status = AsyncProjectJobStatus::Failed;
                record.logs.push(format!("job failed: {error}"));
                record.error = Some(error);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AsyncProjectJobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl AsyncProjectJobStatus {
    const fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone)]
struct AsyncProjectJobRecord {
    status: AsyncProjectJobStatus,
    project: ProjectFile,
    logs: Vec<String>,
    result: Option<ProjectSolveResponse>,
    error: Option<String>,
    cancellation_requested: bool,
}

#[derive(Debug, Clone)]
struct ArtifactRecord {
    kind: ArtifactKind,
    name: String,
    content: String,
}

fn run_project_job(state: AppState, job_id: String, project: ProjectFile) {
    if !state.mark_running(&job_id) {
        return;
    }
    let result = solve_project(&project).map_err(|error| error.message);
    state.complete_job(&job_id, result);
}

fn solve_project(project: &ProjectFile) -> Result<ProjectSolveResponse, ApiError> {
    validate_project(project)?;
    let mut jobs = Vec::with_capacity(project.jobs.len());

    for job in &project.jobs {
        let (mesh, config) = job_to_poisson(job)?;
        let source = config.source;
        let problem = PoissonProblem {
            conductivity: config.conductivity,
            source: move |x, y| source.value_at(x, y),
            dirichlet: config.dirichlet,
        };
        let result = solve_poisson_with_solver(&mesh, &problem, config.solver_options)?;
        jobs.push(ProjectSolveJobResponse {
            id: job.id.clone(),
            status: "completed",
            physics: physics_name(&job.physics),
            values: result.values,
            iterations: result.diagnostics.iterations,
            residual_norm: result.diagnostics.residual_norm,
            diagnostics: DiagnosticsResponse::from(result.diagnostics),
        });
    }

    Ok(ProjectSolveResponse {
        schema_version: project.schema_version,
        status: "completed",
        jobs,
    })
}

fn validate_artifact_content(kind: ArtifactKind, content: &str) -> Result<(), ApiError> {
    match kind {
        ArtifactKind::Mesh => {
            parse_mesh_str(content)?;
        }
        ArtifactKind::Params => {
            parse_params_str(content)?;
        }
        ArtifactKind::Project => {
            parse_project_str(content)?;
        }
        ArtifactKind::Solution => validate_solution_artifact(content)?,
    }
    Ok(())
}

fn validate_solution_artifact(content: &str) -> Result<(), ApiError> {
    if content.lines().any(|line| line.trim() == "node value") {
        Ok(())
    } else {
        Err(ApiError::message(
            "solution artifact must contain a 'node value' header".to_owned(),
        ))
    }
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Debug, Deserialize)]
struct SolvePoissonRequest {
    mesh: MeshRequest,
    problem: ProblemRequest,
    #[serde(default)]
    solver_options: Option<SolverOptionsRequest>,
}

#[derive(Debug, Deserialize)]
struct ProjectEnvelopeRequest {
    project: ProjectFile,
}

#[derive(Debug, Deserialize)]
struct ArtifactUploadRequest {
    kind: ArtifactKind,
    name: String,
    content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ArtifactKind {
    Mesh,
    Params,
    Project,
    Solution,
}

impl ArtifactKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Mesh => "mesh",
            Self::Params => "params",
            Self::Project => "project",
            Self::Solution => "solution",
        }
    }
}

#[derive(Debug, Deserialize)]
struct MeshRequest {
    points: Vec<PointRequest>,
    triangles: Vec<TriangleRequest>,
}

#[derive(Debug, Deserialize)]
struct PointRequest {
    x: f64,
    y: f64,
}

#[derive(Debug, Deserialize)]
struct TriangleRequest {
    nodes: [usize; 3],
}

#[derive(Debug, Deserialize)]
struct ProblemRequest {
    conductivity: f64,
    source: SourceRequest,
    #[serde(default)]
    dirichlet: Vec<DirichletRequest>,
}

#[derive(Debug, Deserialize)]
struct SourceRequest {
    constant: f64,
}

#[derive(Debug, Deserialize)]
struct DirichletRequest {
    node: usize,
    value: f64,
}

#[derive(Debug, Default, Deserialize)]
struct SolverOptionsRequest {
    #[serde(default)]
    max_iterations: Option<usize>,
    #[serde(default)]
    tolerance: Option<f64>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    preconditioner: Option<String>,
    #[serde(default)]
    record_residual_history: Option<bool>,
}

impl TryFrom<SolverOptionsRequest> for LinearSolverOptions {
    type Error = String;

    fn try_from(value: SolverOptionsRequest) -> Result<Self, Self::Error> {
        let defaults = SolverOptions::default();
        Ok(Self {
            max_iterations: value.max_iterations.unwrap_or(defaults.max_iterations),
            tolerance: value.tolerance.unwrap_or(defaults.tolerance),
            backend: match value.backend.as_deref() {
                None | Some("cg") | Some("conjugate_gradient") => {
                    LinearSolverBackend::ConjugateGradient
                }
                Some("gmres") => LinearSolverBackend::Gmres,
                Some("dense_direct") => LinearSolverBackend::DenseDirect,
                Some("sparse_ldl") => LinearSolverBackend::SparseLdl,
                Some(value) => return Err(format!("unsupported solver backend '{value}'")),
            },
            preconditioner: match value.preconditioner.as_deref() {
                None | Some("none") => PreconditionerKind::None,
                Some("jacobi") => PreconditionerKind::Jacobi,
                Some("amg") => PreconditionerKind::Amg,
                Some(value) => return Err(format!("unsupported preconditioner '{value}'")),
            },
            record_residual_history: value.record_residual_history.unwrap_or(false),
        })
    }
}

#[derive(Debug, Serialize)]
struct SolvePoissonResponse {
    values: Vec<f64>,
    iterations: usize,
    residual_norm: f64,
    diagnostics: DiagnosticsResponse,
}

#[derive(Debug, Clone, Serialize)]
struct ProjectValidationResponse {
    schema_version: u32,
    status: &'static str,
    job_count: usize,
    jobs: Vec<ProjectJobSummaryResponse>,
}

#[derive(Debug, Clone, Serialize)]
struct ProjectJobSummaryResponse {
    id: String,
    status: &'static str,
    physics: &'static str,
    points: usize,
    triangles: usize,
}

impl From<&kepler::ProjectJob> for ProjectJobSummaryResponse {
    fn from(value: &kepler::ProjectJob) -> Self {
        Self {
            id: value.id.clone(),
            status: "valid",
            physics: physics_name(&value.physics),
            points: value.mesh.points.len(),
            triangles: value.mesh.triangles.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ProjectSolveResponse {
    schema_version: u32,
    status: &'static str,
    jobs: Vec<ProjectSolveJobResponse>,
}

#[derive(Debug, Clone, Serialize)]
struct ProjectSolveJobResponse {
    id: String,
    status: &'static str,
    physics: &'static str,
    values: Vec<f64>,
    iterations: usize,
    residual_norm: f64,
    diagnostics: DiagnosticsResponse,
}

#[derive(Debug, Clone, Serialize)]
struct DiagnosticsResponse {
    backend: &'static str,
    preconditioner: &'static str,
    converged: bool,
    initial_residual_norm: f64,
    residual_history: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct AsyncProjectSubmitResponse {
    job_id: String,
    status: &'static str,
    status_url: String,
    result_url: String,
}

#[derive(Debug, Serialize)]
struct AsyncProjectStatusResponse {
    job_id: String,
    status: &'static str,
    schema_version: u32,
    project_job_count: usize,
    logs: Vec<String>,
    error: Option<String>,
    result_url: String,
}

impl AsyncProjectStatusResponse {
    fn from_record(job_id: String, record: &AsyncProjectJobRecord) -> Self {
        Self {
            result_url: format!("/projects/jobs/{job_id}/result"),
            job_id,
            status: record.status.as_str(),
            schema_version: record.project.schema_version,
            project_job_count: record.project.jobs.len(),
            logs: record.logs.clone(),
            error: record.error.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct AsyncProjectResultResponse {
    job_id: String,
    status: &'static str,
    result: Option<ProjectSolveResponse>,
    error: Option<String>,
    logs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ArtifactUploadResponse {
    artifact_id: String,
    kind: &'static str,
    name: String,
    size_bytes: usize,
    download_url: String,
}

impl ArtifactUploadResponse {
    fn from_record(artifact_id: &str, record: &ArtifactRecord) -> Self {
        Self {
            artifact_id: artifact_id.to_owned(),
            kind: record.kind.as_str(),
            name: record.name.clone(),
            size_bytes: record.content.len(),
            download_url: format!("/projects/artifacts/{artifact_id}"),
        }
    }
}

#[derive(Debug, Serialize)]
struct ArtifactDownloadResponse {
    artifact_id: String,
    kind: &'static str,
    name: String,
    content: String,
    size_bytes: usize,
}

impl ArtifactDownloadResponse {
    fn from_record(artifact_id: String, record: &ArtifactRecord) -> Self {
        Self {
            artifact_id,
            kind: record.kind.as_str(),
            name: record.name.clone(),
            size_bytes: record.content.len(),
            content: record.content.clone(),
        }
    }
}

impl AsyncProjectResultResponse {
    fn from_record(job_id: String, record: &AsyncProjectJobRecord) -> Self {
        Self {
            job_id,
            status: record.status.as_str(),
            result: record.result.clone(),
            error: record.error.clone(),
            logs: record.logs.clone(),
        }
    }
}

impl From<SolverDiagnostics> for DiagnosticsResponse {
    fn from(value: SolverDiagnostics) -> Self {
        Self {
            backend: match value.backend {
                LinearSolverBackend::ConjugateGradient => "conjugate_gradient",
                LinearSolverBackend::Gmres => "gmres",
                LinearSolverBackend::DenseDirect => "dense_direct",
                LinearSolverBackend::SparseLdl => "sparse_ldl",
            },
            preconditioner: match value.preconditioner {
                PreconditionerKind::None => "none",
                PreconditionerKind::Jacobi => "jacobi",
                PreconditionerKind::Amg => "amg",
            },
            converged: value.converged,
            initial_residual_norm: value.initial_residual_norm,
            residual_history: value.residual_history,
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: &'static str,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(error: impl std::error::Error) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: error.to_string(),
        }
    }

    fn message(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message,
        }
    }

    fn not_found(message: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: self.message,
                code: "bad_request",
            }),
        )
            .into_response()
    }
}

impl From<kepler::MeshError> for ApiError {
    fn from(value: kepler::MeshError) -> Self {
        Self::bad_request(value)
    }
}

impl From<kepler::fem::poisson::PoissonError> for ApiError {
    fn from(value: kepler::fem::poisson::PoissonError) -> Self {
        Self::bad_request(value)
    }
}

impl From<kepler::ProjectError> for ApiError {
    fn from(value: kepler::ProjectError) -> Self {
        Self::bad_request(value)
    }
}

impl From<kepler::FileIoError> for ApiError {
    fn from(value: kepler::FileIoError) -> Self {
        Self::bad_request(value)
    }
}

fn physics_name(physics: &ProjectPhysics) -> &'static str {
    match physics {
        ProjectPhysics::Poisson(_) => "poisson",
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use serde_json::{Value, json};
    use std::time::{Duration, Instant};
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn health_endpoint_reports_ok() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body, json!({ "status": "ok" }));
    }

    #[tokio::test]
    async fn solve_endpoint_returns_known_square_solution() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/solve/poisson")
                    .header("content-type", "application/json")
                    .body(Body::from(square_request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["values"][0], json!(0.0));
        assert_eq!(body["values"][4], json!(1.0 / 12.0));
        assert_eq!(body["iterations"], json!(1));
        assert_eq!(body["diagnostics"]["backend"], json!("conjugate_gradient"));
        assert_eq!(body["diagnostics"]["preconditioner"], json!("none"));
    }

    #[tokio::test]
    async fn solve_endpoint_accepts_solver_stack_options() {
        let mut request = square_request();
        request["solver_options"]["backend"] = json!("gmres");
        request["solver_options"]["preconditioner"] = json!("none");
        request["solver_options"]["record_residual_history"] = json!(true);

        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/solve/poisson")
                    .header("content-type", "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["diagnostics"]["backend"], json!("gmres"));
        assert_eq!(body["diagnostics"]["preconditioner"], json!("none"));
        assert_eq!(
            body["diagnostics"]["residual_history"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn solve_endpoint_rejects_unknown_solver_backend() {
        let mut request = square_request();
        request["solver_options"]["backend"] = json!("magic");

        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/solve/poisson")
                    .header("content-type", "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("unsupported solver backend")
        );
    }

    #[tokio::test]
    async fn solve_endpoint_rejects_invalid_mesh() {
        let mut request = square_request();
        request["mesh"]["triangles"][0]["nodes"] = json!([0, 0, 4]);

        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/solve/poisson")
                    .header("content-type", "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert!(body["error"].as_str().unwrap().contains("duplicate"));
    }

    #[tokio::test]
    async fn project_validate_endpoint_accepts_project_envelope() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/validate")
                    .header("content-type", "application/json")
                    .body(Body::from(project_request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body, expected_project_validate_response());
    }

    #[tokio::test]
    async fn project_solve_endpoint_runs_synchronous_poisson_job() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/solve")
                    .header("content-type", "application/json")
                    .body(Body::from(project_request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body, expected_project_solve_response());
    }

    #[tokio::test]
    #[ignore = "benchmark-style verification; run with `cargo test --bin server -- --ignored --nocapture`"]
    async fn benchmark_rest_project_validate_and_solve_workflow() {
        let app = app();
        let iterations = 25;
        let start = Instant::now();

        for _ in 0..iterations {
            let validate_response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/projects/validate")
                        .header("content-type", "application/json")
                        .body(Body::from(project_request().to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(validate_response.status(), StatusCode::OK);

            let solve_response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/projects/solve")
                        .header("content-type", "application/json")
                        .body(Body::from(project_request().to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(solve_response.status(), StatusCode::OK);
            assert_eq!(
                body_json(solve_response).await,
                expected_project_solve_response()
            );
        }

        let elapsed = start.elapsed();
        eprintln!("rest project validate plus solve: {iterations} iterations in {elapsed:?}");
        assert!(elapsed > Duration::ZERO);
    }

    #[tokio::test]
    async fn project_endpoint_returns_stable_error_schema() {
        let mut request = project_request();
        request["project"]["schema_version"] = json!(99);

        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/validate")
                    .header("content-type", "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body, expected_bad_schema_error_response());
    }

    #[tokio::test]
    async fn async_project_job_runs_to_result() {
        let app = app();
        let submit_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(project_request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(submit_response.status(), StatusCode::OK);
        let submit_body = body_json(submit_response).await;
        let job_id = submit_body["job_id"].as_str().unwrap();
        assert_eq!(submit_body["status"], json!("queued"));

        let result_body = poll_async_result(app, job_id).await;
        assert_eq!(result_body["status"], json!("completed"));
        assert_eq!(result_body["result"], expected_project_solve_response());
        assert!(
            result_body["logs"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| entry == "job completed")
        );
    }

    #[tokio::test]
    async fn async_project_job_status_reports_logs() {
        let app = app();
        let submit_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(project_request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let submit_body = body_json(submit_response).await;
        let job_id = submit_body["job_id"].as_str().unwrap();
        let _ = poll_async_result(app.clone(), job_id).await;

        let status_response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/projects/jobs/{job_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(status_response.status(), StatusCode::OK);
        let status_body = body_json(status_response).await;
        assert_eq!(status_body["job_id"], json!(job_id));
        assert_eq!(status_body["status"], json!("completed"));
        assert_eq!(status_body["project_job_count"], json!(1));
        assert!(status_body["result_url"].as_str().unwrap().contains(job_id));
    }

    #[tokio::test]
    async fn async_project_job_cancel_hook_is_available() {
        let app = app();
        let submit_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(project_request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let submit_body = body_json(submit_response).await;
        let job_id = submit_body["job_id"].as_str().unwrap();

        let cancel_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/projects/jobs/{job_id}/cancel"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(cancel_response.status(), StatusCode::OK);
        let cancel_body = body_json(cancel_response).await;
        assert_eq!(cancel_body["job_id"], json!(job_id));
        assert!(
            ["cancelled", "completed", "running", "queued"]
                .contains(&cancel_body["status"].as_str().unwrap())
        );
        assert!(
            cancel_body["logs"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| entry.as_str().unwrap().contains("cancellation requested"))
        );
    }

    #[tokio::test]
    async fn async_project_job_unknown_id_returns_not_found() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/projects/jobs/missing/result")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = body_json(response).await;
        assert_eq!(body["code"], json!("bad_request"));
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("unknown project job")
        );
    }

    #[tokio::test]
    async fn artifact_upload_and_download_round_trip_mesh() {
        let app = app();
        let upload_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/artifacts")
                    .header("content-type", "application/json")
                    .body(Body::from(mesh_artifact_upload().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(upload_response.status(), StatusCode::OK);
        let upload_body = body_json(upload_response).await;
        let artifact_id = upload_body["artifact_id"].as_str().unwrap();
        assert_eq!(upload_body["kind"], json!("mesh"));
        assert!(
            upload_body["download_url"]
                .as_str()
                .unwrap()
                .contains(artifact_id)
        );

        let download_response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/projects/artifacts/{artifact_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(download_response.status(), StatusCode::OK);
        let download_body = body_json(download_response).await;
        assert_eq!(download_body["artifact_id"], json!(artifact_id));
        assert_eq!(download_body["name"], json!("triangle.mesh"));
        assert_eq!(download_body["content"], json!(triangle_mesh_text()));
    }

    #[tokio::test]
    async fn artifact_upload_validates_params_project_and_solution() {
        for (kind, name, content) in [
            ("params", "triangle.params", params_text()),
            (
                "project",
                "project.json",
                project_request()["project"].to_string(),
            ),
            (
                "solution",
                "result.solution",
                "# kepler solution\nnode value\n0 0\n".to_owned(),
            ),
        ] {
            let response = app()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/projects/artifacts")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            json!({
                                "kind": kind,
                                "name": name,
                                "content": content
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);
            let body = body_json(response).await;
            assert_eq!(body["kind"], json!(kind));
            assert_eq!(body["name"], json!(name));
        }
    }

    #[tokio::test]
    async fn artifact_upload_rejects_invalid_mesh() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/projects/artifacts")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "kind": "mesh",
                            "name": "bad.mesh",
                            "content": "nodes\n0 0.0 0.0\n"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["code"], json!("bad_request"));
        assert!(body["error"].as_str().unwrap().contains("triangles"));
    }

    #[tokio::test]
    async fn artifact_download_unknown_id_returns_not_found() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/projects/artifacts/missing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = body_json(response).await;
        assert!(body["error"].as_str().unwrap().contains("unknown artifact"));
    }

    async fn body_json(response: axum::response::Response) -> Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn expected_project_validate_response() -> Value {
        serde_json::from_str(include_str!(
            "../../examples/data/rest_project_validate_response.json"
        ))
        .unwrap()
    }

    fn expected_project_solve_response() -> Value {
        serde_json::from_str(include_str!(
            "../../examples/data/rest_project_solve_response.json"
        ))
        .unwrap()
    }

    fn expected_bad_schema_error_response() -> Value {
        serde_json::from_str(include_str!(
            "../../examples/data/rest_bad_schema_error_response.json"
        ))
        .unwrap()
    }

    fn mesh_artifact_upload() -> Value {
        serde_json::from_str(include_str!(
            "../../examples/data/rest_mesh_artifact_upload.json"
        ))
        .unwrap()
    }

    async fn poll_async_result(app: Router, job_id: &str) -> Value {
        for _ in 0..10 {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("/projects/jobs/{job_id}/result"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = body_json(response).await;
            if body["status"] == json!("completed") || body["status"] == json!("failed") {
                return body;
            }
            tokio::task::yield_now().await;
        }
        panic!("async project job did not reach terminal state");
    }

    fn square_request() -> Value {
        json!({
            "mesh": {
                "points": [
                    { "x": 0.0, "y": 0.0 },
                    { "x": 1.0, "y": 0.0 },
                    { "x": 1.0, "y": 1.0 },
                    { "x": 0.0, "y": 1.0 },
                    { "x": 0.5, "y": 0.5 }
                ],
                "triangles": [
                    { "nodes": [0, 1, 4] },
                    { "nodes": [1, 2, 4] },
                    { "nodes": [2, 3, 4] },
                    { "nodes": [3, 0, 4] }
                ]
            },
            "problem": {
                "conductivity": 1.0,
                "source": { "constant": 1.0 },
                "dirichlet": [
                    { "node": 0, "value": 0.0 },
                    { "node": 1, "value": 0.0 },
                    { "node": 2, "value": 0.0 },
                    { "node": 3, "value": 0.0 }
                ]
            },
            "solver_options": {
                "max_iterations": 10000,
                "tolerance": 1e-10
            }
        })
    }

    fn project_request() -> Value {
        serde_json::from_str(include_str!(
            "../../examples/data/rest_project_request.json"
        ))
        .unwrap()
    }

    fn triangle_mesh_text() -> &'static str {
        "nodes\n0 0.0 0.0\n1 1.0 0.0\n2 0.0 1.0\n\ntriangles\n0 0 1 2\n"
    }

    fn params_text() -> String {
        "conductivity 1.0\nsource constant 1.0\n\ndirichlet\n0 0.0\n".to_owned()
    }
}
