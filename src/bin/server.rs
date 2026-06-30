use std::{env, net::SocketAddr, process};

use axum::{
    Json, Router,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use kepler::{
    LinearSolverBackend, LinearSolverOptions, Mesh, Point2, PoissonProblem, PreconditionerKind,
    ProjectFile, ProjectPhysics, SolverDiagnostics, SolverOptions, Tri3, job_to_poisson,
    solve_poisson_with_solver, validate_project,
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
    Router::new()
        .route("/health", get(health))
        .route("/solve/poisson", post(solve_poisson_endpoint))
        .route("/projects/validate", post(validate_project_endpoint))
        .route("/projects/solve", post(solve_project_endpoint))
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
    validate_project(&request.project)?;
    let mut jobs = Vec::with_capacity(request.project.jobs.len());

    for job in &request.project.jobs {
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

    Ok(Json(ProjectSolveResponse {
        schema_version: request.project.schema_version,
        status: "completed",
        jobs,
    }))
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
                Some(value) => return Err(format!("unsupported solver backend '{value}'")),
            },
            preconditioner: match value.preconditioner.as_deref() {
                None | Some("none") => PreconditionerKind::None,
                Some("jacobi") => PreconditionerKind::Jacobi,
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

#[derive(Debug, Serialize)]
struct ProjectValidationResponse {
    schema_version: u32,
    status: &'static str,
    job_count: usize,
    jobs: Vec<ProjectJobSummaryResponse>,
}

#[derive(Debug, Serialize)]
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

#[derive(Debug, Serialize)]
struct ProjectSolveResponse {
    schema_version: u32,
    status: &'static str,
    jobs: Vec<ProjectSolveJobResponse>,
}

#[derive(Debug, Serialize)]
struct ProjectSolveJobResponse {
    id: String,
    status: &'static str,
    physics: &'static str,
    values: Vec<f64>,
    iterations: usize,
    residual_norm: f64,
    diagnostics: DiagnosticsResponse,
}

#[derive(Debug, Serialize)]
struct DiagnosticsResponse {
    backend: &'static str,
    preconditioner: &'static str,
    converged: bool,
    initial_residual_norm: f64,
    residual_history: Vec<f64>,
}

impl From<SolverDiagnostics> for DiagnosticsResponse {
    fn from(value: SolverDiagnostics) -> Self {
        Self {
            backend: match value.backend {
                LinearSolverBackend::ConjugateGradient => "conjugate_gradient",
                LinearSolverBackend::Gmres => "gmres",
                LinearSolverBackend::DenseDirect => "dense_direct",
            },
            preconditioner: match value.preconditioner {
                PreconditionerKind::None => "none",
                PreconditionerKind::Jacobi => "jacobi",
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
        assert_eq!(body["status"], json!("valid"));
        assert_eq!(body["job_count"], json!(1));
        assert_eq!(body["jobs"][0]["id"], json!("solve-square"));
        assert_eq!(body["jobs"][0]["physics"], json!("poisson"));
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
        assert_eq!(body["status"], json!("completed"));
        assert_eq!(body["jobs"][0]["status"], json!("completed"));
        assert_eq!(body["jobs"][0]["values"][4], json!(1.0 / 12.0));
        assert_eq!(
            body["jobs"][0]["diagnostics"]["backend"],
            json!("dense_direct")
        );
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
        assert_eq!(body["code"], json!("bad_request"));
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("schema version 99")
        );
    }

    async fn body_json(response: axum::response::Response) -> Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
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
        json!({
            "project": {
                "schema_version": 1,
                "name": "server project",
                "jobs": [
                    {
                        "id": "solve-square",
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
                        "physics": {
                            "kind": "poisson",
                            "conductivity": 1.0,
                            "source": { "kind": "constant", "value": 1.0 },
                            "dirichlet": [
                                { "node": 0, "value": 0.0 },
                                { "node": 1, "value": 0.0 },
                                { "node": 2, "value": 0.0 },
                                { "node": 3, "value": 0.0 }
                            ],
                            "solver_options": {
                                "max_iterations": 10000,
                                "tolerance": 1e-10,
                                "backend": "dense_direct",
                                "preconditioner": "none",
                                "record_residual_history": false
                            }
                        },
                        "output": { "format": "solution" }
                    }
                ]
            }
        })
    }
}
