use std::{env, net::SocketAddr, process};

use axum::{
    Json, Router,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use kepler::{Mesh, Point2, PoissonProblem, SolverOptions, Tri3, solve_poisson};

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

    let options = request.solver_options.unwrap_or_default().into();
    let result = solve_poisson(&mesh, &problem, options)?;

    Ok(Json(SolvePoissonResponse {
        values: result.values,
        iterations: result.iterations,
        residual_norm: result.residual_norm,
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
}

impl From<SolverOptionsRequest> for SolverOptions {
    fn from(value: SolverOptionsRequest) -> Self {
        let defaults = SolverOptions::default();
        Self {
            max_iterations: value.max_iterations.unwrap_or(defaults.max_iterations),
            tolerance: value.tolerance.unwrap_or(defaults.tolerance),
        }
    }
}

#[derive(Debug, Serialize)]
struct SolvePoissonResponse {
    values: Vec<f64>,
    iterations: usize,
    residual_norm: f64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
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
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: self.message,
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
}
