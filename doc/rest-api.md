# REST API

Kepler includes a separate single-binary HTTP server for running solver functionality through JSON REST endpoints.

Run the server:

```shell
cargo run --bin server
```

By default it listens on `127.0.0.1:3000`. To choose a different address:

```shell
cargo run --bin server -- --addr 127.0.0.1:4000
```

## Endpoints

### `GET /health`

Returns a simple server status payload:

```json
{
  "status": "ok"
}
```

### `POST /solve/poisson`

Solves the current 2D scalar Poisson problem from a JSON payload.

Request:

```json
{
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
    "tolerance": 1e-10,
    "backend": "conjugate_gradient",
    "preconditioner": "none",
    "record_residual_history": false
  }
}
```

`solver_options` is optional. Missing iteration and tolerance fields fall back to `SolverOptions::default()`. Supported `backend` values are `conjugate_gradient`, `cg`, `gmres`, and `dense_direct`. Supported `preconditioner` values are `none` and `jacobi`.

Response:

```json
{
  "values": [0.0, 0.0, 0.0, 0.0, 0.08333333333333333],
  "iterations": 1,
  "residual_norm": 0.0,
  "diagnostics": {
    "backend": "conjugate_gradient",
    "preconditioner": "none",
    "converged": true,
    "initial_residual_norm": 0.16666666666666666,
    "residual_history": []
  }
}
```

Errors return HTTP `400` with a JSON body:

```json
{
  "error": "triangle 0 contains duplicate node indices",
  "code": "bad_request"
}
```

### `POST /projects/validate`

Validates a versioned project file without solving it. The body uses a stable envelope:

```json
{
  "project": {
    "schema_version": 1,
    "name": "square poisson",
    "jobs": [
      {
        "id": "solve-square",
        "mesh": {
          "points": [
            { "x": 0.0, "y": 0.0 },
            { "x": 1.0, "y": 0.0 },
            { "x": 0.0, "y": 1.0 }
          ],
          "triangles": [
            { "nodes": [0, 1, 2] }
          ]
        },
        "physics": {
          "kind": "poisson",
          "conductivity": 1.0,
          "source": { "kind": "constant", "value": 1.0 },
          "dirichlet": [
            { "node": 0, "value": 0.0 },
            { "node": 2, "value": 0.0 }
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
}
```

A valid project returns:

```json
{
  "schema_version": 1,
  "status": "valid",
  "job_count": 1,
  "jobs": [
    {
      "id": "solve-square",
      "status": "valid",
      "physics": "poisson",
      "points": 3,
      "triangles": 1
    }
  ]
}
```

### `POST /projects/solve`

Runs all jobs in a v1 project synchronously. The request body uses the same `{ "project": ... }` envelope as `/projects/validate`.

The current implementation supports small synchronous Poisson jobs only. A successful response is also envelope-shaped:

```json
{
  "schema_version": 1,
  "status": "completed",
  "jobs": [
    {
      "id": "solve-square",
      "status": "completed",
      "physics": "poisson",
      "values": [0.0, 0.25, 0.0],
      "iterations": 1,
      "residual_norm": 0.0,
      "diagnostics": {
        "backend": "dense_direct",
        "preconditioner": "none",
        "converged": true,
        "initial_residual_norm": 0.0,
        "residual_history": []
      }
    }
  ]
}
```

Project endpoint errors use the same stable error object:

```json
{
  "error": "unsupported project schema version 99; expected 1",
  "code": "bad_request"
}
```

### `POST /projects/jobs`

Submits a v1 project for asynchronous in-memory execution. The request body uses the same `{ "project": ... }` envelope as `/projects/validate`.

Response:

```json
{
  "job_id": "project-job-1",
  "status": "queued",
  "status_url": "/projects/jobs/project-job-1",
  "result_url": "/projects/jobs/project-job-1/result"
}
```

The server stores submitted jobs in memory and starts a background task immediately. This is intended as a first workflow API layer, not durable production job storage.

### `GET /projects/jobs/{job_id}`

Returns job status, logs, and result location:

```json
{
  "job_id": "project-job-1",
  "status": "completed",
  "schema_version": 1,
  "project_job_count": 1,
  "logs": ["job accepted", "job started", "job completed"],
  "error": null,
  "result_url": "/projects/jobs/project-job-1/result"
}
```

Status values are `queued`, `running`, `completed`, `failed`, and `cancelled`.

### `POST /projects/jobs/{job_id}/cancel`

Requests cancellation for an in-memory job and returns the same status shape as `GET /projects/jobs/{job_id}`. Cancellation is currently a state hook: queued/running jobs are marked cancelled when observed by the job store, but already-running small Poisson solves may complete before the cancellation request is processed.

### `GET /projects/jobs/{job_id}/result`

Returns the current result envelope. Pending jobs return `result: null`; completed jobs include the same `ProjectSolveResponse` shape used by `/projects/solve`.

```json
{
  "job_id": "project-job-1",
  "status": "completed",
  "result": {
    "schema_version": 1,
    "status": "completed",
    "jobs": []
  },
  "error": null,
  "logs": ["job accepted", "job started", "job completed"]
}
```

### `POST /projects/artifacts`

Uploads a text artifact into the in-memory workflow store. This endpoint is JSON-based rather than multipart for now:

```json
{
  "kind": "mesh",
  "name": "square.mesh",
  "content": "nodes\n0 0.0 0.0\n1 1.0 0.0\n2 0.0 1.0\n\ntriangles\n0 0 1 2\n"
}
```

Supported `kind` values are:

- `mesh`: validated with the legacy `.mesh` parser.
- `params`: validated with the legacy `.params` parser.
- `project`: validated as a v1 project JSON document.
- `solution`: validated as a text solution artifact containing a `node value` header.

Response:

```json
{
  "artifact_id": "artifact-1",
  "kind": "mesh",
  "name": "square.mesh",
  "size_bytes": 68,
  "download_url": "/projects/artifacts/artifact-1"
}
```

### `GET /projects/artifacts/{artifact_id}`

Downloads a previously uploaded in-memory artifact:

```json
{
  "artifact_id": "artifact-1",
  "kind": "mesh",
  "name": "square.mesh",
  "content": "nodes\n...",
  "size_bytes": 68
}
```

Artifacts are stored in memory only. They are intended as the first upload/download abstraction for workflow APIs; durable storage, multipart uploads, artifact references from project jobs, and binary result bundles are future work.

## Curl Example

```shell
curl -s http://127.0.0.1:3000/solve/poisson \
  -H 'content-type: application/json' \
  -d '{
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
    }
  }'
```

## Current Scope

The REST API still focuses on small 2D Poisson solves. `/solve/poisson` preserves the original direct endpoint, while `/projects/validate`, `/projects/solve`, `/projects/jobs`, and `/projects/artifacts` introduce versioned project envelopes and in-memory artifact handling for the same supported physics. The library also has dimension-aware topology, geometry annotation, shared condition, Gmsh import, VTK export, 3D `Tet4` Poisson, 2D/3D linear elasticity, steady heat transfer, diffusion-reaction, electrostatics, modal-analysis primitives, and a richer linalg solver stack, but the REST project endpoints do not yet accept generic `ElementKind` payloads, named-region material assignments, arbitrary parameter assignments, general `ConditionSet` payloads, Gmsh uploads, VTK downloads, 3D meshes, heat problems, diffusion-reaction problems, electrostatics problems, elasticity problems, modal problems, nonlinear solves, or transient solves. Async jobs and artifacts are stored in memory only; they are lost when the server exits and do not yet support durable persistence, worker pools, authentication, multipart uploads, artifact references from project jobs, or binary result bundles.
