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
    "tolerance": 1e-10
  }
}
```

`solver_options` is optional. Missing solver option fields fall back to `SolverOptions::default()`.

Response:

```json
{
  "values": [0.0, 0.0, 0.0, 0.0, 0.08333333333333333],
  "iterations": 1,
  "residual_norm": 0.0
}
```

Errors return HTTP `400` with a JSON body:

```json
{
  "error": "triangle 0 contains duplicate node indices"
}
```

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

The REST API mirrors the current solver scope: `Tri3` meshes, constant conductivity, constant source terms, Dirichlet boundaries, and Conjugate Gradient solver options. It does not yet provide asynchronous job storage, uploaded mesh files, authentication, or multiple physics endpoints.
