# Project Workflows

Kepler is starting to move from separate `.mesh` and `.params` files toward versioned project/job descriptions. The first project schema preserves the existing synchronous 2D Poisson workflow while giving future CLI and REST APIs a stable envelope.

## Version 1 Schema

Project files are JSON documents with `schema_version = 1` and one or more jobs:

```json
{
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
          "backend": "conjugate_gradient",
          "preconditioner": "none",
          "record_residual_history": false
        }
      },
      "output": { "format": "solution" }
    }
  ]
}
```

The current model supports:

- 2D `Tri3` meshes embedded in the project.
- Synchronous scalar Poisson jobs.
- Constant source values and nodal Dirichlet constraints.
- Solver backend, preconditioner, tolerance, iteration, and residual-history settings.
- A compact `.solution` output format marker.

## Compatibility

`ProjectFile::from_legacy_poisson` adapts the existing `.mesh` plus `.params` workflow into a versioned project. `job_to_poisson` converts a project job back into the existing `Mesh` and `PoissonFileConfig` types, so current CLI and REST behavior can be preserved while new project commands and endpoints are added.

`parse_project_str`, `read_project_file`, `validate_project`, and `validate_job` provide shared schema parsing and validation for future CLI and REST entry points.

## CLI Commands

The existing legacy solve path remains available:

```shell
kepler solve --mesh square.mesh --params square.params --output square.solution
```

Project files can be validated without running a solve:

```shell
kepler project validate --project square.project.json
```

They can also be inspected for a compact summary:

```shell
kepler project inspect --project square.project.json
```

`inspect` prints the schema version, job count, project name when present, and one line per job with physics kind and mesh size. These commands currently validate v1 synchronous Poisson projects and intentionally do not replace the legacy file-driven solve command.

## Validation Rules

Project validation currently checks:

- Supported schema version.
- At least one job.
- Non-empty, unique job IDs.
- Mesh validity through the existing `Mesh` validator.
- Poisson boundary references through existing parameter validation.

Future workflow sub-steps will add REST project submission envelopes, asynchronous job state, upload/download-oriented artifact handling, and broader physics/job coverage.
