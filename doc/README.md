# Kepler Documentation

This directory documents the solver functionality implemented in the crate.

## Current Topics

- [Minimal FEM Solver](fem-poisson.md): 2D scalar Poisson/steady heat diffusion with P1 triangular elements.
- [Solver Stack](solver-stack.md): linear solver backends, preconditioning, diagnostics, nonlinear solves, and transient solves.
- [Project Workflows](project-workflows.md): versioned project/job schemas and compatibility with legacy mesh/parameter workflows.
- [REST API](rest-api.md): HTTP endpoints for running solver jobs from JSON payloads.

## Keeping Documentation Current

Update these documents whenever a public API, supported equation, boundary condition, solver behavior, example, or validation rule changes. Documentation should describe the implemented behavior, not planned future behavior.
