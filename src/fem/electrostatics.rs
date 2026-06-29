use crate::{
    fem::poisson::{
        PoissonError, PoissonProblem, PoissonProblem3D, PoissonResult, PoissonSolverResult,
        solve_poisson, solve_poisson_3d, solve_poisson_3d_with_solver, solve_poisson_with_solver,
    },
    linalg::{LinearSolverOptions, SolverDiagnostics, SolverOptions},
    mesh::{Mesh, MeshTopology, NodeId},
};

pub struct ElectrostaticProblem<F> {
    pub permittivity: f64,
    pub charge_density: F,
    pub prescribed_potentials: Vec<(NodeId, f64)>,
}

pub struct ElectrostaticProblem3D<F> {
    pub permittivity: f64,
    pub charge_density: F,
    pub prescribed_potentials: Vec<(NodeId, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElectricPotentialResult {
    pub potentials: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElectricPotentialSolverResult {
    pub potentials: Vec<f64>,
    pub diagnostics: SolverDiagnostics,
}

pub fn solve_electrostatics<F>(
    mesh: &Mesh,
    problem: &ElectrostaticProblem<F>,
    options: SolverOptions,
) -> Result<ElectricPotentialResult, PoissonError>
where
    F: Fn(f64, f64) -> f64,
{
    let poisson_problem = PoissonProblem {
        conductivity: problem.permittivity,
        source: &problem.charge_density,
        dirichlet: problem.prescribed_potentials.clone(),
    };
    solve_poisson(mesh, &poisson_problem, options).map(ElectricPotentialResult::from)
}

pub fn solve_electrostatics_with_solver<F>(
    mesh: &Mesh,
    problem: &ElectrostaticProblem<F>,
    options: LinearSolverOptions,
) -> Result<ElectricPotentialSolverResult, PoissonError>
where
    F: Fn(f64, f64) -> f64,
{
    let poisson_problem = PoissonProblem {
        conductivity: problem.permittivity,
        source: &problem.charge_density,
        dirichlet: problem.prescribed_potentials.clone(),
    };
    solve_poisson_with_solver(mesh, &poisson_problem, options)
        .map(ElectricPotentialSolverResult::from)
}

pub fn solve_electrostatics_3d<F>(
    mesh: &MeshTopology<3>,
    problem: &ElectrostaticProblem3D<F>,
    options: SolverOptions,
) -> Result<ElectricPotentialResult, PoissonError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let poisson_problem = PoissonProblem3D {
        conductivity: problem.permittivity,
        source: &problem.charge_density,
        dirichlet: problem.prescribed_potentials.clone(),
    };
    solve_poisson_3d(mesh, &poisson_problem, options).map(ElectricPotentialResult::from)
}

pub fn solve_electrostatics_3d_with_solver<F>(
    mesh: &MeshTopology<3>,
    problem: &ElectrostaticProblem3D<F>,
    options: LinearSolverOptions,
) -> Result<ElectricPotentialSolverResult, PoissonError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let poisson_problem = PoissonProblem3D {
        conductivity: problem.permittivity,
        source: &problem.charge_density,
        dirichlet: problem.prescribed_potentials.clone(),
    };
    solve_poisson_3d_with_solver(mesh, &poisson_problem, options)
        .map(ElectricPotentialSolverResult::from)
}

impl From<PoissonResult> for ElectricPotentialResult {
    fn from(value: PoissonResult) -> Self {
        Self {
            potentials: value.values,
            iterations: value.iterations,
            residual_norm: value.residual_norm,
        }
    }
}

impl From<PoissonSolverResult> for ElectricPotentialSolverResult {
    fn from(value: PoissonSolverResult) -> Self {
        Self {
            potentials: value.values,
            diagnostics: value.diagnostics,
        }
    }
}
