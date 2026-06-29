use crate::{
    fem::poisson::{
        PoissonError, PoissonProblem, PoissonProblem3D, PoissonResult, PoissonSolverResult,
        solve_poisson, solve_poisson_3d, solve_poisson_3d_with_solver, solve_poisson_with_solver,
    },
    linalg::{LinearSolverOptions, SolverDiagnostics, SolverOptions},
    mesh::{Mesh, MeshTopology, NodeId},
};

pub struct SteadyHeatProblem<F> {
    pub thermal_conductivity: f64,
    pub heat_generation: F,
    pub prescribed_temperatures: Vec<(NodeId, f64)>,
}

pub struct SteadyHeatProblem3D<F> {
    pub thermal_conductivity: f64,
    pub heat_generation: F,
    pub prescribed_temperatures: Vec<(NodeId, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemperatureResult {
    pub temperatures: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemperatureSolverResult {
    pub temperatures: Vec<f64>,
    pub diagnostics: SolverDiagnostics,
}

pub fn solve_steady_heat<F>(
    mesh: &Mesh,
    problem: &SteadyHeatProblem<F>,
    options: SolverOptions,
) -> Result<TemperatureResult, PoissonError>
where
    F: Fn(f64, f64) -> f64,
{
    let poisson_problem = PoissonProblem {
        conductivity: problem.thermal_conductivity,
        source: &problem.heat_generation,
        dirichlet: problem.prescribed_temperatures.clone(),
    };
    solve_poisson(mesh, &poisson_problem, options).map(TemperatureResult::from)
}

pub fn solve_steady_heat_with_solver<F>(
    mesh: &Mesh,
    problem: &SteadyHeatProblem<F>,
    options: LinearSolverOptions,
) -> Result<TemperatureSolverResult, PoissonError>
where
    F: Fn(f64, f64) -> f64,
{
    let poisson_problem = PoissonProblem {
        conductivity: problem.thermal_conductivity,
        source: &problem.heat_generation,
        dirichlet: problem.prescribed_temperatures.clone(),
    };
    solve_poisson_with_solver(mesh, &poisson_problem, options).map(TemperatureSolverResult::from)
}

pub fn solve_steady_heat_3d<F>(
    mesh: &MeshTopology<3>,
    problem: &SteadyHeatProblem3D<F>,
    options: SolverOptions,
) -> Result<TemperatureResult, PoissonError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let poisson_problem = PoissonProblem3D {
        conductivity: problem.thermal_conductivity,
        source: &problem.heat_generation,
        dirichlet: problem.prescribed_temperatures.clone(),
    };
    solve_poisson_3d(mesh, &poisson_problem, options).map(TemperatureResult::from)
}

pub fn solve_steady_heat_3d_with_solver<F>(
    mesh: &MeshTopology<3>,
    problem: &SteadyHeatProblem3D<F>,
    options: LinearSolverOptions,
) -> Result<TemperatureSolverResult, PoissonError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let poisson_problem = PoissonProblem3D {
        conductivity: problem.thermal_conductivity,
        source: &problem.heat_generation,
        dirichlet: problem.prescribed_temperatures.clone(),
    };
    solve_poisson_3d_with_solver(mesh, &poisson_problem, options).map(TemperatureSolverResult::from)
}

impl From<PoissonResult> for TemperatureResult {
    fn from(value: PoissonResult) -> Self {
        Self {
            temperatures: value.values,
            iterations: value.iterations,
            residual_norm: value.residual_norm,
        }
    }
}

impl From<PoissonSolverResult> for TemperatureSolverResult {
    fn from(value: PoissonSolverResult) -> Self {
        Self {
            temperatures: value.values,
            diagnostics: value.diagnostics,
        }
    }
}
