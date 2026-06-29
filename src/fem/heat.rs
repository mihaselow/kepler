use std::collections::{BTreeMap, BTreeSet};

use sprs::{CsMat, TriMat};
use thiserror::Error;

use crate::{
    fem::poisson::{
        PoissonError, PoissonProblem, PoissonProblem3D, PoissonResult, PoissonSolverResult,
        local_load, local_stiffness, solve_poisson, solve_poisson_3d, solve_poisson_3d_with_solver,
        solve_poisson_with_solver,
    },
    linalg::{
        LinalgError, LinearSolverOptions, SolverDiagnostics, SolverOptions, TransientSolverOptions,
        solve_linear_transient,
    },
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

pub struct TransientHeatProblem<F> {
    pub thermal_conductivity: f64,
    pub volumetric_heat_capacity: f64,
    pub heat_generation: F,
    pub initial_temperatures: Vec<f64>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct TransientTemperatureStep {
    pub time: f64,
    pub temperatures: Vec<f64>,
    pub diagnostics: SolverDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientTemperatureResult {
    pub steps: Vec<TransientTemperatureStep>,
}

#[derive(Debug, Error, PartialEq)]
pub enum TransientHeatError {
    #[error("volumetric heat capacity must be positive and finite, got {0}")]
    InvalidVolumetricHeatCapacity(f64),
    #[error("initial temperature length {initial_len} does not match mesh node count {node_count}")]
    InitialTemperatureLengthMismatch {
        node_count: usize,
        initial_len: usize,
    },
    #[error("prescribed temperature references node {node_id}, but mesh has {node_count} nodes")]
    BoundaryNodeOutOfBounds { node_id: NodeId, node_count: usize },
    #[error("prescribed temperature node {node_id} was specified more than once")]
    DuplicateBoundaryNode { node_id: NodeId },
    #[error("all temperature degrees of freedom are prescribed")]
    NoActiveDegreesOfFreedom,
    #[error("heat assembly failed")]
    Poisson(#[from] PoissonError),
    #[error("linear transient solver failed")]
    LinearSolve(#[from] LinalgError),
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

pub fn solve_transient_heat<F>(
    mesh: &Mesh,
    problem: &TransientHeatProblem<F>,
    options: TransientSolverOptions,
) -> Result<TransientTemperatureResult, TransientHeatError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    validate_transient_heat_problem(mesh, problem)?;
    let prescribed =
        validate_prescribed_temperatures(mesh.node_count(), &problem.prescribed_temperatures)?;
    let active_nodes = active_nodes(mesh.node_count(), &prescribed);
    if active_nodes.is_empty() {
        return Err(TransientHeatError::NoActiveDegreesOfFreedom);
    }
    let active_map = active_node_map(mesh.node_count(), &active_nodes);
    let stiffness = assemble_heat_stiffness(mesh, problem.thermal_conductivity)?;
    let mass = assemble_lumped_heat_mass(mesh, problem.volumetric_heat_capacity);
    let reduced_stiffness = reduce_matrix(&stiffness, &active_nodes, &active_map);
    let reduced_mass = reduce_matrix(&mass, &active_nodes, &active_map);
    let initial_values = active_nodes
        .iter()
        .map(|&node| problem.initial_temperatures[node])
        .collect();
    let source_values = transient_sources(
        mesh,
        problem,
        &stiffness,
        &active_nodes,
        &prescribed,
        &options,
    )?;
    let time_step = options.time_step;

    let reduced_steps = solve_linear_transient(
        &reduced_mass,
        &reduced_stiffness,
        initial_values,
        move |time| {
            let index = (time / time_step).round() as usize;
            source_values[index].clone()
        },
        options,
    )?;

    let steps = reduced_steps
        .into_iter()
        .map(|step| TransientTemperatureStep {
            time: step.time,
            temperatures: reconstruct_temperatures(
                mesh.node_count(),
                &active_nodes,
                &prescribed,
                &step.values,
            ),
            diagnostics: step.linear_diagnostics,
        })
        .collect();

    Ok(TransientTemperatureResult { steps })
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

fn validate_transient_heat_problem<F>(
    mesh: &Mesh,
    problem: &TransientHeatProblem<F>,
) -> Result<(), TransientHeatError> {
    if !problem.volumetric_heat_capacity.is_finite() || problem.volumetric_heat_capacity <= 0.0 {
        return Err(TransientHeatError::InvalidVolumetricHeatCapacity(
            problem.volumetric_heat_capacity,
        ));
    }
    if problem.initial_temperatures.len() != mesh.node_count() {
        return Err(TransientHeatError::InitialTemperatureLengthMismatch {
            node_count: mesh.node_count(),
            initial_len: problem.initial_temperatures.len(),
        });
    }
    Ok(())
}

fn validate_prescribed_temperatures(
    node_count: usize,
    entries: &[(NodeId, f64)],
) -> Result<BTreeMap<NodeId, f64>, TransientHeatError> {
    let mut prescribed = BTreeMap::new();
    for &(node_id, value) in entries {
        if node_id >= node_count {
            return Err(TransientHeatError::BoundaryNodeOutOfBounds {
                node_id,
                node_count,
            });
        }
        if prescribed.insert(node_id, value).is_some() {
            return Err(TransientHeatError::DuplicateBoundaryNode { node_id });
        }
    }
    Ok(prescribed)
}

fn active_nodes(node_count: usize, prescribed: &BTreeMap<NodeId, f64>) -> Vec<NodeId> {
    let prescribed_nodes: BTreeSet<_> = prescribed.keys().copied().collect();
    (0..node_count)
        .filter(|node| !prescribed_nodes.contains(node))
        .collect()
}

fn active_node_map(node_count: usize, active_nodes: &[NodeId]) -> Vec<Option<usize>> {
    let mut map = vec![None; node_count];
    for (active_index, &node) in active_nodes.iter().enumerate() {
        map[node] = Some(active_index);
    }
    map
}

fn assemble_heat_stiffness(mesh: &Mesh, conductivity: f64) -> Result<CsMat<f64>, PoissonError> {
    let mut triplets = TriMat::with_capacity(
        (mesh.node_count(), mesh.node_count()),
        mesh.triangles().len() * 9,
    );
    for triangle in mesh.triangles() {
        let stiffness = local_stiffness(mesh, triangle, conductivity)?;
        for (local_row, global_row) in triangle.nodes.iter().copied().enumerate() {
            for (local_col, global_col) in triangle.nodes.iter().copied().enumerate() {
                triplets.add_triplet(global_row, global_col, stiffness[local_row][local_col]);
            }
        }
    }
    Ok(triplets.to_csr())
}

fn assemble_lumped_heat_mass(mesh: &Mesh, volumetric_heat_capacity: f64) -> CsMat<f64> {
    let mut nodal_mass = vec![0.0; mesh.node_count()];
    for triangle in mesh.triangles() {
        let [a, b, c] = triangle.nodes.map(|node| mesh.points()[node]);
        let twice_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        let contribution = volumetric_heat_capacity * twice_area.abs() / 6.0;
        for node in triangle.nodes {
            nodal_mass[node] += contribution;
        }
    }
    let mut triplets =
        TriMat::with_capacity((mesh.node_count(), mesh.node_count()), mesh.node_count());
    for (node, mass) in nodal_mass.into_iter().enumerate() {
        triplets.add_triplet(node, node, mass);
    }
    triplets.to_csr()
}

fn reduce_matrix(
    matrix: &CsMat<f64>,
    active_nodes: &[NodeId],
    active_map: &[Option<usize>],
) -> CsMat<f64> {
    let mut triplets = TriMat::new((active_nodes.len(), active_nodes.len()));
    for (row_index, row) in matrix.outer_iterator().enumerate() {
        let Some(active_row) = active_map[row_index] else {
            continue;
        };
        for (col_index, value) in row.iter() {
            if let Some(active_col) = active_map[col_index] {
                triplets.add_triplet(active_row, active_col, *value);
            }
        }
    }
    triplets.to_csr()
}

fn transient_sources<F>(
    mesh: &Mesh,
    problem: &TransientHeatProblem<F>,
    stiffness: &CsMat<f64>,
    active_nodes: &[NodeId],
    prescribed: &BTreeMap<NodeId, f64>,
    options: &TransientSolverOptions,
) -> Result<Vec<Vec<f64>>, TransientHeatError>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let mut sources = Vec::with_capacity(options.steps + 1);
    for step in 0..=options.steps {
        let time = step as f64 * options.time_step;
        let full_load = assemble_heat_load(mesh, |x, y| (problem.heat_generation)(x, y, time))?;
        let mut reduced = Vec::with_capacity(active_nodes.len());
        for &node in active_nodes {
            let mut value = full_load[node];
            if let Some(row) = stiffness.outer_view(node) {
                for (col_index, stiffness_value) in row.iter() {
                    if let Some(boundary_value) = prescribed.get(&col_index) {
                        value -= stiffness_value * boundary_value;
                    }
                }
            }
            reduced.push(value);
        }
        sources.push(reduced);
    }
    Ok(sources)
}

fn assemble_heat_load<F>(mesh: &Mesh, source: F) -> Result<Vec<f64>, PoissonError>
where
    F: Fn(f64, f64) -> f64,
{
    let mut rhs = vec![0.0; mesh.node_count()];
    for triangle in mesh.triangles() {
        let load = local_load(mesh, triangle, &source)?;
        for (local_row, global_row) in triangle.nodes.iter().copied().enumerate() {
            rhs[global_row] += load[local_row];
        }
    }
    Ok(rhs)
}

fn reconstruct_temperatures(
    node_count: usize,
    active_nodes: &[NodeId],
    prescribed: &BTreeMap<NodeId, f64>,
    active_values: &[f64],
) -> Vec<f64> {
    let mut temperatures = vec![0.0; node_count];
    for (&node, &value) in prescribed {
        temperatures[node] = value;
    }
    for (&node, &value) in active_nodes.iter().zip(active_values) {
        temperatures[node] = value;
    }
    temperatures
}
