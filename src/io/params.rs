use std::{collections::BTreeMap, fs, path::Path};

use crate::{
    io::{FileIoError, data_lines, parse_f64, parse_usize},
    linalg::{LinearSolverBackend, LinearSolverOptions, PreconditionerKind, SolverOptions},
    mesh::NodeId,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceConfig {
    Constant(f64),
}

impl SourceConfig {
    pub fn value_at(self, _x: f64, _y: f64) -> f64 {
        match self {
            Self::Constant(value) => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoissonFileConfig {
    pub conductivity: f64,
    pub source: SourceConfig,
    pub dirichlet: Vec<(NodeId, f64)>,
    pub solver_options: LinearSolverOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Dirichlet,
}

pub fn read_params_file(path: impl AsRef<Path>) -> Result<PoissonFileConfig, FileIoError> {
    let path = path.as_ref();
    let input = fs::read_to_string(path).map_err(|source| FileIoError::Read {
        path: path.to_owned(),
        source,
    })?;
    parse_params_str(&input)
}

pub fn parse_params_str(input: &str) -> Result<PoissonFileConfig, FileIoError> {
    let mut conductivity = None;
    let mut source = None;
    let mut solver_options = LinearSolverOptions::default();
    let mut dirichlet = BTreeMap::new();
    let mut section = None;

    for (line, tokens) in data_lines(input) {
        match tokens.as_slice() {
            ["dirichlet"] => section = Some(Section::Dirichlet),
            ["conductivity", value] => {
                reject_duplicate(line, "conductivity", conductivity.is_some())?;
                let value = parse_f64(line, value)?;
                if !value.is_finite() || value <= 0.0 {
                    return Err(FileIoError::InvalidConductivity { value });
                }
                conductivity = Some(value);
                section = None;
            }
            ["source", "constant", value] => {
                reject_duplicate(line, "source", source.is_some())?;
                let value = parse_f64(line, value)?;
                if !value.is_finite() {
                    return Err(FileIoError::InvalidSource { value });
                }
                source = Some(SourceConfig::Constant(value));
                section = None;
            }
            ["source", kind, ..] => {
                return Err(FileIoError::UnsupportedSource {
                    line,
                    kind: (*kind).to_owned(),
                });
            }
            ["solver", "max_iterations", value] => {
                solver_options.max_iterations = parse_usize(line, value)?;
                if solver_options.max_iterations == 0 {
                    return Err(FileIoError::InvalidMaxIterations);
                }
                section = None;
            }
            ["solver", "tolerance", value] => {
                solver_options.tolerance = parse_f64(line, value)?;
                if !solver_options.tolerance.is_finite() || solver_options.tolerance <= 0.0 {
                    return Err(FileIoError::InvalidTolerance {
                        value: solver_options.tolerance,
                    });
                }
                section = None;
            }
            ["solver", "backend", value] => {
                solver_options.backend = parse_backend(line, value)?;
                section = None;
            }
            ["solver", "preconditioner", value] => {
                solver_options.preconditioner = parse_preconditioner(line, value)?;
                section = None;
            }
            ["solver", "record_residual_history", value] => {
                solver_options.record_residual_history = parse_bool(line, value)?;
                section = None;
            }
            ["solver", option, ..] => {
                return Err(FileIoError::UnsupportedSolverOption {
                    line,
                    option: (*option).to_owned(),
                });
            }
            _ => match section {
                Some(Section::Dirichlet) => {
                    let (node_id, value) = parse_dirichlet_line(line, &tokens)?;
                    if dirichlet.insert(node_id, value).is_some() {
                        return Err(FileIoError::DuplicateDirichlet { line, node_id });
                    }
                }
                None => {
                    return Err(FileIoError::InvalidLine {
                        line,
                        expected: "a parameter row or dirichlet section",
                    });
                }
            },
        }
    }

    let conductivity = conductivity.ok_or(FileIoError::MissingParameter {
        name: "conductivity",
    })?;
    let source = source.ok_or(FileIoError::MissingParameter { name: "source" })?;

    Ok(PoissonFileConfig {
        conductivity,
        source,
        dirichlet: dirichlet.into_iter().collect(),
        solver_options,
    })
}

impl PoissonFileConfig {
    pub fn compatibility_solver_options(&self) -> SolverOptions {
        SolverOptions {
            max_iterations: self.solver_options.max_iterations,
            tolerance: self.solver_options.tolerance,
        }
    }
}

pub fn validate_params_for_mesh(
    config: &PoissonFileConfig,
    node_count: usize,
) -> Result<(), FileIoError> {
    for &(node_id, _) in &config.dirichlet {
        if node_id >= node_count {
            return Err(FileIoError::BoundaryNodeOutOfBounds {
                node_id,
                node_count,
            });
        }
    }
    Ok(())
}

fn reject_duplicate(
    line: usize,
    name: &'static str,
    is_duplicate: bool,
) -> Result<(), FileIoError> {
    if is_duplicate {
        Err(FileIoError::DuplicateParameter { line, name })
    } else {
        Ok(())
    }
}

fn parse_dirichlet_line(line: usize, tokens: &[&str]) -> Result<(NodeId, f64), FileIoError> {
    if tokens.len() != 2 {
        return Err(FileIoError::InvalidLine {
            line,
            expected: "dirichlet row: <node_id> <value>",
        });
    }

    Ok((parse_usize(line, tokens[0])?, parse_f64(line, tokens[1])?))
}

fn parse_backend(line: usize, value: &str) -> Result<LinearSolverBackend, FileIoError> {
    match value {
        "cg" | "conjugate_gradient" => Ok(LinearSolverBackend::ConjugateGradient),
        "gmres" => Ok(LinearSolverBackend::Gmres),
        "dense_direct" => Ok(LinearSolverBackend::DenseDirect),
        _ => Err(FileIoError::UnsupportedSolverBackend {
            line,
            backend: value.to_owned(),
        }),
    }
}

fn parse_preconditioner(line: usize, value: &str) -> Result<PreconditionerKind, FileIoError> {
    match value {
        "none" => Ok(PreconditionerKind::None),
        "jacobi" => Ok(PreconditionerKind::Jacobi),
        _ => Err(FileIoError::UnsupportedPreconditioner {
            line,
            preconditioner: value.to_owned(),
        }),
    }
}

fn parse_bool(line: usize, value: &str) -> Result<bool, FileIoError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(FileIoError::ParseBool {
            line,
            value: value.to_owned(),
        }),
    }
}
