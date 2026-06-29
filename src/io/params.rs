use std::{collections::BTreeMap, fs, path::Path};

use crate::{
    io::{FileIoError, data_lines, parse_f64, parse_usize},
    linalg::SolverOptions,
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
    pub solver_options: SolverOptions,
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
    let mut max_iterations = SolverOptions::default().max_iterations;
    let mut tolerance = SolverOptions::default().tolerance;
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
                max_iterations = parse_usize(line, value)?;
                if max_iterations == 0 {
                    return Err(FileIoError::InvalidMaxIterations);
                }
                section = None;
            }
            ["solver", "tolerance", value] => {
                tolerance = parse_f64(line, value)?;
                if !tolerance.is_finite() || tolerance <= 0.0 {
                    return Err(FileIoError::InvalidTolerance { value: tolerance });
                }
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
        solver_options: SolverOptions {
            max_iterations,
            tolerance,
        },
    })
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
