pub mod mesh;
pub mod params;
pub mod solution;

use std::{io, num::ParseFloatError, num::ParseIntError, path::PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FileIoError {
    #[error("failed to read {path}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to write {path}")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("line {line}: expected {expected}")]
    InvalidLine { line: usize, expected: &'static str },
    #[error("line {line}: unknown section '{section}'")]
    UnknownSection { line: usize, section: String },
    #[error("missing required section '{section}'")]
    MissingSection { section: &'static str },
    #[error("line {line}: duplicate id {id}")]
    DuplicateId { line: usize, id: usize },
    #[error("ids must be contiguous and start at 0; expected {expected}, found {found}")]
    NonContiguousId { expected: usize, found: usize },
    #[error("line {line}: failed to parse integer '{value}'")]
    ParseInt {
        line: usize,
        value: String,
        #[source]
        source: ParseIntError,
    },
    #[error("line {line}: failed to parse float '{value}'")]
    ParseFloat {
        line: usize,
        value: String,
        #[source]
        source: ParseFloatError,
    },
    #[error("mesh validation failed")]
    Mesh(#[from] crate::mesh::MeshError),
    #[error("missing required parameter '{name}'")]
    MissingParameter { name: &'static str },
    #[error("line {line}: unsupported source kind '{kind}'")]
    UnsupportedSource { line: usize, kind: String },
    #[error("line {line}: unsupported solver option '{option}'")]
    UnsupportedSolverOption { line: usize, option: String },
    #[error("line {line}: duplicate parameter '{name}'")]
    DuplicateParameter { line: usize, name: &'static str },
    #[error("line {line}: duplicate Dirichlet node {node_id}")]
    DuplicateDirichlet { line: usize, node_id: usize },
    #[error("Dirichlet node {node_id} is out of bounds for mesh with {node_count} nodes")]
    BoundaryNodeOutOfBounds { node_id: usize, node_count: usize },
    #[error("conductivity must be positive and finite, got {value}")]
    InvalidConductivity { value: f64 },
    #[error("source value must be finite, got {value}")]
    InvalidSource { value: f64 },
    #[error("solver max_iterations must be greater than zero")]
    InvalidMaxIterations,
    #[error("solver tolerance must be positive and finite, got {value}")]
    InvalidTolerance { value: f64 },
}

fn parse_usize(line: usize, value: &str) -> Result<usize, FileIoError> {
    value.parse().map_err(|source| FileIoError::ParseInt {
        line,
        value: value.to_owned(),
        source,
    })
}

fn parse_f64(line: usize, value: &str) -> Result<f64, FileIoError> {
    value.parse().map_err(|source| FileIoError::ParseFloat {
        line,
        value: value.to_owned(),
        source,
    })
}

fn data_lines(input: &str) -> impl Iterator<Item = (usize, Vec<&str>)> {
    input.lines().enumerate().filter_map(|(index, raw_line)| {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            None
        } else {
            Some((index + 1, line.split_whitespace().collect()))
        }
    })
}
