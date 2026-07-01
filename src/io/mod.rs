pub mod abaqus;
pub mod cad;
pub mod gmsh;
pub mod mesh;
pub mod params;
pub mod project;
pub mod result_format;
pub mod solution;
pub mod vtk;

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
    #[error("line {line}: unsupported solver backend '{backend}'")]
    UnsupportedSolverBackend { line: usize, backend: String },
    #[error("line {line}: unsupported preconditioner '{preconditioner}'")]
    UnsupportedPreconditioner { line: usize, preconditioner: String },
    #[error("line {line}: failed to parse boolean '{value}'; expected true or false")]
    ParseBool { line: usize, value: String },
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
    #[error("unsupported Gmsh format '{version}'; only ASCII 2.x files are supported")]
    UnsupportedGmshFormat { version: String },
    #[error("line {line}: unsupported Gmsh element type {element_type}")]
    UnsupportedGmshElement { line: usize, element_type: usize },
    #[error("line {line}: Gmsh element references node id {node_id}, but no such node exists")]
    UnknownGmshNode { line: usize, node_id: usize },
    #[error("Gmsh file must contain a $Nodes section")]
    MissingGmshNodes,
    #[error("Gmsh file must contain an $Elements section")]
    MissingGmshElements,
    #[error("VTK scalar field '{name}' has {actual} values, but topology has {expected} points")]
    VtkFieldLengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
}

pub(crate) fn parse_usize(line: usize, value: &str) -> Result<usize, FileIoError> {
    value.parse().map_err(|source| FileIoError::ParseInt {
        line,
        value: value.to_owned(),
        source,
    })
}

pub(crate) fn parse_f64(line: usize, value: &str) -> Result<f64, FileIoError> {
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
