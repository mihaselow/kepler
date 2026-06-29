use std::{collections::BTreeMap, fs, path::Path};

use crate::{
    io::{FileIoError, data_lines, parse_f64, parse_usize},
    mesh::{Mesh, Point2, Tri3},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Nodes,
    Triangles,
}

pub fn read_mesh_file(path: impl AsRef<Path>) -> Result<Mesh, FileIoError> {
    let path = path.as_ref();
    let input = fs::read_to_string(path).map_err(|source| FileIoError::Read {
        path: path.to_owned(),
        source,
    })?;
    parse_mesh_str(&input)
}

pub fn parse_mesh_str(input: &str) -> Result<Mesh, FileIoError> {
    let mut section = None;
    let mut nodes = BTreeMap::new();
    let mut triangles = BTreeMap::new();

    for (line, tokens) in data_lines(input) {
        match tokens.as_slice() {
            ["nodes"] => section = Some(Section::Nodes),
            ["triangles"] => section = Some(Section::Triangles),
            [section_name] if section.is_none() => {
                return Err(FileIoError::UnknownSection {
                    line,
                    section: (*section_name).to_owned(),
                });
            }
            _ => match section {
                Some(Section::Nodes) => {
                    let (id, point) = parse_node_line(line, &tokens)?;
                    if nodes.insert(id, point).is_some() {
                        return Err(FileIoError::DuplicateId { line, id });
                    }
                }
                Some(Section::Triangles) => {
                    let [id, a, b, c] = parse_triangle_line(line, &tokens)?;
                    if triangles.insert(id, Tri3::new([a, b, c])).is_some() {
                        return Err(FileIoError::DuplicateId { line, id });
                    }
                }
                None => {
                    return Err(FileIoError::InvalidLine {
                        line,
                        expected: "a section header",
                    });
                }
            },
        }
    }

    if nodes.is_empty() {
        return Err(FileIoError::MissingSection { section: "nodes" });
    }
    if triangles.is_empty() {
        return Err(FileIoError::MissingSection {
            section: "triangles",
        });
    }

    let points = into_contiguous_values(nodes)?;
    let triangles = into_contiguous_values(triangles)?;
    Ok(Mesh::new(points, triangles)?)
}

fn parse_node_line(line: usize, tokens: &[&str]) -> Result<(usize, Point2), FileIoError> {
    if tokens.len() != 3 {
        return Err(FileIoError::InvalidLine {
            line,
            expected: "node row: <id> <x> <y>",
        });
    }

    Ok((
        parse_usize(line, tokens[0])?,
        Point2::new(parse_f64(line, tokens[1])?, parse_f64(line, tokens[2])?),
    ))
}

fn parse_triangle_line(line: usize, tokens: &[&str]) -> Result<[usize; 4], FileIoError> {
    if tokens.len() != 4 {
        return Err(FileIoError::InvalidLine {
            line,
            expected: "triangle row: <id> <node_a> <node_b> <node_c>",
        });
    }

    Ok([
        parse_usize(line, tokens[0])?,
        parse_usize(line, tokens[1])?,
        parse_usize(line, tokens[2])?,
        parse_usize(line, tokens[3])?,
    ])
}

fn into_contiguous_values<T>(values_by_id: BTreeMap<usize, T>) -> Result<Vec<T>, FileIoError> {
    let mut values = Vec::with_capacity(values_by_id.len());
    for (expected, (found, value)) in values_by_id.into_iter().enumerate() {
        if expected != found {
            return Err(FileIoError::NonContiguousId { expected, found });
        }
        values.push(value);
    }
    Ok(values)
}
