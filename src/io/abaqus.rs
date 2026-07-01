use std::{collections::BTreeMap, fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    annotation::{GeometryAnnotations, MaterialAssignment},
    io::FileIoError,
    mesh::{Cell, ElementKind, Mesh, Point2},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbaqusModel {
    pub nodes: Vec<[f64; 3]>,
    pub elements: Vec<AbaqusElement>,
    pub node_sets: BTreeMap<String, Vec<usize>>,
    pub element_sets: BTreeMap<String, Vec<usize>>,
    pub materials: BTreeMap<String, AbaqusMaterial>,
    pub boundaries: Vec<AbaqusBoundary>,
    pub steps: Vec<AbaqusStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbaqusElement {
    pub kind: String,
    pub nodes: Vec<usize>,
    pub element_set: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbaqusMaterial {
    pub young_modulus: Option<f64>,
    pub poisson_ratio: Option<f64>,
    pub density: Option<f64>,
    pub plastic_points: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbaqusBoundary {
    pub node: usize,
    pub component: usize,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbaqusStep {
    pub name: String,
    pub kind: String,
}

pub fn read_abaqus_file(path: impl AsRef<Path>) -> Result<AbaqusModel, FileIoError> {
    let path = path.as_ref();
    let input = fs::read_to_string(path).map_err(|source| FileIoError::Read {
        path: path.to_owned(),
        source,
    })?;
    parse_abaqus_str(&input)
}

pub fn parse_abaqus_str(input: &str) -> Result<AbaqusModel, FileIoError> {
    let mut nodes = Vec::new();
    let mut elements = Vec::new();
    let mut node_sets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut element_sets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut materials: BTreeMap<String, AbaqusMaterial> = BTreeMap::new();
    let mut boundaries = Vec::new();
    let mut steps = Vec::new();

    let mut current_keyword = String::new();
    let mut current_set_name: Option<String> = None;
    let mut current_material: Option<String> = None;
    let mut current_step: Option<String> = None;
    let mut pending_element_kind = String::from("UNKNOWN");
    let mut pending_element_set: Option<String> = None;

    for (line_number, raw_line) in input.lines().enumerate() {
        let line = raw_line.split('!').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("**") {
            continue;
        }
        if line.starts_with('*') {
            let keyword = line[1..]
                .split(',')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_uppercase();
            current_keyword = keyword.clone();
            current_set_name = parse_parameter(line, "NSET").or_else(|| parse_parameter(line, "ELSET"));
            pending_element_kind = parse_parameter(line, "TYPE").unwrap_or_else(|| "UNKNOWN".to_string());
            pending_element_set = parse_parameter(line, "ELSET");

            if keyword == "MATERIAL" {
                current_material = parse_parameter(line, "NAME");
                if let Some(name) = current_material.clone() {
                    materials.entry(name).or_default();
                }
            }
            if keyword == "STEP" {
                current_step = parse_parameter(line, "NAME");
                if let Some(name) = current_step.clone() {
                    steps.push(AbaqusStep {
                        name,
                        kind: "generic".to_string(),
                    });
                }
            }
            if keyword == "STATIC" {
                if let Some(name) = current_step.clone() {
                    if let Some(step) = steps.iter_mut().find(|step| step.name == name) {
                        step.kind = "static".to_string();
                    }
                }
            }
            if keyword == "DYNAMIC" {
                if let Some(name) = current_step.clone() {
                    if let Some(step) = steps.iter_mut().find(|step| step.name == name) {
                        step.kind = "dynamic".to_string();
                    }
                }
            }
            continue;
        }

        let tokens: Vec<&str> = line.split(',').map(str::trim).collect();
        match current_keyword.as_str() {
            "NODE" => {
                if tokens.len() < 4 {
                    return Err(FileIoError::InvalidLine {
                        line: line_number + 1,
                        expected: "node id, x, y, z",
                    });
                }
                let id = parse_usize(line_number + 1, tokens[0])?;
                let x = parse_f64(line_number + 1, tokens[1])?;
                let y = parse_f64(line_number + 1, tokens[2])?;
                let z = parse_f64(line_number + 1, tokens[3])?;
                ensure_contiguous_id(&nodes, id, line_number + 1)?;
                nodes.push([x, y, z]);
            }
            "ELEMENT" => {
                if tokens.is_empty() {
                    return Err(FileIoError::InvalidLine {
                        line: line_number + 1,
                        expected: "element id and node connectivity",
                    });
                }
                let node_ids = tokens[1..]
                    .iter()
                    .map(|token| parse_usize(line_number + 1, token))
                    .collect::<Result<Vec<_>, _>>()?;
                elements.push(AbaqusElement {
                    kind: pending_element_kind.clone(),
                    nodes: node_ids,
                    element_set: pending_element_set.clone(),
                });
            }
            "NSET" => {
                let set_name = current_set_name.clone().ok_or(FileIoError::InvalidLine {
                    line: line_number + 1,
                    expected: "NSET name on keyword line",
                })?;
                let entry = node_sets.entry(set_name).or_default();
                for token in tokens {
                    if let Some((start, end)) = parse_range(token) {
                        entry.extend(start..=end);
                    } else {
                        entry.push(parse_usize(line_number + 1, token)? - 1);
                    }
                }
            }
            "ELSET" => {
                let set_name = current_set_name.clone().ok_or(FileIoError::InvalidLine {
                    line: line_number + 1,
                    expected: "ELSET name on keyword line",
                })?;
                let entry = element_sets.entry(set_name).or_default();
                for token in tokens {
                    if let Some((start, end)) = parse_range(token) {
                        entry.extend(start..=end);
                    } else {
                        entry.push(parse_usize(line_number + 1, token)? - 1);
                    }
                }
            }
            "BOUNDARY" => {
                if tokens.len() < 3 {
                    return Err(FileIoError::InvalidLine {
                        line: line_number + 1,
                        expected: "node, first_dof, last_dof, value",
                    });
                }
                let node = parse_usize(line_number + 1, tokens[0])? - 1;
                let first_dof = parse_usize(line_number + 1, tokens[1])?;
                let last_dof = parse_usize(line_number + 1, tokens.get(2).copied().unwrap_or(tokens[1]))?;
                let value = tokens
                    .get(3)
                    .map(|token| parse_f64(line_number + 1, token))
                    .transpose()?
                    .unwrap_or(0.0);
                for component in first_dof..=last_dof {
                    boundaries.push(AbaqusBoundary {
                        node,
                        component,
                        value,
                    });
                }
            }
            "ELASTIC" => {
                if let Some(name) = current_material.clone() {
                    if tokens.len() >= 2 {
                        let young = parse_f64(line_number + 1, tokens[0])?;
                        let poisson = parse_f64(line_number + 1, tokens[1])?;
                        let entry = materials.entry(name.clone()).or_default();
                        entry.young_modulus = Some(young);
                        entry.poisson_ratio = Some(poisson);
                    }
                }
            }
            "PLASTIC" => {
                if let Some(name) = current_material.clone() {
                    if tokens.len() >= 2 {
                        let stress = parse_f64(line_number + 1, tokens[0])?;
                        let strain = parse_f64(line_number + 1, tokens[1])?;
                        materials
                            .entry(name)
                            .or_default()
                            .plastic_points
                            .push([stress, strain]);
                    }
                }
            }
            "DENSITY" => {
                if let Some(name) = current_material.clone() {
                    if let Some(token) = tokens.first() {
                        materials.entry(name).or_default().density =
                            Some(parse_f64(line_number + 1, token)?);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(AbaqusModel {
        nodes,
        elements,
        node_sets,
        element_sets,
        materials,
        boundaries,
        steps,
    })
}

impl Default for AbaqusMaterial {
    fn default() -> Self {
        Self {
            young_modulus: None,
            poisson_ratio: None,
            density: None,
            plastic_points: Vec::new(),
        }
    }
}

pub fn abaqus_to_mesh_2d(model: &AbaqusModel) -> Result<Mesh, FileIoError> {
    let points: Vec<Point2> = model
        .nodes
        .iter()
        .map(|node| Point2::new(node[0], node[1]))
        .collect();
    let mut cells = Vec::new();
    for element in &model.elements {
        let kind = map_abaqus_element_kind(&element.kind);
        cells.push(Cell::new(kind, element.nodes.iter().map(|id| id - 1).collect()));
    }
    Mesh::new_with_cells(points, cells).map_err(FileIoError::Mesh)
}

pub fn abaqus_to_annotations(model: &AbaqusModel) -> GeometryAnnotations {
    let mut annotations = GeometryAnnotations::default();
    for (index, (name, _material)) in model.materials.iter().enumerate() {
        annotations = annotations.with_material(MaterialAssignment::new(
            index,
            name.clone(),
            crate::annotation::EntitySelector::region_id(0),
            Vec::new(),
        ));
    }
    annotations
}

fn map_abaqus_element_kind(kind: &str) -> ElementKind {
    match kind.to_ascii_uppercase().as_str() {
        "CPS3" | "CPE3" | "S3" => ElementKind::Tri3,
        "CPS4" | "CPE4" | "S4" => ElementKind::Quad4,
        "C3D4" => ElementKind::Tet4,
        "C3D8" => ElementKind::Hex8,
        _ => ElementKind::Tri3,
    }
}

fn parse_parameter(line: &str, key: &str) -> Option<String> {
    for part in line.split(',') {
        let part = part.trim();
        if let Some((name, value)) = part.split_once('=') {
            if name.trim().eq_ignore_ascii_case(key) {
                return Some(value.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

fn parse_range(token: &str) -> Option<(usize, usize)> {
    let mut parts = token.split(':');
    let start: usize = parts.next()?.parse().ok()?;
    let end: usize = parts.next()?.parse().ok()?;
    Some((start.saturating_sub(1), end.saturating_sub(1)))
}

fn ensure_contiguous_id(nodes: &[[f64; 3]], id: usize, line: usize) -> Result<(), FileIoError> {
    let expected = nodes.len() + 1;
    if id != expected {
        return Err(FileIoError::NonContiguousId {
            expected,
            found: id,
        });
    }
    let _ = line;
    Ok(())
}

fn parse_usize(line: usize, value: &str) -> Result<usize, FileIoError> {
    crate::io::parse_usize(line, value)
}

fn parse_f64(line: usize, value: &str) -> Result<f64, FileIoError> {
    crate::io::parse_f64(line, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_inp_model() {
        let input = r#"
*NODE
1, 0., 0., 0.
2, 1., 0., 0.
3, 0., 1., 0.
*ELEMENT, TYPE=CPS3, ELSET=BLOCK
1, 1, 2, 3
*NSET, NSET=FIXED
1, 3
*MATERIAL, NAME=STEEL
*ELASTIC
210000., 0.3
*DENSITY
7.85e-9
*BOUNDARY
1, 1, 2
*STEP, NAME=LOAD
*STATIC
*END STEP
"#;
        let model = parse_abaqus_str(input).unwrap();
        assert_eq!(model.nodes.len(), 3);
        assert_eq!(model.elements.len(), 1);
        assert_eq!(model.node_sets["FIXED"], vec![0, 2]);
        assert_eq!(model.materials["STEEL"].young_modulus, Some(210000.0));
        assert_eq!(model.steps[0].kind, "static");

        let mesh = abaqus_to_mesh_2d(&model).unwrap();
        assert_eq!(mesh.node_count(), 3);
        assert_eq!(mesh.cells().len(), 1);
    }
}
