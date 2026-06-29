use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use crate::{
    io::{FileIoError, parse_f64, parse_usize},
    mesh::{Cell, ElementKind, EntityDimension, MeshTopology, PointD, Region},
};

const Z_TOLERANCE: f64 = 1.0e-12;
type PhysicalKey = (usize, usize);
type PhysicalNames = BTreeMap<PhysicalKey, PhysicalName>;

#[derive(Debug, Clone, PartialEq)]
pub enum ImportedMesh {
    TwoD(MeshTopology<2>),
    ThreeD(MeshTopology<3>),
}

impl ImportedMesh {
    pub const fn dimension(&self) -> usize {
        match self {
            Self::TwoD(_) => 2,
            Self::ThreeD(_) => 3,
        }
    }

    pub fn region_count(&self) -> usize {
        match self {
            Self::TwoD(topology) => topology.regions().len(),
            Self::ThreeD(topology) => topology.regions().len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PhysicalName {
    dimension: EntityDimension,
    name: String,
}

#[derive(Debug, Clone, PartialEq)]
struct ElementRecord {
    kind: ElementKind,
    physical_tag: Option<usize>,
    nodes: Vec<usize>,
}

pub fn read_gmsh_file(path: impl AsRef<Path>) -> Result<ImportedMesh, FileIoError> {
    let path = path.as_ref();
    let input = fs::read_to_string(path).map_err(|source| FileIoError::Read {
        path: path.to_owned(),
        source,
    })?;
    parse_gmsh_str(&input)
}

pub fn parse_gmsh_str(input: &str) -> Result<ImportedMesh, FileIoError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut physical_names = BTreeMap::new();
    let mut nodes = BTreeMap::new();
    let mut elements = Vec::new();
    let mut saw_nodes = false;
    let mut saw_elements = false;

    let mut index = 0;
    while index < lines.len() {
        let line = lines[index].trim();
        match line {
            "$MeshFormat" => {
                index = parse_mesh_format(&lines, index + 1)?;
            }
            "$PhysicalNames" => {
                let (next, parsed) = parse_physical_names(&lines, index + 1)?;
                physical_names = parsed;
                index = next;
            }
            "$Nodes" => {
                let (next, parsed) = parse_nodes(&lines, index + 1)?;
                nodes = parsed;
                saw_nodes = true;
                index = next;
            }
            "$Elements" => {
                let (next, parsed) = parse_elements(&lines, index + 1)?;
                elements = parsed;
                saw_elements = true;
                index = next;
            }
            _ => index += 1,
        }
    }

    if !saw_nodes {
        return Err(FileIoError::MissingGmshNodes);
    }
    if !saw_elements {
        return Err(FileIoError::MissingGmshElements);
    }

    build_imported_mesh(nodes, elements, physical_names)
}

fn parse_mesh_format(lines: &[&str], index: usize) -> Result<usize, FileIoError> {
    let tokens: Vec<&str> = lines
        .get(index)
        .ok_or(FileIoError::InvalidLine {
            line: index + 1,
            expected: "Gmsh mesh format row",
        })?
        .split_whitespace()
        .collect();
    if tokens.len() < 2 {
        return Err(FileIoError::InvalidLine {
            line: index + 1,
            expected: "<version> <file_type> <data_size>",
        });
    }

    let version = tokens[0];
    let file_type = tokens[1];
    if !version.starts_with("2.") || file_type != "0" {
        return Err(FileIoError::UnsupportedGmshFormat {
            version: version.to_owned(),
        });
    }

    Ok(skip_to_end(lines, index + 1, "$EndMeshFormat"))
}

fn parse_physical_names(
    lines: &[&str],
    index: usize,
) -> Result<(usize, PhysicalNames), FileIoError> {
    let count = parse_count(lines, index, "physical name count")?;
    let mut names = BTreeMap::new();

    for offset in 0..count {
        let line_index = index + 1 + offset;
        let raw_line = lines.get(line_index).ok_or(FileIoError::InvalidLine {
            line: line_index + 1,
            expected: "physical name row",
        })?;
        let tokens: Vec<&str> = raw_line.split_whitespace().collect();
        if tokens.len() < 3 {
            return Err(FileIoError::InvalidLine {
                line: line_index + 1,
                expected: "<dimension> <tag> <name>",
            });
        }

        let dimension_value = parse_usize(line_index + 1, tokens[0])?;
        let physical_tag = parse_usize(line_index + 1, tokens[1])?;
        let dimension = entity_dimension_from_gmsh(dimension_value, line_index + 1)?;
        let name = parse_physical_name(raw_line, line_index + 1)?;

        names.insert(
            (dimension_value, physical_tag),
            PhysicalName { dimension, name },
        );
    }

    Ok((
        skip_to_end(lines, index + 1 + count, "$EndPhysicalNames"),
        names,
    ))
}

fn parse_nodes(
    lines: &[&str],
    index: usize,
) -> Result<(usize, BTreeMap<usize, [f64; 3]>), FileIoError> {
    let count = parse_count(lines, index, "node count")?;
    let mut nodes = BTreeMap::new();

    for offset in 0..count {
        let line_index = index + 1 + offset;
        let tokens: Vec<&str> = lines
            .get(line_index)
            .ok_or(FileIoError::InvalidLine {
                line: line_index + 1,
                expected: "node row",
            })?
            .split_whitespace()
            .collect();
        if tokens.len() != 4 {
            return Err(FileIoError::InvalidLine {
                line: line_index + 1,
                expected: "<node_id> <x> <y> <z>",
            });
        }

        let node_id = parse_usize(line_index + 1, tokens[0])?;
        let coords = [
            parse_f64(line_index + 1, tokens[1])?,
            parse_f64(line_index + 1, tokens[2])?,
            parse_f64(line_index + 1, tokens[3])?,
        ];
        if nodes.insert(node_id, coords).is_some() {
            return Err(FileIoError::DuplicateId {
                line: line_index + 1,
                id: node_id,
            });
        }
    }

    Ok((skip_to_end(lines, index + 1 + count, "$EndNodes"), nodes))
}

fn parse_elements(
    lines: &[&str],
    index: usize,
) -> Result<(usize, Vec<ElementRecord>), FileIoError> {
    let count = parse_count(lines, index, "element count")?;
    let mut elements = Vec::with_capacity(count);

    for offset in 0..count {
        let line_index = index + 1 + offset;
        let tokens: Vec<&str> = lines
            .get(line_index)
            .ok_or(FileIoError::InvalidLine {
                line: line_index + 1,
                expected: "element row",
            })?
            .split_whitespace()
            .collect();
        if tokens.len() < 4 {
            return Err(FileIoError::InvalidLine {
                line: line_index + 1,
                expected: "<id> <type> <tag_count> <tags...> <nodes...>",
            });
        }

        let element_type = parse_usize(line_index + 1, tokens[1])?;
        let kind =
            element_kind_from_gmsh(element_type).ok_or(FileIoError::UnsupportedGmshElement {
                line: line_index + 1,
                element_type,
            })?;
        let tag_count = parse_usize(line_index + 1, tokens[2])?;
        let node_start = 3 + tag_count;
        if tokens.len() != node_start + kind.node_count() {
            return Err(FileIoError::InvalidLine {
                line: line_index + 1,
                expected: "a supported Gmsh element with the expected number of nodes",
            });
        }

        let physical_tag = if tag_count > 0 {
            Some(parse_usize(line_index + 1, tokens[3])?)
        } else {
            None
        };
        let mut nodes = Vec::with_capacity(kind.node_count());
        for token in &tokens[node_start..] {
            nodes.push(parse_usize(line_index + 1, token)?);
        }
        elements.push(ElementRecord {
            kind,
            physical_tag,
            nodes,
        });
    }

    Ok((
        skip_to_end(lines, index + 1 + count, "$EndElements"),
        elements,
    ))
}

fn build_imported_mesh(
    nodes: BTreeMap<usize, [f64; 3]>,
    elements: Vec<ElementRecord>,
    physical_names: PhysicalNames,
) -> Result<ImportedMesh, FileIoError> {
    let node_map: HashMap<usize, usize> = nodes
        .keys()
        .enumerate()
        .map(|(internal_id, gmsh_id)| (*gmsh_id, internal_id))
        .collect();
    let is_2d = nodes.values().all(|coords| coords[2].abs() <= Z_TOLERANCE)
        && elements
            .iter()
            .all(|element| element.kind.entity_dimension().spatial_dimension() <= 2);
    let mut region_ids = BTreeMap::new();
    let mut regions = Vec::new();
    let mut next_region_id = 0;
    let mut cells = Vec::with_capacity(elements.len());

    for element in elements {
        let region = element.physical_tag.map(|physical_tag| {
            let dimension = element.kind.entity_dimension().spatial_dimension();
            let key = (dimension, physical_tag);
            if let Some(region_id) = region_ids.get(&key) {
                return *region_id;
            }

            let physical_name = physical_names.get(&key);
            let region_id = next_region_id;
            next_region_id += 1;
            region_ids.insert(key, region_id);
            regions.push(Region::new(
                region_id,
                physical_name
                    .map(|physical_name| physical_name.name.clone())
                    .unwrap_or_else(|| format!("physical_{dimension}_{physical_tag}")),
                physical_name
                    .map(|physical_name| physical_name.dimension)
                    .unwrap_or_else(|| element.kind.entity_dimension()),
            ));
            region_id
        });

        let mut remapped_nodes = Vec::with_capacity(element.nodes.len());
        for gmsh_node_id in element.nodes {
            remapped_nodes.push(*node_map.get(&gmsh_node_id).ok_or(
                FileIoError::UnknownGmshNode {
                    line: 0,
                    node_id: gmsh_node_id,
                },
            )?);
        }

        let mut cell = Cell::new(element.kind, remapped_nodes);
        if let Some(region_id) = region {
            cell = cell.with_region(region_id);
        }
        cells.push(cell);
    }

    if is_2d {
        Ok(ImportedMesh::TwoD(MeshTopology::with_regions(
            nodes
                .values()
                .map(|coords| PointD::new([coords[0], coords[1]]))
                .collect(),
            cells,
            regions,
        )?))
    } else {
        Ok(ImportedMesh::ThreeD(MeshTopology::with_regions(
            nodes.values().map(|coords| PointD::new(*coords)).collect(),
            cells,
            regions,
        )?))
    }
}

fn parse_count(lines: &[&str], index: usize, expected: &'static str) -> Result<usize, FileIoError> {
    let line = lines.get(index).ok_or(FileIoError::InvalidLine {
        line: index + 1,
        expected,
    })?;
    parse_usize(index + 1, line.trim())
}

fn skip_to_end(lines: &[&str], mut index: usize, end_marker: &str) -> usize {
    while index < lines.len() {
        let line = lines[index].trim();
        index += 1;
        if line == end_marker {
            break;
        }
    }
    index
}

fn entity_dimension_from_gmsh(
    dimension: usize,
    line: usize,
) -> Result<EntityDimension, FileIoError> {
    match dimension {
        0 => Ok(EntityDimension::Point),
        1 => Ok(EntityDimension::Curve),
        2 => Ok(EntityDimension::Surface),
        3 => Ok(EntityDimension::Volume),
        _ => Err(FileIoError::InvalidLine {
            line,
            expected: "physical dimension 0, 1, 2, or 3",
        }),
    }
}

fn element_kind_from_gmsh(element_type: usize) -> Option<ElementKind> {
    match element_type {
        1 => Some(ElementKind::Line2),
        2 => Some(ElementKind::Tri3),
        3 => Some(ElementKind::Quad4),
        4 => Some(ElementKind::Tet4),
        5 => Some(ElementKind::Hex8),
        _ => None,
    }
}

fn parse_physical_name(raw_line: &str, line: usize) -> Result<String, FileIoError> {
    let Some(start) = raw_line.find('"') else {
        return Err(FileIoError::InvalidLine {
            line,
            expected: "quoted physical name",
        });
    };
    let Some(end) = raw_line.rfind('"') else {
        return Err(FileIoError::InvalidLine {
            line,
            expected: "quoted physical name",
        });
    };
    if start == end {
        return Err(FileIoError::InvalidLine {
            line,
            expected: "quoted physical name",
        });
    }
    Ok(raw_line[start + 1..end].to_owned())
}
