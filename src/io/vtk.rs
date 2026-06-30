use std::{fmt::Write as _, fs, path::Path};

use crate::{
    io::FileIoError,
    mesh::{ElementKind, MeshTopology, PointD},
};

#[derive(Debug, Clone, PartialEq)]
pub struct VtkScalarField {
    pub name: String,
    pub values: Vec<f64>,
}

impl VtkScalarField {
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }
}

/// A 3-component vector field attached to VTK points.
#[derive(Debug, Clone, PartialEq)]
pub struct VtkVectorField {
    pub name: String,
    /// One [fx, fy, fz] per node.
    pub values: Vec<[f64; 3]>,
}

impl VtkVectorField {
    pub fn new(name: impl Into<String>, values: Vec<[f64; 3]>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }
}

pub fn write_vtk_legacy_file<const D: usize>(
    path: impl AsRef<Path>,
    topology: &MeshTopology<D>,
    scalar_fields: &[VtkScalarField],
) -> Result<(), FileIoError> {
    let path = path.as_ref();
    let output = format_vtk_legacy(topology, scalar_fields)?;
    fs::write(path, output).map_err(|source| FileIoError::Write {
        path: path.to_owned(),
        source,
    })
}

pub fn format_vtk_legacy<const D: usize>(
    topology: &MeshTopology<D>,
    scalar_fields: &[VtkScalarField],
) -> Result<String, FileIoError> {
    for field in scalar_fields {
        if field.values.len() != topology.points().len() {
            return Err(FileIoError::VtkFieldLengthMismatch {
                name: field.name.clone(),
                expected: topology.points().len(),
                actual: field.values.len(),
            });
        }
    }

    let mut output = String::new();
    writeln!(&mut output, "# vtk DataFile Version 3.0").expect("writing to string cannot fail");
    writeln!(&mut output, "kepler mesh").expect("writing to string cannot fail");
    writeln!(&mut output, "ASCII").expect("writing to string cannot fail");
    writeln!(&mut output, "DATASET UNSTRUCTURED_GRID").expect("writing to string cannot fail");
    writeln!(&mut output, "POINTS {} double", topology.points().len())
        .expect("writing to string cannot fail");
    for point in topology.points() {
        let [x, y, z] = point_to_vtk_coords(point);
        writeln!(&mut output, "{x} {y} {z}").expect("writing to string cannot fail");
    }

    let cell_size: usize = topology
        .cells()
        .iter()
        .map(|cell| 1 + cell.nodes.len())
        .sum();
    writeln!(
        &mut output,
        "CELLS {} {}",
        topology.cells().len(),
        cell_size
    )
    .expect("writing to string cannot fail");
    for cell in topology.cells() {
        write!(&mut output, "{}", cell.nodes.len()).expect("writing to string cannot fail");
        for node in &cell.nodes {
            write!(&mut output, " {node}").expect("writing to string cannot fail");
        }
        writeln!(&mut output).expect("writing to string cannot fail");
    }

    writeln!(&mut output, "CELL_TYPES {}", topology.cells().len())
        .expect("writing to string cannot fail");
    for cell in topology.cells() {
        writeln!(&mut output, "{}", vtk_cell_type(cell.kind))
            .expect("writing to string cannot fail");
    }

    if !scalar_fields.is_empty() {
        writeln!(&mut output, "POINT_DATA {}", topology.points().len())
            .expect("writing to string cannot fail");
        for field in scalar_fields {
            writeln!(
                &mut output,
                "SCALARS {} double 1",
                sanitize_vtk_name(&field.name)
            )
            .expect("writing to string cannot fail");
            writeln!(&mut output, "LOOKUP_TABLE default").expect("writing to string cannot fail");
            for value in &field.values {
                writeln!(&mut output, "{value}").expect("writing to string cannot fail");
            }
        }
    }

    Ok(output)
}

/// Writes a VTK Legacy ASCII file with displacement as a VECTORS field and
/// von Mises stress (plus any extra scalar fields) as SCALARS point data.
///
/// `displacements_3d` must have one `[fx, fy, fz]` entry per node (pad the
/// third component with 0.0 for 2-D problems).  `von_mises` must have one
/// value per node.  Extra scalar fields are appended after von Mises.
pub fn format_vtk_with_stress<const D: usize>(
    topology: &MeshTopology<D>,
    displacement_field: &VtkVectorField,
    von_mises: &VtkScalarField,
    extra_scalar_fields: &[VtkScalarField],
) -> Result<String, FileIoError> {
    let n_pts = topology.points().len();

    // Validate lengths.
    if displacement_field.values.len() != n_pts {
        return Err(FileIoError::VtkFieldLengthMismatch {
            name: displacement_field.name.clone(),
            expected: n_pts,
            actual: displacement_field.values.len(),
        });
    }
    if von_mises.values.len() != n_pts {
        return Err(FileIoError::VtkFieldLengthMismatch {
            name: von_mises.name.clone(),
            expected: n_pts,
            actual: von_mises.values.len(),
        });
    }
    for field in extra_scalar_fields {
        if field.values.len() != n_pts {
            return Err(FileIoError::VtkFieldLengthMismatch {
                name: field.name.clone(),
                expected: n_pts,
                actual: field.values.len(),
            });
        }
    }

    let mut output = String::new();
    writeln!(&mut output, "# vtk DataFile Version 3.0").expect("writing to string cannot fail");
    writeln!(&mut output, "kepler mesh with stress").expect("writing to string cannot fail");
    writeln!(&mut output, "ASCII").expect("writing to string cannot fail");
    writeln!(&mut output, "DATASET UNSTRUCTURED_GRID").expect("writing to string cannot fail");
    writeln!(&mut output, "POINTS {} double", n_pts).expect("writing to string cannot fail");
    for point in topology.points() {
        let [x, y, z] = point_to_vtk_coords(point);
        writeln!(&mut output, "{x} {y} {z}").expect("writing to string cannot fail");
    }

    let cell_size: usize = topology
        .cells()
        .iter()
        .map(|cell| 1 + cell.nodes.len())
        .sum();
    writeln!(
        &mut output,
        "CELLS {} {}",
        topology.cells().len(),
        cell_size
    )
    .expect("writing to string cannot fail");
    for cell in topology.cells() {
        write!(&mut output, "{}", cell.nodes.len()).expect("writing to string cannot fail");
        for node in &cell.nodes {
            write!(&mut output, " {node}").expect("writing to string cannot fail");
        }
        writeln!(&mut output).expect("writing to string cannot fail");
    }

    writeln!(&mut output, "CELL_TYPES {}", topology.cells().len())
        .expect("writing to string cannot fail");
    for cell in topology.cells() {
        writeln!(&mut output, "{}", vtk_cell_type(cell.kind))
            .expect("writing to string cannot fail");
    }

    // Point data: displacement as VECTORS first, then von Mises and extras as SCALARS.
    writeln!(&mut output, "POINT_DATA {n_pts}").expect("writing to string cannot fail");
    writeln!(
        &mut output,
        "VECTORS {} double",
        sanitize_vtk_name(&displacement_field.name)
    )
    .expect("writing to string cannot fail");
    for [fx, fy, fz] in &displacement_field.values {
        writeln!(&mut output, "{fx} {fy} {fz}").expect("writing to string cannot fail");
    }

    // von Mises scalar field.
    writeln!(
        &mut output,
        "SCALARS {} double 1",
        sanitize_vtk_name(&von_mises.name)
    )
    .expect("writing to string cannot fail");
    writeln!(&mut output, "LOOKUP_TABLE default").expect("writing to string cannot fail");
    for value in &von_mises.values {
        writeln!(&mut output, "{value}").expect("writing to string cannot fail");
    }

    // Extra scalar fields.
    for field in extra_scalar_fields {
        writeln!(
            &mut output,
            "SCALARS {} double 1",
            sanitize_vtk_name(&field.name)
        )
        .expect("writing to string cannot fail");
        writeln!(&mut output, "LOOKUP_TABLE default").expect("writing to string cannot fail");
        for value in &field.values {
            writeln!(&mut output, "{value}").expect("writing to string cannot fail");
        }
    }

    Ok(output)
}

fn point_to_vtk_coords<const D: usize>(point: &PointD<D>) -> [f64; 3] {
    [
        point.coords.first().copied().unwrap_or(0.0),
        point.coords.get(1).copied().unwrap_or(0.0),
        point.coords.get(2).copied().unwrap_or(0.0),
    ]
}

fn vtk_cell_type(kind: ElementKind) -> usize {
    match kind {
        ElementKind::Line2 => 3,
        ElementKind::Line3 => 21,
        ElementKind::Tri3 => 5,
        ElementKind::Tri6 => 22,
        ElementKind::Quad4 => 9,
        ElementKind::Quad8 => 23,
        ElementKind::Tet4 => 10,
        ElementKind::Tet10 => 24,
        ElementKind::Hex8 => 12,
        ElementKind::Hex20 => 25,
    }
}

fn sanitize_vtk_name(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "field".to_owned()
    } else {
        sanitized
    }
}
