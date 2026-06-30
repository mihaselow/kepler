use kepler::{
    ElementKind, FileIoError, ImportedMesh, VtkScalarField, format_vtk_legacy, parse_gmsh_str,
};

const GMSH_2D: &str = include_str!("../examples/data/physical_groups_2d.msh");
const VTK_2D_TEMPERATURE: &str =
    include_str!("../examples/data/physical_groups_2d_temperature.vtk");

#[test]
fn gmsh_import_preserves_physical_groups_as_regions() {
    let imported = parse_gmsh_str(GMSH_2D).unwrap();

    let ImportedMesh::TwoD(topology) = imported else {
        panic!("expected a 2D topology");
    };

    assert_eq!(topology.points().len(), 3);
    assert_eq!(topology.cells().len(), 2);
    assert_eq!(topology.regions().len(), 2);
    assert_eq!(topology.regions()[0].name, "left");
    assert_eq!(topology.regions()[1].name, "body");
    assert_eq!(topology.cells()[0].kind, ElementKind::Line2);
    assert_eq!(topology.cells()[0].region, Some(0));
    assert_eq!(topology.cells()[1].kind, ElementKind::Tri3);
    assert_eq!(topology.cells()[1].region, Some(1));
}

#[test]
fn gmsh_import_supports_three_dimensional_tetrahedra() {
    let imported = parse_gmsh_str(GMSH_3D).unwrap();

    let ImportedMesh::ThreeD(topology) = imported else {
        panic!("expected a 3D topology");
    };

    assert_eq!(topology.dimension(), 3);
    assert_eq!(topology.points().len(), 4);
    assert_eq!(topology.cells()[0].kind, ElementKind::Tet4);
    assert_eq!(topology.regions()[0].name, "volume");
}

#[test]
fn gmsh_import_rejects_unknown_element_types() {
    let error = parse_gmsh_str(GMSH_UNSUPPORTED_ELEMENT).unwrap_err();

    assert!(matches!(
        error,
        FileIoError::UnsupportedGmshElement {
            element_type: 15,
            ..
        }
    ));
}

#[test]
fn vtk_export_writes_unstructured_grid_and_scalar_data() {
    let ImportedMesh::TwoD(topology) = parse_gmsh_str(GMSH_2D).unwrap() else {
        panic!("expected a 2D topology");
    };

    let output = format_vtk_legacy(
        &topology,
        &[VtkScalarField::new(
            "temperature field",
            vec![0.0, 0.5, 1.0],
        )],
    )
    .unwrap();

    assert_eq!(output, VTK_2D_TEMPERATURE);
    assert!(output.contains("DATASET UNSTRUCTURED_GRID"));
    assert!(output.contains("POINTS 3 double"));
    assert!(output.contains("CELL_TYPES 2"));
    assert!(output.contains("SCALARS temperature_field double 1"));
    assert!(output.contains("LOOKUP_TABLE default"));
}

#[test]
fn vtk_export_rejects_scalar_fields_with_wrong_length() {
    let ImportedMesh::TwoD(topology) = parse_gmsh_str(GMSH_2D).unwrap() else {
        panic!("expected a 2D topology");
    };

    let error = format_vtk_legacy(&topology, &[VtkScalarField::new("u", vec![1.0])]).unwrap_err();

    assert!(matches!(
        error,
        FileIoError::VtkFieldLengthMismatch {
            name,
            expected: 3,
            actual: 1,
        } if name == "u"
    ));
}

const GMSH_3D: &str = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
3 4 "volume"
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 0.0 0.0 1.0
$EndNodes
$Elements
1
1 4 2 4 0 1 2 3 4
$EndElements
"#;

const GMSH_UNSUPPORTED_ELEMENT: &str = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
1
1 0.0 0.0 0.0
$EndNodes
$Elements
1
1 15 0 1
$EndElements
"#;
