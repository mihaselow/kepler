use kepler::{
    AnnotationError, Cell, ElementKind, EntityDimension, EntitySelector, GeometryAnnotations,
    MaterialAssignment, MeshTopology, Parameter, ParameterAssignment, ParameterValue, PointD,
    Region,
};

#[test]
fn annotations_target_regions_by_name_or_id() {
    let topology = sample_topology();
    let annotations = GeometryAnnotations::new()
        .with_material(MaterialAssignment::new(
            0,
            "steel",
            EntitySelector::region_name("body"),
            vec![
                Parameter::scalar("young_modulus", 210.0e9, Some("Pa")),
                Parameter::scalar("poisson_ratio", 0.3, None::<String>),
            ],
        ))
        .with_parameter(ParameterAssignment::new(
            1,
            "mesh_size",
            EntitySelector::region_id(10),
            ParameterValue::Scalar(0.05),
            Some("m"),
        ));

    let resolved = annotations.validate_for_topology(&topology).unwrap();

    assert_eq!(resolved.materials[0].region_id, 10);
    assert_eq!(resolved.parameters[0].region_id, 10);
    assert_eq!(resolved.material_by_region()[&10].name, "steel");
}

#[test]
fn annotations_reject_unknown_region_names() {
    let topology = sample_topology();
    let annotations = GeometryAnnotations::new().with_material(MaterialAssignment::new(
        0,
        "steel",
        EntitySelector::region_name("missing"),
        vec![],
    ));

    let error = annotations.validate_for_topology(&topology).unwrap_err();

    assert_eq!(
        error,
        AnnotationError::UnknownRegionName {
            name: "missing".to_owned()
        }
    );
}

#[test]
fn annotations_reject_duplicate_material_names() {
    let topology = sample_topology();
    let annotations = GeometryAnnotations::new()
        .with_material(MaterialAssignment::new(
            0,
            "steel",
            EntitySelector::region_id(10),
            vec![],
        ))
        .with_material(MaterialAssignment::new(
            1,
            "steel",
            EntitySelector::region_id(10),
            vec![],
        ));

    let error = annotations.validate_for_topology(&topology).unwrap_err();

    assert_eq!(
        error,
        AnnotationError::DuplicateMaterialName {
            name: "steel".to_owned()
        }
    );
}

#[test]
fn annotations_reject_duplicate_material_properties() {
    let topology = sample_topology();
    let annotations = GeometryAnnotations::new().with_material(MaterialAssignment::new(
        0,
        "steel",
        EntitySelector::region_id(10),
        vec![
            Parameter::scalar("density", 7850.0, Some("kg/m^3")),
            Parameter::scalar("density", 7860.0, Some("kg/m^3")),
        ],
    ));

    let error = annotations.validate_for_topology(&topology).unwrap_err();

    assert_eq!(
        error,
        AnnotationError::DuplicateMaterialProperty {
            name: "density".to_owned()
        }
    );
}

#[test]
fn annotations_reject_duplicate_parameters_on_same_target() {
    let topology = sample_topology();
    let annotations = GeometryAnnotations::new()
        .with_parameter(ParameterAssignment::new(
            0,
            "temperature",
            EntitySelector::region_name("body"),
            ParameterValue::Scalar(300.0),
            Some("K"),
        ))
        .with_parameter(ParameterAssignment::new(
            1,
            "temperature",
            EntitySelector::region_id(10),
            ParameterValue::Scalar(350.0),
            Some("K"),
        ));

    let error = annotations.validate_for_topology(&topology).unwrap_err();

    assert_eq!(
        error,
        AnnotationError::DuplicateParameterForTarget {
            region_id: 10,
            name: "temperature".to_owned()
        }
    );
}

fn sample_topology() -> MeshTopology<2> {
    MeshTopology::with_regions(
        vec![
            PointD::new([0.0, 0.0]),
            PointD::new([1.0, 0.0]),
            PointD::new([0.0, 1.0]),
        ],
        vec![Cell::new(ElementKind::Tri3, vec![0, 1, 2]).with_region(10)],
        vec![Region::new(10, "body", EntityDimension::Surface)],
    )
    .unwrap()
}
