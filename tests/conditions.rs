use kepler::{
    Cell, Condition, ConditionError, ConditionKind, ConditionSet, ConditionSignature, ElementKind,
    EntityDimension, EntitySelector, MeshTopology, ParameterValue, PointD, Region,
};

#[test]
fn conditions_resolve_boundary_domain_and_point_targets() {
    let topology = sample_topology();
    let conditions = ConditionSet::new()
        .with_condition(Condition::new(
            0,
            "fixed temperature",
            EntitySelector::region_name("left"),
            ConditionKind::Dirichlet {
                field: "temperature".to_owned(),
                value: ParameterValue::Scalar(300.0),
            },
        ))
        .with_condition(Condition::new(
            1,
            "heat flux",
            EntitySelector::region_name("left"),
            ConditionKind::HeatFlux {
                value: 25.0,
                units: Some("W/m^2".to_owned()),
            },
        ))
        .with_condition(Condition::new(
            2,
            "gravity",
            EntitySelector::region_name("body"),
            ConditionKind::BodyLoad {
                components: vec![0.0, -9.81],
                units: Some("m/s^2".to_owned()),
            },
        ))
        .with_condition(Condition::new(
            3,
            "probe force",
            EntitySelector::region_name("corner"),
            ConditionKind::PointLoad {
                components: vec![1.0, 0.0],
                units: Some("N".to_owned()),
            },
        ));

    let resolved = conditions.validate_for_topology(&topology).unwrap();
    let by_region = resolved.by_region();

    assert_eq!(resolved.conditions.len(), 4);
    assert_eq!(by_region[&1].len(), 2);
    assert_eq!(by_region[&2].len(), 1);
    assert_eq!(by_region[&3].len(), 1);
}

#[test]
fn pressure_requires_boundary_region() {
    let topology = sample_topology();
    let conditions = ConditionSet::new().with_condition(Condition::new(
        0,
        "pressure",
        EntitySelector::region_name("body"),
        ConditionKind::Pressure {
            value: 100.0,
            units: Some("Pa".to_owned()),
        },
    ));

    let error = conditions.validate_for_topology(&topology).unwrap_err();

    assert!(matches!(
        error,
        ConditionError::InvalidTargetDimension {
            condition_id: 0,
            actual: EntityDimension::Surface,
            ..
        }
    ));
}

#[test]
fn body_load_requires_domain_region() {
    let topology = sample_topology();
    let conditions = ConditionSet::new().with_condition(Condition::new(
        0,
        "gravity",
        EntitySelector::region_name("left"),
        ConditionKind::BodyLoad {
            components: vec![0.0, -9.81],
            units: Some("m/s^2".to_owned()),
        },
    ));

    let error = conditions.validate_for_topology(&topology).unwrap_err();

    assert!(matches!(
        error,
        ConditionError::InvalidTargetDimension {
            condition_id: 0,
            actual: EntityDimension::Curve,
            ..
        }
    ));
}

#[test]
fn duplicate_essential_conditions_for_same_field_and_target_are_rejected() {
    let topology = sample_topology();
    let conditions = ConditionSet::new()
        .with_condition(Condition::new(
            0,
            "temperature a",
            EntitySelector::region_name("left"),
            ConditionKind::Dirichlet {
                field: "temperature".to_owned(),
                value: ParameterValue::Scalar(300.0),
            },
        ))
        .with_condition(Condition::new(
            1,
            "temperature b",
            EntitySelector::region_name("left"),
            ConditionKind::Dirichlet {
                field: "temperature".to_owned(),
                value: ParameterValue::Scalar(350.0),
            },
        ));

    let error = conditions.validate_for_topology(&topology).unwrap_err();

    assert_eq!(
        error,
        ConditionError::DuplicateConditionForTarget {
            region_id: 1,
            signature: ConditionSignature::EssentialField("temperature".to_owned()),
        }
    );
}

#[test]
fn unknown_condition_targets_are_rejected() {
    let topology = sample_topology();
    let conditions = ConditionSet::new().with_condition(Condition::new(
        0,
        "missing",
        EntitySelector::region_name("missing"),
        ConditionKind::HeatFlux {
            value: 10.0,
            units: Some("W/m^2".to_owned()),
        },
    ));

    let error = conditions.validate_for_topology(&topology).unwrap_err();

    assert!(matches!(error, ConditionError::Selector(_)));
}

fn sample_topology() -> MeshTopology<2> {
    MeshTopology::with_regions(
        vec![
            PointD::new([0.0, 0.0]),
            PointD::new([1.0, 0.0]),
            PointD::new([0.0, 1.0]),
        ],
        vec![
            Cell::new(ElementKind::Tri3, vec![0, 1, 2]).with_region(2),
            Cell::new(ElementKind::Line2, vec![0, 2]).with_region(1),
            Cell::new(ElementKind::Line2, vec![0, 1]).with_region(1),
        ],
        vec![
            Region::new(1, "left", EntityDimension::Curve),
            Region::new(2, "body", EntityDimension::Surface),
            Region::new(3, "corner", EntityDimension::Point),
        ],
    )
    .unwrap()
}
