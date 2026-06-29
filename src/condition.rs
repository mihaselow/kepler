use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::{
    annotation::{AnnotationError, EntitySelector, ParameterValue},
    mesh::{EntityDimension, MeshTopology, RegionId},
};

pub type ConditionId = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub id: ConditionId,
    pub name: String,
    pub target: EntitySelector,
    pub kind: ConditionKind,
}

impl Condition {
    pub fn new(
        id: ConditionId,
        name: impl Into<String>,
        target: EntitySelector,
        kind: ConditionKind,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            target,
            kind,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConditionKind {
    Dirichlet {
        field: String,
        value: ParameterValue,
    },
    Neumann {
        field: String,
        flux: ParameterValue,
    },
    Robin {
        field: String,
        coefficient: ParameterValue,
        reference: ParameterValue,
    },
    PointLoad {
        components: Vec<f64>,
        units: Option<String>,
    },
    BodyLoad {
        components: Vec<f64>,
        units: Option<String>,
    },
    Traction {
        components: Vec<f64>,
        units: Option<String>,
    },
    Pressure {
        value: f64,
        units: Option<String>,
    },
    HeatFlux {
        value: f64,
        units: Option<String>,
    },
}

impl ConditionKind {
    fn target_class(&self) -> TargetClass {
        match self {
            Self::Dirichlet { .. } => TargetClass::AnyEntity,
            Self::Neumann { .. } | Self::Robin { .. } | Self::Traction { .. } => {
                TargetClass::Boundary
            }
            Self::PointLoad { .. } => TargetClass::Point,
            Self::BodyLoad { .. } => TargetClass::Domain,
            Self::Pressure { .. } | Self::HeatFlux { .. } => TargetClass::Boundary,
        }
    }

    fn signature(&self) -> ConditionSignature {
        match self {
            Self::Dirichlet { field, .. } => ConditionSignature::EssentialField(field.clone()),
            Self::Neumann { field, .. } => ConditionSignature::NaturalField {
                kind: "neumann",
                field: field.clone(),
            },
            Self::Robin { field, .. } => ConditionSignature::NaturalField {
                kind: "robin",
                field: field.clone(),
            },
            Self::PointLoad { .. } => ConditionSignature::Mechanical("point_load"),
            Self::BodyLoad { .. } => ConditionSignature::Mechanical("body_load"),
            Self::Traction { .. } => ConditionSignature::Mechanical("traction"),
            Self::Pressure { .. } => ConditionSignature::Mechanical("pressure"),
            Self::HeatFlux { .. } => ConditionSignature::Thermal("heat_flux"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConditionSet {
    conditions: Vec<Condition>,
}

impl ConditionSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_condition(mut self, condition: Condition) -> Self {
        self.conditions.push(condition);
        self
    }

    pub fn conditions(&self) -> &[Condition] {
        &self.conditions
    }

    pub fn validate_for_topology<const D: usize>(
        &self,
        topology: &MeshTopology<D>,
    ) -> Result<ResolvedConditionSet, ConditionError> {
        let mut ids = BTreeSet::new();
        let mut resolved = Vec::with_capacity(self.conditions.len());
        let mut signatures = BTreeSet::new();

        for condition in &self.conditions {
            if !ids.insert(condition.id) {
                return Err(ConditionError::DuplicateConditionId { id: condition.id });
            }

            let region_id = condition.target.resolve_region_id(topology.regions())?;
            let region = topology
                .regions()
                .iter()
                .find(|region| region.id == region_id)
                .expect("resolved region id must exist");

            validate_target_dimension(condition, D, region.dimension)?;

            let signature = condition.kind.signature();
            let signature_key = (region_id, signature.clone());
            if !signatures.insert(signature_key) {
                return Err(ConditionError::DuplicateConditionForTarget {
                    region_id,
                    signature,
                });
            }

            resolved.push(ResolvedCondition {
                id: condition.id,
                name: condition.name.clone(),
                region_id,
                kind: condition.kind.clone(),
            });
        }

        Ok(ResolvedConditionSet {
            conditions: resolved,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedConditionSet {
    pub conditions: Vec<ResolvedCondition>,
}

impl ResolvedConditionSet {
    pub fn by_region(&self) -> BTreeMap<RegionId, Vec<&ResolvedCondition>> {
        let mut by_region = BTreeMap::new();
        for condition in &self.conditions {
            by_region
                .entry(condition.region_id)
                .or_insert_with(Vec::new)
                .push(condition);
        }
        by_region
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCondition {
    pub id: ConditionId,
    pub name: String,
    pub region_id: RegionId,
    pub kind: ConditionKind,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConditionSignature {
    EssentialField(String),
    NaturalField { kind: &'static str, field: String },
    Mechanical(&'static str),
    Thermal(&'static str),
}

#[derive(Debug, Error, PartialEq)]
pub enum ConditionError {
    #[error(transparent)]
    Selector(#[from] AnnotationError),
    #[error("condition id {id} is defined more than once")]
    DuplicateConditionId { id: ConditionId },
    #[error(
        "condition {condition_id} targets a {actual:?} region, but {kind:?} requires {expected}"
    )]
    InvalidTargetDimension {
        condition_id: ConditionId,
        kind: ConditionKind,
        expected: &'static str,
        actual: EntityDimension,
    },
    #[error("condition signature {signature:?} is assigned more than once to region {region_id}")]
    DuplicateConditionForTarget {
        region_id: RegionId,
        signature: ConditionSignature,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetClass {
    AnyEntity,
    Point,
    Boundary,
    Domain,
}

fn validate_target_dimension(
    condition: &Condition,
    mesh_dimension: usize,
    region_dimension: EntityDimension,
) -> Result<(), ConditionError> {
    let region_dimension_value = region_dimension.spatial_dimension();
    let is_valid = match condition.kind.target_class() {
        TargetClass::AnyEntity => region_dimension_value <= mesh_dimension,
        TargetClass::Point => region_dimension == EntityDimension::Point,
        TargetClass::Boundary => mesh_dimension > 0 && region_dimension_value + 1 == mesh_dimension,
        TargetClass::Domain => region_dimension_value == mesh_dimension,
    };

    if is_valid {
        Ok(())
    } else {
        Err(ConditionError::InvalidTargetDimension {
            condition_id: condition.id,
            kind: condition.kind.clone(),
            expected: expected_description(condition.kind.target_class()),
            actual: region_dimension,
        })
    }
}

fn expected_description(target_class: TargetClass) -> &'static str {
    match target_class {
        TargetClass::AnyEntity => "any entity within the mesh dimension",
        TargetClass::Point => "a point region",
        TargetClass::Boundary => "a boundary region with dimension mesh_dimension - 1",
        TargetClass::Domain => "a domain region with the same dimension as the mesh",
    }
}
