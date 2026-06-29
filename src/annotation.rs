use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::mesh::{MeshTopology, Region, RegionId};

pub type AnnotationId = usize;
pub type ParameterId = usize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EntitySelector {
    RegionId(RegionId),
    RegionName(String),
}

impl EntitySelector {
    pub fn region_id(region_id: RegionId) -> Self {
        Self::RegionId(region_id)
    }

    pub fn region_name(name: impl Into<String>) -> Self {
        Self::RegionName(name.into())
    }

    pub fn resolve_region_id(&self, regions: &[Region]) -> Result<RegionId, AnnotationError> {
        match self {
            Self::RegionId(region_id) => regions
                .iter()
                .find(|region| region.id == *region_id)
                .map(|region| region.id)
                .ok_or(AnnotationError::UnknownRegionId {
                    region_id: *region_id,
                }),
            Self::RegionName(name) => {
                let mut matches = regions.iter().filter(|region| region.name == *name);
                let Some(region) = matches.next() else {
                    return Err(AnnotationError::UnknownRegionName { name: name.clone() });
                };
                if matches.next().is_some() {
                    return Err(AnnotationError::AmbiguousRegionName { name: name.clone() });
                }
                Ok(region.id)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    Scalar(f64),
    Vector(Vec<f64>),
    Text(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub value: ParameterValue,
    pub units: Option<String>,
}

impl Parameter {
    pub fn scalar(name: impl Into<String>, value: f64, units: Option<impl Into<String>>) -> Self {
        Self {
            name: name.into(),
            value: ParameterValue::Scalar(value),
            units: units.map(Into::into),
        }
    }

    pub fn vector(
        name: impl Into<String>,
        value: Vec<f64>,
        units: Option<impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            value: ParameterValue::Vector(value),
            units: units.map(Into::into),
        }
    }

    pub fn text(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: ParameterValue::Text(value.into()),
            units: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaterialAssignment {
    pub id: AnnotationId,
    pub name: String,
    pub target: EntitySelector,
    pub properties: Vec<Parameter>,
}

impl MaterialAssignment {
    pub fn new(
        id: AnnotationId,
        name: impl Into<String>,
        target: EntitySelector,
        properties: Vec<Parameter>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            target,
            properties,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParameterAssignment {
    pub id: ParameterId,
    pub name: String,
    pub target: EntitySelector,
    pub value: ParameterValue,
    pub units: Option<String>,
}

impl ParameterAssignment {
    pub fn new(
        id: ParameterId,
        name: impl Into<String>,
        target: EntitySelector,
        value: ParameterValue,
        units: Option<impl Into<String>>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            target,
            value,
            units: units.map(Into::into),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GeometryAnnotations {
    materials: Vec<MaterialAssignment>,
    parameters: Vec<ParameterAssignment>,
}

impl GeometryAnnotations {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_material(mut self, material: MaterialAssignment) -> Self {
        self.materials.push(material);
        self
    }

    pub fn with_parameter(mut self, parameter: ParameterAssignment) -> Self {
        self.parameters.push(parameter);
        self
    }

    pub fn materials(&self) -> &[MaterialAssignment] {
        &self.materials
    }

    pub fn parameters(&self) -> &[ParameterAssignment] {
        &self.parameters
    }

    pub fn validate_for_topology<const D: usize>(
        &self,
        topology: &MeshTopology<D>,
    ) -> Result<ResolvedAnnotations, AnnotationError> {
        let mut material_ids = BTreeSet::new();
        let mut material_names = BTreeSet::new();
        let mut resolved_materials = Vec::with_capacity(self.materials.len());

        for material in &self.materials {
            if !material_ids.insert(material.id) {
                return Err(AnnotationError::DuplicateMaterialId { id: material.id });
            }
            if !material_names.insert(material.name.clone()) {
                return Err(AnnotationError::DuplicateMaterialName {
                    name: material.name.clone(),
                });
            }

            let region_id = material.target.resolve_region_id(topology.regions())?;
            validate_parameter_names(&material.properties)?;
            resolved_materials.push(ResolvedMaterialAssignment {
                id: material.id,
                name: material.name.clone(),
                region_id,
                properties: material.properties.clone(),
            });
        }

        let mut parameter_ids = BTreeSet::new();
        let mut parameters_by_target = BTreeSet::new();
        let mut resolved_parameters = Vec::with_capacity(self.parameters.len());

        for parameter in &self.parameters {
            if !parameter_ids.insert(parameter.id) {
                return Err(AnnotationError::DuplicateParameterId { id: parameter.id });
            }
            let region_id = parameter.target.resolve_region_id(topology.regions())?;
            if !parameters_by_target.insert((region_id, parameter.name.clone())) {
                return Err(AnnotationError::DuplicateParameterForTarget {
                    region_id,
                    name: parameter.name.clone(),
                });
            }

            resolved_parameters.push(ResolvedParameterAssignment {
                id: parameter.id,
                name: parameter.name.clone(),
                region_id,
                value: parameter.value.clone(),
                units: parameter.units.clone(),
            });
        }

        Ok(ResolvedAnnotations {
            materials: resolved_materials,
            parameters: resolved_parameters,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAnnotations {
    pub materials: Vec<ResolvedMaterialAssignment>,
    pub parameters: Vec<ResolvedParameterAssignment>,
}

impl ResolvedAnnotations {
    pub fn material_by_region(&self) -> BTreeMap<RegionId, &ResolvedMaterialAssignment> {
        self.materials
            .iter()
            .map(|material| (material.region_id, material))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedMaterialAssignment {
    pub id: AnnotationId,
    pub name: String,
    pub region_id: RegionId,
    pub properties: Vec<Parameter>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedParameterAssignment {
    pub id: ParameterId,
    pub name: String,
    pub region_id: RegionId,
    pub value: ParameterValue,
    pub units: Option<String>,
}

#[derive(Debug, Error, PartialEq)]
pub enum AnnotationError {
    #[error("region id {region_id} does not exist")]
    UnknownRegionId { region_id: RegionId },
    #[error("region name '{name}' does not exist")]
    UnknownRegionName { name: String },
    #[error("region name '{name}' matches multiple regions")]
    AmbiguousRegionName { name: String },
    #[error("material id {id} is defined more than once")]
    DuplicateMaterialId { id: AnnotationId },
    #[error("material name '{name}' is defined more than once")]
    DuplicateMaterialName { name: String },
    #[error("parameter id {id} is defined more than once")]
    DuplicateParameterId { id: ParameterId },
    #[error("parameter '{name}' is assigned more than once to region {region_id}")]
    DuplicateParameterForTarget { region_id: RegionId, name: String },
    #[error("material property '{name}' is defined more than once")]
    DuplicateMaterialProperty { name: String },
}

fn validate_parameter_names(parameters: &[Parameter]) -> Result<(), AnnotationError> {
    let mut seen = BTreeSet::new();
    for parameter in parameters {
        if !seen.insert(parameter.name.clone()) {
            return Err(AnnotationError::DuplicateMaterialProperty {
                name: parameter.name.clone(),
            });
        }
    }
    Ok(())
}
