use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadFileFormat {
    Step,
    Iges,
    Brep,
    Stl,
}

impl CadFileFormat {
    pub const fn as_extension(self) -> &'static str {
        match self {
            Self::Step => "step",
            Self::Iges => "iges",
            Self::Brep => "brep",
            Self::Stl => "stl",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadLengthUnit {
    Millimeter,
    Meter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CadSource {
    pub path: PathBuf,
    pub format: CadFileFormat,
    pub unit: CadLengthUnit,
}

impl CadSource {
    pub fn from_path(
        path: impl Into<PathBuf>,
        unit: CadLengthUnit,
    ) -> Result<Self, CadWorkflowError> {
        let path = path.into();
        let format = infer_cad_format(&path)?;
        Ok(Self { path, format, unit })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadMeshingDimension {
    Surface2D,
    Volume3D,
}

impl CadMeshingDimension {
    const fn gmsh_flag(self) -> &'static str {
        match self {
            Self::Surface2D => "-2",
            Self::Volume3D => "-3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadMeshOutputFormat {
    GmshMsh2,
}

impl CadMeshOutputFormat {
    const fn gmsh_format(self) -> &'static str {
        match self {
            Self::GmshMsh2 => "msh2",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadMeshingOptions {
    pub dimension: CadMeshingDimension,
    #[serde(default)]
    pub max_element_size: Option<f64>,
    #[serde(default)]
    pub element_order: Option<usize>,
    #[serde(default)]
    pub optimize: bool,
    pub output_format: CadMeshOutputFormat,
}

impl Default for CadMeshingOptions {
    fn default() -> Self {
        Self {
            dimension: CadMeshingDimension::Volume3D,
            max_element_size: None,
            element_order: None,
            optimize: false,
            output_format: CadMeshOutputFormat::GmshMsh2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExternalMesher {
    Gmsh {
        #[serde(default = "default_gmsh_executable")]
        executable: String,
    },
}

impl Default for ExternalMesher {
    fn default() -> Self {
        Self::Gmsh {
            executable: default_gmsh_executable(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadMeshingWorkflow {
    pub source: CadSource,
    #[serde(default)]
    pub mesher: ExternalMesher,
    #[serde(default)]
    pub options: CadMeshingOptions,
    pub output_mesh: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalCommand {
    pub program: String,
    pub args: Vec<String>,
}

#[derive(Debug, Error, PartialEq)]
pub enum CadWorkflowError {
    #[error("CAD source path must not be empty")]
    EmptySourcePath,
    #[error("CAD output mesh path must not be empty")]
    EmptyOutputPath,
    #[error("unsupported CAD file extension '{extension}'")]
    UnsupportedCadExtension { extension: String },
    #[error("max_element_size must be positive and finite, got {value}")]
    InvalidMaxElementSize { value: f64 },
    #[error("element_order must be greater than zero")]
    InvalidElementOrder,
    #[error("Gmsh executable must not be empty")]
    EmptyMesherExecutable,
    #[error("CAD workflow currently writes Gmsh .msh output, got '{extension}'")]
    UnsupportedOutputExtension { extension: String },
}

pub fn infer_cad_format(path: impl AsRef<Path>) -> Result<CadFileFormat, CadWorkflowError> {
    let extension =
        path_extension(path.as_ref()).ok_or_else(|| CadWorkflowError::UnsupportedCadExtension {
            extension: String::new(),
        })?;

    match extension.as_str() {
        "step" | "stp" => Ok(CadFileFormat::Step),
        "iges" | "igs" => Ok(CadFileFormat::Iges),
        "brep" => Ok(CadFileFormat::Brep),
        "stl" => Ok(CadFileFormat::Stl),
        _ => Err(CadWorkflowError::UnsupportedCadExtension { extension }),
    }
}

pub fn validate_cad_meshing_workflow(
    workflow: &CadMeshingWorkflow,
) -> Result<(), CadWorkflowError> {
    if workflow.source.path.as_os_str().is_empty() {
        return Err(CadWorkflowError::EmptySourcePath);
    }
    if workflow.output_mesh.as_os_str().is_empty() {
        return Err(CadWorkflowError::EmptyOutputPath);
    }
    if let Some(value) = workflow.options.max_element_size
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(CadWorkflowError::InvalidMaxElementSize { value });
    }
    if matches!(workflow.options.element_order, Some(0)) {
        return Err(CadWorkflowError::InvalidElementOrder);
    }
    if output_extension(&workflow.output_mesh) != Some("msh".to_owned()) {
        return Err(CadWorkflowError::UnsupportedOutputExtension {
            extension: output_extension(&workflow.output_mesh).unwrap_or_default(),
        });
    }

    match &workflow.mesher {
        ExternalMesher::Gmsh { executable } if executable.trim().is_empty() => {
            Err(CadWorkflowError::EmptyMesherExecutable)
        }
        ExternalMesher::Gmsh { .. } => Ok(()),
    }
}

pub fn plan_cad_meshing_command(
    workflow: &CadMeshingWorkflow,
) -> Result<ExternalCommand, CadWorkflowError> {
    validate_cad_meshing_workflow(workflow)?;

    match &workflow.mesher {
        ExternalMesher::Gmsh { executable } => Ok(plan_gmsh_command(workflow, executable)),
    }
}

fn plan_gmsh_command(workflow: &CadMeshingWorkflow, executable: &str) -> ExternalCommand {
    let mut args = vec![
        workflow.source.path.to_string_lossy().into_owned(),
        workflow.options.dimension.gmsh_flag().to_owned(),
        "-format".to_owned(),
        workflow.options.output_format.gmsh_format().to_owned(),
        "-o".to_owned(),
        workflow.output_mesh.to_string_lossy().into_owned(),
    ];

    if let Some(max_element_size) = workflow.options.max_element_size {
        args.push("-clmax".to_owned());
        args.push(max_element_size.to_string());
    }
    if let Some(element_order) = workflow.options.element_order {
        args.push("-order".to_owned());
        args.push(element_order.to_string());
    }
    if workflow.options.optimize {
        args.push("-optimize".to_owned());
    }

    ExternalCommand {
        program: executable.to_owned(),
        args,
    }
}

fn path_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
}

fn output_extension(path: &Path) -> Option<String> {
    path_extension(path)
}

fn default_gmsh_executable() -> String {
    "gmsh".to_owned()
}
