use std::path::PathBuf;

use kepler::{
    CadFileFormat, CadLengthUnit, CadMeshOutputFormat, CadMeshingDimension, CadMeshingOptions,
    CadMeshingWorkflow, CadSource, CadWorkflowError, ExternalCommand, ExternalMesher,
    infer_cad_format, plan_cad_meshing_command, validate_cad_meshing_workflow,
};

#[test]
fn infers_supported_cad_formats_from_extensions() {
    assert_eq!(
        infer_cad_format("bracket.step").unwrap(),
        CadFileFormat::Step
    );
    assert_eq!(
        infer_cad_format("bracket.stp").unwrap(),
        CadFileFormat::Step
    );
    assert_eq!(infer_cad_format("blade.iges").unwrap(), CadFileFormat::Iges);
    assert_eq!(infer_cad_format("blade.igs").unwrap(), CadFileFormat::Iges);
    assert_eq!(infer_cad_format("solid.brep").unwrap(), CadFileFormat::Brep);
    assert_eq!(infer_cad_format("surface.stl").unwrap(), CadFileFormat::Stl);
}

#[test]
fn rejects_unsupported_cad_extensions() {
    let error = infer_cad_format("drawing.dxf").unwrap_err();

    assert!(matches!(
        error,
        CadWorkflowError::UnsupportedCadExtension { extension } if extension == "dxf"
    ));
}

#[test]
fn plans_gmsh_volume_meshing_command_for_external_workflow() {
    let workflow = CadMeshingWorkflow {
        source: CadSource::from_path("models/bracket.step", CadLengthUnit::Millimeter).unwrap(),
        mesher: ExternalMesher::Gmsh {
            executable: "gmsh".to_owned(),
        },
        options: CadMeshingOptions {
            dimension: CadMeshingDimension::Volume3D,
            max_element_size: Some(0.25),
            element_order: Some(2),
            optimize: true,
            output_format: CadMeshOutputFormat::GmshMsh2,
        },
        output_mesh: PathBuf::from("target/bracket.msh"),
    };

    let command = plan_cad_meshing_command(&workflow).unwrap();

    assert_eq!(
        command,
        ExternalCommand {
            program: "gmsh".to_owned(),
            args: vec![
                "models/bracket.step".to_owned(),
                "-3".to_owned(),
                "-format".to_owned(),
                "msh2".to_owned(),
                "-o".to_owned(),
                "target/bracket.msh".to_owned(),
                "-clmax".to_owned(),
                "0.25".to_owned(),
                "-order".to_owned(),
                "2".to_owned(),
                "-optimize".to_owned(),
            ],
        }
    );
}

#[test]
fn plans_gmsh_surface_meshing_command_with_minimal_options() {
    let workflow = CadMeshingWorkflow {
        source: CadSource::from_path("models/shell.igs", CadLengthUnit::Meter).unwrap(),
        mesher: ExternalMesher::default(),
        options: CadMeshingOptions {
            dimension: CadMeshingDimension::Surface2D,
            ..CadMeshingOptions::default()
        },
        output_mesh: PathBuf::from("target/shell.msh"),
    };

    let command = plan_cad_meshing_command(&workflow).unwrap();

    assert_eq!(command.program, "gmsh");
    assert_eq!(
        command.args,
        vec![
            "models/shell.igs".to_owned(),
            "-2".to_owned(),
            "-format".to_owned(),
            "msh2".to_owned(),
            "-o".to_owned(),
            "target/shell.msh".to_owned(),
        ]
    );
}

#[test]
fn rejects_invalid_meshing_options_before_command_planning() {
    let mut workflow = CadMeshingWorkflow {
        source: CadSource::from_path("models/bracket.step", CadLengthUnit::Millimeter).unwrap(),
        mesher: ExternalMesher::default(),
        options: CadMeshingOptions {
            max_element_size: Some(0.0),
            ..CadMeshingOptions::default()
        },
        output_mesh: PathBuf::from("target/bracket.msh"),
    };

    assert!(matches!(
        validate_cad_meshing_workflow(&workflow),
        Err(CadWorkflowError::InvalidMaxElementSize { value }) if value == 0.0
    ));

    workflow.options.max_element_size = Some(1.0);
    workflow.options.element_order = Some(0);
    assert_eq!(
        validate_cad_meshing_workflow(&workflow),
        Err(CadWorkflowError::InvalidElementOrder)
    );

    workflow.options.element_order = Some(1);
    workflow.output_mesh = PathBuf::from("target/bracket.vtk");
    assert!(matches!(
        validate_cad_meshing_workflow(&workflow),
        Err(CadWorkflowError::UnsupportedOutputExtension { extension }) if extension == "vtk"
    ));
}
