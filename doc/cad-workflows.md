# CAD To Mesh Workflows

Kepler does not include a native CAD kernel. CAD support starts with an external workflow boundary: describe the CAD source, meshing options, and output mesh, then build a command plan for a supported external mesher.

## Roadmap Sub-Steps

The CAD workflow roadmap item is split into sub-steps:

- Add a validated workflow model for external CAD meshing and Gmsh command planning.
- Add CLI commands that can print or run a CAD meshing command plan.
- Add REST/project workflow support for CAD source and mesh artifacts.
- Add fixture coverage for CAD command planning and imported Gmsh output.
- Document installation expectations and limitations for external CAD/meshing tools.

The parent roadmap item should remain in progress until all of these sub-steps are implemented, documented, and verified.

## Current Scope

The first implementation supports command planning for Gmsh. It accepts CAD-like source formats that Gmsh can hand to its geometry kernels:

- STEP: `.step`, `.stp`
- IGES: `.iges`, `.igs`
- OpenCASCADE BREP: `.brep`
- STL: `.stl`

The command planner always requests Gmsh MSH 2 output with `-format msh2`, because Kepler's current importer supports ASCII Gmsh 2.x files.

## Rust API

Build a workflow, validate it, then turn it into an external command:

```rust
use std::path::PathBuf;

use kepler::{
    CadLengthUnit, CadMeshOutputFormat, CadMeshingDimension, CadMeshingOptions,
    CadMeshingWorkflow, CadSource, ExternalMesher, plan_cad_meshing_command,
};

let workflow = CadMeshingWorkflow {
    source: CadSource::from_path("models/bracket.step", CadLengthUnit::Millimeter)?,
    mesher: ExternalMesher::default(),
    options: CadMeshingOptions {
        dimension: CadMeshingDimension::Volume3D,
        max_element_size: Some(0.25),
        element_order: Some(2),
        optimize: true,
        output_format: CadMeshOutputFormat::GmshMsh2,
    },
    output_mesh: PathBuf::from("target/bracket.msh"),
};

let command = plan_cad_meshing_command(&workflow)?;
assert_eq!(command.program, "gmsh");
```

The returned command is data only: Kepler does not execute `gmsh` in this API. Callers can inspect, log, sandbox, or run the command according to their environment.

## Current Limitations

- No CAD kernel is linked into Kepler.
- No external process execution is performed by the library API.
- Only Gmsh command planning is supported.
- The output is constrained to `.msh` files intended for the existing Gmsh importer.
- Physical group and naming quality still depend on how the source CAD model and Gmsh workflow define entities.
