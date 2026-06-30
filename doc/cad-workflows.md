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

The current implementation supports command planning for Gmsh from Rust and from the CLI. It accepts CAD-like source formats that Gmsh can hand to its geometry kernels:

- STEP: `.step`, `.stp`
- IGES: `.iges`, `.igs`
- OpenCASCADE BREP: `.brep`
- STL: `.stl`

The command planner always requests Gmsh MSH 2 output with `-format msh2`, because Kepler's current importer supports ASCII Gmsh 2.x files.

## CLI Commands

Print the external meshing command without running it:

```shell
kepler cad plan --input models/bracket.step --output target/bracket.msh --dimension 3 --unit mm --max-element-size 0.25 --element-order 2 --optimize
```

Run the planned command explicitly:

```shell
kepler cad run --input models/bracket.step --output target/bracket.msh --dimension 3 --unit mm
```

Both commands support:

- `--input <path>` for `.step`, `.stp`, `.iges`, `.igs`, `.brep`, or `.stl` files.
- `--output <path.msh>` for Gmsh MSH output.
- `--dimension 2|3` for surface or volume meshing.
- `--unit mm|m` for source model unit metadata.
- `--gmsh <path>` to choose the external executable.
- `--max-element-size <h>`, `--element-order <n>`, and `--optimize` for common Gmsh controls.

`cad run` prints the command before launching the process. Kepler does not currently parse the generated mesh automatically after the external command exits.

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

The returned command is data only: the library API does not execute `gmsh`. Callers can inspect, log, sandbox, or run the command according to their environment. The CLI `cad run` command is the explicit process-execution entry point.

## Current Limitations

- No CAD kernel is linked into Kepler.
- No external process execution is performed by the library API; only the CLI `cad run` command launches a process.
- Only Gmsh command planning is supported.
- The output is constrained to `.msh` files intended for the existing Gmsh importer.
- Generated mesh files are not imported automatically after `cad run`.
- Physical group and naming quality still depend on how the source CAD model and Gmsh workflow define entities.
