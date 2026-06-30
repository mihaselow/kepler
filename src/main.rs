use std::{env, path::PathBuf, process, process::Command as ProcessCommand};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command = match Command::parse(env::args().skip(1)) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message}");
            eprintln!();
            eprintln!("{}", usage());
            process::exit(2);
        }
    };

    match command {
        Command::Solve {
            mesh_path,
            params_path,
            output_path,
        } => solve_from_files(mesh_path, params_path, output_path)?,
        Command::ProjectValidate { project_path } => validate_project_file(project_path)?,
        Command::ProjectInspect { project_path } => inspect_project_file(project_path)?,
        Command::CadPlan { workflow } => print_cad_plan(workflow)?,
        Command::CadRun { workflow } => run_cad_plan(workflow)?,
    }

    Ok(())
}

enum Command {
    Solve {
        mesh_path: PathBuf,
        params_path: PathBuf,
        output_path: PathBuf,
    },
    ProjectValidate {
        project_path: PathBuf,
    },
    ProjectInspect {
        project_path: PathBuf,
    },
    CadPlan {
        workflow: kepler::CadMeshingWorkflow,
    },
    CadRun {
        workflow: kepler::CadMeshingWorkflow,
    },
}

impl Command {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut args = args.into_iter();
        match args.next().as_deref() {
            Some("solve") => parse_solve_args(args),
            Some("project") => parse_project_args(args),
            Some("cad") => parse_cad_args(args),
            Some("--help") | Some("-h") | None => Err(usage()),
            Some(command) => Err(format!("unknown command '{command}'")),
        }
    }
}

fn parse_cad_args(mut args: impl Iterator<Item = String>) -> Result<Command, String> {
    let action = args
        .next()
        .ok_or_else(|| "missing cad subcommand".to_owned())?;
    let mut input_path = None;
    let mut output_path = None;
    let mut unit = kepler::CadLengthUnit::Millimeter;
    let mut dimension = kepler::CadMeshingDimension::Volume3D;
    let mut gmsh_executable = "gmsh".to_owned();
    let mut max_element_size = None;
    let mut element_order = None;
    let mut optimize = false;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--input" => input_path = Some(PathBuf::from(required_value(&mut args, &flag)?)),
            "--output" => output_path = Some(PathBuf::from(required_value(&mut args, &flag)?)),
            "--unit" => unit = parse_cad_unit(&required_value(&mut args, &flag)?)?,
            "--dimension" => dimension = parse_cad_dimension(&required_value(&mut args, &flag)?)?,
            "--gmsh" => gmsh_executable = required_value(&mut args, &flag)?,
            "--max-element-size" => {
                let value = required_value(&mut args, &flag)?;
                max_element_size = Some(
                    value
                        .parse::<f64>()
                        .map_err(|_| format!("invalid value for --max-element-size '{value}'"))?,
                );
            }
            "--element-order" => {
                let value = required_value(&mut args, &flag)?;
                element_order = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid value for --element-order '{value}'"))?,
                );
            }
            "--optimize" => optimize = true,
            _ => return Err(format!("unknown cad option '{flag}'")),
        }
    }

    let input_path = input_path.ok_or_else(|| "missing required option --input".to_owned())?;
    let output_mesh = output_path.ok_or_else(|| "missing required option --output".to_owned())?;
    let source =
        kepler::CadSource::from_path(input_path, unit).map_err(|error| error.to_string())?;
    let workflow = kepler::CadMeshingWorkflow {
        source,
        mesher: kepler::ExternalMesher::Gmsh {
            executable: gmsh_executable,
        },
        options: kepler::CadMeshingOptions {
            dimension,
            max_element_size,
            element_order,
            optimize,
            output_format: kepler::CadMeshOutputFormat::GmshMsh2,
        },
        output_mesh,
    };
    kepler::validate_cad_meshing_workflow(&workflow).map_err(|error| error.to_string())?;

    match action.as_str() {
        "plan" => Ok(Command::CadPlan { workflow }),
        "run" => Ok(Command::CadRun { workflow }),
        _ => Err(format!("unknown cad subcommand '{action}'")),
    }
}

fn parse_project_args(mut args: impl Iterator<Item = String>) -> Result<Command, String> {
    let action = args
        .next()
        .ok_or_else(|| "missing project subcommand".to_owned())?;
    let mut project_path = None;

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--project" => project_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown project option '{flag}'")),
        }
    }

    let project_path =
        project_path.ok_or_else(|| "missing required option --project".to_owned())?;
    match action.as_str() {
        "validate" => Ok(Command::ProjectValidate { project_path }),
        "inspect" => Ok(Command::ProjectInspect { project_path }),
        _ => Err(format!("unknown project subcommand '{action}'")),
    }
}

fn parse_solve_args(mut args: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut mesh_path = None;
    let mut params_path = None;
    let mut output_path = None;

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;

        match flag.as_str() {
            "--mesh" => mesh_path = Some(PathBuf::from(value)),
            "--params" => params_path = Some(PathBuf::from(value)),
            "--output" => output_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown solve option '{flag}'")),
        }
    }

    Ok(Command::Solve {
        mesh_path: mesh_path.ok_or_else(|| "missing required option --mesh".to_owned())?,
        params_path: params_path.ok_or_else(|| "missing required option --params".to_owned())?,
        output_path: output_path.ok_or_else(|| "missing required option --output".to_owned())?,
    })
}

fn required_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_cad_unit(value: &str) -> Result<kepler::CadLengthUnit, String> {
    match value {
        "mm" | "millimeter" | "millimeters" => Ok(kepler::CadLengthUnit::Millimeter),
        "m" | "meter" | "meters" => Ok(kepler::CadLengthUnit::Meter),
        _ => Err(format!("unsupported CAD length unit '{value}'")),
    }
}

fn parse_cad_dimension(value: &str) -> Result<kepler::CadMeshingDimension, String> {
    match value {
        "2" | "2d" | "surface" | "surface2d" => Ok(kepler::CadMeshingDimension::Surface2D),
        "3" | "3d" | "volume" | "volume3d" => Ok(kepler::CadMeshingDimension::Volume3D),
        _ => Err(format!("unsupported CAD meshing dimension '{value}'")),
    }
}

fn solve_from_files(
    mesh_path: PathBuf,
    params_path: PathBuf,
    output_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mesh = kepler::read_mesh_file(mesh_path)?;
    let config = kepler::read_params_file(params_path)?;
    kepler::io::params::validate_params_for_mesh(&config, mesh.node_count())?;

    let source = config.source;
    let problem = kepler::PoissonProblem {
        conductivity: config.conductivity,
        source: move |x, y| source.value_at(x, y),
        dirichlet: config.dirichlet,
    };

    let result = kepler::solve_poisson_with_solver(&mesh, &problem, config.solver_options)?;
    let compatibility_result = kepler::PoissonResult::from(result.clone());
    kepler::write_solution_file(&output_path, &compatibility_result)?;

    println!(
        "wrote {} values to {}",
        result.values.len(),
        output_path.display()
    );
    println!(
        "solver {:?} with {:?}: {} iterations, residual {}",
        result.diagnostics.backend,
        result.diagnostics.preconditioner,
        result.diagnostics.iterations,
        result.diagnostics.residual_norm
    );
    if !result.diagnostics.residual_history.is_empty() {
        println!(
            "residual history: {:?}",
            result.diagnostics.residual_history
        );
    }

    Ok(())
}

fn validate_project_file(project_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let project = kepler::read_project_file(&project_path)?;
    kepler::validate_project(&project)?;
    println!(
        "project {} is valid (schema_version {}, {} job{})",
        project_path.display(),
        project.schema_version,
        project.jobs.len(),
        plural_suffix(project.jobs.len()),
    );
    Ok(())
}

fn inspect_project_file(project_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let project = kepler::read_project_file(&project_path)?;
    kepler::validate_project(&project)?;
    println!("project: {}", project_path.display());
    if let Some(name) = &project.name {
        println!("name: {name}");
    }
    println!("schema_version: {}", project.schema_version);
    println!("jobs: {}", project.jobs.len());
    for job in &project.jobs {
        let physics = match &job.physics {
            kepler::ProjectPhysics::Poisson(_) => "poisson",
        };
        println!(
            "- {}: physics={}, points={}, triangles={}",
            job.id,
            physics,
            job.mesh.points.len(),
            job.mesh.triangles.len()
        );
    }
    Ok(())
}

fn print_cad_plan(workflow: kepler::CadMeshingWorkflow) -> Result<(), Box<dyn std::error::Error>> {
    let command = kepler::plan_cad_meshing_command(&workflow)?;
    println!("{}", format_external_command(&command));
    Ok(())
}

fn run_cad_plan(workflow: kepler::CadMeshingWorkflow) -> Result<(), Box<dyn std::error::Error>> {
    let command = kepler::plan_cad_meshing_command(&workflow)?;
    println!("{}", format_external_command(&command));
    let status = ProcessCommand::new(&command.program)
        .args(&command.args)
        .status()?;
    if !status.success() {
        return Err(format!("CAD mesher exited with status {status}").into());
    }
    Ok(())
}

fn format_external_command(command: &kepler::ExternalCommand) -> String {
    std::iter::once(command.program.as_str())
        .chain(command.args.iter().map(String::as_str))
        .map(shell_quote)
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '='))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn plural_suffix(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

fn usage() -> String {
    [
        "usage:",
        "  kepler solve --mesh <path.mesh> --params <path.params> --output <path.solution>",
        "  kepler project validate --project <path.project.json>",
        "  kepler project inspect --project <path.project.json>",
        "  kepler cad plan --input <model.step> --output <mesh.msh> [--dimension 2|3] [--unit mm|m] [--gmsh <path>] [--max-element-size <h>] [--element-order <n>] [--optimize]",
        "  kepler cad run --input <model.step> --output <mesh.msh> [--dimension 2|3] [--unit mm|m] [--gmsh <path>] [--max-element-size <h>] [--element-order <n>] [--optimize]",
    ]
    .join("\n")
}
