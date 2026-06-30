use std::{env, path::PathBuf, process};

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
}

impl Command {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut args = args.into_iter();
        match args.next().as_deref() {
            Some("solve") => parse_solve_args(args),
            Some("project") => parse_project_args(args),
            Some("--help") | Some("-h") | None => Err(usage()),
            Some(command) => Err(format!("unknown command '{command}'")),
        }
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

fn plural_suffix(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

fn usage() -> String {
    [
        "usage:",
        "  kepler solve --mesh <path.mesh> --params <path.params> --output <path.solution>",
        "  kepler project validate --project <path.project.json>",
        "  kepler project inspect --project <path.project.json>",
    ]
    .join("\n")
}
