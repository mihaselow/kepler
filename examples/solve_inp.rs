//! Load an Abaqus `.inp` benchmark, solve 2D plane elasticity, and verify against
//! the companion `*.verify.json` analytical reference.
//!
//! ```bash
//! cargo run --example solve_inp -- examples/data/abaqus/uniaxial_patch.inp
//! cargo run --example solve_inp -- examples/data/abaqus/cantilever.inp output.vtk
//! ```

use std::{env, path::Path, process};

use kepler::{
    LinearSolverOptions, SolverOptions, VtkScalarField, abaqus_to_elasticity_problem,
    abaqus_to_mesh_2d, read_abaqus_file, read_abaqus_verify_case, solve_elasticity_with_solver,
    verify_elasticity_against_case, write_vtk_legacy_file,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example solve_inp -- <model.inp> [output.vtk]");
        process::exit(1);
    }

    let inp_path = Path::new(&args[1]);
    let verify_path = inp_path.with_extension("verify.json");
    let vtk_path = args.get(2).map(String::as_str);

    println!("Reading Abaqus model: {}", inp_path.display());
    let model = read_abaqus_file(inp_path)?;
    let mesh = abaqus_to_mesh_2d(&model)?;
    println!(
        "  {} nodes, {} elements ({} materials, {} BCs, {} loads)",
        mesh.node_count(),
        mesh.cells().len(),
        model.materials.len(),
        model.boundaries.len(),
        model.cloads.len()
    );

    let verify = read_abaqus_verify_case(&verify_path)?;
    println!("Verification case: {}", verify.description);

    let problem = abaqus_to_elasticity_problem(
        &model,
        &verify.material,
        verify.thickness,
        verify.plane_stress,
    )?;

    let result = solve_elasticity_with_solver(
        &mesh,
        &problem,
        LinearSolverOptions::from(SolverOptions::default()),
    )?;
    verify_elasticity_against_case(&result, &verify)?;

    println!("PASS — all checks within tolerance.");
    println!(
        "  solver iterations: {}, residual: {:.3e}",
        result.diagnostics.iterations, result.diagnostics.residual_norm
    );

    for check in &verify.checks {
        match check {
            kepler::AbaqusVerifyCheck::DisplacementComponent {
                node,
                component,
                expected,
                ..
            } => {
                let actual = if component.eq_ignore_ascii_case("y") {
                    result.displacements[*node][1]
                } else {
                    result.displacements[*node][0]
                };
                println!("  u_{component}[{node}] = {actual:.6e} (ref {expected:.6e})");
            }
            kepler::AbaqusVerifyCheck::StressComponent {
                element,
                component,
                expected,
                ..
            } => {
                let stress = &result.element_stress[*element];
                let actual = match component.to_lowercase().as_str() {
                    "sigma_yy" => stress.sigma_yy,
                    "sigma_xy" => stress.sigma_xy,
                    "von_mises" => stress.von_mises,
                    _ => stress.sigma_xx,
                };
                println!("  {component}[{element}] = {actual:.6e} (ref {expected:.6e})");
            }
        }
    }

    if let Some(path) = vtk_path {
        let topology = mesh.topology()?;
        let ux: Vec<f64> = result.displacements.iter().map(|d| d[0]).collect();
        let uy: Vec<f64> = result.displacements.iter().map(|d| d[1]).collect();
        let von_mises: Vec<f64> = result
            .nodal_stress
            .iter()
            .map(|stress| stress.von_mises)
            .collect();
        let scalar_fields = [
            VtkScalarField::new("ux", ux),
            VtkScalarField::new("uy", uy),
            VtkScalarField::new("von_mises", von_mises),
        ];
        write_vtk_legacy_file(path, &topology, &scalar_fields)?;
        println!("Wrote VTK: {path}");
    }

    Ok(())
}
