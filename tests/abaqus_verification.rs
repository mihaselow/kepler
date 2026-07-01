use kepler::{
    LinearSolverOptions, SolverOptions, abaqus_to_elasticity_problem, abaqus_to_mesh_2d,
    parse_abaqus_str, read_abaqus_verify_case, solve_elasticity_with_solver,
    verify_elasticity_against_case,
};

fn run_case(inp_rel: &str) {
    let inp_path = format!("examples/data/abaqus/{inp_rel}.inp");
    let verify_path = format!("examples/data/abaqus/{inp_rel}.verify.json");
    let input = std::fs::read_to_string(&inp_path).unwrap_or_else(|_| {
        panic!("missing fixture {inp_path}");
    });
    let model = parse_abaqus_str(&input).unwrap();
    let mesh = abaqus_to_mesh_2d(&model).unwrap();
    let verify = read_abaqus_verify_case(&verify_path).unwrap();
    let problem = abaqus_to_elasticity_problem(
        &model,
        &verify.material,
        verify.thickness,
        verify.plane_stress,
    )
    .unwrap();
    let result = solve_elasticity_with_solver(
        &mesh,
        &problem,
        LinearSolverOptions::from(SolverOptions::default()),
    )
    .unwrap();
    verify_elasticity_against_case(&result, &verify).unwrap_or_else(|error| {
        panic!("verification failed for {inp_rel}: {error}");
    });
}

#[test]
fn abaqus_import_parses_block_fixture() {
    let input = include_str!("../examples/data/abaqus/block.inp");
    let model = parse_abaqus_str(input).unwrap();
    assert_eq!(model.nodes.len(), 4);
    assert_eq!(model.elements.len(), 2);
    assert!(model.materials.contains_key("STEEL"));
    assert_eq!(model.steps.len(), 1);
    assert_eq!(model.steps[0].kind, "static");

    let mesh = abaqus_to_mesh_2d(&model).unwrap();
    assert_eq!(mesh.node_count(), 4);
    assert_eq!(mesh.cells().len(), 2);
}

#[test]
fn abaqus_uniaxial_patch_matches_analytical_tension() {
    run_case("uniaxial_patch");
}

#[test]
fn abaqus_cantilever_matches_beam_theory_within_tolerance() {
    run_case("cantilever");
}

#[test]
fn abaqus_cload_and_boundary_import() {
    let input = include_str!("../examples/data/abaqus/cantilever.inp");
    let model = parse_abaqus_str(input).unwrap();
    assert_eq!(model.boundaries.len(), 6);
    assert_eq!(model.cloads.len(), 2);
    assert!((model.cloads[0].value + 500.0).abs() < f64::EPSILON);
}
