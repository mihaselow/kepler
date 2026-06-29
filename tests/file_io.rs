use kepler::{
    FileIoError, PoissonProblem, SolverOptions, format_solution, parse_mesh_str, parse_params_str,
    solve_poisson,
};

const EPS: f64 = 1.0e-12;
const SQUARE_MESH: &str = include_str!("../examples/data/square.mesh");
const SQUARE_PARAMS: &str = include_str!("../examples/data/square.params");

#[test]
fn parses_square_mesh_file_format() {
    let mesh = parse_mesh_str(SQUARE_MESH).unwrap();

    assert_eq!(mesh.node_count(), 5);
    assert_eq!(mesh.triangles().len(), 4);
}

#[test]
fn mesh_parser_rejects_duplicate_node_ids() {
    let error = parse_mesh_str(
        r#"
nodes
0 0.0 0.0
0 1.0 0.0

triangles
0 0 1 2
"#,
    )
    .unwrap_err();

    assert!(matches!(error, FileIoError::DuplicateId { id: 0, .. }));
}

#[test]
fn mesh_parser_rejects_non_contiguous_node_ids() {
    let error = parse_mesh_str(
        r#"
nodes
0 0.0 0.0
2 1.0 0.0
3 0.0 1.0

triangles
0 0 2 3
"#,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        FileIoError::NonContiguousId {
            expected: 1,
            found: 2
        }
    ));
}

#[test]
fn parses_params_file_format() {
    let config = parse_params_str(SQUARE_PARAMS).unwrap();

    assert_close(config.conductivity, 1.0);
    assert_eq!(config.dirichlet.len(), 4);
    assert_eq!(config.solver_options, SolverOptions::default());
}

#[test]
fn params_parser_rejects_unsupported_source() {
    let error = parse_params_str(
        r#"
conductivity 1.0
source expression x
"#,
    )
    .unwrap_err();

    assert!(matches!(error, FileIoError::UnsupportedSource { .. }));
}

#[test]
fn params_parser_rejects_duplicate_dirichlet_nodes() {
    let error = parse_params_str(
        r#"
conductivity 1.0
source constant 1.0

dirichlet
0 0.0
0 1.0
"#,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        FileIoError::DuplicateDirichlet { node_id: 0, .. }
    ));
}

#[test]
fn solution_format_includes_diagnostics_and_ordered_values() {
    let output = format_solution(&kepler::PoissonResult {
        values: vec![0.0, 0.5],
        iterations: 3,
        residual_norm: 1.0e-9,
    });

    assert!(output.contains("# iterations 3"));
    assert!(output.contains("# residual_norm 0.000000001"));
    assert!(output.contains("node value\n0 0\n1 0.5\n"));
}

#[test]
fn file_driven_square_solve_matches_programmatic_reference() {
    let mesh = parse_mesh_str(SQUARE_MESH).unwrap();
    let config = parse_params_str(SQUARE_PARAMS).unwrap();
    kepler::io::params::validate_params_for_mesh(&config, mesh.node_count()).unwrap();
    let source = config.source;
    let problem = PoissonProblem {
        conductivity: config.conductivity,
        source: move |x, y| source.value_at(x, y),
        dirichlet: config.dirichlet,
    };

    let result = solve_poisson(&mesh, &problem, config.solver_options).unwrap();

    assert_close(result.values[4], 1.0 / 12.0);
}

#[test]
fn params_validation_rejects_missing_boundary_nodes() {
    let config = parse_params_str(
        r#"
conductivity 1.0
source constant 1.0

dirichlet
5 0.0
"#,
    )
    .unwrap();

    let error = kepler::io::params::validate_params_for_mesh(&config, 2).unwrap_err();

    assert!(matches!(
        error,
        FileIoError::BoundaryNodeOutOfBounds {
            node_id: 5,
            node_count: 2
        }
    ));
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS,
        "expected {actual} to be within {EPS} of {expected}",
    );
}
