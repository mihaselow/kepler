use kepler::{
    FileIoError, LinearSolverBackend, Mesh, PROJECT_SCHEMA_VERSION, Point2, PreconditionerKind,
    ProjectError, ProjectFile, ProjectLinearSolverBackend, ProjectOutputFormat, Tri3,
    format_project, job_to_poisson, parse_params_str, parse_project_str, validate_project,
};

const PROJECT_JSON: &str = r#"
{
  "schema_version": 1,
  "name": "square poisson",
  "jobs": [
    {
      "id": "solve-square",
      "mesh": {
        "points": [
          { "x": 0.0, "y": 0.0 },
          { "x": 1.0, "y": 0.0 },
          { "x": 0.0, "y": 1.0 }
        ],
        "triangles": [
          { "nodes": [0, 1, 2] }
        ]
      },
      "physics": {
        "kind": "poisson",
        "conductivity": 1.0,
        "source": { "kind": "constant", "value": 1.0 },
        "dirichlet": [
          { "node": 0, "value": 0.0 },
          { "node": 2, "value": 0.0 }
        ],
        "solver_options": {
          "max_iterations": 50,
          "tolerance": 1e-9,
          "backend": "dense_direct",
          "preconditioner": "none",
          "record_residual_history": true
        }
      },
      "output": { "format": "solution" }
    }
  ]
}
"#;

#[test]
fn parses_versioned_poisson_project() {
    let project = parse_project_str(PROJECT_JSON).unwrap();

    assert_eq!(project.schema_version, PROJECT_SCHEMA_VERSION);
    assert_eq!(project.jobs[0].id, "solve-square");
    assert_eq!(
        project.jobs[0].output.as_ref().unwrap().format,
        ProjectOutputFormat::Solution
    );

    let (mesh, config) = job_to_poisson(&project.jobs[0]).unwrap();
    assert_eq!(mesh.node_count(), 3);
    assert_eq!(
        config.solver_options.backend,
        LinearSolverBackend::DenseDirect
    );
    assert_eq!(
        config.solver_options.preconditioner,
        PreconditionerKind::None
    );
    assert!(config.solver_options.record_residual_history);
}

#[test]
fn project_round_trips_from_legacy_mesh_and_params() {
    let mesh = Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ],
        vec![Tri3::new([0, 1, 2])],
    )
    .unwrap();
    let params = parse_params_str(
        r#"
conductivity 2.0
source constant 3.0
solver backend gmres
dirichlet
0 0.0
"#,
    )
    .unwrap();

    let project = ProjectFile::from_legacy_poisson("legacy-solve", &mesh, params);
    validate_project(&project).unwrap();
    let serialized = format_project(&project).unwrap();
    let parsed = parse_project_str(&serialized).unwrap();
    let (_, config) = job_to_poisson(&parsed.jobs[0]).unwrap();

    assert_eq!(parsed.jobs[0].id, "legacy-solve");
    assert_eq!(config.conductivity, 2.0);
    assert_eq!(config.solver_options.backend, LinearSolverBackend::Gmres);
}

#[test]
fn project_rejects_unsupported_schema_version() {
    let error = parse_project_str(
        r#"
{
  "schema_version": 99,
  "jobs": []
}
"#,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        ProjectError::UnsupportedSchemaVersion {
            version: 99,
            expected: PROJECT_SCHEMA_VERSION
        }
    ));
}

#[test]
fn project_rejects_duplicate_job_ids() {
    let mut project = parse_project_str(PROJECT_JSON).unwrap();
    project.jobs.push(project.jobs[0].clone());

    let error = validate_project(&project).unwrap_err();

    assert!(matches!(error, ProjectError::DuplicateJobId { .. }));
}

#[test]
fn project_rejects_out_of_bounds_dirichlet_nodes() {
    let mut project = parse_project_str(PROJECT_JSON).unwrap();
    let kepler::ProjectPhysics::Poisson(problem) = &mut project.jobs[0].physics;
    problem.dirichlet.push(kepler::ProjectDirichlet {
        node: 10,
        value: 0.0,
    });

    let error = validate_project(&project).unwrap_err();

    assert!(matches!(
        error,
        ProjectError::Params {
            source: FileIoError::BoundaryNodeOutOfBounds {
                node_id: 10,
                node_count: 3
            },
            ..
        }
    ));
}

#[test]
fn project_solver_options_have_schema_defaults() {
    let options = kepler::default_project_solver_options();

    assert_eq!(
        options.backend,
        ProjectLinearSolverBackend::ConjugateGradient
    );
    assert_eq!(
        options.max_iterations,
        kepler::SolverOptions::default().max_iterations
    );
}
