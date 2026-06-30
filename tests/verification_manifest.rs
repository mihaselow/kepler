use std::{fs, path::Path};

const MANIFEST: &str = include_str!("../doc/verification.md");

const REQUIRED_COMMANDS: &[&str] = &[
    "cargo fmt",
    "cargo test",
    "cargo clippy --all-targets --all-features",
    "cargo build --bin server",
];

const REQUIRED_TESTS: &[&str] = &[
    "tests/poisson.rs",
    "tests/poisson_3d.rs",
    "tests/mesh_topology.rs",
    "tests/conditions.rs",
    "tests/annotations.rs",
    "tests/heat.rs",
    "tests/diffusion_reaction.rs",
    "tests/electrostatics.rs",
    "tests/manufactured_scalar.rs",
    "tests/elasticity.rs",
    "tests/elasticity_3d.rs",
    "tests/modal.rs",
    "tests/structural_verification.rs",
    "tests/solver_stack.rs",
    "tests/physics_solver_stack.rs",
    "tests/transient_coverage.rs",
    "tests/file_io.rs",
    "tests/mesh_import_export.rs",
    "tests/project_workflow.rs",
    "tests/cli_project.rs",
    "tests/benchmarks.rs",
    "tests/verification_manifest.rs",
    "src/bin/server.rs",
];

const REQUIRED_FIXTURES: &[&str] = &[
    "examples/data/square.mesh",
    "examples/data/square.params",
    "examples/data/square.project.json",
    "examples/data/physical_groups_2d.msh",
    "examples/data/physical_groups_2d_temperature.vtk",
    "examples/data/two_node.solution",
    "examples/data/cli_project_inspect_summary.txt",
    "examples/data/rest_project_request.json",
    "examples/data/rest_project_validate_response.json",
    "examples/data/rest_project_solve_response.json",
    "examples/data/rest_bad_schema_error_response.json",
    "examples/data/rest_mesh_artifact_upload.json",
];

#[test]
fn verification_manifest_references_existing_checks_and_fixtures() {
    for command in REQUIRED_COMMANDS {
        assert!(
            MANIFEST.contains(command),
            "verification manifest should mention required command `{command}`",
        );
    }

    for path in REQUIRED_TESTS.iter().chain(REQUIRED_FIXTURES) {
        assert!(
            Path::new(path).exists(),
            "verification manifest references missing path `{path}`",
        );
        assert!(
            MANIFEST.contains(path),
            "verification manifest should mention `{path}`",
        );
    }
}

#[test]
fn verification_manifest_documents_known_gaps() {
    for gap in [
        "Manufactured-solution",
        "Benchmarks",
        "CAD import",
        "in-memory only",
    ] {
        assert!(
            MANIFEST.contains(gap),
            "verification manifest should document gap `{gap}`",
        );
    }
}

#[test]
fn verification_manifest_documents_quality_gate_guidance() {
    for guidance in [
        "Local Workflow Guidance",
        "CI Guidance",
        "Golden fixture updates should be deliberate",
        "tests/verification_manifest.rs",
        "cargo test --test benchmarks -- --ignored --nocapture",
        "cargo test --bin server -- --ignored --nocapture",
    ] {
        assert!(
            MANIFEST.contains(guidance),
            "verification manifest should document quality-gate guidance `{guidance}`",
        );
    }
}

#[test]
fn verification_manifest_lists_all_integration_test_files() {
    let mut missing = Vec::new();
    for entry in fs::read_dir("tests").unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|value| value.to_str()) != Some("rs") {
            continue;
        }
        let normalized = path.to_string_lossy().replace('\\', "/");
        if !MANIFEST.contains(&normalized) {
            missing.push(normalized);
        }
    }

    assert!(
        missing.is_empty(),
        "verification manifest is missing integration test files: {missing:?}",
    );
}
