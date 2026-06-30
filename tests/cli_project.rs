use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const PROJECT_JSON: &str = r#"
{
  "schema_version": 1,
  "name": "cli project",
  "jobs": [
    {
      "id": "solve-triangle",
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
          "record_residual_history": false
        }
      },
      "output": { "format": "solution" }
    }
  ]
}
"#;

const LEGACY_MESH: &str = r#"
nodes
0 0.0 0.0
1 1.0 0.0
2 0.0 1.0

triangles
0 0 1 2
"#;

const LEGACY_PARAMS: &str = r#"
conductivity 1.0
source constant 1.0
solver backend dense_direct

dirichlet
0 0.0
2 0.0
"#;

#[test]
fn cli_validates_project_file() {
    let dir = temp_test_dir("validate");
    let project_path = write_file(&dir, "project.json", PROJECT_JSON);

    let output = Command::new(env!("CARGO_BIN_EXE_kepler"))
        .args([
            "project",
            "validate",
            "--project",
            project_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("schema_version 1"));
    assert!(stdout.contains("1 job"));
    remove_dir(&dir);
}

#[test]
fn cli_inspects_project_file() {
    let dir = temp_test_dir("inspect");
    let project_path = write_file(&dir, "project.json", PROJECT_JSON);

    let output = Command::new(env!("CARGO_BIN_EXE_kepler"))
        .args([
            "project",
            "inspect",
            "--project",
            project_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("name: cli project"));
    assert!(stdout.contains("- solve-triangle: physics=poisson, points=3, triangles=1"));
    remove_dir(&dir);
}

#[test]
fn cli_rejects_invalid_project_file() {
    let dir = temp_test_dir("invalid");
    let project_path = write_file(
        &dir,
        "project.json",
        r#"{ "schema_version": 99, "jobs": [] }"#,
    );

    let output = Command::new(env!("CARGO_BIN_EXE_kepler"))
        .args([
            "project",
            "validate",
            "--project",
            project_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("UnsupportedSchemaVersion") || stderr.contains("schema version 99"));
    remove_dir(&dir);
}

#[test]
fn cli_preserves_legacy_solve_command() {
    let dir = temp_test_dir("legacy");
    let mesh_path = write_file(&dir, "mesh.mesh", LEGACY_MESH);
    let params_path = write_file(&dir, "params.params", LEGACY_PARAMS);
    let output_path = dir.join("out.solution");

    let output = Command::new(env!("CARGO_BIN_EXE_kepler"))
        .args([
            "solve",
            "--mesh",
            mesh_path.to_str().unwrap(),
            "--params",
            params_path.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let solution = fs::read_to_string(&output_path).unwrap();
    assert!(solution.contains("node value"));
    remove_dir(&dir);
}

fn temp_test_dir(name: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("kepler-cli-{name}-{}-{unique}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_file(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).unwrap();
    path
}

fn remove_dir(dir: &Path) {
    fs::remove_dir_all(dir).unwrap();
}
