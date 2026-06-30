use std::process::Command;

#[test]
fn cli_prints_cad_meshing_plan() {
    let output = Command::new(env!("CARGO_BIN_EXE_kepler"))
        .args([
            "cad",
            "plan",
            "--input",
            "models/bracket.step",
            "--output",
            "target/bracket.msh",
            "--dimension",
            "3",
            "--unit",
            "mm",
            "--max-element-size",
            "0.25",
            "--element-order",
            "2",
            "--optimize",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "gmsh models/bracket.step -3 -format msh2 -o target/bracket.msh -clmax 0.25 -order 2 -optimize\n"
    );
}

#[test]
fn cli_runs_cad_meshing_plan_with_explicit_executable() {
    let output = Command::new(env!("CARGO_BIN_EXE_kepler"))
        .args([
            "cad",
            "run",
            "--input",
            "models/shell.igs",
            "--output",
            "target/shell.msh",
            "--dimension",
            "2d",
            "--unit",
            "m",
            "--gmsh",
            "true",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "true models/shell.igs -2 -format msh2 -o target/shell.msh\n"
    );
}

#[test]
fn cli_rejects_unsupported_cad_input_extension() {
    let output = Command::new(env!("CARGO_BIN_EXE_kepler"))
        .args([
            "cad",
            "plan",
            "--input",
            "models/profile.dxf",
            "--output",
            "target/profile.msh",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unsupported CAD file extension 'dxf'"));
}
