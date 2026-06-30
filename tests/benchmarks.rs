use std::time::{Duration, Instant};

use kepler::{
    ImportedMesh, LinearSolverBackend, LinearSolverOptions, Mesh, Point2, PoissonProblem, Tri3,
    VtkScalarField, fem::poisson::assemble_poisson_system, format_vtk_legacy, job_to_poisson,
    parse_gmsh_str, parse_project_str, solve_poisson_with_solver, validate_project,
};

const GMSH_2D: &str = include_str!("../examples/data/physical_groups_2d.msh");
const SQUARE_PROJECT: &str = include_str!("../examples/data/square.project.json");

#[test]
#[ignore = "benchmark-style verification; run with `cargo test --test benchmarks -- --ignored --nocapture`"]
fn benchmark_poisson_assembly() {
    let mesh = structured_tri_mesh(16);
    let problem = poisson_problem(&mesh);

    let elapsed = repeat("poisson assembly", 50, || {
        let (matrix, rhs) = assemble_poisson_system(&mesh, &problem).unwrap();
        assert_eq!(matrix.rows(), mesh.node_count());
        assert_eq!(rhs.len(), mesh.node_count());
    });

    assert_reasonable(elapsed);
}

#[test]
#[ignore = "benchmark-style verification; run with `cargo test --test benchmarks -- --ignored --nocapture`"]
fn benchmark_poisson_solve() {
    let mesh = structured_tri_mesh(10);
    let problem = poisson_problem(&mesh);
    let options = LinearSolverOptions {
        backend: LinearSolverBackend::DenseDirect,
        ..LinearSolverOptions::default()
    };

    let elapsed = repeat("poisson solve", 15, || {
        let result = solve_poisson_with_solver(&mesh, &problem, options.clone()).unwrap();
        assert_eq!(result.values.len(), mesh.node_count());
        assert!(result.diagnostics.converged);
    });

    assert_reasonable(elapsed);
}

#[test]
#[ignore = "benchmark-style verification; run with `cargo test --test benchmarks -- --ignored --nocapture`"]
fn benchmark_import_export_round_trip() {
    let ImportedMesh::TwoD(topology) = parse_gmsh_str(GMSH_2D).unwrap() else {
        panic!("expected a 2D topology");
    };

    let elapsed = repeat("gmsh import plus vtk export", 100, || {
        let imported = parse_gmsh_str(GMSH_2D).unwrap();
        let ImportedMesh::TwoD(imported_topology) = imported else {
            panic!("expected a 2D topology");
        };
        let output = format_vtk_legacy(
            &imported_topology,
            &[VtkScalarField::new("temperature", vec![0.0, 0.5, 1.0])],
        )
        .unwrap();
        assert!(output.contains("UNSTRUCTURED_GRID"));
    });

    assert_eq!(topology.points().len(), 3);
    assert_reasonable(elapsed);
}

#[test]
#[ignore = "benchmark-style verification; run with `cargo test --test benchmarks -- --ignored --nocapture`"]
fn benchmark_project_parse_validate_and_adapt() {
    let elapsed = repeat("project parse validate adapt", 100, || {
        let project = parse_project_str(SQUARE_PROJECT).unwrap();
        validate_project(&project).unwrap();
        let (mesh, config) = job_to_poisson(&project.jobs[0]).unwrap();
        assert_eq!(mesh.node_count(), 5);
        assert_eq!(config.dirichlet.len(), 4);
    });

    assert_reasonable(elapsed);
}

fn repeat(label: &str, iterations: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    eprintln!("{label}: {iterations} iterations in {elapsed:?}");
    elapsed
}

fn structured_tri_mesh(divisions: usize) -> Mesh {
    let stride = divisions + 1;
    let mut points = Vec::with_capacity(stride * stride);
    for y in 0..=divisions {
        for x in 0..=divisions {
            points.push(Point2::new(
                x as f64 / divisions as f64,
                y as f64 / divisions as f64,
            ));
        }
    }

    let mut triangles = Vec::with_capacity(divisions * divisions * 2);
    for y in 0..divisions {
        for x in 0..divisions {
            let lower_left = y * stride + x;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + stride;
            let upper_right = upper_left + 1;
            triangles.push(Tri3::new([lower_left, lower_right, upper_right]));
            triangles.push(Tri3::new([lower_left, upper_right, upper_left]));
        }
    }

    Mesh::new(points, triangles).unwrap()
}

fn poisson_problem(mesh: &Mesh) -> PoissonProblem<impl Fn(f64, f64) -> f64> {
    let mut dirichlet = Vec::new();
    for (node, point) in mesh.points().iter().enumerate() {
        if point.x == 0.0 || point.x == 1.0 || point.y == 0.0 || point.y == 1.0 {
            dirichlet.push((node, 0.0));
        }
    }

    PoissonProblem {
        conductivity: 1.0,
        source: |_, _| 1.0,
        dirichlet,
    }
}

fn assert_reasonable(elapsed: Duration) {
    assert!(
        elapsed > Duration::ZERO,
        "benchmark-style test should measure nonzero elapsed time",
    );
}
