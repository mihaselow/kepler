use kepler::{
    read_gmsh_file, write_vtk_legacy_file, ElementKind, ImportedMesh, Mesh, MeshTopology,
    Point2, SolverOptions, VtkScalarField,
    // Poisson
    PoissonProblem, PoissonProblem3D, solve_poisson, solve_poisson_3d,
    // Elasticity
    ElasticityProblem, ElasticityProblem3D, ElasticityMaterial, ElasticityMaterial3D,
    ElasticityModel, DisplacementComponent, DisplacementComponent3D, DisplacementConstraint,
    DisplacementConstraint3D, NodalForce, NodalForce3D, solve_elasticity, solve_elasticity_3d,
    // Modal
    ModalProblem, ModalProblem3D, solve_modal, solve_modal_3d,
};
use std::collections::BTreeSet;
use std::env;
use std::process;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: cargo run --example solve_gmsh <mesh.msh> <poisson|elasticity|modal> <output.vtk>");
        process::exit(1);
    }

    let mesh_path = &args[1];
    let physics = args[2].to_lowercase();
    let output_path = &args[3];

    println!("Reading Gmsh file: {mesh_path}...");
    let imported = read_gmsh_file(mesh_path)?;

    match imported {
        ImportedMesh::TwoD(ref topology) => {
            println!(
                "Loaded 2D Gmsh mesh with {} points and {} cells",
                topology.points().len(),
                topology.cells().len()
            );
            solve_2d(topology, &physics, output_path)?;
        }
        ImportedMesh::ThreeD(ref topology) => {
            println!(
                "Loaded 3D Gmsh mesh with {} points and {} cells",
                topology.points().len(),
                topology.cells().len()
            );
            solve_3d(topology, &physics, output_path)?;
        }
    }

    Ok(())
}

fn solve_2d(
    topology: &MeshTopology<2>,
    physics: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Convert MeshTopology<2> to Kepler's 2D Mesh struct
    let points: Vec<Point2> = topology
        .points()
        .iter()
        .map(|pt| Point2::new(pt.coords[0], pt.coords[1]))
        .collect();

    let mut triangles = Vec::new();
    for cell in topology.cells() {
        if cell.kind == ElementKind::Tri3 {
            triangles.push(kepler::Tri3::new([cell.nodes[0], cell.nodes[1], cell.nodes[2]]));
        }
    }

    if triangles.is_empty() {
        return Err("No Tri3 elements found in the 2D mesh (Kepler's 2D solvers only support Tri3)".into());
    }

    let mesh = Mesh::new(points, triangles)?;

    // Identify boundary nodes from region named "boundary_left"
    let boundary_nodes = get_boundary_nodes(topology, "boundary_left");
    println!("Found {} nodes on boundary_left", boundary_nodes.len());

    let mut scalar_fields = Vec::new();

    match physics {
        "poisson" => {
            println!("Setting up 2D Poisson problem...");
            let problem = PoissonProblem {
                conductivity: 1.0,
                source: |_, _| 100.0, // Constant internal heat/charge source
                dirichlet: boundary_nodes.iter().map(|&node| (node, 0.0)).collect(),
            };

            println!("Solving...");
            let result = solve_poisson(&mesh, &problem, SolverOptions::default())?;
            println!(
                "Poisson solve converged. Residual norm: {}",
                result.residual_norm
            );

            scalar_fields.push(VtkScalarField::new("u", result.values));
        }
        "elasticity" => {
            println!("Setting up 2D Elasticity problem (cantilever beam bending)...");
            // Find rightmost node to apply point load
            let rightmost_node = find_rightmost_node(topology);

            let problem = ElasticityProblem {
                material: ElasticityMaterial {
                    young_modulus: 2e11, // Steel Young's Modulus
                    poisson_ratio: 0.3,
                    model: ElasticityModel::PlaneStress,
                },
                thickness: 0.1,
                constraints: boundary_nodes
                    .iter()
                    .flat_map(|&node| {
                        vec![
                            DisplacementConstraint {
                                node,
                                component: DisplacementComponent::X,
                                value: 0.0,
                            },
                            DisplacementConstraint {
                                node,
                                component: DisplacementComponent::Y,
                                value: 0.0,
                            },
                        ]
                    })
                    .collect(),
                forces: vec![NodalForce {
                    node: rightmost_node,
                    fx: 0.0,
                    fy: -1000.0, // Downward point force
                }],
            };

            println!("Solving...");
            let result = solve_elasticity(&mesh, &problem, SolverOptions::default())?;
            println!(
                "Elasticity solve completed. Residual norm: {}",
                result.residual_norm
            );

            let ux: Vec<f64> = result.displacements.iter().map(|d| d[0]).collect();
            let uy: Vec<f64> = result.displacements.iter().map(|d| d[1]).collect();
            let mag: Vec<f64> = result
                .displacements
                .iter()
                .map(|d| (d[0].powi(2) + d[1].powi(2)).sqrt())
                .collect();

            scalar_fields.push(VtkScalarField::new("ux", ux));
            scalar_fields.push(VtkScalarField::new("uy", uy));
            scalar_fields.push(VtkScalarField::new("displacement_magnitude", mag));
        }
        "modal" => {
            println!("Setting up 2D Modal Analysis...");
            let problem = ModalProblem {
                elasticity: ElasticityProblem {
                    material: ElasticityMaterial {
                        young_modulus: 2e11,
                        poisson_ratio: 0.3,
                        model: ElasticityModel::PlaneStress,
                    },
                    thickness: 0.1,
                    constraints: boundary_nodes
                        .iter()
                        .flat_map(|&node| {
                            vec![
                                DisplacementConstraint {
                                    node,
                                    component: DisplacementComponent::X,
                                    value: 0.0,
                                },
                                DisplacementConstraint {
                                    node,
                                    component: DisplacementComponent::Y,
                                    value: 0.0,
                                },
                            ]
                        })
                        .collect(),
                    forces: vec![],
                },
                density: 7850.0, // Steel density
                mode_count: 3,
            };

            println!("Solving...");
            let result = solve_modal(&mesh, &problem)?;
            println!("Modal solve completed successfully.");

            for (i, mode) in result.modes.iter().enumerate() {
                println!(
                    "  Mode {}: Frequency = {:.2} Hz (Angular = {:.2} rad/s)",
                    i + 1,
                    mode.frequency_hz,
                    mode.angular_frequency
                );

                let mag: Vec<f64> = mode
                    .displacements
                    .iter()
                    .map(|d| (d[0].powi(2) + d[1].powi(2)).sqrt())
                    .collect();
                scalar_fields.push(VtkScalarField::new(format!("mode_{}_amplitude", i + 1), mag));
            }
        }
        _ => return Err(format!("Unsupported physics model: {physics}").into()),
    }

    println!("Writing VTK file to {output_path}...");
    write_vtk_legacy_file(output_path, topology, &scalar_fields)?;
    println!("Done!");

    Ok(())
}

fn solve_3d(
    topology: &MeshTopology<3>,
    physics: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Identify boundary nodes from region named "boundary_left"
    let boundary_nodes = get_boundary_nodes(topology, "boundary_left");
    println!("Found {} nodes on boundary_left", boundary_nodes.len());

    let mut scalar_fields = Vec::new();

    match physics {
        "poisson" => {
            println!("Setting up 3D Poisson problem...");
            let problem = PoissonProblem3D {
                conductivity: 1.0,
                source: |_, _, _| 100.0,
                dirichlet: boundary_nodes.iter().map(|&node| (node, 0.0)).collect(),
            };

            println!("Solving...");
            let result = solve_poisson_3d(topology, &problem, SolverOptions::default())?;
            println!(
                "Poisson 3D solve converged. Residual norm: {}",
                result.residual_norm
            );

            scalar_fields.push(VtkScalarField::new("u", result.values));
        }
        "elasticity" => {
            println!("Setting up 3D Elasticity problem (cantilever beam bending)...");
            let rightmost_node = find_rightmost_node(topology);

            let problem = ElasticityProblem3D {
                material: ElasticityMaterial3D {
                    young_modulus: 2e11,
                    poisson_ratio: 0.3,
                },
                constraints: boundary_nodes
                    .iter()
                    .flat_map(|&node| {
                        vec![
                            DisplacementConstraint3D {
                                node,
                                component: DisplacementComponent3D::X,
                                value: 0.0,
                            },
                            DisplacementConstraint3D {
                                node,
                                component: DisplacementComponent3D::Y,
                                value: 0.0,
                            },
                            DisplacementConstraint3D {
                                node,
                                component: DisplacementComponent3D::Z,
                                value: 0.0,
                            },
                        ]
                    })
                    .collect(),
                forces: vec![NodalForce3D {
                    node: rightmost_node,
                    fx: 0.0,
                    fy: -1000.0,
                    fz: 0.0,
                }],
            };

            println!("Solving...");
            let result = solve_elasticity_3d(topology, &problem, SolverOptions::default())?;
            println!(
                "Elasticity 3D solve completed. Residual norm: {}",
                result.residual_norm
            );

            let ux: Vec<f64> = result.displacements.iter().map(|d| d[0]).collect();
            let uy: Vec<f64> = result.displacements.iter().map(|d| d[1]).collect();
            let uz: Vec<f64> = result.displacements.iter().map(|d| d[2]).collect();
            let mag: Vec<f64> = result
                .displacements
                .iter()
                .map(|d| (d[0].powi(2) + d[1].powi(2) + d[2].powi(2)).sqrt())
                .collect();

            scalar_fields.push(VtkScalarField::new("ux", ux));
            scalar_fields.push(VtkScalarField::new("uy", uy));
            scalar_fields.push(VtkScalarField::new("uz", uz));
            scalar_fields.push(VtkScalarField::new("displacement_magnitude", mag));
        }
        "modal" => {
            println!("Setting up 3D Modal Analysis...");
            let problem = ModalProblem3D {
                elasticity: ElasticityProblem3D {
                    material: ElasticityMaterial3D {
                        young_modulus: 2e11,
                        poisson_ratio: 0.3,
                    },
                    constraints: boundary_nodes
                        .iter()
                        .flat_map(|&node| {
                            vec![
                                DisplacementConstraint3D {
                                    node,
                                    component: DisplacementComponent3D::X,
                                    value: 0.0,
                                },
                                DisplacementConstraint3D {
                                    node,
                                    component: DisplacementComponent3D::Y,
                                    value: 0.0,
                                },
                                DisplacementConstraint3D {
                                    node,
                                    component: DisplacementComponent3D::Z,
                                    value: 0.0,
                                },
                            ]
                        })
                        .collect(),
                    forces: vec![],
                },
                density: 7850.0,
                mode_count: 3,
            };

            println!("Solving...");
            let result = solve_modal_3d(topology, &problem)?;
            println!("Modal 3D solve completed successfully.");

            for (i, mode) in result.modes.iter().enumerate() {
                println!(
                    "  Mode {}: Frequency = {:.2} Hz (Angular = {:.2} rad/s)",
                    i + 1,
                    mode.frequency_hz,
                    mode.angular_frequency
                );

                let mag: Vec<f64> = mode
                    .displacements
                    .iter()
                    .map(|d| (d[0].powi(2) + d[1].powi(2) + d[2].powi(2)).sqrt())
                    .collect();
                scalar_fields.push(VtkScalarField::new(format!("mode_{}_amplitude", i + 1), mag));
            }
        }
        _ => return Err(format!("Unsupported physics model: {physics}").into()),
    }

    println!("Writing VTK file to {output_path}...");
    write_vtk_legacy_file(output_path, topology, &scalar_fields)?;
    println!("Done!");

    Ok(())
}

fn get_boundary_nodes<const D: usize>(
    topology: &MeshTopology<D>,
    region_name: &str,
) -> BTreeSet<usize> {
    let mut nodes = BTreeSet::new();
    let region_id = topology
        .regions()
        .iter()
        .find(|r| r.name == region_name)
        .map(|r| r.id);

    if let Some(r_id) = region_id {
        for cell in topology.cells() {
            if cell.region == Some(r_id) {
                for &node in &cell.nodes {
                    nodes.insert(node);
                }
            }
        }
    }
    nodes
}

fn find_rightmost_node<const D: usize>(topology: &MeshTopology<D>) -> usize {
    let mut right_node = 0;
    let mut max_x = -f64::INFINITY;
    for (i, pt) in topology.points().iter().enumerate() {
        if pt.coords[0] > max_x {
            max_x = pt.coords[0];
            right_node = i;
        }
    }
    right_node
}
