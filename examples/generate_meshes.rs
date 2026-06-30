use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let data_dir = Path::new("examples/data");
    std::fs::create_dir_all(data_dir).unwrap();

    // Generate small meshes (for fast runs, modal analysis)
    generate_square_tri3(data_dir, 10, "small");
    generate_cube_tet4(data_dir, 4, "small");
    generate_cube_hex8(data_dir, 4, "small");

    // Generate fine meshes (for detailed solves)
    generate_square_tri3(data_dir, 50, "fine");
    generate_cube_tet4(data_dir, 25, "fine");
    generate_cube_hex8(data_dir, 25, "fine");

    // Custom Kepler format mesh
    generate_square_custom_mesh(data_dir);

    println!("Successfully generated small/fine 2D/3D test mesh files in examples/data/");
}

fn generate_square_tri3(dir: &Path, n: usize, suffix: &str) {
    let filename = format!("square_tri3_{}.msh", suffix);
    let mut file = File::create(dir.join(filename)).unwrap();

    writeln!(file, "$MeshFormat\n2.2 0 8\n$EndMeshFormat").unwrap();
    writeln!(
        file,
        "$PhysicalNames\n2\n2 1 \"domain\"\n1 2 \"boundary_left\"\n$EndPhysicalNames"
    )
    .unwrap();

    // Nodes
    let node_count = (n + 1) * (n + 1);
    writeln!(file, "$Nodes\n{}", node_count).unwrap();
    for r in 0..=n {
        for c in 0..=n {
            let id = r * (n + 1) + c + 1;
            let x = c as f64 / n as f64;
            let y = r as f64 / n as f64;
            writeln!(file, "{} {} {} 0.0", id, x, y).unwrap();
        }
    }
    writeln!(file, "$EndNodes").unwrap();

    // Elements
    let mut elements = Vec::new();
    let mut el_id = 1;

    // Boundary lines on left edge (x = 0)
    for r in 0..n {
        let n0 = r * (n + 1) + 1;
        let n1 = (r + 1) * (n + 1) + 1;
        // type 1 (Line2), 2 tags: physical group 2, elementary entity 1
        elements.push(format!("{} 1 2 2 1 {} {}", el_id, n0, n1));
        el_id += 1;
    }

    // Domain triangles
    for r in 0..n {
        for c in 0..n {
            let n0 = r * (n + 1) + c + 1;
            let n1 = n0 + 1;
            let n2 = (r + 1) * (n + 1) + c + 1;
            let n3 = n2 + 1;

            // type 2 (Tri3), 2 tags: physical group 1, elementary entity 1
            elements.push(format!("{} 2 2 1 1 {} {} {}", el_id, n0, n1, n3));
            el_id += 1;
            elements.push(format!("{} 2 2 1 1 {} {} {}", el_id, n0, n3, n2));
            el_id += 1;
        }
    }

    writeln!(file, "$Elements\n{}", elements.len()).unwrap();
    for el in elements {
        writeln!(file, "{}", el).unwrap();
    }
    writeln!(file, "$EndElements").unwrap();
}

fn generate_cube_tet4(dir: &Path, n: usize, suffix: &str) {
    let filename = format!("cube_tet4_{}.msh", suffix);
    let mut file = File::create(dir.join(filename)).unwrap();

    writeln!(file, "$MeshFormat\n2.2 0 8\n$EndMeshFormat").unwrap();
    writeln!(
        file,
        "$PhysicalNames\n2\n3 1 \"domain\"\n2 2 \"boundary_left\"\n$EndPhysicalNames"
    )
    .unwrap();

    // Nodes
    let node_count = (n + 1) * (n + 1) * (n + 1);
    writeln!(file, "$Nodes\n{}", node_count).unwrap();
    let mut id = 1;
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                let x = i as f64 / n as f64;
                let y = j as f64 / n as f64;
                let z = k as f64 / n as f64;
                writeln!(file, "{} {} {} {}", id, x, y, z).unwrap();
                id += 1;
            }
        }
    }
    writeln!(file, "$EndNodes").unwrap();

    // Helper to get node index (1-based)
    let get_node = |i: usize, j: usize, k: usize| -> usize {
        1 + i + j * (n + 1) + k * (n + 1) * (n + 1)
    };

    let mut elements = Vec::new();
    let mut el_id = 1;

    // Boundary triangles on left face (x = 0)
    for k in 0..n {
        for j in 0..n {
            let f0 = get_node(0, j, k);
            let f1 = get_node(0, j + 1, k);
            let f2 = get_node(0, j + 1, k + 1);
            let f3 = get_node(0, j, k + 1);

            // type 2 (Tri3), 2 tags: physical group 2, elementary entity 1
            elements.push(format!("{} 2 2 2 1 {} {} {}", el_id, f0, f1, f2));
            el_id += 1;
            elements.push(format!("{} 2 2 2 1 {} {} {}", el_id, f0, f2, f3));
            el_id += 1;
        }
    }

    // Domain tetrahedra
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p0 = get_node(i, j, k);
                let p1 = get_node(i + 1, j, k);
                let p2 = get_node(i + 1, j + 1, k);
                let p3 = get_node(i, j + 1, k);
                let p4 = get_node(i, j, k + 1);
                let p5 = get_node(i + 1, j, k + 1);
                let p6 = get_node(i + 1, j + 1, k + 1);
                let p7 = get_node(i, j + 1, k + 1);

                // Split cube into 6 tets (type 4, 2 tags: physical group 1, elementary entity 1)
                elements.push(format!("{} 4 2 1 1 {} {} {} {}", el_id, p0, p1, p2, p6));
                el_id += 1;
                elements.push(format!("{} 4 2 1 1 {} {} {} {}", el_id, p0, p2, p3, p6));
                el_id += 1;
                elements.push(format!("{} 4 2 1 1 {} {} {} {}", el_id, p0, p1, p6, p5));
                el_id += 1;
                elements.push(format!("{} 4 2 1 1 {} {} {} {}", el_id, p0, p5, p4, p6));
                el_id += 1;
                elements.push(format!("{} 4 2 1 1 {} {} {} {}", el_id, p0, p4, p7, p6));
                el_id += 1;
                elements.push(format!("{} 4 2 1 1 {} {} {} {}", el_id, p0, p7, p3, p6));
                el_id += 1;
            }
        }
    }

    writeln!(file, "$Elements\n{}", elements.len()).unwrap();
    for el in elements {
        writeln!(file, "{}", el).unwrap();
    }
    writeln!(file, "$EndElements").unwrap();
}

fn generate_cube_hex8(dir: &Path, n: usize, suffix: &str) {
    let filename = format!("cube_hex8_{}.msh", suffix);
    let mut file = File::create(dir.join(filename)).unwrap();

    writeln!(file, "$MeshFormat\n2.2 0 8\n$EndMeshFormat").unwrap();
    writeln!(
        file,
        "$PhysicalNames\n2\n3 1 \"domain\"\n2 2 \"boundary_left\"\n$EndPhysicalNames"
    )
    .unwrap();

    // Nodes
    let node_count = (n + 1) * (n + 1) * (n + 1);
    writeln!(file, "$Nodes\n{}", node_count).unwrap();
    let mut id = 1;
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                let x = i as f64 / n as f64;
                let y = j as f64 / n as f64;
                let z = k as f64 / n as f64;
                writeln!(file, "{} {} {} {}", id, x, y, z).unwrap();
                id += 1;
            }
        }
    }
    writeln!(file, "$EndNodes").unwrap();

    let get_node = |i: usize, j: usize, k: usize| -> usize {
        1 + i + j * (n + 1) + k * (n + 1) * (n + 1)
    };

    let mut elements = Vec::new();
    let mut el_id = 1;

    // Boundary quads on left face (x = 0)
    for k in 0..n {
        for j in 0..n {
            let f0 = get_node(0, j, k);
            let f1 = get_node(0, j + 1, k);
            let f2 = get_node(0, j + 1, k + 1);
            let f3 = get_node(0, j, k + 1);

            // type 3 (Quad4), 2 tags: physical group 2, elementary entity 1
            elements.push(format!("{} 3 2 2 1 {} {} {} {}", el_id, f0, f1, f2, f3));
            el_id += 1;
        }
    }

    // Domain hexahedra
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p0 = get_node(i, j, k);
                let p1 = get_node(i + 1, j, k);
                let p2 = get_node(i + 1, j + 1, k);
                let p3 = get_node(i, j + 1, k);
                let p4 = get_node(i, j, k + 1);
                let p5 = get_node(i + 1, j, k + 1);
                let p6 = get_node(i + 1, j + 1, k + 1);
                let p7 = get_node(i, j + 1, k + 1);

                // type 5 (Hex8), 2 tags: physical group 1, elementary entity 1
                elements.push(format!(
                    "{} 5 2 1 1 {} {} {} {} {} {} {} {}",
                    el_id, p0, p1, p2, p3, p4, p5, p6, p7
                ));
                el_id += 1;
            }
        }
    }

    writeln!(file, "$Elements\n{}", elements.len()).unwrap();
    for el in elements {
        writeln!(file, "{}", el).unwrap();
    }
    writeln!(file, "$EndElements").unwrap();
}

fn generate_square_custom_mesh(dir: &Path) {
    let n = 22;
    
    // Write square_fine.mesh
    let mut file_mesh = File::create(dir.join("square_fine.mesh")).unwrap();
    writeln!(file_mesh, "nodes").unwrap();
    for r in 0..=n {
        for c in 0..=n {
            let id = r * (n + 1) + c;
            let x = c as f64 / n as f64;
            let y = r as f64 / n as f64;
            writeln!(file_mesh, "{} {} {}", id, x, y).unwrap();
        }
    }
    
    writeln!(file_mesh, "\ntriangles").unwrap();
    let mut el_id = 0;
    for r in 0..n {
        for c in 0..n {
            let n0 = r * (n + 1) + c;
            let n1 = n0 + 1;
            let n2 = (r + 1) * (n + 1) + c;
            let n3 = n2 + 1;
            
            writeln!(file_mesh, "{} {} {} {}", el_id, n0, n1, n3).unwrap();
            el_id += 1;
            writeln!(file_mesh, "{} {} {} {}", el_id, n0, n3, n2).unwrap();
            el_id += 1;
        }
    }
    
    // Write square_fine.params
    let mut file_params = File::create(dir.join("square_fine.params")).unwrap();
    writeln!(file_params, "conductivity 1.0\nsource constant 100.0").unwrap();
    writeln!(file_params, "solver max_iterations 1000\nsolver tolerance 1e-8").unwrap();
    writeln!(file_params, "solver backend conjugate_gradient\nsolver preconditioner jacobi").unwrap();
    writeln!(file_params, "solver record_residual_history false\n\ndirichlet").unwrap();
    
    // Collect and write boundary nodes
    let mut boundary_nodes = std::collections::BTreeSet::new();
    for c in 0..=n {
        boundary_nodes.insert(c);
        boundary_nodes.insert(n * (n + 1) + c);
    }
    for r in 1..n {
        boundary_nodes.insert(r * (n + 1));
        boundary_nodes.insert(r * (n + 1) + n);
    }
    for node in boundary_nodes {
        writeln!(file_params, "{} 0.0", node).unwrap();
    }
}
