use std::f64::consts::PI;

use kepler::{
    BeamSection, Cell, DisplacementComponent, DisplacementComponent3D, DisplacementConstraint,
    DisplacementConstraint3D, ElasticityMaterial, ElasticityMaterial3D, ElasticityModel,
    ElasticityProblem, ElasticityProblem3D, ElementKind, Mesh, MeshTopology, ModalProblem,
    ModalProblem3D, Point2, Point3, PointD, SolverOptions, StructuralComponent,
    StructuralConstraint, StructuralForce, StructuralMaterial, StructuralProblem, Tri3,
    fem::elasticity::{local_elasticity_stiffness, local_tet4_elasticity_stiffness},
    solve_elasticity, solve_elasticity_3d, solve_modal, solve_modal_3d, solve_structural,
};

const EPS: f64 = 1.0e-10;

#[test]
fn elasticity_2d_preserves_affine_displacement_constraints() {
    let mesh = unit_triangle();
    let problem = ElasticityProblem {
        material: material_2d(),
        thickness: 1.0,
        constraints: mesh
            .points()
            .iter()
            .enumerate()
            .flat_map(|(node, point)| {
                let [ux, uy] = affine_displacement_2d(point.x, point.y);
                [
                    DisplacementConstraint {
                        node,
                        component: DisplacementComponent::X,
                        value: ux,
                    },
                    DisplacementConstraint {
                        node,
                        component: DisplacementComponent::Y,
                        value: uy,
                    },
                ]
            })
            .collect(),
        forces: vec![],
    };

    let result = solve_elasticity(&mesh, &problem, SolverOptions::default()).unwrap();

    for (node, point) in mesh.points().iter().enumerate() {
        let expected = affine_displacement_2d(point.x, point.y);
        assert_close(result.displacements[node][0], expected[0]);
        assert_close(result.displacements[node][1], expected[1]);
    }
}

#[test]
fn elasticity_3d_preserves_affine_displacement_constraints() {
    let mesh = unit_tetrahedron();
    let problem = ElasticityProblem3D {
        material: material_3d(),
        constraints: mesh
            .points()
            .iter()
            .enumerate()
            .flat_map(|(node, point)| {
                let [ux, uy, uz] =
                    affine_displacement_3d(point.coords[0], point.coords[1], point.coords[2]);
                [
                    DisplacementConstraint3D {
                        node,
                        component: DisplacementComponent3D::X,
                        value: ux,
                    },
                    DisplacementConstraint3D {
                        node,
                        component: DisplacementComponent3D::Y,
                        value: uy,
                    },
                    DisplacementConstraint3D {
                        node,
                        component: DisplacementComponent3D::Z,
                        value: uz,
                    },
                ]
            })
            .collect(),
        forces: vec![],
    };

    let result = solve_elasticity_3d(&mesh, &problem, SolverOptions::default()).unwrap();

    for (node, point) in mesh.points().iter().enumerate() {
        let expected = affine_displacement_3d(point.coords[0], point.coords[1], point.coords[2]);
        assert_close(result.displacements[node][0], expected[0]);
        assert_close(result.displacements[node][1], expected[1]);
        assert_close(result.displacements[node][2], expected[2]);
    }
}

#[test]
fn local_elasticity_matrices_have_rigid_translation_null_modes() {
    let mesh = unit_triangle();
    let stiffness =
        local_elasticity_stiffness(&mesh, &mesh.triangles()[0], material_2d(), 1.0).unwrap();

    assert_matrix_vector_close_2d(stiffness, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0; 6]);
    assert_matrix_vector_close_2d(stiffness, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [0.0; 6]);

    let mesh_3d = unit_tetrahedron();
    let stiffness_3d =
        local_tet4_elasticity_stiffness(&mesh_3d, [0, 1, 2, 3], material_3d()).unwrap();
    assert_matrix_vector_close_3d(
        stiffness_3d,
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0; 12],
    );
    assert_matrix_vector_close_3d(
        stiffness_3d,
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0; 12],
    );
    assert_matrix_vector_close_3d(
        stiffness_3d,
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0; 12],
    );
}

#[test]
fn modal_2d_matches_one_dof_frequency_reference() {
    let mesh = unit_triangle();
    let density = 2.0;
    let thickness = 1.0;
    let constraints = all_dofs_except_node_component_2d(1, DisplacementComponent::X);
    let stiffness =
        local_elasticity_stiffness(&mesh, &mesh.triangles()[0], material_2d(), thickness).unwrap();
    let active_stiffness = stiffness[2][2];
    let active_mass = density * thickness * 0.5 / 3.0;
    let expected_frequency = (active_stiffness / active_mass).sqrt() / (2.0 * PI);
    let problem = ModalProblem {
        elasticity: ElasticityProblem {
            material: material_2d(),
            thickness,
            constraints,
            forces: vec![],
        },
        density,
        mode_count: 1,
    };

    let result = solve_modal(&mesh, &problem).unwrap();

    assert_close(result.modes[0].frequency_hz, expected_frequency);
    assert!(result.modes[0].displacements[1][0].abs() > 0.0);
    assert_close(result.modes[0].displacements[1][1], 0.0);
}

#[test]
fn modal_3d_matches_one_dof_frequency_reference() {
    let mesh = unit_tetrahedron();
    let density = 24.0;
    let constraints = all_dofs_except_node_component_3d(1, DisplacementComponent3D::X);
    let stiffness = local_tet4_elasticity_stiffness(&mesh, [0, 1, 2, 3], material_3d()).unwrap();
    let active_stiffness = stiffness[3][3];
    let active_mass = density * (1.0 / 6.0) / 4.0;
    let expected_frequency = (active_stiffness / active_mass).sqrt() / (2.0 * PI);
    let problem = ModalProblem3D {
        elasticity: ElasticityProblem3D {
            material: material_3d(),
            constraints,
            forces: vec![],
        },
        density,
        mode_count: 1,
    };

    let result = solve_modal_3d(&mesh, &problem).unwrap();

    assert_close(result.modes[0].frequency_hz, expected_frequency);
    assert!(result.modes[0].displacements[1][0].abs() > 0.0);
    assert_close(result.modes[0].displacements[1][1], 0.0);
    assert_close(result.modes[0].displacements[1][2], 0.0);
}

fn unit_triangle() -> Mesh {
    Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ],
        vec![Tri3::new([0, 1, 2])],
    )
    .unwrap()
}

fn unit_tetrahedron() -> MeshTopology<3> {
    MeshTopology::<3>::new(
        vec![
            PointD::new([0.0, 0.0, 0.0]),
            PointD::new([1.0, 0.0, 0.0]),
            PointD::new([0.0, 1.0, 0.0]),
            PointD::new([0.0, 0.0, 1.0]),
        ],
        vec![Cell::new(ElementKind::Tet4, vec![0, 1, 2, 3])],
    )
    .unwrap()
}

fn material_2d() -> ElasticityMaterial {
    ElasticityMaterial {
        young_modulus: 100.0,
        poisson_ratio: 0.25,
        model: ElasticityModel::PlaneStress,
    }
}

fn material_3d() -> ElasticityMaterial3D {
    ElasticityMaterial3D {
        young_modulus: 100.0,
        poisson_ratio: 0.25,
    }
}

fn affine_displacement_2d(x: f64, y: f64) -> [f64; 2] {
    [0.1 + 0.2 * x - 0.05 * y, -0.3 + 0.15 * x + 0.1 * y]
}

fn affine_displacement_3d(x: f64, y: f64, z: f64) -> [f64; 3] {
    [
        0.1 + 0.2 * x - 0.05 * y + 0.03 * z,
        -0.3 + 0.15 * x + 0.1 * y - 0.02 * z,
        0.2 - 0.04 * x + 0.07 * y + 0.12 * z,
    ]
}

fn all_dofs_except_node_component_2d(
    active_node: usize,
    active_component: DisplacementComponent,
) -> Vec<DisplacementConstraint> {
    let mut constraints = Vec::new();
    for node in 0..3 {
        for component in [DisplacementComponent::X, DisplacementComponent::Y] {
            if node == active_node && component == active_component {
                continue;
            }
            constraints.push(DisplacementConstraint {
                node,
                component,
                value: 0.0,
            });
        }
    }
    constraints
}

fn all_dofs_except_node_component_3d(
    active_node: usize,
    active_component: DisplacementComponent3D,
) -> Vec<DisplacementConstraint3D> {
    let mut constraints = Vec::new();
    for node in 0..4 {
        for component in [
            DisplacementComponent3D::X,
            DisplacementComponent3D::Y,
            DisplacementComponent3D::Z,
        ] {
            if node == active_node && component == active_component {
                continue;
            }
            constraints.push(DisplacementConstraint3D {
                node,
                component,
                value: 0.0,
            });
        }
    }
    constraints
}

fn assert_matrix_vector_close_2d(matrix: [[f64; 6]; 6], vector: [f64; 6], expected: [f64; 6]) {
    for row in 0..6 {
        let actual: f64 = matrix[row]
            .iter()
            .zip(vector)
            .map(|(value, vector_value)| value * vector_value)
            .sum();
        assert_close(actual, expected[row]);
    }
}

fn assert_matrix_vector_close_3d(matrix: [[f64; 12]; 12], vector: [f64; 12], expected: [f64; 12]) {
    for row in 0..12 {
        let actual: f64 = matrix[row]
            .iter()
            .zip(vector)
            .map(|(value, vector_value)| value * vector_value)
            .sum();
        assert_close(actual, expected[row]);
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS,
        "expected {actual} to be within {EPS} of {expected}",
    );
}

#[test]
fn test_beam_3d_element() {
    use kepler::fem::element::Element;
    use kepler::{Beam3D, Point3};
    use std::collections::BTreeMap;

    let nodes = [0, 1];
    let el = Beam3D {
        nodes: &nodes,
        area: 0.01,
        moment_y: 1.0e-5,
        moment_z: 2.0e-5,
        torsional_constant: 3.0e-5,
        local_y_direction: [0.0, 1.0, 0.0],
    };

    let coords = [Point3::new([0.0, 0.0, 0.0]), Point3::new([2.0, 0.0, 0.0])];

    let mut properties = BTreeMap::new();
    properties.insert("young_modulus".to_string(), 200e9);
    properties.insert("poisson_ratio".to_string(), 0.3);

    let k = el.local_stiffness(&coords, &properties).unwrap();
    assert_eq!(k.len(), 12);

    // 1. Axial stiffness: E * A / L = 200e9 * 0.01 / 2.0 = 1e9
    assert!((k[0][0] - 1e9).abs() < 1.0);
    assert!((k[0][6] - -1e9).abs() < 1.0);

    // 2. Torsional stiffness: G * J / L where G = E / 2.6
    let g = 200e9 / 2.6;
    let expected_torsion = g * 3.0e-5 / 2.0;
    assert!((k[3][3] - expected_torsion).abs() < 1.0);
    assert!((k[3][9] - -expected_torsion).abs() < 1.0);

    // 3. Bending around z (in x-y plane): 12 * E * Iz / L^3 = 12 * 200e9 * 2.0e-5 / 8.0 = 6e6
    assert!((k[1][1] - 6e6).abs() < 1.0);
    assert!((k[1][7] - -6e6).abs() < 1.0);

    // 4. Bending around y (in x-z plane): 12 * E * Iy / L^3 = 12 * 200e9 * 1.0e-5 / 8.0 = 3e6
    assert!((k[2][2] - 3e6).abs() < 1.0);
    assert!((k[2][8] - -3e6).abs() < 1.0);

    // 5. Check symmetry
    for (r, row) in k.iter().enumerate().take(12) {
        for (c, &k_rc) in row.iter().enumerate().take(12) {
            assert!(
                (k_rc - k[c][r]).abs() < 1.0e-3,
                "Symmetry failed at ({}, {})",
                r,
                c
            );
        }
    }

    // 6. Rigid translation null modes
    let mut k_arr = [[0.0; 12]; 12];
    for r in 0..12 {
        for c in 0..12 {
            k_arr[r][c] = k[r][c];
        }
    }

    let x_translation = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y_translation = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let z_translation = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    assert_matrix_vector_close_3d(k_arr, x_translation, [0.0; 12]);
    assert_matrix_vector_close_3d(k_arr, y_translation, [0.0; 12]);
    assert_matrix_vector_close_3d(k_arr, z_translation, [0.0; 12]);
}

fn fixed_structural_node(node: usize) -> Vec<StructuralConstraint> {
    [
        StructuralComponent::Ux,
        StructuralComponent::Uy,
        StructuralComponent::Uz,
        StructuralComponent::ThetaX,
        StructuralComponent::ThetaY,
        StructuralComponent::ThetaZ,
    ]
    .into_iter()
    .map(|component| StructuralConstraint {
        node,
        component,
        value: 0.0,
    })
    .collect()
}

#[test]
fn structural_cantilever_beam_tip_deflection_matches_euler_bernoulli() {
    let length = 10.0;
    let young_modulus = 200e9;
    let moment_z = 1.0e-4;
    let force = 1_000.0;

    let mesh = MeshTopology::new(
        vec![
            Point3::new([0.0, 0.0, 0.0]),
            Point3::new([length, 0.0, 0.0]),
        ],
        vec![Cell::new(ElementKind::Line2, vec![0, 1])],
    )
    .unwrap();

    let problem = StructuralProblem {
        material: StructuralMaterial {
            young_modulus,
            poisson_ratio: 0.3,
        },
        beam_section: BeamSection {
            area: 0.01,
            moment_y: 1.0e-5,
            moment_z,
            torsional_constant: 3.0e-5,
            local_y_direction: [0.0, 1.0, 0.0],
        },
        shell_thickness: 0.1,
        constraints: fixed_structural_node(0),
        forces: vec![StructuralForce {
            node: 1,
            component: StructuralComponent::Uy,
            value: force,
        }],
    };

    let result = solve_structural(&mesh, &problem, SolverOptions::default()).unwrap();
    let expected = force * length.powi(3) / (3.0 * young_modulus * moment_z);
    assert_close(result.displacements[1][1], expected);
}

#[test]
fn structural_scordelis_lo_roof_crown_deflects_downward() {
    // Coarse Scordelis-Lo-style cylindrical roof patch (R=25 ft, 40 deg arc, t=0.25 ft).
    // Reference crown displacement for the full benchmark is ~0.3024 in under self-weight;
    // this coarse 2x2 Quad4 mesh checks downward crown motion and reasonable magnitude.
    let radius = 25.0 * 0.3048; // ft to m
    let thickness = 0.25 * 0.3048;
    let length = 25.0 * 0.3048; // quarter-model half-length
    let young_modulus = 4.32e8 * 6894.76; // psi to Pa
    let density = 0.2086 * 27679.9;

    let angles = [0.0_f64, 20.0_f64.to_radians(), 40.0_f64.to_radians()];
    let x_coords = [0.0, length];
    let mut points = Vec::new();
    for &x in &x_coords {
        for &theta in &angles {
            let y = radius * theta.sin();
            let z = radius * (1.0 - theta.cos());
            points.push(Point3::new([x, y, z]));
        }
    }

    let node = |ix: usize, itheta: usize| ix * angles.len() + itheta;
    let cells = vec![
        Cell::new(
            ElementKind::Quad4,
            vec![node(0, 0), node(1, 0), node(1, 1), node(0, 1)],
        ),
        Cell::new(
            ElementKind::Quad4,
            vec![node(0, 1), node(1, 1), node(1, 2), node(0, 2)],
        ),
    ];

    let mesh = MeshTopology::new(points, cells).unwrap();
    let crown_node = node(0, 1);

    let mut constraints = Vec::new();
    // Built-in edge at theta = 0.
    for &n in &[node(0, 0), node(1, 0)] {
        constraints.extend(fixed_structural_node(n));
    }
    // Symmetry plane at x = 0.
    for itheta in 1..angles.len() {
        constraints.push(StructuralConstraint {
            node: node(0, itheta),
            component: StructuralComponent::Ux,
            value: 0.0,
        });
    }

    let elem_area = radius * (angles[2] - angles[0]) * length / 2.0;
    let weight_per_node = density * thickness * elem_area / 4.0;
    let mut forces = Vec::new();
    for ix in 0..x_coords.len() {
        for itheta in 1..angles.len() {
            forces.push(StructuralForce {
                node: node(ix, itheta),
                component: StructuralComponent::Uz,
                value: -weight_per_node,
            });
        }
    }

    let problem = StructuralProblem {
        material: StructuralMaterial {
            young_modulus,
            poisson_ratio: 0.0,
        },
        beam_section: BeamSection::default(),
        shell_thickness: thickness,
        constraints,
        forces,
    };

    let result = solve_structural(&mesh, &problem, SolverOptions::default()).unwrap();
    let crown_uz = result.displacements[crown_node][2];
    assert!(
        crown_uz < 0.0,
        "crown should deflect downward, got uz={crown_uz}"
    );
    assert!(
        crown_uz.abs() > 1.0e-9,
        "crown deflection should be non-trivial"
    );
    assert!(
        crown_uz.abs() < 0.05,
        "crown deflection magnitude unreasonable: {crown_uz}"
    );
}
