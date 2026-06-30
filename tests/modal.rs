use kepler::{
    Cell, DisplacementComponent, DisplacementComponent3D, DisplacementConstraint,
    DisplacementConstraint3D, ElasticityMaterial, ElasticityMaterial3D, ElasticityModel,
    ElasticityProblem, ElasticityProblem3D, ElementKind, Mesh, MeshTopology, ModalError,
    ModalProblem, ModalProblem3D, Point2, PointD, Tri3, solve_modal, solve_modal_3d,
};

#[test]
fn modal_solves_sorted_two_dimensional_modes() {
    let mesh = unit_triangle();
    let problem = ModalProblem {
        elasticity: ElasticityProblem {
            material: material_2d(),
            thickness: 1.0,
            constraints: fixed_nodes_2d(&[0, 2]),
            forces: vec![],
        },
        density: 2.0,
        mode_count: 2,
    };

    let result = solve_modal(&mesh, &problem).unwrap();

    assert_eq!(result.modes.len(), 2);
    assert!(result.modes[0].frequency_hz > 0.0);
    assert!(result.modes[1].frequency_hz >= result.modes[0].frequency_hz);
    assert_eq!(result.modes[0].displacements.len(), mesh.node_count());
    assert_close(result.modes[0].displacements[0][0], 0.0);
    assert_close(result.modes[0].displacements[2][1], 0.0);
}

#[test]
fn modal_solves_sorted_three_dimensional_modes() {
    let mesh = unit_tetrahedron();
    let problem = ModalProblem3D {
        elasticity: ElasticityProblem3D {
            material: material_3d(),
            constraints: fixed_nodes_3d(&[0, 2, 3]),
            forces: vec![],
        },
        density: 2.0,
        mode_count: 3,
    };

    let result = solve_modal_3d(&mesh, &problem).unwrap();

    assert_eq!(result.modes.len(), 3);
    assert!(result.modes[0].frequency_hz > 0.0);
    assert!(result.modes[1].frequency_hz >= result.modes[0].frequency_hz);
    assert!(result.modes[2].frequency_hz >= result.modes[1].frequency_hz);
    assert_eq!(result.modes[0].displacements.len(), mesh.points().len());
    assert_close(result.modes[0].displacements[0][0], 0.0);
    assert_close(result.modes[0].displacements[3][2], 0.0);
}

#[test]
fn modal_limits_requested_modes_to_active_dofs() {
    let mesh = unit_triangle();
    let problem = ModalProblem {
        elasticity: ElasticityProblem {
            material: material_2d(),
            thickness: 1.0,
            constraints: fixed_nodes_2d(&[0, 2]),
            forces: vec![],
        },
        density: 2.0,
        mode_count: 10,
    };

    let result = solve_modal(&mesh, &problem).unwrap();

    assert_eq!(result.modes.len(), 2);
}

#[test]
fn modal_rejects_invalid_density() {
    let mesh = unit_triangle();
    let problem = ModalProblem {
        elasticity: ElasticityProblem {
            material: material_2d(),
            thickness: 1.0,
            constraints: vec![],
            forces: vec![],
        },
        density: 0.0,
        mode_count: 1,
    };

    let error = solve_modal(&mesh, &problem).unwrap_err();

    assert_eq!(error, ModalError::InvalidDensity(0.0));
}

#[test]
fn modal_rejects_fully_constrained_models() {
    let mesh = unit_triangle();
    let problem = ModalProblem {
        elasticity: ElasticityProblem {
            material: material_2d(),
            thickness: 1.0,
            constraints: fixed_nodes_2d(&[0, 1, 2]),
            forces: vec![],
        },
        density: 2.0,
        mode_count: 1,
    };

    let error = solve_modal(&mesh, &problem).unwrap_err();

    assert_eq!(error, ModalError::NoActiveDegreesOfFreedom);
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

fn fixed_nodes_2d(nodes: &[usize]) -> Vec<DisplacementConstraint> {
    nodes
        .iter()
        .flat_map(|&node| {
            [
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
        .collect()
}

fn fixed_nodes_3d(nodes: &[usize]) -> Vec<DisplacementConstraint3D> {
    nodes
        .iter()
        .flat_map(|&node| {
            [
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
        .collect()
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

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-10,
        "expected {actual} to be close to {expected}",
    );
}

#[test]
fn modal_solves_hex8_modes() {
    let mesh = MeshTopology::<3>::new(
        vec![
            PointD::new([0.0, 0.0, 0.0]),
            PointD::new([1.0, 0.0, 0.0]),
            PointD::new([1.0, 1.0, 0.0]),
            PointD::new([0.0, 1.0, 0.0]),
            PointD::new([0.0, 0.0, 1.0]),
            PointD::new([1.0, 0.0, 1.0]),
            PointD::new([1.0, 1.0, 1.0]),
            PointD::new([0.0, 1.0, 1.0]),
        ],
        vec![Cell::new(ElementKind::Hex8, vec![0, 1, 2, 3, 4, 5, 6, 7])],
    )
    .unwrap();

    let problem = ModalProblem3D {
        elasticity: ElasticityProblem3D {
            material: material_3d(),
            constraints: fixed_nodes_3d(&[0, 2, 3, 4]),
            forces: vec![],
        },
        density: 2.0,
        mode_count: 2,
    };

    let result = solve_modal_3d(&mesh, &problem).unwrap();

    assert_eq!(result.modes.len(), 2);
    assert!(result.modes[0].frequency_hz > 0.0);
    assert!(result.modes[1].frequency_hz >= result.modes[0].frequency_hz);
}
