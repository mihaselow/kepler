use kepler::{
    Cell, ElementKind, EntityDimension, Mesh, MeshError, MeshTopology, Point2, PointD, Region, Tri3,
};

#[test]
fn legacy_mesh_exposes_dimension_aware_cells() {
    let mesh = Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ],
        vec![Tri3::new([0, 1, 2])],
    )
    .unwrap();

    assert_eq!(mesh.dimension(), 2);
    assert_eq!(mesh.cells().len(), 1);
    assert_eq!(mesh.cells()[0].kind, ElementKind::Tri3);
    assert_eq!(mesh.cells()[0].nodes, vec![0, 1, 2]);

    let topology = mesh.topology().unwrap();
    assert_eq!(topology.dimension(), 2);
    assert_eq!(topology.cells()[0].kind, ElementKind::Tri3);
}

#[test]
fn topology_accepts_valid_three_dimensional_tetrahedra() {
    let topology = MeshTopology::<3>::new(
        vec![
            PointD::new([0.0, 0.0, 0.0]),
            PointD::new([1.0, 0.0, 0.0]),
            PointD::new([0.0, 1.0, 0.0]),
            PointD::new([0.0, 0.0, 1.0]),
        ],
        vec![Cell::new(ElementKind::Tet4, vec![0, 1, 2, 3])],
    )
    .unwrap();

    assert_eq!(topology.dimension(), 3);
    assert_eq!(topology.points().len(), 4);
    assert_eq!(topology.cells()[0].kind, ElementKind::Tet4);
}

#[test]
fn topology_rejects_volume_elements_in_two_dimensions() {
    let error = MeshTopology::<2>::new(
        vec![
            PointD::new([0.0, 0.0]),
            PointD::new([1.0, 0.0]),
            PointD::new([0.0, 1.0]),
            PointD::new([1.0, 1.0]),
        ],
        vec![Cell::new(ElementKind::Tet4, vec![0, 1, 2, 3])],
    )
    .unwrap_err();

    assert!(matches!(
        error,
        MeshError::ElementDimensionExceedsMesh {
            cell_index: 0,
            element_dimension: 3,
            mesh_dimension: 2,
        }
    ));
}

#[test]
fn topology_rejects_degenerate_tetrahedra() {
    let error = MeshTopology::<3>::new(
        vec![
            PointD::new([0.0, 0.0, 0.0]),
            PointD::new([1.0, 0.0, 0.0]),
            PointD::new([0.0, 1.0, 0.0]),
            PointD::new([1.0, 1.0, 0.0]),
        ],
        vec![Cell::new(ElementKind::Tet4, vec![0, 1, 2, 3])],
    )
    .unwrap_err();

    assert!(matches!(error, MeshError::DegenerateCell { cell_index: 0 }));
}

#[test]
fn topology_validates_region_references_and_dimensions() {
    let topology = MeshTopology::<2>::with_regions(
        vec![
            PointD::new([0.0, 0.0]),
            PointD::new([1.0, 0.0]),
            PointD::new([0.0, 1.0]),
        ],
        vec![Cell::new(ElementKind::Tri3, vec![0, 1, 2]).with_region(7)],
        vec![Region::new(7, "fluid", EntityDimension::Surface)],
    )
    .unwrap();

    assert_eq!(topology.regions()[0].name, "fluid");
    assert_eq!(topology.cells()[0].region, Some(7));
}

#[test]
fn topology_rejects_region_dimension_mismatches() {
    let error = MeshTopology::<2>::with_regions(
        vec![
            PointD::new([0.0, 0.0]),
            PointD::new([1.0, 0.0]),
            PointD::new([0.0, 1.0]),
        ],
        vec![Cell::new(ElementKind::Tri3, vec![0, 1, 2]).with_region(1)],
        vec![Region::new(1, "edge", EntityDimension::Curve)],
    )
    .unwrap_err();

    assert!(matches!(
        error,
        MeshError::RegionDimensionMismatch {
            cell_index: 0,
            region_id: 1,
            cell_dimension: 2,
            region_dimension: 1,
        }
    ));
}
