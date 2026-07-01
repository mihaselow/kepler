use kepler::{abaqus_to_mesh_2d, parse_abaqus_str};

#[test]
fn abaqus_import_parses_block_fixture() {
    let input = include_str!("../examples/data/abaqus/block.inp");
    let model = parse_abaqus_str(input).unwrap();
    assert_eq!(model.nodes.len(), 4);
    assert_eq!(model.elements.len(), 2);
    assert!(model.materials.contains_key("STEEL"));
    assert_eq!(model.steps.len(), 1);
    assert_eq!(model.steps[0].kind, "static");

    let mesh = abaqus_to_mesh_2d(&model).unwrap();
    assert_eq!(mesh.node_count(), 4);
    assert_eq!(mesh.cells().len(), 2);
}
