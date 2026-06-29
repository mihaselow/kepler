use thiserror::Error;

pub type NodeId = usize;
pub type CellId = usize;
pub type FacetId = usize;
pub type RegionId = usize;
pub type MaterialId = usize;
pub type FieldId = usize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointD<const D: usize> {
    pub coords: [f64; D],
}

impl<const D: usize> PointD<D> {
    pub const fn new(coords: [f64; D]) -> Self {
        Self { coords }
    }

    pub const fn dimension(&self) -> usize {
        D
    }
}

pub type Point3 = PointD<3>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub const fn dimension(&self) -> usize {
        2
    }
}

impl From<Point2> for PointD<2> {
    fn from(value: Point2) -> Self {
        Self::new([value.x, value.y])
    }
}

impl From<PointD<2>> for Point2 {
    fn from(value: PointD<2>) -> Self {
        Self::new(value.coords[0], value.coords[1])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityDimension {
    Point,
    Curve,
    Surface,
    Volume,
}

impl EntityDimension {
    pub const fn spatial_dimension(self) -> usize {
        match self {
            Self::Point => 0,
            Self::Curve => 1,
            Self::Surface => 2,
            Self::Volume => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementKind {
    Line2,
    Tri3,
    Quad4,
    Tet4,
    Hex8,
}

impl ElementKind {
    pub const fn node_count(self) -> usize {
        match self {
            Self::Line2 => 2,
            Self::Tri3 => 3,
            Self::Quad4 => 4,
            Self::Tet4 => 4,
            Self::Hex8 => 8,
        }
    }

    pub const fn entity_dimension(self) -> EntityDimension {
        match self {
            Self::Line2 => EntityDimension::Curve,
            Self::Tri3 | Self::Quad4 => EntityDimension::Surface,
            Self::Tet4 | Self::Hex8 => EntityDimension::Volume,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell {
    pub kind: ElementKind,
    pub nodes: Vec<NodeId>,
    pub region: Option<RegionId>,
}

impl Cell {
    pub fn new(kind: ElementKind, nodes: Vec<NodeId>) -> Self {
        Self {
            kind,
            nodes,
            region: None,
        }
    }

    pub fn with_region(mut self, region: RegionId) -> Self {
        self.region = Some(region);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub id: RegionId,
    pub name: String,
    pub dimension: EntityDimension,
}

impl Region {
    pub fn new(id: RegionId, name: impl Into<String>, dimension: EntityDimension) -> Self {
        Self {
            id,
            name: name.into(),
            dimension,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshTopology<const D: usize> {
    points: Vec<PointD<D>>,
    cells: Vec<Cell>,
    regions: Vec<Region>,
}

impl<const D: usize> MeshTopology<D> {
    pub fn new(points: Vec<PointD<D>>, cells: Vec<Cell>) -> Result<Self, MeshError> {
        Self::with_regions(points, cells, Vec::new())
    }

    pub fn with_regions(
        points: Vec<PointD<D>>,
        cells: Vec<Cell>,
        regions: Vec<Region>,
    ) -> Result<Self, MeshError> {
        if points.is_empty() {
            return Err(MeshError::EmptyMesh);
        }

        validate_regions(&regions)?;
        for (cell_index, cell) in cells.iter().enumerate() {
            validate_cell(cell_index, cell, points.len(), D, &regions)?;
            validate_cell_geometry(cell_index, cell, &points)?;
        }

        Ok(Self {
            points,
            cells,
            regions,
        })
    }

    pub fn points(&self) -> &[PointD<D>] {
        &self.points
    }

    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    pub const fn dimension(&self) -> usize {
        D
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tri3 {
    pub nodes: [NodeId; 3],
}

impl Tri3 {
    pub const fn new(nodes: [NodeId; 3]) -> Self {
        Self { nodes }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    points: Vec<Point2>,
    triangles: Vec<Tri3>,
    cells: Vec<Cell>,
    regions: Vec<Region>,
}

impl Mesh {
    pub fn new(points: Vec<Point2>, triangles: Vec<Tri3>) -> Result<Self, MeshError> {
        if points.is_empty() {
            return Err(MeshError::EmptyMesh);
        }

        for (triangle_index, triangle) in triangles.iter().enumerate() {
            validate_triangle_indices(triangle_index, triangle, points.len())?;

            let [a, b, c] = triangle.nodes.map(|node| points[node]);
            let twice_area = triangle_twice_area(a, b, c);
            if twice_area.abs() <= f64::EPSILON {
                return Err(MeshError::DegenerateTriangle { triangle_index });
            }
        }

        let cells = triangles
            .iter()
            .map(|triangle| Cell::new(ElementKind::Tri3, triangle.nodes.to_vec()))
            .collect();

        Ok(Self {
            points,
            triangles,
            cells,
            regions: Vec::new(),
        })
    }

    pub fn points(&self) -> &[Point2] {
        &self.points
    }

    pub fn triangles(&self) -> &[Tri3] {
        &self.triangles
    }

    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    pub fn node_count(&self) -> usize {
        self.points.len()
    }

    pub const fn dimension(&self) -> usize {
        2
    }

    pub fn topology(&self) -> Result<MeshTopology<2>, MeshError> {
        MeshTopology::with_regions(
            self.points.iter().copied().map(PointD::from).collect(),
            self.cells.clone(),
            self.regions.clone(),
        )
    }

    pub fn triangle_area(&self, triangle: &Tri3) -> f64 {
        let [a, b, c] = triangle.nodes.map(|node| self.points[node]);
        0.5 * triangle_twice_area(a, b, c).abs()
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum MeshError {
    #[error("mesh must contain at least one point")]
    EmptyMesh,
    #[error("triangle {triangle_index} references node {node_id}, but mesh has {node_count} nodes")]
    InvalidNodeIndex {
        triangle_index: usize,
        node_id: NodeId,
        node_count: usize,
    },
    #[error("triangle {triangle_index} contains duplicate node indices")]
    DuplicateTriangleNode { triangle_index: usize },
    #[error("triangle {triangle_index} has zero area")]
    DegenerateTriangle { triangle_index: usize },
    #[error("cell {cell_index} has {actual} nodes, but {kind:?} requires {expected}")]
    InvalidCellNodeCount {
        cell_index: usize,
        kind: ElementKind,
        expected: usize,
        actual: usize,
    },
    #[error("cell {cell_index} references node {node_id}, but mesh has {node_count} nodes")]
    InvalidCellNodeIndex {
        cell_index: usize,
        node_id: NodeId,
        node_count: usize,
    },
    #[error("cell {cell_index} contains duplicate node indices")]
    DuplicateCellNode { cell_index: usize },
    #[error(
        "cell {cell_index} has topological dimension {element_dimension}, but mesh dimension is {mesh_dimension}"
    )]
    ElementDimensionExceedsMesh {
        cell_index: usize,
        element_dimension: usize,
        mesh_dimension: usize,
    },
    #[error("cell {cell_index} has zero measure")]
    DegenerateCell { cell_index: usize },
    #[error("region {region_id} is defined more than once")]
    DuplicateRegion { region_id: RegionId },
    #[error("cell {cell_index} references region {region_id}, but no such region exists")]
    UnknownRegion {
        cell_index: usize,
        region_id: RegionId,
    },
    #[error(
        "cell {cell_index} with dimension {cell_dimension} cannot belong to region {region_id} with dimension {region_dimension}"
    )]
    RegionDimensionMismatch {
        cell_index: usize,
        region_id: RegionId,
        cell_dimension: usize,
        region_dimension: usize,
    },
}

fn validate_triangle_indices(
    triangle_index: usize,
    triangle: &Tri3,
    node_count: usize,
) -> Result<(), MeshError> {
    let [a, b, c] = triangle.nodes;
    if a == b || a == c || b == c {
        return Err(MeshError::DuplicateTriangleNode { triangle_index });
    }

    for node_id in triangle.nodes {
        if node_id >= node_count {
            return Err(MeshError::InvalidNodeIndex {
                triangle_index,
                node_id,
                node_count,
            });
        }
    }

    Ok(())
}

fn triangle_twice_area(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
}

fn validate_regions(regions: &[Region]) -> Result<(), MeshError> {
    let mut seen = std::collections::BTreeSet::new();
    for region in regions {
        if !seen.insert(region.id) {
            return Err(MeshError::DuplicateRegion {
                region_id: region.id,
            });
        }
    }
    Ok(())
}

fn validate_cell(
    cell_index: usize,
    cell: &Cell,
    node_count: usize,
    mesh_dimension: usize,
    regions: &[Region],
) -> Result<(), MeshError> {
    let expected = cell.kind.node_count();
    let actual = cell.nodes.len();
    if expected != actual {
        return Err(MeshError::InvalidCellNodeCount {
            cell_index,
            kind: cell.kind,
            expected,
            actual,
        });
    }

    if cell.kind.entity_dimension().spatial_dimension() > mesh_dimension {
        return Err(MeshError::ElementDimensionExceedsMesh {
            cell_index,
            element_dimension: cell.kind.entity_dimension().spatial_dimension(),
            mesh_dimension,
        });
    }

    let mut seen_nodes = std::collections::BTreeSet::new();
    for &node_id in &cell.nodes {
        if node_id >= node_count {
            return Err(MeshError::InvalidCellNodeIndex {
                cell_index,
                node_id,
                node_count,
            });
        }
        if !seen_nodes.insert(node_id) {
            return Err(MeshError::DuplicateCellNode { cell_index });
        }
    }

    if let Some(region_id) = cell.region {
        let Some(region) = regions.iter().find(|region| region.id == region_id) else {
            return Err(MeshError::UnknownRegion {
                cell_index,
                region_id,
            });
        };
        let cell_dimension = cell.kind.entity_dimension().spatial_dimension();
        let region_dimension = region.dimension.spatial_dimension();
        if cell_dimension != region_dimension {
            return Err(MeshError::RegionDimensionMismatch {
                cell_index,
                region_id,
                cell_dimension,
                region_dimension,
            });
        }
    }

    Ok(())
}

fn validate_cell_geometry<const D: usize>(
    cell_index: usize,
    cell: &Cell,
    points: &[PointD<D>],
) -> Result<(), MeshError> {
    match cell.kind {
        ElementKind::Line2 => {
            let a = points[cell.nodes[0]].coords;
            let b = points[cell.nodes[1]].coords;
            let length_squared: f64 = a
                .iter()
                .zip(b)
                .map(|(a_value, b_value)| (b_value - a_value).powi(2))
                .sum();
            if length_squared <= f64::EPSILON {
                return Err(MeshError::DegenerateCell { cell_index });
            }
        }
        ElementKind::Tri3 if D >= 2 => {
            let a = points[cell.nodes[0]].coords;
            let b = points[cell.nodes[1]].coords;
            let c = points[cell.nodes[2]].coords;
            let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            if twice_area.abs() <= f64::EPSILON {
                return Err(MeshError::DegenerateCell { cell_index });
            }
        }
        ElementKind::Tet4 if D == 3 => {
            let a = points[cell.nodes[0]].coords;
            let b = points[cell.nodes[1]].coords;
            let c = points[cell.nodes[2]].coords;
            let d = points[cell.nodes[3]].coords;
            let volume_six = determinant_3(
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
                [d[0] - a[0], d[1] - a[1], d[2] - a[2]],
            );
            if volume_six.abs() <= f64::EPSILON {
                return Err(MeshError::DegenerateCell { cell_index });
            }
        }
        _ => {}
    }

    Ok(())
}

fn determinant_3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}
