use thiserror::Error;

pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
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

        Ok(Self { points, triangles })
    }

    pub fn points(&self) -> &[Point2] {
        &self.points
    }

    pub fn triangles(&self) -> &[Tri3] {
        &self.triangles
    }

    pub fn node_count(&self) -> usize {
        self.points.len()
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
