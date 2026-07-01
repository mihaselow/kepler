use crate::mesh::{ElementKind, Mesh};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundarySegment {
    pub nodes: [usize; 2],
}

/// Extracts boundary segments from a 2D mesh (edges that belong to only one cell).
pub fn extract_boundary_segments(mesh: &Mesh) -> Vec<BoundarySegment> {
    let mut edge_counts = HashMap::new();

    for cell in mesh.cells() {
        match cell.kind {
            ElementKind::Tri3 => {
                let n0 = cell.nodes[0];
                let n1 = cell.nodes[1];
                let n2 = cell.nodes[2];

                let edges = [(n0, n1), (n1, n2), (n2, n0)];
                for &(u, v) in &edges {
                    let key = if u < v { (u, v) } else { (v, u) };
                    *edge_counts.entry(key).or_insert(0) += 1;
                }
            }
            ElementKind::Quad4 => {
                let n0 = cell.nodes[0];
                let n1 = cell.nodes[1];
                let n2 = cell.nodes[2];
                let n3 = cell.nodes[3];

                let edges = [(n0, n1), (n1, n2), (n2, n3), (n3, n0)];
                for &(u, v) in &edges {
                    let key = if u < v { (u, v) } else { (v, u) };
                    *edge_counts.entry(key).or_insert(0) += 1;
                }
            }
            _ => {}
        }
    }

    let mut boundary = Vec::new();
    for (key, count) in edge_counts {
        if count == 1 {
            boundary.push(BoundarySegment {
                nodes: [key.0, key.1],
            });
        }
    }
    boundary
}

pub struct SpatialHashGrid2D {
    pub cell_size: f64,
    pub segment_buckets: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialHashGrid2D {
    pub fn new(cell_size: f64) -> Self {
        Self {
            cell_size,
            segment_buckets: HashMap::new(),
        }
    }

    pub fn hash_point(&self, x: f64, y: f64) -> (i32, i32) {
        (
            (x / self.cell_size).floor() as i32,
            (y / self.cell_size).floor() as i32,
        )
    }

    pub fn insert_segments(&mut self, mesh: &Mesh, segments: &[BoundarySegment]) {
        self.segment_buckets.clear();
        for (idx, seg) in segments.iter().enumerate() {
            let p0 = mesh.points()[seg.nodes[0]];
            let p1 = mesh.points()[seg.nodes[1]];

            let x_min = p0.x.min(p1.x);
            let x_max = p0.x.max(p1.x);
            let y_min = p0.y.min(p1.y);
            let y_max = p0.y.max(p1.y);

            let (i_min, j_min) = self.hash_point(x_min, y_min);
            let (i_max, j_max) = self.hash_point(x_max, y_max);

            for i in i_min..=i_max {
                for j in j_min..=j_max {
                    self.segment_buckets
                        .entry((i, j))
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
            }
        }
    }

    pub fn query_candidates(&self, x: f64, y: f64) -> Vec<usize> {
        let (i_node, j_node) = self.hash_point(x, y);
        let mut candidates = std::collections::BTreeSet::new();

        for i in (i_node - 1)..=(i_node + 1) {
            for j in (j_node - 1)..=(j_node + 1) {
                if let Some(segs) = self.segment_buckets.get(&(i, j)) {
                    for &seg_idx in segs {
                        candidates.insert(seg_idx);
                    }
                }
            }
        }

        candidates.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Cell, Point2};

    #[test]
    fn test_extract_boundary_segments() {
        // Create 2 contiguous Tri3 cells sharing one edge
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        let cells = vec![
            Cell::new(ElementKind::Tri3, vec![0, 1, 2]),
            Cell::new(ElementKind::Tri3, vec![0, 2, 3]),
        ];

        let mesh = Mesh::new_with_cells(points, cells).unwrap();
        let boundary = extract_boundary_segments(&mesh);

        // Edges (0,1), (1,2), (2,3), (3,0) should each occur once
        // Shared edge (0,2) occurs twice, so it should not be in the boundary.
        assert_eq!(boundary.len(), 4);

        let mut boundary_edges: Vec<_> = boundary
            .iter()
            .map(|seg| {
                let u = seg.nodes[0];
                let v = seg.nodes[1];
                if u < v { (u, v) } else { (v, u) }
            })
            .collect();
        boundary_edges.sort();

        assert_eq!(boundary_edges, vec![(0, 1), (0, 3), (1, 2), (2, 3)]);
    }

    #[test]
    fn test_spatial_hash_grid() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];

        let cells = vec![Cell::new(ElementKind::Quad4, vec![0, 1, 2, 3])];

        let mesh = Mesh::new_with_cells(points, cells).unwrap();
        let boundary = extract_boundary_segments(&mesh);

        let mut grid = SpatialHashGrid2D::new(1.0);
        grid.insert_segments(&mesh, &boundary);

        // A node at (0.5, 0.5) should be close to segments:
        // Segment 0: (0,1) from (0,0) to (2,0) -> overlaps (0,0), (1,0) grid cells
        // Segment 3: (3,0) from (0,2) to (0,0) -> overlaps (0,0), (0,1) grid cells
        let candidates = grid.query_candidates(0.5, 0.5);
        assert!(!candidates.is_empty());
    }
}
