//! Parallel assembly utilities for FEM stiffness and mass matrix construction.
use rayon::prelude::*;
use sprs::TriMat;

/// A sparse matrix entry produced by one element.
#[derive(Debug, Clone)]
pub struct Triplet {
    pub row: usize,
    pub col: usize,
    pub val: f64,
}

/// Merges per-element triplet vecs into a TriMat.
pub fn merge_triplets(size: usize, element_triplets: Vec<Vec<Triplet>>) -> TriMat<f64> {
    let total: usize = element_triplets.iter().map(|v| v.len()).sum();
    let mut tri = TriMat::with_capacity((size, size), total);
    for triplets in element_triplets {
        for t in triplets {
            tri.add_triplet(t.row, t.col, t.val);
        }
    }
    tri
}

/// Parallel map over elements, computing Vec<Triplet> for each.
pub fn par_assemble<T, F>(elements: &[T], f: F) -> Vec<Vec<Triplet>>
where
    T: Sync,
    F: Fn(usize, &T) -> Option<Vec<Triplet>> + Sync,
{
    elements
        .par_iter()
        .enumerate()
        .filter_map(|(i, elem)| f(i, elem))
        .collect()
}
