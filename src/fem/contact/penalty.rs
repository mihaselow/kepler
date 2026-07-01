use sprs::TriMat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContactPair {
    pub candidate_node: usize,
    pub segment_nodes: [usize; 2],
}

/// Evaluates a single frictionless node-to-segment contact element using the penalty method.
/// Returns (forces, stiffness, gap) if in contact.
/// Force vector layout: [f_cx, f_cy, f_m1x, f_m1y, f_m2x, f_m2y]
/// Stiffness matrix layout: 6x6
pub fn evaluate_penalty_contact(
    x_c: [f64; 2],
    x_m1: [f64; 2],
    x_m2: [f64; 2],
    penalty: f64,
) -> Option<([f64; 6], [[f64; 6]; 6], f64)> {
    let dx = x_m2[0] - x_m1[0];
    let dy = x_m2[1] - x_m1[1];
    let len_sq = dx * dx + dy * dy;
    if len_sq <= 1e-14 {
        return None;
    }
    let len = len_sq.sqrt();

    // Normal vector pointing outwards (e.g. rotated 90 deg counter-clockwise)
    let nx = -dy / len;
    let ny = dx / len;

    // Projection coordinate xi along the segment
    let rx = x_c[0] - x_m1[0];
    let ry = x_c[1] - x_m1[1];
    let xi = (rx * dx + ry * dy) / len_sq;

    // Check if node projects onto the segment
    if xi < 0.0 || xi > 1.0 {
        return None;
    }

    // Normal gap (negative values represent penetration)
    let gap = rx * nx + ry * ny;

    if gap >= 0.0 {
        return None; // No penetration, no contact
    }

    // N vector: [n_cx, n_cy, n_m1x, n_m1y, n_m2x, n_m2y]
    let n = [nx, ny, -(1.0 - xi) * nx, -(1.0 - xi) * ny, -xi * nx, -xi * ny];

    // Contact force: f = -epsilon_N * gap * N
    let mut forces = [0.0; 6];
    for i in 0..6 {
        forces[i] = -penalty * gap * n[i];
    }

    // Tangent stiffness: K = epsilon_N * N * N^T
    let mut stiffness = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            stiffness[i][j] = penalty * n[i] * n[j];
        }
    }

    Some((forces, stiffness, gap))
}

/// Adds penalty contact contributions to global forces and tangent stiffness.
pub fn assemble_penalty_contact(
    node_coords: &[[f64; 2]],
    u: &[f64],
    pairs: &[ContactPair],
    penalty: f64,
    f_int: &mut [f64],
    triplets: &mut TriMat<f64>,
    bc_set: &std::collections::BTreeSet<usize>,
) {
    for pair in pairs {
        let c = pair.candidate_node;
        let m1 = pair.segment_nodes[0];
        let m2 = pair.segment_nodes[1];

        let x_c = [node_coords[c][0] + u[c * 2], node_coords[c][1] + u[c * 2 + 1]];
        let x_m1 = [node_coords[m1][0] + u[m1 * 2], node_coords[m1][1] + u[m1 * 2 + 1]];
        let x_m2 = [node_coords[m2][0] + u[m2 * 2], node_coords[m2][1] + u[m2 * 2 + 1]];

        if let Some((forces, stiffness, _)) = evaluate_penalty_contact(x_c, x_m1, x_m2, penalty) {
            let global_dofs = [c * 2, c * 2 + 1, m1 * 2, m1 * 2 + 1, m2 * 2, m2 * 2 + 1];

            // Assemble forces
            for i in 0..6 {
                f_int[global_dofs[i]] += forces[i];
            }

            // Assemble stiffness
            for i in 0..6 {
                let r = global_dofs[i];
                if bc_set.contains(&r) {
                    continue;
                }
                for j in 0..6 {
                    let col = global_dofs[j];
                    if bc_set.contains(&col) {
                        continue;
                    }
                    triplets.add_triplet(r, col, stiffness[i][j]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_penalty_contact() {
        let x_c = [0.5, -0.1]; // Penetrating the segment
        let x_m1 = [0.0, 0.0];
        let x_m2 = [1.0, 0.0];
        let penalty = 1e6;

        let result = evaluate_penalty_contact(x_c, x_m1, x_m2, penalty);
        assert!(result.is_some());

        let (forces, stiffness, gap) = result.unwrap();
        assert!(gap < 0.0);
        // Force on candidate node should push it back in the positive y direction (outward normal)
        assert!(forces[1] > 0.0);
        assert_eq!(stiffness[1][1], penalty);
    }
}
