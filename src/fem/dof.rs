use crate::mesh::NodeId;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DOFManager {
    // Maps (NodeId, FieldName) -> Global Equation Index
    dof_map: BTreeMap<(NodeId, String), usize>,
    total_dofs: usize,
}

impl DOFManager {
    /// Creates a new, empty degree of freedom manager.
    pub fn new() -> Self {
        Self {
            dof_map: BTreeMap::new(),
            total_dofs: 0,
        }
    }

    /// Registers a degree of freedom at a node for a specific field/variable,
    /// returning the global equation index. If it is already registered,
    /// returns the existing index.
    pub fn register_dof(&mut self, node: NodeId, field: &str) -> usize {
        let key = (node, field.to_owned());
        if let Some(&id) = self.dof_map.get(&key) {
            id
        } else {
            let id = self.total_dofs;
            self.dof_map.insert(key, id);
            self.total_dofs += 1;
            id
        }
    }

    /// Returns the global equation index for a given node's degree of freedom,
    /// or `None` if it has not been registered.
    pub fn get_eq_index(&self, node: NodeId, field: &str) -> Option<usize> {
        self.dof_map.get(&(node, field.to_owned())).copied()
    }

    /// Returns the total number of unconstrained degrees of freedom registered.
    pub fn total_dof_count(&self) -> usize {
        self.total_dofs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dof_manager_registration() {
        let mut dm = DOFManager::new();
        assert_eq!(dm.total_dof_count(), 0);

        // Register first DOF
        let idx0_u = dm.register_dof(0, "u");
        assert_eq!(idx0_u, 0);
        assert_eq!(dm.total_dof_count(), 1);

        // Register same DOF again (should return same index)
        assert_eq!(dm.register_dof(0, "u"), 0);
        assert_eq!(dm.total_dof_count(), 1);

        // Register second DOF at same node, different field
        let idx0_v = dm.register_dof(0, "v");
        assert_eq!(idx0_v, 1);
        assert_eq!(dm.total_dof_count(), 2);

        // Register DOF at different node
        let idx1_u = dm.register_dof(1, "u");
        assert_eq!(idx1_u, 2);
        assert_eq!(dm.total_dof_count(), 3);

        // Test retrieval
        assert_eq!(dm.get_eq_index(0, "u"), Some(0));
        assert_eq!(dm.get_eq_index(0, "v"), Some(1));
        assert_eq!(dm.get_eq_index(1, "u"), Some(2));
        assert_eq!(dm.get_eq_index(1, "v"), None);
    }
}
