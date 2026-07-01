use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeplerResultFile {
    pub schema_version: u32,
    pub mesh: KeplerResultMesh,
    pub steps: Vec<KeplerResultStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeplerResultMesh {
    pub points: Vec<[f64; 3]>,
    pub cells: Vec<KeplerResultCell>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeplerResultCell {
    pub kind: String,
    pub nodes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeplerResultStep {
    pub time: f64,
    pub displacements: Vec<[f64; 3]>,
    pub stress: Option<Vec<[f64; 6]>>,
}

#[derive(Debug, Error)]
pub enum ResultIoError {
    #[error("failed to read result file {path}")]
    Read {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write result file {path}")]
    Write {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize result JSON")]
    Json(#[from] serde_json::Error),
    #[error("HDF5 support is not enabled; build with `--features hdf5`")]
    Hdf5Unavailable,
    #[cfg(feature = "hdf5")]
    #[error("HDF5 error: {0}")]
    Hdf5(String),
}

pub const RESULT_SCHEMA_VERSION: u32 = 1;

pub fn write_json_result(path: impl AsRef<Path>, result: &KeplerResultFile) -> Result<(), ResultIoError> {
    let path = path.as_ref();
    let payload = serde_json::to_vec_pretty(result)?;
    fs::write(path, payload).map_err(|source| ResultIoError::Write {
        path: path.to_owned(),
        source,
    })
}

pub fn read_json_result(path: impl AsRef<Path>) -> Result<KeplerResultFile, ResultIoError> {
    let path = path.as_ref();
    let payload = fs::read(path).map_err(|source| ResultIoError::Read {
        path: path.to_owned(),
        source,
    })?;
    Ok(serde_json::from_slice(&payload)?)
}

pub fn write_hdf5_result(path: impl AsRef<Path>, result: &KeplerResultFile) -> Result<(), ResultIoError> {
    #[cfg(feature = "hdf5")]
    {
        use hdf5::prelude::*;
        let path = path.as_ref();
        let file = File::create(path).map_err(|source| ResultIoError::Write {
            path: path.to_owned(),
            source,
        })?;
        let group = file
            .create_group("mesh")
            .map_err(|error| ResultIoError::Hdf5(error.to_string()))?;
        let points: Vec<f64> = result
            .mesh
            .points
            .iter()
            .flat_map(|point| point.iter().copied())
            .collect();
        group
            .new_dataset::<f64>()
            .shape([result.mesh.points.len(), 3])
            .create("nodes")
            .map_err(|error| ResultIoError::Hdf5(error.to_string()))?
            .write(&points)
            .map_err(|error| ResultIoError::Hdf5(error.to_string()))?;

        let steps = file
            .create_group("steps")
            .map_err(|error| ResultIoError::Hdf5(error.to_string()))?;
        for (index, step) in result.steps.iter().enumerate() {
            let step_group = steps
                .create_group(&format!("step_{index}"))
                .map_err(|error| ResultIoError::Hdf5(error.to_string()))?;
            let flat: Vec<f64> = step
                .displacements
                .iter()
                .flat_map(|value| value.iter().copied())
                .collect();
            step_group
                .new_dataset::<f64>()
                .shape([step.displacements.len(), 3])
                .create("displacements")
                .map_err(|error| ResultIoError::Hdf5(error.to_string()))?
                .write(&flat)
                .map_err(|error| ResultIoError::Hdf5(error.to_string()))?;
        }
        Ok(())
    }
    #[cfg(not(feature = "hdf5"))]
    {
        let _ = (path, result);
        Err(ResultIoError::Hdf5Unavailable)
    }
}

pub fn write_result_file(
    path: impl AsRef<Path>,
    result: &KeplerResultFile,
    prefer_hdf5: bool,
) -> Result<(), ResultIoError> {
    if prefer_hdf5 {
        match write_hdf5_result(&path, result) {
            Ok(()) => return Ok(()),
            Err(ResultIoError::Hdf5Unavailable) => {}
            Err(error) => return Err(error),
        }
    }
    let json_path = path.as_ref().with_extension("json");
    write_json_result(json_path, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_result_round_trip() {
        let result = KeplerResultFile {
            schema_version: RESULT_SCHEMA_VERSION,
            mesh: KeplerResultMesh {
                points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                cells: vec![KeplerResultCell {
                    kind: "tri3".to_string(),
                    nodes: vec![0, 1],
                }],
            },
            steps: vec![KeplerResultStep {
                time: 0.0,
                displacements: vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                stress: None,
            }],
        };
        let path = std::env::temp_dir().join("kepler_result_test.json");
        write_json_result(&path, &result).unwrap();
        let loaded = read_json_result(&path).unwrap();
        assert_eq!(loaded, result);
        let _ = fs::remove_file(path);
    }
}
