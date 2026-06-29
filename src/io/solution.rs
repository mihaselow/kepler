use std::{fmt::Write as _, fs, path::Path};

use crate::{fem::poisson::PoissonResult, io::FileIoError};

pub fn write_solution_file(
    path: impl AsRef<Path>,
    result: &PoissonResult,
) -> Result<(), FileIoError> {
    let path = path.as_ref();
    let output = format_solution(result);
    fs::write(path, output).map_err(|source| FileIoError::Write {
        path: path.to_owned(),
        source,
    })
}

pub fn format_solution(result: &PoissonResult) -> String {
    let mut output = String::new();
    writeln!(&mut output, "# kepler solution").expect("writing to string cannot fail");
    writeln!(&mut output, "# iterations {}", result.iterations)
        .expect("writing to string cannot fail");
    writeln!(&mut output, "# residual_norm {}", result.residual_norm)
        .expect("writing to string cannot fail");
    writeln!(&mut output, "node value").expect("writing to string cannot fail");

    for (node_id, value) in result.values.iter().enumerate() {
        writeln!(&mut output, "{node_id} {value}").expect("writing to string cannot fail");
    }

    output
}
