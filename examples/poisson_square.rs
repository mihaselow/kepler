use kepler::{Mesh, Point2, PoissonProblem, SolverOptions, Tri3, solve_poisson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = Mesh::new(
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.5, 0.5),
        ],
        vec![
            Tri3::new([0, 1, 4]),
            Tri3::new([1, 2, 4]),
            Tri3::new([2, 3, 4]),
            Tri3::new([3, 0, 4]),
        ],
    )?;

    let problem = PoissonProblem {
        conductivity: 1.0,
        source: |_, _| 1.0,
        dirichlet: vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)],
    };

    let result = solve_poisson(&mesh, &problem, SolverOptions::default())?;

    for (node_id, value) in result.values.iter().enumerate() {
        println!("u[{node_id}] = {value:.6}");
    }

    Ok(())
}
