# Abaqus INP Verification Benchmarks

These fixtures exercise Kepler's `*NODE` / `*ELEMENT` / `*MATERIAL` / `*BOUNDARY` / `*CLOAD`
import path and compare Kepler elasticity results against analytical references.

All models use SI units (m, N, Pa) unless noted.

## Benchmarks

| File | Problem | Reference | Typical command |
|------|---------|-----------|-----------------|
| `uniaxial_patch.inp` | Single CPS4, uniform x-tension | `σ_xx = F/A`, `u_x = σ L / E` | see below |
| `cantilever.inp` | 10×2 CPS4 cantilever, tip load | Euler–Bernoulli `δ = PL³/(3EI)` (35% coarse-mesh tol.) | see below |
| `block.inp` | Mixed CPS4/CPS3 import smoke test | parse-only | `cargo test abaqus_import` |

Companion `*.verify.json` files list expected displacements/stresses and tolerances.

### Run a verification case

```bash
cargo run --example solve_inp -- examples/data/abaqus/uniaxial_patch.inp
cargo run --example solve_inp -- examples/data/abaqus/cantilever.inp
```

Optional VTK output:

```bash
cargo run --example solve_inp -- examples/data/abaqus/uniaxial_patch.inp output.vtk
```

### Automated tests

```bash
cargo test --test abaqus_verification
```

## Reference formulas

**Uniaxial patch** (plane stress, thickness `t`, width `h`, length `L`):

- `σ_xx = F_x / (h · t)`
- `ε_xx = σ_xx / E`
- `u_x(x) = ε_xx · x`

**Cantilever** (unit depth `b = t = 1`, height `h`, length `L`, tip load `P`):

- `I = b h³ / 12`
- `δ_tip = P L³ / (3 E I)`

The 10×2 CPS4 mesh for `cantilever.inp` is expected to match beam theory within ~35% (documented in the JSON tolerance).

## External benchmarks

These cases are inspired by standard FEM patch-test and beam-theory references (e.g. MacNeal & Harder patch tests, Timoshenko beam theory). They are authored for Kepler rather than copied from vendor input decks, so they stay small and license-friendly.
