# gw_fopt

`gw_fopt` is a mixed Python/Rust scientific package for first-order phase-transition bubble dynamics and gravitational-wave spectra.

The package has two main Python namespaces:

- `gw_fopt.bubble_dynamics`: Python-side bubble profiles, PDE evolution, fitting utilities, and visualization.
- `gw_fopt.bubble_gw`: Rust-backed gravitational-wave calculations exposed to Python through PyO3.

The native Rust crate is `bubble_gw`, located under `rust/bubble-gw`.

## Main Workflows

- [Two bubbles](two_bubbles.md): solve the `(1+1)D` two-bubble field evolution and compute the exact GW spectrum.
- [Lattice bubbles](lattice_bubbles.md): define lattices, attach bubbles, validate configurations, and generate boundary bubbles.
- [Generalized bulk-flow](generalized_bulkflow.md): compute approximate many-bubble GW spectra from nucleation four-vectors.
- [Sampling utilities](sample.md): generate adaptive one-dimensional grids for expensive scans.

## Minimal Example

=== "Python"

    ```python
    from gw_fopt.bubble_gw import many_bubbles, two_bubbles, utils

    samples = utils.sample(
        start=1.0,
        stop=100.0,
        n_sample=5,
        n_grid=2,
        n_iter=0,
        sample_type="log",
    )
    ```

=== "Rust"

    ```rust
    use bubble_gw::many_bubbles;
    use bubble_gw::two_bubbles;
    ```

## Local Preview

Build and preview the guide with:

```bash
uv sync --extra docs
uv run zensical serve
```

To attach generated Rust API documentation under `site/rustdoc`, run:

```bash
cd rust
cargo doc -p bubble_gw --no-deps
cd ..
rm -rf site/rustdoc
mkdir -p site/rustdoc
cp -R rust/target/doc/. site/rustdoc/
```
