# Installation

All usage paths require a Rust toolchain because the Python extension is built from the Rust workspace.

Install Rust if needed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Package Use

=== "Python"

    From the repository root:

    ```bash
    uv sync
    source .venv/bin/activate
    ```

    Rebuild the PyO3 extension after changing Rust code exposed to Python:

    ```bash
    uv run maturin develop --release
    ```

=== "Rust"

    The native crate lives in `rust/bubble-gw` and is part of the Rust workspace:

    ```bash
    cd rust
    cargo build -p bubble_gw --release
    cargo test -p bubble_gw
    ```

## Documentation Tools

=== "Python"

    Install the documentation dependencies:

    ```bash
    uv sync --extra docs
    ```

    Serve the guide locally:

    ```bash
    uv run zensical serve
    ```

=== "Rust"

    Generate native Rust API documentation:

    ```bash
    cd rust
    cargo doc -p bubble_gw --no-deps
    ```

    Copy the generated Rust API documentation into the Zensical output from the repository root:

    ```bash
    rm -rf site/rustdoc
    mkdir -p site/rustdoc
    cp -R rust/target/doc/. site/rustdoc/
    ```
