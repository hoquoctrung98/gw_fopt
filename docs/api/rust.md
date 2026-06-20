# Rust API

The native Rust crate is `bubble_gw`, located in `rust/bubble-gw`.

Build the Rust API documentation with:

```bash
cd rust
cargo doc -p bubble_gw --no-deps
```

After building the Zensical site, copy generated Rust documentation into `site/rustdoc` from the repository root:

```bash
rm -rf site/rustdoc
mkdir -p site/rustdoc
cp -R rust/target/doc/. site/rustdoc/
```

Then open the copied API docs at <a href="/rustdoc/bubble_gw/index.html">/rustdoc/bubble_gw/index.html</a>.

The Rust API is intentionally kept usable independently from the Python bindings. PyO3 wrapper code lives in `rust/py-bubble-gw`.
