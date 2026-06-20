# C/C++ bindings for `bubble_gw`

This crate exposes a C ABI over the Rust `bubble_gw` crate. It is intentionally
separate from both the Rust core crate and the PyO3 binding crate:

- `../bubble-gw`: Rust numerical implementation.
- `../py-bubble-gw`: Python bindings.
- `../c-bubble-gw`: C/C++ ABI and generated headers.

## Build

From `rust/`:

```bash
cargo build -p c-bubble-gw --release
```

This produces dynamic and static libraries under `rust/target/release/`, with
the public header committed at `include/bubble_gw.h`.

## Header generation

The header is managed with `cbindgen`. Regenerate it from `rust/` with:

```bash
cbindgen --config c-bubble-gw/cbindgen.toml \
  --crate c-bubble-gw \
  --lang c \
  --output c-bubble-gw/include/bubble_gw.h
```

The committed header is C++ compatible through `extern "C"` guards. If a
separate C++-flavored header is desired later, use the same config with
`--lang c++`.

## API conventions

- All exported functions use the `bgw_` prefix.
- Rust-owned objects are opaque handles and must be released with their matching
  `bgw_*_free` function.
- Numerical arrays are caller-owned, C-contiguous, row-major buffers.
- Functions return `BgwStatus`; details for the latest failure are available via
  `bgw_last_error_message()`.
- Optional floating-point defaults use `NAN` as the C-side sentinel.
