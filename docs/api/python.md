# Python API

The Python package is split between pure-Python workflow utilities and Rust-backed GW calculations.

## `gw_fopt.bubble_dynamics`

::: gw_fopt.bubble_dynamics

## Bubble Simulation

::: gw_fopt.bubble_dynamics.bubble_simulator

## Utilities

::: gw_fopt.bubble_dynamics.utils

## Visualization

::: gw_fopt.bubble_dynamics.visualizer

## `gw_fopt.bubble_gw`

The `gw_fopt.bubble_gw` namespace is backed by the PyO3 extension built from `rust/py-bubble-gw`.
Its public submodules are:

- `gw_fopt.bubble_gw.two_bubbles`
- `gw_fopt.bubble_gw.many_bubbles`
- `gw_fopt.bubble_gw.utils`

The complete native implementation is documented in the [Rust API](rust.md).
