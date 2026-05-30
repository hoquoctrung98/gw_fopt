{
  description = "gw_fopt lightweight native dependency shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    nixpkgs,
    rust-overlay,
    ...
  }: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    devShells = forAllSystems (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [rust-overlay.overlays.default];
      };
      nativeLibraries = with pkgs; [
        # hdf5
        # openblas
      ];
    in {
      default = pkgs.mkShell {
        packages = with pkgs; [
          pkg-config
          git
          uv
          (rust-bin.nightly.latest.default.override {
            extensions = [
              "clippy"
              "rustfmt"
            ];
          })
        ];

        env = {
          UV_PYTHON_DOWNLOADS = "auto";
          UV_LINK_MODE = "clone";
          RUST_BACKTRACE = "1";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath nativeLibraries;
        };

        shellHook = ''
          echo "gw_fopt Nix shell"
          echo "  Nix:    native libraries, uv, and Rust toolchain"
          echo "  Python: managed by uv"
          echo "  Rust:   managed by nixpkgs + rust-overlay"
          echo ""

          if ! command -v uv >/dev/null 2>&1; then
            echo "warning: uv is not on PATH."
          fi

          echo "Typical workflow:"
          echo "  uv sync"
          echo "  source .venv/bin/activate"
          echo "  uv run maturin develop --release"
        '';
      };
    });
  };
}
