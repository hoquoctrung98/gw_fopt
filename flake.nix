{
  description = "gw_fopt lightweight native dependency shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    devShells = forAllSystems (system: let
      pkgs = import nixpkgs {inherit system;};
      nativeLibraries = with pkgs; [
        # stdenv.cc.cc.lib
        # openssl
        # libffi
        # hdf5
        # openblas
      ];
    in {
      default = pkgs.mkShell {
        packages = with pkgs; [
          pkg-config
          git
        ];

        env = {
          UV_PYTHON_DOWNLOADS = "managed";
          UV_LINK_MODE = "copy";
          RUST_BACKTRACE = "1";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath nativeLibraries;
        };

        shellHook = ''
          echo "gw_fopt Nix shell"
          echo "  Nix:    native libraries and build helpers only"
          echo "  Python: managed by uv"
          echo "  Rust:   managed by your external toolchain"
          echo ""

          if ! command -v uv >/dev/null 2>&1; then
            echo "warning: uv is not on PATH. Install uv outside Nix or expose it before entering this shell."
          fi

          if ! command -v cargo >/dev/null 2>&1; then
            echo "warning: cargo is not on PATH. Load your external Rust toolchain before entering this shell."
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
