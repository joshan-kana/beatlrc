{
  description = "Offline, beat-synchronised LRC generation";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      systems = [
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      pkgsFor = system: import nixpkgs { inherit system; };
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python312
              uv
              ffmpeg
              libsndfile
              pkg-config
              ruff
            ];
            shellHook = ''
              export UV_PYTHON_DOWNLOADS=never
              export UV_PYTHON="$(command -v python3)"
            '';
          };
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        {
          unit =
            pkgs.runCommand "beatlrc-unit-tests"
              {
                nativeBuildInputs = [ pkgs.python312 ];
                src = self;
              }
              ''
                cd "$src"
                PYTHONPATH=src python3 -m unittest discover -s tests -v
                touch "$out"
              '';
        }
      );

      formatter = forAllSystems (system: (pkgsFor system).nixfmt-tree);
    };
}
