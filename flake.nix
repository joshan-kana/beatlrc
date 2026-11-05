{
  description = "Offline, beat-synchronised LRC generation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
    }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
      inherit (pkgs) lib;
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      python = pkgs.python312;
      pythonBase = pkgs.callPackage pyproject-nix.build.packages { inherit python; };
      pythonSet = pythonBase.overrideScope (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
          (final: prev: {
            # MLX's Metal wheel installs libmlx.dylib in a second package.
            # Its binary extension must resolve that Nix store path directly.
            mlx = prev.mlx.overrideAttrs (old: {
              postFixup = (old.postFixup or "") + ''
                for extension in "$out/${python.sitePackages}/mlx/core."*.so; do
                  install_name_tool -add_rpath \
                    "${final.mlx-metal}/${python.sitePackages}/mlx/lib" \
                    "$extension"
                  /usr/bin/codesign -f -s - "$extension"
                done
              '';
            });
          })
        ]
      );
      venv = pythonSet.mkVirtualEnv "lyr-env" workspace.deps.all;
      inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
      application = mkApplication {
        inherit venv;
        package = pythonSet.lyr;
      };
      package = pkgs.symlinkJoin {
        name = "lyr";
        paths = [ application ];
        nativeBuildInputs = [ pkgs.makeWrapper ];
        postBuild = ''
          wrapProgram "$out/bin/lyr" \
            --prefix PATH : ${lib.makeBinPath [ pkgs.ffmpeg ]}
        '';
      };
    in
    {
      packages.${system} = {
        default = package;
        lyr = package;
      };

      apps.${system}.default = {
        type = "app";
        program = "${package}/bin/lyr";
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          package
          python
          pkgs.ruff
          pkgs.uv
        ];
      };

      checks.${system} = {
        package = package;
        unit =
          pkgs.runCommand "lyr-unit-tests"
            {
              nativeBuildInputs = [ python ];
              src = self;
            }
            ''
              cd "$src"
              PYTHONPATH=src python3 -m unittest discover -s tests -v
              touch "$out"
            '';
      };

      formatter.${system} = pkgs.nixfmt-tree;
    };
}
