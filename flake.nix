{
  description = "nix flake for ZiHuan AIBot development";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Python 环境
            python311
            uv
            just
            poppler_utils
            nodePackages.localtunnel

            # C++ 开发工具
            cmake
            ninja
            gtest
            boost
            fmt
            spdlog
            yaml-cpp
            ncurses
            openssl
            onnxruntime

            # 数据库
            mysql80
            weaviate
          ];

          buildInputs = with pkgs; [
            gcc
            clang
            clang-tools
            stdenv.cc.cc.lib
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath (with pkgs; [
              stdenv.cc.cc.lib
              zlib
              boost
              fmt
              yaml-cpp
              mysql80
              onnxruntime
            ])}
            if [ ! -d ".venv" ]; then
              python -m venv .venv
            fi
            source .venv/bin/activate
          '';
        };
      });
}