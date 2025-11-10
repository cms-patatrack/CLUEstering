{
  description = "C++ dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
    in pkgs.mkShell {
      buildInputs = [
        pkgs.gcc12
        pkgs.gcc13
        pkgs.llvmPackages_15.clang
        pkgs.boost
        pkgs.tbb
        pkgs.libomp
        pkgs.cmake
      ];
    };
  };
}
