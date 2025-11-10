{
  description = "C++ dev environment with multiple compilers";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux = {
      # Shells for different compilers
      gcc12 = let pkgs = import nixpkgs { system = "x86_64-linux"; }; in
        pkgs.mkShell {
          buildInputs = [
            pkgs.gcc12
            pkgs.boost
            pkgs.tbb
            pkgs.libomp
            pkgs.cmake
          ];
        };

      gcc13 = let pkgs = import nixpkgs { system = "x86_64-linux"; }; in
        pkgs.mkShell {
          buildInputs = [
            pkgs.gcc13
            pkgs.boost
            pkgs.tbb
            pkgs.libomp
            pkgs.cmake
          ];
        };

      clang15 = let pkgs = import nixpkgs { system = "x86_64-linux"; }; in
        pkgs.mkShell {
          buildInputs = [
            pkgs.llvmPackages_15.clang
            pkgs.boost
            pkgs.tbb
            pkgs.libomp
            pkgs.cmake
          ];
        };
    };
  };
}
