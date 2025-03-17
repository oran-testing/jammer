{
  description = "RT Jammer";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
  let
    pkgs = import nixpkgs { system = "x86_64-linux"; };
    nativeBuildInputs = with pkgs; [
      ninja
      cmake
      gcc
      uhd
      yaml-cpp
      pkg-config
      boost
    ];
  in
  {
    inherit nativeBuildInputs;

    packages.x86_64-linux.default = pkgs.stdenv.mkDerivation {
      pname = "jammer";
      version = "1.0";

      src = ./.;
      inherit nativeBuildInputs;

      installPhase = ''
        mkdir -p $out/bin
        cp jammer $out/bin
      '';
    };

    devShells.x86_64-linux.default = pkgs.mkShell {
      inherit nativeBuildInputs;
    };

  };
}
