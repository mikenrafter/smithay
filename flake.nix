{
  description = "VulkanRenderer smithay (COSMIC e3d461a pin) plus a NixOS overlay that vendors this tree into cosmic-comp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    cosmic-comp.url = "github:Pheoxy/cosmic-comp/vulkan-renderer-e3d461a";
    cosmic-comp.flake = false;
  };

  outputs =
    {
      self,
      nixpkgs,
      cosmic-comp,
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      mkOverlay = import ./nix/overlay.nix {
        inherit cosmic-comp;
        smithay = self;
      };
    in
    {
      overlays.cosmic-vulkan = mkOverlay;
      overlays.default = mkOverlay { };

      nixosModules.cosmic-vulkan = import ./nix/module.nix {
        overlay = self.overlays.default;
      };
      nixosModules.default = self.nixosModules.cosmic-vulkan;

      devShells = forAllSystems (system: {
        profiling = import ./nix/profiling-shell.nix {
          pkgs = nixpkgs.legacyPackages.${system};
        };
        default = self.devShells.${system}.profiling;
      });
    };
}
