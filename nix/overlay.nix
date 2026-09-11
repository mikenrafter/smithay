# Overlay for NixOS users who want to replace nixpkgs cosmic-comp with this
# VulkanRenderer bring-up (smithay e3d461a pin + cosmic-comp vulkan branch).
#
# Exclusive KMS Vulkan, not default COSMIC GLES. Do not mix with GLES
# hybrid-export smithay patches. cargoHash changes when either tree changes;
# override it if fetchCargoVendor fails:
#
#   nixpkgs.overlays = [
#     (inputs.smithay-vulkan.overlays.cosmic-vulkan {
#       cargoHash = "sha256-...";
#     })
#   ];
#
{ cosmic-comp, smithay }:
{
  cargoHash ? "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
}:
final: prev:
let
  src = prev.runCommand "cosmic-comp-vulkan-src" { } ''
    cp -a ${cosmic-comp}/. "$out"
    chmod -R u+w "$out"
    rm -rf "$out/smithay" "$out/target" "$out/result"
    cp -a ${smithay} "$out/smithay"
    chmod -R u+w "$out/smithay"
    rm -rf "$out/smithay/target" "$out/smithay/result" "$out/smithay/.git"
    substituteInPlace "$out/Cargo.toml" \
      --replace-fail 'smithay = { path = "../smithay" }' \
                     'smithay = { path = "./smithay" }'
  '';
  version = "${oldVersion}-vulkan";
  oldVersion = prev.cosmic-comp.version or "1.0.0";
  libdisplayInfo =
    prev.libdisplay-info_0_3 or prev.libdisplay-info;
in
{
  cosmic-comp = prev.cosmic-comp.overrideAttrs (old: {
    inherit src version;
    cargoHash = cargoHash;
    cargoDeps = prev.rustPlatform.fetchCargoVendor {
      inherit src version;
      inherit (old) pname;
      hash = cargoHash;
    };
    buildFeatures = (old.buildFeatures or [ ]) ++ [ "renderer_vulkan" ];
    cargoBuildFeatures = (old.cargoBuildFeatures or [ ]) ++ [ "renderer_vulkan" ];
    buildInputs =
      (builtins.filter (
        p: (p.pname or "") != "libdisplay-info"
      ) (old.buildInputs or [ ]))
      ++ [
        prev.vulkan-loader
        libdisplayInfo
      ];
  });
}
