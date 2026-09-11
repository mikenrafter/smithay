# Profiling / bring-up tools for VulkanRenderer + cosmic-comp.
#   nix develop github:Pheoxy/smithay/add-vulkan-renderer-support-cosmic-e3d461a#profiling
{ pkgs }:
let
  lib = pkgs.lib;
  libs = with pkgs; [
    wayland
    libxkbcommon
    libglvnd
    mesa
    libdrm
    libgbm
    vulkan-loader
    vulkan-headers
    vulkan-validation-layers
    pixman
    libinput
    seatd
    systemd
    libx11
    libxcursor
    libxrandr
    libxi
    libxext
  ];
in
pkgs.mkShell {
  name = "smithay-vulkan-profiling";
  packages = with pkgs; [
    rustc
    cargo
    rustfmt
    clippy
    pkg-config
    tracy
    vulkan-tools
    vulkan-validation-layers
    vulkan-loader
    renderdoc
    perf
    hotspot
    cargo-flamegraph
    drm_info
    shaderc
    spirv-tools
    gfxreconstruct
  ];
  buildInputs = libs;
  shellHook = ''
    export LD_LIBRARY_PATH="${lib.makeLibraryPath libs}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [ -d /run/opengl-driver/share/vulkan/icd.d ]; then
      export VK_DRIVER_FILES="$(printf '%s:' /run/opengl-driver/share/vulkan/icd.d/*.json)"
      export VK_DRIVER_FILES="''${VK_DRIVER_FILES%:}"
    fi
    if [ -d ${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d ]; then
      export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d''${VK_LAYER_PATH:+:$VK_LAYER_PATH}"
    fi
    echo "smithay-vulkan-profiling: tracy, renderdoc, perf/hotspot, cargo-flamegraph, vulkan-tools, drm_info"
    echo "cosmic-comp tracy: cargo run --features profile-with-tracy"
  '';
}
