# NixOS module: apply the Vulkan cosmic-comp overlay and mark the session.
# Importing this replaces GLES cosmic-comp. Keep it off the default generation
# if you still want epoch COSMIC; use a specialisation or an opt-in host.
{ overlay }:
{ lib, ... }:
{
  nixpkgs.overlays = [ overlay ];
  environment.sessionVariables.COSMIC_RENDERER = "vulkan";
}
