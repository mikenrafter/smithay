use std::ffi::CStr;

use crate::backend::{
    allocator::{Fourcc, Modifier, format::FormatSet},
    vulkan::PhysicalDevice,
};

use super::VulkanError;

/// Top-level Vulkan renderer capabilities.
#[non_exhaustive]
#[derive(Debug, Default, Clone)]
pub struct VulkanRendererCapabilities {
    /// Device availability capabilities.
    pub device: VulkanDeviceCapabilities,
    /// Memory and dmabuf import capabilities.
    pub import: VulkanImportCapabilities,
    /// Export capabilities.
    pub export: VulkanExportCapabilities,
    /// Rendering capabilities.
    pub rendering: VulkanRenderingCapabilities,
    /// Synchronization capabilities.
    pub sync: VulkanSyncCapabilities,
    /// Colour/HDR-adjacent capabilities.
    pub color: VulkanColorCapabilities,
    /// Per-format and per-modifier capabilities for future device-backed probing.
    pub formats: VulkanFormatCapabilities,
}

/// Per-format and per-modifier Vulkan renderer capabilities.
///
/// [`FormatSet`] values contain DRM fourcc/modifier pairs and are empty until device-backed
/// format/modifier discovery exists.
#[non_exhaustive]
#[derive(Debug, Default, Clone)]
pub struct VulkanFormatCapabilities {
    /// Detailed records keyed by DRM format and modifier.
    pub records: Vec<VulkanFormatCapabilityRecord>,
    /// Formats usable as sampled textures.
    pub sampled: FormatSet,
    /// Formats usable for shared-memory uploads.
    pub memory_import: FormatSet,
    /// Formats usable for dmabuf imports.
    pub dmabuf_import: FormatSet,
    /// Formats usable as render targets.
    pub render_target: FormatSet,
    /// Formats exportable as dmabufs.
    pub dmabuf_export: FormatSet,
    /// Formats usable as blit/copy sources.
    pub blit_src: FormatSet,
    /// Formats usable as blit/copy destinations.
    pub blit_dst: FormatSet,
    /// Formats usable for linear/readback paths.
    pub linear: FormatSet,
}

/// Capability record for a DRM format and modifier pair.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanFormatCapabilityRecord {
    /// DRM format code.
    pub format: Fourcc,
    /// Modifier discovered for this format.
    pub modifier: Modifier,
    /// Per-usage capability bits for this format/modifier pair.
    pub usages: VulkanFormatUsage,
}

/// Per-usage capability bits for a Vulkan format record.
#[non_exhaustive]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VulkanFormatUsage {
    /// Usable as a sampled texture.
    pub sampled: bool,
    /// Usable for memory uploads.
    pub memory_import: bool,
    /// Usable for dmabuf imports.
    pub dmabuf_import: bool,
    /// Usable as a render target.
    pub render_target: bool,
    /// Exportable as a dmabuf.
    pub dmabuf_export: bool,
    /// Usable as a blit/copy source.
    pub blit_src: bool,
    /// Usable as a blit/copy destination.
    pub blit_dst: bool,
    /// Usable for linear/readback paths.
    pub linear: bool,
}

impl VulkanFormatCapabilities {
    /// Discovers Vulkan renderer format capabilities for a physical device.
    ///
    /// This is a stub in the scaffold. The future implementation must query every `(format,
    /// modifier)` pair for each intended Vulkan image usage and must not advertise linux-dmabuf
    /// pairs that `ImportDma` cannot import.
    pub fn discover(_physical_device: &PhysicalDevice) -> Result<Self, VulkanError> {
        Err(VulkanError::UnsupportedOperation("format capability discovery"))
    }
}

impl VulkanRendererCapabilities {
    pub(super) fn for_initialized_device(enabled_extensions: &[&CStr]) -> Self {
        Self {
            device: VulkanDeviceCapabilities {
                available: true,
                multi_gpu: false,
                extensions: enabled_extensions
                    .iter()
                    .map(|extension| extension.to_string_lossy().into_owned())
                    .collect(),
            },
            ..Self::default()
        }
    }
}

/// Vulkan device availability capabilities.
#[non_exhaustive]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VulkanDeviceCapabilities {
    /// Whether a Vulkan device is available to this renderer.
    pub available: bool,
    /// Whether multi-GPU operation is supported.
    pub multi_gpu: bool,
    /// Device extensions known to be available for renderer use.
    pub extensions: Vec<String>,
}

/// Vulkan import capabilities.
#[non_exhaustive]
#[derive(Debug, Default, Clone)]
pub struct VulkanImportCapabilities {
    /// Whether memory imports are supported.
    pub memory: bool,
    /// Whether dmabuf imports are supported.
    pub dmabuf: bool,
    /// Whether modifier-aware dmabuf imports are supported.
    pub modifiers: bool,
}

/// Vulkan export capabilities.
#[non_exhaustive]
#[derive(Debug, Default, Clone)]
pub struct VulkanExportCapabilities {
    /// Whether dmabuf export is supported.
    pub dmabuf: bool,
    /// Whether modifier-aware export is supported.
    pub modifiers: bool,
}

/// Vulkan rendering capabilities.
#[non_exhaustive]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VulkanRenderingCapabilities {
    /// Whether offscreen rendering is supported.
    pub offscreen: bool,
    /// Whether blit/copy operations are supported.
    pub blit: bool,
    /// Whether 10-bit render targets are supported.
    pub render_target_10bit: bool,
    /// Whether fp16 render targets are supported.
    pub render_target_fp16: bool,
}

/// Vulkan synchronization capabilities.
#[non_exhaustive]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VulkanSyncCapabilities {
    /// Whether explicit synchronization is supported.
    pub explicit: bool,
}

/// Vulkan colour and HDR-adjacent capabilities.
#[non_exhaustive]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VulkanColorCapabilities {
    /// Whether colour transform hooks are supported.
    pub color_transform_hooks: bool,
    /// Whether HDR-ready targets are supported.
    pub hdr_ready_targets: bool,
}
