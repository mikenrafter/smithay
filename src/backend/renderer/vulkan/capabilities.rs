use std::ffi::CStr;

use ash::vk;

use crate::backend::{
    allocator::{Format, Fourcc, Modifier, format::FormatSet},
    vulkan::PhysicalDevice,
};

use super::{VulkanError, format::renderer_format_infos};

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
    /// Raw per-format Vulkan image feature capabilities.
    pub formats: VulkanFormatCapabilities,
}

/// Raw per-format Vulkan image feature capabilities.
///
/// [`FormatSet`] values use Smithay's DRM format representation, but non-empty sets in this
/// structure do not imply that the matching Smithay renderer trait is implemented. In particular,
/// `Modifier::Invalid` records internal Vulkan optimal-tiling support and must not be advertised as
/// a linux-dmabuf modifier pair.
#[non_exhaustive]
#[derive(Debug, Default, Clone)]
pub struct VulkanFormatCapabilities {
    /// Detailed records keyed by DRM format and renderer tiling marker.
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
    /// Formats usable as blit sources.
    pub blit_src: FormatSet,
    /// Formats usable as blit destinations.
    pub blit_dst: FormatSet,
    /// Formats usable as transfer/copy sources.
    pub transfer_src: FormatSet,
    /// Formats usable as transfer/copy destinations.
    pub transfer_dst: FormatSet,
    /// Formats usable for linear/readback paths.
    pub linear: FormatSet,
}

/// Capability record for a DRM format and renderer tiling marker.
///
/// `Modifier::Invalid` means Vulkan optimal tiling for renderer-internal images, not an explicit
/// dmabuf modifier. `Modifier::Linear` means Vulkan linear tiling.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanFormatCapabilityRecord {
    /// DRM format code.
    pub format: Fourcc,
    /// Renderer tiling marker for this format.
    pub modifier: Modifier,
    /// Per-usage capability bits for this format and tiling marker.
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
    /// Usable as a blit source.
    pub blit_src: bool,
    /// Usable as a blit destination.
    pub blit_dst: bool,
    /// Usable as a transfer/copy source.
    pub transfer_src: bool,
    /// Usable as a transfer/copy destination.
    pub transfer_dst: bool,
    /// Usable for linear/readback paths.
    pub linear: bool,
}

impl VulkanFormatCapabilities {
    /// Discovers Vulkan renderer format capabilities for a physical device.
    ///
    /// This probes sampled, render-target, blit, transfer, and linear tiling support for the renderer's
    /// static format table. Import and export format sets remain empty until those traits are
    /// implemented and can import or export the advertised pairs.
    pub fn discover(physical_device: &PhysicalDevice) -> Result<Self, VulkanError> {
        let mut records = Vec::new();
        let mut sampled = Vec::new();
        let mut render_target = Vec::new();
        let mut blit_src = Vec::new();
        let mut blit_dst = Vec::new();
        let mut transfer_src = Vec::new();
        let mut transfer_dst = Vec::new();
        let mut linear = Vec::new();

        for info in renderer_format_infos() {
            let mut properties = vk::FormatProperties2::default();

            unsafe { physical_device.get_format_properties(info.vk_format, &mut properties) };

            let format_properties = properties.format_properties;
            let optimal_usage = format_usage_from_features(format_properties.optimal_tiling_features);
            if optimal_usage.any_supported() {
                let format = format_with_modifier(info.fourcc, Modifier::Invalid);

                if optimal_usage.sampled {
                    sampled.push(format);
                }
                if optimal_usage.render_target {
                    render_target.push(format);
                }
                if optimal_usage.blit_src {
                    blit_src.push(format);
                }
                if optimal_usage.blit_dst {
                    blit_dst.push(format);
                }
                if optimal_usage.transfer_src {
                    transfer_src.push(format);
                }
                if optimal_usage.transfer_dst {
                    transfer_dst.push(format);
                }

                records.push(VulkanFormatCapabilityRecord {
                    format: info.fourcc,
                    modifier: Modifier::Invalid,
                    usages: optimal_usage,
                });
            }

            let mut linear_usage = format_usage_from_features(format_properties.linear_tiling_features);
            if linear_tiling_supported(format_properties.linear_tiling_features) {
                linear_usage.linear = true;
                let format = format_with_modifier(info.fourcc, Modifier::Linear);

                if linear_usage.sampled {
                    sampled.push(format);
                }
                if linear_usage.render_target {
                    render_target.push(format);
                }
                if linear_usage.blit_src {
                    blit_src.push(format);
                }
                if linear_usage.blit_dst {
                    blit_dst.push(format);
                }
                if linear_usage.transfer_src {
                    transfer_src.push(format);
                }
                if linear_usage.transfer_dst {
                    transfer_dst.push(format);
                }
                linear.push(format);
                records.push(VulkanFormatCapabilityRecord {
                    format: info.fourcc,
                    modifier: Modifier::Linear,
                    usages: linear_usage,
                });
            }
        }

        Ok(Self {
            records,
            sampled: sampled.into_iter().collect(),
            memory_import: FormatSet::default(),
            dmabuf_import: FormatSet::default(),
            render_target: render_target.into_iter().collect(),
            dmabuf_export: FormatSet::default(),
            blit_src: blit_src.into_iter().collect(),
            blit_dst: blit_dst.into_iter().collect(),
            transfer_src: transfer_src.into_iter().collect(),
            transfer_dst: transfer_dst.into_iter().collect(),
            linear: linear.into_iter().collect(),
        })
    }
}

impl VulkanFormatUsage {
    fn any_supported(&self) -> bool {
        self.sampled
            || self.memory_import
            || self.dmabuf_import
            || self.render_target
            || self.dmabuf_export
            || self.blit_src
            || self.blit_dst
            || self.transfer_src
            || self.transfer_dst
            || self.linear
    }
}

fn format_with_modifier(format: Fourcc, modifier: Modifier) -> Format {
    Format {
        code: format,
        modifier,
    }
}

pub(super) fn format_usage_from_features(features: vk::FormatFeatureFlags) -> VulkanFormatUsage {
    VulkanFormatUsage {
        sampled: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
        render_target: features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT),
        blit_src: features.contains(vk::FormatFeatureFlags::BLIT_SRC),
        blit_dst: features.contains(vk::FormatFeatureFlags::BLIT_DST),
        transfer_src: features.contains(vk::FormatFeatureFlags::TRANSFER_SRC),
        transfer_dst: features.contains(vk::FormatFeatureFlags::TRANSFER_DST),
        ..VulkanFormatUsage::default()
    }
}

pub(super) fn linear_tiling_supported(features: vk::FormatFeatureFlags) -> bool {
    features.intersects(
        vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST
            | vk::FormatFeatureFlags::BLIT_SRC
            | vk::FormatFeatureFlags::BLIT_DST,
    )
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
    /// Whether Smithay `Blit` operations are supported.
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
