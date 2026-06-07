use std::ffi::CStr;

use ash::{ext, khr, vk};

use crate::backend::{
    allocator::{Format, Fourcc, Modifier, format::FormatSet},
    vulkan::{PhysicalDevice, version::Version},
};

use super::{
    VulkanError,
    format::{get_format_info, renderer_format_infos},
};

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
    /// External-memory prerequisite discovery.
    pub external_memory: VulkanExternalMemoryCapabilities,
}

/// Raw per-format Vulkan image feature capabilities.
///
/// Records describe renderer-internal Vulkan support. Smithay-facing [`FormatSet`] values remain
/// limited to import/export traits and stay empty until those traits are implemented.
#[non_exhaustive]
#[derive(Debug, Default, Clone)]
pub struct VulkanFormatCapabilities {
    /// Detailed records keyed by DRM format and Vulkan image tiling.
    pub records: Vec<VulkanFormatCapabilityRecord>,
    /// Formats usable for shared-memory uploads.
    pub memory_import: FormatSet,
    /// Formats usable for dmabuf imports.
    pub dmabuf_import: FormatSet,
    /// Formats exportable as dmabufs.
    pub dmabuf_export: FormatSet,
}

/// Capability record for a DRM format and Vulkan image tiling.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanFormatCapabilityRecord {
    /// DRM format code.
    pub format: Fourcc,
    /// Vulkan image tiling queried for this format.
    pub tiling: VulkanFormatTiling,
    /// Per-usage capability bits for this format and tiling marker.
    pub usages: VulkanFormatUsage,
}

/// Vulkan image tiling queried for a renderer-internal format record.
#[non_exhaustive]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanFormatTiling {
    /// Vulkan optimal tiling.
    Optimal,
    /// Vulkan linear tiling.
    Linear,
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
    /// Usable as a framebuffer color attachment.
    pub color_attachment: bool,
    /// Usable as a framebuffer color attachment with blending.
    pub color_attachment_blend: bool,
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
}

impl VulkanFormatCapabilities {
    /// Discovers Vulkan renderer format capabilities for a physical device.
    ///
    /// This probes sampled, color-attachment, blit, transfer, and linear tiling support for the
    /// renderer's static format table. Import and export format sets remain empty until those
    /// traits are implemented and can import or export the advertised pairs.
    pub fn discover(physical_device: &PhysicalDevice) -> Result<Self, VulkanError> {
        let mut records = Vec::new();
        let mut memory_import = Vec::new();

        for info in renderer_format_infos() {
            let mut properties = vk::FormatProperties2::default();

            unsafe { physical_device.get_format_properties(info.vk_format, &mut properties) };

            let format_properties = properties.format_properties;
            let mut optimal_usage = format_usage_from_features(format_properties.optimal_tiling_features);
            optimal_usage.memory_import = optimal_usage.sampled && optimal_usage.transfer_dst;
            if optimal_usage.memory_import {
                memory_import.push(Format {
                    code: info.fourcc,
                    modifier: Modifier::Invalid,
                });
            }
            if optimal_usage.any_supported() {
                records.push(VulkanFormatCapabilityRecord {
                    format: info.fourcc,
                    tiling: VulkanFormatTiling::Optimal,
                    usages: optimal_usage,
                });
            }

            let linear_usage = format_usage_from_features(format_properties.linear_tiling_features);
            if linear_tiling_supported(format_properties.linear_tiling_features) {
                records.push(VulkanFormatCapabilityRecord {
                    format: info.fourcc,
                    tiling: VulkanFormatTiling::Linear,
                    usages: linear_usage,
                });
            }
        }

        Ok(Self {
            records,
            memory_import: memory_import.into_iter().collect(),
            dmabuf_import: FormatSet::default(),
            dmabuf_export: FormatSet::default(),
        })
    }

    pub(super) fn render_target_formats(&self) -> FormatSet {
        self.records
            .iter()
            .filter(|record| {
                record.tiling == VulkanFormatTiling::Optimal
                    && get_format_info(record.format)
                        .map(|info| !info.is_10bit)
                        .unwrap_or(false)
                    && record.usages.color_attachment
                    && record.usages.color_attachment_blend
                    && record.usages.transfer_src
                    && record.usages.transfer_dst
            })
            .map(|record| Format {
                code: record.format,
                modifier: Modifier::Invalid,
            })
            .collect()
    }
}

impl VulkanFormatUsage {
    fn any_supported(&self) -> bool {
        self.sampled
            || self.memory_import
            || self.dmabuf_import
            || self.color_attachment
            || self.color_attachment_blend
            || self.dmabuf_export
            || self.blit_src
            || self.blit_dst
            || self.transfer_src
            || self.transfer_dst
    }
}

pub(super) fn format_usage_from_features(features: vk::FormatFeatureFlags) -> VulkanFormatUsage {
    VulkanFormatUsage {
        sampled: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
        color_attachment: features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT),
        color_attachment_blend: features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
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
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
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

/// Vulkan external-memory prerequisite discovery.
///
/// These fields only describe physical-device support for renderer-side dmabuf plumbing. They do
/// not mean that the renderer enabled the device extensions, can import/export dmabufs, or should
/// advertise any Smithay-facing dmabuf format set.
#[non_exhaustive]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VulkanExternalMemoryCapabilities {
    /// Whether `VK_EXT_external_memory_dma_buf` is supported.
    pub dmabuf_external_memory: bool,
    /// Whether `VK_KHR_external_memory_fd` is supported.
    pub external_memory_fd: bool,
    /// Whether `VK_EXT_image_drm_format_modifier` is supported.
    pub drm_format_modifiers: bool,
    /// Whether `VK_KHR_image_format_list` is available, either as Vulkan 1.2 core or as an extension.
    pub image_format_list: bool,
    /// Whether the known renderer dmabuf external-memory prerequisites are all available.
    pub prerequisites_available: bool,
}

impl VulkanExternalMemoryCapabilities {
    pub(super) fn discover(physical_device: &PhysicalDevice) -> Self {
        Self::from_device_extension_support(physical_device.api_version(), |extension| {
            physical_device.has_device_extension(extension)
        })
    }

    pub(super) fn from_device_extension_support(
        api_version: Version,
        mut has_device_extension: impl FnMut(&CStr) -> bool,
    ) -> Self {
        let dmabuf_external_memory = has_device_extension(ext::external_memory_dma_buf::NAME);
        let external_memory_fd = has_device_extension(khr::external_memory_fd::NAME);
        let drm_format_modifiers = has_device_extension(ext::image_drm_format_modifier::NAME);
        let image_format_list =
            api_version >= Version::VERSION_1_2 || has_device_extension(khr::image_format_list::NAME);
        let prerequisites_available =
            dmabuf_external_memory && external_memory_fd && drm_format_modifiers && image_format_list;

        Self {
            dmabuf_external_memory,
            external_memory_fd,
            drm_format_modifiers,
            image_format_list,
            prerequisites_available,
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
    /// Whether same-format CPU-memory readback from offscreen framebuffers is supported.
    pub memory: bool,
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
