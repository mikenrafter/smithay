use std::ffi::CStr;

use ash::{ext, khr, vk};

use crate::backend::{
    allocator::{Format, Fourcc, Modifier, format::FormatSet},
    vulkan::{PhysicalDevice, version::Version},
};

use super::{
    VulkanError,
    format::{get_format_info, renderer_format_infos},
    image::VulkanDmabufImportState,
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
    /// External synchronization prerequisite discovery.
    pub external_sync: VulkanExternalSyncCapabilities,
}

/// Raw per-format Vulkan image feature capabilities.
///
/// Records describe renderer-internal Vulkan support. Smithay-facing [`FormatSet`] values are
/// limited to implemented renderer traits. The dmabuf import and render-target sets are probed
/// development-path sets and are separated from the fully integrated rendering capability bits
/// below.
#[non_exhaustive]
#[derive(Debug, Default, Clone)]
pub struct VulkanFormatCapabilities {
    /// Detailed records keyed by DRM format and Vulkan image tiling.
    pub records: Vec<VulkanFormatCapabilityRecord>,
    /// Detailed records keyed by DRM format and DRM format modifier.
    #[allow(dead_code)]
    pub(crate) modifier_records: Vec<VulkanDrmFormatModifierCapabilityRecord>,
    /// Formats usable for shared-memory uploads.
    pub memory_import: FormatSet,
    /// Probed sampled single-plane dmabuf import format/modifier pairs.
    ///
    /// This is a raw Vulkan external-memory format set. Public ImportDma advertisement remains
    /// gated separately.
    pub dmabuf_import: FormatSet,
    /// Formats exportable as dmabufs.
    pub dmabuf_export: FormatSet,
    /// Formats probed as usable by the explicit Vulkan dmabuf render-target development path.
    ///
    /// This is a raw Vulkan external-memory format set, not a broad compositor capability. Public
    /// [`VulkanRenderingCapabilities`] bits decide whether those formats are advertised as fully
    /// integrated renderer functionality.
    pub dmabuf_render_target: FormatSet,
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

/// Capability record for a DRM format modifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VulkanDrmFormatModifierCapabilityRecord {
    /// DRM format code.
    pub format: Fourcc,
    /// DRM format modifier.
    pub modifier: Modifier,
    /// Number of memory planes used by this format/modifier pair.
    pub plane_count: u32,
    /// Per-usage capability bits for this format/modifier pair.
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
    /// renderer's static format table. Sampled single-plane modifier pairs are recorded in
    /// `dmabuf_import`; public ImportDma advertisement remains gated. Export stays empty until
    /// that trait is implemented.
    pub fn discover(
        physical_device: &PhysicalDevice,
        external_memory: &VulkanExternalMemoryCapabilities,
    ) -> Result<Self, VulkanError> {
        let mut records = Vec::new();
        let mut modifier_records = Vec::new();
        let mut memory_import = Vec::new();
        let mut dmabuf_import = Vec::new();
        let mut dmabuf_render_target = Vec::new();

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

            if should_query_modifier_properties(external_memory) {
                for modifier_properties in physical_device
                    .get_format_modifier_properties(info.vk_format)
                    .unwrap_or_default()
                {
                    let record = modifier_record_from_properties(info.fourcc, modifier_properties);
                    let is_importable_render_target = !info.is_10bit
                        && record.plane_count == 1
                        && record.usages.color_attachment
                        && record.usages.color_attachment_blend
                        && dmabuf_render_target_external_importable(
                            physical_device,
                            info.vk_format,
                            record.modifier,
                        )?;
                    if is_importable_render_target {
                        dmabuf_render_target.push(Format {
                            code: record.format,
                            modifier: record.modifier,
                        });
                    }
                    if record.usages.sampled && record.plane_count == 1 {
                        dmabuf_import.push(Format {
                            code: record.format,
                            modifier: record.modifier,
                        });
                    }
                    if record.usages.any_supported() {
                        modifier_records.push(record);
                    }
                }
            }
        }

        Ok(Self {
            records,
            modifier_records,
            memory_import: memory_import.into_iter().collect(),
            dmabuf_import: dmabuf_import.into_iter().collect(),
            dmabuf_export: FormatSet::default(),
            dmabuf_render_target: dmabuf_render_target.into_iter().collect(),
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

    #[allow(dead_code)]
    pub(crate) fn dmabuf_import_record(
        &self,
        import: &VulkanDmabufImportState,
    ) -> Option<&VulkanDrmFormatModifierCapabilityRecord> {
        self.modifier_records.iter().find(|record| {
            record.format == import.format()
                && record.modifier == import.modifier()
                && record.plane_count as usize == import.plane_count()
                && record.usages.sampled
        })
    }

    #[allow(dead_code)]
    pub(crate) fn dmabuf_render_target_record(
        &self,
        import: &VulkanDmabufImportState,
    ) -> Option<&VulkanDrmFormatModifierCapabilityRecord> {
        self.modifier_records.iter().find(|record| {
            record.format == import.format()
                && record.modifier == import.modifier()
                && get_format_info(record.format)
                    .map(|info| !info.is_10bit)
                    .unwrap_or(false)
                && record.plane_count == 1
                && record.plane_count as usize == import.plane_count()
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
        })
    }

    #[allow(dead_code)]
    pub(crate) fn has_sampled_dmabuf_modifier_record(&self, import: &VulkanDmabufImportState) -> bool {
        self.dmabuf_import_record(import).is_some()
    }

    #[allow(dead_code)]
    pub(crate) fn has_dmabuf_render_target_modifier_record(&self, import: &VulkanDmabufImportState) -> bool {
        self.dmabuf_render_target_record(import).is_some()
    }
}

pub(super) fn should_query_modifier_properties(external_memory: &VulkanExternalMemoryCapabilities) -> bool {
    external_memory.prerequisites_available
        && external_memory.dmabuf_external_memory
        && external_memory.external_memory_fd
        && external_memory.drm_format_modifiers
        && external_memory.foreign_queue_family
        && external_memory.image_format_list
}

fn dmabuf_render_target_external_importable(
    physical_device: &PhysicalDevice,
    format: vk::Format,
    modifier: Modifier,
) -> Result<bool, VulkanError> {
    let mut external_image_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut drm_format_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(modifier.into())
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let format_info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .flags(vk::ImageCreateFlags::empty())
        .push_next(&mut external_image_info)
        .push_next(&mut drm_format_info);
    let mut external_properties = vk::ExternalImageFormatProperties::default();
    let mut image_properties = vk::ImageFormatProperties2::default().push_next(&mut external_properties);

    // SAFETY: `physical_device` belongs to its retained instance. The pNext chains live for the
    // duration of the call and request only support properties for a 2D color-attachment image with
    // DRM-format-modifier tiling and DMA_BUF external memory; callers only reach this helper after
    // the required external-memory and DRM-modifier prerequisites were discovered.
    let result = unsafe {
        physical_device
            .instance()
            .handle()
            .get_physical_device_image_format_properties2(
                physical_device.handle(),
                &format_info,
                &mut image_properties,
            )
    };

    match result {
        Ok(()) => {
            let image_format_properties = image_properties.image_format_properties;
            let _ = image_properties;
            let external_memory_properties = external_properties.external_memory_properties;
            Ok(external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
                && external_memory_properties
                    .compatible_handle_types
                    .contains(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                && image_format_properties
                    .sample_counts
                    .contains(vk::SampleCountFlags::TYPE_1))
        }
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(false),
        Err(err) => Err(VulkanError::from(err)),
    }
}

pub(super) fn modifier_record_from_properties(
    format: Fourcc,
    properties: vk::DrmFormatModifierPropertiesEXT,
) -> VulkanDrmFormatModifierCapabilityRecord {
    VulkanDrmFormatModifierCapabilityRecord {
        format,
        modifier: Modifier::from(properties.drm_format_modifier),
        plane_count: properties.drm_format_modifier_plane_count,
        usages: format_usage_from_features(properties.drm_format_modifier_tiling_features),
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
    /// Whether `VK_EXT_queue_family_foreign` is supported for foreign producer ownership transfer.
    pub foreign_queue_family: bool,
    /// Whether `VK_KHR_image_format_list` is available, either as Vulkan 1.2 core or as an extension.
    pub image_format_list: bool,
    /// Whether the known renderer dmabuf external-memory prerequisites are all available.
    pub prerequisites_available: bool,
}

impl VulkanExternalMemoryCapabilities {
    #[allow(dead_code)]
    pub(super) fn required_device_extensions(api_version: Version) -> Vec<&'static CStr> {
        let mut extensions = vec![
            ext::external_memory_dma_buf::NAME,
            khr::external_memory_fd::NAME,
            ext::image_drm_format_modifier::NAME,
            ext::queue_family_foreign::NAME,
        ];

        if api_version < Version::VERSION_1_2 {
            extensions.push(khr::image_format_list::NAME);
        }

        extensions
    }

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
        let foreign_queue_family = has_device_extension(ext::queue_family_foreign::NAME);
        let image_format_list =
            api_version >= Version::VERSION_1_2 || has_device_extension(khr::image_format_list::NAME);
        let prerequisites_available = dmabuf_external_memory
            && external_memory_fd
            && drm_format_modifiers
            && foreign_queue_family
            && image_format_list;

        Self {
            dmabuf_external_memory,
            external_memory_fd,
            drm_format_modifiers,
            foreign_queue_family,
            image_format_list,
            prerequisites_available,
        }
    }
}

/// Vulkan external synchronization prerequisite discovery.
///
/// These fields only describe physical-device support for future Vulkan sync-file semaphore
/// plumbing. They do not mean that the renderer can already import/export sync files or advertise
/// Smithay explicit synchronization support.
#[non_exhaustive]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VulkanExternalSyncCapabilities {
    /// Whether external semaphore support is available via Vulkan 1.1 core or `VK_KHR_external_semaphore`.
    pub external_semaphore: bool,
    /// Whether `VK_KHR_external_semaphore_fd` is supported.
    pub external_semaphore_fd: bool,
    /// Whether `SYNC_FD` external semaphores report import support.
    pub sync_file_importable: bool,
    /// Whether `SYNC_FD` external semaphores report export support.
    pub sync_file_exportable: bool,
    /// Whether semaphores imported from `SYNC_FD` can be exported again as `SYNC_FD`.
    pub sync_file_export_from_imported: bool,
    /// Whether the known renderer sync-file semaphore prerequisites are all available.
    pub prerequisites_available: bool,
}

impl VulkanExternalSyncCapabilities {
    #[allow(dead_code)]
    pub(super) fn required_device_extensions(api_version: Version) -> Vec<&'static CStr> {
        let mut extensions = Vec::new();
        if api_version < Version::VERSION_1_1 {
            extensions.push(khr::external_semaphore::NAME);
        }
        extensions.push(khr::external_semaphore_fd::NAME);
        extensions
    }

    pub(super) fn discover(physical_device: &PhysicalDevice) -> Self {
        let mut capabilities =
            Self::from_device_extension_support(physical_device.api_version(), |extension| {
                physical_device.has_device_extension(extension)
            });

        if capabilities.prerequisites_available {
            let external_semaphore_info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            let mut external_semaphore_properties = vk::ExternalSemaphoreProperties::default();
            // SAFETY: `physical_device` was enumerated from this live instance. Vulkan 1.1 core or
            // VK_KHR_external_semaphore provides this query, and `prerequisites_available` also
            // requires VK_KHR_external_semaphore_fd. Input/output pointers refer to stack storage
            // valid for the duration of the call.
            unsafe {
                physical_device
                    .instance()
                    .handle()
                    .get_physical_device_external_semaphore_properties(
                        physical_device.handle(),
                        &external_semaphore_info,
                        &mut external_semaphore_properties,
                    )
            };
            capabilities.apply_sync_file_properties(external_semaphore_properties);
        }

        capabilities
    }

    pub(super) fn from_device_extension_support(
        api_version: Version,
        mut has_device_extension: impl FnMut(&CStr) -> bool,
    ) -> Self {
        let external_semaphore =
            api_version >= Version::VERSION_1_1 || has_device_extension(khr::external_semaphore::NAME);
        let external_semaphore_fd = has_device_extension(khr::external_semaphore_fd::NAME);
        let prerequisites_available = external_semaphore && external_semaphore_fd;

        Self {
            external_semaphore,
            external_semaphore_fd,
            sync_file_importable: false,
            sync_file_exportable: false,
            sync_file_export_from_imported: false,
            prerequisites_available,
        }
    }

    pub(super) fn apply_sync_file_properties(&mut self, properties: vk::ExternalSemaphoreProperties<'_>) {
        let sync_fd = vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD;
        let features = properties.external_semaphore_features;
        self.sync_file_importable = features.contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE);
        self.sync_file_exportable = features.contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE);
        self.sync_file_export_from_imported = self.sync_file_importable
            && self.sync_file_exportable
            && properties.export_from_imported_handle_types.contains(sync_fd);
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
    /// Whether sampled dmabuf import formats were probed.
    ///
    /// This is raw Vulkan capability, not public ImportDma advertisement.
    pub dmabuf: bool,
    /// Whether modifier-aware sampled dmabuf import formats were probed.
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
    /// Whether fully integrated imported dmabuf renderer framebuffers are supported.
    ///
    /// This remains false while Vulkan dmabuf render targets require explicit development-fork
    /// ownership/layout/synchronization contracts.
    pub dmabuf_targets: bool,
    /// Whether modifier-aware fully integrated imported dmabuf render targets are supported.
    ///
    /// This remains false while Vulkan dmabuf render targets require explicit development-fork
    /// ownership/layout/synchronization contracts.
    pub dmabuf_target_modifiers: bool,
    /// Whether the Vulkan dmabuf render-target development path has probed formats.
    ///
    /// This indicates the explicit unsafe development API has probed formats and gates this fork's
    /// conservative generic [`Bind<Dmabuf>`](crate::backend::renderer::Bind) path. The generic path
    /// discards previous contents and forces full repaint; callers using the explicit path must satisfy
    /// [`crate::backend::renderer::vulkan::VulkanRenderer::bind_dmabuf_render_target`] safety
    /// requirements. This flag does not imply broad imported-dmabuf renderer-framebuffer support.
    pub dmabuf_target_development: bool,
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
