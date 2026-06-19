use std::{
    cell::Cell,
    ffi::CStr,
    fs::File,
    marker::PhantomData,
    os::unix::io::{AsFd, OwnedFd},
    sync::Arc,
};

use ash::{ext, khr, vk};

use crate::backend::allocator::{
    Buffer, Format, Fourcc, Modifier,
    dmabuf::{AsDmabuf, Dmabuf, DmabufFlags},
    vulkan::{
        ImageUsageFlags, VulkanAllocator, VulkanAllocatorDmabufForeignReleaseEvidence,
        VulkanAllocatorForeignReleaseError, VulkanImage,
    },
};
use crate::backend::renderer::sync::Interrupted;
use crate::backend::renderer::{
    Bind, Color32F, DebugFlags, ExportMem, Frame, ImportDma, ImportMem, Offscreen, RenderTargetLifecycle,
    Renderer, SurfaceCacheTextureReleaseError, Texture, TextureMapping,
    sync::{Fence, SyncPoint},
};
use crate::backend::vulkan::{Instance, PhysicalDevice, version::Version};
use crate::utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform, user_data::UserDataMap};
#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
use crate::wayland::drm_syncobj::DrmSyncPoint;

use super::capabilities::{
    format_usage_from_features, linear_tiling_supported, modifier_record_from_properties,
    should_query_modifier_properties,
};
use super::device::{
    VulkanDeviceState, VulkanDmabufExternalImageFormatProperties, VulkanSampledDmabufForeignReleaseError,
    VulkanSampledTexturePipelineShaders, VulkanShaderSpirv, VulkanSharedImageSyncState,
    VulkanSubmitSynchronization, VulkanSyncFileImport, VulkanSyncFileSemaphorePayloadState,
    classify_sampled_dmabuf_release_submit_error_for_tests, dmabuf_import_memory_type_bits,
    dmabuf_plane_layouts, dmabuf_render_target_foreign_acquire_barrier,
    dmabuf_render_target_foreign_release_barrier, find_memory_type_index, image_copy_buffer_offset,
    image_copy_required_size, image_layout_transition, plan_dmabuf_render_target_foreign_acquire_barrier,
    plan_dmabuf_render_target_foreign_release_barrier, plan_sampled_dmabuf_foreign_acquire_barrier,
    plan_sampled_dmabuf_foreign_release_barrier, project_dmabuf_render_target_sync_after_pending_acquire,
    sampled_dmabuf_foreign_acquire_barrier, sampled_dmabuf_foreign_release_barrier, select_queue_families,
    tightly_packed_image_size, validate_submit_wait_stage, vulkan_filter,
};
use super::error::vulkan_api_result_invalidates_context;
use super::format::is_10bit;
use super::image::{
    VulkanDmabufImportState, VulkanDmabufPlane, VulkanExternalImageAcquireKind, VulkanExternalImageOwnership,
    VulkanExternalImageReleaseKind, VulkanExternalMemoryHandleType, VulkanExternalMemoryState,
    VulkanImageLayoutState, VulkanImageSource, VulkanImageState, VulkanImageSyncState, VulkanImageUsage,
    VulkanSampledDmabufRelease, clear_damage_to_clear_areas, clip_render_texture_draw_area,
    damage_to_scissor_areas, dmabuf_acquired_image_state, dmabuf_acquired_render_target_image_state,
    dmabuf_import_image_state, dmabuf_render_target_image_state, draw_solid_damage_to_clear_areas,
    render_texture_damage_to_scissor_areas, source_to_uv_rect,
};
use super::*;

#[derive(Debug)]
struct InterruptedFence;

impl Fence for InterruptedFence {
    fn is_signaled(&self) -> bool {
        false
    }

    fn wait(&self) -> Result<(), Interrupted> {
        Err(Interrupted)
    }

    fn is_exportable(&self) -> bool {
        false
    }

    fn export(&self) -> Option<OwnedFd> {
        None
    }
}

#[derive(Debug)]
struct SignaledExportableFence;

impl Fence for SignaledExportableFence {
    fn is_signaled(&self) -> bool {
        true
    }

    fn wait(&self) -> Result<(), Interrupted> {
        Ok(())
    }

    fn is_exportable(&self) -> bool {
        true
    }

    fn export(&self) -> Option<OwnedFd> {
        panic!("already-signaled sync points must not be exported")
    }
}

#[derive(Debug)]
struct CpuWaitFence {
    exportable: bool,
    exports_fd: bool,
}

impl Fence for CpuWaitFence {
    fn is_signaled(&self) -> bool {
        false
    }

    fn wait(&self) -> Result<(), Interrupted> {
        Ok(())
    }

    fn is_exportable(&self) -> bool {
        self.exportable
    }

    fn export(&self) -> Option<OwnedFd> {
        if self.exports_fd {
            Some(File::open("/dev/null").unwrap().into())
        } else {
            None
        }
    }
}

fn frame_for_tests(
    context_id: ContextId<VulkanTexture>,
    output_size: Size<i32, Physical>,
    transform: Transform,
) -> VulkanFrame<'static, 'static> {
    VulkanFrame {
        context_id,
        output_size,
        transform,
        device: None,
        target: None,
        _renderer: PhantomData,
    }
}

fn texture_for_tests(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> VulkanTexture {
    VulkanTexture {
        context_id: ContextId::new(),
        image: VulkanImageState::new_for_tests(size, format),
        sampled_image: None,
        sampled_dmabuf_release: None,
        sampled_dmabuf: None,
        y_inverted: false,
    }
}

fn render_target_for_tests(
    context_id: ContextId<VulkanTexture>,
    source: VulkanImageSource,
    size: Size<i32, BufferCoord>,
    format: Option<Fourcc>,
) -> VulkanRenderTarget<'static> {
    let mut image = VulkanImageState::new_for_tests(size, format);
    image.source = source;

    VulkanRenderTarget {
        context_id,
        image,
        color_image: None,
        dmabuf: None,
        _target: PhantomData,
    }
}

fn dmabuf_for_tests() -> Dmabuf {
    dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Invalid,
        DmabufFlags::empty(),
        &[(0, 0, 4)],
    )
}

fn dmabuf_with_planes_for_tests(
    size: Size<i32, BufferCoord>,
    format: Fourcc,
    modifier: Modifier,
    flags: DmabufFlags,
    planes: &[(u32, u32, u32)],
) -> Dmabuf {
    let mut builder = Dmabuf::builder(size, format, modifier, flags);
    for &(idx, offset, stride) in planes {
        builder.add_plane(File::open("/dev/null").unwrap().into(), idx, offset, stride);
    }
    builder.build().unwrap()
}

fn extension_names_for_tests(extensions: Vec<&'static CStr>) -> Vec<String> {
    extensions
        .into_iter()
        .map(|extension| extension.to_string_lossy().into_owned())
        .collect()
}

fn dmabuf_external_image_properties_for_tests(
    importable: bool,
    max_extent: vk::Extent3D,
    sample_counts: vk::SampleCountFlags,
) -> VulkanDmabufExternalImageFormatProperties {
    let external_memory_features = if importable {
        vk::ExternalMemoryFeatureFlags::IMPORTABLE
    } else {
        vk::ExternalMemoryFeatureFlags::empty()
    };
    let compatible_handle_types = if importable {
        vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT
    } else {
        vk::ExternalMemoryHandleTypeFlags::empty()
    };

    VulkanDmabufExternalImageFormatProperties {
        image_format_properties: vk::ImageFormatProperties {
            max_extent,
            max_mip_levels: 1,
            max_array_layers: 1,
            sample_counts,
            max_resource_size: 1,
        },
        external_memory_properties: vk::ExternalMemoryProperties {
            external_memory_features,
            export_from_imported_handle_types: vk::ExternalMemoryHandleTypeFlags::empty(),
            compatible_handle_types,
        },
        importable,
        dedicated_only: false,
    }
}

fn render_target_format_record_for_tests(
    format: Fourcc,
    tiling: VulkanFormatTiling,
    color_attachment_blend: bool,
) -> VulkanFormatCapabilityRecord {
    VulkanFormatCapabilityRecord {
        format,
        tiling,
        usages: VulkanFormatUsage {
            sampled: true,
            color_attachment: true,
            color_attachment_blend,
            transfer_src: true,
            transfer_dst: true,
            ..VulkanFormatUsage::default()
        },
    }
}

fn memory_properties_for_tests(
    memory_types: &[vk::MemoryPropertyFlags],
) -> vk::PhysicalDeviceMemoryProperties {
    let mut properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: memory_types.len() as u32,
        ..Default::default()
    };

    for (index, flags) in memory_types.iter().copied().enumerate() {
        properties.memory_types[index] = vk::MemoryType {
            property_flags: flags,
            heap_index: 0,
        };
    }

    properties
}

fn has_probed_format_support(caps: &VulkanFormatCapabilities) -> bool {
    caps.records.iter().any(|record| {
        record.usages.sampled
            || record.usages.color_attachment
            || record.usages.color_attachment_blend
            || record.usages.blit_src
            || record.usages.blit_dst
            || record.usages.transfer_src
            || record.usages.transfer_dst
    })
}

#[test]
fn renderer_context_id_is_stable_and_unique_per_instance() {
    let renderer = VulkanRenderer::new_scaffold_for_tests();
    let other_renderer = VulkanRenderer::new_scaffold_for_tests();

    assert_eq!(renderer.context_id(), renderer.context_id());
    assert_ne!(renderer.context_id(), other_renderer.context_id());
}

#[test]
fn vulkan_renderer_default_capabilities_are_false() {
    let caps = VulkanRenderer::scaffold_capabilities();
    assert!(!caps.device.available);
    assert!(!caps.device.multi_gpu);
    assert!(caps.device.extensions.is_empty());
    assert!(!caps.import.memory);
    assert!(!caps.import.dmabuf);
    assert!(!caps.import.modifiers);
    assert!(!caps.export.memory);
    assert!(!caps.export.dmabuf);
    assert!(!caps.export.modifiers);
    assert!(!caps.rendering.offscreen);
    assert!(!caps.rendering.dmabuf_targets);
    assert!(!caps.rendering.dmabuf_target_modifiers);
    assert!(!caps.rendering.dmabuf_target_development);
    assert!(!caps.rendering.blit);
    assert!(!caps.rendering.render_target_10bit);
    assert!(!caps.rendering.render_target_fp16);
    assert!(!caps.sync.explicit);
    assert!(!caps.color.color_transform_hooks);
    assert!(!caps.color.hdr_ready_targets);
    assert!(!caps.external_memory.dmabuf_external_memory);
    assert!(!caps.external_memory.external_memory_fd);
    assert!(!caps.external_memory.drm_format_modifiers);
    assert!(!caps.external_memory.foreign_queue_family);
    assert!(!caps.external_memory.image_format_list);
    assert!(!caps.external_memory.prerequisites_available);
    assert!(!caps.external_sync.external_semaphore);
    assert!(!caps.external_sync.external_semaphore_fd);
    assert!(!caps.external_sync.sync_file_importable);
    assert!(!caps.external_sync.sync_file_exportable);
    assert!(!caps.external_sync.sync_file_export_from_imported);
    assert!(!caps.external_sync.prerequisites_available);
}

#[test]
fn vulkan_format_capability_matrix_defaults_empty() {
    let caps = VulkanRendererCapabilities::default();
    assert!(caps.formats.records.is_empty());
    assert!(caps.formats.modifier_records.is_empty());
    assert!(caps.formats.memory_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_export.iter().next().is_none());
    assert!(caps.formats.dmabuf_render_target.iter().next().is_none());
}

#[test]
fn wayland_protocol_capabilities_are_not_advertised_by_default() {
    let caps = VulkanRendererCapabilities::default();

    assert!(!caps.import.memory);
    assert!(!caps.import.dmabuf);
    assert!(!caps.import.modifiers);
    assert!(!caps.export.memory);
    assert!(!caps.export.dmabuf);
    assert!(!caps.export.modifiers);
    assert!(!caps.rendering.dmabuf_targets);
    assert!(!caps.rendering.dmabuf_target_modifiers);
    assert!(!caps.rendering.dmabuf_target_development);
    assert!(!caps.sync.explicit);
    assert!(!caps.external_sync.prerequisites_available);
    assert!(caps.formats.memory_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_export.iter().next().is_none());
    assert!(caps.formats.dmabuf_render_target.iter().next().is_none());
}

#[cfg(feature = "wayland_frontend")]
#[test]
fn wayland_import_all_uses_shared_buffer_integration() {
    fn assert_import_all<R: crate::backend::renderer::ImportAll>() {
        let _ = R::import_buffer_from_surface_state;
    }

    assert_import_all::<VulkanRenderer>();
}

#[test]
fn vulkan_format_capability_record_is_per_tiling_marker() {
    let record = VulkanFormatCapabilityRecord {
        format: Fourcc::Argb8888,
        tiling: VulkanFormatTiling::Linear,
        usages: VulkanFormatUsage {
            sampled: true,
            memory_import: true,
            transfer_src: true,
            ..VulkanFormatUsage::default()
        },
    };

    assert_eq!(record.format, Fourcc::Argb8888);
    assert_eq!(record.tiling, VulkanFormatTiling::Linear);
    assert!(record.usages.sampled);
    assert!(record.usages.memory_import);
    assert!(!record.usages.dmabuf_import);
    assert!(record.usages.transfer_src);
}

#[test]
fn external_memory_capability_discovery_tracks_prerequisites_without_advertising_dmabuf() {
    let supported = [
        ext::external_memory_dma_buf::NAME,
        khr::external_memory_fd::NAME,
        ext::image_drm_format_modifier::NAME,
        ext::queue_family_foreign::NAME,
        khr::image_format_list::NAME,
    ];
    let caps =
        VulkanExternalMemoryCapabilities::from_device_extension_support(Version::VERSION_1_1, |name| {
            supported.iter().any(|supported| *supported == name)
        });

    assert!(caps.dmabuf_external_memory);
    assert!(caps.external_memory_fd);
    assert!(caps.drm_format_modifiers);
    assert!(caps.foreign_queue_family);
    assert!(caps.image_format_list);
    assert!(caps.prerequisites_available);

    let renderer_caps = VulkanRendererCapabilities {
        external_memory: caps,
        ..VulkanRendererCapabilities::default()
    };
    assert!(!renderer_caps.import.dmabuf);
    assert!(!renderer_caps.import.modifiers);
    assert!(!renderer_caps.export.dmabuf);
    assert!(!renderer_caps.export.modifiers);
    assert!(!renderer_caps.rendering.dmabuf_targets);
    assert!(!renderer_caps.rendering.dmabuf_target_modifiers);
    assert!(!renderer_caps.rendering.dmabuf_target_development);
    assert!(renderer_caps.formats.dmabuf_import.iter().next().is_none());
    assert!(renderer_caps.formats.dmabuf_export.iter().next().is_none());
    assert!(renderer_caps.formats.dmabuf_render_target.iter().next().is_none());
}

#[test]
fn external_memory_capability_discovery_requires_modifier_dependency() {
    let supported = [
        ext::external_memory_dma_buf::NAME,
        khr::external_memory_fd::NAME,
        ext::image_drm_format_modifier::NAME,
        ext::queue_family_foreign::NAME,
    ];
    let caps =
        VulkanExternalMemoryCapabilities::from_device_extension_support(Version::VERSION_1_1, |name| {
            supported.iter().any(|supported| *supported == name)
        });
    assert!(!caps.image_format_list);
    assert!(!caps.prerequisites_available);

    let core_image_format_list_caps =
        VulkanExternalMemoryCapabilities::from_device_extension_support(Version::VERSION_1_2, |name| {
            supported.iter().any(|supported| *supported == name)
        });
    assert!(core_image_format_list_caps.image_format_list);
    assert!(core_image_format_list_caps.prerequisites_available);
}

#[test]
fn external_memory_capability_discovery_requires_foreign_queue_family() {
    let supported = [
        ext::external_memory_dma_buf::NAME,
        khr::external_memory_fd::NAME,
        ext::image_drm_format_modifier::NAME,
        khr::image_format_list::NAME,
    ];
    let caps =
        VulkanExternalMemoryCapabilities::from_device_extension_support(Version::VERSION_1_1, |name| {
            supported.iter().any(|supported| *supported == name)
        });

    assert!(!caps.foreign_queue_family);
    assert!(!caps.prerequisites_available);
}

#[test]
fn external_sync_capability_discovery_tracks_sync_file_prerequisites_without_advertising() {
    let supported = [khr::external_semaphore::NAME, khr::external_semaphore_fd::NAME];
    let caps = VulkanExternalSyncCapabilities::from_device_extension_support(Version::VERSION_1_0, |name| {
        supported.iter().any(|supported| *supported == name)
    });

    assert!(caps.external_semaphore);
    assert!(caps.external_semaphore_fd);
    assert!(caps.prerequisites_available);
    assert!(!caps.sync_file_importable);
    assert!(!caps.sync_file_exportable);
    assert!(!caps.sync_file_export_from_imported);

    let renderer_caps = VulkanRendererCapabilities {
        external_sync: caps,
        ..VulkanRendererCapabilities::default()
    };
    assert!(!renderer_caps.sync.explicit);
    assert!(!renderer_caps.import.dmabuf);
    assert!(!renderer_caps.export.dmabuf);
    assert!(!renderer_caps.rendering.dmabuf_targets);
    assert!(!renderer_caps.rendering.dmabuf_target_development);
}

#[test]
fn external_sync_capability_discovery_requires_fd_extension() {
    let caps = VulkanExternalSyncCapabilities::from_device_extension_support(Version::VERSION_1_0, |name| {
        name == khr::external_semaphore::NAME
    });

    assert!(caps.external_semaphore);
    assert!(!caps.external_semaphore_fd);
    assert!(!caps.prerequisites_available);
}

#[test]
fn external_sync_capability_discovery_uses_core_external_semaphore() {
    let caps = VulkanExternalSyncCapabilities::from_device_extension_support(Version::VERSION_1_1, |name| {
        name == khr::external_semaphore_fd::NAME
    });

    assert!(caps.external_semaphore);
    assert!(caps.external_semaphore_fd);
    assert!(caps.prerequisites_available);
    assert!(!caps.sync_file_importable);
    assert!(!caps.sync_file_exportable);
    assert!(!caps.sync_file_export_from_imported);
}

#[test]
fn external_sync_file_properties_map_import_export_features() {
    let mut caps =
        VulkanExternalSyncCapabilities::from_device_extension_support(Version::VERSION_1_1, |name| {
            name == khr::external_semaphore_fd::NAME
        });
    let unsupported_properties = vk::ExternalSemaphoreProperties::default()
        .export_from_imported_handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);

    caps.apply_sync_file_properties(unsupported_properties);
    assert!(!caps.sync_file_importable);
    assert!(!caps.sync_file_exportable);
    assert!(!caps.sync_file_export_from_imported);

    let sync_file_properties = vk::ExternalSemaphoreProperties::default()
        .compatible_handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .external_semaphore_features(
            vk::ExternalSemaphoreFeatureFlags::IMPORTABLE | vk::ExternalSemaphoreFeatureFlags::EXPORTABLE,
        )
        .export_from_imported_handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);

    caps.apply_sync_file_properties(sync_file_properties);
    assert!(caps.sync_file_importable);
    assert!(caps.sync_file_exportable);
    assert!(caps.sync_file_export_from_imported);

    let import_only_properties = vk::ExternalSemaphoreProperties::default()
        .compatible_handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .external_semaphore_features(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE);

    caps.apply_sync_file_properties(import_only_properties);
    assert!(caps.sync_file_importable);
    assert!(!caps.sync_file_exportable);
    assert!(!caps.sync_file_export_from_imported);

    let export_only_properties = vk::ExternalSemaphoreProperties::default()
        .external_semaphore_features(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
        .export_from_imported_handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);

    caps.apply_sync_file_properties(export_only_properties);
    assert!(!caps.sync_file_importable);
    assert!(caps.sync_file_exportable);
    assert!(!caps.sync_file_export_from_imported);
}

#[test]
fn sync_file_semaphore_helpers_require_capabilities_before_device_lookup() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(!device.can_export_sync_file());
    assert!(matches!(
        // SAFETY: The helper returns before using the import payload because sync-file import
        // capabilities are disabled on an empty test device.
        unsafe { device.import_sync_file_semaphore(VulkanSyncFileImport::AlreadySignaled) },
        Err(VulkanError::UnsupportedOperation("sync-file semaphore import"))
    ));
    assert!(matches!(
        device.create_exportable_sync_file_semaphore(),
        Err(VulkanError::UnsupportedOperation("sync-file semaphore export"))
    ));
}

#[test]
fn submit_wait_stage_validation_rejects_empty_masks() {
    assert!(matches!(
        validate_submit_wait_stage(vk::PipelineStageFlags::empty()),
        Err(VulkanError::UnsupportedOperation("semaphore wait stage"))
    ));
    assert!(validate_submit_wait_stage(vk::PipelineStageFlags::TOP_OF_PIPE).is_ok());
    assert!(validate_submit_wait_stage(vk::PipelineStageFlags::FRAGMENT_SHADER).is_ok());
}

#[test]
fn external_sync_required_device_extensions_track_api_version_dependencies() {
    assert_eq!(
        VulkanExternalSyncCapabilities::required_device_extensions(Version::VERSION_1_0),
        vec![khr::external_semaphore::NAME, khr::external_semaphore_fd::NAME]
    );
    assert_eq!(
        VulkanExternalSyncCapabilities::required_device_extensions(Version::VERSION_1_1),
        vec![khr::external_semaphore_fd::NAME]
    );
}

#[test]
fn external_memory_required_device_extensions_track_api_version_dependencies() {
    assert_eq!(
        VulkanExternalMemoryCapabilities::required_device_extensions(Version::VERSION_1_1),
        vec![
            ext::external_memory_dma_buf::NAME,
            khr::external_memory_fd::NAME,
            ext::image_drm_format_modifier::NAME,
            ext::queue_family_foreign::NAME,
            khr::image_format_list::NAME,
        ]
    );
    assert_eq!(
        VulkanExternalMemoryCapabilities::required_device_extensions(Version::VERSION_1_2),
        vec![
            ext::external_memory_dma_buf::NAME,
            khr::external_memory_fd::NAME,
            ext::image_drm_format_modifier::NAME,
            ext::queue_family_foreign::NAME,
        ]
    );
}

#[test]
fn drm_modifier_queries_require_full_external_memory_prerequisites() {
    let partial = VulkanExternalMemoryCapabilities {
        drm_format_modifiers: true,
        image_format_list: true,
        ..VulkanExternalMemoryCapabilities::default()
    };
    assert!(!should_query_modifier_properties(&partial));

    let inconsistent = VulkanExternalMemoryCapabilities {
        prerequisites_available: true,
        ..partial
    };
    assert!(!should_query_modifier_properties(&inconsistent));

    let complete = VulkanExternalMemoryCapabilities {
        dmabuf_external_memory: true,
        external_memory_fd: true,
        drm_format_modifiers: true,
        foreign_queue_family: true,
        image_format_list: true,
        prerequisites_available: true,
    };
    assert!(should_query_modifier_properties(&complete));
}

#[test]
fn drm_modifier_capability_records_map_vulkan_properties_without_advertising_dmabuf() {
    let record = modifier_record_from_properties(
        Fourcc::Abgr8888,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | vk::FormatFeatureFlags::TRANSFER_DST,
        },
    );

    assert_eq!(record.format, Fourcc::Abgr8888);
    assert_eq!(record.modifier, Modifier::Linear);
    assert_eq!(record.plane_count, 1);
    assert!(record.usages.sampled);
    assert!(record.usages.color_attachment);
    assert!(record.usages.transfer_dst);
    assert!(!record.usages.dmabuf_import);
    assert!(!record.usages.dmabuf_export);

    let renderer_caps = VulkanRendererCapabilities {
        formats: VulkanFormatCapabilities {
            modifier_records: vec![record],
            ..VulkanFormatCapabilities::default()
        },
        ..VulkanRendererCapabilities::default()
    };
    assert!(!renderer_caps.import.dmabuf);
    assert!(!renderer_caps.export.dmabuf);
    assert!(!renderer_caps.rendering.dmabuf_targets);
    assert!(!renderer_caps.rendering.dmabuf_target_modifiers);
    assert!(!renderer_caps.rendering.dmabuf_target_development);
    assert!(renderer_caps.formats.dmabuf_import.iter().next().is_none());
    assert!(renderer_caps.formats.dmabuf_export.iter().next().is_none());
    assert!(renderer_caps.formats.dmabuf_render_target.iter().next().is_none());
}

#[test]
fn drm_modifier_capability_lookup_requires_sampled_exact_match() {
    let sampled_record = modifier_record_from_properties(
        Fourcc::Nv12,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 2,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        },
    );
    let non_sampled_record = modifier_record_from_properties(
        Fourcc::Abgr8888,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::COLOR_ATTACHMENT,
        },
    );
    let caps = VulkanFormatCapabilities {
        modifier_records: vec![sampled_record, non_sampled_record],
        ..VulkanFormatCapabilities::default()
    };

    let importable_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4), (1, 4, 4)],
    );
    let importable = VulkanDmabufImportState::from_dmabuf(&importable_dmabuf).unwrap();
    assert!(caps.has_sampled_dmabuf_modifier_record(&importable));
    assert!(caps.dmabuf_import_record(&importable).is_some());

    let wrong_modifier_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Nv12,
        Modifier::Invalid,
        DmabufFlags::empty(),
        &[(0, 0, 4), (1, 4, 4)],
    );
    let wrong_modifier = VulkanDmabufImportState::from_dmabuf(&wrong_modifier_dmabuf).unwrap();
    assert!(!caps.has_sampled_dmabuf_modifier_record(&wrong_modifier));

    let wrong_plane_count_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4)],
    );
    let wrong_plane_count = VulkanDmabufImportState::from_dmabuf(&wrong_plane_count_dmabuf).unwrap();
    assert!(!caps.has_sampled_dmabuf_modifier_record(&wrong_plane_count));

    let unsupported_format_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Xrgb8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4)],
    );
    let unsupported_format = VulkanDmabufImportState::from_dmabuf(&unsupported_format_dmabuf).unwrap();
    assert!(!caps.has_sampled_dmabuf_modifier_record(&unsupported_format));

    let non_sampled_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4)],
    );
    let non_sampled = VulkanDmabufImportState::from_dmabuf(&non_sampled_dmabuf).unwrap();
    assert!(!caps.has_sampled_dmabuf_modifier_record(&non_sampled));
    assert!(caps.dmabuf_import.iter().next().is_none());
    assert!(caps.dmabuf_export.iter().next().is_none());
    assert!(caps.dmabuf_render_target.iter().next().is_none());
}

#[test]
fn drm_modifier_capability_lookup_requires_single_plane_color_attachment_target() {
    let render_target_record = modifier_record_from_properties(
        Fourcc::Abgr8888,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
        },
    );
    let ten_bit_record = modifier_record_from_properties(
        Fourcc::Xrgb2101010,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
        },
    );
    let color_only_record = modifier_record_from_properties(
        Fourcc::Argb8888,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::COLOR_ATTACHMENT,
        },
    );
    let sampled_record = modifier_record_from_properties(
        Fourcc::Xrgb8888,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        },
    );
    let multiplane_color_record = modifier_record_from_properties(
        Fourcc::Nv12,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 2,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::COLOR_ATTACHMENT,
        },
    );
    let caps = VulkanFormatCapabilities {
        modifier_records: vec![
            render_target_record,
            ten_bit_record,
            color_only_record,
            sampled_record,
            multiplane_color_record,
        ],
        ..VulkanFormatCapabilities::default()
    };

    let render_target_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let render_target_import = VulkanDmabufImportState::from_dmabuf(&render_target_dmabuf).unwrap();
    assert!(caps.has_dmabuf_render_target_modifier_record(&render_target_import));
    assert!(caps.dmabuf_render_target_record(&render_target_import).is_some());
    assert!(!caps.has_sampled_dmabuf_modifier_record(&render_target_import));

    let ten_bit_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Xrgb2101010,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let ten_bit_import = VulkanDmabufImportState::from_dmabuf(&ten_bit_dmabuf).unwrap();
    assert!(!caps.has_dmabuf_render_target_modifier_record(&ten_bit_import));

    let sampled_only_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Xrgb8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let sampled_only_import = VulkanDmabufImportState::from_dmabuf(&sampled_only_dmabuf).unwrap();
    assert!(!caps.has_dmabuf_render_target_modifier_record(&sampled_only_import));

    let color_only_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Argb8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let color_only_import = VulkanDmabufImportState::from_dmabuf(&color_only_dmabuf).unwrap();
    assert!(!caps.has_dmabuf_render_target_modifier_record(&color_only_import));

    let multiplane_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4), (1, 4, 4)],
    );
    let multiplane_import = VulkanDmabufImportState::from_dmabuf(&multiplane_dmabuf).unwrap();
    assert!(!caps.has_dmabuf_render_target_modifier_record(&multiplane_import));

    assert!(caps.dmabuf_import.iter().next().is_none());
    assert!(caps.dmabuf_export.iter().next().is_none());
    assert!(caps.dmabuf_render_target.iter().next().is_none());
}

#[test]
fn vulkan_device_state_uninitialized_starts_empty() {
    let device = VulkanDeviceState::empty_for_tests();
    assert!(device.instance.is_none());
    assert!(device.physical_device.is_none());
    assert!(device.logical_device.is_none());
    assert!(device.memory_properties.is_none());
    assert!(!device.capabilities.device.available);
    assert!(device.enabled_extensions.is_empty());
    assert!(device.external_memory_fns.is_none());
    assert!(device.external_sync_fns.is_none());
    assert_eq!(device.queue_families.graphics, None);
    assert_eq!(device.queue_families.transfer, None);
    assert!(device.queues.graphics.is_none());
    assert!(device.queues.transfer.is_none());
    assert!(device.graphics_command_pool.is_none());
    assert!(device.transfer_command_pool.is_none());
}

#[test]
fn dmabuf_external_image_format_query_is_disabled_without_prerequisites() {
    let device = VulkanDeviceState::empty_for_tests();
    let dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let import = VulkanDmabufImportState::from_dmabuf(&dmabuf).unwrap();

    assert!(
        device
            .dmabuf_external_image_format_properties(&import)
            .unwrap()
            .is_none()
    );
    assert!(device.dmabuf_import_candidate(&import).unwrap().is_none());
    assert!(device.dmabuf_render_target_candidate(&import).unwrap().is_none());
    assert!(
        device
            .create_dmabuf_render_target_image(&import)
            .unwrap()
            .is_none()
    );
    assert!(
        device
            .create_bound_dmabuf_render_target_image(&dmabuf)
            .unwrap()
            .is_none()
    );
    assert!(
        device
            .dmabuf_render_target_external_image_format_properties(&import)
            .unwrap()
            .is_none()
    );
    assert!(device.create_dmabuf_import_image(&import).unwrap().is_none());
    assert!(
        device
            .create_bound_dmabuf_import_image(&dmabuf)
            .unwrap()
            .is_none()
    );
    assert!(
        device
            .create_dmabuf_sampled_image_resources(&dmabuf, TextureFilter::Linear, TextureFilter::Nearest)
            .unwrap()
            .is_none()
    );
    assert!(
        // SAFETY: This uninitialized-device test returns before any Vulkan image import or acquire
        // operation because external-memory prerequisites are unavailable.
        unsafe {
            device.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout(
                &dmabuf,
                TextureFilter::Linear,
                TextureFilter::Nearest,
                None,
            )
        }
        .unwrap()
        .is_none()
    );
    assert!(
        // SAFETY: This uninitialized-device test returns before any Vulkan image import or acquire
        // operation because external-memory prerequisites are unavailable. The signaled sync point
        // does not export or import any fd.
        unsafe {
            device.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point(
                &dmabuf,
                TextureFilter::Linear,
                TextureFilter::Nearest,
                Some(&SyncPoint::signaled()),
            )
        }
        .unwrap()
        .is_none()
    );
    assert!(
        // SAFETY: This uninitialized-device test returns before any Vulkan image import or acquire
        // operation because external-memory prerequisites are unavailable.
        unsafe { device.create_acquired_dmabuf_render_target_image(&dmabuf, false, None) }
            .unwrap()
            .is_none()
    );
    assert!(
        // SAFETY: This uninitialized-device test returns before any Vulkan image import or acquire
        // operation because external-memory prerequisites are unavailable. The signaled sync point
        // does not export or import any fd.
        unsafe {
            device.create_acquired_dmabuf_render_target_image_with_sync_point(
                &dmabuf,
                false,
                Some(&SyncPoint::signaled()),
            )
        }
        .unwrap()
        .is_none()
    );
    assert!(
        // SAFETY: This uninitialized-device test returns before any Vulkan image import, sync wait,
        // fd import, or acquire operation because external-memory prerequisites are unavailable.
        unsafe {
            device.create_acquired_dmabuf_render_target_image_with_sync_point(
                &dmabuf,
                false,
                Some(&SyncPoint::from(InterruptedFence)),
            )
        }
        .unwrap()
        .is_none()
    );
    assert!(
        // SAFETY: This uninitialized-device test returns before any Vulkan image import, sync wait,
        // fd import, or acquire operation because external-memory prerequisites are unavailable.
        unsafe {
            device.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point(
                &dmabuf,
                TextureFilter::Linear,
                TextureFilter::Nearest,
                Some(&SyncPoint::from(InterruptedFence)),
            )
        }
        .unwrap()
        .is_none()
    );
}

#[test]
fn sync_point_wait_semaphore_falls_back_without_vulkan_sync_file_import() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(
        unsafe {
            // SAFETY: A signaled sync point does not export or import any fd.
            device.import_sync_point_wait_semaphore(&SyncPoint::signaled())
        }
        .unwrap()
        .is_none()
    );
    assert!(
        unsafe {
            // SAFETY: Already-signaled sync points are short-circuited before fd export/import.
            device.import_sync_point_wait_semaphore(&SyncPoint::from(SignaledExportableFence))
        }
        .unwrap()
        .is_none()
    );
    assert!(
        unsafe {
            // SAFETY: This fence is non-exportable, so the helper uses only CPU waiting.
            device.import_sync_point_wait_semaphore(&SyncPoint::from(CpuWaitFence {
                exportable: false,
                exports_fd: false,
            }))
        }
        .unwrap()
        .is_none()
    );
    assert!(
        unsafe {
            // SAFETY: The empty test device cannot import sync-file fds, so this falls back to CPU wait.
            device.import_sync_point_wait_semaphore(&SyncPoint::from(CpuWaitFence {
                exportable: true,
                exports_fd: false,
            }))
        }
        .unwrap()
        .is_none()
    );
    assert!(
        unsafe {
            // SAFETY: The empty test device cannot import sync-file fds, so this falls back to CPU wait
            // without exporting the test fd.
            device.import_sync_point_wait_semaphore(&SyncPoint::from(CpuWaitFence {
                exportable: true,
                exports_fd: true,
            }))
        }
        .unwrap()
        .is_none()
    );
    assert!(matches!(
        unsafe {
            // SAFETY: This non-exportable fence exercises CPU wait interruption only.
            device.import_sync_point_wait_semaphore(&SyncPoint::from(InterruptedFence))
        },
        Err(VulkanError::SyncInterrupted)
    ));
}

#[test]
fn sync_file_release_fence_exports_and_waits_pollable_fd() {
    let signaled = sync_point_from_sync_file(None);
    assert!(!signaled.contains_fence());
    assert!(signaled.is_reached());
    assert!(signaled.wait().is_ok());

    // `/dev/null` is used only as an always-ready pollable fd for the private fence wrapper. Runtime
    // Vulkan paths pass Linux sync-file fds exported from Vulkan semaphores.
    let fd: OwnedFd = File::open("/dev/null").unwrap().into();
    let sync = sync_point_from_sync_file(Some(fd));
    assert!(sync.contains_fence());
    assert!(sync.is_exportable());
    assert!(sync.export().is_some());
    assert!(sync.is_reached());
    assert!(sync.wait().is_ok());
}

#[test]
fn dmabuf_import_image_state_tracks_sampled_dmabuf_without_advertising_renderability() {
    let dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::Y_INVERT,
        &[(0, 0, 16)],
    );
    let import = VulkanDmabufImportState::from_dmabuf(&dmabuf).unwrap();
    let image = dmabuf_import_image_state(&import);

    assert_eq!(image.size, (4, 3).into());
    assert_eq!(image.format, Some(Fourcc::Abgr8888));
    assert_eq!(image.source, VulkanImageSource::DmabufImport);
    assert!(image.usage.sampled);
    assert!(!image.usage.transfer_dst);
    assert!(!image.usage.color_attachment);
    assert_eq!(image.layout, VulkanImageLayoutState::Undefined);
    assert!(image.sync.external_acquire_pending);
    assert_eq!(
        image.sync.external_ownership,
        VulkanExternalImageOwnership::ForeignUnknown
    );
    assert!(!image.sync.pending_write);
    assert!(!image.sync.exportable_sync);
    assert!(import.y_inverted);

    let acquired = dmabuf_acquired_image_state(&import);
    assert_eq!(acquired.size, (4, 3).into());
    assert_eq!(acquired.format, Some(Fourcc::Abgr8888));
    assert_eq!(acquired.source, VulkanImageSource::DmabufImport);
    assert!(acquired.usage.sampled);
    assert_eq!(acquired.layout, VulkanImageLayoutState::ShaderReadOnly);
    assert!(!acquired.sync.external_acquire_pending);
    assert_eq!(
        acquired.sync.external_ownership,
        VulkanExternalImageOwnership::Local
    );
}

#[test]
fn dmabuf_render_target_image_state_tracks_color_attachment_without_advertising_support() {
    let dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let import = VulkanDmabufImportState::from_dmabuf(&dmabuf).unwrap();
    let image = dmabuf_render_target_image_state(&import);

    assert_eq!(image.size, (4, 3).into());
    assert_eq!(image.format, Some(Fourcc::Abgr8888));
    assert_eq!(image.source, VulkanImageSource::DmabufImport);
    assert!(image.usage.color_attachment);
    assert!(!image.usage.sampled);
    assert!(!image.usage.transfer_src);
    assert!(!image.usage.transfer_dst);
    assert_eq!(image.layout, VulkanImageLayoutState::Undefined);
    assert!(image.sync.external_acquire_pending);
    assert_eq!(
        image.sync.external_ownership,
        VulkanExternalImageOwnership::ForeignUnknown
    );
    assert!(!image.sync.pending_write);
    assert!(!image.sync.exportable_sync);

    let acquired = dmabuf_acquired_render_target_image_state(&import);
    assert_eq!(acquired.size, (4, 3).into());
    assert_eq!(acquired.format, Some(Fourcc::Abgr8888));
    assert_eq!(acquired.source, VulkanImageSource::RenderTarget);
    assert!(acquired.usage.color_attachment);
    assert!(!acquired.usage.sampled);
    assert_eq!(acquired.layout, VulkanImageLayoutState::ColorAttachment);
    assert!(!acquired.sync.external_acquire_pending);
    assert_eq!(
        acquired.sync.external_ownership,
        VulkanExternalImageOwnership::Local
    );
}

#[test]
fn dmabuf_import_image_plane_layouts_track_metadata() {
    let dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4), (1, 16, 4)],
    );
    let import = VulkanDmabufImportState::from_dmabuf(&dmabuf).unwrap();
    let layouts = dmabuf_plane_layouts(&import);

    assert_eq!(layouts.len(), 2);
    assert_eq!(layouts[0].offset, 0);
    assert_eq!(layouts[0].row_pitch, 4);
    assert_eq!(layouts[0].size, 0);
    assert_eq!(layouts[0].array_pitch, 0);
    assert_eq!(layouts[0].depth_pitch, 0);
    assert_eq!(layouts[1].offset, 16);
    assert_eq!(layouts[1].row_pitch, 4);
}

#[test]
fn dmabuf_import_memory_type_bits_intersect_image_and_fd_requirements() {
    assert_eq!(dmabuf_import_memory_type_bits(0b1110, 0b1010), 0b1010);
    assert_eq!(dmabuf_import_memory_type_bits(0b0100, 0b0010), 0);
}

#[test]
fn dmabuf_import_candidate_requires_importable_single_sample_extent() {
    let dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let import = VulkanDmabufImportState::from_dmabuf(&dmabuf).unwrap();

    let supported = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 4,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
    );
    assert!(supported.supports_sampled_import(&import));

    let mut zero_width_import = import.clone();
    zero_width_import.size = (0, 3).into();
    assert!(!supported.supports_sampled_import(&zero_width_import));

    let mut zero_height_import = import.clone();
    zero_height_import.size = (4, 0).into();
    assert!(!supported.supports_sampled_import(&zero_height_import));

    let non_importable = dmabuf_external_image_properties_for_tests(
        false,
        vk::Extent3D {
            width: 4,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
    );
    assert!(!non_importable.supports_sampled_import(&import));

    let too_small = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 3,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
    );
    assert!(!too_small.supports_sampled_import(&import));

    let too_short = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 4,
            height: 2,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
    );
    assert!(!too_short.supports_sampled_import(&import));

    let zero_depth = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 4,
            height: 3,
            depth: 0,
        },
        vk::SampleCountFlags::TYPE_1,
    );
    assert!(!zero_depth.supports_sampled_import(&import));

    let mut zero_layers = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 4,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
    );
    zero_layers.image_format_properties.max_array_layers = 0;
    assert!(!zero_layers.supports_sampled_import(&import));

    let mut zero_mip_levels = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 4,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
    );
    zero_mip_levels.image_format_properties.max_mip_levels = 0;
    assert!(!zero_mip_levels.supports_sampled_import(&import));

    let no_single_sample = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 4,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_2,
    );
    assert!(!no_single_sample.supports_sampled_import(&import));

    let mut dedicated_only = supported;
    dedicated_only.dedicated_only = true;
    dedicated_only.external_memory_properties.external_memory_features |=
        vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY;
    assert!(dedicated_only.supports_sampled_import(&import));
    assert!(dedicated_only.dedicated_only);
}

#[test]
fn dmabuf_render_target_candidate_requires_importable_single_plane_single_sample_extent() {
    let dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let import = VulkanDmabufImportState::from_dmabuf(&dmabuf).unwrap();
    let supported = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 4,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
    );

    assert!(supported.supports_render_target_import(&import));

    let multiplane_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4), (1, 4, 4)],
    );
    let multiplane_import = VulkanDmabufImportState::from_dmabuf(&multiplane_dmabuf).unwrap();
    assert!(!supported.supports_render_target_import(&multiplane_import));

    let too_small = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 3,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
    );
    assert!(!too_small.supports_render_target_import(&import));

    let no_single_sample = dmabuf_external_image_properties_for_tests(
        true,
        vk::Extent3D {
            width: 4,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_2,
    );
    assert!(!no_single_sample.supports_render_target_import(&import));

    let non_importable = dmabuf_external_image_properties_for_tests(
        false,
        vk::Extent3D {
            width: 4,
            height: 3,
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
    );
    assert!(!non_importable.supports_render_target_import(&import));
}

#[test]
fn memory_type_lookup_selects_supported_required_properties() {
    let properties = memory_properties_for_tests(&[
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
    ]);

    assert_eq!(
        find_memory_type_index(
            &properties,
            0b10,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
        .unwrap(),
        1
    );
}

#[test]
fn memory_type_lookup_can_require_coherent_host_visible_memory() {
    let properties = memory_properties_for_tests(&[
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ]);

    assert_eq!(
        find_memory_type_index(
            &properties,
            0b11,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap(),
        1
    );
}

#[test]
fn memory_type_lookup_rejects_missing_required_properties() {
    let properties = memory_properties_for_tests(&[vk::MemoryPropertyFlags::HOST_VISIBLE]);

    assert!(matches!(
        find_memory_type_index(&properties, 0b1, vk::MemoryPropertyFlags::DEVICE_LOCAL),
        Err(VulkanError::MemoryTypeUnsupported)
    ));
}

#[test]
fn memory_type_lookup_respects_type_bits() {
    let properties = memory_properties_for_tests(&[
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
    ]);

    assert!(matches!(
        find_memory_type_index(&properties, 0b01, vk::MemoryPropertyFlags::HOST_VISIBLE),
        Err(VulkanError::MemoryTypeUnsupported)
    ));
}

#[test]
fn memory_type_lookup_requires_initialized_memory_properties() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(matches!(
        device.find_memory_type_index(u32::MAX, vk::MemoryPropertyFlags::empty()),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing memory properties"
    ));
}

#[test]
fn image_layout_transition_requires_matching_image_usage() {
    assert!(
        image_layout_transition(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST,
        )
        .is_ok()
    );
    assert!(matches!(
        image_layout_transition(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageUsageFlags::SAMPLED,
        ),
        Err(VulkanError::UnsupportedOperation(
            "image transfer destination usage"
        ))
    ));
    assert!(
        image_layout_transition(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageUsageFlags::SAMPLED,
        )
        .is_ok()
    );
    assert!(matches!(
        image_layout_transition(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST,
        ),
        Err(VulkanError::UnsupportedOperation("image sampled usage"))
    ));
    assert!(
        image_layout_transition(
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        )
        .is_ok()
    );
    assert!(matches!(
        image_layout_transition(
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageUsageFlags::SAMPLED,
        ),
        Err(VulkanError::UnsupportedOperation(
            "image transfer destination usage"
        ))
    ));
    assert!(
        image_layout_transition(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .is_ok()
    );
    assert!(matches!(
        image_layout_transition(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST,
        ),
        Err(VulkanError::UnsupportedOperation("image transfer source usage"))
    ));
    assert!(
        image_layout_transition(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .is_ok()
    );
    assert!(matches!(
        image_layout_transition(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC,
        ),
        Err(VulkanError::UnsupportedOperation(
            "image transfer destination usage"
        ))
    ));
    assert!(
        image_layout_transition(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .is_ok()
    );
    assert!(
        image_layout_transition(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .is_ok()
    );
    assert!(
        image_layout_transition(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .is_ok()
    );
    assert!(
        image_layout_transition(
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .is_ok()
    );
}

#[test]
fn sampled_dmabuf_foreign_barriers_require_known_external_layout() {
    let usage = vk::ImageUsageFlags::SAMPLED;
    let fresh_import_sync = VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
        ..VulkanImageSyncState::default()
    };
    let released_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    let local_sync = VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    };
    let pending_local_sync = VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    };
    let no_pending_sync = VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::ForeignKnownGeneral,
        ..VulkanImageSyncState::default()
    };

    assert!(fresh_import_sync.external_acquire_pending);
    assert_eq!(
        fresh_import_sync.external_ownership,
        VulkanExternalImageOwnership::ForeignUnknown
    );
    assert_eq!(fresh_import_sync.known_foreign_layout(), None);
    assert!(released_sync.external_acquire_pending);
    assert_eq!(
        released_sync.external_ownership,
        VulkanExternalImageOwnership::ForeignKnownGeneral
    );
    assert_eq!(
        released_sync.known_foreign_layout(),
        Some(vk::ImageLayout::GENERAL)
    );
    let mut acquire_transition_sync = released_sync;
    assert!(matches!(
        // SAFETY: This intentionally checks precondition validation and returns before mutating
        // because no acquire transfer is pending.
        unsafe { acquire_transition_sync.complete_sampled_dmabuf_foreign_acquire() },
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    acquire_transition_sync
        .begin_sampled_dmabuf_foreign_acquire()
        .unwrap();
    assert!(acquire_transition_sync.external_acquire_pending);
    assert_eq!(
        acquire_transition_sync.external_ownership,
        VulkanExternalImageOwnership::AcquirePending
    );
    assert_eq!(acquire_transition_sync.known_foreign_layout(), None);
    assert!(!acquire_transition_sync.is_locally_usable());
    assert!(matches!(
        plan_sampled_dmabuf_foreign_acquire_barrier(&acquire_transition_sync, 2, usage),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending transfer only after the corresponding Vulkan barrier has completed.
    unsafe {
        acquire_transition_sync
            .complete_sampled_dmabuf_foreign_acquire()
            .unwrap()
    };
    assert!(!acquire_transition_sync.external_acquire_pending);
    assert_eq!(
        acquire_transition_sync.external_ownership,
        VulkanExternalImageOwnership::Local
    );
    assert!(acquire_transition_sync.is_locally_usable());
    let mut aborted_release_sync = acquire_transition_sync;
    aborted_release_sync
        .begin_sampled_dmabuf_foreign_release()
        .unwrap();
    assert_eq!(
        aborted_release_sync.external_ownership,
        VulkanExternalImageOwnership::ReleasePending
    );
    assert!(!aborted_release_sync.is_locally_usable());
    assert!(matches!(
        plan_sampled_dmabuf_foreign_release_barrier(
            &aborted_release_sync,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            2,
            usage,
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    aborted_release_sync
        .abort_sampled_dmabuf_foreign_release()
        .unwrap();
    assert_eq!(
        aborted_release_sync.external_ownership,
        VulkanExternalImageOwnership::Local
    );
    assert!(matches!(
        // SAFETY: This intentionally checks precondition validation and returns before mutating
        // because no release transfer is pending.
        unsafe { acquire_transition_sync.complete_sampled_dmabuf_foreign_release() },
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    acquire_transition_sync
        .begin_sampled_dmabuf_foreign_release()
        .unwrap();
    assert_eq!(
        acquire_transition_sync.external_ownership,
        VulkanExternalImageOwnership::ReleasePending
    );
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending transfer only after the corresponding Vulkan barrier has completed.
    unsafe {
        acquire_transition_sync
            .complete_sampled_dmabuf_foreign_release()
            .unwrap()
    };
    assert!(acquire_transition_sync.external_acquire_pending);
    assert_eq!(
        acquire_transition_sync.external_ownership,
        VulkanExternalImageOwnership::ForeignKnownGeneral
    );
    assert_eq!(
        acquire_transition_sync.known_foreign_layout(),
        Some(vk::ImageLayout::GENERAL)
    );
    let mut aborted_acquire_sync = released_sync;
    aborted_acquire_sync
        .begin_sampled_dmabuf_foreign_acquire()
        .unwrap();
    aborted_acquire_sync
        .abort_sampled_dmabuf_foreign_acquire()
        .unwrap();
    assert_eq!(aborted_acquire_sync, released_sync);
    for mut invalid_acquire in [fresh_import_sync, local_sync, no_pending_sync] {
        assert!(matches!(
            invalid_acquire.begin_sampled_dmabuf_foreign_acquire(),
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
        assert!(matches!(
            // SAFETY: This intentionally checks precondition validation and returns before mutating.
            unsafe { invalid_acquire.complete_sampled_dmabuf_foreign_acquire() },
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
    }
    for mut invalid_release in [fresh_import_sync, released_sync, pending_local_sync] {
        assert!(matches!(
            invalid_release.begin_sampled_dmabuf_foreign_release(),
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
        assert!(matches!(
            // SAFETY: This intentionally checks precondition validation and returns before mutating.
            unsafe { invalid_release.complete_sampled_dmabuf_foreign_release() },
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
    }
    assert!(matches!(
        plan_sampled_dmabuf_foreign_acquire_barrier(&no_pending_sync, 2, usage),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    assert_eq!(
        plan_sampled_dmabuf_foreign_acquire_barrier(&local_sync, 2, usage).unwrap(),
        None
    );
    assert!(matches!(
        plan_sampled_dmabuf_foreign_acquire_barrier(&fresh_import_sync, 2, usage),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    assert!(matches!(
        plan_sampled_dmabuf_foreign_acquire_barrier(&released_sync, vk::QUEUE_FAMILY_FOREIGN_EXT, usage,),
        Err(VulkanError::UnsupportedOperation("dmabuf queue family"))
    ));

    assert!(matches!(
        sampled_dmabuf_foreign_acquire_barrier(vk::ImageLayout::UNDEFINED, 0, usage),
        Err(VulkanError::UnsupportedOperation("dmabuf external layout"))
    ));
    assert!(matches!(
        sampled_dmabuf_foreign_acquire_barrier(vk::ImageLayout::PREINITIALIZED, 0, usage),
        Err(VulkanError::UnsupportedOperation("dmabuf external layout"))
    ));
    assert!(matches!(
        sampled_dmabuf_foreign_acquire_barrier(
            vk::ImageLayout::GENERAL,
            0,
            vk::ImageUsageFlags::TRANSFER_DST,
        ),
        Err(VulkanError::UnsupportedOperation("image sampled usage"))
    ));
    assert!(matches!(
        sampled_dmabuf_foreign_release_barrier(vk::QUEUE_FAMILY_FOREIGN_EXT, usage),
        Err(VulkanError::UnsupportedOperation("dmabuf queue family"))
    ));
    assert!(matches!(
        sampled_dmabuf_foreign_release_barrier(0, vk::ImageUsageFlags::TRANSFER_DST),
        Err(VulkanError::UnsupportedOperation("image sampled usage"))
    ));
    for special_queue_family in [
        vk::QUEUE_FAMILY_IGNORED,
        vk::QUEUE_FAMILY_EXTERNAL,
        vk::QUEUE_FAMILY_FOREIGN_EXT,
    ] {
        assert!(matches!(
            sampled_dmabuf_foreign_acquire_barrier(vk::ImageLayout::GENERAL, special_queue_family, usage),
            Err(VulkanError::UnsupportedOperation("dmabuf queue family"))
        ));
        assert!(matches!(
            sampled_dmabuf_foreign_release_barrier(special_queue_family, usage),
            Err(VulkanError::UnsupportedOperation("dmabuf queue family"))
        ));
    }

    let acquire = sampled_dmabuf_foreign_acquire_barrier(vk::ImageLayout::GENERAL, 2, usage).unwrap();
    assert_eq!(acquire.src_stage, vk::PipelineStageFlags::TOP_OF_PIPE);
    assert_eq!(acquire.dst_stage, vk::PipelineStageFlags::FRAGMENT_SHADER);
    assert_eq!(acquire.src_access, vk::AccessFlags::empty());
    assert_eq!(acquire.dst_access, vk::AccessFlags::SHADER_READ);
    assert_eq!(acquire.old_layout, vk::ImageLayout::GENERAL);
    assert_eq!(acquire.new_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    assert_eq!(acquire.src_queue_family_index, vk::QUEUE_FAMILY_FOREIGN_EXT);
    assert_eq!(acquire.dst_queue_family_index, 2);
    assert_eq!(
        plan_sampled_dmabuf_foreign_acquire_barrier(&released_sync, 2, usage).unwrap(),
        Some(acquire)
    );
    let release = sampled_dmabuf_foreign_release_barrier(2, usage).unwrap();
    assert_eq!(release.src_stage, vk::PipelineStageFlags::FRAGMENT_SHADER);
    assert_eq!(release.dst_stage, vk::PipelineStageFlags::BOTTOM_OF_PIPE);
    assert_eq!(release.src_access, vk::AccessFlags::SHADER_READ);
    assert_eq!(release.dst_access, vk::AccessFlags::empty());
    assert_eq!(release.old_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    assert_eq!(release.new_layout, vk::ImageLayout::GENERAL);
    assert_eq!(release.src_queue_family_index, 2);
    assert_eq!(release.dst_queue_family_index, vk::QUEUE_FAMILY_FOREIGN_EXT);
    assert_eq!(
        plan_sampled_dmabuf_foreign_release_barrier(
            &VulkanImageSyncState::default(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            2,
            usage,
        )
        .unwrap(),
        None
    );
    assert!(matches!(
        plan_sampled_dmabuf_foreign_release_barrier(
            &released_sync,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            2,
            usage,
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    assert!(matches!(
        plan_sampled_dmabuf_foreign_release_barrier(
            &pending_local_sync,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            2,
            usage,
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    assert!(matches!(
        plan_sampled_dmabuf_foreign_release_barrier(&local_sync, vk::ImageLayout::GENERAL, 2, usage,),
        Err(VulkanError::UnsupportedOperation("dmabuf local layout"))
    ));
    assert_eq!(
        plan_sampled_dmabuf_foreign_release_barrier(
            &local_sync,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            2,
            usage,
        )
        .unwrap(),
        Some(release)
    );
    let image = vk::Image::null();
    let vk_barrier = acquire.to_color_image_memory_barrier(image);
    assert_eq!(vk_barrier.s_type, vk::StructureType::IMAGE_MEMORY_BARRIER);
    assert!(vk_barrier.p_next.is_null());
    assert_eq!(vk_barrier.src_access_mask, acquire.src_access);
    assert_eq!(vk_barrier.dst_access_mask, acquire.dst_access);
    assert_eq!(vk_barrier.old_layout, acquire.old_layout);
    assert_eq!(vk_barrier.new_layout, acquire.new_layout);
    assert_eq!(vk_barrier.src_queue_family_index, acquire.src_queue_family_index);
    assert_eq!(vk_barrier.dst_queue_family_index, acquire.dst_queue_family_index);
    assert_eq!(vk_barrier.image, image);
    assert_eq!(
        vk_barrier.subresource_range.aspect_mask,
        vk::ImageAspectFlags::COLOR
    );
    assert_eq!(vk_barrier.subresource_range.base_mip_level, 0);
    assert_eq!(vk_barrier.subresource_range.level_count, 1);
    assert_eq!(vk_barrier.subresource_range.base_array_layer, 0);
    assert_eq!(vk_barrier.subresource_range.layer_count, 1);

    let release_vk_barrier = release.to_color_image_memory_barrier(image);
    assert_eq!(release_vk_barrier.src_access_mask, release.src_access);
    assert_eq!(release_vk_barrier.dst_access_mask, release.dst_access);
    assert_eq!(release_vk_barrier.old_layout, release.old_layout);
    assert_eq!(release_vk_barrier.new_layout, release.new_layout);
    assert_eq!(
        release_vk_barrier.src_queue_family_index,
        release.src_queue_family_index
    );
    assert_eq!(
        release_vk_barrier.dst_queue_family_index,
        release.dst_queue_family_index
    );
}

#[test]
fn dmabuf_render_target_foreign_barriers_distinguish_discard_and_preserve_acquire() {
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    let fresh_import_sync = VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
        ..VulkanImageSyncState::default()
    };
    let released_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    let local_sync = VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    };
    let pending_local_sync = VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    };

    let discard_acquire =
        dmabuf_render_target_foreign_acquire_barrier(vk::ImageLayout::UNDEFINED, 2, usage).unwrap();
    assert_eq!(discard_acquire.src_stage, vk::PipelineStageFlags::TOP_OF_PIPE);
    assert_eq!(
        discard_acquire.dst_stage,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
    );
    assert_eq!(discard_acquire.src_access, vk::AccessFlags::empty());
    assert_eq!(
        discard_acquire.dst_access,
        vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(discard_acquire.old_layout, vk::ImageLayout::UNDEFINED);
    assert_eq!(
        discard_acquire.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        discard_acquire.src_queue_family_index,
        vk::QUEUE_FAMILY_FOREIGN_EXT
    );
    assert_eq!(discard_acquire.dst_queue_family_index, 2);
    assert_eq!(
        plan_dmabuf_render_target_foreign_acquire_barrier(&fresh_import_sync, 2, usage, false).unwrap(),
        Some(discard_acquire)
    );
    assert!(matches!(
        plan_dmabuf_render_target_foreign_acquire_barrier(&fresh_import_sync, 2, usage, true),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));

    let preserve_acquire =
        dmabuf_render_target_foreign_acquire_barrier(vk::ImageLayout::GENERAL, 2, usage).unwrap();
    assert_eq!(preserve_acquire.old_layout, vk::ImageLayout::GENERAL);
    assert_eq!(
        preserve_acquire.new_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        plan_dmabuf_render_target_foreign_acquire_barrier(&released_sync, 2, usage, true).unwrap(),
        Some(preserve_acquire)
    );
    assert_eq!(
        plan_dmabuf_render_target_foreign_acquire_barrier(&local_sync, 2, usage, false).unwrap(),
        None
    );
    assert!(matches!(
        dmabuf_render_target_foreign_acquire_barrier(vk::ImageLayout::PREINITIALIZED, 2, usage),
        Err(VulkanError::UnsupportedOperation("dmabuf external layout"))
    ));
    assert!(matches!(
        dmabuf_render_target_foreign_acquire_barrier(
            vk::ImageLayout::UNDEFINED,
            2,
            vk::ImageUsageFlags::SAMPLED,
        ),
        Err(VulkanError::UnsupportedOperation("image color attachment usage"))
    ));
    assert!(matches!(
        dmabuf_render_target_foreign_acquire_barrier(
            vk::ImageLayout::UNDEFINED,
            vk::QUEUE_FAMILY_FOREIGN_EXT,
            usage,
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf queue family"))
    ));

    let release = dmabuf_render_target_foreign_release_barrier(2, usage).unwrap();
    assert_eq!(release.src_stage, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
    assert_eq!(release.dst_stage, vk::PipelineStageFlags::BOTTOM_OF_PIPE);
    assert_eq!(
        release.src_access,
        vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(release.dst_access, vk::AccessFlags::empty());
    assert_eq!(release.old_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    assert_eq!(release.new_layout, vk::ImageLayout::GENERAL);
    assert_eq!(release.src_queue_family_index, 2);
    assert_eq!(release.dst_queue_family_index, vk::QUEUE_FAMILY_FOREIGN_EXT);
    assert_eq!(
        plan_dmabuf_render_target_foreign_release_barrier(
            &local_sync,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            2,
            usage,
        )
        .unwrap(),
        Some(release)
    );
    assert_eq!(
        plan_dmabuf_render_target_foreign_release_barrier(
            &VulkanImageSyncState::default(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            2,
            usage,
        )
        .unwrap(),
        None
    );
    assert!(matches!(
        plan_dmabuf_render_target_foreign_release_barrier(&local_sync, vk::ImageLayout::GENERAL, 2, usage,),
        Err(VulkanError::UnsupportedOperation("dmabuf local layout"))
    ));
    assert!(matches!(
        plan_dmabuf_render_target_foreign_release_barrier(
            &pending_local_sync,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            2,
            usage,
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    let mut acquire_pending_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    acquire_pending_sync
        .begin_dmabuf_render_target_foreign_acquire(true)
        .unwrap();
    assert!(matches!(
        plan_dmabuf_render_target_foreign_release_barrier(
            &acquire_pending_sync,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            2,
            usage,
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    assert!(matches!(
        dmabuf_render_target_foreign_release_barrier(vk::QUEUE_FAMILY_FOREIGN_EXT, usage),
        Err(VulkanError::UnsupportedOperation("dmabuf queue family"))
    ));
    assert!(matches!(
        dmabuf_render_target_foreign_release_barrier(2, vk::ImageUsageFlags::SAMPLED),
        Err(VulkanError::UnsupportedOperation("image color attachment usage"))
    ));

    let image = vk::Image::null();
    let vk_barrier = discard_acquire.to_color_image_memory_barrier(image);
    assert_eq!(vk_barrier.s_type, vk::StructureType::IMAGE_MEMORY_BARRIER);
    assert_eq!(vk_barrier.image, image);
    assert_eq!(vk_barrier.old_layout, vk::ImageLayout::UNDEFINED);
    assert_eq!(vk_barrier.new_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    assert_eq!(
        vk_barrier.subresource_range.aspect_mask,
        vk::ImageAspectFlags::COLOR
    );
    assert_eq!(vk_barrier.subresource_range.level_count, 1);
    assert_eq!(vk_barrier.subresource_range.layer_count, 1);
}

#[test]
fn dmabuf_render_target_acquire_sync_restores_discard_or_preserve_source_on_abort() {
    let mut discard_sync = VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
        ..VulkanImageSyncState::default()
    };
    let discard_restore = discard_sync
        .begin_dmabuf_render_target_foreign_acquire(false)
        .unwrap();
    assert_eq!(
        discard_restore.ownership(),
        VulkanExternalImageOwnership::ForeignUnknown
    );
    assert_eq!(
        discard_sync.external_ownership,
        VulkanExternalImageOwnership::AcquirePending
    );
    assert!(discard_sync.external_acquire_pending);
    discard_sync
        .abort_dmabuf_render_target_foreign_acquire(discard_restore)
        .unwrap();
    assert_eq!(
        discard_sync.external_ownership,
        VulkanExternalImageOwnership::ForeignUnknown
    );
    assert!(discard_sync.external_acquire_pending);

    let mut preserve_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    let preserve_restore = preserve_sync
        .begin_dmabuf_render_target_foreign_acquire(true)
        .unwrap();
    assert_eq!(
        preserve_restore.ownership(),
        VulkanExternalImageOwnership::ForeignKnownGeneral
    );
    assert_eq!(
        preserve_sync.external_ownership,
        VulkanExternalImageOwnership::AcquirePending
    );
    preserve_sync
        .abort_dmabuf_render_target_foreign_acquire(preserve_restore)
        .unwrap();
    assert_eq!(
        preserve_sync.external_ownership,
        VulkanExternalImageOwnership::ForeignKnownGeneral
    );
    assert_eq!(
        preserve_sync.known_foreign_layout(),
        Some(vk::ImageLayout::GENERAL)
    );

    let mut completed_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    completed_sync
        .begin_dmabuf_render_target_foreign_acquire(true)
        .unwrap();
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending transfer only after the corresponding Vulkan barrier has completed.
    unsafe {
        completed_sync
            .complete_dmabuf_render_target_foreign_acquire()
            .unwrap()
    };
    assert_eq!(
        completed_sync.external_ownership,
        VulkanExternalImageOwnership::Local
    );
    assert!(!completed_sync.external_acquire_pending);
    assert!(completed_sync.is_locally_usable());

    assert!(matches!(
        VulkanImageSyncState {
            external_acquire_pending: true,
            external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
            ..VulkanImageSyncState::default()
        }
        .begin_dmabuf_render_target_foreign_acquire(true),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    assert!(matches!(
        completed_sync.abort_dmabuf_render_target_foreign_acquire(
            VulkanImageSyncState::foreign_known_general_for_dmabuf_import()
                .begin_dmabuf_render_target_foreign_acquire(true)
                .unwrap()
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));

    let mut pending_without_restore_token = VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::AcquirePending,
        external_acquire_kind: VulkanExternalImageAcquireKind::RenderTarget,
        ..VulkanImageSyncState::default()
    };
    // SAFETY: This intentionally checks precondition validation and returns before mutating because
    // a render-target acquire completion must have a matching restore token from begin.
    unsafe {
        assert!(matches!(
            pending_without_restore_token.complete_dmabuf_render_target_foreign_acquire(),
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
    }

    let mut mismatch_sync = VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
        ..VulkanImageSyncState::default()
    };
    let correct_restore = mismatch_sync
        .begin_dmabuf_render_target_foreign_acquire(false)
        .unwrap();
    let wrong_restore = VulkanImageSyncState::foreign_known_general_for_dmabuf_import()
        .begin_dmabuf_render_target_foreign_acquire(true)
        .unwrap();
    assert!(matches!(
        mismatch_sync.abort_dmabuf_render_target_foreign_acquire(wrong_restore),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    assert_eq!(
        mismatch_sync.external_ownership,
        VulkanExternalImageOwnership::AcquirePending
    );
    mismatch_sync
        .abort_dmabuf_render_target_foreign_acquire(correct_restore)
        .unwrap();
    assert_eq!(
        mismatch_sync.external_ownership,
        VulkanExternalImageOwnership::ForeignUnknown
    );

    let mut render_target_pending_sync = VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
        ..VulkanImageSyncState::default()
    };
    let render_target_restore = render_target_pending_sync
        .begin_dmabuf_render_target_foreign_acquire(false)
        .unwrap();
    assert!(matches!(
        render_target_pending_sync.abort_sampled_dmabuf_foreign_acquire(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    // SAFETY: The call is expected to fail before changing host-side state because this is a
    // render-target acquire, not a sampled acquire.
    unsafe {
        assert!(matches!(
            render_target_pending_sync.complete_sampled_dmabuf_foreign_acquire(),
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
    }
    assert_eq!(
        render_target_pending_sync.external_ownership,
        VulkanExternalImageOwnership::AcquirePending
    );
    render_target_pending_sync
        .abort_dmabuf_render_target_foreign_acquire(render_target_restore)
        .unwrap();

    let mut sampled_pending_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    sampled_pending_sync
        .begin_sampled_dmabuf_foreign_acquire()
        .unwrap();
    assert!(matches!(
        sampled_pending_sync.abort_dmabuf_render_target_foreign_acquire(
            VulkanImageSyncState::foreign_known_general_for_dmabuf_import()
                .begin_dmabuf_render_target_foreign_acquire(true)
                .unwrap()
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    // SAFETY: The call is expected to fail before changing host-side state because this is a sampled
    // acquire, not a render-target acquire.
    unsafe {
        assert!(matches!(
            sampled_pending_sync.complete_dmabuf_render_target_foreign_acquire(),
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
    }
    sampled_pending_sync
        .abort_sampled_dmabuf_foreign_acquire()
        .unwrap();

    let shared = VulkanSharedImageSyncState::new(VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
        ..VulkanImageSyncState::default()
    });
    let restore = shared.begin_dmabuf_render_target_foreign_acquire(false).unwrap();
    assert_eq!(restore.ownership(), VulkanExternalImageOwnership::ForeignUnknown);
    assert_eq!(
        shared.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::AcquirePending
    );
    shared
        .abort_dmabuf_render_target_foreign_acquire(restore)
        .unwrap();
    assert_eq!(
        shared.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::ForeignUnknown
    );

    let shared_complete =
        VulkanSharedImageSyncState::new(VulkanImageSyncState::foreign_known_general_for_dmabuf_import());
    shared_complete
        .begin_dmabuf_render_target_foreign_acquire(true)
        .unwrap();
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending transfer only after the corresponding Vulkan barrier has completed.
    unsafe {
        shared_complete
            .complete_dmabuf_render_target_foreign_acquire_for_tests()
            .unwrap()
    };
    assert_eq!(
        shared_complete.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::Local
    );
}

#[test]
fn dmabuf_render_target_release_sync_rejects_sampled_cross_operations() {
    let mut render_target_release = VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    };
    render_target_release
        .begin_dmabuf_render_target_foreign_release()
        .unwrap();
    assert_eq!(
        render_target_release.external_ownership,
        VulkanExternalImageOwnership::ReleasePending
    );
    assert_eq!(
        render_target_release.external_release_kind,
        VulkanExternalImageReleaseKind::RenderTarget
    );
    assert!(!render_target_release.external_acquire_pending);
    assert!(!render_target_release.is_locally_usable());
    assert!(matches!(
        render_target_release.abort_sampled_dmabuf_foreign_release(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    // SAFETY: The call is expected to fail before changing host-side state because this is a
    // render-target release, not a sampled release.
    unsafe {
        assert!(matches!(
            render_target_release.complete_sampled_dmabuf_foreign_release(),
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
    }
    render_target_release
        .abort_dmabuf_render_target_foreign_release()
        .unwrap();
    assert_eq!(
        render_target_release.external_ownership,
        VulkanExternalImageOwnership::Local
    );
    assert_eq!(
        render_target_release.external_release_kind,
        VulkanExternalImageReleaseKind::None
    );

    render_target_release
        .begin_dmabuf_render_target_foreign_release()
        .unwrap();
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending transfer only after the corresponding Vulkan barrier has completed.
    unsafe {
        render_target_release
            .complete_dmabuf_render_target_foreign_release()
            .unwrap()
    };
    assert_eq!(
        render_target_release.external_ownership,
        VulkanExternalImageOwnership::ForeignKnownGeneral
    );
    assert!(render_target_release.external_acquire_pending);
    assert_eq!(
        render_target_release.external_release_kind,
        VulkanExternalImageReleaseKind::None
    );
    assert_eq!(
        render_target_release.known_foreign_layout(),
        Some(vk::ImageLayout::GENERAL)
    );

    let mut sampled_release = VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    };
    sampled_release.begin_sampled_dmabuf_foreign_release().unwrap();
    assert_eq!(
        sampled_release.external_release_kind,
        VulkanExternalImageReleaseKind::Sampled
    );
    assert!(matches!(
        sampled_release.abort_dmabuf_render_target_foreign_release(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    // SAFETY: The call is expected to fail before changing host-side state because this is a sampled
    // release, not a render-target release.
    unsafe {
        assert!(matches!(
            sampled_release.complete_dmabuf_render_target_foreign_release(),
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
    }
    sampled_release.abort_sampled_dmabuf_foreign_release().unwrap();

    assert!(matches!(
        VulkanImageSyncState::default().begin_dmabuf_render_target_foreign_release(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    assert!(matches!(
        VulkanImageSyncState {
            external_ownership: VulkanExternalImageOwnership::Local,
            external_acquire_pending: true,
            ..VulkanImageSyncState::default()
        }
        .begin_dmabuf_render_target_foreign_release(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));

    let shared = VulkanSharedImageSyncState::new(VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    });
    shared.begin_dmabuf_render_target_foreign_release().unwrap();
    assert_eq!(
        shared.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::ReleasePending
    );
    shared.abort_dmabuf_render_target_foreign_release().unwrap();
    assert_eq!(
        shared.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::Local
    );

    let shared_complete = VulkanSharedImageSyncState::new(VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    });
    shared_complete
        .begin_dmabuf_render_target_foreign_release()
        .unwrap();
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending transfer only after the corresponding Vulkan barrier has completed.
    unsafe {
        shared_complete
            .complete_dmabuf_render_target_foreign_release_for_tests()
            .unwrap()
    };
    assert_eq!(
        shared_complete.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::ForeignKnownGeneral
    );
    assert!(shared_complete.get().unwrap().external_acquire_pending());
}

#[test]
fn dmabuf_render_target_release_can_nest_after_pending_acquire_host_side() {
    let mut abort_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    let acquire_restore = abort_sync
        .begin_dmabuf_render_target_foreign_acquire(true)
        .unwrap();
    abort_sync.begin_dmabuf_render_target_foreign_release().unwrap();
    assert_eq!(
        abort_sync.external_ownership,
        VulkanExternalImageOwnership::ReleasePending
    );
    assert!(abort_sync.external_acquire_pending);
    assert_eq!(
        abort_sync.external_acquire_kind,
        VulkanExternalImageAcquireKind::RenderTarget
    );
    assert_eq!(
        abort_sync.external_release_kind,
        VulkanExternalImageReleaseKind::RenderTarget
    );
    abort_sync.abort_dmabuf_render_target_foreign_release().unwrap();
    assert_eq!(
        abort_sync.external_ownership,
        VulkanExternalImageOwnership::AcquirePending
    );
    assert!(abort_sync.external_acquire_pending);
    assert_eq!(
        abort_sync.external_acquire_kind,
        VulkanExternalImageAcquireKind::RenderTarget
    );
    abort_sync
        .abort_dmabuf_render_target_foreign_acquire(acquire_restore)
        .unwrap();
    assert_eq!(
        abort_sync,
        VulkanImageSyncState::foreign_known_general_for_dmabuf_import()
    );

    let mut complete_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    complete_sync
        .begin_dmabuf_render_target_foreign_acquire(true)
        .unwrap();
    complete_sync
        .begin_dmabuf_render_target_foreign_release()
        .unwrap();
    // SAFETY: This intentionally checks precondition validation and returns before mutating because
    // release completion must happen after the acquire barrier has completed.
    unsafe {
        assert!(matches!(
            complete_sync.complete_dmabuf_render_target_foreign_release(),
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
    }
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending acquire only after the corresponding Vulkan barrier has completed.
    unsafe {
        complete_sync
            .complete_dmabuf_render_target_foreign_acquire()
            .unwrap()
    };
    assert_eq!(
        complete_sync.external_ownership,
        VulkanExternalImageOwnership::ReleasePending
    );
    assert!(!complete_sync.external_acquire_pending);
    assert_eq!(
        complete_sync.external_acquire_kind,
        VulkanExternalImageAcquireKind::None
    );
    assert_eq!(
        complete_sync.external_release_kind,
        VulkanExternalImageReleaseKind::RenderTarget
    );
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending release only after the corresponding Vulkan barrier has completed.
    unsafe {
        complete_sync
            .complete_dmabuf_render_target_foreign_release()
            .unwrap()
    };
    assert_eq!(
        complete_sync,
        VulkanImageSyncState::foreign_known_general_for_dmabuf_import()
    );
}

#[test]
fn dmabuf_render_target_pending_acquire_projects_to_release_plannable_local_sync() {
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    let mut acquire_pending_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    acquire_pending_sync
        .begin_dmabuf_render_target_foreign_acquire(true)
        .unwrap();

    let projected = project_dmabuf_render_target_sync_after_pending_acquire(acquire_pending_sync).unwrap();
    assert_eq!(projected.external_ownership, VulkanExternalImageOwnership::Local);
    assert!(!projected.external_acquire_pending);
    assert_eq!(
        projected.external_acquire_kind,
        VulkanExternalImageAcquireKind::None
    );
    assert_eq!(projected.render_target_acquire_restore_token, None);
    assert_eq!(
        plan_dmabuf_render_target_foreign_release_barrier(
            &projected,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            2,
            usage,
        )
        .unwrap(),
        Some(dmabuf_render_target_foreign_release_barrier(2, usage).unwrap())
    );

    assert!(matches!(
        project_dmabuf_render_target_sync_after_pending_acquire(VulkanImageSyncState::default()),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));

    let mut discard_pending_sync = VulkanImageSyncState {
        external_acquire_pending: true,
        external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
        ..VulkanImageSyncState::default()
    };
    discard_pending_sync
        .begin_dmabuf_render_target_foreign_acquire(false)
        .unwrap();
    let projected_discard =
        project_dmabuf_render_target_sync_after_pending_acquire(discard_pending_sync).unwrap();
    assert_eq!(
        projected_discard.external_ownership,
        VulkanExternalImageOwnership::Local
    );
    assert!(!projected_discard.external_acquire_pending);
    assert_eq!(
        projected_discard.external_acquire_kind,
        VulkanExternalImageAcquireKind::None
    );
    assert_eq!(projected_discard.render_target_acquire_restore_token, None);
    assert_eq!(projected_discard.known_foreign_layout(), None);

    let mut nested_release_pending = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    nested_release_pending
        .begin_dmabuf_render_target_foreign_acquire(true)
        .unwrap();
    nested_release_pending
        .begin_dmabuf_render_target_foreign_release()
        .unwrap();
    assert!(matches!(
        project_dmabuf_render_target_sync_after_pending_acquire(nested_release_pending),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
}

#[test]
fn shared_image_sync_state_reserves_aborts_and_completes_dmabuf_ownership() {
    let acquire_sync =
        VulkanSharedImageSyncState::new(VulkanImageSyncState::foreign_known_general_for_dmabuf_import());

    acquire_sync.begin_sampled_dmabuf_foreign_acquire().unwrap();
    assert_eq!(
        acquire_sync.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::AcquirePending
    );
    assert!(!acquire_sync.get().unwrap().is_locally_usable());
    assert!(matches!(
        acquire_sync.begin_sampled_dmabuf_foreign_acquire(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    acquire_sync.abort_sampled_dmabuf_foreign_acquire().unwrap();
    assert_eq!(
        acquire_sync.get().unwrap(),
        VulkanImageSyncState::foreign_known_general_for_dmabuf_import()
    );

    acquire_sync.begin_sampled_dmabuf_foreign_acquire().unwrap();
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending transfer only after the corresponding Vulkan barrier has completed.
    unsafe {
        acquire_sync
            .complete_sampled_dmabuf_foreign_acquire_for_tests()
            .unwrap()
    };
    assert_eq!(
        acquire_sync.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::Local
    );
    assert!(acquire_sync.get().unwrap().is_locally_usable());
    assert!(matches!(
        acquire_sync.abort_sampled_dmabuf_foreign_acquire(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));

    let release_sync = VulkanSharedImageSyncState::new(VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    });
    release_sync.begin_sampled_dmabuf_foreign_release().unwrap();
    assert_eq!(
        release_sync.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::ReleasePending
    );
    assert!(!release_sync.get().unwrap().is_locally_usable());
    assert!(matches!(
        release_sync.begin_sampled_dmabuf_foreign_release(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    release_sync.abort_sampled_dmabuf_foreign_release().unwrap();
    assert_eq!(
        release_sync.get().unwrap().external_ownership(),
        VulkanExternalImageOwnership::Local
    );

    release_sync.begin_sampled_dmabuf_foreign_release().unwrap();
    // SAFETY: This unit test exercises only the host-side state transition; production callers may
    // complete a pending transfer only after the corresponding Vulkan barrier has completed.
    unsafe {
        release_sync
            .complete_sampled_dmabuf_foreign_release_for_tests()
            .unwrap()
    };
    assert_eq!(
        release_sync.get().unwrap(),
        VulkanImageSyncState::foreign_known_general_for_dmabuf_import()
    );
    assert!(matches!(
        release_sync.abort_sampled_dmabuf_foreign_release(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
}

#[test]
fn tightly_packed_image_size_matches_supported_renderer_formats() {
    let extent = vk::Extent3D {
        width: 2,
        height: 3,
        depth: 1,
    };

    assert_eq!(
        tightly_packed_image_size(vk::Format::R8G8B8A8_UNORM, extent).unwrap(),
        24
    );
    assert_eq!(
        tightly_packed_image_size(vk::Format::B8G8R8A8_UNORM, extent).unwrap(),
        24
    );
    assert_eq!(
        tightly_packed_image_size(vk::Format::R5G6B5_UNORM_PACK16, extent).unwrap(),
        12
    );
    assert!(matches!(
        tightly_packed_image_size(
            vk::Format::R8G8B8A8_UNORM,
            vk::Extent3D {
                width: 0,
                height: 1,
                depth: 1,
            },
        ),
        Err(VulkanError::UnsupportedOperation("zero-sized image"))
    ));
    assert!(matches!(
        tightly_packed_image_size(vk::Format::D32_SFLOAT, extent),
        Err(VulkanError::UnsupportedOperation("tightly packed image format"))
    ));
}

#[test]
fn shm_buffer_copy_repacks_offset_and_stride() {
    let data = [
        0xaa, 0xbb, 0x00, 0x00, // padding before buffer offset
        0x01, 0x02, 0x03, 0x04, // row 0 pixels
        0xee, 0xff, // row 0 stride padding
        0x05, 0x06, 0x07, 0x08, // row 1 pixels
        0xcc, 0xdd, // row 1 stride padding
    ];

    let packed = super::copy_shm_buffer_to_tightly_packed(4, 2, 2, 6, 2, data.as_ptr(), data.len()).unwrap();
    assert_eq!(packed, [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);

    assert!(matches!(
        super::copy_shm_buffer_to_tightly_packed(4, 2, 2, 3, 2, data.as_ptr(), data.len()),
        Err(VulkanError::UnsupportedOperation("wl_shm buffer layout"))
    ));
    assert!(matches!(
        super::copy_shm_buffer_to_tightly_packed(4, 2, 2, 6, 2, data.as_ptr(), 13),
        Err(VulkanError::UnsupportedOperation("wl_shm buffer length"))
    ));
    assert!(matches!(
        super::copy_shm_buffer_to_tightly_packed(4, 2, 2, 6, 2, std::ptr::null(), data.len()),
        Err(VulkanError::UnsupportedOperation("wl_shm buffer"))
    ));
}

#[test]
fn image_copy_required_size_accounts_for_regions_and_row_stride() {
    let extent = vk::Extent3D {
        width: 2,
        height: 2,
        depth: 1,
    };

    assert_eq!(
        image_copy_required_size(vk::Format::R8G8B8A8_UNORM, 0, 0, 0, extent).unwrap(),
        16
    );
    assert_eq!(
        image_copy_required_size(vk::Format::R8G8B8A8_UNORM, 4, 4, 2, extent).unwrap(),
        28
    );
    assert_eq!(
        image_copy_buffer_offset(vk::Format::R8G8B8A8_UNORM, 4, 1, 1).unwrap(),
        20
    );
    assert!(matches!(
        image_copy_required_size(vk::Format::R8G8B8A8_UNORM, 0, 1, 0, extent),
        Err(VulkanError::UnsupportedOperation("image copy row length"))
    ));
    assert!(matches!(
        image_copy_required_size(vk::Format::R8G8B8A8_UNORM, 0, 0, 1, extent),
        Err(VulkanError::UnsupportedOperation("image copy image height"))
    ));
}

#[test]
fn texture_filters_map_to_vulkan_filters() {
    assert_eq!(vulkan_filter(TextureFilter::Linear), vk::Filter::LINEAR);
    assert_eq!(vulkan_filter(TextureFilter::Nearest), vk::Filter::NEAREST);
}

#[test]
fn buffer_creation_requires_initialized_device() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(matches!(
        device.create_buffer(4096, vk::BufferUsageFlags::TRANSFER_SRC),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.destroy_buffer(vk::Buffer::null()),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.buffer_memory_requirements(vk::Buffer::null()),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.allocate_memory(4096, 0),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.free_memory(vk::DeviceMemory::null()),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.bind_buffer_memory(vk::Buffer::null(), vk::DeviceMemory::null(), 0),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.map_memory(vk::DeviceMemory::null(), 0, 4096),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.unmap_memory(vk::DeviceMemory::null()),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.flush_mapped_memory_range(vk::DeviceMemory::null(), 0, 4096),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.create_host_visible_buffer(4096, vk::BufferUsageFlags::TRANSFER_SRC),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.create_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
        ),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.destroy_image(vk::Image::null()),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.image_memory_requirements(vk::Image::null()),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.bind_image_memory(vk::Image::null(), vk::DeviceMemory::null(), 0),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
    assert!(matches!(
        device.create_bound_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing instance"
    ));
    assert!(matches!(
        device.create_sampler(TextureFilter::Linear, TextureFilter::Nearest),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
}

#[test]
fn graphics_command_buffer_allocation_requires_initialized_device() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(matches!(
        device.allocate_graphics_command_buffer(),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
}

#[test]
fn transfer_command_buffer_allocation_requires_initialized_device() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(matches!(
        device.allocate_transfer_command_buffer(),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
}

#[test]
fn shader_module_creation_rejects_empty_code() {
    // SAFETY: Empty input is rejected before a `VulkanShaderSpirv` value is produced, so this does
    // not claim invalid SPIR-V is usable by Vulkan.
    let empty_spirv = unsafe { VulkanShaderSpirv::from_words_unchecked(&[]) };

    assert!(matches!(
        empty_spirv,
        Err(VulkanError::UnsupportedOperation("shader module code"))
    ));
}

#[test]
fn sampled_texture_pipeline_shader_pair_rejects_empty_modules() {
    // SAFETY: Empty vertex input is rejected before a shader-pair value is produced, so this does
    // not claim invalid SPIR-V is usable by Vulkan.
    let empty_vertex = unsafe {
        VulkanSampledTexturePipelineShaders::from_spirv_unchecked(
            vk::Format::R8G8B8A8_UNORM,
            &[],
            TEXTURED_FRAGMENT_SHADER_SPIRV,
        )
    };
    // SAFETY: Empty fragment input is rejected before a shader-pair value is produced, so this does
    // not claim invalid SPIR-V is usable by Vulkan.
    let empty_fragment = unsafe {
        VulkanSampledTexturePipelineShaders::from_spirv_unchecked(
            vk::Format::R8G8B8A8_UNORM,
            TEXTURED_VERTEX_SHADER_SPIRV,
            &[],
        )
    };

    assert!(matches!(
        empty_vertex,
        Err(VulkanError::UnsupportedOperation("shader module code"))
    ));
    assert!(matches!(
        empty_fragment,
        Err(VulkanError::UnsupportedOperation("shader module code"))
    ));
}

#[test]
fn pipeline_layout_creation_requires_initialized_device() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(matches!(
        device.create_empty_pipeline_layout(),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
}

#[test]
fn sampled_texture_descriptor_set_layout_creation_requires_initialized_device() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(matches!(
        device.create_sampled_texture_descriptor_set_layout(),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
}

#[test]
fn sampled_texture_descriptor_pool_rejects_zero_capacity() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(matches!(
        device.create_sampled_texture_descriptor_pool(0),
        Err(VulkanError::UnsupportedOperation("descriptor pool capacity"))
    ));
}

#[test]
fn sampled_texture_descriptor_pool_creation_requires_initialized_device() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(matches!(
        device.create_sampled_texture_descriptor_pool(1),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
}

#[test]
fn sampled_texture_pipeline_layout_creation_requires_initialized_device() {
    let device = VulkanDeviceState::empty_for_tests();

    assert!(matches!(
        device.create_sampled_texture_pipeline_layout(),
        Err(VulkanError::DeviceInitializationFailed(message)) if message == "missing logical device"
    ));
}

#[test]
fn vulkan_renderer_builder_requires_explicit_physical_device() {
    assert!(matches!(
        VulkanRenderer::builder().build(),
        Err(VulkanError::VulkanUnavailable)
    ));
}

#[test]
fn offscreen_render_target_creation_requires_initialized_device() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();

    assert!(matches!(
        renderer.create_offscreen_render_target(Fourcc::Argb8888, (1, 1).into()),
        Err(VulkanError::VulkanUnavailable)
    ));
    assert!(matches!(
        renderer.create_offscreen_render_target(Fourcc::Argb8888, (0, 1).into()),
        Err(VulkanError::VulkanUnavailable)
    ));
}

#[test]
fn offscreen_render_target_clear_rejects_foreign_targets() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut target = VulkanRenderTarget::new_for_tests((1, 1).into(), Some(Fourcc::Argb8888));

    assert!(matches!(
        renderer.clear_offscreen_render_target(&mut target, Color32F::BLACK),
        Err(VulkanError::UnsupportedOperation("foreign offscreen target"))
    ));
}

#[test]
fn offscreen_render_target_read_rejects_foreign_targets() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut target = VulkanRenderTarget::new_for_tests((1, 1).into(), Some(Fourcc::Argb8888));

    assert!(matches!(
        renderer.read_offscreen_render_target(&mut target),
        Err(VulkanError::UnsupportedOperation("foreign offscreen target"))
    ));
}

#[test]
fn clear_color_value_preserves_format_alpha_semantics() {
    let clear =
        super::clear_color_value_for_format(Fourcc::Argb8888, Color32F::new(0.25, 0.5, 0.75, 0.5)).unwrap();

    assert_eq!(unsafe { clear.float32 }, [0.25, 0.5, 0.75, 0.5]);

    let opaque_clear =
        super::clear_color_value_for_format(Fourcc::Xrgb8888, Color32F::new(0.25, 0.5, 0.75, 0.5)).unwrap();

    assert_eq!(unsafe { opaque_clear.float32 }, [0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn initialized_device_capabilities_do_not_enable_format_backed_rendering_before_discovery() {
    let caps = VulkanRendererCapabilities::for_initialized_device(&[]);

    assert!(caps.device.available);
    assert!(!caps.device.multi_gpu);
    assert!(caps.device.extensions.is_empty());
    assert!(!caps.import.memory);
    assert!(!caps.import.dmabuf);
    assert!(!caps.export.memory);
    assert!(!caps.export.dmabuf);
    assert!(!caps.rendering.offscreen);
    assert!(!caps.rendering.dmabuf_targets);
    assert!(!caps.rendering.dmabuf_target_modifiers);
    assert!(!caps.rendering.dmabuf_target_development);
    assert!(!caps.rendering.blit);
    assert!(!caps.sync.explicit);
    assert!(caps.formats.records.is_empty());
}

#[test]
fn public_offscreen_create_buffer_requires_initialized_device() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();

    assert!(matches!(
        Offscreen::<VulkanRenderTarget<'static>>::create_buffer(
            &mut renderer,
            Fourcc::Abgr8888,
            (1, 1).into(),
        ),
        Err(VulkanError::UnsupportedFormat(Fourcc::Abgr8888))
    ));

    renderer.capabilities.formats.records = vec![render_target_format_record_for_tests(
        Fourcc::Abgr8888,
        VulkanFormatTiling::Optimal,
        true,
    )];

    assert!(matches!(
        Offscreen::<VulkanRenderTarget<'static>>::create_buffer(
            &mut renderer,
            Fourcc::Abgr8888,
            (1, 1).into(),
        ),
        Err(VulkanError::VulkanUnavailable)
    ));
}

#[test]
fn public_bind_rejects_invalid_vulkan_targets_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut foreign_target = render_target_for_tests(
        ContextId::new(),
        VulkanImageSource::Offscreen,
        (1, 1).into(),
        Some(Fourcc::Abgr8888),
    );
    let mut non_offscreen_target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::Uninitialized,
        (1, 1).into(),
        Some(Fourcc::Abgr8888),
    );
    let mut missing_image_target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::Offscreen,
        (1, 1).into(),
        Some(Fourcc::Abgr8888),
    );

    assert!(matches!(
        Bind::bind(&mut renderer, &mut foreign_target),
        Err(VulkanError::UnsupportedOperation("foreign render target"))
    ));
    assert!(matches!(
        Bind::bind(&mut renderer, &mut non_offscreen_target),
        Err(VulkanError::UnsupportedOperation("render target"))
    ));
    assert!(matches!(
        Bind::bind(&mut renderer, &mut missing_image_target),
        Err(VulkanError::UnsupportedOperation("render target image"))
    ));
}

#[test]
fn public_dmabuf_bind_uses_discard_acquire_path() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    renderer.capabilities.rendering.dmabuf_target_development = true;
    let mut dmabuf = dmabuf_for_tests();
    let default_acquire = VulkanDmabufRenderTargetAcquire::default();
    let signaled_sync = SyncPoint::signaled();
    let preserve_acquire = VulkanDmabufRenderTargetAcquire::preserve(Some(&signaled_sync));

    let formats = <VulkanRenderer as Bind<Dmabuf>>::supported_formats(&renderer)
        .expect("Vulkan dmabuf render targets have an explicit format set");
    assert!(formats.iter().next().is_none());
    assert!(!default_acquire.preserve_contents);
    assert!(default_acquire.acquire_sync.is_none());
    assert!(preserve_acquire.preserve_contents);
    assert!(preserve_acquire.acquire_sync.is_some());
    assert_eq!(
        <VulkanRenderer as RenderTargetLifecycle<Dmabuf>>::target_age(&renderer, &dmabuf, 3),
        0
    );
    assert!(matches!(
        <VulkanRenderer as Bind<Dmabuf>>::bind(&mut renderer, &mut dmabuf),
        Err(VulkanError::VulkanUnavailable)
    ));
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the helper returns before any
        // Vulkan import or ownership-transfer operation can occur.
        unsafe { renderer.create_acquired_dmabuf_render_target(&dmabuf, false, None) },
        Err(VulkanError::VulkanUnavailable)
    ));
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the helper returns before any
        // sync-point wait, Vulkan import, or ownership-transfer operation can occur.
        unsafe {
            renderer.create_acquired_dmabuf_render_target_with_sync_point(
                &dmabuf,
                false,
                Some(&SyncPoint::signaled()),
            )
        },
        Err(VulkanError::VulkanUnavailable)
    ));
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the public explicit API returns
        // before any sync-point wait, Vulkan import, or ownership-transfer operation can occur.
        unsafe {
            renderer.bind_dmabuf_render_target(&mut dmabuf, VulkanDmabufRenderTargetAcquire::discard())
        },
        Err(VulkanError::VulkanUnavailable)
    ));
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the public explicit API returns
        // before any sync-point wait, Vulkan import, or ownership-transfer operation can occur.
        unsafe { renderer.bind_dmabuf_render_target(&mut dmabuf, preserve_acquire) },
        Err(VulkanError::VulkanUnavailable)
    ));

    let mut discard_target = unsafe {
        // SAFETY: This scaffold renderer has no Vulkan device, so binding the wrapper returns before
        // any Vulkan import or ownership-transfer operation can occur.
        VulkanDmabufRenderTarget::discard(&mut dmabuf)
    };
    assert!(!discard_target.preserve_contents());
    assert!(!discard_target.acquire().preserve_contents);
    assert_eq!(
        <VulkanRenderer as RenderTargetLifecycle<VulkanDmabufRenderTarget<'_, '_>>>::target_age(
            &renderer,
            &discard_target,
            3,
        ),
        0
    );
    assert!(matches!(
        <VulkanRenderer as Bind<VulkanDmabufRenderTarget<'_, '_>>>::bind(&mut renderer, &mut discard_target),
        Err(VulkanError::VulkanUnavailable)
    ));
    drop(discard_target);

    let mut preserve_target = unsafe {
        // SAFETY: This scaffold renderer has no Vulkan device, so binding the wrapper returns before
        // any sync-point wait, Vulkan import, or ownership-transfer operation can occur.
        VulkanDmabufRenderTarget::preserve(&mut dmabuf, Some(&signaled_sync))
    };
    assert!(preserve_target.preserve_contents());
    assert!(preserve_target.acquire().acquire_sync.is_some());
    assert_eq!(
        <VulkanRenderer as RenderTargetLifecycle<VulkanDmabufRenderTarget<'_, '_>>>::target_age(
            &renderer,
            &preserve_target,
            3,
        ),
        3
    );
    assert!(matches!(
        <VulkanRenderer as Bind<VulkanDmabufRenderTarget<'_, '_>>>::bind(&mut renderer, &mut preserve_target),
        Err(VulkanError::VulkanUnavailable)
    ));
    drop(preserve_target);

    let mut owned_discard_target = unsafe {
        // SAFETY: This scaffold renderer has no Vulkan device, so binding the wrapper returns before
        // any Vulkan import or ownership-transfer operation can occur.
        VulkanOwnedDmabufRenderTarget::discard(dmabuf.clone())
    };
    assert!(!owned_discard_target.preserve_contents());
    assert_eq!(owned_discard_target.dmabuf().size(), dmabuf.size());
    assert_eq!(
        <VulkanRenderer as RenderTargetLifecycle<VulkanOwnedDmabufRenderTarget<'_>>>::target_age(
            &renderer,
            &owned_discard_target,
            3,
        ),
        0
    );
    assert!(matches!(
        <VulkanRenderer as Bind<VulkanOwnedDmabufRenderTarget<'_>>>::bind(
            &mut renderer,
            &mut owned_discard_target,
        ),
        Err(VulkanError::VulkanUnavailable)
    ));

    let mut owned_preserve_target = unsafe {
        // SAFETY: This scaffold renderer has no Vulkan device, so binding the wrapper returns before
        // any sync-point wait, Vulkan import, or ownership-transfer operation can occur.
        VulkanOwnedDmabufRenderTarget::preserve(dmabuf, Some(&signaled_sync))
    };
    assert!(owned_preserve_target.preserve_contents());
    assert!(owned_preserve_target.acquire().acquire_sync.is_some());
    assert_eq!(
        <VulkanRenderer as RenderTargetLifecycle<VulkanOwnedDmabufRenderTarget<'_>>>::target_age(
            &renderer,
            &owned_preserve_target,
            3,
        ),
        3
    );
    assert!(matches!(
        <VulkanRenderer as Bind<VulkanOwnedDmabufRenderTarget<'_>>>::bind(
            &mut renderer,
            &mut owned_preserve_target,
        ),
        Err(VulkanError::VulkanUnavailable)
    ));
}

#[test]
fn dmabuf_loopback_evidence_is_identity_bound_and_not_public_advertised() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let dmabuf = dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4)],
    );
    let unrelated_dmabuf = dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let evidence = unsafe {
        // SAFETY: This unit test validates token routing only; no Vulkan acquire/import is reached
        // with this token before the scaffold device lookup fails.
        VulkanDmabufLoopbackImportEvidence::new(dmabuf.weak(), SyncPoint::signaled())
    };

    assert!(evidence.is_for_dmabuf(&dmabuf));
    assert!(!evidence.is_for_dmabuf(&unrelated_dmabuf));
    assert!(evidence.acquire_sync().is_reached());
    assert!(
        renderer
            .validate_dmabuf_loopback_import_evidence(&dmabuf, &evidence)
            .is_ok()
    );
    assert!(matches!(
        renderer.validate_dmabuf_loopback_import_evidence(&unrelated_dmabuf, &evidence),
        Err(VulkanError::UnsupportedOperation("dmabuf loopback evidence"))
    ));
    assert!(matches!(
        // SAFETY: This test intentionally validates identity rejection before any Vulkan operation
        // can use the evidence.
        unsafe { renderer.import_dmabuf_texture_from_loopback(&unrelated_dmabuf, evidence) },
        Err(VulkanError::UnsupportedOperation("dmabuf loopback evidence"))
    ));

    let evidence = unsafe {
        // SAFETY: This scaffold renderer has no Vulkan device, so the helper returns before any
        // Vulkan import or ownership-transfer operation can occur.
        VulkanDmabufLoopbackImportEvidence::new(dmabuf.weak(), SyncPoint::signaled())
    };
    renderer.capabilities.formats.modifier_records = vec![modifier_record_from_properties(
        Fourcc::Abgr8888,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        },
    )];
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the helper returns before any
        // Vulkan import or ownership-transfer operation can occur.
        unsafe { renderer.import_dmabuf_texture_from_loopback(&dmabuf, evidence) },
        Err(VulkanError::VulkanUnavailable)
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_public_advertisement_contract(),
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"))
    ));
    assert!(renderer.dmabuf_formats().iter().next().is_none());

    let mut offscreen_target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::Offscreen,
        (1, 1).into(),
        Some(Fourcc::Abgr8888),
    );
    assert!(matches!(
        renderer.release_dmabuf_render_target_for_sampled_loopback(&mut offscreen_target, false),
        Err(VulkanError::UnsupportedOperation("dmabuf loopback render target"))
    ));
}

#[test]
fn allocator_release_evidence_is_consumed_and_identity_bound_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut dmabuf = dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4)],
    );
    let mut unrelated_dmabuf = dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );

    let evidence = unsafe {
        // SAFETY: This unit test validates identity routing only. It uses the evidence only with a
        // scaffold renderer and expects rejection before any Vulkan operation can occur.
        VulkanAllocatorDmabufForeignReleaseEvidence::new_for_tests(&dmabuf)
    };
    assert!(matches!(
        unsafe {
            // SAFETY: The helper must reject the mismatched dmabuf identity before reaching Vulkan.
            renderer.bind_allocator_released_dmabuf_render_target(&mut unrelated_dmabuf, evidence)
        },
        Err(VulkanError::UnsupportedOperation(
            "allocator dmabuf release evidence"
        ))
    ));

    let evidence = unsafe {
        // SAFETY: This scaffold renderer has no Vulkan device, so a matching evidence token can only
        // route as far as device lookup.
        VulkanAllocatorDmabufForeignReleaseEvidence::new_for_tests(&dmabuf)
    };
    assert!(matches!(
        unsafe {
            // SAFETY: This test validates that matching evidence is consumed by the intended helper
            // and then reaches the normal Vulkan render-target bind path, which fails at device lookup.
            renderer.bind_allocator_released_dmabuf_render_target(&mut dmabuf, evidence)
        },
        Err(VulkanError::VulkanUnavailable)
    ));
}

struct RuntimeDmabufLoopbackCandidate {
    renderer: VulkanRenderer,
    dmabuf: Dmabuf,
    format: Format,
    image: VulkanImage,
    allocator: VulkanAllocator,
}

fn runtime_dmabuf_loopback_candidate(test_name: &str) -> Option<RuntimeDmabufLoopbackCandidate> {
    let instance = match Instance::new(Version::VERSION_1_3, None) {
        Ok(instance) => instance,
        Err(err) => {
            eprintln!("skipping {test_name}: failed to create instance: {err:?}");
            return None;
        }
    };

    let devices = match PhysicalDevice::enumerate(&instance) {
        Ok(devices) => devices,
        Err(err) => {
            eprintln!("skipping {test_name}: failed to enumerate devices: {err:?}");
            return None;
        }
    };

    let usage = ImageUsageFlags::COLOR_ATTACHMENT
        | ImageUsageFlags::SAMPLED
        | ImageUsageFlags::TRANSFER_SRC
        | ImageUsageFlags::TRANSFER_DST;
    let mut found_extension_capable_device = false;
    let mut created_renderer_allocator_pair = false;
    let mut setup_errors = Vec::new();

    for physical_device in devices {
        if !VulkanAllocator::required_extensions(&physical_device)
            .into_iter()
            .all(|extension| physical_device.has_device_extension(extension))
        {
            continue;
        }
        found_extension_capable_device = true;

        let renderer = match VulkanRenderer::builder()
            .with_physical_device(physical_device.clone())
            .build()
        {
            Ok(renderer) => renderer,
            Err(err) => {
                setup_errors.push(format!("{} renderer: {err:?}", physical_device.name()));
                continue;
            }
        };
        let mut allocator = match VulkanAllocator::new(&physical_device, usage) {
            Ok(allocator) => allocator,
            Err(err) => {
                setup_errors.push(format!("{} allocator: {err:?}", physical_device.name()));
                continue;
            }
        };
        created_renderer_allocator_pair = true;

        let candidates = renderer
            .capabilities()
            .formats
            .modifier_records
            .iter()
            .filter(|record| {
                record.plane_count == 1
                    && record.usages.sampled
                    && record.usages.color_attachment
                    && record.usages.color_attachment_blend
                    && get_format_info(record.format)
                        .map(|info| !info.is_10bit)
                        .unwrap_or(false)
            })
            .map(|record| Format {
                code: record.format,
                modifier: record.modifier,
            })
            .collect::<Vec<_>>();

        for format in candidates {
            if !allocator.is_format_supported(format, usage) {
                continue;
            }
            if !renderer
                .capabilities()
                .formats
                .dmabuf_render_target
                .contains(&format)
            {
                continue;
            }

            let image = match allocator.create_buffer_with_usage(4, 4, format.code, &[format.modifier], usage)
            {
                Ok(image) => image,
                Err(err) => {
                    setup_errors.push(format!(
                        "{} {:?} {:?} allocation: {err:?}",
                        physical_device.name(),
                        format.code,
                        format.modifier
                    ));
                    continue;
                }
            };
            let dmabuf = match image.export() {
                Ok(dmabuf) => dmabuf,
                Err(err) => {
                    setup_errors.push(format!(
                        "{} {:?} {:?} export: {err:?}",
                        physical_device.name(),
                        format.code,
                        format.modifier
                    ));
                    continue;
                }
            };

            assert_eq!(dmabuf.format(), format);
            assert!(validate_dmabuf_render_target_metadata(&dmabuf).is_ok());
            assert!(renderer.validate_sampled_dmabuf_import_metadata(&dmabuf).is_ok());
            assert!(!renderer.capabilities().import.dmabuf);
            assert!(
                renderer
                    .capabilities()
                    .formats
                    .dmabuf_import
                    .iter()
                    .next()
                    .is_none()
            );
            assert!(renderer.dmabuf_formats().iter().next().is_none());
            assert!(matches!(
                renderer.validate_sampled_dmabuf_public_advertisement_contract(),
                Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"))
            ));

            return Some(RuntimeDmabufLoopbackCandidate {
                renderer,
                dmabuf,
                format,
                image,
                allocator,
            });
        }
    }

    if !found_extension_capable_device {
        eprintln!("skipping {test_name}: no device supports allocator extensions");
        return None;
    }

    if !created_renderer_allocator_pair {
        panic!(
            "failed to create Vulkan renderer/allocator pair for extension-capable devices: {setup_errors:?}"
        );
    }

    if !setup_errors.is_empty() {
        eprintln!("Vulkan loopback setup/allocation/export errors while searching: {setup_errors:?}");
    }
    eprintln!("skipping {test_name}: no common exportable sampled/render-target modifier");
    None
}

#[test]
#[ignore = "requires a working Vulkan loader, physical device and dmabuf-exportable loopback format"]
fn runtime_dmabuf_loopback_prerequisites_find_common_exportable_modifier() {
    let Some(candidate) = runtime_dmabuf_loopback_candidate("Vulkan dmabuf loopback prerequisite test")
    else {
        return;
    };
    assert_eq!(candidate.dmabuf.size(), Size::from((4, 4)));
    assert_eq!(candidate.dmabuf.format(), candidate.format);
    assert_eq!(candidate.image.size(), candidate.dmabuf.size());
    assert_eq!(candidate.image.format(), candidate.format);
    assert!(candidate.allocator.is_format_supported(
        candidate.format,
        ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::SAMPLED
            | ImageUsageFlags::TRANSFER_SRC
            | ImageUsageFlags::TRANSFER_DST,
    ));
}

#[test]
#[ignore = "requires a working Vulkan loader, physical device and dmabuf-exportable loopback format"]
fn runtime_dmabuf_loopback_imports_released_render_target_as_sampled_texture() {
    let Some(mut candidate) = runtime_dmabuf_loopback_candidate("Vulkan dmabuf loopback import test") else {
        return;
    };

    assert!(candidate.renderer.dmabuf_formats().iter().next().is_none());
    assert!(matches!(
        candidate
            .renderer
            .validate_sampled_dmabuf_public_advertisement_contract(),
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"))
    ));
    let allocator_release = unsafe {
        // SAFETY: The dmabuf was just exported from `candidate.image`, and this ignored runtime test
        // does not hand it to any other API before asking the allocator to release the fresh image to
        // FOREIGN/GENERAL for the renderer acquire below.
        candidate
            .allocator
            .release_dmabuf_to_foreign_general(&candidate.image, &candidate.dmabuf)
    }
    .expect("release allocator dmabuf to foreign GENERAL");
    assert!(matches!(
        unsafe {
            // SAFETY: Same no-intervening-use condition as above; this call must fail on allocator
            // release state before recording a second Vulkan release.
            candidate
                .allocator
                .release_dmabuf_to_foreign_general(&candidate.image, &candidate.dmabuf)
        },
        Err(VulkanAllocatorForeignReleaseError::InvalidState(
            "Vulkan allocator dmabuf foreign release state"
        ))
    ));

    let mut target = unsafe {
        // SAFETY: `allocator_release` proves that the allocator-owned image backing this exported
        // dmabuf was released to VK_QUEUE_FAMILY_FOREIGN_EXT in GENERAL layout. There is no
        // intervening access before this renderer acquire.
        candidate
            .renderer
            .bind_allocator_released_dmabuf_render_target(&mut candidate.dmabuf, allocator_release)
    }
    .expect("bind allocator-released dmabuf as Vulkan render target")
    .expect("renderer should advertise the selected dmabuf render-target modifier");

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((4, 4)))];
        let mut frame = candidate
            .renderer
            .render(&mut target, (4, 4).into(), Transform::Normal)
            .expect("render into loopback dmabuf target");
        frame
            .clear(Color32F::new(0.25, 0.5, 0.75, 1.0), &full_damage)
            .expect("clear loopback dmabuf render target");
    }
    assert_eq!(target.image.layout, VulkanImageLayoutState::ColorAttachment);

    let evidence = candidate
        .renderer
        .release_dmabuf_render_target_for_sampled_loopback(&mut target, false)
        .expect("release loopback render target to foreign GENERAL")
        .expect("released loopback render target should produce sampled import evidence");
    assert_eq!(target.image.layout, VulkanImageLayoutState::Undefined);
    drop(target);
    assert!(evidence.is_for_dmabuf(&candidate.dmabuf));
    assert!(evidence.acquire_sync().is_reached());

    let texture = unsafe {
        // SAFETY: `evidence` was produced by releasing the same Smithay dmabuf identity immediately
        // above, and there is no intervening access, acquire, release, or layout/ownership transition
        // before this sampled loopback import.
        candidate
            .renderer
            .import_dmabuf_texture_from_loopback(&candidate.dmabuf, evidence)
    }
    .expect("import released loopback dmabuf as sampled texture")
    .expect("selected modifier should support sampled dmabuf import");

    assert_eq!(texture.width(), 4);
    assert_eq!(texture.height(), 4);
    assert_eq!(texture.format(), Some(candidate.format.code));
    assert!(texture.has_sampled_image_for_tests());
    assert!(candidate.renderer.dmabuf_formats().iter().next().is_none());
    assert!(matches!(
        candidate
            .renderer
            .validate_sampled_dmabuf_public_advertisement_contract(),
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"))
    ));

    let (released, release_sync) = candidate
        .renderer
        .release_imported_dmabuf_texture_to_foreign_general_sync_point(&texture, false)
        .expect("release sampled loopback texture back to foreign GENERAL");
    assert!(released);
    assert!(release_sync.is_reached());
}

#[test]
fn public_dmabuf_bind_gates_render_target_formats() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut dmabuf = dmabuf_for_tests();
    renderer.capabilities.formats.dmabuf_render_target = [Format {
        code: Fourcc::Abgr8888,
        modifier: Modifier::Linear,
    }]
    .into_iter()
    .collect();

    // This scaffold state intentionally patches only the raw/probed format set. Runtime discovery
    // is responsible for promoting that probe result into the development capability bit; the
    // `Bind<Dmabuf>` format surface must not enable itself from raw formats alone.
    assert!(!renderer.capabilities.rendering.dmabuf_targets);
    assert!(!renderer.capabilities.rendering.dmabuf_target_modifiers);
    assert!(!renderer.capabilities.rendering.dmabuf_target_development);
    assert!(
        renderer
            .capabilities
            .formats
            .dmabuf_render_target
            .iter()
            .any(|format| { format.code == Fourcc::Abgr8888 && format.modifier == Modifier::Linear })
    );
    let formats = <VulkanRenderer as Bind<Dmabuf>>::supported_formats(&renderer)
        .expect("Vulkan dmabuf Bind has an explicit render-target format set");
    assert!(formats.iter().next().is_none());
    let explicit_formats =
        <VulkanRenderer as Bind<VulkanDmabufRenderTarget<'static, 'static>>>::supported_formats(&renderer)
            .expect("Vulkan explicit dmabuf render targets have a gated format set");
    assert!(explicit_formats.iter().next().is_none());
    let owned_explicit_formats =
        <VulkanRenderer as Bind<VulkanOwnedDmabufRenderTarget<'static>>>::supported_formats(&renderer)
            .expect("Vulkan owned explicit dmabuf render targets have a gated format set");
    assert!(owned_explicit_formats.iter().next().is_none());
    assert!(matches!(
        <VulkanRenderer as Bind<Dmabuf>>::bind(&mut renderer, &mut dmabuf),
        Err(VulkanError::NotPublicAdvertised("dmabuf render target"))
    ));
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the explicit development path
        // returns before any Vulkan import or ownership-transfer operation can occur. The important
        // contract here is that the explicit path remains validation-reachable even while its format
        // surface is not advertised.
        unsafe {
            renderer.bind_dmabuf_render_target(&mut dmabuf, VulkanDmabufRenderTargetAcquire::discard())
        },
        Err(VulkanError::VulkanUnavailable)
    ));
    {
        let mut explicit_target = unsafe {
            // SAFETY: This scaffold renderer has no Vulkan device, so binding the wrapper returns
            // before any Vulkan import or ownership-transfer operation can occur.
            VulkanDmabufRenderTarget::discard(&mut dmabuf)
        };
        assert!(matches!(
            <VulkanRenderer as Bind<VulkanDmabufRenderTarget<'_, '_>>>::bind(
                &mut renderer,
                &mut explicit_target,
            ),
            Err(VulkanError::VulkanUnavailable)
        ));
    }
    let mut owned_explicit_target = unsafe {
        // SAFETY: This scaffold renderer has no Vulkan device, so binding the wrapper returns before
        // any Vulkan import or ownership-transfer operation can occur.
        VulkanOwnedDmabufRenderTarget::discard(dmabuf.clone())
    };
    assert!(matches!(
        <VulkanRenderer as Bind<VulkanOwnedDmabufRenderTarget<'_>>>::bind(
            &mut renderer,
            &mut owned_explicit_target,
        ),
        Err(VulkanError::VulkanUnavailable)
    ));

    renderer.capabilities.rendering.dmabuf_target_development = true;
    let formats = <VulkanRenderer as Bind<Dmabuf>>::supported_formats(&renderer)
        .expect("Vulkan dmabuf Bind has a gated render-target format set");
    assert!(
        formats
            .iter()
            .any(|format| { format.code == Fourcc::Abgr8888 && format.modifier == Modifier::Linear })
    );
    let explicit_formats =
        <VulkanRenderer as Bind<VulkanDmabufRenderTarget<'static, 'static>>>::supported_formats(&renderer)
            .expect("Vulkan explicit dmabuf render targets have a gated format set");
    assert!(
        explicit_formats
            .iter()
            .any(|format| { format.code == Fourcc::Abgr8888 && format.modifier == Modifier::Linear })
    );
    let owned_explicit_formats =
        <VulkanRenderer as Bind<VulkanOwnedDmabufRenderTarget<'static>>>::supported_formats(&renderer)
            .expect("Vulkan owned explicit dmabuf render targets have a gated format set");
    assert!(
        owned_explicit_formats
            .iter()
            .any(|format| { format.code == Fourcc::Abgr8888 && format.modifier == Modifier::Linear })
    );
}

#[test]
fn public_dmabuf_bind_validates_metadata_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    renderer.capabilities.rendering.dmabuf_target_development = true;
    let mut zero_width = dmabuf_with_planes_for_tests(
        (0, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4)],
    );
    let mut multi_plane = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4), (1, 4, 4)],
    );

    assert!(matches!(
        <VulkanRenderer as Bind<Dmabuf>>::bind(&mut renderer, &mut zero_width),
        Err(VulkanError::UnsupportedOperation("dmabuf size"))
    ));
    assert!(matches!(
        <VulkanRenderer as Bind<Dmabuf>>::bind(&mut renderer, &mut multi_plane),
        Err(VulkanError::UnsupportedOperation("dmabuf render target planes"))
    ));
    assert!(matches!(
        // SAFETY: Invalid metadata is rejected before any Vulkan import or ownership-transfer
        // operation can occur.
        unsafe {
            renderer.bind_dmabuf_render_target(&mut zero_width, VulkanDmabufRenderTargetAcquire::discard())
        },
        Err(VulkanError::UnsupportedOperation("dmabuf size"))
    ));
    assert!(matches!(
        // SAFETY: Invalid metadata is rejected before any Vulkan import or ownership-transfer
        // operation can occur.
        unsafe {
            renderer.bind_dmabuf_render_target(&mut multi_plane, VulkanDmabufRenderTargetAcquire::discard())
        },
        Err(VulkanError::UnsupportedOperation("dmabuf render target planes"))
    ));
    assert!(matches!(
        // SAFETY: Invalid metadata is rejected before any Vulkan import or ownership-transfer
        // operation can occur.
        unsafe { renderer.create_acquired_dmabuf_render_target(&zero_width, false, None) },
        Err(VulkanError::UnsupportedOperation("dmabuf size"))
    ));
    assert!(matches!(
        // SAFETY: Invalid metadata is rejected before any Vulkan import or ownership-transfer
        // operation can occur.
        unsafe { renderer.create_acquired_dmabuf_render_target(&multi_plane, false, None) },
        Err(VulkanError::UnsupportedOperation("dmabuf render target planes"))
    ));
    assert!(matches!(
        // SAFETY: Invalid metadata is rejected before any sync-point wait, Vulkan import, or
        // ownership-transfer operation can occur.
        unsafe {
            renderer.create_acquired_dmabuf_render_target_with_sync_point(
                &zero_width,
                false,
                Some(&SyncPoint::from(InterruptedFence)),
            )
        },
        Err(VulkanError::UnsupportedOperation("dmabuf size"))
    ));
    assert!(matches!(
        // SAFETY: Invalid metadata is rejected before any sync-point wait, Vulkan import, or
        // ownership-transfer operation can occur.
        unsafe {
            renderer.create_acquired_dmabuf_render_target_with_sync_point(
                &multi_plane,
                false,
                Some(&SyncPoint::from(InterruptedFence)),
            )
        },
        Err(VulkanError::UnsupportedOperation("dmabuf render target planes"))
    ));
}

#[test]
fn internal_dmabuf_render_target_release_rejects_preconditions_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut foreign_target = render_target_for_tests(
        ContextId::new(),
        VulkanImageSource::RenderTarget,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );
    let mut wrong_source = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::Offscreen,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );
    let mut missing_image = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::RenderTarget,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );
    let mut released_target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::RenderTarget,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );
    released_target.image.sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();

    assert!(matches!(
        renderer.release_acquired_dmabuf_render_target_to_foreign_general(&mut foreign_target, false),
        Err(VulkanError::UnsupportedOperation("foreign dmabuf render target"))
    ));
    assert!(matches!(
        renderer.release_acquired_dmabuf_render_target_to_foreign_general(&mut wrong_source, false),
        Err(VulkanError::UnsupportedOperation("dmabuf render target"))
    ));
    assert!(matches!(
        renderer.release_acquired_dmabuf_render_target_to_foreign_general(&mut missing_image, false),
        Err(VulkanError::UnsupportedOperation("dmabuf render target image"))
    ));
    assert!(matches!(
        renderer.release_acquired_dmabuf_render_target_to_foreign_general(&mut released_target, false),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    assert!(matches!(
        renderer
            .release_acquired_dmabuf_render_target_to_foreign_general_sync_point(&mut missing_image, true),
        Err(VulkanError::UnsupportedOperation("dmabuf render target image"))
    ));
    assert!(matches!(
        renderer.release_dmabuf_render_target_after_render_error(&mut missing_image),
        Err(VulkanError::UnsupportedOperation("dmabuf render target image"))
    ));
    assert!(matches!(
        <VulkanRenderer as RenderTargetLifecycle<VulkanDmabufRenderTarget<'static, 'static>>>::release_after_render_error(
            &mut renderer,
            &mut missing_image,
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf render target image"))
    ));
}

#[test]
fn public_dmabuf_import_gates_formats() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let dmabuf = dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4)],
    );
    let format = Format {
        code: Fourcc::Abgr8888,
        modifier: Modifier::Linear,
    };
    renderer.capabilities.formats.dmabuf_import = [format].into_iter().collect();
    renderer.capabilities.formats.modifier_records = vec![modifier_record_from_properties(
        Fourcc::Abgr8888,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        },
    )];

    assert!(!renderer.capabilities.import.dmabuf);
    assert!(
        renderer
            .capabilities
            .formats
            .dmabuf_import
            .iter()
            .any(|candidate| *candidate == format)
    );
    assert!(renderer.dmabuf_formats().iter().next().is_none());
    assert!(!renderer.has_dmabuf_format(format));

    renderer.capabilities.import.dmabuf = true;
    assert!(
        renderer
            .capabilities
            .formats
            .dmabuf_import
            .iter()
            .next()
            .is_some()
    );
    assert!(renderer.dmabuf_formats().iter().next().is_none());
    assert!(!renderer.has_dmabuf_format(format));

    assert!(matches!(
        renderer.import_dmabuf(&dmabuf, None),
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"))
    ));
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the explicit validation-stage
        // helper returns before any Vulkan import, fd import, or ownership-transfer operation can
        // occur. The important contract here is that known-layout sampled dmabuf import remains
        // reachable without advertising generic ImportDma support.
        unsafe { renderer.import_dmabuf_texture_with_known_general_layout(&dmabuf, None) },
        Err(VulkanError::VulkanUnavailable)
    ));
    let known_layout_evidence = unsafe {
        // SAFETY: These unit tests only validate contract routing before device lookup; they perform
        // no Vulkan import, acquire, or sampling operation with the constructed evidence.
        SampledDmabufKnownLayoutEvidence::foreign_general(dmabuf.weak())
    };
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the helper returns before any
        // Vulkan import or ownership-transfer operation can occur.
        unsafe {
            renderer.create_imported_dmabuf_texture_with_known_general_layout(
                &dmabuf,
                known_layout_evidence.clone(),
                None,
            )
        },
        Err(VulkanError::VulkanUnavailable)
    ));
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the helper returns before any
        // Vulkan import, fd import, or ownership-transfer operation can occur.
        unsafe {
            renderer.create_imported_dmabuf_texture_with_known_general_layout_and_sync_point(
                &dmabuf,
                known_layout_evidence.clone(),
                Some(&SyncPoint::signaled()),
            )
        },
        Err(VulkanError::VulkanUnavailable)
    ));
    let release_ownership_called = Cell::new(false);
    let release_ownership = || {
        release_ownership_called.set(true);
        panic!("release ownership must not be consumed before device lookup succeeds")
    };
    assert!(matches!(
        // SAFETY: This scaffold renderer has no Vulkan device, so the helper returns before any
        // Vulkan import, fd import, ownership-transfer, or release signaling operation can occur.
        unsafe {
            renderer.create_imported_dmabuf_texture_with_known_general_layout_release_and_sync_point(
                &dmabuf,
                known_layout_evidence,
                Some(&SyncPoint::from(SignaledExportableFence)),
                release_ownership,
            )
        },
        Err(VulkanError::VulkanUnavailable)
    ));
    assert!(!release_ownership_called.get());
}

#[test]
fn sampled_dmabuf_wayland_policy_does_not_public_advertise_import_dma() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let format = Format {
        code: Fourcc::Abgr8888,
        modifier: Modifier::Linear,
    };
    let dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    renderer.capabilities.import.dmabuf = true;
    renderer.capabilities.external_memory.foreign_queue_family = true;
    renderer.capabilities.formats.dmabuf_import = [format].into_iter().collect();
    renderer.capabilities.formats.modifier_records = vec![modifier_record_from_properties(
        Fourcc::Abgr8888,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        },
    )];

    let import = VulkanDmabufImportState::from_dmabuf(&dmabuf).unwrap();
    assert!(renderer.validate_sampled_dmabuf_import_metadata(&dmabuf).is_ok());

    let acquire_sync = SyncPoint::from(SignaledExportableFence);
    let acquire_evidence = SampledDmabufAcquireSyncEvidence::new(&dmabuf, acquire_sync);
    let release_evidence = renderer
        .validate_sampled_dmabuf_wayland_release_point_contract(&dmabuf, true)
        .unwrap();
    let external_state = SampledDmabufWaylandForeignGeneralEvidence::new_for_tests(&dmabuf);
    let external_state_sources = renderer
        .sampled_dmabuf_wayland_external_state_evidence_sources(
            &dmabuf,
            SampledDmabufWaylandLayoutHistory::NoRendererHistory,
            Some(&external_state),
        )
        .unwrap();
    let policy_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &dmabuf,
        &import,
        &acquire_evidence,
        &release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    )
    .with_external_state_sources(external_state_sources)
    .with_texture_cache_replacement_release_reachability(
        SampledDmabufWaylandTextureCacheReplacementReleaseReachability::new_for_tests(&dmabuf),
    )
    .with_texture_cache_release_hook(SampledDmabufWaylandTextureCacheReleaseHook::new_for_tests(
        &dmabuf,
    ))
    .with_texture_cache_release_lifecycle(Some(
        SampledDmabufWaylandTextureCacheReleaseLifecycle::new_for_tests(&dmabuf),
    ))
    .with_release_ownership(SampledDmabufReleaseOwnershipEvidence::new_for_tests(&dmabuf));

    let wayland_policy = renderer
        .validate_sampled_dmabuf_wayland_vulkan_interop_policy(&policy_context)
        .unwrap();
    let known_layout = renderer
        .validate_sampled_dmabuf_known_layout_contract(&dmabuf, wayland_policy)
        .unwrap();
    assert!(known_layout.is_for_dmabuf(&dmabuf));
    assert_eq!(
        renderer.sampled_dmabuf_public_import_contracts(),
        SampledDmabufPublicImportContracts {
            raw_import_capability: true,
            advertised_formats: true,
            public_external_state_policy: false,
            public_import_lifecycle: false,
        }
    );

    assert!(matches!(
        renderer.validate_sampled_dmabuf_public_advertisement_contract(),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf public external-state policy"
        ))
    ));
    assert!(renderer.dmabuf_formats().iter().next().is_none());
    assert!(!renderer.has_dmabuf_format(format));
    assert!(matches!(
        renderer.import_dmabuf(&dmabuf, None),
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"))
    ));
}

#[test]
fn sampled_dmabuf_import_validation_guards_metadata_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let valid_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );

    assert!(matches!(
        renderer.validate_sampled_dmabuf_import_metadata(&valid_dmabuf),
        Err(VulkanError::MissingCapability("sampled dmabuf format/modifier"))
    ));

    renderer.capabilities.formats.modifier_records = vec![modifier_record_from_properties(
        Fourcc::Abgr8888,
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: Modifier::Linear.into(),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::SAMPLED_IMAGE,
        },
    )];
    assert!(
        renderer
            .validate_sampled_dmabuf_import_metadata(&valid_dmabuf)
            .is_ok()
    );

    let zero_width = dmabuf_with_planes_for_tests(
        (0, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let multi_plane = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4), (1, 12, 4)],
    );
    let implicit_modifier = dmabuf_for_tests();
    let unsupported_format = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Yuyv,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 8)],
    );

    assert!(matches!(
        renderer.validate_sampled_dmabuf_import_metadata(&zero_width),
        Err(VulkanError::UnsupportedOperation("dmabuf size"))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_import_metadata(&multi_plane),
        Err(VulkanError::UnsupportedOperation("sampled dmabuf planes"))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_import_metadata(&implicit_modifier),
        Err(VulkanError::MissingCapability("sampled dmabuf explicit modifier"))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_import_metadata(&unsupported_format),
        Err(VulkanError::UnsupportedFormat(Fourcc::Yuyv))
    ));
}

#[test]
fn sampled_dmabuf_import_contract_scaffold_marks_remaining_steps() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let advertised_format = Format {
        code: Fourcc::Abgr8888,
        modifier: Modifier::Linear,
    };

    assert_eq!(
        renderer.sampled_dmabuf_public_import_contracts(),
        SampledDmabufPublicImportContracts {
            raw_import_capability: false,
            advertised_formats: false,
            public_external_state_policy: false,
            public_import_lifecycle: false,
        }
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_public_advertisement_contract(),
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"))
    ));
    renderer.capabilities.import.dmabuf = true;
    assert_eq!(
        renderer.sampled_dmabuf_public_import_contracts(),
        SampledDmabufPublicImportContracts {
            raw_import_capability: true,
            advertised_formats: false,
            public_external_state_policy: false,
            public_import_lifecycle: false,
        }
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_public_advertisement_contract(),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf advertised formats"
        ))
    ));
    renderer.capabilities.formats.dmabuf_import = [advertised_format].into_iter().collect();
    assert_eq!(
        renderer.sampled_dmabuf_public_import_contracts(),
        SampledDmabufPublicImportContracts {
            raw_import_capability: true,
            advertised_formats: true,
            public_external_state_policy: false,
            public_import_lifecycle: false,
        }
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_public_advertisement_contract(),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf public external-state policy"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_public_external_state_contract(),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf public external-state policy"
        ))
    ));
    assert!(matches!(
        SampledDmabufPublicImportContracts {
            raw_import_capability: true,
            advertised_formats: true,
            public_external_state_policy: true,
            public_import_lifecycle: false,
        }
        .validate(),
        Err(VulkanError::MissingCapability("sampled dmabuf import lifecycle"))
    ));
    assert!(
        SampledDmabufPublicImportContracts {
            raw_import_capability: true,
            advertised_formats: true,
            public_external_state_policy: true,
            public_import_lifecycle: true,
        }
        .validate()
        .is_ok()
    );

    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_acquire_sync_contract(None),
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf implicit sync"))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_acquire_sync_contract(Some(&SyncPoint::signaled())),
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf implicit sync"))
    ));
    let explicit_acquire = SyncPoint::from(SignaledExportableFence);
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_acquire_sync_contract(Some(&explicit_acquire))
            .is_ok()
    );
    let policy_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    let policy_acquire_evidence =
        SampledDmabufAcquireSyncEvidence::new(&policy_dmabuf, explicit_acquire.clone());
    let policy_import = VulkanDmabufImportState::from_dmabuf(&policy_dmabuf).unwrap();
    let policy_release_evidence = renderer
        .validate_sampled_dmabuf_wayland_release_point_contract(&policy_dmabuf, true)
        .unwrap();
    let mut history_renderer = VulkanRenderer::new_scaffold_for_tests();
    assert_eq!(
        history_renderer.sampled_dmabuf_layout_history(&policy_dmabuf),
        SampledDmabufWaylandLayoutHistory::NoRendererHistory
    );
    history_renderer.record_sampled_dmabuf_released_to_foreign_general(&policy_dmabuf);
    assert_eq!(
        history_renderer.sampled_dmabuf_layout_history(&policy_dmabuf),
        SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral
    );
    assert_eq!(
        history_renderer.sampled_dmabuf_layout_history(&policy_dmabuf.clone()),
        SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral
    );
    let unrelated_dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 16)],
    );
    assert_eq!(
        history_renderer.sampled_dmabuf_layout_history(&unrelated_dmabuf),
        SampledDmabufWaylandLayoutHistory::NoRendererHistory
    );
    let unrelated_release_evidence = renderer
        .validate_sampled_dmabuf_wayland_release_point_contract(&unrelated_dmabuf, true)
        .unwrap();
    let mismatched_release_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &unrelated_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_release_sync_policy(&mismatched_release_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf release evidence identity"
        ))
    ));
    let unrelated_acquire_evidence =
        SampledDmabufAcquireSyncEvidence::new(&unrelated_dmabuf, explicit_acquire.clone());
    let mismatched_acquire_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &unrelated_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_acquire_sync_policy(&mismatched_acquire_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf acquire sync identity"
        ))
    ));
    history_renderer.record_sampled_dmabuf_locally_acquired(&policy_dmabuf);
    assert_eq!(
        history_renderer.sampled_dmabuf_layout_history(&policy_dmabuf),
        SampledDmabufWaylandLayoutHistory::LocallyAcquired
    );
    history_renderer.record_sampled_dmabuf_released_to_foreign_general(&policy_dmabuf);
    assert_eq!(
        history_renderer.sampled_dmabuf_layout_history(&policy_dmabuf),
        SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral
    );
    {
        let temporary_dmabuf = dmabuf_with_planes_for_tests(
            (4, 3).into(),
            Fourcc::Abgr8888,
            Modifier::Linear,
            DmabufFlags::empty(),
            &[(0, 0, 16)],
        );
        history_renderer.record_sampled_dmabuf_released_to_foreign_general(&temporary_dmabuf);
        assert_eq!(history_renderer.sampled_dmabuf_layout_history.len(), 2);
    }
    history_renderer.prune_sampled_dmabuf_layout_history();
    assert_eq!(history_renderer.sampled_dmabuf_layout_history.len(), 1);
    let policy_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_context_metadata(&policy_context)
            .is_ok()
    );
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_current_reacquire_layout_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::NoRendererHistory,
            None,
        ),
        Ok(None)
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_current_reacquire_foreign_general_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::NoRendererHistory,
            None,
        ),
        Ok(None)
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_first_import_layout_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::NoRendererHistory,
            None,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout policy"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_external_state_evidence_sources(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::NoRendererHistory,
            None,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout policy"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_first_import_foreign_general_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::NoRendererHistory,
            None,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan foreign GENERAL policy"
        ))
    ));
    let wayland_external_state = SampledDmabufWaylandForeignGeneralEvidence::new_for_tests(&policy_dmabuf);
    let first_import_sources = renderer
        .sampled_dmabuf_wayland_external_state_evidence_sources(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::NoRendererHistory,
            Some(&wayland_external_state),
        )
        .unwrap();
    assert!(first_import_sources.first_import_layout.is_some());
    assert!(first_import_sources.first_import_foreign_general.is_some());
    assert!(first_import_sources.current_reacquire_layout.is_none());
    assert!(first_import_sources.current_reacquire_foreign_general.is_none());
    let mut external_state_lifecycle_renderer = VulkanRenderer::new_scaffold_for_tests();
    external_state_lifecycle_renderer
        .capabilities
        .external_memory
        .foreign_queue_family = true;
    let external_state_lifecycle_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    )
    .with_external_state_sources(first_import_sources.clone())
    .with_texture_cache_replacement_release_reachability(
        SampledDmabufWaylandTextureCacheReplacementReleaseReachability::new_for_tests(&policy_dmabuf),
    )
    .with_texture_cache_release_hook(SampledDmabufWaylandTextureCacheReleaseHook::new_for_tests(
        &policy_dmabuf,
    ))
    .with_release_ownership(SampledDmabufReleaseOwnershipEvidence::new_for_tests(
        &policy_dmabuf,
    ));
    assert!(matches!(
        external_state_lifecycle_renderer
            .validate_sampled_dmabuf_wayland_vulkan_interop_policy(&external_state_lifecycle_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan texture-cache release call sites"
        ))
    ));
    let external_state_complete_lifecycle_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    )
    .with_external_state_sources(first_import_sources.clone())
    .with_texture_cache_replacement_release_reachability(
        SampledDmabufWaylandTextureCacheReplacementReleaseReachability::new_for_tests(&policy_dmabuf),
    )
    .with_texture_cache_release_hook(SampledDmabufWaylandTextureCacheReleaseHook::new_for_tests(
        &policy_dmabuf,
    ))
    .with_texture_cache_release_lifecycle(Some(
        SampledDmabufWaylandTextureCacheReleaseLifecycle::new_for_tests(&policy_dmabuf),
    ))
    .with_release_ownership(SampledDmabufReleaseOwnershipEvidence::new_for_tests(
        &policy_dmabuf,
    ));
    assert!(
        external_state_lifecycle_renderer
            .validate_sampled_dmabuf_wayland_vulkan_interop_policy(&external_state_complete_lifecycle_context)
            .is_ok()
    );
    let mismatched_wayland_external_state =
        SampledDmabufWaylandForeignGeneralEvidence::new_for_tests(&unrelated_dmabuf);
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_external_state_evidence_sources(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::NoRendererHistory,
            Some(&mismatched_wayland_external_state),
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland external-state identity"
        ))
    ));
    let user_data_external_state = UserDataMap::new();
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_user_data_foreign_general_evidence(
            &user_data_external_state,
            &policy_dmabuf,
        ),
        Ok(None)
    ));
    unsafe {
        // SAFETY: This unit test only validates evidence storage and identity routing; it performs
        // no Vulkan import, acquire, sampling, or release operation with the constructed evidence.
        VulkanRenderer::mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
            &user_data_external_state,
            &policy_dmabuf,
        )
        .unwrap();
    }
    let stored_external_state = renderer
        .sampled_dmabuf_wayland_user_data_foreign_general_evidence(&user_data_external_state, &policy_dmabuf)
        .unwrap()
        .unwrap();
    assert!(stored_external_state.is_for_dmabuf(&policy_dmabuf));
    assert!(!stored_external_state.is_for_dmabuf(&unrelated_dmabuf));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_user_data_foreign_general_evidence(
            &user_data_external_state,
            &unrelated_dmabuf,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland external-state identity"
        ))
    ));
    let user_data_lifecycle = UserDataMap::new();
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_user_data_texture_cache_release_lifecycle(
            &user_data_lifecycle,
            &policy_dmabuf,
        ),
        Ok(None)
    ));
    unsafe {
        // SAFETY: This unit test validates only marker storage and identity checks. It does not
        // import, sample, release, reset, destroy, or otherwise use a real Wayland surface cache.
        renderer
            .mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
                &user_data_lifecycle,
                &policy_dmabuf,
            )
            .unwrap();
    }
    let stored_lifecycle = renderer
        .sampled_dmabuf_wayland_user_data_texture_cache_release_lifecycle(
            &user_data_lifecycle,
            &policy_dmabuf,
        )
        .unwrap()
        .unwrap();
    assert!(stored_lifecycle.is_for_dmabuf(&policy_dmabuf));
    assert!(!stored_lifecycle.is_for_dmabuf(&unrelated_dmabuf));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_user_data_texture_cache_release_lifecycle(
            &user_data_lifecycle,
            &unrelated_dmabuf,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland texture-cache release lifecycle identity"
        ))
    ));
    let unrelated_renderer = VulkanRenderer::new_scaffold_for_tests();
    assert!(matches!(
        unrelated_renderer.sampled_dmabuf_wayland_user_data_texture_cache_release_lifecycle(
            &user_data_lifecycle,
            &policy_dmabuf,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland texture-cache release lifecycle renderer identity"
        ))
    ));
    let mismatched_import_dmabuf = dmabuf_with_planes_for_tests(
        (8, 3).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 32)],
    );
    let mismatched_import = VulkanDmabufImportState::from_dmabuf(&mismatched_import_dmabuf).unwrap();
    let mismatched_import_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &mismatched_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_context_metadata(&mismatched_import_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland import metadata"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&mismatched_import_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland import metadata"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_vulkan_interop_policy_contracts(&mismatched_import_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland import metadata"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_known_layout_contract(
            &policy_dmabuf,
            SampledDmabufLayoutEvidence::WaylandDmabuf,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf known-layout contract"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout policy"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_vulkan_interop_policy_contracts(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout policy"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_first_import_layout_policy(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout policy"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_reacquire_layout_policy(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan reacquire layout history"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_layout_policy(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout policy"
        ))
    ));
    let mut first_import_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    first_import_context.first_import_layout = Some(
        SampledDmabufWaylandFirstImportLayoutEvidence::new_for_tests(&unrelated_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_first_import_layout_policy(&first_import_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland first-import identity"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_layout_policy(&first_import_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland first-import identity"
        ))
    ));
    first_import_context.first_import_layout = Some(
        SampledDmabufWaylandFirstImportLayoutEvidence::new_for_tests(&policy_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_first_import_layout_policy(&first_import_context),
        Ok(policy) if policy.is_for_dmabuf(&policy_dmabuf)
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_layout_policy(&first_import_context),
        Ok(SampledDmabufWaylandLayoutPolicy::FirstImport(_))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_foreign_general_policy(&first_import_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan foreign GENERAL policy"
        ))
    ));
    first_import_context.first_import_foreign_general = Some(unsafe {
        // SAFETY: This unit test only validates contract routing; it performs no Vulkan import,
        // acquire, or sampling operation with the constructed evidence.
        SampledDmabufKnownLayoutEvidence::foreign_general(unrelated_dmabuf.weak())
    });
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_foreign_general_policy(&first_import_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland foreign GENERAL identity"
        ))
    ));
    first_import_context.first_import_foreign_general = Some(unsafe {
        // SAFETY: This unit test only validates contract routing; it performs no Vulkan import,
        // acquire, or sampling operation with the constructed evidence.
        SampledDmabufKnownLayoutEvidence::foreign_general(policy_dmabuf.weak())
    });
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_foreign_general_policy(&first_import_context)
            .is_ok()
    );
    first_import_context.release_ownership = Some(SampledDmabufReleaseOwnershipEvidence::new_for_tests(
        &policy_dmabuf,
    ));
    first_import_context.texture_cache_replacement_release_reachability =
        Some(SampledDmabufWaylandTextureCacheReplacementReleaseReachability::new_for_tests(&policy_dmabuf));
    first_import_context.texture_cache_release_hook = Some(
        SampledDmabufWaylandTextureCacheReleaseHook::new_for_tests(&policy_dmabuf),
    );
    first_import_context.texture_cache_release_lifecycle = Some(
        SampledDmabufWaylandTextureCacheReleaseLifecycle::new_for_tests(&policy_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&first_import_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan queue-family capability"
        ))
    ));
    let renderer_release_history_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_current_reacquire_layout_policy(
            &renderer_release_history_context
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan current reacquire layout policy"
        ))
    ));
    assert!(matches!(
        renderer
            .validate_sampled_dmabuf_wayland_first_import_layout_policy(&renderer_release_history_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout history"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_reacquire_layout_policy(&renderer_release_history_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan current reacquire layout policy"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_layout_policy(&renderer_release_history_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan current reacquire layout policy"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_current_reacquire_layout_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
            None,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan current reacquire layout policy"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_current_reacquire_foreign_general_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
            None,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan current reacquire layout policy"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_external_state_evidence_sources(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
            None,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan current reacquire layout policy"
        ))
    ));
    let current_reacquire_sources = renderer
        .sampled_dmabuf_wayland_external_state_evidence_sources(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
            Some(&wayland_external_state),
        )
        .unwrap();
    assert!(current_reacquire_sources.first_import_layout.is_none());
    assert!(current_reacquire_sources.first_import_foreign_general.is_none());
    assert!(current_reacquire_sources.current_reacquire_layout.is_some());
    assert!(
        current_reacquire_sources
            .current_reacquire_foreign_general
            .is_some()
    );
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_first_import_layout_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
            None,
        ),
        Ok(None)
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_first_import_foreign_general_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
            None,
        ),
        Ok(None)
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_foreign_general_policy(&renderer_release_history_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan current reacquire layout policy"
        ))
    ));
    let local_acquire_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::LocallyAcquired,
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_layout_policy(&local_acquire_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan unreleased local acquire"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_first_import_layout_policy(&local_acquire_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan unreleased local acquire"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_current_reacquire_layout_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::LocallyAcquired,
            None,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan unreleased local acquire"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_current_reacquire_foreign_general_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::LocallyAcquired,
            None,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan unreleased local acquire"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_external_state_evidence_sources(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::LocallyAcquired,
            None,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan unreleased local acquire"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_first_import_layout_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::LocallyAcquired,
            None,
        ),
        Ok(None)
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_first_import_foreign_general_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::LocallyAcquired,
            None,
        ),
        Ok(None)
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_foreign_general_policy(&local_acquire_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan unreleased local acquire"
        ))
    ));
    let mut current_reacquire_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
    );
    current_reacquire_context.current_reacquire_layout = Some(
        SampledDmabufWaylandCurrentReacquireLayoutEvidence::new_for_tests(&unrelated_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_current_reacquire_layout_policy(&current_reacquire_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland current reacquire identity"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_layout_policy(&current_reacquire_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland current reacquire identity"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&current_reacquire_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland current reacquire identity"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_foreign_general_policy(&current_reacquire_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland current reacquire identity"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_current_reacquire_foreign_general_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
            current_reacquire_context.current_reacquire_layout.as_ref(),
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland current reacquire identity"
        ))
    ));
    current_reacquire_context.current_reacquire_layout = Some(
        SampledDmabufWaylandCurrentReacquireLayoutEvidence::new_for_tests(&policy_dmabuf),
    );
    current_reacquire_context.current_reacquire_foreign_general = renderer
        .sampled_dmabuf_wayland_current_reacquire_foreign_general_evidence(
            &policy_dmabuf,
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
            current_reacquire_context.current_reacquire_layout.as_ref(),
        )
        .unwrap();
    current_reacquire_context.release_ownership = Some(SampledDmabufReleaseOwnershipEvidence::new_for_tests(
        &policy_dmabuf,
    ));
    current_reacquire_context.texture_cache_replacement_release_reachability =
        Some(SampledDmabufWaylandTextureCacheReplacementReleaseReachability::new_for_tests(&policy_dmabuf));
    current_reacquire_context.texture_cache_release_hook = Some(
        SampledDmabufWaylandTextureCacheReleaseHook::new_for_tests(&policy_dmabuf),
    );
    current_reacquire_context.texture_cache_release_lifecycle = Some(
        SampledDmabufWaylandTextureCacheReleaseLifecycle::new_for_tests(&policy_dmabuf),
    );
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_current_reacquire_layout_policy(&current_reacquire_context)
            .is_ok()
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_layout_policy(&current_reacquire_context),
        Ok(SampledDmabufWaylandLayoutPolicy::Reacquire(_))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&current_reacquire_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan queue-family capability"
        ))
    ));
    assert!(matches!(
        renderer.sampled_dmabuf_wayland_vulkan_interop_policy_contracts(&current_reacquire_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan queue-family capability"
        ))
    ));
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_foreign_general_policy(&current_reacquire_context)
            .is_ok()
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_foreign_general_policy(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout policy"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_queue_family_policy(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan queue-family capability"
        ))
    ));
    renderer.capabilities.external_memory.foreign_queue_family = true;
    let validated_queue_policy = renderer
        .validate_sampled_dmabuf_wayland_queue_family_policy(&policy_context)
        .unwrap();
    assert!(validated_queue_policy.is_for_dmabuf(&policy_dmabuf));
    assert!(!validated_queue_policy.is_for_dmabuf(&unrelated_dmabuf));
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_queue_family_policy(&policy_context)
            .is_ok()
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&current_reacquire_context),
        Ok(SampledDmabufLayoutEvidence::SmithayWaylandVulkanPolicy(_))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&first_import_context),
        Ok(SampledDmabufLayoutEvidence::SmithayWaylandVulkanPolicy(_))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&renderer_release_history_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan current reacquire layout policy"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan first-import layout policy"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_public_advertisement_contract(),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf public external-state policy"
        ))
    ));
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_acquire_sync_policy(&policy_context)
            .is_ok()
    );
    let validated_acquire_policy = renderer
        .validate_sampled_dmabuf_wayland_acquire_sync_policy(&policy_context)
        .unwrap();
    assert!(validated_acquire_policy.is_for_dmabuf(&policy_dmabuf));
    assert!(!validated_acquire_policy.is_for_dmabuf(&unrelated_dmabuf));
    let signaled_acquire = SyncPoint::signaled();
    let signaled_acquire_evidence = SampledDmabufAcquireSyncEvidence::new(&policy_dmabuf, signaled_acquire);
    let implicit_policy_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &signaled_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_acquire_sync_policy(&implicit_policy_context),
        Err(VulkanError::NotPublicAdvertised("sampled dmabuf implicit sync"))
    ));
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_release_sync_policy(&policy_context)
            .is_ok()
    );
    let validated_release_policy = renderer
        .validate_sampled_dmabuf_wayland_release_sync_policy(&policy_context)
        .unwrap();
    assert!(validated_release_policy.is_for_dmabuf(&policy_dmabuf));
    assert!(!validated_release_policy.is_for_dmabuf(&unrelated_dmabuf));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_release_ownership_policy(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland release ownership transfer"
        ))
    ));
    assert!(matches!(
        renderer
            .validate_sampled_dmabuf_wayland_release_ownership_availability_contract(&policy_dmabuf, false),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland release ownership transfer"
        ))
    ));
    let production_release_ownership_evidence = renderer
        .validate_sampled_dmabuf_wayland_release_ownership_availability_contract(&policy_dmabuf, true)
        .unwrap();
    assert!(production_release_ownership_evidence.is_for_dmabuf(&policy_dmabuf));
    assert!(!production_release_ownership_evidence.is_for_dmabuf(&unrelated_dmabuf));
    let mut mismatched_release_ownership_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    mismatched_release_ownership_context.release_ownership = Some(
        SampledDmabufReleaseOwnershipEvidence::new_for_tests(&unrelated_dmabuf),
    );
    assert!(matches!(
        renderer
            .validate_sampled_dmabuf_wayland_release_ownership_policy(&mismatched_release_ownership_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland release ownership identity"
        ))
    ));
    let mut release_ownership_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    release_ownership_context.release_ownership = Some(production_release_ownership_evidence);
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_release_ownership_policy(&release_ownership_context)
            .is_ok()
    );
    let stale_cache_policy_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        false,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_texture_cache_policy(&stale_cache_policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan texture-cache policy"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_texture_cache_replacement_reachability_contract(
            &policy_dmabuf,
            false,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan import_surface post-retired-release call site"
        ))
    ));
    let replacement_release_reachability = renderer
        .validate_sampled_dmabuf_wayland_texture_cache_replacement_reachability_contract(&policy_dmabuf, true)
        .unwrap();
    assert!(replacement_release_reachability.is_for_dmabuf(&policy_dmabuf));
    assert!(!replacement_release_reachability.is_for_dmabuf(&unrelated_dmabuf));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_texture_cache_policy(&policy_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan import_surface post-retired-release call site"
        ))
    ));
    let mut mismatched_cache_replacement_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    mismatched_cache_replacement_context.texture_cache_replacement_release_reachability = Some(
        SampledDmabufWaylandTextureCacheReplacementReleaseReachability::new_for_tests(&unrelated_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_texture_cache_policy(&mismatched_cache_replacement_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland import_surface post-retired-release call site identity"
        ))
    ));
    let mut missing_cache_hook_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    missing_cache_hook_context.texture_cache_replacement_release_reachability =
        Some(replacement_release_reachability.clone());
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_texture_cache_policy(&missing_cache_hook_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan texture-cache release hook"
        ))
    ));
    let mut mismatched_cache_hook_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    mismatched_cache_hook_context.texture_cache_replacement_release_reachability =
        Some(replacement_release_reachability.clone());
    mismatched_cache_hook_context.texture_cache_release_hook = Some(
        SampledDmabufWaylandTextureCacheReleaseHook::new_for_tests(&unrelated_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_texture_cache_policy(&mismatched_cache_hook_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland texture-cache release hook identity"
        ))
    ));
    let mut missing_cache_lifecycle_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    missing_cache_lifecycle_context.texture_cache_replacement_release_reachability =
        Some(replacement_release_reachability.clone());
    missing_cache_lifecycle_context.texture_cache_release_hook = Some(
        SampledDmabufWaylandTextureCacheReleaseHook::new_for_tests(&policy_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_texture_cache_policy(&missing_cache_lifecycle_context),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan texture-cache release call sites"
        ))
    ));
    let mut mismatched_cache_lifecycle_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    mismatched_cache_lifecycle_context.texture_cache_replacement_release_reachability =
        Some(replacement_release_reachability.clone());
    mismatched_cache_lifecycle_context.texture_cache_release_hook = Some(
        SampledDmabufWaylandTextureCacheReleaseHook::new_for_tests(&policy_dmabuf),
    );
    mismatched_cache_lifecycle_context.texture_cache_release_lifecycle = Some(
        SampledDmabufWaylandTextureCacheReleaseLifecycle::new_for_tests(&unrelated_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_texture_cache_policy(&mismatched_cache_lifecycle_context),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland texture-cache release lifecycle identity"
        ))
    ));
    let mut cache_lifecycle_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
        &policy_dmabuf,
        &policy_import,
        &policy_acquire_evidence,
        &policy_release_evidence,
        true,
        SampledDmabufWaylandLayoutHistory::NoRendererHistory,
    );
    cache_lifecycle_context.texture_cache_replacement_release_reachability =
        Some(replacement_release_reachability);
    cache_lifecycle_context.texture_cache_release_hook = Some(
        SampledDmabufWaylandTextureCacheReleaseHook::new_for_tests(&policy_dmabuf),
    );
    cache_lifecycle_context.texture_cache_release_lifecycle = Some(
        SampledDmabufWaylandTextureCacheReleaseLifecycle::new_for_tests(&policy_dmabuf),
    );
    assert!(
        renderer
            .validate_sampled_dmabuf_wayland_texture_cache_policy(&cache_lifecycle_context)
            .is_ok()
    );
    let validated_cache_policy = renderer
        .validate_sampled_dmabuf_wayland_texture_cache_policy(&cache_lifecycle_context)
        .unwrap();
    assert!(validated_cache_policy.is_for_dmabuf(&policy_dmabuf));
    assert!(!validated_cache_policy.is_for_dmabuf(&unrelated_dmabuf));
    let mut wayland_vulkan_contracts = SampledDmabufWaylandVulkanInteropPolicyContracts::default();
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan layout policy"
        ))
    ));
    wayland_vulkan_contracts.layout = Some(SampledDmabufWaylandLayoutPolicy::Reacquire(
        SampledDmabufWaylandReacquireLayoutPolicy::new_for_tests(&unrelated_dmabuf),
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland layout identity"
        ))
    ));
    wayland_vulkan_contracts.layout = Some(SampledDmabufWaylandLayoutPolicy::Reacquire(
        SampledDmabufWaylandReacquireLayoutPolicy::new_for_tests(&policy_dmabuf),
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan foreign GENERAL policy"
        ))
    ));
    wayland_vulkan_contracts.foreign_general = Some(unsafe {
        // SAFETY: This unit test only validates contract routing; it performs no Vulkan import,
        // acquire, or sampling operation with the constructed evidence.
        SampledDmabufKnownLayoutEvidence::foreign_general(unrelated_dmabuf.weak())
    });
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland foreign GENERAL identity"
        ))
    ));
    wayland_vulkan_contracts.foreign_general = Some(unsafe {
        // SAFETY: This unit test only validates contract routing; it performs no Vulkan import,
        // acquire, or sampling operation with the constructed evidence.
        SampledDmabufKnownLayoutEvidence::foreign_general(policy_dmabuf.weak())
    });
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan queue-family policy"
        ))
    ));
    wayland_vulkan_contracts.queue_family_transfer = Some(
        SampledDmabufWaylandQueueFamilyPolicy::new_for_tests(&unrelated_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland queue-family identity"
        ))
    ));
    wayland_vulkan_contracts.queue_family_transfer = Some(
        SampledDmabufWaylandQueueFamilyPolicy::new_for_tests(&policy_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan acquire sync policy"
        ))
    ));
    wayland_vulkan_contracts.acquire_sync = Some(SampledDmabufWaylandAcquireSyncPolicy::new_for_tests(
        &unrelated_dmabuf,
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland acquire sync identity"
        ))
    ));
    wayland_vulkan_contracts.acquire_sync = Some(SampledDmabufWaylandAcquireSyncPolicy::new_for_tests(
        &policy_dmabuf,
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan release sync policy"
        ))
    ));
    wayland_vulkan_contracts.release_sync = Some(SampledDmabufWaylandReleaseSyncPolicy::new_for_tests(
        &unrelated_dmabuf,
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland release sync identity"
        ))
    ));
    wayland_vulkan_contracts.release_sync = Some(SampledDmabufWaylandReleaseSyncPolicy::new_for_tests(
        &policy_dmabuf,
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland Vulkan texture-cache policy"
        ))
    ));
    wayland_vulkan_contracts.texture_cache_reuse = Some(
        SampledDmabufWaylandTextureCachePolicy::new_for_tests(&unrelated_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland texture-cache identity"
        ))
    ));
    wayland_vulkan_contracts.texture_cache_reuse = Some(
        SampledDmabufWaylandTextureCachePolicy::new_for_tests(&policy_dmabuf),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland release ownership transfer"
        ))
    ));
    wayland_vulkan_contracts.release_ownership = Some(SampledDmabufReleaseOwnershipEvidence::new_for_tests(
        &unrelated_dmabuf,
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
            &policy_dmabuf,
            &wayland_vulkan_contracts,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland release ownership identity"
        ))
    ));
    wayland_vulkan_contracts.release_ownership = Some(SampledDmabufReleaseOwnershipEvidence::new_for_tests(
        &policy_dmabuf,
    ));
    let smithay_wayland_policy = SampledDmabufLayoutEvidence::SmithayWaylandVulkanPolicy(
        renderer
            .validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
                &policy_dmabuf,
                &wayland_vulkan_contracts,
            )
            .unwrap(),
    );
    let mut unrelated_wayland_vulkan_contracts = wayland_vulkan_contracts.clone();
    unrelated_wayland_vulkan_contracts.layout = Some(SampledDmabufWaylandLayoutPolicy::Reacquire(
        SampledDmabufWaylandReacquireLayoutPolicy::new_for_tests(&unrelated_dmabuf),
    ));
    unrelated_wayland_vulkan_contracts.foreign_general = Some(unsafe {
        // SAFETY: This unit test only validates contract routing; it performs no Vulkan import,
        // acquire, or sampling operation with the constructed evidence.
        SampledDmabufKnownLayoutEvidence::foreign_general(unrelated_dmabuf.weak())
    });
    unrelated_wayland_vulkan_contracts.queue_family_transfer = Some(
        SampledDmabufWaylandQueueFamilyPolicy::new_for_tests(&unrelated_dmabuf),
    );
    unrelated_wayland_vulkan_contracts.acquire_sync = Some(
        SampledDmabufWaylandAcquireSyncPolicy::new_for_tests(&unrelated_dmabuf),
    );
    unrelated_wayland_vulkan_contracts.release_sync = Some(
        SampledDmabufWaylandReleaseSyncPolicy::new_for_tests(&unrelated_dmabuf),
    );
    unrelated_wayland_vulkan_contracts.texture_cache_reuse = Some(
        SampledDmabufWaylandTextureCachePolicy::new_for_tests(&unrelated_dmabuf),
    );
    unrelated_wayland_vulkan_contracts.release_ownership = Some(
        SampledDmabufReleaseOwnershipEvidence::new_for_tests(&unrelated_dmabuf),
    );
    let mismatched_smithay_wayland_policy = SampledDmabufLayoutEvidence::SmithayWaylandVulkanPolicy(
        renderer
            .validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
                &unrelated_dmabuf,
                &unrelated_wayland_vulkan_contracts,
            )
            .unwrap(),
    );
    assert!(matches!(
        renderer.validate_sampled_dmabuf_known_layout_contract(
            &policy_dmabuf,
            mismatched_smithay_wayland_policy,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland policy identity"
        ))
    ));
    let mismatched_wayland_foreign_general_policy =
        SampledDmabufLayoutEvidence::SmithayWaylandVulkanPolicy(SampledDmabufWaylandVulkanInteropPolicy {
            dmabuf: policy_dmabuf.weak(),
            foreign_general: unsafe {
                // SAFETY: This unit test only validates contract routing; it performs no Vulkan
                // import, acquire, or sampling operation with the constructed evidence.
                SampledDmabufKnownLayoutEvidence::foreign_general(unrelated_dmabuf.weak())
            },
        });
    assert!(matches!(
        renderer.validate_sampled_dmabuf_known_layout_contract(
            &policy_dmabuf,
            mismatched_wayland_foreign_general_policy,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf Wayland foreign GENERAL identity"
        ))
    ));
    assert!(
        renderer
            .validate_sampled_dmabuf_known_layout_contract(&policy_dmabuf, smithay_wayland_policy)
            .is_ok()
    );
    let known_layout_evidence = SampledDmabufLayoutEvidence::KnownForeignGeneral(unsafe {
        // SAFETY: This unit test only validates contract routing; it performs no Vulkan import,
        // acquire, or sampling operation with the constructed evidence.
        SampledDmabufKnownLayoutEvidence::foreign_general(policy_dmabuf.weak())
    });
    assert!(
        renderer
            .validate_sampled_dmabuf_known_layout_contract(&policy_dmabuf, known_layout_evidence)
            .is_ok()
    );
    let mismatched_known_layout_evidence = SampledDmabufLayoutEvidence::KnownForeignGeneral(unsafe {
        // SAFETY: This unit test only validates contract routing; it performs no Vulkan import,
        // acquire, or sampling operation with the constructed evidence.
        SampledDmabufKnownLayoutEvidence::foreign_general(unrelated_dmabuf.weak())
    });
    assert!(matches!(
        renderer
            .validate_sampled_dmabuf_known_layout_contract(&policy_dmabuf, mismatched_known_layout_evidence),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf known-layout identity"
        ))
    ));
    assert!(matches!(
        renderer.validate_sampled_dmabuf_wayland_release_point_contract(&policy_dmabuf, false),
        Err(VulkanError::MissingCapability(
            "sampled dmabuf release point contract"
        ))
    ));
    let release_ownership = SampledDmabufReleaseOwnership::new_for_tests(&policy_dmabuf);
    let mismatched_release_ownership = SampledDmabufReleaseOwnership::new_for_tests(&unrelated_dmabuf);
    assert!(matches!(
        renderer.validate_sampled_dmabuf_release_lifecycle_contract(
            &policy_dmabuf,
            &mismatched_release_ownership,
        ),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf release ownership identity"
        ))
    ));
    renderer
        .validate_sampled_dmabuf_release_lifecycle_contract(&policy_dmabuf, &release_ownership)
        .unwrap();
    let release_obligation = release_ownership.into_release();
    assert!(release_obligation.signal_wayland_release_once().is_ok());
    assert!(release_obligation.signal_wayland_release_once().is_ok());
    let release = VulkanSampledDmabufRelease::validation_stage_without_wayland_point();
    assert!(release.signal_wayland_release_once().is_ok());
    assert!(release.signal_wayland_release_once().is_ok());
    let invalid_sync_file = File::open("/dev/null").unwrap();
    assert!(
        release
            .satisfy_wayland_release_once(Some(invalid_sync_file.as_fd()))
            .is_ok()
    );
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[test]
fn sampled_dmabuf_release_keeps_wayland_point_after_failed_satisfaction() {
    let release = VulkanSampledDmabufRelease::wayland_syncobj(DrmSyncPoint::invalid_for_tests(1).unwrap());

    assert!(matches!(
        release.signal_wayland_release_once(),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf release point signal"
        ))
    ));
    assert!(matches!(
        release.signal_wayland_release_once(),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf release point signal"
        ))
    ));

    let release = VulkanSampledDmabufRelease::wayland_syncobj(DrmSyncPoint::invalid_for_tests(2).unwrap());
    let invalid_sync_file = File::open("/dev/null").unwrap();
    assert!(matches!(
        release.satisfy_wayland_release_once(Some(invalid_sync_file.as_fd())),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf release point import sync file"
        ))
    ));
    assert!(matches!(
        release.satisfy_wayland_release_once(Some(invalid_sync_file.as_fd())),
        Err(VulkanError::UnsupportedOperation(
            "sampled dmabuf release point import sync file"
        ))
    ));
}

#[test]
fn internal_dmabuf_texture_release_rejects_preconditions_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut foreign_texture = texture_for_tests((1, 1).into(), Some(Fourcc::Abgr8888));
    foreign_texture.image.source = VulkanImageSource::DmabufImport;
    let mut memory_texture = texture_for_tests((1, 1).into(), Some(Fourcc::Abgr8888));
    memory_texture.context_id = renderer.context_id();
    memory_texture.image.source = VulkanImageSource::MemoryUpload;
    let mut missing_sampled_image_texture = texture_for_tests((1, 1).into(), Some(Fourcc::Abgr8888));
    missing_sampled_image_texture.context_id = renderer.context_id();
    missing_sampled_image_texture.image.source = VulkanImageSource::DmabufImport;
    missing_sampled_image_texture.image.layout = VulkanImageLayoutState::ShaderReadOnly;
    missing_sampled_image_texture.image.sync = VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    };

    assert!(matches!(
        renderer.release_imported_dmabuf_texture_to_foreign_general(&foreign_texture, false),
        Err(VulkanError::UnsupportedOperation("foreign dmabuf texture"))
    ));
    assert!(matches!(
        renderer.release_imported_dmabuf_texture_to_foreign_general(&memory_texture, false),
        Err(VulkanError::UnsupportedOperation("dmabuf texture"))
    ));
    assert!(
        renderer
            .release_retired_wayland_texture_for_cache(&memory_texture)
            .is_ok()
    );
    assert!(Renderer::release_imported_texture_for_surface_cache(&mut renderer, &memory_texture).is_ok());
    let mut release_obligation_texture = missing_sampled_image_texture.clone();
    release_obligation_texture.sampled_dmabuf_release =
        Some(VulkanSampledDmabufRelease::validation_stage_without_wayland_point());
    assert!(matches!(
        renderer.release_retired_wayland_texture_for_cache(&release_obligation_texture),
        Err(SurfaceCacheTextureReleaseError::RetrySafe(
            VulkanError::UnsupportedOperation("dmabuf texture sampled image")
        ))
    ));
    assert!(matches!(
        Renderer::release_imported_texture_for_surface_cache(&mut renderer, &release_obligation_texture),
        Err(SurfaceCacheTextureReleaseError::RetrySafe(
            VulkanError::UnsupportedOperation("dmabuf texture sampled image")
        ))
    ));
    assert!(matches!(
        renderer.release_imported_dmabuf_texture_to_foreign_general(&release_obligation_texture, true),
        Err(VulkanError::UnsupportedOperation("dmabuf texture sampled image"))
    ));
    assert!(matches!(
        renderer.release_imported_dmabuf_texture_to_foreign_general(&missing_sampled_image_texture, true),
        Err(VulkanError::UnsupportedOperation("dmabuf texture sampled image"))
    ));
    assert!(matches!(
        renderer.release_imported_dmabuf_texture_to_foreign_general_sync_point(
            &missing_sampled_image_texture,
            true
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf texture sampled image"))
    ));
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[test]
fn sampled_cache_release_point_failure_is_committed_side_effect() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut texture = texture_for_tests((1, 1).into(), Some(Fourcc::Abgr8888));
    texture.context_id = renderer.context_id();
    texture.image.source = VulkanImageSource::DmabufImport;
    texture.sampled_dmabuf_release = Some(VulkanSampledDmabufRelease::wayland_syncobj(
        DrmSyncPoint::invalid_for_tests(1).unwrap(),
    ));

    assert!(matches!(
        renderer.complete_sampled_dmabuf_cache_release_after_device_release(&texture, None),
        Err(SurfaceCacheTextureReleaseError::ReleaseSideEffectsCommitted(
            VulkanError::UnsupportedOperation("sampled dmabuf release point signal")
        ))
    ));
}

#[test]
fn sampled_release_submit_errors_are_classified_by_queue_acceptance() {
    assert!(matches!(
        classify_sampled_dmabuf_release_submit_error_for_tests(
            false,
            false,
            VulkanError::UnsupportedOperation("submit")
        ),
        VulkanSampledDmabufForeignReleaseError::RetrySafe(VulkanError::UnsupportedOperation("submit"))
    ));
    assert!(matches!(
        classify_sampled_dmabuf_release_submit_error_for_tests(
            true,
            false,
            VulkanError::UnsupportedOperation("submit")
        ),
        VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(VulkanError::UnsupportedOperation("submit"))
    ));
    assert!(matches!(
        classify_sampled_dmabuf_release_submit_error_for_tests(
            false,
            true,
            VulkanError::UnsupportedOperation("submit")
        ),
        VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(VulkanError::UnsupportedOperation("submit"))
    ));
}

#[test]
fn sampled_cache_release_maps_device_release_classification() {
    assert!(matches!(
        sampled_dmabuf_cache_device_release_error(VulkanSampledDmabufForeignReleaseError::RetrySafe(
            VulkanError::UnsupportedOperation("release")
        )),
        SurfaceCacheTextureReleaseError::RetrySafe(VulkanError::UnsupportedOperation("release"))
    ));
    assert!(matches!(
        sampled_dmabuf_cache_device_release_error(VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(
            VulkanError::UnsupportedOperation("release")
        )),
        SurfaceCacheTextureReleaseError::ReleaseSideEffectsCommitted(VulkanError::UnsupportedOperation(
            "release"
        ))
    ));
}

#[test]
fn public_export_mem_rejects_invalid_vulkan_targets_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let foreign_target = render_target_for_tests(
        ContextId::new(),
        VulkanImageSource::Offscreen,
        (1, 1).into(),
        Some(Fourcc::Abgr8888),
    );
    let non_offscreen_target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::Uninitialized,
        (1, 1).into(),
        Some(Fourcc::Abgr8888),
    );
    let missing_image_target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::Offscreen,
        (1, 1).into(),
        Some(Fourcc::Abgr8888),
    );
    let region = Rectangle::new((0, 0).into(), (1, 1).into());

    assert!(matches!(
        renderer.copy_framebuffer(&foreign_target, region, Fourcc::Abgr8888),
        Err(VulkanError::UnsupportedOperation("foreign render target"))
    ));
    assert!(matches!(
        renderer.copy_framebuffer(&non_offscreen_target, region, Fourcc::Abgr8888),
        Err(VulkanError::UnsupportedOperation("render target"))
    ));
    assert!(matches!(
        renderer.copy_framebuffer(&missing_image_target, region, Fourcc::Argb8888),
        Err(VulkanError::UnsupportedFormat(Fourcc::Argb8888))
    ));
    assert!(matches!(
        renderer.copy_framebuffer(&missing_image_target, region, Fourcc::Abgr8888),
        Err(VulkanError::VulkanUnavailable)
    ));
}

#[test]
fn public_export_mem_rejects_texture_readback_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let foreign_texture = texture_for_tests((1, 1).into(), Some(Fourcc::Abgr8888));
    let texture = VulkanTexture {
        context_id: renderer.context_id(),
        ..texture_for_tests((1, 1).into(), Some(Fourcc::Abgr8888))
    };
    let region = Rectangle::new((0, 0).into(), (1, 1).into());

    assert!(matches!(
        renderer.can_read_texture(&foreign_texture),
        Err(VulkanError::UnsupportedOperation("foreign memory texture"))
    ));
    assert!(matches!(renderer.can_read_texture(&texture), Ok(false)));
    assert!(matches!(
        renderer.copy_texture(&foreign_texture, region, Fourcc::Abgr8888),
        Err(VulkanError::UnsupportedOperation("foreign memory texture"))
    ));
    assert!(matches!(
        renderer.copy_texture(&texture, region, Fourcc::Abgr8888),
        Err(VulkanError::UnsupportedOperation("texture memory export"))
    ));
}

#[test]
fn vulkan_memory_mapping_reports_readback_metadata() {
    let mapping = VulkanMemoryMapping {
        data: vec![1, 2, 3, 4, 5, 6, 7, 8],
        size: Size::from((2, 1)),
        format: Fourcc::Abgr8888,
        flipped: false,
    };

    assert_eq!(mapping.width(), 2);
    assert_eq!(mapping.height(), 1);
    assert_eq!(mapping.size(), Size::from((2, 1)));
    assert_eq!(Texture::format(&mapping), Some(Fourcc::Abgr8888));
    assert_eq!(TextureMapping::format(&mapping), Fourcc::Abgr8888);
    assert!(!mapping.flipped());

    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    assert_eq!(renderer.map_texture(&mapping).unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn public_bind_supported_formats_match_probed_render_targets() {
    let renderer = VulkanRenderer::new_scaffold_for_tests();
    let formats = <VulkanRenderer as Bind<VulkanRenderTarget<'static>>>::supported_formats(&renderer)
        .expect("Vulkan render targets have format metadata");

    assert!(formats.iter().next().is_none());
}

#[test]
fn public_bind_supported_formats_filter_provisional_render_target_capabilities() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    renderer.capabilities.formats.records = vec![
        render_target_format_record_for_tests(Fourcc::Abgr8888, VulkanFormatTiling::Optimal, true),
        render_target_format_record_for_tests(Fourcc::Xrgb2101010, VulkanFormatTiling::Optimal, true),
        render_target_format_record_for_tests(Fourcc::Argb8888, VulkanFormatTiling::Optimal, false),
        render_target_format_record_for_tests(Fourcc::Xrgb8888, VulkanFormatTiling::Linear, true),
    ];

    let formats = <VulkanRenderer as Bind<VulkanRenderTarget<'static>>>::supported_formats(&renderer)
        .expect("Vulkan render targets have format metadata");

    assert!(formats.contains(&Format {
        code: Fourcc::Abgr8888,
        modifier: Modifier::Invalid,
    }));
    assert!(!formats.contains(&Format {
        code: Fourcc::Xrgb2101010,
        modifier: Modifier::Invalid,
    }));
    assert!(!formats.contains(&Format {
        code: Fourcc::Argb8888,
        modifier: Modifier::Invalid,
    }));
    assert!(!formats.contains(&Format {
        code: Fourcc::Xrgb8888,
        modifier: Modifier::Invalid,
    }));
    assert!(renderer.render_target_format_supported(Fourcc::Abgr8888));
    assert!(!renderer.render_target_format_supported(Fourcc::Xrgb2101010));
    assert!(!renderer.render_target_format_supported(Fourcc::Argb8888));
    assert!(!renderer.render_target_format_supported(Fourcc::Xrgb8888));
}

#[test]
fn uninitialized_renderer_reports_no_memory_import_formats() {
    let renderer = VulkanRenderer::new_scaffold_for_tests();

    assert!(renderer.mem_formats().next().is_none());
}

#[test]
fn memory_update_rejects_foreign_renderer_textures() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let texture = texture_for_tests((1, 1).into(), Some(Fourcc::Abgr8888));

    assert!(matches!(
        renderer.update_memory(
            &texture,
            &[0x00, 0x00, 0x00, 0x00],
            Rectangle::new((0, 0).into(), (1, 1).into()),
        ),
        Err(VulkanError::UnsupportedOperation("foreign memory texture"))
    ));
}

#[test]
fn memory_update_region_validation_rejects_out_of_bounds_regions() {
    assert!(matches!(
        super::update_region_to_vk((4, 4).into(), Rectangle::new((1, 1).into(), (2, 2).into())),
        Ok((
            vk::Offset3D { x: 1, y: 1, z: 0 },
            vk::Extent3D {
                width: 2,
                height: 2,
                depth: 1,
            },
        ))
    ));
    assert!(matches!(
        super::update_region_to_vk((4, 4).into(), Rectangle::new((-1, 0).into(), (1, 1).into())),
        Err(VulkanError::UnsupportedOperation("memory update region"))
    ));
    assert!(matches!(
        super::update_region_to_vk((4, 4).into(), Rectangle::new((3, 3).into(), (2, 1).into())),
        Err(VulkanError::UnsupportedOperation("memory update region"))
    ));
}

#[test]
fn image_region_to_vk_rejects_invalid_readback_regions() {
    let image_size = Size::<i32, BufferCoord>::from((4, 3));

    assert!(matches!(
        super::image_region_to_vk(
            image_size,
            Rectangle::new((-1, 0).into(), (1, 1).into()),
            "test region"
        ),
        Err(VulkanError::UnsupportedOperation("test region"))
    ));
    assert!(matches!(
        super::image_region_to_vk(
            image_size,
            Rectangle::new((0, 0).into(), (0, 1).into()),
            "test region"
        ),
        Err(VulkanError::UnsupportedOperation("test region"))
    ));
    assert!(matches!(
        super::image_region_to_vk(
            image_size,
            Rectangle::new((3, 0).into(), (2, 1).into()),
            "test region"
        ),
        Err(VulkanError::UnsupportedOperation("test region"))
    ));
    assert!(matches!(
        super::image_region_to_vk(
            image_size,
            Rectangle::new((0, 2).into(), (1, 2).into()),
            "test region"
        ),
        Err(VulkanError::UnsupportedOperation("test region"))
    ));
    assert!(matches!(
        super::image_region_to_vk(
            Size::<i32, BufferCoord>::from((i32::MAX, 1)),
            Rectangle::new((i32::MAX, 0).into(), (1, 1).into()),
            "test region"
        ),
        Err(VulkanError::UnsupportedOperation("test region"))
    ));
    assert!(matches!(
        super::image_region_to_vk(
            Size::<i32, BufferCoord>::from((1, i32::MAX)),
            Rectangle::new((0, i32::MAX).into(), (1, 1).into()),
            "test region"
        ),
        Err(VulkanError::UnsupportedOperation("test region"))
    ));
}

#[test]
fn image_region_to_vk_converts_valid_readback_regions() {
    let (offset, extent) = super::image_region_to_vk(
        Size::<i32, BufferCoord>::from((4, 3)),
        Rectangle::new((1, 1).into(), (2, 1).into()),
        "test region",
    )
    .unwrap();

    assert_eq!(offset, vk::Offset3D { x: 1, y: 1, z: 0 });
    assert_eq!(
        extent,
        vk::Extent3D {
            width: 2,
            height: 1,
            depth: 1,
        }
    );
}

#[test]
fn format_usage_maps_vulkan_feature_flags() {
    let usage = format_usage_from_features(
        vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
            | vk::FormatFeatureFlags::BLIT_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST,
    );

    assert!(usage.sampled);
    assert!(usage.color_attachment);
    assert!(usage.color_attachment_blend);
    assert!(usage.blit_src);
    assert!(!usage.blit_dst);
    assert!(!usage.transfer_src);
    assert!(usage.transfer_dst);
    assert!(!usage.memory_import);
    assert!(!usage.dmabuf_import);
    assert!(!usage.dmabuf_export);
}

#[test]
fn linear_tiling_support_is_explicitly_limited_to_usable_features() {
    assert!(linear_tiling_supported(vk::FormatFeatureFlags::TRANSFER_SRC));
    assert!(linear_tiling_supported(vk::FormatFeatureFlags::SAMPLED_IMAGE));
    assert!(!linear_tiling_supported(vk::FormatFeatureFlags::empty()));
}

#[test]
fn queue_family_selection_requires_graphics() {
    let families = [vk::QueueFamilyProperties {
        queue_flags: vk::QueueFlags::TRANSFER,
        queue_count: 1,
        ..Default::default()
    }];

    assert!(matches!(
        select_queue_families(&families),
        Err(VulkanError::QueueFamilyUnsupported)
    ));
}

#[test]
fn queue_family_selection_prefers_dedicated_transfer() {
    let families = [
        vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER,
            queue_count: 1,
            ..Default::default()
        },
        vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::TRANSFER,
            queue_count: 1,
            ..Default::default()
        },
    ];

    let selected = select_queue_families(&families).unwrap();

    assert_eq!(selected.graphics, Some(0));
    assert_eq!(selected.transfer, Some(1));
}

#[test]
fn queue_family_selection_uses_graphics_for_transfer_fallback() {
    let families = [vk::QueueFamilyProperties {
        queue_flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER,
        queue_count: 1,
        ..Default::default()
    }];

    let selected = select_queue_families(&families).unwrap();

    assert_eq!(selected.graphics, Some(0));
    assert_eq!(selected.transfer, Some(0));
}

#[test]
fn vulkan_image_state_tracks_target_metadata() {
    let target = VulkanRenderTarget::new_for_tests((2, 3).into(), Some(Fourcc::Argb8888));
    assert_eq!(target.width(), 2);
    assert_eq!(target.height(), 3);
    assert_eq!(target.size(), Size::from((2, 3)));
    assert_eq!(Texture::format(&target), Some(Fourcc::Argb8888));
    assert_eq!(target.image.source, VulkanImageSource::Uninitialized);
    assert_eq!(target.image.layout, VulkanImageLayoutState::Undefined);
    assert_eq!(target.image.usage, VulkanImageUsage::default());
    assert_eq!(target.image.sync, VulkanImageSyncState::default());
    assert!(!target.has_color_image_for_tests());
}

#[test]
fn vulkan_texture_tracks_texture_metadata() {
    let texture = texture_for_tests((5, 7).into(), None);

    assert_eq!(texture.width(), 5);
    assert_eq!(texture.height(), 7);
    assert_eq!(texture.size(), Size::from((5, 7)));
    assert_eq!(Texture::format(&texture), None);
    assert_eq!(texture.image.source, VulkanImageSource::Uninitialized);
    assert_eq!(texture.image.layout, VulkanImageLayoutState::Undefined);
    assert_eq!(texture.image.usage, VulkanImageUsage::default());
    assert_eq!(texture.image.sync, VulkanImageSyncState::default());
}

#[test]
fn vulkan_api_errors_convert_cleanly() {
    assert!(matches!(
        VulkanError::from(ash::vk::Result::ERROR_DEVICE_LOST),
        VulkanError::VulkanApi(ash::vk::Result::ERROR_DEVICE_LOST)
    ));
}

#[test]
fn vulkan_errors_map_to_swap_buffers_error() {
    let context_lost_errors = [
        VulkanError::VulkanUnavailable,
        VulkanError::MissingRequiredExtension("VK_EXT_test".to_owned()),
        VulkanError::DeviceLost,
        VulkanError::DeviceInitializationFailed("test".to_owned()),
        VulkanError::QueueFamilyUnsupported,
        VulkanError::ExternalMemoryUnsupported,
        VulkanError::VulkanApi(ash::vk::Result::ERROR_DEVICE_LOST),
        VulkanError::VulkanApi(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY),
        VulkanError::VulkanApi(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
        VulkanError::VulkanApi(ash::vk::Result::ERROR_INITIALIZATION_FAILED),
    ];

    for err in context_lost_errors {
        assert!(
            matches!(
                crate::backend::SwapBuffersError::from(err),
                crate::backend::SwapBuffersError::ContextLost(_)
            ),
            "expected context-lost classification"
        );
    }

    let temporary_errors = [
        VulkanError::UnsupportedOperation("test"),
        VulkanError::NotPublicAdvertised("test"),
        VulkanError::MissingCapability("test"),
        VulkanError::UnsupportedFormat(Fourcc::Argb8888),
        VulkanError::UnsupportedModifier,
        VulkanError::MemoryTypeUnsupported,
        VulkanError::SyncInterrupted,
        VulkanError::VulkanApi(ash::vk::Result::ERROR_FORMAT_NOT_SUPPORTED),
    ];

    for err in temporary_errors {
        assert!(
            matches!(
                crate::backend::SwapBuffersError::from(err),
                crate::backend::SwapBuffersError::TemporaryFailure(_)
            ),
            "expected temporary-failure classification"
        );
    }
}

#[test]
fn vulkan_api_context_invalidation_policy_is_explicit() {
    for result in [
        ash::vk::Result::ERROR_DEVICE_LOST,
        ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ash::vk::Result::ERROR_INITIALIZATION_FAILED,
        ash::vk::Result::ERROR_EXTENSION_NOT_PRESENT,
        ash::vk::Result::ERROR_FEATURE_NOT_PRESENT,
        ash::vk::Result::ERROR_INCOMPATIBLE_DRIVER,
    ] {
        assert!(
            vulkan_api_result_invalidates_context(result),
            "{result:?} should invalidate the renderer context"
        );
    }

    for result in [
        ash::vk::Result::ERROR_FORMAT_NOT_SUPPORTED,
        ash::vk::Result::TIMEOUT,
    ] {
        assert!(
            !vulkan_api_result_invalidates_context(result),
            "{result:?} should remain retryable or operation-local"
        );
    }
}

#[test]
fn external_memory_metadata_tracks_planes() {
    let memory = VulkanExternalMemoryState {
        handle_type: VulkanExternalMemoryHandleType::Dmabuf,
        format: Fourcc::Argb8888,
        modifier: crate::backend::allocator::Modifier::Linear,
        planes: vec![VulkanDmabufPlane {
            plane_idx: 0,
            offset: 0,
            stride: 4,
        }],
        disjoint: false,
    };
    let import = VulkanDmabufImportState {
        size: (1, 1).into(),
        memory,
        y_inverted: false,
    };

    assert_eq!(import.memory.handle_type, VulkanExternalMemoryHandleType::Dmabuf);
    assert_eq!(import.memory.format, Fourcc::Argb8888);
    assert_eq!(
        import.memory.modifier,
        crate::backend::allocator::Modifier::Linear
    );
    assert_eq!(import.memory.planes.len(), 1);
    assert!(!import.memory.disjoint);
    assert!(!import.y_inverted);
}

#[test]
fn dmabuf_import_state_extracts_metadata() {
    let dmabuf = dmabuf_with_planes_for_tests(
        (4, 3).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::Y_INVERT,
        &[(0, 16, 32), (1, 48, 16)],
    );

    let import = VulkanDmabufImportState::from_dmabuf(&dmabuf).unwrap();

    assert_eq!(import.size, Size::from((4, 3)));
    assert_eq!(import.memory.handle_type, VulkanExternalMemoryHandleType::Dmabuf);
    assert_eq!(import.memory.format, Fourcc::Nv12);
    assert_eq!(import.memory.modifier, Modifier::Linear);
    assert_eq!(import.memory.planes.len(), 2);
    assert_eq!(
        import.memory.planes,
        vec![
            VulkanDmabufPlane {
                plane_idx: 0,
                offset: 16,
                stride: 32,
            },
            VulkanDmabufPlane {
                plane_idx: 1,
                offset: 48,
                stride: 16,
            },
        ]
    );
    assert!(import.memory.disjoint);
    assert!(import.y_inverted);
}

#[test]
fn dmabuf_import_state_rejects_invalid_metadata() {
    let zero_width = dmabuf_with_planes_for_tests(
        (0, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4)],
    );
    assert!(matches!(
        VulkanDmabufImportState::from_dmabuf(&zero_width),
        Err(VulkanError::UnsupportedOperation("dmabuf size"))
    ));

    let missing_plane_zero = dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(1, 0, 4)],
    );
    assert!(matches!(
        VulkanDmabufImportState::from_dmabuf(&missing_plane_zero),
        Err(VulkanError::UnsupportedOperation("dmabuf plane index"))
    ));

    let duplicate_plane_zero = dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4), (0, 4, 4)],
    );
    assert!(matches!(
        VulkanDmabufImportState::from_dmabuf(&duplicate_plane_zero),
        Err(VulkanError::UnsupportedOperation("dmabuf plane index"))
    ));

    let plane_gap = dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Nv12,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 4), (2, 4, 4)],
    );
    assert!(matches!(
        VulkanDmabufImportState::from_dmabuf(&plane_gap),
        Err(VulkanError::UnsupportedOperation("dmabuf plane index"))
    ));

    let zero_stride = dmabuf_with_planes_for_tests(
        (1, 1).into(),
        Fourcc::Abgr8888,
        Modifier::Linear,
        DmabufFlags::empty(),
        &[(0, 0, 0)],
    );
    assert!(matches!(
        VulkanDmabufImportState::from_dmabuf(&zero_stride),
        Err(VulkanError::UnsupportedOperation("dmabuf stride"))
    ));
}

#[test]
fn vulkan_renderer_reports_unusable_and_waits() {
    assert!(matches!(
        VulkanRenderer::new(),
        Err(VulkanError::VulkanUnavailable)
    ));

    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    assert!(!renderer.is_device_initialized());
    assert!(renderer.wait(&SyncPoint::signaled()).is_ok());
}

#[test]
fn unsupported_frame_creation_returns_clean_error() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut target = VulkanRenderTarget::new_for_tests((1, 1).into(), Some(Fourcc::Argb8888));
    let err = renderer
        .render(&mut target, (1, 1).into(), Transform::Normal)
        .unwrap_err();
    assert!(matches!(err, VulkanError::UnsupportedOperation(_)));
}

#[test]
fn renderer_render_rejects_foreign_targets_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut target = render_target_for_tests(
        ContextId::new(),
        VulkanImageSource::Offscreen,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );

    assert!(matches!(
        renderer.render(&mut target, (1, 1).into(), Transform::Normal),
        Err(VulkanError::UnsupportedOperation("foreign render target"))
    ));
}

#[test]
fn renderer_render_rejects_non_offscreen_targets_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::Uninitialized,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );

    assert!(matches!(
        renderer.render(&mut target, (1, 1).into(), Transform::Normal),
        Err(VulkanError::UnsupportedOperation("render target"))
    ));
}

#[test]
fn renderer_render_accepts_internal_dmabuf_target_source_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::RenderTarget,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );

    assert!(matches!(
        renderer.render(&mut target, (1, 1).into(), Transform::Normal),
        Err(VulkanError::UnsupportedOperation("render target image"))
    ));
}

#[test]
fn renderer_render_rejects_released_dmabuf_target_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::RenderTarget,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );
    target.image.sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();

    assert!(matches!(
        renderer.render(&mut target, (1, 1).into(), Transform::Normal),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
}

#[test]
fn renderer_render_validates_output_size_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::Offscreen,
        (2, 2).into(),
        Some(Fourcc::Argb8888),
    );

    assert!(matches!(
        renderer.render(&mut target, (0, 2).into(), Transform::Normal),
        Err(VulkanError::UnsupportedOperation("frame size"))
    ));
    assert!(matches!(
        renderer.render(&mut target, (1, 2).into(), Transform::Normal),
        Err(VulkanError::UnsupportedOperation("frame size"))
    ));
}

#[test]
fn renderer_render_rejects_missing_offscreen_image_before_device_lookup() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut target = render_target_for_tests(
        renderer.context_id(),
        VulkanImageSource::Offscreen,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );

    assert!(matches!(
        renderer.render(&mut target, (1, 1).into(), Transform::Normal),
        Err(VulkanError::UnsupportedOperation("render target image"))
    ));
}

#[test]
fn frame_metadata_accessors_report_uninitialized_state() {
    let context_id = ContextId::new();
    let frame = frame_for_tests(context_id.clone(), (11, 13).into(), Transform::Flipped270);

    assert_eq!(frame.context_id(), context_id);
    assert_eq!(frame.output_size(), Size::from((11, 13)));
    assert_eq!(frame.transformation(), Transform::Flipped270);
}

#[test]
fn frame_noops_on_empty_damage() {
    let mut frame = frame_for_tests(ContextId::new(), (11, 13).into(), Transform::Flipped270);
    let texture = texture_for_tests((1, 1).into(), Some(Fourcc::Argb8888));

    assert!(frame.wait(&SyncPoint::signaled()).is_ok());
    assert!(frame.clear(Color32F::TRANSPARENT, &[]).is_ok());
    assert!(
        frame
            .draw_solid(Rectangle::from_size((1, 1).into()), &[], Color32F::TRANSPARENT)
            .is_ok()
    );
    assert!(
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((1.0, 1.0).into()),
                Rectangle::from_size((1, 1).into()),
                &[],
                &[],
                Transform::Normal,
                1.0,
            )
            .is_ok()
    );
}

#[test]
fn frame_rejects_non_empty_work() {
    let mut frame = frame_for_tests(ContextId::new(), (11, 13).into(), Transform::Flipped270);
    let texture = texture_for_tests((1, 1).into(), Some(Fourcc::Argb8888));
    let damage = [Rectangle::from_size((1, 1).into())];

    assert!(matches!(
        frame.clear(Color32F::TRANSPARENT, &damage),
        Err(VulkanError::UnsupportedOperation(_))
    ));
    assert!(matches!(
        frame.draw_solid(
            Rectangle::from_size((1, 1).into()),
            &damage,
            Color32F::TRANSPARENT
        ),
        Err(VulkanError::UnsupportedOperation(_))
    ));
    assert!(matches!(
        frame.render_texture_from_to(
            &texture,
            Rectangle::from_size((1.0, 1.0).into()),
            Rectangle::from_size((1, 1).into()),
            &damage,
            &[],
            Transform::Normal,
            1.0,
        ),
        Err(VulkanError::UnsupportedOperation(_))
    ));
}

#[test]
fn frame_clear_accepts_partial_damage_before_device_lookup() {
    let mut frame = frame_for_tests(ContextId::new(), (4, 4).into(), Transform::Normal);
    let partial_damage = [Rectangle::new((1, 1).into(), (2, 2).into())];
    let overlapping_damage = [
        Rectangle::new((0, 0).into(), (2, 2).into()),
        Rectangle::new((1, 1).into(), (2, 2).into()),
    ];

    assert!(matches!(
        frame.clear(Color32F::TRANSPARENT, &partial_damage),
        Err(VulkanError::UnsupportedOperation("clear device"))
    ));
    assert!(matches!(
        frame.clear(Color32F::TRANSPARENT, &overlapping_damage),
        Err(VulkanError::UnsupportedOperation("clear device"))
    ));
    assert!(
        frame
            .clear(
                Color32F::TRANSPARENT,
                &[Rectangle::new((5, 0).into(), (1, 1).into())]
            )
            .is_ok()
    );

    let mut transformed_frame = frame_for_tests(ContextId::new(), (4, 4).into(), Transform::Flipped270);
    assert!(matches!(
        transformed_frame.clear(Color32F::TRANSPARENT, &partial_damage),
        Err(VulkanError::UnsupportedOperation("clear transform"))
    ));
}

#[test]
fn frame_draw_solid_rejects_preconditions_before_device_lookup() {
    let context_id = ContextId::new();
    let full_damage = [Rectangle::from_size((4, 4).into())];

    fn assert_draw_solid_error(
        frame_transform: Transform,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
        expected: &'static str,
    ) {
        let mut frame = frame_for_tests(ContextId::new(), (4, 4).into(), frame_transform);

        assert!(
            matches!(
                frame.draw_solid(dst, damage, color),
                Err(VulkanError::UnsupportedOperation(reason)) if reason == expected
            ),
            "expected UnsupportedOperation({expected:?})"
        );
    }

    assert_draw_solid_error(
        Transform::_90,
        Rectangle::from_size((4, 4).into()),
        &full_damage,
        Color32F::BLACK,
        "draw solid transform",
    );
    assert_draw_solid_error(
        Transform::Normal,
        Rectangle::from_size((4, 4).into()),
        &full_damage,
        Color32F::new(0.0, 0.0, 0.0, 0.5),
        "draw solid device",
    );
    assert_draw_solid_error(
        Transform::Normal,
        Rectangle::from_size((4, 4).into()),
        &[
            Rectangle::new((0, 0).into(), (2, 2).into()),
            Rectangle::new((1, 1).into(), (2, 2).into()),
        ],
        Color32F::new(0.0, 0.0, 0.0, 0.5),
        "draw solid damage",
    );
    assert_draw_solid_error(
        Transform::Normal,
        Rectangle::new((3, 0).into(), (2, 4).into()),
        &full_damage,
        Color32F::BLACK,
        "draw solid device",
    );
    assert_draw_solid_error(
        Transform::Normal,
        Rectangle::from_size((4, 4).into()),
        &[
            Rectangle::new((0, 0).into(), (2, 2).into()),
            Rectangle::new((1, 1).into(), (2, 2).into()),
        ],
        Color32F::BLACK,
        "draw solid device",
    );

    let mut frame = frame_for_tests(context_id, (4, 4).into(), Transform::Normal);
    assert!(
        frame
            .draw_solid(
                Rectangle::new((5, 0).into(), (1, 1).into()),
                &full_damage,
                Color32F::BLACK,
            )
            .is_ok()
    );
    assert!(matches!(
        frame.draw_solid(Rectangle::from_size((4, 4).into()), &full_damage, Color32F::BLACK),
        Err(VulkanError::UnsupportedOperation("draw solid device"))
    ));
}

#[test]
fn frame_finish_is_signaled_without_submitted_work() {
    let frame = frame_for_tests(ContextId::new(), (1, 1).into(), Transform::Normal);
    let sync = frame.finish().unwrap();

    assert!(sync.is_reached());
    assert!(!sync.contains_fence());
    assert!(!sync.is_exportable());
}

#[test]
fn frame_finish_releases_internal_dmabuf_target_preconditions() {
    let context_id = ContextId::new();
    let mut missing_image = render_target_for_tests(
        context_id.clone(),
        VulkanImageSource::RenderTarget,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );
    let frame = VulkanFrame {
        context_id: context_id.clone(),
        output_size: (1, 1).into(),
        transform: Transform::Normal,
        device: None,
        target: Some(&mut missing_image),
        _renderer: PhantomData,
    };

    assert!(matches!(
        frame.finish(),
        Err(VulkanError::UnsupportedOperation("dmabuf render target image"))
    ));

    let mut released_target = render_target_for_tests(
        context_id.clone(),
        VulkanImageSource::RenderTarget,
        (1, 1).into(),
        Some(Fourcc::Argb8888),
    );
    released_target.image.sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
    let frame = VulkanFrame {
        context_id,
        output_size: (1, 1).into(),
        transform: Transform::Normal,
        device: None,
        target: Some(&mut released_target),
        _renderer: PhantomData,
    };

    assert!(matches!(
        frame.finish(),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
}

#[test]
fn frame_render_texture_supports_source_transforms_before_device_lookup() {
    let context_id = ContextId::new();
    let mut texture = texture_for_tests((2, 3).into(), Some(Fourcc::Argb8888));
    texture.context_id = context_id.clone();
    let damage = [Rectangle::from_size((8, 6).into())];

    for transform in [
        Transform::Normal,
        Transform::_90,
        Transform::_180,
        Transform::_270,
        Transform::Flipped,
        Transform::Flipped90,
        Transform::Flipped180,
        Transform::Flipped270,
    ] {
        let mut frame = frame_for_tests(context_id.clone(), (8, 6).into(), Transform::Normal);

        assert!(
            matches!(
                frame.render_texture_from_to(
                    &texture,
                    Rectangle::from_size((2.0, 3.0).into()),
                    Rectangle::from_size((8, 6).into()),
                    &damage,
                    &[],
                    transform,
                    0.5,
                ),
                Err(VulkanError::UnsupportedOperation("render texture device"))
            ),
            "transform {transform:?}"
        );
    }
}

#[test]
fn frame_render_texture_rejects_narrow_path_preconditions_before_device_lookup() {
    let output_size = Size::from((8, 6));
    let full_src = Rectangle::from_size((2.0, 3.0).into());
    let full_dst = Rectangle::from_size(output_size);
    let full_damage = [Rectangle::from_size(output_size)];
    let opaque_region = [Rectangle::from_size((1, 1).into())];

    fn assert_render_texture_error(
        texture: &VulkanTexture,
        frame_context_id: ContextId<VulkanTexture>,
        frame_transform: Transform,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
        expected: &'static str,
    ) {
        let mut frame = frame_for_tests(frame_context_id, (8, 6).into(), frame_transform);

        assert!(
            matches!(
                frame.render_texture_from_to(
                    texture,
                    src,
                    dst,
                    damage,
                    opaque_regions,
                    src_transform,
                    alpha,
                ),
                Err(VulkanError::UnsupportedOperation(reason)) if reason == expected
            ),
            "expected UnsupportedOperation({expected:?})"
        );
    }

    let frame_context_id = ContextId::new();
    let mut texture = texture_for_tests((2, 3).into(), Some(Fourcc::Argb8888));

    assert_render_texture_error(
        &texture,
        frame_context_id.clone(),
        Transform::Normal,
        full_src,
        full_dst,
        &full_damage,
        &[],
        Transform::Normal,
        1.0,
        "foreign render texture",
    );

    texture.context_id = frame_context_id.clone();

    for sync in [
        VulkanImageSyncState {
            external_acquire_pending: true,
            ..VulkanImageSyncState::default()
        },
        VulkanImageSyncState {
            external_acquire_pending: true,
            external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
            ..VulkanImageSyncState::default()
        },
        VulkanImageSyncState::foreign_known_general_for_dmabuf_import(),
        VulkanImageSyncState {
            external_acquire_pending: true,
            external_ownership: VulkanExternalImageOwnership::AcquirePending,
            ..VulkanImageSyncState::default()
        },
        VulkanImageSyncState {
            external_acquire_pending: true,
            external_ownership: VulkanExternalImageOwnership::Local,
            ..VulkanImageSyncState::default()
        },
        VulkanImageSyncState {
            external_ownership: VulkanExternalImageOwnership::ReleasePending,
            ..VulkanImageSyncState::default()
        },
    ] {
        let mut externally_owned_texture = texture.clone();
        externally_owned_texture.image.sync = sync;
        let mut frame = frame_for_tests(frame_context_id.clone(), output_size, Transform::Normal);
        assert!(matches!(
            frame.render_texture_from_to(
                &externally_owned_texture,
                full_src,
                full_dst,
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            ),
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        ));
    }

    assert_render_texture_error(
        &texture,
        frame_context_id.clone(),
        Transform::Normal,
        full_src,
        full_dst,
        &[Rectangle::from_size((1, 1).into())],
        &[],
        Transform::Normal,
        1.0,
        "render texture device",
    );
    assert_render_texture_error(
        &texture,
        frame_context_id.clone(),
        Transform::Normal,
        Rectangle::new((1.0, 0.0).into(), (2.0, 3.0).into()),
        full_dst,
        &full_damage,
        &[],
        Transform::Normal,
        1.0,
        "render texture source",
    );
    assert_render_texture_error(
        &texture,
        frame_context_id.clone(),
        Transform::Normal,
        full_src,
        Rectangle::new((7, 0).into(), (2, 6).into()),
        &full_damage,
        &[],
        Transform::Normal,
        1.0,
        "render texture device",
    );

    let mut frame = frame_for_tests(frame_context_id.clone(), output_size, Transform::Normal);
    assert!(
        frame
            .render_texture_from_to(
                &texture,
                full_src,
                Rectangle::new((8, 0).into(), (2, 6).into()),
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .is_ok()
    );

    assert_render_texture_error(
        &texture,
        frame_context_id.clone(),
        Transform::Normal,
        full_src,
        full_dst,
        &full_damage,
        &opaque_region,
        Transform::Normal,
        1.0,
        "render texture device",
    );
    assert_render_texture_error(
        &texture,
        frame_context_id.clone(),
        Transform::Normal,
        full_src,
        full_dst,
        &full_damage,
        &[],
        Transform::_90,
        1.0,
        "render texture device",
    );
    assert_render_texture_error(
        &texture,
        frame_context_id.clone(),
        Transform::_90,
        full_src,
        full_dst,
        &full_damage,
        &[],
        Transform::Normal,
        1.0,
        "render texture transform",
    );
    for alpha in [-0.1, 1.1, f32::NAN] {
        assert_render_texture_error(
            &texture,
            frame_context_id.clone(),
            Transform::Normal,
            full_src,
            full_dst,
            &full_damage,
            &[],
            Transform::Normal,
            alpha,
            "render texture alpha",
        );
    }

    texture.y_inverted = true;
    assert_render_texture_error(
        &texture,
        frame_context_id,
        Transform::Normal,
        full_src,
        full_dst,
        &full_damage,
        &[],
        Transform::Normal,
        1.0,
        "render texture device",
    );
}

#[test]
fn source_to_uv_rect_flips_y_for_y_inverted_textures() {
    assert_eq!(
        source_to_uv_rect(
            (2, 4).into(),
            Rectangle::new((0.0, 1.0).into(), (2.0, 2.0).into()),
            false,
            Transform::Normal,
        ),
        Some(([0.0, 0.25], [1.0, 0.0], [0.0, 0.5]))
    );
    assert_eq!(
        source_to_uv_rect(
            (2, 4).into(),
            Rectangle::new((0.0, 1.0).into(), (2.0, 2.0).into()),
            true,
            Transform::Normal,
        ),
        Some(([0.0, 0.75], [1.0, 0.0], [0.0, -0.5]))
    );
    assert_eq!(
        source_to_uv_rect(
            (2, 4).into(),
            Rectangle::new((0.0, 0.0).into(), (2.0, 1.0).into()),
            true,
            Transform::Normal,
        ),
        Some(([0.0, 1.0], [1.0, 0.0], [0.0, -0.25]))
    );
    assert_eq!(
        source_to_uv_rect(
            (2, 4).into(),
            Rectangle::new((0.0, 2.0).into(), (2.0, 1.0).into()),
            true,
            Transform::Normal,
        ),
        Some(([0.0, 0.5], [1.0, 0.0], [0.0, -0.25]))
    );
}

#[test]
fn source_to_uv_rect_supports_source_transforms() {
    let src = Rectangle::new((1.0, 1.0).into(), (2.0, 2.0).into());
    let texture_size = Size::<i32, BufferCoord>::from((4, 4));

    assert_eq!(
        source_to_uv_rect(texture_size, src, false, Transform::Normal),
        Some(([0.25, 0.25], [0.5, 0.0], [0.0, 0.5]))
    );
    assert_eq!(
        source_to_uv_rect(texture_size, src, false, Transform::_90),
        Some(([0.25, 0.75], [0.0, -0.5], [0.5, 0.0]))
    );
    assert_eq!(
        source_to_uv_rect(texture_size, src, false, Transform::_180),
        Some(([0.75, 0.75], [-0.5, 0.0], [0.0, -0.5]))
    );
    assert_eq!(
        source_to_uv_rect(texture_size, src, false, Transform::_270),
        Some(([0.75, 0.25], [0.0, 0.5], [-0.5, 0.0]))
    );
    assert_eq!(
        source_to_uv_rect(texture_size, src, false, Transform::Flipped),
        Some(([0.75, 0.25], [-0.5, 0.0], [0.0, 0.5]))
    );
    assert_eq!(
        source_to_uv_rect(texture_size, src, false, Transform::Flipped90),
        Some(([0.25, 0.25], [0.0, 0.5], [0.5, 0.0]))
    );
    assert_eq!(
        source_to_uv_rect(texture_size, src, false, Transform::Flipped180),
        Some(([0.25, 0.75], [0.5, 0.0], [0.0, -0.5]))
    );
    assert_eq!(
        source_to_uv_rect(texture_size, src, false, Transform::Flipped270),
        Some(([0.75, 0.75], [0.0, -0.5], [-0.5, 0.0]))
    );
}

#[test]
fn damage_to_scissor_areas_clips_damage_to_destination_and_output() {
    let output_size = Size::<i32, Physical>::from((6, 4));
    let dst = Rectangle::new((1, 1).into(), (4, 2).into());
    let damage = [
        Rectangle::new((0, 0).into(), (3, 2).into()),
        Rectangle::new((3, 1).into(), (4, 4).into()),
        Rectangle::new((0, 3).into(), (1, 1).into()),
        Rectangle::new((2, 2).into(), (0, 1).into()),
    ];
    let scissors = damage_to_scissor_areas(output_size, dst, &damage).unwrap();

    assert_eq!(
        scissors,
        vec![
            vk::Rect2D {
                offset: vk::Offset2D { x: 1, y: 1 },
                extent: vk::Extent2D { width: 3, height: 2 },
            },
            vk::Rect2D {
                offset: vk::Offset2D { x: 4, y: 2 },
                extent: vk::Extent2D { width: 1, height: 1 },
            },
        ]
    );
}

#[test]
fn damage_to_scissor_areas_ignores_non_overlapping_damage() {
    let output_size = Size::<i32, Physical>::from((4, 4));
    let dst = Rectangle::new((1, 1).into(), (2, 2).into());
    let damage = [Rectangle::new((2, 2).into(), (1, 1).into())];

    assert_eq!(
        damage_to_scissor_areas(output_size, dst, &damage),
        Some(Vec::new())
    );
    assert_eq!(
        damage_to_scissor_areas((0, 4).into(), dst, &[Rectangle::from_size((1, 1).into())]),
        None
    );
}

#[test]
fn damage_to_scissor_areas_rejects_overlapping_clipped_damage() {
    let output_size = Size::<i32, Physical>::from((4, 4));
    let dst = Rectangle::new((1, 1).into(), (3, 3).into());
    let damage = [
        Rectangle::new((0, 0).into(), (2, 2).into()),
        Rectangle::new((1, 1).into(), (2, 2).into()),
    ];

    assert_eq!(damage_to_scissor_areas(output_size, dst, &damage), None);
}

#[test]
fn clear_damage_to_clear_areas_clips_and_accepts_overlap() {
    let output_size = Size::<i32, Physical>::from((4, 1));
    let damage = [
        Rectangle::new((0, 0).into(), (2, 1).into()),
        Rectangle::new((1, 0).into(), (4, 1).into()),
        Rectangle::new((5, 0).into(), (1, 1).into()),
    ];

    assert_eq!(
        clear_damage_to_clear_areas(output_size, &damage),
        Some(vec![
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: 2, height: 1 },
            },
            vk::Rect2D {
                offset: vk::Offset2D { x: 1, y: 0 },
                extent: vk::Extent2D { width: 3, height: 1 },
            },
        ])
    );
    assert_eq!(
        clear_damage_to_clear_areas(output_size, &[Rectangle::new((5, 0).into(), (1, 1).into())]),
        Some(Vec::new())
    );
    assert_eq!(
        clear_damage_to_clear_areas((0, 1).into(), &[Rectangle::from_size((1, 1).into())]),
        None
    );
}

#[test]
fn draw_solid_damage_to_clear_areas_clips_and_accepts_overlap() {
    let output_size = Size::<i32, Physical>::from((4, 1));
    let dst = Rectangle::new((1, 0).into(), (4, 1).into());
    let damage = [
        Rectangle::new((0, 0).into(), (2, 1).into()),
        Rectangle::new((1, 0).into(), (3, 1).into()),
    ];

    assert_eq!(
        draw_solid_damage_to_clear_areas(output_size, dst, &damage),
        Some(vec![
            vk::Rect2D {
                offset: vk::Offset2D { x: 1, y: 0 },
                extent: vk::Extent2D { width: 2, height: 1 },
            },
            vk::Rect2D {
                offset: vk::Offset2D { x: 2, y: 0 },
                extent: vk::Extent2D { width: 2, height: 1 },
            },
        ])
    );
    assert_eq!(
        draw_solid_damage_to_clear_areas(output_size, Rectangle::new((5, 0).into(), (1, 1).into()), &damage,),
        Some(Vec::new())
    );
}

#[test]
fn render_texture_damage_to_scissor_areas_splits_opaque_regions() {
    let output_size = Size::<i32, Physical>::from((6, 1));
    let dst = Rectangle::new((1, 0).into(), (4, 1).into());
    let damage = [Rectangle::from_size((4, 1).into())];
    let opaque_regions = [Rectangle::new((1, 0).into(), (2, 1).into())];

    let (non_opaque, opaque) =
        render_texture_damage_to_scissor_areas(output_size, dst, &damage, &opaque_regions, false, 1.0)
            .unwrap();

    assert_eq!(
        non_opaque,
        vec![
            vk::Rect2D {
                offset: vk::Offset2D { x: 1, y: 0 },
                extent: vk::Extent2D { width: 1, height: 1 },
            },
            vk::Rect2D {
                offset: vk::Offset2D { x: 4, y: 0 },
                extent: vk::Extent2D { width: 1, height: 1 },
            },
        ]
    );
    assert_eq!(
        opaque,
        vec![vk::Rect2D {
            offset: vk::Offset2D { x: 2, y: 0 },
            extent: vk::Extent2D { width: 2, height: 1 },
        }]
    );
}

#[test]
fn render_texture_damage_to_scissor_areas_ignores_opaque_regions_with_global_alpha() {
    let output_size = Size::<i32, Physical>::from((6, 1));
    let dst = Rectangle::new((1, 0).into(), (4, 1).into());
    let damage = [Rectangle::from_size((4, 1).into())];
    let opaque_regions = [Rectangle::new((1, 0).into(), (2, 1).into())];

    let (non_opaque, opaque) =
        render_texture_damage_to_scissor_areas(output_size, dst, &damage, &opaque_regions, false, 0.5)
            .unwrap();

    assert_eq!(
        non_opaque,
        vec![vk::Rect2D {
            offset: vk::Offset2D { x: 1, y: 0 },
            extent: vk::Extent2D { width: 4, height: 1 },
        }]
    );
    assert!(opaque.is_empty());
}

#[test]
fn render_texture_damage_to_scissor_areas_treats_implicit_opaque_as_opaque() {
    let output_size = Size::<i32, Physical>::from((6, 1));
    let dst = Rectangle::new((1, 0).into(), (4, 1).into());
    let damage = [Rectangle::from_size((4, 1).into())];
    let opaque_regions = [Rectangle::new((1, 0).into(), (1, 1).into())];

    let (non_opaque, opaque) =
        render_texture_damage_to_scissor_areas(output_size, dst, &damage, &opaque_regions, true, 1.0)
            .unwrap();

    assert!(non_opaque.is_empty());
    assert_eq!(
        opaque,
        vec![vk::Rect2D {
            offset: vk::Offset2D { x: 1, y: 0 },
            extent: vk::Extent2D { width: 4, height: 1 },
        }]
    );
}

#[test]
fn clip_render_texture_draw_area_preserves_visible_destinations() {
    let output_size = Size::<i32, Physical>::from((8, 6));
    let dst = Rectangle::new((1, 2).into(), (4, 3).into());

    let (draw_area, uv_origin, uv_x_axis, uv_y_axis) =
        clip_render_texture_draw_area(output_size, dst, [0.25, 0.5], [0.5, 0.0], [0.0, 0.25])
            .unwrap()
            .unwrap();

    assert_eq!(
        draw_area,
        vk::Rect2D {
            offset: vk::Offset2D { x: 1, y: 2 },
            extent: vk::Extent2D { width: 4, height: 3 },
        }
    );
    assert_eq!(uv_origin, [0.25, 0.5]);
    assert_eq!(uv_x_axis, [0.5, 0.0]);
    assert_eq!(uv_y_axis, [0.0, 0.25]);
}

#[test]
fn clip_render_texture_draw_area_remaps_uvs_for_output_clipping() {
    let output_size = Size::<i32, Physical>::from((4, 3));
    let dst = Rectangle::new((-2, -1).into(), (8, 4).into());

    let (draw_area, uv_origin, uv_x_axis, uv_y_axis) =
        clip_render_texture_draw_area(output_size, dst, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0])
            .unwrap()
            .unwrap();

    assert_eq!(
        draw_area,
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: 4, height: 3 },
        }
    );
    assert_eq!(uv_origin, [0.25, 0.25]);
    assert_eq!(uv_x_axis, [0.5, 0.0]);
    assert_eq!(uv_y_axis, [0.0, 0.75]);
}

#[test]
fn clip_render_texture_draw_area_handles_transformed_uv_axes() {
    let output_size = Size::<i32, Physical>::from((4, 4));
    let dst = Rectangle::new((2, 1).into(), (4, 4).into());

    let (draw_area, uv_origin, uv_x_axis, uv_y_axis) =
        clip_render_texture_draw_area(output_size, dst, [0.0, 1.0], [0.0, -1.0], [1.0, 0.0])
            .unwrap()
            .unwrap();

    assert_eq!(
        draw_area,
        vk::Rect2D {
            offset: vk::Offset2D { x: 2, y: 1 },
            extent: vk::Extent2D { width: 2, height: 3 },
        }
    );
    assert_eq!(uv_origin, [0.0, 1.0]);
    assert_eq!(uv_x_axis, [0.0, -0.5]);
    assert_eq!(uv_y_axis, [0.75, 0.0]);
}

#[test]
fn clip_render_texture_draw_area_reports_offscreen_and_invalid_destinations() {
    let output_size = Size::<i32, Physical>::from((4, 3));

    assert_eq!(
        clip_render_texture_draw_area(
            output_size,
            Rectangle::new((4, 0).into(), (2, 2).into()),
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ),
        Some(None)
    );
    assert_eq!(
        clip_render_texture_draw_area(
            output_size,
            Rectangle::new((0, 0).into(), (0, 2).into()),
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ),
        None
    );
    assert_eq!(
        clip_render_texture_draw_area(
            (0, 3).into(),
            Rectangle::new((0, 0).into(), (2, 2).into()),
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ),
        None
    );
}

#[test]
fn frame_render_texture_reaches_device_lookup_for_supported_narrow_preconditions() {
    let context_id = ContextId::new();
    let mut texture = texture_for_tests((2, 3).into(), Some(Fourcc::Argb8888));
    texture.context_id = context_id;
    let damage = [Rectangle::from_size((8, 6).into())];

    for alpha in [1.0, 0.5] {
        for src in [
            Rectangle::from_size((2.0, 3.0).into()),
            Rectangle::new((1.0, 1.0).into(), (1.0, 2.0).into()),
        ] {
            for dst in [
                Rectangle::from_size((8, 6).into()),
                Rectangle::new((1, 2).into(), (3, 4).into()),
            ] {
                let mut frame = frame_for_tests(texture.context_id.clone(), (8, 6).into(), Transform::Normal);

                assert!(
                    matches!(
                        frame.render_texture_from_to(
                            &texture,
                            src,
                            dst,
                            &damage,
                            &[],
                            Transform::Normal,
                            alpha,
                        ),
                        Err(VulkanError::UnsupportedOperation("render texture device"))
                    ),
                    "alpha {alpha:?} src {src:?} dst {dst:?}"
                );
            }
        }
    }
}

#[test]
fn renderer_filters_roundtrip_as_state() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    assert_eq!(renderer.downscale_filter, TextureFilter::Linear);
    assert_eq!(renderer.upscale_filter, TextureFilter::Linear);

    assert!(renderer.downscale_filter(TextureFilter::Nearest).is_ok());
    assert!(renderer.upscale_filter(TextureFilter::Nearest).is_ok());
    assert_eq!(renderer.downscale_filter, TextureFilter::Nearest);
    assert_eq!(renderer.upscale_filter, TextureFilter::Nearest);
}

#[test]
fn renderer_debug_flags_roundtrip() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    assert!(renderer.debug_flags().is_empty());

    renderer.set_debug_flags(DebugFlags::TINT);
    assert_eq!(renderer.debug_flags(), DebugFlags::TINT);
}

#[test]
fn renderer_cleanup_texture_cache_is_noop() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    assert!(renderer.cleanup_texture_cache().is_ok());
}

#[test]
fn renderer_wait_on_signaled_sync_succeeds() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    assert!(renderer.wait(&SyncPoint::signaled()).is_ok());
}

#[test]
fn wait_interruption_maps_to_sync_interrupted() {
    let sync = SyncPoint::from(InterruptedFence);
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut frame = frame_for_tests(ContextId::new(), (1, 1).into(), Transform::Normal);

    assert!(matches!(renderer.wait(&sync), Err(VulkanError::SyncInterrupted)));
    assert!(matches!(frame.wait(&sync), Err(VulkanError::SyncInterrupted)));
}

const TEXTURED_VERTEX_SHADER_SPIRV: &[u32] = &[
    0x07230203, 0x00010000, 0x0008000b, 0x00000033, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000001f, 0x00000023, 0x0000002f,
    0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00050005,
    0x0000000c, 0x69736f70, 0x6e6f6974, 0x00000073, 0x00030005, 0x00000013, 0x00737675, 0x00060005,
    0x0000001d, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000001d, 0x00000000,
    0x505f6c67, 0x7469736f, 0x006e6f69, 0x00070006, 0x0000001d, 0x00000001, 0x505f6c67, 0x746e696f,
    0x657a6953, 0x00000000, 0x00070006, 0x0000001d, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369,
    0x0065636e, 0x00070006, 0x0000001d, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e,
    0x00030005, 0x0000001f, 0x00000000, 0x00060005, 0x00000023, 0x565f6c67, 0x65747265, 0x646e4978,
    0x00007865, 0x00040005, 0x0000002f, 0x76755f76, 0x00000000, 0x00030047, 0x0000001d, 0x00000002,
    0x00050048, 0x0000001d, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000001d, 0x00000001,
    0x0000000b, 0x00000001, 0x00050048, 0x0000001d, 0x00000002, 0x0000000b, 0x00000003, 0x00050048,
    0x0000001d, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000023, 0x0000000b, 0x0000002a,
    0x00040047, 0x0000002f, 0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000002,
    0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000003,
    0x0004001c, 0x0000000a, 0x00000007, 0x00000009, 0x00040020, 0x0000000b, 0x00000006, 0x0000000a,
    0x0004003b, 0x0000000b, 0x0000000c, 0x00000006, 0x0004002b, 0x00000006, 0x0000000d, 0xbf800000,
    0x0005002c, 0x00000007, 0x0000000e, 0x0000000d, 0x0000000d, 0x0004002b, 0x00000006, 0x0000000f,
    0x40400000, 0x0005002c, 0x00000007, 0x00000010, 0x0000000f, 0x0000000d, 0x0005002c, 0x00000007,
    0x00000011, 0x0000000d, 0x0000000f, 0x0006002c, 0x0000000a, 0x00000012, 0x0000000e, 0x00000010,
    0x00000011, 0x0004003b, 0x0000000b, 0x00000013, 0x00000006, 0x0004002b, 0x00000006, 0x00000014,
    0x00000000, 0x0005002c, 0x00000007, 0x00000015, 0x00000014, 0x00000014, 0x0004002b, 0x00000006,
    0x00000016, 0x40000000, 0x0005002c, 0x00000007, 0x00000017, 0x00000016, 0x00000014, 0x0005002c,
    0x00000007, 0x00000018, 0x00000014, 0x00000016, 0x0006002c, 0x0000000a, 0x00000019, 0x00000015,
    0x00000017, 0x00000018, 0x00040017, 0x0000001a, 0x00000006, 0x00000004, 0x0004002b, 0x00000008,
    0x0000001b, 0x00000001, 0x0004001c, 0x0000001c, 0x00000006, 0x0000001b, 0x0006001e, 0x0000001d,
    0x0000001a, 0x00000006, 0x0000001c, 0x0000001c, 0x00040020, 0x0000001e, 0x00000003, 0x0000001d,
    0x0004003b, 0x0000001e, 0x0000001f, 0x00000003, 0x00040015, 0x00000020, 0x00000020, 0x00000001,
    0x0004002b, 0x00000020, 0x00000021, 0x00000000, 0x00040020, 0x00000022, 0x00000001, 0x00000020,
    0x0004003b, 0x00000022, 0x00000023, 0x00000001, 0x00040020, 0x00000025, 0x00000006, 0x00000007,
    0x0004002b, 0x00000006, 0x00000028, 0x3f800000, 0x00040020, 0x0000002c, 0x00000003, 0x0000001a,
    0x00040020, 0x0000002e, 0x00000003, 0x00000007, 0x0004003b, 0x0000002e, 0x0000002f, 0x00000003,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e,
    0x0000000c, 0x00000012, 0x0003003e, 0x00000013, 0x00000019, 0x0004003d, 0x00000020, 0x00000024,
    0x00000023, 0x00050041, 0x00000025, 0x00000026, 0x0000000c, 0x00000024, 0x0004003d, 0x00000007,
    0x00000027, 0x00000026, 0x00050051, 0x00000006, 0x00000029, 0x00000027, 0x00000000, 0x00050051,
    0x00000006, 0x0000002a, 0x00000027, 0x00000001, 0x00070050, 0x0000001a, 0x0000002b, 0x00000029,
    0x0000002a, 0x00000014, 0x00000028, 0x00050041, 0x0000002c, 0x0000002d, 0x0000001f, 0x00000021,
    0x0003003e, 0x0000002d, 0x0000002b, 0x0004003d, 0x00000020, 0x00000030, 0x00000023, 0x00050041,
    0x00000025, 0x00000031, 0x00000013, 0x00000030, 0x0004003d, 0x00000007, 0x00000032, 0x00000031,
    0x0003003e, 0x0000002f, 0x00000032, 0x000100fd, 0x00010038,
];

const TEXTURED_FRAGMENT_SHADER_SPIRV: &[u32] = &[
    0x07230203, 0x00010000, 0x0008000b, 0x00000014, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00000011, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00050005, 0x00000009, 0x5f74756f, 0x6f6c6f63, 0x00000072, 0x00030005, 0x0000000d,
    0x00786574, 0x00040005, 0x00000011, 0x76755f76, 0x00000000, 0x00040047, 0x00000009, 0x0000001e,
    0x00000000, 0x00040047, 0x0000000d, 0x00000021, 0x00000000, 0x00040047, 0x0000000d, 0x00000022,
    0x00000000, 0x00040047, 0x00000011, 0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006,
    0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009,
    0x00000003, 0x00090019, 0x0000000a, 0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x00000001, 0x00000000, 0x0003001b, 0x0000000b, 0x0000000a, 0x00040020, 0x0000000c, 0x00000000,
    0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000000, 0x00040017, 0x0000000f, 0x00000006,
    0x00000002, 0x00040020, 0x00000010, 0x00000001, 0x0000000f, 0x0004003b, 0x00000010, 0x00000011,
    0x00000001, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x0004003d, 0x0000000b, 0x0000000e, 0x0000000d, 0x0004003d, 0x0000000f, 0x00000012, 0x00000011,
    0x00050057, 0x00000007, 0x00000013, 0x0000000e, 0x00000012, 0x0003003e, 0x00000009, 0x00000013,
    0x000100fd, 0x00010038,
];

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_renderer_builder_initializes_with_first_physical_device() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");
    let api_version = physical_device.api_version();

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device.clone())
        .build()
        .unwrap();
    let other_renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let caps = renderer.capabilities().clone();

    assert!(renderer.is_device_initialized());
    let device = renderer.device.as_ref().unwrap();
    let other_device = other_renderer.device.as_ref().unwrap();
    assert!(device.memory_properties.is_some());
    assert!(
        device
            .find_memory_type_index(u32::MAX, vk::MemoryPropertyFlags::empty())
            .is_ok()
    );
    let graphics_family = device.queue_families.graphics.unwrap();
    let transfer_family = device.queue_families.transfer.unwrap();
    let graphics_queue = device.queues.graphics.as_ref().unwrap();
    let transfer_queue = device.queues.transfer.as_ref().unwrap();
    let graphics_command_pool = device.graphics_command_pool.as_ref().unwrap();
    let transfer_command_pool = device.transfer_command_pool.as_ref().unwrap();
    assert_eq!(graphics_queue.queue_family_index(), graphics_family);
    assert_eq!(transfer_queue.queue_family_index(), transfer_family);
    assert_eq!(graphics_command_pool.queue_family_index(), graphics_family);
    assert_eq!(transfer_command_pool.queue_family_index(), transfer_family);
    let buffer = device
        .create_buffer(
            4096,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap();
    assert_ne!(buffer, vk::Buffer::null());
    let requirements = device.buffer_memory_requirements(buffer).unwrap();
    assert!(requirements.size >= 4096);
    assert_ne!(requirements.memory_type_bits, 0);
    assert!(
        device
            .find_memory_type_index(requirements.memory_type_bits, vk::MemoryPropertyFlags::empty())
            .is_ok()
    );
    let memory_type_index = device
        .find_memory_type_index(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let memory = device
        .allocate_memory(requirements.size, memory_type_index)
        .unwrap();
    assert_ne!(memory, vk::DeviceMemory::null());
    device.bind_buffer_memory(buffer, memory, 0).unwrap();
    let mapped = device.map_memory(memory, 0, 4096).unwrap();
    assert!(!mapped.is_null());
    let pattern = [0x51, 0x52, 0x53, 0x54];
    unsafe {
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), mapped.cast::<u8>(), pattern.len());
        assert_eq!(
            std::slice::from_raw_parts(mapped.cast::<u8>(), pattern.len()),
            pattern
        );
    }
    device.unmap_memory(memory).unwrap();
    device.free_memory(memory).unwrap();
    device.destroy_buffer(buffer).unwrap();
    let host_visible_buffer = device
        .create_host_visible_buffer(
            4096,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap();
    assert_ne!(host_visible_buffer.buffer(), vk::Buffer::null());
    host_visible_buffer.write(&[0x61, 0x62, 0x63, 0x64]).unwrap();
    drop(host_visible_buffer);
    let image = device
        .create_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        )
        .unwrap();
    assert_ne!(image, vk::Image::null());
    let image_requirements = device.image_memory_requirements(image).unwrap();
    assert_ne!(image_requirements.memory_type_bits, 0);
    device.destroy_image(image).unwrap();
    let owned_image = device
        .create_bound_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .unwrap();
    assert_ne!(owned_image.image(), vk::Image::null());
    assert_ne!(owned_image.memory(), vk::DeviceMemory::null());
    assert_eq!(owned_image.extent().width, 1);
    assert_eq!(owned_image.extent().height, 1);
    assert_eq!(owned_image.extent().depth, 1);
    assert_eq!(owned_image.format(), vk::Format::R8G8B8A8_UNORM);
    assert!(owned_image.usage().contains(vk::ImageUsageFlags::TRANSFER_DST));
    assert!(owned_image.usage().contains(vk::ImageUsageFlags::SAMPLED));
    assert_eq!(owned_image.layout().unwrap(), vk::ImageLayout::UNDEFINED);
    assert_eq!(owned_image.external_memory_handle_type(), None);
    assert!(matches!(
        device.create_bound_image(
            vk::Extent3D {
                width: 0,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
        Err(VulkanError::UnsupportedOperation("zero-sized image"))
    ));
    assert!(matches!(
        device.create_bound_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::empty(),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
        Err(VulkanError::UnsupportedOperation("empty image usage"))
    ));
    let copy_buffer = device
        .create_host_visible_buffer(4, vk::BufferUsageFlags::TRANSFER_SRC)
        .unwrap();
    let foreign_copy_buffer = other_device
        .create_host_visible_buffer(4, vk::BufferUsageFlags::TRANSFER_SRC)
        .unwrap();
    assert_eq!(copy_buffer.size(), 4);
    assert!(copy_buffer.usage().contains(vk::BufferUsageFlags::TRANSFER_SRC));
    copy_buffer.write(&[0xff, 0x00, 0x00, 0xff]).unwrap();
    let foreign_color_image = other_device
        .create_bound_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .unwrap();
    assert!(matches!(
        device.create_color_attachment_image_view(&foreign_color_image),
        Err(VulkanError::UnsupportedOperation("image view device"))
    ));
    assert!(matches!(
        device.clear_color_attachment_image(
            &foreign_color_image,
            vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0]
            },
        ),
        Err(VulkanError::UnsupportedOperation("image device"))
    ));
    assert!(matches!(
        device.clear_color_attachment_image_in(
            &foreign_color_image,
            vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0]
            },
            &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: 1, height: 1 },
            }],
        ),
        Err(VulkanError::UnsupportedOperation("image device"))
    ));
    let mut graphics_command_buffer = device.allocate_graphics_command_buffer().unwrap();
    let mut transfer_command_buffer = device.allocate_transfer_command_buffer().unwrap();
    assert_ne!(graphics_command_buffer.handle(), vk::CommandBuffer::null());
    assert_ne!(transfer_command_buffer.handle(), vk::CommandBuffer::null());
    assert_eq!(graphics_command_buffer.queue_family_index(), graphics_family);
    assert_eq!(transfer_command_buffer.queue_family_index(), transfer_family);
    assert!(matches!(
        device.transition_image_layout(
            &mut graphics_command_buffer,
            &owned_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        Err(VulkanError::UnsupportedOperation("command buffer recording"))
    ));
    assert!(matches!(
        device.record_sampled_dmabuf_foreign_acquire_barrier(&mut graphics_command_buffer, &owned_image),
        Err(VulkanError::UnsupportedOperation("command buffer recording"))
    ));
    assert!(matches!(
        device.record_sampled_dmabuf_foreign_release_barrier(&mut graphics_command_buffer, &owned_image),
        Err(VulkanError::UnsupportedOperation("command buffer recording"))
    ));
    assert!(matches!(
        device.record_dmabuf_render_target_foreign_acquire_barrier(
            &mut graphics_command_buffer,
            &owned_image,
            false,
        ),
        Err(VulkanError::UnsupportedOperation("command buffer recording"))
    ));
    assert!(matches!(
        device
            .record_dmabuf_render_target_foreign_release_barrier(&mut graphics_command_buffer, &owned_image),
        Err(VulkanError::UnsupportedOperation("command buffer recording"))
    ));
    assert!(matches!(
        device.end_command_buffer(&mut transfer_command_buffer),
        Err(VulkanError::UnsupportedOperation("command buffer recording"))
    ));
    assert!(!graphics_command_buffer.is_recording());
    assert!(matches!(
        device.submit_transfer_command_buffer_and_wait(&mut transfer_command_buffer),
        Err(VulkanError::UnsupportedOperation("command buffer executable"))
    ));
    device.begin_command_buffer(&mut graphics_command_buffer).unwrap();
    assert!(graphics_command_buffer.is_recording());
    assert!(matches!(
        device.begin_command_buffer(&mut graphics_command_buffer),
        Err(VulkanError::UnsupportedOperation("command buffer recording"))
    ));
    assert!(matches!(
        device.submit_graphics_command_buffer_and_wait(&mut graphics_command_buffer),
        Err(VulkanError::UnsupportedOperation("command buffer executable"))
    ));
    assert!(matches!(
        device.record_sampled_dmabuf_foreign_acquire_barrier(&mut graphics_command_buffer, &owned_image),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.record_sampled_dmabuf_foreign_release_barrier(&mut graphics_command_buffer, &owned_image),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.record_dmabuf_render_target_foreign_acquire_barrier(
            &mut graphics_command_buffer,
            &owned_image,
            false,
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device
            .record_dmabuf_render_target_foreign_release_barrier(&mut graphics_command_buffer, &owned_image),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.submit_sampled_dmabuf_foreign_acquire(&owned_image, None),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.submit_sampled_dmabuf_foreign_release(&owned_image, None),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.release_sampled_dmabuf_to_foreign_general(&owned_image, false),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.release_sampled_dmabuf_to_foreign_general(&owned_image, true),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.submit_dmabuf_render_target_foreign_acquire(&owned_image, false, None),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.submit_dmabuf_render_target_foreign_release(&owned_image, None),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.release_dmabuf_render_target_to_foreign_general(&owned_image, false),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.release_dmabuf_render_target_to_foreign_general(&owned_image, true),
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    ));
    assert!(matches!(
        device.copy_buffer_to_image(
            &mut graphics_command_buffer,
            &foreign_copy_buffer,
            &owned_image,
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        ),
        Err(VulkanError::UnsupportedOperation("command buffer buffer device"))
    ));
    assert!(matches!(
        device.copy_image_to_buffer(
            &mut graphics_command_buffer,
            &owned_image,
            &foreign_copy_buffer,
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        ),
        Err(VulkanError::UnsupportedOperation("command buffer buffer device"))
    ));
    owned_image
        .set_sync_state(VulkanImageSyncState {
            external_ownership: VulkanExternalImageOwnership::ReleasePending,
            ..VulkanImageSyncState::default()
        })
        .unwrap();
    assert!(matches!(
        device.transition_image_layout(
            &mut graphics_command_buffer,
            &owned_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    ));
    owned_image
        .set_sync_state(VulkanImageSyncState::default())
        .unwrap();
    if graphics_family != transfer_family {
        let mut wrong_family_command_buffer = device.allocate_transfer_command_buffer().unwrap();
        device
            .begin_command_buffer(&mut wrong_family_command_buffer)
            .unwrap();
        device
            .end_command_buffer(&mut wrong_family_command_buffer)
            .unwrap();
        assert!(matches!(
            device.submit_graphics_command_buffer_and_wait(&mut wrong_family_command_buffer),
            Err(VulkanError::UnsupportedOperation("command buffer queue family"))
        ));

        let mut transfer_barrier_command_buffer = device.allocate_transfer_command_buffer().unwrap();
        device
            .begin_command_buffer(&mut transfer_barrier_command_buffer)
            .unwrap();
        assert!(matches!(
            device.record_sampled_dmabuf_foreign_acquire_barrier(
                &mut transfer_barrier_command_buffer,
                &owned_image,
            ),
            Err(VulkanError::UnsupportedOperation("command buffer graphics queue"))
        ));
        assert!(matches!(
            device.record_sampled_dmabuf_foreign_release_barrier(
                &mut transfer_barrier_command_buffer,
                &owned_image,
            ),
            Err(VulkanError::UnsupportedOperation("command buffer graphics queue"))
        ));
        assert!(matches!(
            device.record_dmabuf_render_target_foreign_acquire_barrier(
                &mut transfer_barrier_command_buffer,
                &owned_image,
                false,
            ),
            Err(VulkanError::UnsupportedOperation("command buffer graphics queue"))
        ));
        assert!(matches!(
            device.record_dmabuf_render_target_foreign_release_barrier(
                &mut transfer_barrier_command_buffer,
                &owned_image,
            ),
            Err(VulkanError::UnsupportedOperation("command buffer graphics queue"))
        ));
    }
    assert!(matches!(
        device.copy_buffer_to_image(
            &mut graphics_command_buffer,
            &copy_buffer,
            &owned_image,
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        ),
        Err(VulkanError::UnsupportedOperation("image copy layout"))
    ));
    device
        .transition_image_layout(
            &mut graphics_command_buffer,
            &owned_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )
        .unwrap();
    device
        .copy_buffer_to_image(
            &mut graphics_command_buffer,
            &copy_buffer,
            &owned_image,
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        )
        .unwrap();
    drop(copy_buffer);
    device
        .transition_image_layout(
            &mut graphics_command_buffer,
            &owned_image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )
        .unwrap();
    assert_eq!(owned_image.layout().unwrap(), vk::ImageLayout::UNDEFINED);
    device.end_command_buffer(&mut graphics_command_buffer).unwrap();
    assert!(!graphics_command_buffer.is_recording());
    assert!(graphics_command_buffer.is_executable_for_tests());
    assert!(matches!(
        device.end_command_buffer(&mut graphics_command_buffer),
        Err(VulkanError::UnsupportedOperation("command buffer recording"))
    ));
    device
        .submit_graphics_command_buffer_and_wait(&mut graphics_command_buffer)
        .unwrap();
    assert!(graphics_command_buffer.is_submitted_for_tests());
    assert!(matches!(
        device.submit_graphics_command_buffer_and_wait(&mut graphics_command_buffer),
        Err(VulkanError::UnsupportedOperation("command buffer executable"))
    ));
    assert_eq!(
        owned_image.layout().unwrap(),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
    device.begin_command_buffer(&mut transfer_command_buffer).unwrap();
    device.end_command_buffer(&mut transfer_command_buffer).unwrap();
    device
        .submit_transfer_command_buffer_and_wait(&mut transfer_command_buffer)
        .unwrap();
    if caps.external_sync.sync_file_importable && caps.external_sync.sync_file_exportable {
        // SAFETY: `AlreadySignaled` uses Vulkan's special `-1` sync-file import value, not an
        // application-owned fd.
        let acquire_semaphore = unsafe {
            device
                .import_sync_file_semaphore(VulkanSyncFileImport::AlreadySignaled)
                .unwrap()
        };
        let release_semaphore = device.create_exportable_sync_file_semaphore().unwrap();
        assert_eq!(
            acquire_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Signaled
        );
        assert_eq!(
            release_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Unsignaled
        );
        assert!(matches!(
            device.submit_sampled_dmabuf_foreign_acquire(&owned_image, Some(&acquire_semaphore)),
            Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
        ));
        assert_eq!(
            acquire_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Signaled
        );
        assert!(matches!(
            device.submit_sampled_dmabuf_foreign_release(&owned_image, Some(&release_semaphore)),
            Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
        ));
        assert_eq!(
            release_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Unsignaled
        );
        assert!(matches!(
            device
                .submit_dmabuf_render_target_foreign_acquire(&owned_image, false, Some(&acquire_semaphore),),
            Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
        ));
        assert_eq!(
            acquire_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Signaled
        );
        assert!(matches!(
            device.submit_dmabuf_render_target_foreign_release(&owned_image, Some(&release_semaphore)),
            Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
        ));
        assert_eq!(
            release_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Unsignaled
        );
        let mut synchronized_command_buffer = device.allocate_graphics_command_buffer().unwrap();
        device
            .begin_command_buffer(&mut synchronized_command_buffer)
            .unwrap();
        device
            .end_command_buffer(&mut synchronized_command_buffer)
            .unwrap();

        let empty_wait_stage = VulkanSubmitSynchronization::default()
            .wait_sync_file(&acquire_semaphore, vk::PipelineStageFlags::empty());
        assert!(matches!(
            // SAFETY: This intentionally exercises pre-submit validation and returns before any
            // Vulkan queue operation because the wait stage is invalid.
            unsafe {
                device.submit_graphics_command_buffer_and_wait_with_synchronization(
                    &mut synchronized_command_buffer,
                    &empty_wait_stage,
                )
            },
            Err(VulkanError::UnsupportedOperation("semaphore wait stage"))
        ));
        let duplicate_semaphore = VulkanSubmitSynchronization::default()
            .wait_sync_file(&acquire_semaphore, vk::PipelineStageFlags::TOP_OF_PIPE)
            .signal_sync_file(&acquire_semaphore);
        assert!(matches!(
            // SAFETY: This intentionally exercises pre-submit validation and returns before any
            // Vulkan queue operation because the same binary semaphore appears twice.
            unsafe {
                device.submit_graphics_command_buffer_and_wait_with_synchronization(
                    &mut synchronized_command_buffer,
                    &duplicate_semaphore,
                )
            },
            Err(VulkanError::UnsupportedOperation("semaphore submit duplicate"))
        ));

        let synchronization = VulkanSubmitSynchronization::default()
            .wait_sync_file(&acquire_semaphore, vk::PipelineStageFlags::TOP_OF_PIPE)
            .signal_sync_file(&release_semaphore);
        // SAFETY: `acquire_semaphore` was imported with Vulkan's already-signaled `-1` sync-file
        // payload and has not been waited on yet. `release_semaphore` was freshly created for
        // export and has not been signaled yet. `TOP_OF_PIPE` is valid for the graphics queue.
        unsafe {
            device
                .submit_graphics_command_buffer_and_wait_with_synchronization(
                    &mut synchronized_command_buffer,
                    &synchronization,
                )
                .unwrap()
        };
        assert!(synchronized_command_buffer.is_submitted_for_tests());
        assert_eq!(
            acquire_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Unsignaled
        );
        assert_eq!(
            release_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Signaled
        );
        let mut repeated_signal_command_buffer = device.allocate_graphics_command_buffer().unwrap();
        device
            .begin_command_buffer(&mut repeated_signal_command_buffer)
            .unwrap();
        device
            .end_command_buffer(&mut repeated_signal_command_buffer)
            .unwrap();
        let repeated_signal = VulkanSubmitSynchronization::default().signal_sync_file(&release_semaphore);
        assert!(matches!(
            // SAFETY: This intentionally exercises pre-submit payload-state validation and returns
            // before any Vulkan queue operation because `release_semaphore` is already signaled.
            unsafe {
                device.submit_graphics_command_buffer_and_wait_with_synchronization(
                    &mut repeated_signal_command_buffer,
                    &repeated_signal,
                )
            },
            Err(VulkanError::UnsupportedOperation("semaphore signal payload"))
        ));
        // SAFETY: The synchronized submit above completed before this export, and the release
        // semaphore was created for sync-file export on this device.
        let _release_sync_file = unsafe { device.export_sync_file_semaphore(&release_semaphore).unwrap() };
        assert_eq!(
            release_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Unsignaled
        );
        assert!(matches!(
            // SAFETY: This intentionally exercises pre-export payload-state validation and returns
            // before Vulkan because the previous export consumed the sync-file semaphore payload.
            unsafe { device.export_sync_file_semaphore(&release_semaphore) },
            Err(VulkanError::UnsupportedOperation("sync-file semaphore payload"))
        ));
        // SAFETY: The previous export consumed `release_semaphore` back to unsignaled, the command
        // buffer is still executable because the earlier repeated-signal attempt failed before
        // queue submission, and there are no wait semaphores.
        unsafe {
            device
                .submit_graphics_command_buffer_and_wait_with_synchronization(
                    &mut repeated_signal_command_buffer,
                    &repeated_signal,
                )
                .unwrap()
        };
        assert_eq!(
            release_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Signaled
        );

        let async_release_semaphore = device.create_exportable_sync_file_semaphore().unwrap();
        let mut async_release_command_buffer = device.allocate_graphics_command_buffer().unwrap();
        device
            .begin_command_buffer(&mut async_release_command_buffer)
            .unwrap();
        device
            .end_command_buffer(&mut async_release_command_buffer)
            .unwrap();
        let async_release = VulkanSubmitSynchronization::default().signal_sync_file(&async_release_semaphore);
        // SAFETY: The command buffer is executable, the signal semaphore is unsignaled and belongs
        // to this device, and there are no wait semaphores.
        let async_submission = unsafe {
            device
                .submit_graphics_command_buffer_with_synchronization_for_tests(
                    async_release_command_buffer,
                    &async_release,
                )
                .unwrap()
        };
        assert_eq!(
            async_release_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::PendingSignal
        );
        // SAFETY: The signal operation has been submitted and has no unsubmitted dependencies, so
        // SYNC_FD export of the pending signal payload is valid.
        let _async_release_sync_file = unsafe {
            device
                .export_sync_file_semaphore(&async_release_semaphore)
                .unwrap()
        };
        assert_eq!(
            async_release_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Unsignaled
        );
        async_submission.wait_complete().unwrap();
        assert_eq!(
            async_release_semaphore.payload_state_for_tests().unwrap(),
            VulkanSyncFileSemaphorePayloadState::Unsignaled
        );
    }
    drop(owned_image);
    let uploaded_image = device
        .create_uploaded_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            &[0x00, 0xff, 0x00, 0xff, 0xaa],
        )
        .unwrap();
    assert_ne!(uploaded_image.image(), vk::Image::null());
    assert_eq!(uploaded_image.extent().width, 1);
    assert_eq!(uploaded_image.format(), vk::Format::R8G8B8A8_UNORM);
    assert_eq!(
        uploaded_image.layout().unwrap(),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
    let uploaded_view = device.create_image_view(&uploaded_image).unwrap();
    assert_ne!(uploaded_view.handle(), vk::ImageView::null());
    assert_eq!(uploaded_view.image(), uploaded_image.image());
    let uploaded_sampler = device
        .create_sampler(TextureFilter::Nearest, TextureFilter::Linear)
        .unwrap();
    assert_ne!(uploaded_sampler.handle(), vk::Sampler::null());
    assert_eq!(uploaded_sampler.min_filter(), TextureFilter::Nearest);
    assert_eq!(uploaded_sampler.mag_filter(), TextureFilter::Linear);
    drop(uploaded_sampler);
    drop(uploaded_view);
    assert!(matches!(
        device.create_uploaded_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            &[0x00, 0xff, 0x00],
        ),
        Err(VulkanError::UnsupportedOperation("image upload data"))
    ));
    let sampled_image = device
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            &[0x00, 0x00, 0xff, 0xff],
            TextureFilter::Linear,
            TextureFilter::Linear,
        )
        .unwrap();
    assert_ne!(sampled_image.image().image(), vk::Image::null());
    assert_ne!(sampled_image.view().handle(), vk::ImageView::null());
    assert_ne!(sampled_image.sampler().handle(), vk::Sampler::null());
    assert_eq!(
        sampled_image.image().layout().unwrap(),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
    let update_buffer = device
        .create_host_visible_buffer(16, vk::BufferUsageFlags::TRANSFER_SRC)
        .unwrap();
    update_buffer
        .write(&[
            0xff, 0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ])
        .unwrap();
    let mut update_command_buffer = device.allocate_graphics_command_buffer().unwrap();
    device.begin_command_buffer(&mut update_command_buffer).unwrap();
    device
        .transition_image_layout(
            &mut update_command_buffer,
            sampled_image.image(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )
        .unwrap();
    device
        .copy_buffer_region_to_image(
            &mut update_command_buffer,
            &update_buffer,
            sampled_image.image(),
            image_copy_buffer_offset(vk::Format::R8G8B8A8_UNORM, 2, 1, 0).unwrap(),
            2,
            2,
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        )
        .unwrap();
    device
        .transition_image_layout(
            &mut update_command_buffer,
            sampled_image.image(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )
        .unwrap();
    device.end_command_buffer(&mut update_command_buffer).unwrap();
    device
        .submit_graphics_command_buffer_and_wait(&mut update_command_buffer)
        .unwrap();
    assert_eq!(
        sampled_image.image().layout().unwrap(),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (1, 1).into(),
        Fourcc::Abgr8888,
        sampled_image,
        true,
    );
    assert_eq!(texture.width(), 1);
    assert_eq!(texture.height(), 1);
    assert_eq!(texture.format(), Some(Fourcc::Abgr8888));
    assert!(texture.has_sampled_image_for_tests());
    assert!(texture.is_y_inverted_for_tests());
    assert!(caps.device.available);
    let mut expected_required_extensions = Vec::new();
    if caps.external_memory.prerequisites_available {
        expected_required_extensions.extend(VulkanExternalMemoryCapabilities::required_device_extensions(
            api_version,
        ));
    }
    if caps.external_sync.prerequisites_available {
        expected_required_extensions.extend(VulkanExternalSyncCapabilities::required_device_extensions(
            api_version,
        ));
    }
    let expected_enabled_extensions = extension_names_for_tests(expected_required_extensions);
    assert_eq!(caps.device.extensions, expected_enabled_extensions);
    assert_eq!(device.enabled_extensions, expected_enabled_extensions);
    assert_eq!(
        device.external_memory_fns.is_some(),
        caps.external_memory.prerequisites_available
    );
    assert_eq!(
        device.external_sync_fns.is_some(),
        caps.external_sync.prerequisites_available
    );
    assert_eq!(
        caps.import.memory,
        caps.formats.memory_import.iter().next().is_some()
    );
    assert!(!caps.import.dmabuf);
    assert_eq!(caps.export.memory, caps.rendering.offscreen);
    assert!(!caps.export.dmabuf);
    let has_dmabuf_render_target_formats = caps.formats.dmabuf_render_target.iter().next().is_some();
    assert!(!caps.rendering.dmabuf_targets);
    assert!(!caps.rendering.dmabuf_target_modifiers);
    assert_eq!(
        caps.rendering.dmabuf_target_development,
        has_dmabuf_render_target_formats
    );
    assert!(
        caps.formats
            .dmabuf_render_target
            .iter()
            .all(|format| matches!(is_10bit(format.code), Ok(false)))
    );
    assert!(!caps.sync.explicit);
    if let Some(record) = caps
        .formats
        .modifier_records
        .iter()
        .find(|record| record.usages.sampled && record.plane_count > 0)
    {
        let planes = (0..record.plane_count)
            .map(|idx| (idx, idx * 16, 16))
            .collect::<Vec<_>>();
        let dmabuf = dmabuf_with_planes_for_tests(
            (4, 3).into(),
            record.format,
            record.modifier,
            DmabufFlags::empty(),
            &planes,
        );
        let import = VulkanDmabufImportState::from_dmabuf(&dmabuf).unwrap();
        let properties = device.dmabuf_external_image_format_properties(&import).unwrap();
        if let Some(properties) = properties {
            assert_eq!(properties.image_format_properties.max_extent.depth, 1);
            assert!(
                properties
                    .external_memory_properties
                    .compatible_handle_types
                    .contains(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            );
        }
        let candidate = device.dmabuf_import_candidate(&import).unwrap();
        if let Some(candidate) = candidate {
            assert!(candidate.properties.importable);
            assert!(candidate.properties.supports_sampled_import(&import));
            assert_eq!(candidate.dedicated_only, candidate.properties.dedicated_only);
        }
        assert!(!caps.import.dmabuf);
        assert!(caps.formats.dmabuf_import.iter().next().is_none());
    }
    assert!(caps.rendering.offscreen);
    assert!(!caps.rendering.blit);
    assert!(!caps.sync.explicit);
    assert!(
        has_probed_format_support(&caps.formats),
        "expected builder-initialized renderer to expose probed non-import/export format support"
    );
    if caps.import.memory {
        let import_format = renderer.mem_formats().next().unwrap();
        let imported_texture = renderer
            .import_memory(
                &[
                    0xff, 0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff,
                ],
                import_format,
                (2, 2).into(),
                true,
            )
            .unwrap();
        assert_eq!(imported_texture.width(), 2);
        assert_eq!(imported_texture.height(), 2);
        assert_eq!(imported_texture.format(), Some(import_format));
        assert!(imported_texture.has_sampled_image_for_tests());
        assert!(imported_texture.is_y_inverted_for_tests());
        renderer
            .update_memory(
                &imported_texture,
                &[
                    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0x00,
                    0xff,
                ],
                Rectangle::new((1, 1).into(), (1, 1).into()),
            )
            .unwrap();
    }
    let retained_offscreen_target = if let Some(offscreen_format) = caps
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    {
        let mut target = renderer
            .create_offscreen_render_target(offscreen_format, (2, 2).into())
            .unwrap();
        assert_eq!(target.context_id_for_tests(), renderer.context_id());
        assert_eq!(target.width(), 2);
        assert_eq!(target.height(), 2);
        assert_eq!(target.format(), Some(offscreen_format));
        assert_eq!(target.image.source, VulkanImageSource::Offscreen);
        assert_eq!(target.image.layout, VulkanImageLayoutState::Undefined);
        assert!(target.image.usage.color_attachment);
        assert!(target.image.usage.transfer_src);
        assert!(target.image.usage.transfer_dst);
        assert!(target.has_color_image_for_tests());
        let color_image = target.color_image.as_ref().unwrap();
        assert_ne!(color_image.image(), vk::Image::null());
        assert!(
            color_image
                .usage()
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        );
        assert!(color_image.usage().contains(vk::ImageUsageFlags::TRANSFER_SRC));
        assert!(color_image.usage().contains(vk::ImageUsageFlags::TRANSFER_DST));
        assert_eq!(color_image.layout().unwrap(), vk::ImageLayout::UNDEFINED);
        renderer
            .clear_offscreen_render_target(&mut target, Color32F::new(0.2, 0.4, 0.6, 1.0))
            .unwrap();
        assert_eq!(target.image.layout, VulkanImageLayoutState::TransferDst);
        let color_image = target.color_image.as_ref().unwrap();
        assert_eq!(
            color_image.layout().unwrap(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        );
        let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
        assert_eq!(target.image.layout, VulkanImageLayoutState::TransferSrc);
        let color_image = target.color_image.as_ref().unwrap();
        assert_eq!(
            color_image.layout().unwrap(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );
        assert_eq!(readback, [51, 102, 153, 255].repeat(4));
        let public_readback = renderer
            .copy_framebuffer(
                &target,
                Rectangle::new((1, 0).into(), (1, 2).into()),
                offscreen_format,
            )
            .unwrap();
        assert_eq!(public_readback.width(), 1);
        assert_eq!(public_readback.height(), 2);
        assert_eq!(Texture::format(&public_readback), Some(offscreen_format));
        assert!(!public_readback.flipped());
        let public_readback_data = renderer.map_texture(&public_readback).unwrap();
        assert_eq!(public_readback_data, [51, 102, 153, 255].repeat(2));
        {
            let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((2, 2)))];
            let renderer_context = renderer.context_id();
            let mut frame = renderer
                .render(&mut target, (2, 2).into(), Transform::Normal)
                .unwrap();
            assert_eq!(frame.context_id(), renderer_context);
            assert_eq!(frame.output_size(), Size::from((2, 2)));
            assert_eq!(frame.transformation(), Transform::Normal);
            frame
                .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)
                .unwrap();
            assert!(frame.finish().unwrap().is_reached());
        }
        assert_eq!(target.image.layout, VulkanImageLayoutState::ColorAttachment);
        let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
        assert_eq!(readback, [255, 0, 0, 255].repeat(4));
        let public_readback = renderer
            .copy_framebuffer(
                &target,
                Rectangle::new((0, 0).into(), (2, 1).into()),
                offscreen_format,
            )
            .unwrap();
        assert_eq!(
            renderer.map_texture(&public_readback).unwrap(),
            [255, 0, 0, 255].repeat(2)
        );
        Some(target)
    } else {
        None
    };
    assert!(caps.formats.dmabuf_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_export.iter().next().is_none());
    assert!(
        caps.formats
            .dmabuf_render_target
            .iter()
            .all(|format| matches!(is_10bit(format.code), Ok(false)))
    );
    drop(renderer);
    drop(retained_offscreen_target);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_sampled_texture_descriptor_resources_create_with_first_physical_device() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let device = renderer.device.as_ref().unwrap();

    let descriptor_set_layout = device.create_sampled_texture_descriptor_set_layout().unwrap();
    assert_ne!(descriptor_set_layout.handle(), vk::DescriptorSetLayout::null());

    let descriptor_pool = device.create_sampled_texture_descriptor_pool(2).unwrap();
    assert_ne!(descriptor_pool.handle(), vk::DescriptorPool::null());
    assert_eq!(descriptor_pool.max_sets(), 2);

    let sampled_texture_layout = device.create_sampled_texture_pipeline_layout().unwrap();
    assert_ne!(
        sampled_texture_layout.descriptor_set_layout().handle(),
        vk::DescriptorSetLayout::null()
    );
    assert_ne!(
        sampled_texture_layout.pipeline_layout().handle(),
        vk::PipelineLayout::null()
    );

    let sampled_image = Arc::new(
        device
            .create_uploaded_sampled_image(
                vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                vk::Format::R8G8B8A8_UNORM,
                &[0x00, 0x00, 0xff, 0xff],
                TextureFilter::Linear,
                TextureFilter::Linear,
            )
            .unwrap(),
    );
    let descriptor_set = device
        .create_sampled_texture_descriptor_set(
            &descriptor_pool,
            sampled_texture_layout.descriptor_set_layout(),
            Arc::clone(&sampled_image),
        )
        .unwrap();
    assert_ne!(descriptor_set.handle(), vk::DescriptorSet::null());
    assert_eq!(descriptor_set.pool().max_sets(), 2);
    assert_eq!(
        descriptor_set.sampled_image().image().image(),
        sampled_image.image().image()
    );

    let Some(pipeline_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
        })
        .map(|record| super::get_render_vk_format(record.format).unwrap())
    else {
        return;
    };
    // SAFETY: These words were generated from local GLSL shaders by glslangValidator for this
    // runtime smoke test. They contain compatible vertex/fragment `main` entry points, no non-built-in
    // vertex inputs, matching location interfaces, set 0 binding 0 as a combined image sampler, and
    // one color output compatible with the selected color-attachment format.
    let shaders = unsafe {
        VulkanSampledTexturePipelineShaders::from_spirv_unchecked(
            pipeline_format,
            TEXTURED_VERTEX_SHADER_SPIRV,
            TEXTURED_FRAGMENT_SHADER_SPIRV,
        )
    }
    .unwrap();
    let graphics_pipeline = device
        .create_sampled_texture_graphics_pipeline(shaders, true)
        .unwrap();
    assert_ne!(graphics_pipeline.render_pass().handle(), vk::RenderPass::null());
    assert!(graphics_pipeline.blend_enabled());
    assert_ne!(
        graphics_pipeline.layout().pipeline_layout().handle(),
        vk::PipelineLayout::null()
    );
    assert_ne!(graphics_pipeline.pipeline().handle(), vk::Pipeline::null());

    let target = device
        .create_offscreen_color_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            pipeline_format,
        )
        .unwrap();
    device
        .render_sampled_texture_to_color_image(&target, &descriptor_set, &graphics_pipeline)
        .unwrap();
    assert_eq!(
        target.layout().unwrap(),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    let readback = device.read_image_to_tightly_packed_buffer(&target).unwrap();
    assert_eq!(readback, [0, 0, 255, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_builtin_graphics_pipelines_are_cached() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let device = renderer.device.as_ref().unwrap();
    let Some(pipeline_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
        })
        .map(|record| super::get_render_vk_format(record.format).unwrap())
    else {
        return;
    };

    let sampled_blend_a = device
        .builtin_sampled_texture_graphics_pipeline(pipeline_format, true)
        .unwrap();
    let sampled_blend_b = device
        .builtin_sampled_texture_graphics_pipeline(pipeline_format, true)
        .unwrap();
    let sampled_opaque = device
        .builtin_sampled_texture_graphics_pipeline(pipeline_format, false)
        .unwrap();
    assert!(Arc::ptr_eq(&sampled_blend_a, &sampled_blend_b));
    assert!(!Arc::ptr_eq(&sampled_blend_a, &sampled_opaque));

    let load_render_pass = device.single_color_load_render_pass(pipeline_format).unwrap();
    let sampled_layout = device.sampled_texture_pipeline_layout().unwrap();
    assert!(Arc::ptr_eq(sampled_blend_a.render_pass_arc(), &load_render_pass));
    assert!(Arc::ptr_eq(sampled_opaque.render_pass_arc(), &load_render_pass));
    assert!(Arc::ptr_eq(sampled_blend_a.layout_arc(), &sampled_layout));
    assert!(Arc::ptr_eq(sampled_opaque.layout_arc(), &sampled_layout));

    let solid_blend_a = device
        .builtin_solid_color_graphics_pipeline(pipeline_format, true)
        .unwrap();
    let solid_blend_b = device
        .builtin_solid_color_graphics_pipeline(pipeline_format, true)
        .unwrap();
    let solid_opaque = device
        .builtin_solid_color_graphics_pipeline(pipeline_format, false)
        .unwrap();
    assert!(Arc::ptr_eq(&solid_blend_a, &solid_blend_b));
    assert!(!Arc::ptr_eq(&solid_blend_a, &solid_opaque));
    let solid_layout = device.solid_color_pipeline_layout().unwrap();
    assert!(Arc::ptr_eq(solid_blend_a.render_pass_arc(), &load_render_pass));
    assert!(Arc::ptr_eq(solid_opaque.render_pass_arc(), &load_render_pass));
    assert!(Arc::ptr_eq(solid_blend_a.layout_arc(), &solid_layout));
    assert!(Arc::ptr_eq(solid_opaque.layout_arc(), &solid_layout));

    let clear_render_pass_a = device.single_color_clear_render_pass(pipeline_format).unwrap();
    let clear_render_pass_b = device.single_color_clear_render_pass(pipeline_format).unwrap();
    assert!(Arc::ptr_eq(&clear_render_pass_a, &clear_render_pass_b));
    assert!(!Arc::ptr_eq(&clear_render_pass_a, &load_render_pass));

    let descriptor_layout_a = device.sampled_texture_descriptor_set_layout().unwrap();
    let descriptor_layout_b = device.sampled_texture_descriptor_set_layout().unwrap();
    assert!(Arc::ptr_eq(&descriptor_layout_a, &descriptor_layout_b));
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_vulkan_texture_clones_share_sampled_image_sync_state() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(format).unwrap(),
            &[0, 0, 0, 255],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture =
        VulkanTexture::from_sampled_image(renderer.context_id(), (1, 1).into(), format, sampled_image, false);
    let cloned_texture = texture.clone();
    let foreign_sync = VulkanImageSyncState::foreign_known_general_for_dmabuf_import();

    assert_eq!(
        texture.sync_state_for_tests().unwrap(),
        VulkanImageSyncState::default()
    );
    cloned_texture
        .set_sampled_image_sync_state_for_tests(foreign_sync)
        .unwrap();
    assert_eq!(texture.sync_state_for_tests().unwrap(), foreign_sync);
    assert_eq!(cloned_texture.sync_state_for_tests().unwrap(), foreign_sync);

    let local_sync = VulkanImageSyncState {
        external_ownership: VulkanExternalImageOwnership::Local,
        ..VulkanImageSyncState::default()
    };
    texture
        .set_sampled_image_sync_state_for_tests(local_sync)
        .unwrap();
    assert_eq!(texture.sync_state_for_tests().unwrap(), local_sync);
    assert_eq!(cloned_texture.sync_state_for_tests().unwrap(), local_sync);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_draws_uploaded_sampled_image() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[0x00, 0x00, 0xff, 0xff],
            TextureFilter::Linear,
            TextureFilter::Linear,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (1, 1).into(),
        render_format,
        sampled_image,
        false,
    );
    let mut target = renderer
        .create_offscreen_render_target(render_format, (1, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((1, 1)))];
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((1.0, 1.0).into()),
                Rectangle::from_size((1, 1).into()),
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    assert_eq!(target.image.layout, VulkanImageLayoutState::ColorAttachment);
    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(readback, [0, 0, 255, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_draws_imported_memory_texture() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.memory_import
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let texture = renderer
        .import_memory(&[0x00, 0xff, 0x00, 0xff], render_format, (1, 1).into(), false)
        .unwrap();
    let mut target = renderer
        .create_offscreen_render_target(render_format, (1, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((1, 1)))];
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((1.0, 1.0).into()),
                Rectangle::from_size((1, 1).into()),
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(readback, [0, 255, 0, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_draw_solid_respects_destination_local_damage() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.color_attachment
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let mut target = renderer
        .create_offscreen_render_target(render_format, (4, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((4, 1)))];
        let overlapping_damage = [
            Rectangle::new((0, 0).into(), (2, 1).into()),
            Rectangle::new((1, 0).into(), (2, 1).into()),
        ];
        let clipped_damage = [Rectangle::from_size(Size::<i32, Physical>::from((3, 1)))];
        let mut frame = renderer
            .render(&mut target, (4, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)
            .unwrap();
        frame
            .draw_solid(
                Rectangle::new((1, 0).into(), (2, 1).into()),
                &overlapping_damage,
                Color32F::new(0.0, 0.0, 1.0, 1.0),
            )
            .unwrap();
        frame
            .draw_solid(
                Rectangle::new((3, 0).into(), (3, 1).into()),
                &clipped_damage,
                Color32F::new(0.0, 1.0, 0.0, 1.0),
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(
        readback,
        [255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 255, 0, 255]
    );
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_draw_solid_blends_translucent_color() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let mut target = renderer
        .create_offscreen_render_target(render_format, (1, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((1, 1)))];
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)
            .unwrap();
        frame
            .draw_solid(
                Rectangle::from_size((1, 1).into()),
                &full_damage,
                Color32F::new(0.0, 0.0, 0.5, 0.5),
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert!((127..=128).contains(&readback[0]), "red channel {readback:?}");
    assert_eq!(readback[1], 0);
    assert!((127..=128).contains(&readback[2]), "blue channel {readback:?}");
    assert_eq!(readback[3], 255);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_clear_respects_partial_damage() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.color_attachment
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let mut target = renderer
        .create_offscreen_render_target(render_format, (4, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((4, 1)))];
        let partial_damage = [Rectangle::new((1, 0).into(), (2, 1).into())];
        let mut frame = renderer
            .render(&mut target, (4, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)
            .unwrap();
        frame
            .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &partial_damage)
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(
        readback,
        [255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255]
    );
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_public_offscreen_bind_renders_uploaded_sampled_image() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[0x00, 0x00, 0xff, 0xff],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (1, 1).into(),
        render_format,
        sampled_image,
        false,
    );
    let mut target =
        Offscreen::<VulkanRenderTarget<'static>>::create_buffer(&mut renderer, render_format, (1, 1).into())
            .unwrap();
    let mut framebuffer = Bind::bind(&mut renderer, &mut target).unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((1, 1)))];
        let mut frame = renderer
            .render(&mut framebuffer, (1, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((1.0, 1.0).into()),
                Rectangle::from_size((1, 1).into()),
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut framebuffer).unwrap();
    assert_eq!(readback, [0, 0, 255, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_flips_y_inverted_texture() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 1,
                height: 2,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[255, 0, 0, 255, 0, 0, 255, 255],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (1, 2).into(),
        render_format,
        sampled_image,
        true,
    );
    let mut target =
        Offscreen::<VulkanRenderTarget<'static>>::create_buffer(&mut renderer, render_format, (1, 2).into())
            .unwrap();
    let mut framebuffer = Bind::bind(&mut renderer, &mut target).unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((1, 2)))];
        let mut frame = renderer
            .render(&mut framebuffer, (1, 2).into(), Transform::Normal)
            .unwrap();

        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((1.0, 2.0).into()),
                Rectangle::from_size((1, 2).into()),
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut framebuffer).unwrap();
    assert_eq!(readback, [0, 0, 255, 255, 255, 0, 0, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_crops_y_inverted_texture() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 1,
                height: 4,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[
                255, 0, 0, 255, // red
                0, 255, 0, 255, // green
                0, 0, 255, 255, // blue
                255, 255, 255, 255, // white
            ],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (1, 4).into(),
        render_format,
        sampled_image,
        true,
    );
    let mut target =
        Offscreen::<VulkanRenderTarget<'static>>::create_buffer(&mut renderer, render_format, (1, 1).into())
            .unwrap();
    let mut framebuffer = Bind::bind(&mut renderer, &mut target).unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((1, 1)))];
        let mut frame = renderer
            .render(&mut framebuffer, (1, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .render_texture_from_to(
                &texture,
                Rectangle::new((0.0, 0.0).into(), (1.0, 1.0).into()),
                Rectangle::from_size((1, 1).into()),
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut framebuffer).unwrap();
    assert_eq!(readback, [255, 255, 255, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_applies_source_transforms() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 2,
                height: 2,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[
                255, 0, 0, 255, // top-left
                0, 255, 0, 255, // top-right
                0, 0, 255, 255, // bottom-left
                255, 255, 255, 255, // bottom-right
            ],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (2, 2).into(),
        render_format,
        sampled_image,
        false,
    );
    let cases: &[(Transform, &[u8])] = &[
        (
            Transform::_90,
            &[0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 255, 255, 0, 255, 0, 255],
        ),
        (
            Transform::Flipped,
            &[0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255, 0, 0, 255, 255],
        ),
        (
            Transform::Flipped90,
            &[255, 0, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255],
        ),
        (
            Transform::Flipped180,
            &[0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 255, 0, 255],
        ),
        (
            Transform::Flipped270,
            &[255, 255, 255, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 255],
        ),
        (
            Transform::_180,
            &[255, 255, 255, 255, 0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255],
        ),
        (
            Transform::_270,
            &[0, 255, 0, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 0, 255, 255],
        ),
    ];

    for (transform, expected) in cases {
        let mut target = Offscreen::<VulkanRenderTarget<'static>>::create_buffer(
            &mut renderer,
            render_format,
            (2, 2).into(),
        )
        .unwrap();
        let mut framebuffer = Bind::bind(&mut renderer, &mut target).unwrap();

        {
            let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((2, 2)))];
            let mut frame = renderer
                .render(&mut framebuffer, (2, 2).into(), Transform::Normal)
                .unwrap();

            frame
                .render_texture_from_to(
                    &texture,
                    Rectangle::from_size((2.0, 2.0).into()),
                    Rectangle::from_size((2, 2).into()),
                    &full_damage,
                    &[],
                    *transform,
                    1.0,
                )
                .unwrap();
            assert!(frame.finish().unwrap().is_reached());
        }

        let readback = renderer.read_offscreen_render_target(&mut framebuffer).unwrap();
        assert_eq!(&readback, expected, "transform {transform:?}");
    }
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_draws_to_destination_rectangle() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[0x00, 0x00, 0xff, 0xff],
            TextureFilter::Linear,
            TextureFilter::Linear,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (1, 1).into(),
        render_format,
        sampled_image,
        false,
    );
    let mut target = renderer
        .create_offscreen_render_target(render_format, (2, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((2, 1)))];
        let mut frame = renderer
            .render(&mut target, (2, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)
            .unwrap();
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((1.0, 1.0).into()),
                Rectangle::new((1, 0).into(), (1, 1).into()),
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(readback, [255, 0, 0, 255, 0, 0, 255, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_respects_partial_damage_scissor() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 2,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[0, 0, 255, 255, 0, 0, 255, 255],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (2, 1).into(),
        render_format,
        sampled_image,
        false,
    );
    let mut target = renderer
        .create_offscreen_render_target(render_format, (4, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((4, 1)))];
        let partial_damage = [Rectangle::new((1, 0).into(), (1, 1).into())];
        let mut frame = renderer
            .render(&mut target, (4, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)
            .unwrap();
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((2.0, 1.0).into()),
                Rectangle::new((1, 0).into(), (2, 1).into()),
                &partial_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(
        readback,
        [255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 0, 255]
    );
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_clips_partial_destination() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 4,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[
                255, 0, 0, 255, // clipped left
                0, 255, 0, 255, // visible x = 0
                0, 0, 255, 255, // visible x = 1
                255, 255, 255, 255, // visible x = 2
            ],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (4, 1).into(),
        render_format,
        sampled_image,
        false,
    );
    let mut target = renderer
        .create_offscreen_render_target(render_format, (3, 1).into())
        .unwrap();

    {
        let clear_damage = [Rectangle::from_size(Size::<i32, Physical>::from((3, 1)))];
        let texture_damage = [Rectangle::from_size(Size::<i32, Physical>::from((4, 1)))];
        let mut frame = renderer
            .render(&mut target, (3, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .clear(Color32F::new(0.0, 0.0, 0.0, 1.0), &clear_damage)
            .unwrap();
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((4.0, 1.0).into()),
                Rectangle::new((-1, 0).into(), (4, 1).into()),
                &texture_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(readback, [0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_draws_cropped_source() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 2,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[255, 0, 0, 255, 0, 0, 255, 255],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (2, 1).into(),
        render_format,
        sampled_image,
        false,
    );
    let mut target = renderer
        .create_offscreen_render_target(render_format, (2, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((2, 1)))];
        let mut frame = renderer
            .render(&mut target, (2, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .render_texture_from_to(
                &texture,
                Rectangle::new((1.0, 0.0).into(), (1.0, 1.0).into()),
                Rectangle::from_size((2, 1).into()),
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(readback, [0, 0, 255, 255, 0, 0, 255, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_blends_global_alpha() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[0, 0, 255, 255],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (1, 1).into(),
        render_format,
        sampled_image,
        false,
    );
    let mut target = renderer
        .create_offscreen_render_target(render_format, (1, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((1, 1)))];
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)
            .unwrap();
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((1.0, 1.0).into()),
                Rectangle::from_size((1, 1).into()),
                &full_damage,
                &[],
                Transform::Normal,
                0.5,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert!((127..=128).contains(&readback[0]), "red channel {readback:?}");
    assert_eq!(readback[1], 0);
    assert!((127..=128).contains(&readback[2]), "blue channel {readback:?}");
    assert_eq!(readback[3], 255);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_accepts_opaque_regions() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(render_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 2,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(render_format).unwrap(),
            &[0, 0, 255, 255, 0, 0, 128, 128],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (2, 1).into(),
        render_format,
        sampled_image,
        false,
    );
    let mut target = renderer
        .create_offscreen_render_target(render_format, (4, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((4, 1)))];
        let texture_damage = [Rectangle::from_size(Size::<i32, Physical>::from((2, 1)))];
        let opaque_regions = [Rectangle::from_size(Size::<i32, Physical>::from((1, 1)))];
        let mut frame = renderer
            .render(&mut target, (4, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)
            .unwrap();
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((2.0, 1.0).into()),
                Rectangle::new((1, 0).into(), (2, 1).into()),
                &texture_damage,
                &opaque_regions,
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(&readback[0..4], &[255, 0, 0, 255]);
    assert_eq!(&readback[4..8], &[0, 0, 255, 255]);
    assert!((127..=128).contains(&readback[8]), "red channel {readback:?}");
    assert_eq!(readback[9], 0);
    assert!((127..=128).contains(&readback[10]), "blue channel {readback:?}");
    assert_eq!(readback[11], 255);
    assert_eq!(&readback[12..16], &[255, 0, 0, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_frame_render_texture_forces_opaque_alpha_formats() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let mut renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let Some(target_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Abgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
                && record.usages.color_attachment
                && record.usages.color_attachment_blend
                && record.usages.transfer_src
                && record.usages.transfer_dst
        })
        .map(|record| record.format)
    else {
        return;
    };
    let Some(texture_format) = renderer
        .capabilities()
        .formats
        .records
        .iter()
        .find(|record| {
            record.format == Fourcc::Xbgr8888
                && record.tiling == VulkanFormatTiling::Optimal
                && record.usages.sampled
        })
        .map(|record| record.format)
    else {
        return;
    };

    let sampled_image = renderer
        .device
        .as_ref()
        .unwrap()
        .create_uploaded_sampled_image(
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            super::get_render_vk_format(texture_format).unwrap(),
            &[0, 0, 255, 0],
            TextureFilter::Nearest,
            TextureFilter::Nearest,
        )
        .unwrap();
    let texture = VulkanTexture::from_sampled_image(
        renderer.context_id(),
        (1, 1).into(),
        texture_format,
        sampled_image,
        false,
    );
    let mut target = renderer
        .create_offscreen_render_target(target_format, (1, 1).into())
        .unwrap();

    {
        let full_damage = [Rectangle::from_size(Size::<i32, Physical>::from((1, 1)))];
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();

        frame
            .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full_damage)
            .unwrap();
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((1.0, 1.0).into()),
                Rectangle::from_size((1, 1).into()),
                &full_damage,
                &[],
                Transform::Normal,
                1.0,
            )
            .unwrap();
        assert!(frame.finish().unwrap().is_reached());
    }

    let readback = renderer.read_offscreen_render_target(&mut target).unwrap();
    assert_eq!(readback, [0, 0, 255, 255]);
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_format_discovery_finds_device_backed_formats_without_import_export() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let external_memory = VulkanExternalMemoryCapabilities::discover(&physical_device);
    let caps = VulkanFormatCapabilities::discover(&physical_device, &external_memory).unwrap();

    assert!(
        has_probed_format_support(&caps),
        "expected at least one probed renderer-internal format record"
    );
    assert_eq!(
        caps.memory_import.iter().next().is_some(),
        caps.records.iter().any(|record| record.usages.memory_import)
    );
    assert!(caps.dmabuf_import.iter().next().is_none());
    assert!(caps.dmabuf_export.iter().next().is_none());
    assert!(
        caps.dmabuf_render_target
            .iter()
            .all(|format| matches!(is_10bit(format.code), Ok(false)))
    );
}

#[test]
#[ignore = "Default Vulkan device selection is not implemented yet"]
fn future_renderer_initialization_enables_device_capabilities() {
    let renderer = VulkanRenderer::new().unwrap();
    let caps = renderer.capabilities();

    assert!(renderer.is_device_initialized());
    assert!(caps.device.available);
    assert!(!caps.device.extensions.is_empty());
}

#[test]
#[ignore = "Vulkan dmabuf import/export is not implemented yet"]
fn future_wayland_dmabuf_advertisement_has_importable_pairs() {
    let renderer = VulkanRenderer::new().unwrap();
    let caps = renderer.capabilities();

    assert!(caps.import.dmabuf);
    assert!(caps.formats.dmabuf_import.iter().next().is_some());
    if caps.export.dmabuf {
        assert!(caps.formats.dmabuf_export.iter().next().is_some());
    }
}

#[test]
#[ignore = "Vulkan explicit synchronization is not implemented yet"]
fn future_explicit_sync_capability_requires_exportable_sync() {
    let renderer = VulkanRenderer::new().unwrap();
    let caps = renderer.capabilities();

    assert!(caps.sync.explicit);
}
