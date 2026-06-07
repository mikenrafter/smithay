use std::{ffi::CStr, fs::File, marker::PhantomData, os::unix::io::OwnedFd, sync::Arc};

use ash::{ext, khr, vk};

use crate::backend::allocator::{
    Format, Fourcc, Modifier,
    dmabuf::{Dmabuf, DmabufFlags},
};
use crate::backend::renderer::sync::Interrupted;
use crate::backend::renderer::{
    Bind, Color32F, DebugFlags, ExportMem, Frame, ImportDma, ImportMem, Offscreen, Renderer, Texture,
    TextureMapping, sync::Fence,
};
use crate::backend::vulkan::{Instance, PhysicalDevice, version::Version};
use crate::utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform};

use super::capabilities::{
    format_usage_from_features, linear_tiling_supported, modifier_record_from_properties,
    should_query_modifier_properties,
};
use super::device::{
    VulkanDeviceState, VulkanSampledTexturePipelineShaders, VulkanShaderSpirv, find_memory_type_index,
    image_copy_buffer_offset, image_copy_required_size, image_layout_transition, select_queue_families,
    tightly_packed_image_size, vulkan_filter,
};
use super::error::vulkan_api_result_invalidates_context;
use super::image::{
    VulkanDmabufImportState, VulkanDmabufPlane, VulkanExternalMemoryHandleType, VulkanExternalMemoryState,
    VulkanImageLayoutState, VulkanImageSource, VulkanImageState, VulkanImageSyncState, VulkanImageUsage,
    clear_damage_to_clear_areas, damage_to_scissor_areas, draw_solid_damage_to_clear_areas,
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
    assert!(!caps.rendering.blit);
    assert!(!caps.rendering.render_target_10bit);
    assert!(!caps.rendering.render_target_fp16);
    assert!(!caps.sync.explicit);
    assert!(!caps.color.color_transform_hooks);
    assert!(!caps.color.hdr_ready_targets);
    assert!(!caps.external_memory.dmabuf_external_memory);
    assert!(!caps.external_memory.external_memory_fd);
    assert!(!caps.external_memory.drm_format_modifiers);
    assert!(!caps.external_memory.image_format_list);
    assert!(!caps.external_memory.prerequisites_available);
}

#[test]
fn vulkan_format_capability_matrix_defaults_empty() {
    let caps = VulkanRendererCapabilities::default();
    assert!(caps.formats.records.is_empty());
    assert!(caps.formats.modifier_records.is_empty());
    assert!(caps.formats.memory_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_export.iter().next().is_none());
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
    assert!(!caps.sync.explicit);
    assert!(caps.formats.memory_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_export.iter().next().is_none());
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
        khr::image_format_list::NAME,
    ];
    let caps =
        VulkanExternalMemoryCapabilities::from_device_extension_support(Version::VERSION_1_1, |name| {
            supported.iter().any(|supported| *supported == name)
        });

    assert!(caps.dmabuf_external_memory);
    assert!(caps.external_memory_fd);
    assert!(caps.drm_format_modifiers);
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
    assert!(renderer_caps.formats.dmabuf_import.iter().next().is_none());
    assert!(renderer_caps.formats.dmabuf_export.iter().next().is_none());
}

#[test]
fn external_memory_capability_discovery_requires_modifier_dependency() {
    let supported = [
        ext::external_memory_dma_buf::NAME,
        khr::external_memory_fd::NAME,
        ext::image_drm_format_modifier::NAME,
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
fn external_memory_required_device_extensions_track_api_version_dependencies() {
    assert_eq!(
        VulkanExternalMemoryCapabilities::required_device_extensions(Version::VERSION_1_1),
        vec![
            ext::external_memory_dma_buf::NAME,
            khr::external_memory_fd::NAME,
            ext::image_drm_format_modifier::NAME,
            khr::image_format_list::NAME,
        ]
    );
    assert_eq!(
        VulkanExternalMemoryCapabilities::required_device_extensions(Version::VERSION_1_2),
        vec![
            ext::external_memory_dma_buf::NAME,
            khr::external_memory_fd::NAME,
            ext::image_drm_format_modifier::NAME,
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
    assert!(renderer_caps.formats.dmabuf_import.iter().next().is_none());
    assert!(renderer_caps.formats.dmabuf_export.iter().next().is_none());
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
    assert_eq!(device.queue_families.graphics, None);
    assert_eq!(device.queue_families.transfer, None);
    assert!(device.queues.graphics.is_none());
    assert!(device.queues.transfer.is_none());
    assert!(device.graphics_command_pool.is_none());
    assert!(device.transfer_command_pool.is_none());
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
fn public_dmabuf_bind_is_explicitly_unsupported() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let mut dmabuf = dmabuf_for_tests();

    let formats = <VulkanRenderer as Bind<Dmabuf>>::supported_formats(&renderer)
        .expect("Vulkan dmabuf render targets have an explicit format set");
    assert!(formats.iter().next().is_none());
    assert!(matches!(
        <VulkanRenderer as Bind<Dmabuf>>::bind(&mut renderer, &mut dmabuf),
        Err(VulkanError::UnsupportedOperation("dmabuf render target"))
    ));
}

#[test]
fn public_dmabuf_import_is_explicitly_unsupported() {
    let mut renderer = VulkanRenderer::new_scaffold_for_tests();
    let dmabuf = dmabuf_for_tests();
    let format = Format {
        code: Fourcc::Abgr8888,
        modifier: Modifier::Invalid,
    };

    assert!(renderer.dmabuf_formats().iter().next().is_none());
    assert!(!renderer.has_dmabuf_format(format));
    assert!(matches!(
        renderer.import_dmabuf(&dmabuf, None),
        Err(VulkanError::UnsupportedOperation("dmabuf import"))
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
        "render texture destination",
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
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let caps = renderer.capabilities().clone();

    assert!(renderer.is_device_initialized());
    let device = renderer.device.as_ref().unwrap();
    assert!(device.memory_properties.is_some());
    assert!(
        device
            .find_memory_type_index(u32::MAX, vk::MemoryPropertyFlags::empty())
            .is_ok()
    );
    assert!(device.graphics_command_pool.is_some());
    assert!(device.transfer_command_pool.is_some());
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
    assert_eq!(copy_buffer.size(), 4);
    assert!(copy_buffer.usage().contains(vk::BufferUsageFlags::TRANSFER_SRC));
    copy_buffer.write(&[0xff, 0x00, 0x00, 0xff]).unwrap();
    let mut graphics_command_buffer = device.allocate_graphics_command_buffer().unwrap();
    let mut transfer_command_buffer = device.allocate_transfer_command_buffer().unwrap();
    assert_ne!(graphics_command_buffer.handle(), vk::CommandBuffer::null());
    assert_ne!(transfer_command_buffer.handle(), vk::CommandBuffer::null());
    device.begin_command_buffer(&mut graphics_command_buffer).unwrap();
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
    device
        .submit_graphics_command_buffer_and_wait(&mut graphics_command_buffer)
        .unwrap();
    assert_eq!(
        owned_image.layout().unwrap(),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    );
    drop(owned_image);
    device.begin_command_buffer(&mut transfer_command_buffer).unwrap();
    device.end_command_buffer(&mut transfer_command_buffer).unwrap();
    device
        .submit_transfer_command_buffer_and_wait(&mut transfer_command_buffer)
        .unwrap();
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
    let expected_enabled_extensions = if caps.external_memory.prerequisites_available {
        extension_names_for_tests(VulkanExternalMemoryCapabilities::required_device_extensions(
            api_version,
        ))
    } else {
        Vec::new()
    };
    assert_eq!(caps.device.extensions, expected_enabled_extensions);
    assert_eq!(device.enabled_extensions, expected_enabled_extensions);
    assert_eq!(
        caps.import.memory,
        caps.formats.memory_import.iter().next().is_some()
    );
    assert!(!caps.import.dmabuf);
    assert_eq!(caps.export.memory, caps.rendering.offscreen);
    assert!(!caps.export.dmabuf);
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
