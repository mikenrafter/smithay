use std::{marker::PhantomData, os::unix::io::OwnedFd};

use ash::vk;

use crate::backend::allocator::Fourcc;
use crate::backend::renderer::sync::Interrupted;
use crate::backend::renderer::{Color32F, DebugFlags, Frame, Renderer, Texture, sync::Fence};
use crate::backend::vulkan::{Instance, PhysicalDevice, version::Version};
use crate::utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform};

use super::capabilities::{format_usage_from_features, linear_tiling_supported};
use super::device::{
    VulkanDeviceState, find_memory_type_index, image_layout_transition, select_queue_families,
};
use super::error::vulkan_api_result_invalidates_context;
use super::image::{
    VulkanDmabufImportState, VulkanDmabufPlane, VulkanExternalMemoryHandleType, VulkanExternalMemoryState,
    VulkanImageLayoutState, VulkanImageSource, VulkanImageState, VulkanImageSyncState, VulkanImageUsage,
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
        _renderer: PhantomData,
        _target: PhantomData,
    }
}

fn texture_for_tests(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> VulkanTexture {
    VulkanTexture {
        image: VulkanImageState::new_for_tests(size, format),
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
    assert!(!caps.export.dmabuf);
    assert!(!caps.export.modifiers);
    assert!(!caps.rendering.offscreen);
    assert!(!caps.rendering.blit);
    assert!(!caps.rendering.render_target_10bit);
    assert!(!caps.rendering.render_target_fp16);
    assert!(!caps.sync.explicit);
    assert!(!caps.color.color_transform_hooks);
    assert!(!caps.color.hdr_ready_targets);
}

#[test]
fn vulkan_format_capability_matrix_defaults_empty() {
    let caps = VulkanRendererCapabilities::default();
    assert!(caps.formats.records.is_empty());
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
fn vulkan_device_state_placeholder_starts_empty() {
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
    assert!(matches!(
        image_layout_transition(
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        ),
        Err(VulkanError::UnsupportedOperation("image layout transition"))
    ));
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
fn vulkan_renderer_builder_is_scaffold_only() {
    assert!(matches!(
        VulkanRenderer::builder().build(),
        Err(VulkanError::VulkanUnavailable)
    ));
}

#[test]
fn initialized_device_capabilities_do_not_enable_renderer_operations() {
    let caps = VulkanRendererCapabilities::for_initialized_device(&[]);

    assert!(caps.device.available);
    assert!(!caps.device.multi_gpu);
    assert!(caps.device.extensions.is_empty());
    assert!(!caps.import.memory);
    assert!(!caps.import.dmabuf);
    assert!(!caps.export.dmabuf);
    assert!(!caps.rendering.offscreen);
    assert!(!caps.rendering.blit);
    assert!(!caps.sync.explicit);
    assert!(caps.formats.records.is_empty());
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
fn vulkan_image_state_placeholder_tracks_target_metadata() {
    let target = VulkanRenderTarget::new_for_tests((2, 3).into(), Some(Fourcc::Argb8888));
    assert_eq!(target.width(), 2);
    assert_eq!(target.height(), 3);
    assert_eq!(target.size(), Size::from((2, 3)));
    assert_eq!(Texture::format(&target), Some(Fourcc::Argb8888));
    assert_eq!(target.image.source, VulkanImageSource::Uninitialized);
    assert_eq!(target.image.layout, VulkanImageLayoutState::Undefined);
    assert_eq!(target.image.usage, VulkanImageUsage::default());
    assert_eq!(target.image.sync, VulkanImageSyncState::default());
}

#[test]
fn vulkan_texture_placeholder_tracks_texture_metadata() {
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
fn external_memory_metadata_placeholder_tracks_planes() {
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
    };

    assert_eq!(import.memory.handle_type, VulkanExternalMemoryHandleType::Dmabuf);
    assert_eq!(import.memory.format, Fourcc::Argb8888);
    assert_eq!(
        import.memory.modifier,
        crate::backend::allocator::Modifier::Linear
    );
    assert_eq!(import.memory.planes.len(), 1);
    assert!(!import.memory.disjoint);
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
fn frame_metadata_accessors_report_scaffold_state() {
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
fn frame_finish_is_signaled_without_submitted_work() {
    let frame = frame_for_tests(ContextId::new(), (1, 1).into(), Transform::Normal);
    let sync = frame.finish().unwrap();

    assert!(sync.is_reached());
    assert!(!sync.contains_fence());
    assert!(!sync.is_exportable());
}

#[test]
fn frame_render_texture_rejects_every_transform_cleanly() {
    let texture = texture_for_tests((2, 3).into(), Some(Fourcc::Argb8888));
    let transforms = [
        Transform::Normal,
        Transform::_90,
        Transform::_180,
        Transform::_270,
        Transform::Flipped,
        Transform::Flipped90,
        Transform::Flipped180,
        Transform::Flipped270,
    ];
    let damage = [Rectangle::from_size((8, 6).into())];

    for transform in transforms {
        let mut frame = frame_for_tests(ContextId::new(), (8, 6).into(), Transform::Normal);

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
                Err(VulkanError::UnsupportedOperation(_))
            ),
            "transform {transform:?}"
        );
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

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_renderer_builder_initializes_with_first_physical_device() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let renderer = VulkanRenderer::builder()
        .with_physical_device(physical_device)
        .build()
        .unwrap();
    let caps = renderer.capabilities();

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
    let mut graphics_command_buffer = device.allocate_graphics_command_buffer().unwrap();
    let mut transfer_command_buffer = device.allocate_transfer_command_buffer().unwrap();
    assert_ne!(graphics_command_buffer.handle(), vk::CommandBuffer::null());
    assert_ne!(transfer_command_buffer.handle(), vk::CommandBuffer::null());
    device.begin_command_buffer(&mut graphics_command_buffer).unwrap();
    device
        .transition_image_layout(
            &mut graphics_command_buffer,
            &owned_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )
        .unwrap();
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
    assert!(caps.device.available);
    assert!(caps.device.extensions.is_empty());
    assert!(!caps.import.memory);
    assert!(!caps.import.dmabuf);
    assert!(!caps.export.dmabuf);
    assert!(!caps.rendering.offscreen);
    assert!(!caps.rendering.blit);
    assert!(!caps.sync.explicit);
    assert!(
        has_probed_format_support(&caps.formats),
        "expected builder-initialized renderer to expose probed non-import/export format support"
    );
    assert!(caps.formats.memory_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_import.iter().next().is_none());
    assert!(caps.formats.dmabuf_export.iter().next().is_none());
}

#[test]
#[ignore = "requires a working Vulkan loader and physical device"]
fn runtime_format_discovery_finds_device_backed_formats_without_import_export() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("No physical devices");

    let caps = VulkanFormatCapabilities::discover(&physical_device).unwrap();

    assert!(
        has_probed_format_support(&caps),
        "expected at least one probed renderer-internal format record"
    );
    assert!(caps.memory_import.iter().next().is_none());
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
