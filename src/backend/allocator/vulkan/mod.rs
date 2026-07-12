//! Module for Buffers created using Vulkan.
//!
//! The [`VulkanAllocator`] type implements the [`Allocator`] trait and [`VulkanImage`] implements [`Buffer`].
//! A [`VulkanImage`] may be exported as a [dmabuf](super::dmabuf).
//!
//! The Vulkan allocator supports up to Vulkan 1.3.
//!
//! The Vulkan allocator requires the following device extensions (and their dependencies):
//! - `VK_EXT_image_drm_format_modifier`
//! - `VK_EXT_external_memory_dmabuf`
//! - `VK_KHR_external_memory_fd`
//!
//! Additionally the Vulkan allocator may enable the following extensions if available:
//! - `VK_EXT_4444_formats`
//!
//! To get the required extensions a device must support to use the Vulkan allocator, use
//! [`VulkanAllocator::required_extensions`].

pub mod format;

use std::{
    ffi::CStr,
    fmt,
    os::unix::io::{FromRawFd, OwnedFd},
    sync::{Arc, Mutex, Weak, mpsc},
};

use ash::{ext, khr, vk};
use bitflags::bitflags;
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
use tracing::instrument;

#[cfg(feature = "backend_drm")]
use crate::backend::drm::DrmNode;
use crate::{
    backend::{
        allocator::dmabuf::DmabufFlags,
        vulkan::{PhysicalDevice, version::Version},
    },
    utils::{Buffer as BufferCoord, Size},
};

use super::{
    Allocator, Buffer,
    dmabuf::{AsDmabuf, Dmabuf, MAX_PLANES, WeakDmabuf},
};

bitflags! {
    /// Flags to indicate the intended usage for a buffer.
    ///
    /// The usage flags may influence whether a buffer with a specific format and modifier combination may
    /// successfully be imported by a graphics api when it is exported as a [`Dmabuf`].
    ///
    /// For example, if a [`Buffer`] is used for scan-out, it is only necessary to specify the color
    /// attachment usage. However, if the exported buffer is used by a client and intended to be rendered to,
    /// then it is possible that the import will fail because the buffer was not allocated with right usage
    /// for the format and modifier combination.
    ///
    /// The default usage set when creating a [`VulkanAllocator`] guarantees that an exported buffer may be
    /// imported successfully with the same usage.
    ///
    /// If you need to allocate buffers with different usages dynamically, then you may use
    /// [`VulkanAllocator::create_buffer_with_usage`].
    ///
    /// [`VulkanAllocator::is_format_supported`] can check if the combination of format, modifier and usage
    /// flags are supported.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ImageUsageFlags: vk::Flags {
        /// The image may be the source of a transfer command.
        ///
        /// This allows the content of the exported buffer to be downloaded from the GPU.
        const TRANSFER_SRC = vk::ImageUsageFlags::TRANSFER_SRC.as_raw();

        /// The image may be the destination of a transfer command.
        ///
        /// This allows the content of the exported buffer to be modified by a memory upload to the GPU.
        const TRANSFER_DST = vk::ImageUsageFlags::TRANSFER_DST.as_raw();

        /// Image may be sampled in a shader.
        ///
        /// This should be used if the exported buffer will be used as a texture.
        const SAMPLED = vk::ImageUsageFlags::SAMPLED.as_raw();

        /// The image may be used in a color attachment.
        ///
        /// This should be used if the exported buffer will be rendered to.
        const COLOR_ATTACHMENT = vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw();
    }
}

/// Error type for [`VulkanAllocator`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The allocator could not be created.
    ///
    /// This can occur for a few reasons:
    /// - No suitable queue family was found.
    #[error("could not create allocator")]
    Setup,

    /// The size specified to create the buffer was too small.
    ///
    /// The size must be greater than `0 x 0` pixels.
    #[error("invalid buffer size")]
    InvalidSize,

    /// The specified format is not supported.
    ///
    /// This error may occur for a few reasons:
    /// 1. There is no equivalent Vulkan format for the specified format code.
    /// 2. The driver does not support any of the specified modifiers for the format code.
    /// 3. The driver does support some of the format, but the size of the buffer requested is too large.
    /// 4. The usage is empty
    #[error("format is not supported")]
    UnsupportedFormat,

    /// The image memory requirements did not allow any usable memory type.
    #[error("memory type is not supported")]
    UnsupportedMemoryType,

    /// Some error from the Vulkan driver.
    #[error(transparent)]
    Vk(#[from] vk::Result),
}

/// An allocator which uses Vulkan to create buffers.
pub struct VulkanAllocator {
    formats: Vec<FormatEntry>,
    images: Vec<ImageInner>,
    image_release_states: Vec<(ImageInner, VulkanAllocatorDmabufReleaseState)>,
    default_usage: ImageUsageFlags,
    remaining_allocations: u32,
    extension_fns: ExtensionFns,
    dropped_recv: mpsc::Receiver<ImageInner>,
    dropped_sender: mpsc::Sender<ImageInner>,
    phd: PhysicalDevice,
    #[cfg(feature = "backend_drm")]
    node: Option<DrmNode>,
    device: Arc<ash::Device>,
    foreign_queue_family_enabled: bool,
    release_queue_family_index: u32,
    release_queue: vk::Queue,
    release_command_pool: Option<vk::CommandPool>,
    release_submission_failed: bool,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanAllocatorDmabufForeignReleaseEvidence {
    image: ImageInner,
    dmabuf: WeakDmabuf,
    _private: (),
}

impl VulkanAllocatorDmabufForeignReleaseEvidence {
    #[allow(dead_code)]
    pub(crate) fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    /// # Safety
    ///
    /// The returned evidence is for unit-test routing only and must not be used for a real Vulkan
    /// acquire/import operation.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) unsafe fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            image: ImageInner {
                image: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
            },
            dmabuf: dmabuf.weak(),
            _private: (),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VulkanAllocatorDmabufReleaseState {
    FreshLocalUndefined,
    ReleasedForeignGeneral,
}

fn image_release_state_index(
    states: &[(ImageInner, VulkanAllocatorDmabufReleaseState)],
    image: ImageInner,
) -> Option<usize> {
    states.iter().position(|(inner, _state)| *inner == image)
}

fn dmabuf_was_exported_from_image(exports: &[WeakDmabuf], dmabuf: &Dmabuf) -> bool {
    exports
        .iter()
        .any(|export| export.upgrade().as_ref() == Some(dmabuf))
}

fn validate_allocator_release_dmabuf_metadata(
    image_size: Size<i32, BufferCoord>,
    image_format: DrmFormat,
    image_plane_count: u32,
    dmabuf: &Dmabuf,
) -> Result<(), VulkanAllocatorForeignReleaseError> {
    if image_plane_count != 1 {
        return Err(VulkanAllocatorForeignReleaseError::MissingCapability(
            "Vulkan allocator dmabuf foreign release planes",
        ));
    }
    if dmabuf.num_planes() != image_plane_count as usize {
        return Err(VulkanAllocatorForeignReleaseError::InvalidState(
            "Vulkan allocator dmabuf foreign release planes",
        ));
    }
    if image_size != dmabuf.size() || image_format != dmabuf.format() {
        return Err(VulkanAllocatorForeignReleaseError::InvalidState(
            "Vulkan allocator dmabuf foreign release identity",
        ));
    }

    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum VulkanAllocatorForeignReleaseError {
    #[error("foreign Vulkan allocator image")]
    ForeignImage,
    #[error("invalid allocator image release state: {0}")]
    InvalidState(&'static str),
    #[error("missing capability: {0}")]
    MissingCapability(&'static str),
    #[error(transparent)]
    Vk(#[from] vk::Result),
}

impl fmt::Debug for VulkanAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanAllocator")
            .field("formats", &self.formats)
            .field("images", &self.images)
            .field("image_release_states", &self.image_release_states)
            .field("default_usage", &self.default_usage)
            .field("remaining_allocations", &self.remaining_allocations)
            .field("dropped_recv", &self.dropped_recv)
            .field("dropped_sender", &self.dropped_sender)
            .field("phd", &self.phd)
            .field("foreign_queue_family_enabled", &self.foreign_queue_family_enabled)
            .field("release_queue_family_index", &self.release_queue_family_index)
            .field("release_queue", &self.release_queue)
            .field("release_command_pool", &self.release_command_pool)
            .field("release_submission_failed", &self.release_submission_failed)
            .finish()
    }
}

impl VulkanAllocator {
    /// Maximum supported version instance version that may be used with the allocator.
    pub const MAX_INSTANCE_VERSION: Version = Version::VERSION_1_3;

    /// Returns the list of device extensions required by the Vulkan allocator.
    ///
    /// This function may return a different list for each [`PhysicalDevice`], meaning each device should be
    /// filtered using it's own call to this function.
    pub fn required_extensions(phd: &PhysicalDevice) -> Vec<&'static CStr> {
        // Always required extensions
        let mut extensions = vec![
            ext::image_drm_format_modifier::NAME,
            ext::external_memory_dma_buf::NAME,
            khr::external_memory_fd::NAME,
        ];

        if phd.api_version() < Version::VERSION_1_2 {
            // VK_EXT_image_drm_format_modifier requires VK_KHR_image_format_list.
            // VK_KHR_image_format_list is part of the core API in Vulkan 1.2
            extensions.push(khr::image_format_list::NAME);
        }

        // Optional extensions:

        // VK_EXT_4444_formats is part of the core API in Vulkan 1.3. Although not always supported
        // (see the 1.3 features to enable)
        if phd.api_version() < Version::VERSION_1_3 {
            // In 1.2 and below, the device must support the extension to use it.
            if phd.has_device_extension(ext::_4444_formats::NAME) {
                extensions.push(ext::_4444_formats::NAME);
            }
        }

        extensions
    }

    /// Creates a [`VulkanAllocator`].
    ///
    /// # Panics
    ///
    /// - If the version of instance which created the [`PhysicalDevice`] is higher than [`VulkanAllocator::MAX_INSTANCE_VERSION`].
    /// - If the default [`ImageUsageFlags`] are empty.
    #[instrument(err, skip(phd), fields(physical_device = phd.name()))]
    pub fn new(phd: &PhysicalDevice, default_usage: ImageUsageFlags) -> Result<VulkanAllocator, Error> {
        // Panic if the instance version is too high
        if phd.instance().api_version() > Self::MAX_INSTANCE_VERSION {
            panic!("Exceeded maximum instance api version for VulkanAllocator (1.3 max)")
        }

        // VUID-VkPhysicalDeviceImageFormatInfo2-usage-requiredbitmask
        // At least one image usage flag must be specified.
        if default_usage.is_empty() {
            panic!("Default usage flags for allocator are empty")
        }

        // Get required extensions
        let mut extensions = Self::required_extensions(phd);
        let foreign_queue_family_enabled = phd.has_device_extension(ext::queue_family_foreign::NAME);
        if foreign_queue_family_enabled {
            extensions.push(ext::queue_family_foreign::NAME);
        }
        let extension_pointers = extensions.iter().copied().map(CStr::as_ptr).collect::<Vec<_>>();

        // We don't actually submit any commands to the queue, but Vulkan requires that we create devices with
        // at least one queue (VUID-VkDeviceCreateInfo-queueCreateInfoCount-arraylength).
        let queue_families = unsafe {
            phd.instance()
                .handle()
                .get_physical_device_queue_family_properties(phd.handle())
        };

        let queue_family_index = queue_families
            .iter()
            // Find a queue with transfer
            .position(|properties| properties.queue_flags.contains(vk::QueueFlags::TRANSFER))
            // If there is no transfer queue family, then try graphics (which must support transfer)
            .or_else(|| {
                queue_families
                    .iter()
                    .position(|properties| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            })
            .ok_or(Error::Setup)?;

        // TODO: Enable 4444 formats features
        let queue_create_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index as u32)
            .queue_priorities(&[0.0])];
        let create_info = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&extension_pointers)
            .queue_create_infos(&queue_create_info);

        let instance = phd.instance().handle();
        let device = unsafe { instance.create_device(phd.handle(), &create_info, None) }?;
        // SAFETY: `queue_create_info` above creates one queue for `queue_family_index`, so queue
        // index 0 is in range for this logical device.
        let release_queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

        // Load extension functions
        let extension_fns = ExtensionFns {
            ext_image_format_modifier: ext::image_drm_format_modifier::Device::new(instance, &device),
            khr_external_memory_fd: khr::external_memory_fd::Device::new(instance, &device),
        };

        let (dropped_sender, dropped_recv) = mpsc::channel();

        #[cfg(feature = "backend_drm")]
        let node = phd
            .render_node()
            .ok()
            .flatten()
            .or_else(|| phd.primary_node().ok().flatten());

        let mut allocator = VulkanAllocator {
            formats: Vec::new(),
            images: Vec::new(),
            image_release_states: Vec::new(),
            default_usage,
            remaining_allocations: phd.limits().max_memory_allocation_count,
            extension_fns,
            dropped_recv,
            dropped_sender,
            phd: phd.clone(),
            #[cfg(feature = "backend_drm")]
            node,
            device: Arc::new(device),
            foreign_queue_family_enabled,
            release_queue_family_index: queue_family_index as u32,
            release_queue,
            release_command_pool: None,
            release_submission_failed: false,
        };

        allocator.init_formats();

        Ok(allocator)
    }

    /// Returns whether this allocator supports the specified format with the usage flags.
    pub fn is_format_supported(&self, format: DrmFormat, usage: ImageUsageFlags) -> bool {
        // VUID-VkPhysicalDeviceImageFormatInfo2-usage-requiredbitmask
        // At least one image usage flag must be specified.
        if usage.is_empty() {
            return false;
        }

        if self.format_plane_count(format).is_none() {
            return false;
        }

        // TODO: Check if the extents are also valid?
        // Vulkan states a maximum extent size for images.
        // This may also be useful as a function on Allocator.
        matches!(
            unsafe { self.get_format_info(format, vk::ImageUsageFlags::from_raw(usage.bits())) },
            Ok(Some(_))
        )
    }

    /// Try to create a buffer with the given dimensions, pixel format and usage flags.
    ///
    /// This may return [`Err`] for one of the following reasons:
    /// - The `usage` is empty.
    /// - The `fourcc` format is not supported.
    /// - All of the allowed `modifiers` are not supported.
    /// - The size of the buffer is too large for the `usage`, `fourcc` format or `modifiers`.
    /// - The `fourcc` format and `modifiers` do not support the specified usage.
    #[instrument(level = "trace", err)]
    #[profiling::function]
    pub fn create_buffer_with_usage(
        &mut self,
        width: u32,
        height: u32,
        fourcc: DrmFourcc,
        modifiers: &[DrmModifier],
        usage: ImageUsageFlags,
    ) -> Result<VulkanImage, Error> {
        self.cleanup();

        let vk_format = format::get_vk_format(fourcc).ok_or(Error::UnsupportedFormat)?;
        let vk_usage = vk::ImageUsageFlags::from_raw(usage.bits());

        // VUID-VkImageCreateInfo-extent-00944, VUID-VkImageCreateInfo-extent-00945
        if width == 0 || height == 0 {
            return Err(Error::InvalidSize);
        }

        // VUID-VkPhysicalDeviceImageFormatInfo2-usage-requiredbitmask
        // At least one image usage flag must be specified.
        if usage.is_empty() {
            return Err(Error::UnsupportedFormat);
        }

        // VUID-VkImageCreateInfo-usage-00964, VUID-VkImageCreateInfo-usage-00965
        if usage.contains(ImageUsageFlags::COLOR_ATTACHMENT) {
            let limits = self.phd.limits();

            if width > limits.max_framebuffer_width || height > limits.max_framebuffer_height {
                return Err(Error::InvalidSize);
            }
        }

        // Filter out any format + modifier combinations that are not supported
        let modifiers = self.filter_modifiers(width, height, vk_usage, fourcc, modifiers)?;

        // VUID-VkImageDrmFormatModifierListCreateInfoEXT-drmFormatModifierCount-arraylength
        if modifiers.is_empty() {
            return Err(Error::UnsupportedFormat);
        }

        unsafe { self.create_image(width, height, vk_format, vk_usage, fourcc, &modifiers[..]) }
    }

    /// Returns the [`PhysicalDevice`] this allocator was created with.
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.phd
    }

    /// Release an allocator-owned dmabuf image to foreign ownership in `GENERAL` layout.
    ///
    /// This validation-stage helper is used by renderer loopback development to establish the first
    /// Vulkan ownership/layout contract for a fresh allocator image. It only succeeds once for an
    /// allocator-owned single-plane image that has not been released before. The returned evidence is
    /// not public API and does not advertise generic dmabuf import or render-target support.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no foreign consumer has accessed, acquired, released, or otherwise
    /// transitioned the exported dmabuf between [`VulkanImage::export`] and this call. This helper
    /// records a release from the allocator's fresh local `UNDEFINED` state; that is only valid while
    /// the exported image has not been used through another API or queue-family owner.
    #[allow(dead_code)]
    pub(crate) unsafe fn release_dmabuf_to_foreign_general(
        &mut self,
        image: &VulkanImage,
        dmabuf: &Dmabuf,
    ) -> Result<VulkanAllocatorDmabufForeignReleaseEvidence, VulkanAllocatorForeignReleaseError> {
        let Some(device) = image.device.upgrade() else {
            return Err(VulkanAllocatorForeignReleaseError::ForeignImage);
        };
        if !Arc::ptr_eq(&device, &self.device) || !self.images.contains(&image.inner) {
            return Err(VulkanAllocatorForeignReleaseError::ForeignImage);
        }
        if self.release_submission_failed {
            return Err(VulkanAllocatorForeignReleaseError::InvalidState(
                "Vulkan allocator dmabuf foreign release submission state",
            ));
        }
        let state_index = image_release_state_index(&self.image_release_states, image.inner)
            .ok_or(VulkanAllocatorForeignReleaseError::ForeignImage)?;
        if self.image_release_states[state_index].1 != VulkanAllocatorDmabufReleaseState::FreshLocalUndefined
        {
            return Err(VulkanAllocatorForeignReleaseError::InvalidState(
                "Vulkan allocator dmabuf foreign release state",
            ));
        }
        validate_allocator_release_dmabuf_metadata(
            image.size(),
            image.format(),
            image.format_plane_count,
            dmabuf,
        )?;
        let exports = image.exports.lock().map_err(|_| {
            VulkanAllocatorForeignReleaseError::InvalidState(
                "Vulkan allocator dmabuf foreign release identity",
            )
        })?;
        let exported_from_image = dmabuf_was_exported_from_image(&exports, dmabuf);
        if !exported_from_image {
            return Err(VulkanAllocatorForeignReleaseError::InvalidState(
                "Vulkan allocator dmabuf foreign release identity",
            ));
        }
        if !self.foreign_queue_family_enabled {
            return Err(VulkanAllocatorForeignReleaseError::MissingCapability(
                "Vulkan allocator dmabuf foreign queue family",
            ));
        }

        if self.release_command_pool.is_none() {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.release_queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            self.release_command_pool = Some(unsafe {
                // SAFETY: `release_queue_family_index` was selected from this physical device's
                // queue-family properties and included in this logical device's queue creation. The
                // resulting command buffers are reserved for this allocator's release queue. No
                // allocation callbacks are used.
                self.device.create_command_pool(&command_pool_info, None)?
            });
        }
        let release_command_pool =
            self.release_command_pool
                .ok_or(VulkanAllocatorForeignReleaseError::MissingCapability(
                    "Vulkan allocator dmabuf foreign release command pool",
                ))?;

        self.submit_dmabuf_foreign_release(image.inner.image, release_command_pool)?;
        self.image_release_states[state_index].1 = VulkanAllocatorDmabufReleaseState::ReleasedForeignGeneral;

        Ok(VulkanAllocatorDmabufForeignReleaseEvidence {
            image: image.inner,
            dmabuf: dmabuf.weak(),
            _private: (),
        })
    }

    fn submit_dmabuf_foreign_release(
        &mut self,
        image: vk::Image,
        command_pool: vk::CommandPool,
    ) -> Result<(), VulkanAllocatorForeignReleaseError> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe {
            // SAFETY: `command_pool` was created from this logical device and is owned by this
            // allocator. The allocation request asks for one primary command buffer.
            self.device.allocate_command_buffers(&allocate_info)?
        };
        let command_buffer = command_buffers[0];
        let mut free_command_buffer = true;
        let mut submitted = false;

        let result = (|| {
            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                // SAFETY: `command_buffer` was allocated from `command_pool` above and is in the
                // initial state. The command pool is allocator-private.
                self.device.begin_command_buffer(command_buffer, &begin_info)?;
            }

            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(self.release_queue_family_index)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe {
                // SAFETY: `command_buffer` is recording. The barrier releases this allocator-owned,
                // single-plane color image from the allocator queue family to FOREIGN ownership and
                // transitions it from its fresh `UNDEFINED` layout to `GENERAL` for the renderer-side
                // acquire contract. No memory dependencies are required for previous writes because
                // this allocator path has not submitted writes to the image.
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
                // SAFETY: `command_buffer` is recording and all commands have been recorded.
                self.device.end_command_buffer(command_buffer)?;
            }

            let command_buffers = [command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            unsafe {
                // SAFETY: `release_queue` belongs to `release_queue_family_index`, and
                // `command_buffer` was allocated from a command pool for the same family. The submit
                // has no semaphore waits/signals and uses no fence; completion is made synchronous by
                // `queue_wait_idle` below before the command buffer is freed.
                self.device
                    .queue_submit(self.release_queue, &submit_info, vk::Fence::null())?;
                submitted = true;
                if let Err(err) = self.device.queue_wait_idle(self.release_queue) {
                    self.release_submission_failed = true;
                    return Err(VulkanAllocatorForeignReleaseError::from(err));
                }
            }

            Ok(())
        })();

        if result.is_ok() {
            unsafe {
                // SAFETY: Queue completion was observed before freeing this command buffer. The
                // command buffer was allocated from `command_pool` above.
                self.device.free_command_buffers(command_pool, &[command_buffer]);
            }
            free_command_buffer = false;
        }

        if free_command_buffer && !submitted {
            unsafe {
                // SAFETY: On errors before a successful submit, the command buffer cannot be in
                // flight. The command buffer was allocated from `command_pool` above.
                self.device.free_command_buffers(command_pool, &[command_buffer]);
            }
        }

        result
    }
}

impl Allocator for VulkanAllocator {
    type Buffer = VulkanImage;
    type Error = Error;

    fn create_buffer(
        &mut self,
        width: u32,
        height: u32,
        fourcc: DrmFourcc,
        modifiers: &[DrmModifier],
    ) -> Result<VulkanImage, Self::Error> {
        self.create_buffer_with_usage(width, height, fourcc, modifiers, self.default_usage)
    }
}

fn find_memory_type_index(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
) -> Result<u32, Error> {
    let mut fallback = None;

    for index in 0..memory_properties.memory_type_count {
        let memory_type_supported = (memory_type_bits & (1u32 << index)) != 0;
        if !memory_type_supported {
            continue;
        }

        let properties = memory_properties.memory_types[index as usize].property_flags;
        if properties.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            return Ok(index);
        }

        fallback.get_or_insert(index);
    }

    fallback.ok_or(Error::UnsupportedMemoryType)
}

fn supports_dma_buf_export(properties: vk::ExternalMemoryProperties) -> bool {
    properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
        && properties
            .compatible_handle_types
            .contains(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
}

fn requires_dedicated_allocation(properties: vk::ExternalMemoryProperties) -> bool {
    properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
}

impl Drop for VulkanAllocator {
    fn drop(&mut self) {
        unsafe {
            if self.release_submission_failed {
                let _ = self.device.device_wait_idle();
            }
            if let Some(command_pool) = self.release_command_pool.take() {
                // SAFETY: Successful allocator release submissions wait for queue idle before
                // freeing their command buffer. If a submitted wait failed, the allocator is marked
                // invalid for later release evidence and teardown first attempts device idle; if
                // that also fails, this is treated as device-lost teardown. No allocation callbacks
                // were used when creating the pool.
                self.device.destroy_command_pool(command_pool, None);
            }

            for image in &self.images {
                self.device.destroy_image(image.image, None);
                self.device.free_memory(image.memory, None);
            }

            self.device.destroy_device(None);
        }
    }
}

/// Vulkan image object.
///
/// This type implements [`Buffer`] and the underlying image may be exported as a dmabuf.
pub struct VulkanImage {
    inner: ImageInner,
    width: u32,
    height: u32,
    format: DrmFormat,
    allocation_info: VulkanImageAllocationInfo,
    #[cfg(feature = "backend_drm")]
    node: Option<DrmNode>,
    /// The number of planes the image has for dmabuf export.
    format_plane_count: u32,
    khr_external_memory_fd: khr::external_memory_fd::Device,
    dropped_sender: mpsc::Sender<ImageInner>,
    device: Weak<ash::Device>,
    exports: Arc<Mutex<Vec<WeakDmabuf>>>,
}

impl fmt::Debug for VulkanImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("allocation_info", &self.allocation_info)
            .field("inner", &self.inner)
            .finish()
    }
}

impl Buffer for VulkanImage {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn size(&self) -> Size<i32, BufferCoord> {
        (self.width as i32, self.height as i32).into()
    }

    fn format(&self) -> DrmFormat {
        self.format
    }
}

impl AsDmabuf for VulkanImage {
    type Error = ExportError;

    #[profiling::function]
    fn export(&self) -> Result<Dmabuf, Self::Error> {
        let device = self.device.upgrade().ok_or(ExportError::AllocatorDestroyed)?;

        // Implementation may be broken if the plane count is wrong.
        if self.format_plane_count == 0 {
            return Err(ExportError::Failed);
        }

        assert!(
            self.format_plane_count as usize <= MAX_PLANES,
            "Vulkan implementation reported too many planes"
        );

        let create_info = vk::MemoryGetFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            // VUID-VkMemoryGetFdInfoKHR-handleType-00671: Memory was allocated with DMA_BUF_EXT
            .memory(self.inner.memory);

        let fd = unsafe { self.khr_external_memory_fd.get_memory_fd(&create_info) }?;
        // SAFETY: `vkGetMemoryFdKHR` creates a new file descriptor owned by the caller.
        let fd = Arc::new(unsafe { OwnedFd::from_raw_fd(fd) });
        let mut builder = Dmabuf::builder(
            self.size(),
            self.format().code,
            self.format().modifier,
            DmabufFlags::empty(),
        );

        for idx in 0..self.format_plane_count {
            // get_image_subresource_layout only gets the layout of one memory plane. This mask specifies
            // which plane should the layout be obtained for.
            let aspect_mask = match idx {
                0 => vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                1 => vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
                2 => vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
                3 => vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
                _ => unreachable!(),
            };

            // VUID-vkGetImageSubresourceLayout-image-02270: All allocate images are created with drm tiling
            let subresource = vk::ImageSubresource::default().aspect_mask(aspect_mask);
            let layout = unsafe { device.get_image_subresource_layout(self.inner.image, subresource) };
            let (offset, stride) = dmabuf_plane_layout(layout).ok_or(ExportError::Failed)?;
            builder.add_plane(fd.clone(), offset, stride);
        }

        #[cfg(feature = "backend_drm")]
        if let Some(node) = self.node {
            builder.set_node(node);
        }

        let dmabuf = builder.build().unwrap();
        self.exports
            .lock()
            .map_err(|_| ExportError::Failed)?
            .push(dmabuf.weak());

        Ok(dmabuf)
    }
}

impl Drop for VulkanImage {
    fn drop(&mut self) {
        let _ = self.dropped_sender.send(self.inner);
    }
}

/// The error type for exporting a [`VulkanImage`] as a [`Dmabuf`].
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// The image could not export a dmabuf since the allocator has been destroyed.
    #[error("allocator has been destroyed")]
    AllocatorDestroyed,

    /// The allocator could not export a dmabuf for an implementation dependent reason.
    #[error("could not export a dmabuf")]
    Failed,

    /// Vulkan API error.
    #[error(transparent)]
    Vk(#[from] vk::Result),
}

fn dmabuf_plane_layout(layout: vk::SubresourceLayout) -> Option<(u32, u32)> {
    let offset = u32::try_from(layout.offset).ok()?;
    let stride = u32::try_from(layout.row_pitch).ok()?;

    if stride == 0 {
        return None;
    }

    Some((offset, stride))
}

fn dmabuf_plane_count(plane_count: u32) -> Option<u32> {
    (1..=MAX_PLANES as u32)
        .contains(&plane_count)
        .then_some(plane_count)
}

fn image_format_properties_support_extent(
    properties: vk::ImageFormatProperties,
    width: u32,
    height: u32,
) -> bool {
    let max_extent = properties.max_extent;

    // VUID-VkImageCreateInfo-extent-02252
    max_extent.width >= width
        // VUID-VkImageCreateInfo-extent-02253
        && max_extent.height >= height
        // VUID-VkImageCreateInfo-extent-02254
        // VUID-VkImageCreateInfo-extent-00946
        // VUID-VkImageCreateInfo-imageType-00957
        && max_extent.depth >= 1
        // VUID-VkImageCreateInfo-samples-02258
        && properties.sample_counts.contains(vk::SampleCountFlags::TYPE_1)
}

fn ensure_allocation_available(remaining_allocations: u32) -> Result<(), Error> {
    if remaining_allocations == 0 {
        Err(Error::Vk(vk::Result::ERROR_TOO_MANY_OBJECTS))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ImageInner {
    // TODO: image usage?
    image: vk::Image,
    /// The first entry will always have a non-null device memory handle.
    ///
    /// The other three entries may be a null handle or valid device memory (the latter with disjoint dmabufs).
    memory: vk::DeviceMemory,
}

#[derive(Debug)]
struct FormatEntry {
    format: DrmFormat,
    modifier_properties: vk::DrmFormatModifierPropertiesEXT,
}

#[derive(Debug, Clone, Copy)]
struct ExternalImageFormatInfo {
    image_format_properties: vk::ImageFormatProperties,
    dedicated_only: bool,
}

#[derive(Clone, Copy)]
struct VulkanImageAllocationInfo {
    modifier_tiling_features: vk::FormatFeatureFlags,
    memory_type_bits: u32,
    memory_size: vk::DeviceSize,
    memory_alignment: vk::DeviceSize,
    memory_type_index: u32,
    memory_type_flags: vk::MemoryPropertyFlags,
    external_dedicated_only: bool,
    memory_dedicated_required: bool,
    memory_dedicated_preferred: bool,
    dedicated: bool,
}

impl fmt::Debug for VulkanImageAllocationInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanImageAllocationInfo")
            .field("modifier_tiling_features", &self.modifier_tiling_features)
            .field("memory_type_bits", &self.memory_type_bits)
            .field("memory_size", &self.memory_size)
            .field("memory_alignment", &self.memory_alignment)
            .field("memory_type_index", &self.memory_type_index)
            .field("memory_type_flags", &self.memory_type_flags)
            .field("external_dedicated_only", &self.external_dedicated_only)
            .field("memory_dedicated_required", &self.memory_dedicated_required)
            .field("memory_dedicated_preferred", &self.memory_dedicated_preferred)
            .field("dedicated", &self.dedicated)
            .finish()
    }
}

struct ExtensionFns {
    /// Functions to get DRM format information.
    ///
    /// These are required, and therefore you may assume this functionality is available.
    ext_image_format_modifier: ext::image_drm_format_modifier::Device,
    /// Functions used for dmabuf import and export.
    ///
    /// If this is [`Some`], then the allocator will support dmabuf import and export operations.
    khr_external_memory_fd: khr::external_memory_fd::Device,
}

impl VulkanAllocator {
    fn init_formats(&mut self) {
        for &fourcc in format::known_formats() {
            let vk_format = format::get_vk_format(fourcc).unwrap();
            let modifier_properties = self
                .phd
                .get_format_modifier_properties(vk_format)
                .expect("The Vulkan allocator requires VK_EXT_image_drm_format_modifier");

            for modifier_properties in modifier_properties {
                self.formats.push(FormatEntry {
                    format: DrmFormat {
                        code: fourcc,
                        modifier: DrmModifier::from(modifier_properties.drm_format_modifier),
                    },
                    modifier_properties,
                });
            }
        }
    }

    /// Returns whether the format + modifier combination and the usage flags are supported.
    unsafe fn get_format_info(
        &self,
        format: DrmFormat,
        usage: vk::ImageUsageFlags,
    ) -> Result<Option<ExternalImageFormatInfo>, vk::Result> {
        let vk_format = format::get_vk_format(format.code);

        match vk_format {
            Some(vk_format) => {
                let mut external_image_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let mut image_drm_format_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
                    .drm_format_modifier(format.modifier.into())
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let format_info = vk::PhysicalDeviceImageFormatInfo2::default()
                    .format(vk_format)
                    .ty(vk::ImageType::TYPE_2D)
                    .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                    .usage(usage)
                    .flags(vk::ImageCreateFlags::empty())
                    // VUID-VkPhysicalDeviceImageFormatInfo2-tiling-02249
                    .push_next(&mut external_image_info)
                    .push_next(&mut image_drm_format_info);
                let mut external_properties = vk::ExternalImageFormatProperties::default();
                let mut image_format_properties =
                    vk::ImageFormatProperties2::default().push_next(&mut external_properties);

                // VUID-vkGetPhysicalDeviceImageFormatProperties-tiling-02248: Must use vkGetPhysicalDeviceImageFormatProperties2
                let result = unsafe {
                    self.phd
                        .instance()
                        .handle()
                        .get_physical_device_image_format_properties2(
                            self.phd.handle(),
                            &format_info,
                            &mut image_format_properties,
                        )
                };

                match result {
                    Ok(()) => {
                        let image_format_properties = image_format_properties.image_format_properties;
                        let external_memory_properties = external_properties.external_memory_properties;
                        Ok(supports_dma_buf_export(external_memory_properties).then_some(
                            ExternalImageFormatInfo {
                                image_format_properties,
                                dedicated_only: requires_dedicated_allocation(external_memory_properties),
                            },
                        ))
                    }
                    Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(None),
                    Err(result) => Err(result),
                }
            }

            None => Ok(None),
        }
    }

    fn filter_modifiers(
        &self,
        width: u32,
        height: u32,
        vk_usage: vk::ImageUsageFlags,
        fourcc: DrmFourcc,
        modifiers: &[DrmModifier],
    ) -> Result<Vec<u64>, Error> {
        let mut filtered = Vec::new();

        for modifier in modifiers.iter().copied() {
            let format = DrmFormat {
                code: fourcc,
                modifier,
            };

            if self.format_plane_count(format).is_none() {
                continue;
            }

            let info = unsafe { self.get_format_info(format, vk_usage)? };
            let Some(info) = info else {
                continue;
            };

            if image_format_properties_support_extent(info.image_format_properties, width, height) {
                filtered.push(modifier.into());
            }
        }

        Ok(filtered)
    }

    /// # Safety
    ///
    /// * The list of modifiers must be supported for the given format and image usage flags.
    /// * The extent of the image must be within the maximum extents Vulkan tells.
    unsafe fn create_image(
        &mut self,
        width: u32,
        height: u32,
        vk_format: vk::Format,
        vk_usage: vk::ImageUsageFlags,
        fourcc: DrmFourcc,
        modifiers: &[u64],
    ) -> Result<VulkanImage, Error> {
        assert!(width > 0);
        assert!(height > 0);

        // Ensure maximum allocations are not exceeded.
        ensure_allocation_available(self.remaining_allocations)?;

        // Now that the list of valid modifiers is known, create an image using one of the modifiers.
        let mut modifier_list =
            vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(modifiers);
        let mut external_memory_image_create_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width,
                height,
                // VUID-VkImageCreateInfo-extent-00946
                // VUID-VkImageCreateInfo-imageType-00957
                depth: 1,
            })
            // VUID-VkImageCreateInfo-samples-parameter
            .samples(vk::SampleCountFlags::TYPE_1)
            // VUID-VkImageCreateInfo-mipLevels-00947
            .mip_levels(1)
            // VUID-VkImageCreateInfo-arrayLayers-00948
            .array_layers(1)
            // VUID-VkImageCreateInfo-pNext-02262
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            // VUID-VkImageCreateInfo-usage-requiredbitmask
            // FIXME: We don't assert any usage flags are set
            .usage(vk_usage)
            // VUID-VkImageCreateInfo-initialLayout-00993
            .initial_layout(vk::ImageLayout::UNDEFINED)
            // VUID-VkImageCreateInfo-tiling-02261
            .push_next(&mut modifier_list)
            // TODO: VUID-VkImageCreateInfo-pNext-00990
            .push_next(&mut external_memory_image_create_info);

        // The image is placed in a scope guard to safely handle future allocation failures.
        let mut guard = scopeguard::guard(
            ImageInner {
                // This is the only spot where ? may be used to detect and error since no previous handles have been created.
                image: unsafe { self.device.create_image(&image_create_info, None) }?,
                memory: vk::DeviceMemory::null(),
            },
            |inner| unsafe {
                self.device.destroy_image(inner.image, None);
                if inner.memory != vk::DeviceMemory::null() {
                    self.device.free_memory(inner.memory, None);
                }
            },
        );

        // Get the modifier Vulkan created the image using.
        let format = {
            let mut image_modifier_properties = vk::ImageDrmFormatModifierPropertiesEXT::default();

            unsafe {
                self.extension_fns
                    .ext_image_format_modifier
                    .get_image_drm_format_modifier_properties(guard.image, &mut image_modifier_properties)
            }?;

            DrmFormat {
                code: fourcc,
                modifier: DrmModifier::from(image_modifier_properties.drm_format_modifier),
            }
        };

        // Now that we know the plane count, get the number of planes for the format + modifier
        let format_entry = self
            .formats
            .iter()
            .find(|entry| entry.format == format)
            .ok_or(Error::UnsupportedFormat)?;
        let format_plane_count =
            dmabuf_plane_count(format_entry.modifier_properties.drm_format_modifier_plane_count)
                .ok_or(Error::UnsupportedFormat)?;
        let external_format_info =
            unsafe { self.get_format_info(format, vk_usage)? }.ok_or(Error::UnsupportedFormat)?;

        // Allocate image memory.
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut memory_requirements =
            vk::MemoryRequirements2::default().push_next(&mut dedicated_requirements);
        let memory_requirements_info = vk::ImageMemoryRequirementsInfo2::default().image(guard.image);
        unsafe {
            self.device
                .get_image_memory_requirements2(&memory_requirements_info, &mut memory_requirements)
        };
        let memory_reqs = memory_requirements.memory_requirements;
        let memory_dedicated_required = dedicated_requirements.requires_dedicated_allocation == vk::TRUE;
        let memory_dedicated_preferred = dedicated_requirements.prefers_dedicated_allocation == vk::TRUE;
        let memory_properties = unsafe {
            self.phd
                .instance()
                .handle()
                .get_physical_device_memory_properties(self.phd.handle())
        };
        let memory_type_index = find_memory_type_index(&memory_properties, memory_reqs.memory_type_bits)?;
        let memory_type_flags = memory_properties.memory_types[memory_type_index as usize].property_flags;
        let mut export_memory_allocate_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let dedicated = external_format_info.dedicated_only || memory_dedicated_required;
        let mut dedicated_allocate_info = vk::MemoryDedicatedAllocateInfo::default().image(guard.image);
        let alloc_create_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_reqs.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_memory_allocate_info);
        let alloc_create_info = if dedicated {
            alloc_create_info.push_next(&mut dedicated_allocate_info)
        } else {
            alloc_create_info
        };

        unsafe {
            // Allocate memory for the image.
            guard.memory = self.device.allocate_memory(&alloc_create_info, None)?;
            // Finally bind the memory to the image
            self.device.bind_image_memory(guard.image, guard.memory, 0)?;
        }

        // Initialization is complete, prevent the scope guard from running it's dropfn.
        let inner = scopeguard::ScopeGuard::into_inner(guard);

        // Track the image for destruction.
        self.images.push(inner);
        self.image_release_states
            .push((inner, VulkanAllocatorDmabufReleaseState::FreshLocalUndefined));

        self.remaining_allocations -= 1;

        Ok(VulkanImage {
            inner,
            width,
            height,
            format,
            allocation_info: VulkanImageAllocationInfo {
                modifier_tiling_features: format_entry
                    .modifier_properties
                    .drm_format_modifier_tiling_features,
                memory_type_bits: memory_reqs.memory_type_bits,
                memory_size: memory_reqs.size,
                memory_alignment: memory_reqs.alignment,
                memory_type_index,
                memory_type_flags,
                external_dedicated_only: external_format_info.dedicated_only,
                memory_dedicated_required,
                memory_dedicated_preferred,
                dedicated,
            },
            format_plane_count,
            khr_external_memory_fd: self.extension_fns.khr_external_memory_fd.clone(),
            dropped_sender: self.dropped_sender.clone(),
            device: Arc::downgrade(&self.device),
            exports: Arc::new(Mutex::new(Vec::new())),
            #[cfg(feature = "backend_drm")]
            node: self.node,
        })
    }

    fn cleanup(&mut self) {
        let dropped = self.dropped_recv.try_iter().collect::<Vec<_>>();
        self.image_release_states
            .retain(|(image, _state)| !dropped.contains(image));

        self.images.retain(|image| {
            // Only drop if the
            let drop = dropped.contains(image);

            if drop {
                // Destroy the underlying image resource
                unsafe {
                    self.device.destroy_image(image.image, None);
                    self.device.free_memory(image.memory, None);
                }

                self.remaining_allocations = self
                    .remaining_allocations
                    .checked_add(1)
                    .expect("Remaining allocations overflowed");
                debug_assert!(
                    self.phd.limits().max_memory_allocation_count >= self.remaining_allocations,
                    "Too many allocations released",
                );
            }

            // If the image was dropped, return false
            !drop
        })
    }

    fn format_plane_count(&self, format: DrmFormat) -> Option<u32> {
        self.formats
            .iter()
            .find(|entry| entry.format == format)
            .and_then(|entry| dmabuf_plane_count(entry.modifier_properties.drm_format_modifier_plane_count))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Error, ImageInner, ImageUsageFlags, VulkanAllocator, VulkanAllocatorDmabufReleaseState,
        VulkanAllocatorForeignReleaseError, dmabuf_plane_count, dmabuf_plane_layout,
        dmabuf_was_exported_from_image, ensure_allocation_available, find_memory_type_index,
        image_format_properties_support_extent, image_release_state_index, requires_dedicated_allocation,
        supports_dma_buf_export, validate_allocator_release_dmabuf_metadata,
    };
    use crate::backend::{
        allocator::{
            Allocator, Buffer, Format, Fourcc, Modifier,
            dmabuf::{AsDmabuf, Dmabuf, DmabufFlags},
        },
        vulkan::{Instance, PhysicalDevice, version::Version},
    };
    use ash::vk;
    use ash::vk::Handle;
    use std::{fs::File, os::fd::OwnedFd};

    fn dmabuf_for_identity_tests() -> Dmabuf {
        let fd = OwnedFd::from(File::open("/dev/null").unwrap());
        let mut builder = Dmabuf::builder((1, 1), Fourcc::Abgr8888, Modifier::Linear, DmabufFlags::empty());
        assert!(builder.add_plane(fd, 0, 4));
        builder.build().unwrap()
    }

    fn dmabuf_with_planes_for_identity_tests(planes: &[(u32, u32, u32)]) -> Dmabuf {
        let mut builder = Dmabuf::builder((1, 1), Fourcc::Abgr8888, Modifier::Linear, DmabufFlags::empty());
        for &(_idx, offset, stride) in planes {
            let fd = OwnedFd::from(File::open("/dev/null").unwrap());
            assert!(builder.add_plane(fd, offset, stride));
        }
        builder.build().unwrap()
    }

    fn memory_properties_for_tests(flags: &[vk::MemoryPropertyFlags]) -> vk::PhysicalDeviceMemoryProperties {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: flags.len() as u32,
            ..vk::PhysicalDeviceMemoryProperties::default()
        };
        for (idx, flags) in flags.iter().copied().enumerate() {
            properties.memory_types[idx].property_flags = flags;
        }
        properties
    }

    #[test]
    fn memory_type_lookup_respects_supported_bits() {
        let properties = memory_properties_for_tests(&[
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        ]);

        assert_eq!(find_memory_type_index(&properties, 0b10).unwrap(), 1);
        assert!(matches!(
            find_memory_type_index(&properties, 0b00),
            Err(Error::UnsupportedMemoryType)
        ));
    }

    #[test]
    fn memory_type_lookup_prefers_device_local() {
        let properties = memory_properties_for_tests(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ]);

        assert_eq!(find_memory_type_index(&properties, 0b11).unwrap(), 1);
    }

    #[test]
    fn memory_type_lookup_falls_back_to_first_supported_type() {
        let properties = memory_properties_for_tests(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_COHERENT,
        ]);

        assert_eq!(find_memory_type_index(&properties, 0b11).unwrap(), 0);
    }

    #[test]
    fn external_memory_properties_require_dma_buf_export_support() {
        let supported = vk::ExternalMemoryProperties {
            external_memory_features: vk::ExternalMemoryFeatureFlags::EXPORTABLE,
            export_from_imported_handle_types: vk::ExternalMemoryHandleTypeFlags::empty(),
            compatible_handle_types: vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
        };
        assert!(supports_dma_buf_export(supported));

        assert!(!supports_dma_buf_export(vk::ExternalMemoryProperties {
            external_memory_features: vk::ExternalMemoryFeatureFlags::empty(),
            ..supported
        }));
        assert!(!supports_dma_buf_export(vk::ExternalMemoryProperties {
            compatible_handle_types: vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
            ..supported
        }));
    }

    #[test]
    fn external_memory_properties_track_dedicated_only() {
        assert!(requires_dedicated_allocation(vk::ExternalMemoryProperties {
            external_memory_features: vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY,
            export_from_imported_handle_types: vk::ExternalMemoryHandleTypeFlags::empty(),
            compatible_handle_types: vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
        }));
        assert!(!requires_dedicated_allocation(
            vk::ExternalMemoryProperties::default()
        ));
    }

    #[test]
    fn dmabuf_plane_layout_requires_representable_metadata() {
        assert_eq!(
            dmabuf_plane_layout(vk::SubresourceLayout {
                offset: 16,
                row_pitch: 256,
                ..vk::SubresourceLayout::default()
            }),
            Some((16, 256)),
        );

        assert_eq!(
            dmabuf_plane_layout(vk::SubresourceLayout {
                offset: u64::from(u32::MAX) + 1,
                row_pitch: 256,
                ..vk::SubresourceLayout::default()
            }),
            None,
        );
        assert_eq!(
            dmabuf_plane_layout(vk::SubresourceLayout {
                offset: 16,
                row_pitch: u64::from(u32::MAX) + 1,
                ..vk::SubresourceLayout::default()
            }),
            None,
        );
        assert_eq!(
            dmabuf_plane_layout(vk::SubresourceLayout {
                offset: 16,
                row_pitch: 0,
                ..vk::SubresourceLayout::default()
            }),
            None,
        );
    }

    #[test]
    fn dmabuf_plane_count_requires_supported_plane_range() {
        assert_eq!(dmabuf_plane_count(0), None);
        assert_eq!(dmabuf_plane_count(1), Some(1));
        assert_eq!(dmabuf_plane_count(4), Some(4));
        assert_eq!(dmabuf_plane_count(5), None);
    }

    #[test]
    fn image_format_properties_must_support_requested_extent() {
        let properties = vk::ImageFormatProperties {
            max_extent: vk::Extent3D {
                width: 64,
                height: 32,
                depth: 1,
            },
            sample_counts: vk::SampleCountFlags::TYPE_1,
            ..vk::ImageFormatProperties::default()
        };

        assert!(image_format_properties_support_extent(properties, 64, 32));
        assert!(!image_format_properties_support_extent(
            vk::ImageFormatProperties {
                max_extent: vk::Extent3D {
                    width: 63,
                    ..properties.max_extent
                },
                ..properties
            },
            64,
            32,
        ));
        assert!(!image_format_properties_support_extent(
            vk::ImageFormatProperties {
                max_extent: vk::Extent3D {
                    height: 31,
                    ..properties.max_extent
                },
                ..properties
            },
            64,
            32,
        ));
        assert!(!image_format_properties_support_extent(
            vk::ImageFormatProperties {
                max_extent: vk::Extent3D {
                    depth: 0,
                    ..properties.max_extent
                },
                ..properties
            },
            64,
            32,
        ));
        assert!(!image_format_properties_support_extent(
            vk::ImageFormatProperties {
                sample_counts: vk::SampleCountFlags::TYPE_2,
                ..properties
            },
            64,
            32,
        ));
    }

    #[test]
    fn allocation_availability_guard_rejects_exhaustion() {
        assert!(ensure_allocation_available(1).is_ok());
        assert!(matches!(
            ensure_allocation_available(0),
            Err(Error::Vk(vk::Result::ERROR_TOO_MANY_OBJECTS))
        ));
    }

    #[test]
    fn dmabuf_export_identity_matches_only_tracked_live_exports() {
        let exported = dmabuf_for_identity_tests();
        let unrelated = dmabuf_for_identity_tests();
        let exports = [exported.weak()];

        assert!(dmabuf_was_exported_from_image(&exports, &exported));
        assert!(!dmabuf_was_exported_from_image(&exports, &unrelated));

        let stale_export = {
            let stale = dmabuf_for_identity_tests();
            stale.weak()
        };
        assert!(!dmabuf_was_exported_from_image(&[stale_export], &exported));
    }

    #[test]
    fn allocator_release_metadata_validates_planes_size_and_format() {
        let exported = dmabuf_for_identity_tests();
        assert!(
            validate_allocator_release_dmabuf_metadata(exported.size(), exported.format(), 1, &exported,)
                .is_ok()
        );

        let multi_plane = dmabuf_with_planes_for_identity_tests(&[(0, 0, 4), (1, 4, 4)]);
        assert!(matches!(
            validate_allocator_release_dmabuf_metadata(exported.size(), exported.format(), 1, &multi_plane),
            Err(VulkanAllocatorForeignReleaseError::InvalidState(
                "Vulkan allocator dmabuf foreign release planes"
            ))
        ));
        assert!(matches!(
            validate_allocator_release_dmabuf_metadata(exported.size(), exported.format(), 2, &exported),
            Err(VulkanAllocatorForeignReleaseError::MissingCapability(
                "Vulkan allocator dmabuf foreign release planes"
            ))
        ));

        assert!(matches!(
            validate_allocator_release_dmabuf_metadata((2, 1).into(), exported.format(), 1, &exported),
            Err(VulkanAllocatorForeignReleaseError::InvalidState(
                "Vulkan allocator dmabuf foreign release identity"
            ))
        ));
        assert!(matches!(
            validate_allocator_release_dmabuf_metadata(
                exported.size(),
                Format {
                    code: Fourcc::Argb8888,
                    modifier: Modifier::Linear,
                },
                1,
                &exported,
            ),
            Err(VulkanAllocatorForeignReleaseError::InvalidState(
                "Vulkan allocator dmabuf foreign release identity"
            ))
        ));
        assert!(matches!(
            validate_allocator_release_dmabuf_metadata(
                exported.size(),
                Format {
                    code: Fourcc::Abgr8888,
                    modifier: Modifier::Invalid,
                },
                1,
                &exported,
            ),
            Err(VulkanAllocatorForeignReleaseError::InvalidState(
                "Vulkan allocator dmabuf foreign release identity"
            ))
        ));
    }

    #[test]
    fn image_release_state_lookup_is_exact_image_identity() {
        let image_a = ImageInner {
            image: vk::Image::from_raw(1),
            memory: vk::DeviceMemory::from_raw(11),
        };
        let image_b = ImageInner {
            image: vk::Image::from_raw(2),
            memory: vk::DeviceMemory::from_raw(22),
        };
        let states = [
            (image_a, VulkanAllocatorDmabufReleaseState::FreshLocalUndefined),
            (image_b, VulkanAllocatorDmabufReleaseState::ReleasedForeignGeneral),
        ];

        assert_eq!(image_release_state_index(&states, image_a), Some(0));
        assert_eq!(image_release_state_index(&states, image_b), Some(1));
        assert_eq!(
            image_release_state_index(
                &states,
                ImageInner {
                    image: vk::Image::from_raw(1),
                    memory: vk::DeviceMemory::from_raw(99),
                },
            ),
            None,
        );
    }

    #[test]
    #[ignore = "requires a working Vulkan loader, physical device and dmabuf-exportable format"]
    fn runtime_allocator_exports_dmabuf() {
        let instance = match Instance::new(Version::VERSION_1_3, None) {
            Ok(instance) => instance,
            Err(err) => {
                eprintln!("skipping Vulkan allocator export test: failed to create instance: {err:?}");
                return;
            }
        };

        let devices = match PhysicalDevice::enumerate(&instance) {
            Ok(devices) => devices,
            Err(err) => {
                eprintln!("skipping Vulkan allocator export test: failed to enumerate devices: {err:?}");
                return;
            }
        };
        let mut found_extension_capable_device = false;
        let mut created_allocator = false;
        let mut allocator_setup_errors = Vec::new();

        for physical_device in devices {
            if !VulkanAllocator::required_extensions(&physical_device)
                .into_iter()
                .all(|extension| physical_device.has_device_extension(extension))
            {
                continue;
            }

            found_extension_capable_device = true;

            let usage = ImageUsageFlags::COLOR_ATTACHMENT;
            let mut allocator = match VulkanAllocator::new(&physical_device, usage) {
                Ok(allocator) => allocator,
                Err(err) => {
                    allocator_setup_errors.push(format!("{}: {err:?}", physical_device.name()));
                    continue;
                }
            };
            created_allocator = true;

            let format = allocator
                .formats
                .iter()
                .map(|entry| entry.format)
                .find(|format| allocator.is_format_supported(*format, usage));
            let Some(format) = format else {
                continue;
            };

            let image = allocator
                .create_buffer(64, 64, format.code, &[format.modifier])
                .expect("create exportable Vulkan image");
            let dmabuf = image.export().expect("export Vulkan image as dmabuf");

            assert_eq!(image.size(), dmabuf.size());
            assert_eq!(image.format(), dmabuf.format());
            assert_eq!(dmabuf.num_planes(), image.format_plane_count as usize);
            assert!(dmabuf.num_planes() > 0);
            assert_eq!(dmabuf.handles().count(), dmabuf.num_planes());
            assert_eq!(dmabuf.offsets().count(), dmabuf.num_planes());
            assert_eq!(dmabuf.strides().count(), dmabuf.num_planes());
            assert!(dmabuf.strides().all(|stride| stride > 0));

            return;
        }

        if !found_extension_capable_device {
            eprintln!("skipping Vulkan allocator export test: no device supports required extensions");
            return;
        }

        if !created_allocator {
            panic!(
                "failed to create Vulkan allocator for extension-capable devices: {allocator_setup_errors:?}"
            );
        }

        if !allocator_setup_errors.is_empty() {
            eprintln!(
                "Vulkan allocator setup failed on some devices while looking for exportable formats: \
                 {allocator_setup_errors:?}"
            );
        }

        eprintln!("skipping Vulkan allocator export test: no dmabuf-exportable color format");
    }
}
