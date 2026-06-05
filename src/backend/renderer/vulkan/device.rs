use std::{
    ffi::c_void,
    ptr,
    sync::{Arc, Mutex, MutexGuard},
};

use ash::vk;

use crate::backend::vulkan::{Instance, PhysicalDevice};

use super::{VulkanError, VulkanRendererCapabilities};

/// Device state placeholder for the future Vulkan renderer implementation.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanDeviceState {
    pub(super) graphics_command_pool: Option<Arc<VulkanCommandPool>>,
    pub(super) transfer_command_pool: Option<Arc<VulkanCommandPool>>,
    pub(super) queues: VulkanQueues,
    pub(super) queue_families: VulkanQueueFamilies,
    pub(super) logical_device: Option<VulkanLogicalDevice>,
    pub(super) physical_device: Option<PhysicalDevice>,
    pub(super) instance: Option<Instance>,
    pub(super) memory_properties: Option<vk::PhysicalDeviceMemoryProperties>,
    pub(super) capabilities: VulkanRendererCapabilities,
    pub(super) enabled_extensions: Vec<String>,
}

impl VulkanDeviceState {
    pub(super) fn new(physical_device: PhysicalDevice) -> Result<Self, VulkanError> {
        let instance = physical_device.instance().clone();
        let queue_properties = unsafe {
            instance
                .handle()
                .get_physical_device_queue_family_properties(physical_device.handle())
        };
        let memory_properties = unsafe {
            instance
                .handle()
                .get_physical_device_memory_properties(physical_device.handle())
        };
        let queue_families = select_queue_families(&queue_properties)?;
        let mut capabilities = VulkanRendererCapabilities::for_initialized_device(&[]);
        capabilities.formats = super::VulkanFormatCapabilities::discover(&physical_device)?;

        let queue_priorities = [1.0];
        let queue_create_infos = queue_families
            .unique_indices()
            .map(|queue_family_index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&queue_priorities)
            })
            .collect::<Vec<_>>();
        let create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);

        let logical_device = unsafe {
            instance
                .handle()
                .create_device(physical_device.handle(), &create_info, None)
        }
        .map_err(VulkanError::from)?;
        let logical_device = VulkanLogicalDevice::new(logical_device);

        let graphics_family = queue_families
            .graphics
            .ok_or(VulkanError::QueueFamilyUnsupported)?;
        let graphics_queue =
            VulkanQueue::new(unsafe { logical_device.handle().get_device_queue(graphics_family, 0) });
        let transfer_queue = match queue_families.transfer {
            Some(transfer_family) if transfer_family == graphics_family => Some(graphics_queue.clone()),
            Some(transfer_family) => Some(VulkanQueue::new(unsafe {
                logical_device.handle().get_device_queue(transfer_family, 0)
            })),
            None => None,
        };
        let queues = VulkanQueues {
            graphics: Some(graphics_queue),
            transfer: transfer_queue,
        };

        let graphics_command_pool = create_command_pool(&logical_device, graphics_family)?;
        let transfer_command_pool = queue_families
            .transfer
            .map(|family| create_command_pool(&logical_device, family))
            .transpose()?;

        Ok(Self {
            graphics_command_pool: Some(graphics_command_pool),
            transfer_command_pool,
            queues,
            queue_families,
            logical_device: Some(logical_device),
            physical_device: Some(physical_device),
            instance: Some(instance),
            memory_properties: Some(memory_properties),
            capabilities,
            enabled_extensions: Vec::new(),
        })
    }

    #[cfg(test)]
    pub(super) fn empty_for_tests() -> Self {
        Self {
            graphics_command_pool: None,
            transfer_command_pool: None,
            queues: VulkanQueues::default(),
            queue_families: VulkanQueueFamilies::default(),
            logical_device: None,
            physical_device: None,
            instance: None,
            memory_properties: None,
            capabilities: VulkanRendererCapabilities::default(),
            enabled_extensions: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub(super) fn allocate_graphics_command_buffer(&self) -> Result<VulkanCommandBuffer, VulkanError> {
        self.logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let command_pool = self.graphics_command_pool.as_ref().ok_or_else(|| {
            VulkanError::DeviceInitializationFailed("missing graphics command pool".to_owned())
        })?;

        allocate_command_buffer(command_pool)
    }

    #[allow(dead_code)]
    pub(super) fn allocate_transfer_command_buffer(&self) -> Result<VulkanCommandBuffer, VulkanError> {
        self.logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let command_pool = self.transfer_command_pool.as_ref().ok_or_else(|| {
            VulkanError::DeviceInitializationFailed("missing transfer command pool".to_owned())
        })?;

        allocate_command_buffer(command_pool)
    }

    #[allow(dead_code)]
    pub(super) fn begin_command_buffer(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
    ) -> Result<(), VulkanError> {
        begin_command_buffer(command_buffer)
    }

    #[allow(dead_code)]
    pub(super) fn end_command_buffer(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
    ) -> Result<(), VulkanError> {
        let _pool_guard = command_buffer.command_pool.lock_host_access()?;

        unsafe {
            command_buffer
                .command_pool
                .logical_device
                .handle()
                .end_command_buffer(command_buffer.handle)
        }
        .map_err(VulkanError::from)
    }

    #[allow(dead_code)]
    pub(super) fn submit_graphics_command_buffer_and_wait(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
    ) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let queue =
            self.queues.graphics.as_ref().ok_or_else(|| {
                VulkanError::DeviceInitializationFailed("missing graphics queue".to_owned())
            })?;

        submit_command_buffer_and_wait(logical_device, queue, command_buffer)
    }

    #[allow(dead_code)]
    pub(super) fn submit_transfer_command_buffer_and_wait(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
    ) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let queue =
            self.queues.transfer.as_ref().ok_or_else(|| {
                VulkanError::DeviceInitializationFailed("missing transfer queue".to_owned())
            })?;

        submit_command_buffer_and_wait(logical_device, queue, command_buffer)
    }

    #[allow(dead_code)]
    pub(super) fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, VulkanError> {
        let memory_properties = self
            .memory_properties
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing memory properties".to_owned()))?;

        find_memory_type_index(memory_properties, memory_type_bits, required_properties)
    }

    #[allow(dead_code)]
    pub(super) fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<vk::Buffer, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        create_buffer(logical_device, size, usage)
    }

    #[allow(dead_code)]
    pub(super) fn destroy_buffer(&self, buffer: vk::Buffer) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        unsafe { logical_device.handle().destroy_buffer(buffer, None) };
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn buffer_memory_requirements(
        &self,
        buffer: vk::Buffer,
    ) -> Result<vk::MemoryRequirements, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        Ok(unsafe { logical_device.handle().get_buffer_memory_requirements(buffer) })
    }

    #[allow(dead_code)]
    pub(super) fn allocate_memory(
        &self,
        size: vk::DeviceSize,
        memory_type_index: u32,
    ) -> Result<vk::DeviceMemory, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        allocate_memory(logical_device, size, memory_type_index)
    }

    #[allow(dead_code)]
    pub(super) fn free_memory(&self, memory: vk::DeviceMemory) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        unsafe { logical_device.handle().free_memory(memory, None) };
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn bind_buffer_memory(
        &self,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    ) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        unsafe { logical_device.handle().bind_buffer_memory(buffer, memory, offset) }
            .map_err(VulkanError::from)
    }

    #[allow(dead_code)]
    pub(super) fn map_memory(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> Result<*mut c_void, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        unsafe {
            logical_device
                .handle()
                .map_memory(memory, offset, size, vk::MemoryMapFlags::empty())
        }
        .map_err(VulkanError::from)
    }

    #[allow(dead_code)]
    pub(super) fn unmap_memory(&self, memory: vk::DeviceMemory) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        unsafe { logical_device.handle().unmap_memory(memory) };
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn flush_mapped_memory_range(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let ranges = [vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(offset)
            .size(size)];

        unsafe { logical_device.handle().flush_mapped_memory_ranges(&ranges) }.map_err(VulkanError::from)
    }

    #[allow(dead_code)]
    pub(super) fn create_host_visible_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<VulkanHostVisibleBuffer, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();
        let memory_properties = self
            .memory_properties
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing memory properties".to_owned()))?;
        let buffer = create_buffer(&logical_device, size, usage)?;
        let requirements = unsafe { logical_device.handle().get_buffer_memory_requirements(buffer) };
        let memory_type_index = match find_memory_type_index(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(err) => {
                unsafe { logical_device.handle().destroy_buffer(buffer, None) };
                return Err(err);
            }
        };
        let memory = match allocate_memory(&logical_device, requirements.size, memory_type_index) {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { logical_device.handle().destroy_buffer(buffer, None) };
                return Err(err);
            }
        };

        if let Err(err) = unsafe { logical_device.handle().bind_buffer_memory(buffer, memory, 0) }
            .map_err(VulkanError::from)
        {
            unsafe {
                logical_device.handle().free_memory(memory, None);
                logical_device.handle().destroy_buffer(buffer, None);
            }
            return Err(err);
        }

        Ok(VulkanHostVisibleBuffer {
            inner: Arc::new(VulkanHostVisibleBufferInner {
                logical_device,
                buffer,
                memory,
                size,
                usage,
            }),
        })
    }

    #[allow(dead_code)]
    pub(super) fn create_image(
        &self,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<vk::Image, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        create_image(logical_device, extent, format, usage)
    }

    #[allow(dead_code)]
    pub(super) fn image_memory_requirements(
        &self,
        image: vk::Image,
    ) -> Result<vk::MemoryRequirements, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        Ok(unsafe { logical_device.handle().get_image_memory_requirements(image) })
    }

    #[allow(dead_code)]
    pub(super) fn bind_image_memory(
        &self,
        image: vk::Image,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    ) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        unsafe { logical_device.handle().bind_image_memory(image, memory, offset) }.map_err(VulkanError::from)
    }

    #[allow(dead_code)]
    pub(super) fn transition_image_layout(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        image: &VulkanOwnedImage,
        new_layout: vk::ImageLayout,
    ) -> Result<(), VulkanError> {
        transition_image_layout(command_buffer, image, new_layout)
    }

    #[allow(dead_code)]
    pub(super) fn copy_buffer_to_image(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        buffer: &VulkanHostVisibleBuffer,
        image: &VulkanOwnedImage,
        extent: vk::Extent3D,
    ) -> Result<(), VulkanError> {
        copy_buffer_to_image(command_buffer, buffer, image, extent)
    }

    #[allow(dead_code)]
    pub(super) fn create_bound_image(
        &self,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        required_memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<VulkanOwnedImage, VulkanError> {
        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing instance".to_owned()))?;
        let physical_device = self
            .physical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing physical device".to_owned()))?;
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();
        let memory_properties = self
            .memory_properties
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing memory properties".to_owned()))?;

        let _properties =
            validate_optimal_2d_image_support(instance, physical_device, extent, format, usage)?;

        create_bound_image(
            &logical_device,
            memory_properties,
            extent,
            format,
            usage,
            required_memory_properties,
        )
    }

    #[allow(dead_code)]
    pub(super) fn destroy_image(&self, image: vk::Image) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        unsafe { logical_device.handle().destroy_image(image, None) };
        Ok(())
    }
}

fn create_command_pool(
    logical_device: &VulkanLogicalDevice,
    queue_family_index: u32,
) -> Result<Arc<VulkanCommandPool>, VulkanError> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let handle = unsafe {
        logical_device
            .handle()
            .create_command_pool(&command_pool_info, None)
    }
    .map_err(VulkanError::from)?;

    Ok(Arc::new(VulkanCommandPool {
        logical_device: logical_device.clone(),
        handle,
        host_access: Mutex::new(()),
    }))
}

fn allocate_command_buffer(
    command_pool: &Arc<VulkanCommandPool>,
) -> Result<VulkanCommandBuffer, VulkanError> {
    let _pool_guard = command_pool.lock_host_access()?;
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool.handle)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffers = unsafe {
        command_pool
            .logical_device
            .handle()
            .allocate_command_buffers(&allocate_info)
    }
    .map_err(VulkanError::from)?;

    let handle = command_buffers
        .into_iter()
        .next()
        .ok_or_else(|| VulkanError::DeviceInitializationFailed("no command buffer allocated".to_owned()))?;

    Ok(VulkanCommandBuffer {
        command_pool: Arc::clone(command_pool),
        handle,
        pending_image_layouts: Vec::new(),
        referenced_buffers: Vec::new(),
        referenced_images: Vec::new(),
    })
}

fn begin_command_buffer(command_buffer: &mut VulkanCommandBuffer) -> Result<(), VulkanError> {
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;
    command_buffer.pending_image_layouts.clear();
    command_buffer.referenced_buffers.clear();
    command_buffer.referenced_images.clear();
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .begin_command_buffer(command_buffer.handle, &begin_info)
    }
    .map_err(VulkanError::from)
}

fn submit_command_buffer_and_wait(
    logical_device: &VulkanLogicalDevice,
    queue: &VulkanQueue,
    command_buffer: &mut VulkanCommandBuffer,
) -> Result<(), VulkanError> {
    let fence_info = vk::FenceCreateInfo::default();
    let fence =
        unsafe { logical_device.handle().create_fence(&fence_info, None) }.map_err(VulkanError::from)?;
    let command_buffers = [command_buffer.handle];
    let submit_infos = [vk::SubmitInfo::default().command_buffers(&command_buffers)];

    let result = command_buffer
        .command_pool
        .lock_host_access()
        .and_then(|_pool_guard| {
            queue.lock_host_access().and_then(|_queue_guard| {
                unsafe {
                    logical_device
                        .handle()
                        .queue_submit(queue.handle, &submit_infos, fence)
                }
                .map_err(VulkanError::from)
            })
        })
        .and_then(|_| {
            unsafe { logical_device.handle().wait_for_fences(&[fence], true, u64::MAX) }
                .map_err(VulkanError::from)
        });

    unsafe { logical_device.handle().destroy_fence(fence, None) };

    result.and_then(|_| command_buffer.commit_pending_image_layouts())
}

fn transition_image_layout(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
    new_layout: vk::ImageLayout,
) -> Result<(), VulkanError> {
    let old_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    if old_layout == new_layout {
        return Ok(());
    }

    let transition = image_layout_transition(old_layout, new_layout, image.usage())?;
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(transition.src_access)
        .dst_access_mask(transition.dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image.image())
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_pipeline_barrier(
                command_buffer.handle,
                transition.src_stage,
                transition.dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
    };

    command_buffer
        .pending_image_layouts
        .push(VulkanPendingImageLayout {
            image: image.image(),
            resource: Arc::clone(&image.inner),
            new_layout,
        });

    Ok(())
}

fn copy_buffer_to_image(
    command_buffer: &mut VulkanCommandBuffer,
    buffer: &VulkanHostVisibleBuffer,
    image: &VulkanOwnedImage,
    extent: vk::Extent3D,
) -> Result<(), VulkanError> {
    if !buffer.usage().contains(vk::BufferUsageFlags::TRANSFER_SRC) {
        return Err(VulkanError::UnsupportedOperation("buffer transfer source usage"));
    }
    if buffer.size() == 0 {
        return Err(VulkanError::UnsupportedOperation("empty image copy buffer"));
    }
    if !image.usage().contains(vk::ImageUsageFlags::TRANSFER_DST) {
        return Err(VulkanError::UnsupportedOperation(
            "image transfer destination usage",
        ));
    }
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
        return Err(VulkanError::UnsupportedOperation("zero-sized image copy"));
    }
    let image_extent = image.extent();
    if extent.width > image_extent.width
        || extent.height > image_extent.height
        || extent.depth > image_extent.depth
    {
        return Err(VulkanError::UnsupportedOperation("image copy extent"));
    }

    let image_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    if image_layout != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
        return Err(VulkanError::UnsupportedOperation("image copy layout"));
    }

    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(extent);
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_copy_buffer_to_image(
                command_buffer.handle,
                buffer.buffer(),
                image.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            )
    };

    command_buffer.referenced_buffers.push(Arc::clone(&buffer.inner));
    command_buffer.referenced_images.push(Arc::clone(&image.inner));

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VulkanLayoutTransition {
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
}

pub(super) fn image_layout_transition(
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    usage: vk::ImageUsageFlags,
) -> Result<VulkanLayoutTransition, VulkanError> {
    match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
            if !usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
                return Err(VulkanError::UnsupportedOperation(
                    "image transfer destination usage",
                ));
            }

            Ok(VulkanLayoutTransition {
                src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
                src_access: vk::AccessFlags::empty(),
                dst_access: vk::AccessFlags::TRANSFER_WRITE,
            })
        }
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
            if !usage.contains(vk::ImageUsageFlags::SAMPLED) {
                return Err(VulkanError::UnsupportedOperation("image sampled usage"));
            }

            Ok(VulkanLayoutTransition {
                src_stage: vk::PipelineStageFlags::TRANSFER,
                dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                src_access: vk::AccessFlags::TRANSFER_WRITE,
                dst_access: vk::AccessFlags::SHADER_READ,
            })
        }
        _ => Err(VulkanError::UnsupportedOperation("image layout transition")),
    }
}

pub(super) fn find_memory_type_index(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    required_properties: vk::MemoryPropertyFlags,
) -> Result<u32, VulkanError> {
    for index in 0..memory_properties.memory_type_count {
        let memory_type_supported = (memory_type_bits & (1u32 << index)) != 0;
        let properties = memory_properties.memory_types[index as usize].property_flags;

        if memory_type_supported && properties.contains(required_properties) {
            return Ok(index);
        }
    }

    Err(VulkanError::MemoryTypeUnsupported)
}

fn create_buffer(
    logical_device: &VulkanLogicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<vk::Buffer, VulkanError> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    unsafe { logical_device.handle().create_buffer(&buffer_info, None) }.map_err(VulkanError::from)
}

fn allocate_memory(
    logical_device: &VulkanLogicalDevice,
    size: vk::DeviceSize,
    memory_type_index: u32,
) -> Result<vk::DeviceMemory, VulkanError> {
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(memory_type_index);

    unsafe { logical_device.handle().allocate_memory(&allocate_info, None) }.map_err(VulkanError::from)
}

fn create_image(
    logical_device: &VulkanLogicalDevice,
    extent: vk::Extent3D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<vk::Image, VulkanError> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    unsafe { logical_device.handle().create_image(&image_info, None) }.map_err(VulkanError::from)
}

fn validate_optimal_2d_image_support(
    instance: &Instance,
    physical_device: &PhysicalDevice,
    extent: vk::Extent3D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<vk::ImageFormatProperties, VulkanError> {
    if usage.is_empty() {
        return Err(VulkanError::UnsupportedOperation("empty image usage"));
    }
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
        return Err(VulkanError::UnsupportedOperation("zero-sized image"));
    }

    let format_info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .flags(vk::ImageCreateFlags::empty());
    let mut image_format_properties = vk::ImageFormatProperties2::default();

    unsafe {
        instance.handle().get_physical_device_image_format_properties2(
            physical_device.handle(),
            &format_info,
            &mut image_format_properties,
        )
    }
    .map_err(|err| {
        if err == vk::Result::ERROR_FORMAT_NOT_SUPPORTED {
            VulkanError::UnsupportedOperation("image format unsupported")
        } else {
            VulkanError::from(err)
        }
    })?;

    let properties = image_format_properties.image_format_properties;
    if properties.max_extent.width < extent.width
        || properties.max_extent.height < extent.height
        || properties.max_extent.depth < extent.depth
    {
        return Err(VulkanError::UnsupportedOperation("image extent"));
    }
    if !properties.sample_counts.contains(vk::SampleCountFlags::TYPE_1) {
        return Err(VulkanError::UnsupportedOperation("image sample count"));
    }

    Ok(properties)
}

fn create_bound_image(
    logical_device: &VulkanLogicalDevice,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    extent: vk::Extent3D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
) -> Result<VulkanOwnedImage, VulkanError> {
    if usage.is_empty() {
        return Err(VulkanError::UnsupportedOperation("empty image usage"));
    }
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
        return Err(VulkanError::UnsupportedOperation("zero-sized image"));
    }

    let image = create_image(logical_device, extent, format, usage)?;
    let requirements = unsafe { logical_device.handle().get_image_memory_requirements(image) };
    let memory_type_index = match find_memory_type_index(
        memory_properties,
        requirements.memory_type_bits,
        required_memory_properties,
    ) {
        Ok(index) => index,
        Err(err) => {
            unsafe { logical_device.handle().destroy_image(image, None) };
            return Err(err);
        }
    };
    let memory = match allocate_memory(logical_device, requirements.size, memory_type_index) {
        Ok(memory) => memory,
        Err(err) => {
            unsafe { logical_device.handle().destroy_image(image, None) };
            return Err(err);
        }
    };

    if let Err(err) =
        unsafe { logical_device.handle().bind_image_memory(image, memory, 0) }.map_err(VulkanError::from)
    {
        unsafe {
            logical_device.handle().free_memory(memory, None);
            logical_device.handle().destroy_image(image, None);
        }
        return Err(err);
    }

    Ok(VulkanOwnedImage {
        inner: Arc::new(VulkanOwnedImageInner {
            logical_device: logical_device.clone(),
            image,
            memory,
            extent,
            format,
            usage,
            layout: Mutex::new(vk::ImageLayout::UNDEFINED),
        }),
    })
}

/// Logical device owner placeholder.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct VulkanLogicalDevice(Arc<ash::Device>);

impl VulkanLogicalDevice {
    fn new(device: ash::Device) -> Self {
        Self(Arc::new(device))
    }

    pub(super) fn handle(&self) -> &ash::Device {
        &self.0
    }
}

impl Drop for VulkanLogicalDevice {
    fn drop(&mut self) {
        if let Some(device) = Arc::get_mut(&mut self.0) {
            unsafe { device.destroy_device(None) };
        }
    }
}

impl std::fmt::Debug for VulkanLogicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("VulkanLogicalDevice").field(&"_").finish()
    }
}

fn host_synchronization_failed() -> VulkanError {
    VulkanError::DeviceInitializationFailed("Vulkan host synchronization lock poisoned".to_owned())
}

/// Synchronized queue handle for renderer submissions.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VulkanQueue {
    handle: vk::Queue,
    host_access: Arc<Mutex<()>>,
}

impl VulkanQueue {
    fn new(handle: vk::Queue) -> Self {
        Self {
            handle,
            host_access: Arc::new(Mutex::new(())),
        }
    }

    fn lock_host_access(&self) -> Result<MutexGuard<'_, ()>, VulkanError> {
        self.host_access.lock().map_err(|_| host_synchronization_failed())
    }
}

/// Synchronized command pool owner.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanCommandPool {
    logical_device: VulkanLogicalDevice,
    handle: vk::CommandPool,
    host_access: Mutex<()>,
}

impl VulkanCommandPool {
    fn lock_host_access(&self) -> Result<MutexGuard<'_, ()>, VulkanError> {
        self.host_access.lock().map_err(|_| host_synchronization_failed())
    }
}

impl Drop for VulkanCommandPool {
    fn drop(&mut self) {
        let _pool_guard = self
            .host_access
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            self.logical_device
                .handle()
                .destroy_command_pool(self.handle, None)
        };
    }
}

/// Primary command buffer owned by its command pool.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanCommandBuffer {
    command_pool: Arc<VulkanCommandPool>,
    handle: vk::CommandBuffer,
    pending_image_layouts: Vec<VulkanPendingImageLayout>,
    referenced_buffers: Vec<Arc<VulkanHostVisibleBufferInner>>,
    referenced_images: Vec<Arc<VulkanOwnedImageInner>>,
}

impl VulkanCommandBuffer {
    pub(super) fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    fn pending_layout_for(&self, image: &VulkanOwnedImage) -> Result<Option<vk::ImageLayout>, VulkanError> {
        Ok(self
            .pending_image_layouts
            .iter()
            .rev()
            .find(|pending| pending.image == image.image())
            .map(|pending| pending.new_layout))
    }

    fn commit_pending_image_layouts(&mut self) -> Result<(), VulkanError> {
        for pending in &self.pending_image_layouts {
            *pending
                .resource
                .layout
                .lock()
                .map_err(|_| host_synchronization_failed())? = pending.new_layout;
        }

        self.pending_image_layouts.clear();
        self.referenced_buffers.clear();
        self.referenced_images.clear();
        Ok(())
    }
}

#[derive(Debug)]
struct VulkanPendingImageLayout {
    image: vk::Image,
    resource: Arc<VulkanOwnedImageInner>,
    new_layout: vk::ImageLayout,
}

impl Drop for VulkanCommandBuffer {
    fn drop(&mut self) {
        let _pool_guard = self
            .command_pool
            .host_access
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            self.command_pool
                .logical_device
                .handle()
                .free_command_buffers(self.command_pool.handle, &[self.handle])
        };
    }
}

/// Vulkan image bound to owned device memory.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VulkanOwnedImage {
    inner: Arc<VulkanOwnedImageInner>,
}

/// Shared owned image resource kept alive by command buffers that reference it.
#[allow(dead_code)]
#[derive(Debug)]
struct VulkanOwnedImageInner {
    logical_device: VulkanLogicalDevice,
    image: vk::Image,
    memory: vk::DeviceMemory,
    extent: vk::Extent3D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    layout: Mutex<vk::ImageLayout>,
}

impl VulkanOwnedImage {
    pub(super) fn image(&self) -> vk::Image {
        self.inner.image
    }

    pub(super) fn memory(&self) -> vk::DeviceMemory {
        self.inner.memory
    }

    pub(super) fn extent(&self) -> vk::Extent3D {
        self.inner.extent
    }

    pub(super) fn format(&self) -> vk::Format {
        self.inner.format
    }

    pub(super) fn usage(&self) -> vk::ImageUsageFlags {
        self.inner.usage
    }

    pub(super) fn layout(&self) -> Result<vk::ImageLayout, VulkanError> {
        self.inner
            .layout
            .lock()
            .map(|layout| *layout)
            .map_err(|_| host_synchronization_failed())
    }
}

impl Drop for VulkanOwnedImageInner {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.handle().destroy_image(self.image, None);
            self.logical_device.handle().free_memory(self.memory, None);
        }
    }
}

/// Host-visible buffer owner for staging-style uploads.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VulkanHostVisibleBuffer {
    inner: Arc<VulkanHostVisibleBufferInner>,
}

/// Shared host-visible buffer resource kept alive by command buffers that reference it.
#[allow(dead_code)]
#[derive(Debug)]
struct VulkanHostVisibleBufferInner {
    logical_device: VulkanLogicalDevice,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
}

impl VulkanHostVisibleBuffer {
    pub(super) fn buffer(&self) -> vk::Buffer {
        self.inner.buffer
    }

    pub(super) fn size(&self) -> vk::DeviceSize {
        self.inner.size
    }

    pub(super) fn usage(&self) -> vk::BufferUsageFlags {
        self.inner.usage
    }

    pub(super) fn write(&self, data: &[u8]) -> Result<(), VulkanError> {
        if data.len() as vk::DeviceSize > self.inner.size {
            return Err(VulkanError::UnsupportedOperation("mapped buffer write size"));
        }

        let mapped = unsafe {
            self.inner.logical_device.handle().map_memory(
                self.inner.memory,
                0,
                self.inner.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(VulkanError::from)?;

        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast::<u8>(), data.len());
        }

        unsafe { self.inner.logical_device.handle().unmap_memory(self.inner.memory) };

        Ok(())
    }
}

impl Drop for VulkanHostVisibleBufferInner {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.handle().destroy_buffer(self.buffer, None);
            self.logical_device.handle().free_memory(self.memory, None);
        }
    }
}

/// Queue family placeholders discovered during device initialization.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct VulkanQueueFamilies {
    pub(super) graphics: Option<u32>,
    pub(super) transfer: Option<u32>,
}

impl VulkanQueueFamilies {
    fn unique_indices(&self) -> impl Iterator<Item = u32> {
        let graphics = self.graphics.into_iter();
        let transfer = self.transfer.filter(|transfer| Some(*transfer) != self.graphics);

        graphics.chain(transfer)
    }
}

/// Vulkan queues selected for renderer submissions.
#[derive(Debug, Default, Clone)]
pub(crate) struct VulkanQueues {
    pub(super) graphics: Option<VulkanQueue>,
    pub(super) transfer: Option<VulkanQueue>,
}

pub(super) fn select_queue_families(
    queue_properties: &[vk::QueueFamilyProperties],
) -> Result<VulkanQueueFamilies, VulkanError> {
    let graphics = queue_properties
        .iter()
        .position(|properties| {
            properties.queue_count > 0 && properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .and_then(|idx| idx.try_into().ok())
        .ok_or(VulkanError::QueueFamilyUnsupported)?;

    let transfer = queue_properties
        .iter()
        .position(|properties| {
            properties.queue_count > 0
                && properties.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .or_else(|| {
            queue_properties.iter().position(|properties| {
                properties.queue_count > 0 && properties.queue_flags.contains(vk::QueueFlags::TRANSFER)
            })
        })
        .and_then(|idx| idx.try_into().ok())
        .or(Some(graphics));

    Ok(VulkanQueueFamilies {
        graphics: Some(graphics),
        transfer,
    })
}
