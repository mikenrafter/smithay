use std::{
    ffi::c_void,
    ptr,
    sync::{Arc, Mutex, MutexGuard},
};

use ash::vk;

use crate::backend::{
    renderer::TextureFilter,
    vulkan::{Instance, PhysicalDevice},
};

use super::{VulkanError, VulkanRendererCapabilities};

const SAMPLED_TEXTURE_FULL_UV_RECT: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const SAMPLED_TEXTURE_DRAW_CONSTANT_SIZE: u32 = 24;

#[derive(Debug, Clone, Copy)]
pub(super) struct VulkanSampledTextureDrawConstants {
    pub(super) draw_area: vk::Rect2D,
    pub(super) uv_rect: [f32; 4],
    pub(super) alpha: f32,
    pub(super) force_opaque_alpha: bool,
}

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
        capabilities.import.memory = capabilities.formats.memory_import.iter().next().is_some();
        capabilities.rendering.offscreen = capabilities
            .formats
            .render_target_formats()
            .iter()
            .next()
            .is_some();

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
        let logical_device = VulkanLogicalDevice::new(logical_device, instance.clone());

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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn copy_buffer_region_to_image(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        buffer: &VulkanHostVisibleBuffer,
        image: &VulkanOwnedImage,
        buffer_offset: vk::DeviceSize,
        buffer_row_length: u32,
        buffer_image_height: u32,
        image_offset: vk::Offset3D,
        extent: vk::Extent3D,
    ) -> Result<(), VulkanError> {
        copy_buffer_region_to_image(
            command_buffer,
            buffer,
            image,
            buffer_offset,
            buffer_row_length,
            buffer_image_height,
            image_offset,
            extent,
        )
    }

    #[allow(dead_code)]
    pub(super) fn create_uploaded_image(
        &self,
        extent: vk::Extent3D,
        format: vk::Format,
        data: &[u8],
    ) -> Result<VulkanOwnedImage, VulkanError> {
        let required_size = tightly_packed_image_size(format, extent)?;
        if (data.len() as vk::DeviceSize) < required_size {
            return Err(VulkanError::UnsupportedOperation("image upload data"));
        }
        let required_len = usize::try_from(required_size)
            .map_err(|_| VulkanError::UnsupportedOperation("image data size"))?;

        let staging = self.create_host_visible_buffer(required_size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        staging.write(&data[..required_len])?;
        let image = self.create_bound_image(
            extent,
            format,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let mut command_buffer = self.allocate_graphics_command_buffer()?;

        self.begin_command_buffer(&mut command_buffer)?;
        self.transition_image_layout(&mut command_buffer, &image, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        self.copy_buffer_to_image(&mut command_buffer, &staging, &image, extent)?;
        self.transition_image_layout(
            &mut command_buffer,
            &image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;
        self.end_command_buffer(&mut command_buffer)?;
        self.submit_graphics_command_buffer_and_wait(&mut command_buffer)?;

        Ok(image)
    }

    #[allow(dead_code)]
    pub(super) fn create_image_view(&self, image: &VulkanOwnedImage) -> Result<VulkanImageView, VulkanError> {
        self.logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        create_image_view(image)
    }

    #[allow(dead_code)]
    pub(super) fn create_sampler(
        &self,
        min_filter: TextureFilter,
        mag_filter: TextureFilter,
    ) -> Result<VulkanSampler, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();

        create_sampler(&logical_device, min_filter, mag_filter)
    }

    #[allow(dead_code)]
    pub(super) fn create_uploaded_sampled_image(
        &self,
        extent: vk::Extent3D,
        format: vk::Format,
        data: &[u8],
        min_filter: TextureFilter,
        mag_filter: TextureFilter,
    ) -> Result<VulkanSampledImage, VulkanError> {
        let image = self.create_uploaded_image(extent, format, data)?;
        let view = self.create_image_view(&image)?;
        let sampler = self.create_sampler(min_filter, mag_filter)?;

        Ok(VulkanSampledImage { sampler, view, image })
    }

    #[allow(dead_code)]
    pub(super) fn create_offscreen_color_image(
        &self,
        extent: vk::Extent3D,
        format: vk::Format,
    ) -> Result<VulkanOwnedImage, VulkanError> {
        self.create_bound_image(
            extent,
            format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    #[allow(dead_code)]
    pub(super) fn clear_offscreen_color_image(
        &self,
        image: &VulkanOwnedImage,
        color: vk::ClearColorValue,
    ) -> Result<(), VulkanError> {
        let mut command_buffer = self.allocate_graphics_command_buffer()?;

        self.begin_command_buffer(&mut command_buffer)?;
        self.transition_image_layout(&mut command_buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        self.clear_color_image(&mut command_buffer, image, color)?;
        self.end_command_buffer(&mut command_buffer)?;
        self.submit_graphics_command_buffer_and_wait(&mut command_buffer)
    }

    #[allow(dead_code)]
    pub(super) fn clear_color_image(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        image: &VulkanOwnedImage,
        color: vk::ClearColorValue,
    ) -> Result<(), VulkanError> {
        clear_color_image(command_buffer, image, color)
    }

    #[allow(dead_code)]
    pub(super) fn clear_color_attachment_image(
        &self,
        image: &VulkanOwnedImage,
        color: vk::ClearColorValue,
    ) -> Result<(), VulkanError> {
        let view = self.create_color_attachment_image_view(image)?;
        let render_pass = create_single_color_render_pass(&image.inner.logical_device, image.format())?;
        let framebuffer = create_single_color_framebuffer(&render_pass, &view, image.extent())?;
        let mut command_buffer = self.allocate_graphics_command_buffer()?;

        self.begin_command_buffer(&mut command_buffer)?;
        self.transition_image_layout(
            &mut command_buffer,
            image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;
        record_color_attachment_clear(&mut command_buffer, image, &render_pass, &framebuffer, color)?;
        self.end_command_buffer(&mut command_buffer)?;
        self.submit_graphics_command_buffer_and_wait(&mut command_buffer)
    }

    #[allow(dead_code)]
    pub(super) fn create_color_attachment_image_view(
        &self,
        image: &VulkanOwnedImage,
    ) -> Result<VulkanImageView, VulkanError> {
        self.logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        create_color_attachment_image_view(image)
    }

    #[allow(dead_code)]
    pub(super) fn create_shader_module(
        &self,
        spirv: VulkanShaderSpirv<'_>,
    ) -> Result<VulkanShaderModule, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();

        create_shader_module(&logical_device, spirv)
    }

    #[allow(dead_code)]
    pub(super) fn create_empty_pipeline_layout(&self) -> Result<VulkanPipelineLayout, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();

        create_empty_pipeline_layout(&logical_device)
    }

    #[allow(dead_code)]
    pub(super) fn create_sampled_texture_descriptor_set_layout(
        &self,
    ) -> Result<VulkanDescriptorSetLayout, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();

        create_sampled_texture_descriptor_set_layout(&logical_device)
    }

    #[allow(dead_code)]
    pub(super) fn create_sampled_texture_descriptor_pool(
        &self,
        max_sets: u32,
    ) -> Result<VulkanDescriptorPool, VulkanError> {
        if max_sets == 0 {
            return Err(VulkanError::UnsupportedOperation("descriptor pool capacity"));
        }

        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();

        create_sampled_texture_descriptor_pool(&logical_device, max_sets)
    }

    #[allow(dead_code)]
    pub(super) fn create_sampled_texture_pipeline_layout(
        &self,
    ) -> Result<VulkanSampledTexturePipelineLayout, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();

        let descriptor_set_layout = create_sampled_texture_descriptor_set_layout(&logical_device)?;
        let pipeline_layout = create_pipeline_layout_for_descriptor_set_layout(&descriptor_set_layout)?;

        Ok(VulkanSampledTexturePipelineLayout {
            pipeline_layout,
            descriptor_set_layout,
        })
    }

    #[allow(dead_code)]
    pub(super) fn create_sampled_texture_descriptor_set(
        &self,
        pool: &VulkanDescriptorPool,
        descriptor_set_layout: &VulkanDescriptorSetLayout,
        sampled_image: Arc<VulkanSampledImage>,
    ) -> Result<VulkanSampledTextureDescriptorSet, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;

        create_sampled_texture_descriptor_set(logical_device, pool, descriptor_set_layout, sampled_image)
    }

    #[allow(dead_code)]
    pub(super) fn create_sampled_texture_graphics_pipeline(
        &self,
        shaders: VulkanSampledTexturePipelineShaders<'_>,
    ) -> Result<VulkanSampledTextureGraphicsPipeline, VulkanError> {
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

        let _format_properties = validate_optimal_2d_image_support(
            instance,
            physical_device,
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            shaders.color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )?;
        // SAFETY: `physical_device` belongs to `instance`, and querying format properties is
        // read-only with no additional extension or lifetime requirements.
        let format_properties = unsafe {
            instance
                .handle()
                .get_physical_device_format_properties(physical_device.handle(), shaders.color_format)
        };
        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
        {
            return Err(VulkanError::UnsupportedOperation(
                "graphics pipeline blend format",
            ));
        }

        let vertex_shader = create_shader_module(&logical_device, shaders.vertex)?;
        let fragment_shader = create_shader_module(&logical_device, shaders.fragment)?;
        let descriptor_set_layout = create_sampled_texture_descriptor_set_layout(&logical_device)?;
        let pipeline_layout = create_pipeline_layout_for_descriptor_set_layout(&descriptor_set_layout)?;
        let sampled_texture_layout = VulkanSampledTexturePipelineLayout {
            pipeline_layout,
            descriptor_set_layout,
        };
        let render_pass = create_single_color_load_render_pass(&logical_device, shaders.color_format)?;
        let pipeline = create_sampled_texture_graphics_pipeline(
            &logical_device,
            &render_pass,
            sampled_texture_layout.pipeline_layout(),
            &vertex_shader,
            &fragment_shader,
        )?;

        Ok(VulkanSampledTextureGraphicsPipeline {
            color_format: shaders.color_format,
            render_pass,
            layout: sampled_texture_layout,
            pipeline,
        })
    }

    #[allow(dead_code)]
    pub(super) fn create_builtin_sampled_texture_graphics_pipeline(
        &self,
        color_format: vk::Format,
    ) -> Result<VulkanSampledTextureGraphicsPipeline, VulkanError> {
        // SAFETY: These built-in shader modules were generated from local GLSL by glslangValidator.
        // They contain compatible vertex/fragment `main` entry points, no non-built-in vertex
        // inputs, matching location interfaces, set 0 binding 0 as a combined image sampler, and
        // one color output compatible with the renderer's UNORM color-attachment formats. The
        // fragment shader reads a 24-byte push-constant block containing a UV rectangle, global
        // alpha value, and opaque-alpha flag covered by the pipeline layout.
        let shaders = unsafe {
            VulkanSampledTexturePipelineShaders::from_spirv_unchecked(
                color_format,
                BUILTIN_TEXTURED_VERTEX_SHADER_SPIRV,
                BUILTIN_TEXTURED_FRAGMENT_SHADER_SPIRV,
            )
        }?;

        self.create_sampled_texture_graphics_pipeline(shaders)
    }

    #[allow(dead_code)]
    pub(super) fn render_sampled_texture_to_color_image(
        &self,
        target: &VulkanOwnedImage,
        descriptor_set: &VulkanSampledTextureDescriptorSet,
        pipeline: &VulkanSampledTextureGraphicsPipeline,
    ) -> Result<(), VulkanError> {
        let draw_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: target.extent().width,
                height: target.extent().height,
            },
        };

        self.render_sampled_texture_to_color_image_in(
            target,
            descriptor_set,
            pipeline,
            VulkanSampledTextureDrawConstants {
                draw_area,
                uv_rect: SAMPLED_TEXTURE_FULL_UV_RECT,
                alpha: 1.0,
                force_opaque_alpha: false,
            },
        )
    }

    #[allow(dead_code)]
    pub(super) fn render_sampled_texture_to_color_image_in(
        &self,
        target: &VulkanOwnedImage,
        descriptor_set: &VulkanSampledTextureDescriptorSet,
        pipeline: &VulkanSampledTextureGraphicsPipeline,
        draw_constants: VulkanSampledTextureDrawConstants,
    ) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        validate_sampled_texture_draw_inputs(logical_device, target, descriptor_set, pipeline)?;
        validate_sampled_texture_draw_area(target, draw_constants.draw_area)?;
        validate_sampled_texture_draw_constants(draw_constants)?;

        let view = self.create_color_attachment_image_view(target)?;
        let framebuffer = create_single_color_framebuffer(pipeline.render_pass(), &view, target.extent())?;
        let mut command_buffer = self.allocate_graphics_command_buffer()?;

        self.begin_command_buffer(&mut command_buffer)?;
        self.transition_image_layout(
            &mut command_buffer,
            target,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;
        synchronize_color_attachment_load(&mut command_buffer, target)?;
        record_sampled_texture_draw(
            &mut command_buffer,
            target,
            pipeline,
            descriptor_set,
            &framebuffer,
            draw_constants,
        )?;
        self.end_command_buffer(&mut command_buffer)?;
        self.submit_graphics_command_buffer_and_wait(&mut command_buffer)
    }

    #[allow(dead_code)]
    pub(super) fn read_image_to_tightly_packed_buffer(
        &self,
        image: &VulkanOwnedImage,
    ) -> Result<Vec<u8>, VulkanError> {
        let readback_size = tightly_packed_image_size(image.format(), image.extent())?;
        let readback_buffer =
            self.create_host_visible_buffer(readback_size, vk::BufferUsageFlags::TRANSFER_DST)?;
        let mut command_buffer = self.allocate_graphics_command_buffer()?;

        self.begin_command_buffer(&mut command_buffer)?;
        self.transition_image_layout(&mut command_buffer, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)?;
        self.copy_image_to_buffer(&mut command_buffer, image, &readback_buffer, image.extent())?;
        self.end_command_buffer(&mut command_buffer)?;
        self.submit_graphics_command_buffer_and_wait(&mut command_buffer)?;

        readback_buffer.read()
    }

    #[allow(dead_code)]
    pub(super) fn copy_image_to_buffer(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        image: &VulkanOwnedImage,
        buffer: &VulkanHostVisibleBuffer,
        extent: vk::Extent3D,
    ) -> Result<(), VulkanError> {
        copy_image_to_buffer(command_buffer, image, buffer, extent)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn update_uploaded_image_region(
        &self,
        image: &VulkanOwnedImage,
        data: &[u8],
        buffer_offset: vk::DeviceSize,
        buffer_row_length: u32,
        buffer_image_height: u32,
        image_offset: vk::Offset3D,
        extent: vk::Extent3D,
    ) -> Result<(), VulkanError> {
        let buffer_size = vk::DeviceSize::try_from(data.len())
            .map_err(|_| VulkanError::UnsupportedOperation("image update data size"))?;
        let staging = self.create_host_visible_buffer(buffer_size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        staging.write(data)?;
        let mut command_buffer = self.allocate_graphics_command_buffer()?;

        self.begin_command_buffer(&mut command_buffer)?;
        self.transition_image_layout(&mut command_buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        self.copy_buffer_region_to_image(
            &mut command_buffer,
            &staging,
            image,
            buffer_offset,
            buffer_row_length,
            buffer_image_height,
            image_offset,
            extent,
        )?;
        self.transition_image_layout(
            &mut command_buffer,
            image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;
        self.end_command_buffer(&mut command_buffer)?;
        self.submit_graphics_command_buffer_and_wait(&mut command_buffer)
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
    copy_buffer_region_to_image(
        command_buffer,
        buffer,
        image,
        0,
        0,
        0,
        vk::Offset3D { x: 0, y: 0, z: 0 },
        extent,
    )
}

#[allow(clippy::too_many_arguments)]
fn copy_buffer_region_to_image(
    command_buffer: &mut VulkanCommandBuffer,
    buffer: &VulkanHostVisibleBuffer,
    image: &VulkanOwnedImage,
    buffer_offset: vk::DeviceSize,
    buffer_row_length: u32,
    buffer_image_height: u32,
    image_offset: vk::Offset3D,
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
    if image_offset.x < 0 || image_offset.y < 0 || image_offset.z < 0 {
        return Err(VulkanError::UnsupportedOperation("image copy offset"));
    }
    let image_extent = image.extent();
    let image_offset_x = image_offset.x as u32;
    let image_offset_y = image_offset.y as u32;
    let image_offset_z = image_offset.z as u32;
    if image_offset_x
        .checked_add(extent.width)
        .is_none_or(|width| width > image_extent.width)
        || image_offset_y
            .checked_add(extent.height)
            .is_none_or(|height| height > image_extent.height)
        || image_offset_z
            .checked_add(extent.depth)
            .is_none_or(|depth| depth > image_extent.depth)
    {
        return Err(VulkanError::UnsupportedOperation("image copy extent"));
    }
    let required_size = image_copy_required_size(
        image.format(),
        buffer_offset,
        buffer_row_length,
        buffer_image_height,
        extent,
    )?;
    if required_size > buffer.size() {
        return Err(VulkanError::UnsupportedOperation("image copy buffer size"));
    }

    let image_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    if image_layout != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
        return Err(VulkanError::UnsupportedOperation("image copy layout"));
    }

    let region = vk::BufferImageCopy::default()
        .buffer_offset(buffer_offset)
        .buffer_row_length(buffer_row_length)
        .buffer_image_height(buffer_image_height)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(image_offset)
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

fn clear_color_image(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
    color: vk::ClearColorValue,
) -> Result<(), VulkanError> {
    if !image.usage().contains(vk::ImageUsageFlags::TRANSFER_DST) {
        return Err(VulkanError::UnsupportedOperation(
            "image transfer destination usage",
        ));
    }

    let image_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    if image_layout != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
        return Err(VulkanError::UnsupportedOperation("image clear layout"));
    }

    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_clear_color_image(
                command_buffer.handle,
                image.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &color,
                &[range],
            )
    };

    command_buffer.referenced_images.push(Arc::clone(&image.inner));

    Ok(())
}

fn record_color_attachment_clear(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
    render_pass: &VulkanRenderPass,
    framebuffer: &VulkanFramebuffer,
    color: vk::ClearColorValue,
) -> Result<(), VulkanError> {
    if !image.usage().contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
    }

    let image_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    if image_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
        return Err(VulkanError::UnsupportedOperation("image color attachment layout"));
    }

    let clear_values = [vk::ClearValue { color }];
    let render_area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: image.extent().width,
            height: image.extent().height,
        },
    };
    let begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass.handle)
        .framebuffer(framebuffer.handle)
        .render_area(render_area)
        .clear_values(&clear_values);
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_begin_render_pass(command_buffer.handle, &begin_info, vk::SubpassContents::INLINE);
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_end_render_pass(command_buffer.handle);
    }

    command_buffer.referenced_images.push(Arc::clone(&image.inner));

    Ok(())
}

fn record_sampled_texture_draw(
    command_buffer: &mut VulkanCommandBuffer,
    target: &VulkanOwnedImage,
    pipeline: &VulkanSampledTextureGraphicsPipeline,
    descriptor_set: &VulkanSampledTextureDescriptorSet,
    framebuffer: &VulkanFramebuffer,
    draw_constants: VulkanSampledTextureDrawConstants,
) -> Result<(), VulkanError> {
    if !target.usage().contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
    }
    validate_sampled_texture_draw_inputs(
        &command_buffer.command_pool.logical_device,
        target,
        descriptor_set,
        pipeline,
    )?;
    validate_sampled_texture_draw_area(target, draw_constants.draw_area)?;
    validate_sampled_texture_draw_constants(draw_constants)?;

    let image_layout = command_buffer
        .pending_layout_for(target)?
        .unwrap_or(target.layout()?);
    if image_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
        return Err(VulkanError::UnsupportedOperation("image color attachment layout"));
    }

    let render_area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: target.extent().width,
            height: target.extent().height,
        },
    };
    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 0.0],
        },
    }];
    let begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(pipeline.render_pass().handle())
        .framebuffer(framebuffer.handle)
        .render_area(render_area)
        .clear_values(&clear_values);
    let viewport = [vk::Viewport {
        x: draw_constants.draw_area.offset.x as f32,
        y: draw_constants.draw_area.offset.y as f32,
        width: draw_constants.draw_area.extent.width as f32,
        height: draw_constants.draw_area.extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [draw_constants.draw_area];
    let descriptor_sets = [descriptor_set.handle()];
    let draw_constant_bytes = sampled_texture_draw_constant_bytes(draw_constants);
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    // SAFETY: All bound objects were created from the same logical device by private constructors.
    // `target` is in COLOR_ATTACHMENT_OPTIMAL for the duration of the render pass, the framebuffer
    // uses the same render pass as `pipeline`, dynamic viewport/scissor are set before drawing, and
    // `descriptor_set` retains the sampled image resources it references, and the pipeline layout
    // contains a fragment-stage push-constant range covering the 24 bytes written here. The command
    // buffer is host synchronized by the command-pool lock.
    unsafe {
        let device = command_buffer.command_pool.logical_device.handle();
        device.cmd_begin_render_pass(command_buffer.handle, &begin_info, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(
            command_buffer.handle,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline().handle(),
        );
        device.cmd_bind_descriptor_sets(
            command_buffer.handle,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.layout().pipeline_layout().handle(),
            0,
            &descriptor_sets,
            &[],
        );
        device.cmd_set_viewport(command_buffer.handle, 0, &viewport);
        device.cmd_set_scissor(command_buffer.handle, 0, &scissors);
        device.cmd_push_constants(
            command_buffer.handle,
            pipeline.layout().pipeline_layout().handle(),
            vk::ShaderStageFlags::FRAGMENT,
            0,
            &draw_constant_bytes,
        );
        device.cmd_draw(command_buffer.handle, 3, 1, 0, 0);
        device.cmd_end_render_pass(command_buffer.handle);
    }

    command_buffer.referenced_images.push(Arc::clone(&target.inner));
    command_buffer
        .referenced_images
        .push(Arc::clone(&descriptor_set.sampled_image().image().inner));

    Ok(())
}

fn validate_sampled_texture_draw_inputs(
    logical_device: &VulkanLogicalDevice,
    target: &VulkanOwnedImage,
    descriptor_set: &VulkanSampledTextureDescriptorSet,
    pipeline: &VulkanSampledTextureGraphicsPipeline,
) -> Result<(), VulkanError> {
    if target.format() != pipeline.color_format() {
        return Err(VulkanError::UnsupportedOperation("graphics pipeline format"));
    }
    if !logical_device.is_same_device(&target.inner.logical_device)
        || !logical_device.is_same_device(pipeline.render_pass().logical_device())
        || !logical_device.is_same_device(pipeline.layout().pipeline_layout().logical_device())
        || !logical_device.is_same_device(pipeline.pipeline().logical_device())
        || !logical_device.is_same_device(descriptor_set.pool().logical_device())
        || !logical_device.is_same_device(descriptor_set.sampled_image().logical_device())
    {
        return Err(VulkanError::UnsupportedOperation("sampled texture draw device"));
    }
    if descriptor_set.sampled_image().image().layout()? != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
        return Err(VulkanError::UnsupportedOperation("sampled texture layout"));
    }

    Ok(())
}

fn validate_sampled_texture_draw_area(
    target: &VulkanOwnedImage,
    draw_area: vk::Rect2D,
) -> Result<(), VulkanError> {
    if draw_area.extent.width == 0 || draw_area.extent.height == 0 {
        return Err(VulkanError::UnsupportedOperation("sampled texture draw area"));
    }
    if draw_area.offset.x < 0 || draw_area.offset.y < 0 {
        return Err(VulkanError::UnsupportedOperation("sampled texture draw area"));
    }

    let x_end = u32::try_from(draw_area.offset.x)
        .ok()
        .and_then(|x| x.checked_add(draw_area.extent.width))
        .ok_or(VulkanError::UnsupportedOperation("sampled texture draw area"))?;
    let y_end = u32::try_from(draw_area.offset.y)
        .ok()
        .and_then(|y| y.checked_add(draw_area.extent.height))
        .ok_or(VulkanError::UnsupportedOperation("sampled texture draw area"))?;

    if x_end > target.extent().width || y_end > target.extent().height {
        return Err(VulkanError::UnsupportedOperation("sampled texture draw area"));
    }

    Ok(())
}

fn validate_sampled_texture_draw_constants(
    draw_constants: VulkanSampledTextureDrawConstants,
) -> Result<(), VulkanError> {
    let [u_offset, v_offset, u_scale, v_scale] = draw_constants.uv_rect;
    const UV_RECT_EPSILON: f32 = 0.000_001;
    let u_end = u_offset + u_scale;
    let v_end = v_offset + v_scale;
    let uv_range = -UV_RECT_EPSILON..=1.0 + UV_RECT_EPSILON;

    if !draw_constants
        .uv_rect
        .iter()
        .all(|component| component.is_finite())
        || !draw_constants.alpha.is_finite()
        || u_scale <= 0.0
        || v_scale == 0.0
        || !uv_range.contains(&u_offset)
        || !uv_range.contains(&u_end)
        || !uv_range.contains(&v_offset)
        || !uv_range.contains(&v_end)
        || !(0.0..=1.0).contains(&draw_constants.alpha)
    {
        return Err(VulkanError::UnsupportedOperation(
            "sampled texture draw constants",
        ));
    }

    Ok(())
}

fn sampled_texture_draw_constant_bytes(draw_constants: VulkanSampledTextureDrawConstants) -> [u8; 24] {
    let [u_offset, v_offset, u_scale, v_scale] = draw_constants.uv_rect;
    let force_opaque_alpha = if draw_constants.force_opaque_alpha {
        1.0f32
    } else {
        0.0
    };
    let mut bytes = [0; 24];
    bytes[0..4].copy_from_slice(&u_offset.to_ne_bytes());
    bytes[4..8].copy_from_slice(&v_offset.to_ne_bytes());
    bytes[8..12].copy_from_slice(&u_scale.to_ne_bytes());
    bytes[12..16].copy_from_slice(&v_scale.to_ne_bytes());
    bytes[16..20].copy_from_slice(&draw_constants.alpha.to_ne_bytes());
    bytes[20..24].copy_from_slice(&force_opaque_alpha.to_ne_bytes());
    bytes
}

fn synchronize_color_attachment_load(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
) -> Result<(), VulkanError> {
    if !image.usage().contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
    }

    let image_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    if image_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
        return Err(VulkanError::UnsupportedOperation("image color attachment layout"));
    }

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
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

    // SAFETY: The image is a color attachment in COLOR_ATTACHMENT_OPTIMAL. This same-layout
    // barrier makes prior transfer/color attachment writes visible to the following load-op render
    // pass and subsequent color writes. Queue-family ownership is unchanged.
    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_pipeline_barrier(
                command_buffer.handle,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[barrier],
            )
    };

    command_buffer.referenced_images.push(Arc::clone(&image.inner));

    Ok(())
}

fn copy_image_to_buffer(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
    buffer: &VulkanHostVisibleBuffer,
    extent: vk::Extent3D,
) -> Result<(), VulkanError> {
    if !image.usage().contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        return Err(VulkanError::UnsupportedOperation("image transfer source usage"));
    }
    if !buffer.usage().contains(vk::BufferUsageFlags::TRANSFER_DST) {
        return Err(VulkanError::UnsupportedOperation(
            "buffer transfer destination usage",
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
    let required_size = tightly_packed_image_size(image.format(), extent)?;
    if required_size > buffer.size() {
        return Err(VulkanError::UnsupportedOperation("image copy buffer size"));
    }

    let image_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    if image_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
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
            .cmd_copy_image_to_buffer(
                command_buffer.handle,
                image.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.buffer(),
                &[region],
            )
    };

    command_buffer.referenced_images.push(Arc::clone(&image.inner));
    command_buffer.referenced_buffers.push(Arc::clone(&buffer.inner));

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
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
            if !usage.contains(vk::ImageUsageFlags::SAMPLED) {
                return Err(VulkanError::UnsupportedOperation("image sampled usage"));
            }
            if !usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
                return Err(VulkanError::UnsupportedOperation(
                    "image transfer destination usage",
                ));
            }

            Ok(VulkanLayoutTransition {
                src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
                src_access: vk::AccessFlags::SHADER_READ,
                dst_access: vk::AccessFlags::TRANSFER_WRITE,
            })
        }
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => {
            if !usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
                return Err(VulkanError::UnsupportedOperation(
                    "image transfer destination usage",
                ));
            }
            if !usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
                return Err(VulkanError::UnsupportedOperation("image transfer source usage"));
            }

            Ok(VulkanLayoutTransition {
                src_stage: vk::PipelineStageFlags::TRANSFER,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
                src_access: vk::AccessFlags::TRANSFER_WRITE,
                dst_access: vk::AccessFlags::TRANSFER_READ,
            })
        }
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
            if !usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
                return Err(VulkanError::UnsupportedOperation("image transfer source usage"));
            }
            if !usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
                return Err(VulkanError::UnsupportedOperation(
                    "image transfer destination usage",
                ));
            }

            Ok(VulkanLayoutTransition {
                src_stage: vk::PipelineStageFlags::TRANSFER,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
                src_access: vk::AccessFlags::TRANSFER_READ,
                dst_access: vk::AccessFlags::TRANSFER_WRITE,
            })
        }
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => {
            if !usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
                return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
            }

            Ok(VulkanLayoutTransition {
                src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access: vk::AccessFlags::empty(),
                dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            })
        }
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => {
            if !usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
                return Err(VulkanError::UnsupportedOperation("image transfer source usage"));
            }
            if !usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
                return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
            }

            Ok(VulkanLayoutTransition {
                src_stage: vk::PipelineStageFlags::TRANSFER,
                dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access: vk::AccessFlags::TRANSFER_READ,
                dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            })
        }
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => {
            if !usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
                return Err(VulkanError::UnsupportedOperation(
                    "image transfer destination usage",
                ));
            }
            if !usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
                return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
            }

            Ok(VulkanLayoutTransition {
                src_stage: vk::PipelineStageFlags::TRANSFER,
                dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access: vk::AccessFlags::TRANSFER_WRITE,
                dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            })
        }
        (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => {
            if !usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
                return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
            }
            if !usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
                return Err(VulkanError::UnsupportedOperation("image transfer source usage"));
            }

            Ok(VulkanLayoutTransition {
                src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
                src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access: vk::AccessFlags::TRANSFER_READ,
            })
        }
        _ => Err(VulkanError::UnsupportedOperation("image layout transition")),
    }
}

pub(super) fn image_copy_required_size(
    format: vk::Format,
    buffer_offset: vk::DeviceSize,
    buffer_row_length: u32,
    buffer_image_height: u32,
    extent: vk::Extent3D,
) -> Result<vk::DeviceSize, VulkanError> {
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
        return Err(VulkanError::UnsupportedOperation("zero-sized image copy"));
    }
    if buffer_row_length != 0 && buffer_row_length < extent.width {
        return Err(VulkanError::UnsupportedOperation("image copy row length"));
    }
    if buffer_image_height != 0 && buffer_image_height < extent.height {
        return Err(VulkanError::UnsupportedOperation("image copy image height"));
    }

    let bytes_per_texel = format_bytes_per_texel(format)?;
    let row_length = if buffer_row_length == 0 {
        extent.width
    } else {
        buffer_row_length
    };
    let image_height = if buffer_image_height == 0 {
        extent.height
    } else {
        buffer_image_height
    };

    u64::from(extent.depth - 1)
        .checked_mul(u64::from(image_height))
        .and_then(|rows| rows.checked_add(u64::from(extent.height - 1)))
        .and_then(|rows| rows.checked_mul(u64::from(row_length)))
        .and_then(|texels| texels.checked_add(u64::from(extent.width)))
        .and_then(|texels| texels.checked_mul(bytes_per_texel))
        .and_then(|bytes| bytes.checked_add(buffer_offset))
        .ok_or(VulkanError::UnsupportedOperation("image copy data size"))
}

pub(super) fn image_copy_buffer_offset(
    format: vk::Format,
    row_length: u32,
    x: u32,
    y: u32,
) -> Result<vk::DeviceSize, VulkanError> {
    let bytes_per_texel = format_bytes_per_texel(format)?;

    u64::from(y)
        .checked_mul(u64::from(row_length))
        .and_then(|texels| texels.checked_add(u64::from(x)))
        .and_then(|texels| texels.checked_mul(bytes_per_texel))
        .ok_or(VulkanError::UnsupportedOperation("image copy data offset"))
}

fn format_bytes_per_texel(format: vk::Format) -> Result<vk::DeviceSize, VulkanError> {
    match format {
        vk::Format::R5G6B5_UNORM_PACK16 => Ok(2),
        vk::Format::B8G8R8A8_UNORM
        | vk::Format::R8G8B8A8_UNORM
        | vk::Format::A8B8G8R8_UNORM_PACK32
        | vk::Format::A2R10G10B10_UNORM_PACK32
        | vk::Format::A2B10G10R10_UNORM_PACK32 => Ok(4),
        _ => Err(VulkanError::UnsupportedOperation("tightly packed image format")),
    }
}

pub(super) fn tightly_packed_image_size(
    format: vk::Format,
    extent: vk::Extent3D,
) -> Result<vk::DeviceSize, VulkanError> {
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
        return Err(VulkanError::UnsupportedOperation("zero-sized image"));
    }

    let bytes_per_texel = format_bytes_per_texel(format)?;

    u64::from(extent.width)
        .checked_mul(u64::from(extent.height))
        .and_then(|size| size.checked_mul(u64::from(extent.depth)))
        .and_then(|size| size.checked_mul(bytes_per_texel))
        .ok_or(VulkanError::UnsupportedOperation("image data size"))
}

fn create_image_view(image: &VulkanOwnedImage) -> Result<VulkanImageView, VulkanError> {
    if !image.usage().contains(vk::ImageUsageFlags::SAMPLED) {
        return Err(VulkanError::UnsupportedOperation("image sampled usage"));
    }

    create_image_view_for_color_aspect(image)
}

fn create_color_attachment_image_view(image: &VulkanOwnedImage) -> Result<VulkanImageView, VulkanError> {
    if !image.usage().contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
    }

    create_image_view_for_color_aspect(image)
}

fn create_image_view_for_color_aspect(image: &VulkanOwnedImage) -> Result<VulkanImageView, VulkanError> {
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image.image())
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(image.format())
        .components(vk::ComponentMapping::default())
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe {
        image
            .inner
            .logical_device
            .handle()
            .create_image_view(&view_info, None)
    }
    .map_err(VulkanError::from)?;

    Ok(VulkanImageView {
        image: Arc::clone(&image.inner),
        view,
    })
}

fn create_sampler(
    logical_device: &VulkanLogicalDevice,
    min_filter: TextureFilter,
    mag_filter: TextureFilter,
) -> Result<VulkanSampler, VulkanError> {
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vulkan_filter(mag_filter))
        .min_filter(vulkan_filter(min_filter))
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .mip_lod_bias(0.0)
        .anisotropy_enable(false)
        .max_anisotropy(1.0)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .min_lod(0.0)
        .max_lod(0.0)
        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
        .unnormalized_coordinates(false);
    let sampler =
        unsafe { logical_device.handle().create_sampler(&sampler_info, None) }.map_err(VulkanError::from)?;

    Ok(VulkanSampler {
        logical_device: logical_device.clone(),
        sampler,
        min_filter,
        mag_filter,
    })
}

pub(super) fn vulkan_filter(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Linear => vk::Filter::LINEAR,
        TextureFilter::Nearest => vk::Filter::NEAREST,
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
pub(crate) struct VulkanLogicalDevice {
    device: Arc<ash::Device>,
    _instance: Instance,
}

impl VulkanLogicalDevice {
    fn new(device: ash::Device, instance: Instance) -> Self {
        Self {
            device: Arc::new(device),
            _instance: instance,
        }
    }

    pub(super) fn handle(&self) -> &ash::Device {
        &self.device
    }

    fn is_same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.device, &other.device)
    }
}

impl Drop for VulkanLogicalDevice {
    fn drop(&mut self) {
        if let Some(device) = Arc::get_mut(&mut self.device) {
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

/// Image view for a sampled Vulkan image.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanImageView {
    image: Arc<VulkanOwnedImageInner>,
    view: vk::ImageView,
}

#[allow(dead_code)]
impl VulkanImageView {
    pub(super) fn handle(&self) -> vk::ImageView {
        self.view
    }

    pub(super) fn image(&self) -> vk::Image {
        self.image.image
    }
}

impl Drop for VulkanImageView {
    fn drop(&mut self) {
        unsafe {
            self.image
                .logical_device
                .handle()
                .destroy_image_view(self.view, None)
        };
    }
}

#[derive(Debug)]
pub(crate) struct VulkanRenderPass {
    logical_device: VulkanLogicalDevice,
    handle: vk::RenderPass,
}

#[allow(dead_code)]
impl VulkanRenderPass {
    pub(super) fn handle(&self) -> vk::RenderPass {
        self.handle
    }

    fn logical_device(&self) -> &VulkanLogicalDevice {
        &self.logical_device
    }
}

impl Drop for VulkanRenderPass {
    fn drop(&mut self) {
        unsafe {
            self.logical_device
                .handle()
                .destroy_render_pass(self.handle, None)
        };
    }
}

#[derive(Debug)]
struct VulkanFramebuffer {
    logical_device: VulkanLogicalDevice,
    handle: vk::Framebuffer,
}

impl Drop for VulkanFramebuffer {
    fn drop(&mut self) {
        unsafe {
            self.logical_device
                .handle()
                .destroy_framebuffer(self.handle, None)
        };
    }
}

/// Vulkan shader module owner for future graphics pipelines.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanShaderModule {
    logical_device: VulkanLogicalDevice,
    handle: vk::ShaderModule,
}

#[allow(dead_code)]
impl VulkanShaderModule {
    pub(super) fn handle(&self) -> vk::ShaderModule {
        self.handle
    }

    fn logical_device(&self) -> &VulkanLogicalDevice {
        &self.logical_device
    }
}

/// SPIR-V shader-module code that has been validated by the caller.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct VulkanShaderSpirv<'code> {
    words: &'code [u32],
}

#[allow(dead_code)]
impl<'code> VulkanShaderSpirv<'code> {
    /// Creates a SPIR-V code wrapper without validating the module contents.
    ///
    /// # Safety
    ///
    /// If this returns `Ok`, the caller must ensure `words` contains valid SPIR-V code for a
    /// Vulkan shader module. The slice type guarantees `pCode` alignment and `codeSize` being a
    /// multiple of four. This constructor only rejects empty input before a wrapper is created.
    pub(super) unsafe fn from_words_unchecked(words: &'code [u32]) -> Result<Self, VulkanError> {
        if words.is_empty() {
            return Err(VulkanError::UnsupportedOperation("shader module code"));
        }

        Ok(Self { words })
    }

    fn words(&self) -> &'code [u32] {
        self.words
    }
}

/// Pair of SPIR-V modules compatible with the sampled-texture graphics pipeline scaffold.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct VulkanSampledTexturePipelineShaders<'code> {
    color_format: vk::Format,
    vertex: VulkanShaderSpirv<'code>,
    fragment: VulkanShaderSpirv<'code>,
}

#[allow(dead_code)]
impl<'code> VulkanSampledTexturePipelineShaders<'code> {
    /// Creates a sampled-texture shader pair without validating module contents or interfaces.
    ///
    /// # Safety
    ///
    /// The caller must ensure both slices contain valid SPIR-V modules for Vulkan shader modules.
    /// The vertex module must provide a `main` entry point with the vertex execution model. The
    /// vertex module must not declare non-built-in vertex input attributes, because the scaffold
    /// pipeline uses an empty vertex-input state. The fragment module must provide a `main` entry
    /// point with the fragment execution model. Their location interfaces must match, the fragment
    /// module must use descriptor set 0 binding 0 as a single `COMBINED_IMAGE_SAMPLER`. If the
    /// fragment module reads push constants, those reads must fit inside the first 24 bytes provided
    /// by this sampled-texture pipeline layout. Its color output must be compatible with a single
    /// `color_format` color attachment in subpass 0 of the render pass used by the scaffold.
    pub(super) unsafe fn from_spirv_unchecked(
        color_format: vk::Format,
        vertex_words: &'code [u32],
        fragment_words: &'code [u32],
    ) -> Result<Self, VulkanError> {
        Ok(Self {
            color_format,
            // SAFETY: The safety contract of this constructor includes the shader-module validity
            // required by `VulkanShaderSpirv::from_words_unchecked` for both modules.
            vertex: unsafe { VulkanShaderSpirv::from_words_unchecked(vertex_words)? },
            // SAFETY: The safety contract of this constructor includes the shader-module validity
            // required by `VulkanShaderSpirv::from_words_unchecked` for both modules.
            fragment: unsafe { VulkanShaderSpirv::from_words_unchecked(fragment_words)? },
        })
    }
}

const BUILTIN_TEXTURED_VERTEX_SHADER_SPIRV: &[u32] = &[
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

const BUILTIN_TEXTURED_FRAGMENT_SHADER_SPIRV: &[u32] = &[
    0x07230203, 0x00010000, 0x0008000b, 0x00000037, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x0000001a, 0x00000031, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072, 0x00030005, 0x0000000d, 0x00786574,
    0x00060005, 0x0000000f, 0x77617244, 0x736e6f43, 0x746e6174, 0x00000073, 0x00050006, 0x0000000f,
    0x00000000, 0x725f7675, 0x00746365, 0x00050006, 0x0000000f, 0x00000001, 0x68706c61, 0x00000061,
    0x00080006, 0x0000000f, 0x00000002, 0x63726f66, 0x706f5f65, 0x65757161, 0x706c615f, 0x00006168,
    0x00030005, 0x00000011, 0x00006370, 0x00040005, 0x0000001a, 0x76755f76, 0x00000000, 0x00050005,
    0x00000031, 0x5f74756f, 0x6f6c6f63, 0x00000072, 0x00040047, 0x0000000d, 0x00000021, 0x00000000,
    0x00040047, 0x0000000d, 0x00000022, 0x00000000, 0x00030047, 0x0000000f, 0x00000002, 0x00050048,
    0x0000000f, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000f, 0x00000001, 0x00000023,
    0x00000010, 0x00050048, 0x0000000f, 0x00000002, 0x00000023, 0x00000014, 0x00040047, 0x0000001a,
    0x0000001e, 0x00000000, 0x00040047, 0x00000031, 0x0000001e, 0x00000000, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007,
    0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000007, 0x00000007, 0x00090019, 0x0000000a,
    0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x0003001b,
    0x0000000b, 0x0000000a, 0x00040020, 0x0000000c, 0x00000000, 0x0000000b, 0x0004003b, 0x0000000c,
    0x0000000d, 0x00000000, 0x0005001e, 0x0000000f, 0x00000007, 0x00000006, 0x00000006, 0x00040020,
    0x00000010, 0x00000009, 0x0000000f, 0x0004003b, 0x00000010, 0x00000011, 0x00000009, 0x00040015,
    0x00000012, 0x00000020, 0x00000001, 0x0004002b, 0x00000012, 0x00000013, 0x00000000, 0x00040017,
    0x00000014, 0x00000006, 0x00000002, 0x00040020, 0x00000015, 0x00000009, 0x00000007, 0x00040020,
    0x00000019, 0x00000001, 0x00000014, 0x0004003b, 0x00000019, 0x0000001a, 0x00000001, 0x0004002b,
    0x00000012, 0x00000022, 0x00000002, 0x00040020, 0x00000023, 0x00000009, 0x00000006, 0x0004002b,
    0x00000006, 0x00000026, 0x00000000, 0x00020014, 0x00000027, 0x0004002b, 0x00000006, 0x0000002b,
    0x3f800000, 0x00040015, 0x0000002c, 0x00000020, 0x00000000, 0x0004002b, 0x0000002c, 0x0000002d,
    0x00000003, 0x00040020, 0x0000002e, 0x00000007, 0x00000006, 0x00040020, 0x00000030, 0x00000003,
    0x00000007, 0x0004003b, 0x00000030, 0x00000031, 0x00000003, 0x0004002b, 0x00000012, 0x00000033,
    0x00000001, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x0004003b, 0x00000008, 0x00000009, 0x00000007, 0x0004003d, 0x0000000b, 0x0000000e, 0x0000000d,
    0x00050041, 0x00000015, 0x00000016, 0x00000011, 0x00000013, 0x0004003d, 0x00000007, 0x00000017,
    0x00000016, 0x0007004f, 0x00000014, 0x00000018, 0x00000017, 0x00000017, 0x00000000, 0x00000001,
    0x0004003d, 0x00000014, 0x0000001b, 0x0000001a, 0x00050041, 0x00000015, 0x0000001c, 0x00000011,
    0x00000013, 0x0004003d, 0x00000007, 0x0000001d, 0x0000001c, 0x0007004f, 0x00000014, 0x0000001e,
    0x0000001d, 0x0000001d, 0x00000002, 0x00000003, 0x00050085, 0x00000014, 0x0000001f, 0x0000001b,
    0x0000001e, 0x00050081, 0x00000014, 0x00000020, 0x00000018, 0x0000001f, 0x00050057, 0x00000007,
    0x00000021, 0x0000000e, 0x00000020, 0x0003003e, 0x00000009, 0x00000021, 0x00050041, 0x00000023,
    0x00000024, 0x00000011, 0x00000022, 0x0004003d, 0x00000006, 0x00000025, 0x00000024, 0x000500b7,
    0x00000027, 0x00000028, 0x00000025, 0x00000026, 0x000300f7, 0x0000002a, 0x00000000, 0x000400fa,
    0x00000028, 0x00000029, 0x0000002a, 0x000200f8, 0x00000029, 0x00050041, 0x0000002e, 0x0000002f,
    0x00000009, 0x0000002d, 0x0003003e, 0x0000002f, 0x0000002b, 0x000200f9, 0x0000002a, 0x000200f8,
    0x0000002a, 0x0004003d, 0x00000007, 0x00000032, 0x00000009, 0x00050041, 0x00000023, 0x00000034,
    0x00000011, 0x00000033, 0x0004003d, 0x00000006, 0x00000035, 0x00000034, 0x0005008e, 0x00000007,
    0x00000036, 0x00000032, 0x00000035, 0x0003003e, 0x00000031, 0x00000036, 0x000100fd, 0x00010038,
];

impl Drop for VulkanShaderModule {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was created from `self.logical_device` and this owner destroys it
        // exactly once. `VulkanLogicalDevice` is retained by value, so the device outlives the
        // shader module. The module is not shared, satisfying host synchronization for destruction.
        unsafe {
            self.logical_device
                .handle()
                .destroy_shader_module(self.handle, None)
        };
    }
}

fn create_shader_module(
    logical_device: &VulkanLogicalDevice,
    spirv: VulkanShaderSpirv<'_>,
) -> Result<VulkanShaderModule, VulkanError> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(spirv.words());
    // SAFETY: `logical_device` is a live Vulkan device. `VulkanShaderSpirv` guarantees non-empty
    // caller-validated SPIR-V, while `&[u32]` gives `pCode` proper alignment and a `codeSize` that
    // is a multiple of four. No allocation callbacks are used.
    let handle = unsafe { logical_device.handle().create_shader_module(&create_info, None) }
        .map_err(VulkanError::from)?;

    Ok(VulkanShaderModule {
        logical_device: logical_device.clone(),
        handle,
    })
}

/// Vulkan pipeline layout owner for future graphics pipelines.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanPipelineLayout {
    logical_device: VulkanLogicalDevice,
    handle: vk::PipelineLayout,
}

#[allow(dead_code)]
impl VulkanPipelineLayout {
    pub(super) fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }

    fn logical_device(&self) -> &VulkanLogicalDevice {
        &self.logical_device
    }
}

impl Drop for VulkanPipelineLayout {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was created from `self.logical_device` and this owner destroys it
        // exactly once. `VulkanLogicalDevice` is retained by value, so the device outlives the
        // pipeline layout. The layout is not shared, satisfying host synchronization for destruction.
        unsafe {
            self.logical_device
                .handle()
                .destroy_pipeline_layout(self.handle, None)
        };
    }
}

fn create_empty_pipeline_layout(
    logical_device: &VulkanLogicalDevice,
) -> Result<VulkanPipelineLayout, VulkanError> {
    let create_info = vk::PipelineLayoutCreateInfo::default();
    // SAFETY: `logical_device` is a live Vulkan device. The create info has no descriptor set
    // layouts or push-constant ranges, which is valid for an empty pipeline layout, and no
    // allocation callbacks are used.
    let handle = unsafe { logical_device.handle().create_pipeline_layout(&create_info, None) }
        .map_err(VulkanError::from)?;

    Ok(VulkanPipelineLayout {
        logical_device: logical_device.clone(),
        handle,
    })
}

/// Vulkan graphics pipeline owner for future textured rendering.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanGraphicsPipeline {
    logical_device: VulkanLogicalDevice,
    handle: vk::Pipeline,
}

#[allow(dead_code)]
impl VulkanGraphicsPipeline {
    pub(super) fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    fn logical_device(&self) -> &VulkanLogicalDevice {
        &self.logical_device
    }
}

impl Drop for VulkanGraphicsPipeline {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was created from `self.logical_device` and this owner destroys it
        // exactly once. `VulkanLogicalDevice` is retained by value, so the device outlives the
        // graphics pipeline. The pipeline is not shared, satisfying host synchronization.
        unsafe { self.logical_device.handle().destroy_pipeline(self.handle, None) };
    }
}

/// Descriptor-set layout for binding one sampled texture to a fragment shader.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanDescriptorSetLayout {
    logical_device: VulkanLogicalDevice,
    handle: vk::DescriptorSetLayout,
}

#[allow(dead_code)]
impl VulkanDescriptorSetLayout {
    pub(super) fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }

    fn logical_device(&self) -> &VulkanLogicalDevice {
        &self.logical_device
    }
}

impl Drop for VulkanDescriptorSetLayout {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was created from `self.logical_device` and this owner destroys it
        // exactly once. `VulkanLogicalDevice` is retained by value, so the device outlives the
        // descriptor-set layout. The layout is not shared, satisfying host synchronization.
        unsafe {
            self.logical_device
                .handle()
                .destroy_descriptor_set_layout(self.handle, None)
        };
    }
}

fn create_sampled_texture_descriptor_set_layout(
    logical_device: &VulkanLogicalDevice,
) -> Result<VulkanDescriptorSetLayout, VulkanError> {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: `logical_device` is a live Vulkan device. The single binding has descriptor count 1,
    // a valid descriptor type, and a non-empty shader stage mask. No immutable samplers or
    // allocation callbacks are used.
    let handle = unsafe {
        logical_device
            .handle()
            .create_descriptor_set_layout(&create_info, None)
    }
    .map_err(VulkanError::from)?;

    Ok(VulkanDescriptorSetLayout {
        logical_device: logical_device.clone(),
        handle,
    })
}

/// Pipeline layout and descriptor-set layout pair for future sampled-texture rendering.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanSampledTexturePipelineLayout {
    pipeline_layout: VulkanPipelineLayout,
    descriptor_set_layout: VulkanDescriptorSetLayout,
}

#[allow(dead_code)]
impl VulkanSampledTexturePipelineLayout {
    pub(super) fn pipeline_layout(&self) -> &VulkanPipelineLayout {
        &self.pipeline_layout
    }

    pub(super) fn descriptor_set_layout(&self) -> &VulkanDescriptorSetLayout {
        &self.descriptor_set_layout
    }
}

/// Render-pass, pipeline-layout, and graphics-pipeline bundle for future sampled-texture draws.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanSampledTextureGraphicsPipeline {
    color_format: vk::Format,
    render_pass: VulkanRenderPass,
    layout: VulkanSampledTexturePipelineLayout,
    pipeline: VulkanGraphicsPipeline,
}

#[allow(dead_code)]
impl VulkanSampledTextureGraphicsPipeline {
    pub(super) fn color_format(&self) -> vk::Format {
        self.color_format
    }

    pub(super) fn render_pass(&self) -> &VulkanRenderPass {
        &self.render_pass
    }

    pub(super) fn layout(&self) -> &VulkanSampledTexturePipelineLayout {
        &self.layout
    }

    pub(super) fn pipeline(&self) -> &VulkanGraphicsPipeline {
        &self.pipeline
    }
}

fn create_pipeline_layout_for_descriptor_set_layout(
    descriptor_set_layout: &VulkanDescriptorSetLayout,
) -> Result<VulkanPipelineLayout, VulkanError> {
    let set_layouts = [descriptor_set_layout.handle()];
    let push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(SAMPLED_TEXTURE_DRAW_CONSTANT_SIZE)];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constant_ranges);
    // SAFETY: `descriptor_set_layout.logical_device` is a live Vulkan device and owns the
    // descriptor-set layout handle used here, so the set layout and pipeline layout belong to the
    // same device. The push-constant range is 24 bytes, starts at offset 0, is a multiple of 4, and
    // is exposed to the fragment shader. No allocation callbacks are used.
    let handle = unsafe {
        descriptor_set_layout
            .logical_device
            .handle()
            .create_pipeline_layout(&create_info, None)
    }
    .map_err(VulkanError::from)?;

    Ok(VulkanPipelineLayout {
        logical_device: descriptor_set_layout.logical_device.clone(),
        handle,
    })
}

/// Descriptor pool for future sampled-texture descriptor sets.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VulkanDescriptorPool {
    inner: Arc<VulkanDescriptorPoolInner>,
}

#[derive(Debug)]
struct VulkanDescriptorPoolInner {
    logical_device: VulkanLogicalDevice,
    handle: vk::DescriptorPool,
    max_sets: u32,
    host_access: Mutex<()>,
}

#[allow(dead_code)]
impl VulkanDescriptorPool {
    pub(super) fn handle(&self) -> vk::DescriptorPool {
        self.inner.handle
    }

    pub(super) fn max_sets(&self) -> u32 {
        self.inner.max_sets
    }

    fn lock_host_access(&self) -> Result<MutexGuard<'_, ()>, VulkanError> {
        self.inner
            .host_access
            .lock()
            .map_err(|_| host_synchronization_failed())
    }

    fn logical_device(&self) -> &VulkanLogicalDevice {
        &self.inner.logical_device
    }
}

impl Drop for VulkanDescriptorPoolInner {
    fn drop(&mut self) {
        let _pool_guard = self
            .host_access
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: `self.handle` was created from `self.logical_device` and this owner destroys it
        // exactly once. `VulkanLogicalDevice` is retained by value, so the device outlives the
        // descriptor pool. The host-access lock serializes destruction with future pool allocation,
        // free, and reset operations that use the same lock.
        unsafe {
            self.logical_device
                .handle()
                .destroy_descriptor_pool(self.handle, None)
        };
    }
}

fn create_sampled_texture_descriptor_pool(
    logical_device: &VulkanLogicalDevice,
    max_sets: u32,
) -> Result<VulkanDescriptorPool, VulkanError> {
    if max_sets == 0 {
        return Err(VulkanError::UnsupportedOperation("descriptor pool capacity"));
    }

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(max_sets)];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(max_sets)
        .pool_sizes(&pool_sizes);
    // SAFETY: `logical_device` is a live Vulkan device. Local validation rejects zero `max_sets`,
    // and this create info provides the same non-zero combined-image-sampler descriptor count. No
    // allocation callbacks are used.
    let handle = unsafe { logical_device.handle().create_descriptor_pool(&create_info, None) }
        .map_err(VulkanError::from)?;

    Ok(VulkanDescriptorPool {
        inner: Arc::new(VulkanDescriptorPoolInner {
            logical_device: logical_device.clone(),
            handle,
            max_sets,
            host_access: Mutex::new(()),
        }),
    })
}

fn create_sampled_texture_graphics_pipeline(
    logical_device: &VulkanLogicalDevice,
    render_pass: &VulkanRenderPass,
    pipeline_layout: &VulkanPipelineLayout,
    vertex_shader: &VulkanShaderModule,
    fragment_shader: &VulkanShaderModule,
) -> Result<VulkanGraphicsPipeline, VulkanError> {
    if !logical_device.is_same_device(render_pass.logical_device())
        || !logical_device.is_same_device(pipeline_layout.logical_device())
        || !logical_device.is_same_device(vertex_shader.logical_device())
        || !logical_device.is_same_device(fragment_shader.logical_device())
    {
        return Err(VulkanError::UnsupportedOperation("graphics pipeline device"));
    }

    let entry_point = c"main";
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader.handle())
            .name(entry_point),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader.handle())
            .name(entry_point),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout.handle())
        .render_pass(render_pass.handle())
        .subpass(0);
    // SAFETY: All handles are validated to belong to `logical_device`. Shader modules contain
    // caller-validated SPIR-V and remain alive for the duration of pipeline creation. The render
    // pass has one blend-capable color attachment at subpass 0, and the fixed-function state
    // describes a simple triangle-list pipeline with dynamic viewport/scissor and premultiplied
    // alpha blending. All create-info slices live through the call and no allocation callbacks are
    // used.
    let pipelines = unsafe {
        logical_device
            .handle()
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .map_err(|(pipelines, err)| {
        for pipeline in pipelines {
            // SAFETY: `pipeline` was returned by the failed create call for this device and is not
            // otherwise owned.
            unsafe { logical_device.handle().destroy_pipeline(pipeline, None) };
        }
        VulkanError::from(err)
    })?;

    let handle = pipelines
        .into_iter()
        .next()
        .ok_or_else(|| VulkanError::DeviceInitializationFailed("no graphics pipeline created".to_owned()))?;

    Ok(VulkanGraphicsPipeline {
        logical_device: logical_device.clone(),
        handle,
    })
}

/// Descriptor set binding one uploaded sampled image.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanSampledTextureDescriptorSet {
    pool: VulkanDescriptorPool,
    sampled_image: Arc<VulkanSampledImage>,
    handle: vk::DescriptorSet,
}

#[allow(dead_code)]
impl VulkanSampledTextureDescriptorSet {
    pub(super) fn handle(&self) -> vk::DescriptorSet {
        self.handle
    }

    pub(super) fn pool(&self) -> &VulkanDescriptorPool {
        &self.pool
    }

    pub(super) fn sampled_image(&self) -> &Arc<VulkanSampledImage> {
        &self.sampled_image
    }
}

fn create_sampled_texture_descriptor_set(
    logical_device: &VulkanLogicalDevice,
    pool: &VulkanDescriptorPool,
    descriptor_set_layout: &VulkanDescriptorSetLayout,
    sampled_image: Arc<VulkanSampledImage>,
) -> Result<VulkanSampledTextureDescriptorSet, VulkanError> {
    if !logical_device.is_same_device(pool.logical_device())
        || !logical_device.is_same_device(descriptor_set_layout.logical_device())
        || !logical_device.is_same_device(sampled_image.logical_device())
    {
        return Err(VulkanError::UnsupportedOperation("descriptor set device"));
    }
    if sampled_image.image().layout()? != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
        return Err(VulkanError::UnsupportedOperation("sampled texture layout"));
    }

    let _pool_guard = pool.lock_host_access()?;
    let set_layouts = [descriptor_set_layout.handle()];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool.handle())
        .set_layouts(&set_layouts);
    // SAFETY: `pool` and `descriptor_set_layout` are validated to belong to `logical_device`.
    // The pool is host-locked for allocation and the set-layout slice lives through the call.
    let descriptor_sets = unsafe { logical_device.handle().allocate_descriptor_sets(&allocate_info) }
        .map_err(VulkanError::from)?;
    let handle = descriptor_sets
        .into_iter()
        .next()
        .ok_or_else(|| VulkanError::DeviceInitializationFailed("no descriptor set allocated".to_owned()))?;
    let image_infos = [vk::DescriptorImageInfo::default()
        .sampler(sampled_image.sampler().handle())
        .image_view(sampled_image.view().handle())
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(handle)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_infos)];
    // SAFETY: `handle` was allocated from `pool` on `logical_device`, binding 0 exists in
    // `descriptor_set_layout` as one combined-image-sampler descriptor, and `sampled_image` retains
    // the sampler and image view referenced by this write. The descriptor set is newly allocated and
    // not concurrently accessed.
    unsafe { logical_device.handle().update_descriptor_sets(&writes, &[]) };

    Ok(VulkanSampledTextureDescriptorSet {
        pool: pool.clone(),
        sampled_image,
        handle,
    })
}

fn create_single_color_render_pass(
    logical_device: &VulkanLogicalDevice,
    format: vk::Format,
) -> Result<VulkanRenderPass, VulkanError> {
    create_single_color_render_pass_with_load_op(logical_device, format, vk::AttachmentLoadOp::CLEAR)
}

fn create_single_color_load_render_pass(
    logical_device: &VulkanLogicalDevice,
    format: vk::Format,
) -> Result<VulkanRenderPass, VulkanError> {
    create_single_color_render_pass_with_load_op(logical_device, format, vk::AttachmentLoadOp::LOAD)
}

fn create_single_color_render_pass_with_load_op(
    logical_device: &VulkanLogicalDevice,
    format: vk::Format,
    load_op: vk::AttachmentLoadOp,
) -> Result<VulkanRenderPass, VulkanError> {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(load_op)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let color_attachments = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachments)];
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    let handle = unsafe {
        logical_device
            .handle()
            .create_render_pass(&render_pass_info, None)
    }
    .map_err(VulkanError::from)?;

    Ok(VulkanRenderPass {
        logical_device: logical_device.clone(),
        handle,
    })
}

fn create_single_color_framebuffer(
    render_pass: &VulkanRenderPass,
    view: &VulkanImageView,
    extent: vk::Extent3D,
) -> Result<VulkanFramebuffer, VulkanError> {
    let attachments = [view.handle()];
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass.handle)
        .attachments(&attachments)
        .width(extent.width)
        .height(extent.height)
        .layers(1);
    let handle = unsafe {
        render_pass
            .logical_device
            .handle()
            .create_framebuffer(&framebuffer_info, None)
    }
    .map_err(VulkanError::from)?;

    Ok(VulkanFramebuffer {
        logical_device: render_pass.logical_device.clone(),
        handle,
    })
}

/// Vulkan sampler for uploaded textures.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanSampler {
    logical_device: VulkanLogicalDevice,
    sampler: vk::Sampler,
    min_filter: TextureFilter,
    mag_filter: TextureFilter,
}

#[allow(dead_code)]
impl VulkanSampler {
    pub(super) fn handle(&self) -> vk::Sampler {
        self.sampler
    }

    pub(super) fn min_filter(&self) -> TextureFilter {
        self.min_filter
    }

    pub(super) fn mag_filter(&self) -> TextureFilter {
        self.mag_filter
    }
}

impl Drop for VulkanSampler {
    fn drop(&mut self) {
        unsafe { self.logical_device.handle().destroy_sampler(self.sampler, None) };
    }
}

/// Uploaded sampled image bundle for the future Vulkan texture path.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanSampledImage {
    sampler: VulkanSampler,
    view: VulkanImageView,
    image: VulkanOwnedImage,
}

#[allow(dead_code)]
impl VulkanSampledImage {
    pub(super) fn image(&self) -> &VulkanOwnedImage {
        &self.image
    }

    pub(super) fn view(&self) -> &VulkanImageView {
        &self.view
    }

    pub(super) fn sampler(&self) -> &VulkanSampler {
        &self.sampler
    }

    fn logical_device(&self) -> &VulkanLogicalDevice {
        &self.image.inner.logical_device
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

    pub(super) fn read(&self) -> Result<Vec<u8>, VulkanError> {
        let len = usize::try_from(self.inner.size)
            .map_err(|_| VulkanError::UnsupportedOperation("mapped buffer read size"))?;
        let mapped = unsafe {
            self.inner.logical_device.handle().map_memory(
                self.inner.memory,
                0,
                self.inner.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(VulkanError::from)?;

        let mut data = vec![0; len];
        unsafe {
            ptr::copy_nonoverlapping(mapped.cast::<u8>(), data.as_mut_ptr(), len);
        }

        unsafe { self.inner.logical_device.handle().unmap_memory(self.inner.memory) };

        Ok(data)
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
