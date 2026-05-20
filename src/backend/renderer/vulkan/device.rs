use std::sync::Arc;

use ash::vk;

use crate::backend::vulkan::{Instance, PhysicalDevice};

use super::{VulkanError, VulkanRendererCapabilities};

/// Device state placeholder for the future Vulkan renderer implementation.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanDeviceState {
    pub(super) graphics_command_pool: Option<ash::vk::CommandPool>,
    pub(super) transfer_command_pool: Option<ash::vk::CommandPool>,
    pub(super) queues: VulkanQueues,
    pub(super) queue_families: VulkanQueueFamilies,
    pub(super) logical_device: Option<VulkanLogicalDevice>,
    pub(super) physical_device: Option<PhysicalDevice>,
    pub(super) instance: Option<Instance>,
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

        let queues = VulkanQueues {
            graphics: queue_families
                .graphics
                .map(|family| unsafe { logical_device.handle().get_device_queue(family, 0) }),
            transfer: queue_families
                .transfer
                .map(|family| unsafe { logical_device.handle().get_device_queue(family, 0) }),
        };

        let graphics_family = queue_families
            .graphics
            .ok_or(VulkanError::QueueFamilyUnsupported)?;
        let graphics_command_pool = create_command_pool(&logical_device, graphics_family)?;
        let transfer_command_pool = match queue_families
            .transfer
            .map(|family| create_command_pool(&logical_device, family))
            .transpose()
        {
            Ok(pool) => pool,
            Err(err) => {
                unsafe {
                    logical_device
                        .handle()
                        .destroy_command_pool(graphics_command_pool, None)
                };
                return Err(err);
            }
        };

        Ok(Self {
            graphics_command_pool: Some(graphics_command_pool),
            transfer_command_pool,
            queues,
            queue_families,
            logical_device: Some(logical_device),
            physical_device: Some(physical_device),
            instance: Some(instance),
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
            capabilities: VulkanRendererCapabilities::default(),
            enabled_extensions: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub(super) fn allocate_graphics_command_buffer(&self) -> Result<vk::CommandBuffer, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let command_pool = self
            .graphics_command_pool
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing graphics command pool".to_owned()))?;

        allocate_command_buffer(logical_device, command_pool)
    }

    #[allow(dead_code)]
    pub(super) fn allocate_transfer_command_buffer(&self) -> Result<vk::CommandBuffer, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let command_pool = self
            .transfer_command_pool
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing transfer command pool".to_owned()))?;

        allocate_command_buffer(logical_device, command_pool)
    }
}

impl Drop for VulkanDeviceState {
    fn drop(&mut self) {
        if let Some(logical_device) = self.logical_device.as_ref() {
            if let Some(command_pool) = self.transfer_command_pool.take() {
                unsafe { logical_device.handle().destroy_command_pool(command_pool, None) };
            }
            if let Some(command_pool) = self.graphics_command_pool.take() {
                unsafe { logical_device.handle().destroy_command_pool(command_pool, None) };
            }
        }
    }
}

fn create_command_pool(
    logical_device: &VulkanLogicalDevice,
    queue_family_index: u32,
) -> Result<vk::CommandPool, VulkanError> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    unsafe {
        logical_device
            .handle()
            .create_command_pool(&command_pool_info, None)
    }
    .map_err(VulkanError::from)
}

fn allocate_command_buffer(
    logical_device: &VulkanLogicalDevice,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, VulkanError> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffers = unsafe { logical_device.handle().allocate_command_buffers(&allocate_info) }
        .map_err(VulkanError::from)?;

    command_buffers
        .into_iter()
        .next()
        .ok_or_else(|| VulkanError::DeviceInitializationFailed("no command buffer allocated".to_owned()))
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VulkanQueues {
    pub(super) graphics: Option<ash::vk::Queue>,
    pub(super) transfer: Option<ash::vk::Queue>,
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
