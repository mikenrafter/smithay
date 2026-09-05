use std::sync::Arc;

use crate::backend::vulkan::{Instance, PhysicalDevice};

use super::VulkanRendererCapabilities;

/// Device state placeholder for the future Vulkan renderer implementation.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanDeviceState {
    pub(super) command_pool: Option<ash::vk::CommandPool>,
    pub(super) queues: VulkanQueues,
    pub(super) queue_families: VulkanQueueFamilies,
    pub(super) logical_device: Option<VulkanLogicalDevice>,
    pub(super) physical_device: Option<PhysicalDevice>,
    pub(super) instance: Option<Instance>,
    pub(super) capabilities: VulkanRendererCapabilities,
    pub(super) enabled_extensions: Vec<String>,
}

/// Logical device owner placeholder.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct VulkanLogicalDevice(Arc<ash::Device>);

impl std::fmt::Debug for VulkanLogicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("VulkanLogicalDevice").field(&"_").finish()
    }
}

impl VulkanDeviceState {
    #[cfg(test)]
    pub(super) fn empty_for_tests() -> Self {
        Self {
            command_pool: None,
            queues: VulkanQueues::default(),
            queue_families: VulkanQueueFamilies::default(),
            logical_device: None,
            physical_device: None,
            instance: None,
            capabilities: VulkanRendererCapabilities::default(),
            enabled_extensions: Vec::new(),
        }
    }
}

/// Queue family placeholders discovered during device initialization.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct VulkanQueueFamilies {
    pub(super) graphics: Option<u32>,
    pub(super) transfer: Option<u32>,
}

/// Vulkan queues selected for renderer submissions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VulkanQueues {
    pub(super) graphics: Option<ash::vk::Queue>,
    pub(super) transfer: Option<ash::vk::Queue>,
}
