use ash::vk;

use crate::backend::{SwapBuffersError, allocator::Fourcc};

/// Host wait timeout for GPU fences, in nanoseconds.
///
/// `vkWaitForFences` / `vkAcquireNextImageKHR` with `u64::MAX` can wedge a compositor thread
/// forever. Five seconds is longer than shader compile hitching and still a hang if the GPU
/// never signals.
pub(crate) const HOST_FENCE_WAIT_TIMEOUT_NS: u64 = 5_000_000_000;

/// Wait for fences with [`HOST_FENCE_WAIT_TIMEOUT_NS`]. Timeout does not fall through to
/// unbounded `vkQueueWaitIdle`.
pub(crate) fn wait_for_fences(
    device: &ash::Device,
    fences: &[vk::Fence],
    wait_all: bool,
) -> Result<(), VulkanError> {
    match unsafe { device.wait_for_fences(fences, wait_all, HOST_FENCE_WAIT_TIMEOUT_NS) } {
        Ok(()) => Ok(()),
        Err(vk::Result::TIMEOUT) => Err(VulkanError::SyncTimeout),
        Err(err) => Err(VulkanError::from(err)),
    }
}

/// Error returned by the Vulkan renderer.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    /// Vulkan renderer support is not available.
    #[error("Vulkan renderer is unavailable")]
    VulkanUnavailable,
    /// A required Vulkan extension is missing.
    #[error("Missing required Vulkan extension: {0}")]
    MissingRequiredExtension(String),
    /// The requested operation is not supported by the current Vulkan renderer capability set.
    #[error("Unsupported Vulkan renderer operation: {0}")]
    UnsupportedOperation(&'static str),
    /// The requested operation is on an intended development path but is not public-advertised yet.
    #[error("Vulkan renderer path is not public-advertised yet: {0}")]
    NotPublicAdvertised(&'static str),
    /// The requested operation requires a Vulkan/device capability that is not available.
    #[error("Vulkan renderer missing capability: {0}")]
    MissingCapability(&'static str),
    /// The requested format is unsupported.
    #[error("Unsupported Vulkan renderer format: {0:?}")]
    UnsupportedFormat(Fourcc),
    /// The requested modifier is unsupported.
    #[error("Unsupported Vulkan renderer modifier")]
    UnsupportedModifier,
    /// The Vulkan device was lost.
    #[error("Vulkan renderer device lost")]
    DeviceLost,
    /// Waiting for synchronization was interrupted.
    #[error("Vulkan renderer synchronization wait interrupted")]
    SyncInterrupted,
    /// A host fence or WSI wait exceeded [`HOST_FENCE_WAIT_TIMEOUT_NS`].
    ///
    /// The GPU may still be using submitted work. Treat the renderer as unusable; do not
    /// follow up with unbounded `vkQueueWaitIdle`.
    #[error("Vulkan renderer synchronization wait timed out")]
    SyncTimeout,
    /// Vulkan API returned an error.
    #[error("Vulkan renderer API error: {0:?}")]
    VulkanApi(ash::vk::Result),
    /// Renderer device initialization failed.
    #[error("Vulkan renderer device initialization failed: {0}")]
    DeviceInitializationFailed(String),
    /// No usable queue family was available for the requested operation.
    #[error("Vulkan renderer queue family unsupported")]
    QueueFamilyUnsupported,
    /// No usable memory type was available for the requested operation.
    #[error("Vulkan renderer memory type unsupported")]
    MemoryTypeUnsupported,
    /// External memory support required by the requested operation is unavailable.
    #[error("Vulkan renderer external memory unsupported")]
    ExternalMemoryUnsupported,
}

impl From<ash::vk::Result> for VulkanError {
    fn from(err: ash::vk::Result) -> Self {
        VulkanError::VulkanApi(err)
    }
}

impl From<VulkanError> for SwapBuffersError {
    fn from(err: VulkanError) -> Self {
        match err {
            VulkanError::VulkanUnavailable
            | VulkanError::MissingRequiredExtension(_)
            | VulkanError::DeviceLost
            | VulkanError::DeviceInitializationFailed(_)
            | VulkanError::QueueFamilyUnsupported
            | VulkanError::ExternalMemoryUnsupported
            | VulkanError::SyncTimeout => SwapBuffersError::ContextLost(Box::new(err)),
            VulkanError::VulkanApi(result) if vulkan_api_result_invalidates_context(result) => {
                SwapBuffersError::ContextLost(Box::new(err))
            }
            VulkanError::UnsupportedOperation(_)
            | VulkanError::NotPublicAdvertised(_)
            | VulkanError::MissingCapability(_)
            | VulkanError::UnsupportedFormat(_)
            | VulkanError::UnsupportedModifier
            | VulkanError::MemoryTypeUnsupported
            | VulkanError::SyncInterrupted
            | VulkanError::VulkanApi(_) => SwapBuffersError::TemporaryFailure(Box::new(err)),
        }
    }
}

pub(crate) fn vulkan_api_result_invalidates_context(result: ash::vk::Result) -> bool {
    matches!(
        result,
        ash::vk::Result::ERROR_DEVICE_LOST
            | ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY
            | ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            | ash::vk::Result::ERROR_INITIALIZATION_FAILED
            | ash::vk::Result::ERROR_EXTENSION_NOT_PRESENT
            | ash::vk::Result::ERROR_FEATURE_NOT_PRESENT
            | ash::vk::Result::ERROR_INCOMPATIBLE_DRIVER
    )
}
