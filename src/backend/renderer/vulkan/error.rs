use crate::backend::{SwapBuffersError, allocator::Fourcc};

/// Error returned by the Vulkan renderer scaffold.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    /// Vulkan renderer support is not available.
    #[error("Vulkan renderer is unavailable")]
    VulkanUnavailable,
    /// A required Vulkan extension is missing.
    #[error("Missing required Vulkan extension: {0}")]
    MissingRequiredExtension(String),
    /// The requested operation is not supported by the scaffold.
    #[error("Unsupported Vulkan renderer operation: {0}")]
    UnsupportedOperation(&'static str),
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
    /// Vulkan API returned an error.
    #[error("Vulkan renderer API error: {0:?}")]
    VulkanApi(ash::vk::Result),
    /// Renderer device initialization failed.
    #[error("Vulkan renderer device initialization failed: {0}")]
    DeviceInitializationFailed(String),
    /// No usable queue family was available for the requested operation.
    #[error("Vulkan renderer queue family unsupported")]
    QueueFamilyUnsupported,
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
            | VulkanError::UnsupportedOperation(_)
            | VulkanError::DeviceLost
            | VulkanError::DeviceInitializationFailed(_)
            | VulkanError::QueueFamilyUnsupported
            | VulkanError::ExternalMemoryUnsupported => SwapBuffersError::ContextLost(Box::new(err)),
            VulkanError::VulkanApi(result) if vulkan_api_result_invalidates_context(result) => {
                SwapBuffersError::ContextLost(Box::new(err))
            }
            err => SwapBuffersError::TemporaryFailure(Box::new(err)),
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
