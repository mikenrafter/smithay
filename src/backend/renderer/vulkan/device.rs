use std::{
    collections::HashMap,
    ffi::c_void,
    fmt,
    mem::ManuallyDrop,
    os::fd::{AsFd, AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
    ptr,
    sync::{Arc, Mutex, MutexGuard},
};

use ash::{ext, khr, vk, vk::Handle};

use crate::backend::{
    allocator::dmabuf::Dmabuf,
    renderer::{TextureFilter, sync::SyncPoint},
    vulkan::{Instance, PhysicalDevice},
};

use super::{
    VulkanError, VulkanRendererCapabilities,
    error::vulkan_api_result_invalidates_context,
    format::get_render_vk_format,
    image::{
        VulkanDmabufImportState, VulkanDmabufRenderTargetAcquireRestore, VulkanExternalImageOwnership,
        VulkanExternalMemoryHandleType, VulkanImageSyncState, dmabuf_import_sync_state,
    },
};

const SAMPLED_TEXTURE_DRAW_CONSTANT_SIZE: u32 = 32;
const SOLID_COLOR_DRAW_CONSTANT_SIZE: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum VulkanSingleColorRenderPassLoadOp {
    Clear,
    Load,
}

impl VulkanSingleColorRenderPassLoadOp {
    fn to_vk(self) -> vk::AttachmentLoadOp {
        match self {
            VulkanSingleColorRenderPassLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
            VulkanSingleColorRenderPassLoadOp::Load => vk::AttachmentLoadOp::LOAD,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct VulkanSampledTextureDrawConstants {
    pub(super) draw_area: vk::Rect2D,
    pub(super) scissor_area: vk::Rect2D,
    pub(super) uv_origin: [f32; 2],
    pub(super) uv_x_axis: [f32; 2],
    pub(super) uv_y_axis: [f32; 2],
    pub(super) alpha: f32,
    pub(super) force_opaque_alpha: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct VulkanSolidColorDrawConstants {
    pub(super) draw_area: vk::Rect2D,
    pub(super) scissor_area: vk::Rect2D,
    pub(super) color: [f32; 4],
}

/// Error classification for releasing a sampled dmabuf image to foreign ownership.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum VulkanSampledDmabufForeignReleaseError {
    /// The release was not submitted; renderer-local ownership state may be retried.
    RetrySafe(VulkanError),
    /// Queue release submission was accepted; the same texture must not be retried as locally owned.
    ReleaseSubmitted(VulkanError),
}

impl VulkanSampledDmabufForeignReleaseError {
    pub(crate) fn into_inner(self) -> VulkanError {
        match self {
            VulkanSampledDmabufForeignReleaseError::RetrySafe(err)
            | VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(err) => err,
        }
    }
}

/// Error classification for acquiring a sampled dmabuf image from foreign ownership.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum VulkanSampledDmabufForeignAcquireError {
    /// The acquire was not submitted; no Vulkan ownership-transfer side effect occurred.
    RetrySafe(VulkanError),
    /// Queue acquire submission was accepted or completion became unknowable; ownership must not be
    /// treated as still safely foreign without further device/context recovery, and acquire wait
    /// semaphore payloads may have been consumed or left pending by the queue submission.
    AcquireSubmitted {
        err: VulkanError,
        sampled_image: Option<VulkanSampledImage>,
    },
}

impl VulkanSampledDmabufForeignAcquireError {
    pub(crate) fn into_inner(self) -> VulkanError {
        match self {
            VulkanSampledDmabufForeignAcquireError::RetrySafe(err)
            | VulkanSampledDmabufForeignAcquireError::AcquireSubmitted { err, .. } => err,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
enum VulkanReadySubmitError<T> {
    /// The ready callback was not called and no queue submit was attempted.
    ReservationFailed(VulkanError),
    /// The ready callback returned an error and no queue submit was attempted.
    ReadyCallbackFailed(VulkanError),
    /// The ready callback succeeded, but `vkQueueSubmit` was not accepted.
    ReadyCallbackCommitted { err: VulkanError, ready: T },
    /// Queue submit was accepted, or completion became unknowable, after the ready callback.
    Submitted { err: VulkanError, ready: T },
}

impl<T> VulkanReadySubmitError<T> {
    fn into_inner(self) -> VulkanError {
        match self {
            VulkanReadySubmitError::ReservationFailed(err)
            | VulkanReadySubmitError::ReadyCallbackFailed(err)
            | VulkanReadySubmitError::ReadyCallbackCommitted { err, .. }
            | VulkanReadySubmitError::Submitted { err, .. } => err,
        }
    }
}

/// Error classification for acquiring a sampled dmabuf after an external ready callback ran.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum VulkanSampledDmabufReadyAcquireError<T> {
    /// The ready callback was not called; no Vulkan ownership-transfer side effect occurred.
    RetrySafe(VulkanError),
    /// The ready callback failed before queue submit was attempted.
    ReadyCallbackFailed(VulkanError),
    /// The ready callback succeeded, but queue submit was not accepted. The callback's side effect
    /// must be treated as committed by the caller.
    ReadyCallbackCommitted { err: VulkanError, ready: T },
    /// Queue acquire submission was accepted or completion became unknowable after the callback.
    AcquireSubmitted {
        err: VulkanError,
        sampled_image: VulkanSampledImage,
        ready: T,
    },
}

#[allow(dead_code)]
impl<T> VulkanSampledDmabufReadyAcquireError<T> {
    pub(crate) fn into_inner(self) -> VulkanError {
        match self {
            VulkanSampledDmabufReadyAcquireError::RetrySafe(err)
            | VulkanSampledDmabufReadyAcquireError::ReadyCallbackFailed(err)
            | VulkanSampledDmabufReadyAcquireError::ReadyCallbackCommitted { err, .. }
            | VulkanSampledDmabufReadyAcquireError::AcquireSubmitted { err, .. } => err,
        }
    }
}

pub(super) struct VulkanExternalMemoryDeviceFunctions {
    #[allow(dead_code)]
    pub(super) image_drm_format_modifier: ext::image_drm_format_modifier::Device,
    #[allow(dead_code)]
    pub(super) external_memory_fd: khr::external_memory_fd::Device,
}

pub(super) struct VulkanExternalSyncDeviceFunctions {
    #[allow(dead_code)]
    pub(super) external_semaphore_fd: khr::external_semaphore_fd::Device,
}

/// Binary Vulkan semaphore used for future sync-file interop.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VulkanSyncFileSemaphore {
    inner: Arc<VulkanSyncFileSemaphoreInner>,
}

#[derive(Debug)]
struct VulkanSyncFileSemaphoreInner {
    logical_device: VulkanLogicalDevice,
    handle: vk::Semaphore,
    created_for_sync_file_export: bool,
    imported_from_sync_file: bool,
    host_access: Mutex<()>,
    payload_state: Mutex<VulkanSyncFileSemaphorePayloadState>,
}

/// Sync-file payload to import into a Vulkan semaphore.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum VulkanSyncFileImport {
    Fd(OwnedFd),
    AlreadySignaled,
}

/// Coarse binary semaphore payload state for sync-file submit scaffolding.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VulkanSyncFileSemaphorePayloadState {
    Unsignaled,
    AwaitableSyncFileImport,
    PendingWait,
    PendingSignal,
    Signaled,
}

#[derive(Debug, Clone)]
struct VulkanPendingSemaphorePayload {
    semaphore: VulkanSyncFileSemaphore,
    previous_state: VulkanSyncFileSemaphorePayloadState,
}

/// A sync-file semaphore wait attached to a single Vulkan queue submit.
#[allow(dead_code)]
pub(super) struct VulkanSubmitSemaphoreWait<'a> {
    pub(super) semaphore: &'a VulkanSyncFileSemaphore,
    pub(super) dst_stage: vk::PipelineStageFlags,
}

/// Binary semaphore synchronization attached to a single Vulkan queue submit.
#[allow(dead_code)]
#[derive(Default)]
pub(super) struct VulkanSubmitSynchronization<'a> {
    pub(super) wait_semaphores: Vec<VulkanSubmitSemaphoreWait<'a>>,
    pub(super) signal_semaphores: Vec<&'a VulkanSyncFileSemaphore>,
}

impl VulkanSyncFileImport {
    fn into_raw_fd(self) -> i32 {
        match self {
            VulkanSyncFileImport::Fd(fd) => fd.into_raw_fd(),
            VulkanSyncFileImport::AlreadySignaled => -1,
        }
    }
}

#[allow(dead_code)]
impl<'a> VulkanSubmitSynchronization<'a> {
    pub(super) fn wait_sync_file(
        mut self,
        semaphore: &'a VulkanSyncFileSemaphore,
        dst_stage: vk::PipelineStageFlags,
    ) -> Self {
        self.wait_semaphores
            .push(VulkanSubmitSemaphoreWait { semaphore, dst_stage });
        self
    }

    pub(super) fn signal_sync_file(mut self, semaphore: &'a VulkanSyncFileSemaphore) -> Self {
        self.signal_semaphores.push(semaphore);
        self
    }

    fn validate(&self, logical_device: &VulkanLogicalDevice) -> Result<(), VulkanError> {
        let mut seen_handles = Vec::with_capacity(self.wait_semaphores.len() + self.signal_semaphores.len());

        for wait in &self.wait_semaphores {
            validate_submit_wait_stage(wait.dst_stage)?;
            validate_submit_semaphore(logical_device, wait.semaphore, &mut seen_handles)?;
        }
        for semaphore in &self.signal_semaphores {
            validate_submit_semaphore(logical_device, semaphore, &mut seen_handles)?;
        }

        Ok(())
    }

    fn wait_handles(&self) -> Vec<vk::Semaphore> {
        self.wait_semaphores
            .iter()
            .map(|wait| wait.semaphore.handle())
            .collect()
    }

    fn wait_stage_masks(&self) -> Vec<vk::PipelineStageFlags> {
        self.wait_semaphores.iter().map(|wait| wait.dst_stage).collect()
    }

    fn signal_handles(&self) -> Vec<vk::Semaphore> {
        self.signal_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect()
    }

    fn lock_host_access(&self) -> Result<Vec<MutexGuard<'_, ()>>, VulkanError> {
        let mut semaphores = Vec::with_capacity(self.wait_semaphores.len() + self.signal_semaphores.len());
        for wait in &self.wait_semaphores {
            semaphores.push(wait.semaphore);
        }
        for semaphore in &self.signal_semaphores {
            semaphores.push(*semaphore);
        }
        semaphores.sort_by_key(|semaphore| semaphore.handle().as_raw());

        let mut guards = Vec::with_capacity(semaphores.len());
        for semaphore in semaphores {
            guards.push(semaphore.lock_host_access()?);
        }

        Ok(guards)
    }

    fn mark_payloads_pending_submit(&self) -> Result<Vec<VulkanPendingSemaphorePayload>, VulkanError> {
        let mut pending = Vec::with_capacity(self.wait_semaphores.len() + self.signal_semaphores.len());

        for wait in &self.wait_semaphores {
            let mut state = wait
                .semaphore
                .inner
                .payload_state
                .lock()
                .map_err(|_| host_synchronization_failed())?;
            match *state {
                VulkanSyncFileSemaphorePayloadState::AwaitableSyncFileImport
                | VulkanSyncFileSemaphorePayloadState::Signaled => {
                    pending.push(VulkanPendingSemaphorePayload {
                        semaphore: wait.semaphore.clone(),
                        previous_state: *state,
                    });
                    *state = VulkanSyncFileSemaphorePayloadState::PendingWait;
                }
                VulkanSyncFileSemaphorePayloadState::Unsignaled
                | VulkanSyncFileSemaphorePayloadState::PendingWait
                | VulkanSyncFileSemaphorePayloadState::PendingSignal => {
                    drop(state);
                    restore_pending_semaphore_payloads(&pending)?;
                    return Err(VulkanError::UnsupportedOperation("semaphore wait payload"));
                }
            }
        }
        for semaphore in &self.signal_semaphores {
            let mut state = semaphore
                .inner
                .payload_state
                .lock()
                .map_err(|_| host_synchronization_failed())?;
            match *state {
                VulkanSyncFileSemaphorePayloadState::Unsignaled => {
                    pending.push(VulkanPendingSemaphorePayload {
                        semaphore: (*semaphore).clone(),
                        previous_state: *state,
                    });
                    *state = VulkanSyncFileSemaphorePayloadState::PendingSignal;
                }
                VulkanSyncFileSemaphorePayloadState::AwaitableSyncFileImport
                | VulkanSyncFileSemaphorePayloadState::PendingWait
                | VulkanSyncFileSemaphorePayloadState::PendingSignal
                | VulkanSyncFileSemaphorePayloadState::Signaled => {
                    drop(state);
                    restore_pending_semaphore_payloads(&pending)?;
                    return Err(VulkanError::UnsupportedOperation("semaphore signal payload"));
                }
            }
        }

        Ok(pending)
    }
}

fn restore_pending_semaphore_payloads(pending: &[VulkanPendingSemaphorePayload]) -> Result<(), VulkanError> {
    for pending in pending.iter().rev() {
        pending.semaphore.set_payload_state(pending.previous_state)?;
    }
    Ok(())
}

fn complete_pending_semaphore_payloads(pending: &[VulkanPendingSemaphorePayload]) -> Result<(), VulkanError> {
    for pending in pending {
        let state = *pending
            .semaphore
            .inner
            .payload_state
            .lock()
            .map_err(|_| host_synchronization_failed())?;
        let completed_state = match state {
            VulkanSyncFileSemaphorePayloadState::PendingWait => {
                VulkanSyncFileSemaphorePayloadState::Unsignaled
            }
            VulkanSyncFileSemaphorePayloadState::PendingSignal => {
                VulkanSyncFileSemaphorePayloadState::Signaled
            }
            VulkanSyncFileSemaphorePayloadState::Unsignaled
                if pending.previous_state == VulkanSyncFileSemaphorePayloadState::Unsignaled =>
            {
                // A pending signal may already have been exported as SYNC_FD. Export consumes the
                // semaphore payload back to the unsignaled state, but the submitted batch still
                // refers to the semaphore object until the fence completes.
                VulkanSyncFileSemaphorePayloadState::Unsignaled
            }
            VulkanSyncFileSemaphorePayloadState::Unsignaled
            | VulkanSyncFileSemaphorePayloadState::AwaitableSyncFileImport
            | VulkanSyncFileSemaphorePayloadState::Signaled => {
                return Err(VulkanError::UnsupportedOperation("semaphore submit payload"));
            }
        };
        pending.semaphore.set_payload_state(completed_state)?;
    }

    Ok(())
}

#[allow(dead_code)]
pub(super) fn validate_submit_wait_stage(dst_stage: vk::PipelineStageFlags) -> Result<(), VulkanError> {
    if dst_stage.is_empty() {
        return Err(VulkanError::UnsupportedOperation("semaphore wait stage"));
    }

    Ok(())
}

fn validate_submit_semaphore(
    logical_device: &VulkanLogicalDevice,
    semaphore: &VulkanSyncFileSemaphore,
    seen_handles: &mut Vec<vk::Semaphore>,
) -> Result<(), VulkanError> {
    if !logical_device.is_same_device(&semaphore.inner.logical_device) {
        return Err(VulkanError::UnsupportedOperation("semaphore device"));
    }
    if seen_handles.contains(&semaphore.handle()) {
        return Err(VulkanError::UnsupportedOperation("semaphore submit duplicate"));
    }

    seen_handles.push(semaphore.handle());
    Ok(())
}

#[allow(dead_code)]
impl VulkanSyncFileSemaphore {
    pub(super) fn handle(&self) -> vk::Semaphore {
        self.inner.handle
    }

    #[cfg(test)]
    pub(super) fn payload_state_for_tests(&self) -> Result<VulkanSyncFileSemaphorePayloadState, VulkanError> {
        self.inner
            .payload_state
            .lock()
            .map(|state| *state)
            .map_err(|_| host_synchronization_failed())
    }

    fn lock_host_access(&self) -> Result<MutexGuard<'_, ()>, VulkanError> {
        self.inner
            .host_access
            .lock()
            .map_err(|_| host_synchronization_failed())
    }

    fn set_payload_state(&self, new_state: VulkanSyncFileSemaphorePayloadState) -> Result<(), VulkanError> {
        *self
            .inner
            .payload_state
            .lock()
            .map_err(|_| host_synchronization_failed())? = new_state;
        Ok(())
    }

    fn mark_sync_file_imported(&self, already_signaled: bool) -> Result<(), VulkanError> {
        self.set_payload_state(if already_signaled {
            VulkanSyncFileSemaphorePayloadState::Signaled
        } else {
            VulkanSyncFileSemaphorePayloadState::AwaitableSyncFileImport
        })
    }

    fn exportable_payload_state(&self) -> Result<VulkanSyncFileSemaphorePayloadState, VulkanError> {
        let state = *self
            .inner
            .payload_state
            .lock()
            .map_err(|_| host_synchronization_failed())?;

        match state {
            VulkanSyncFileSemaphorePayloadState::Signaled
            | VulkanSyncFileSemaphorePayloadState::PendingSignal => Ok(state),
            VulkanSyncFileSemaphorePayloadState::Unsignaled
            | VulkanSyncFileSemaphorePayloadState::AwaitableSyncFileImport
            | VulkanSyncFileSemaphorePayloadState::PendingWait => {
                Err(VulkanError::UnsupportedOperation("sync-file semaphore payload"))
            }
        }
    }

    fn can_export_sync_file(&self, export_from_imported: bool) -> bool {
        self.inner.created_for_sync_file_export
            && (!self.inner.imported_from_sync_file || export_from_imported)
    }
}

impl Drop for VulkanSyncFileSemaphoreInner {
    fn drop(&mut self) {
        let _guard = self
            .host_access
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: `self.handle` was created from `self.logical_device`, has not been destroyed yet,
        // and the host-access mutex prevents concurrent host operations on this semaphore while it
        // is being destroyed. No allocation callbacks are used.
        unsafe { self.logical_device.handle().destroy_semaphore(self.handle, None) };
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VulkanDmabufExternalImageFormatProperties {
    pub(super) image_format_properties: vk::ImageFormatProperties,
    pub(super) external_memory_properties: vk::ExternalMemoryProperties,
    pub(super) importable: bool,
    pub(super) dedicated_only: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VulkanDmabufImportCandidate {
    pub(super) properties: VulkanDmabufExternalImageFormatProperties,
    pub(super) dedicated_only: bool,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanDmabufImportImage {
    pub(super) image: VulkanUnboundImage,
    pub(super) memory_requirements: vk::MemoryRequirements,
    pub(super) candidate: VulkanDmabufImportCandidate,
}

impl VulkanDmabufExternalImageFormatProperties {
    fn supports_single_sample_import_extent(&self, import: &VulkanDmabufImportState) -> bool {
        let Ok(width) = u32::try_from(import.size.w) else {
            return false;
        };
        let Ok(height) = u32::try_from(import.size.h) else {
            return false;
        };

        if width == 0 || height == 0 {
            return false;
        }

        self.importable
            && self
                .image_format_properties
                .sample_counts
                .contains(vk::SampleCountFlags::TYPE_1)
            && self.image_format_properties.max_mip_levels >= 1
            && self.image_format_properties.max_array_layers >= 1
            && self.image_format_properties.max_extent.width >= width
            && self.image_format_properties.max_extent.height >= height
            && self.image_format_properties.max_extent.depth >= 1
    }

    pub(super) fn supports_sampled_import(&self, import: &VulkanDmabufImportState) -> bool {
        self.supports_single_sample_import_extent(import)
    }

    pub(super) fn supports_render_target_import(&self, import: &VulkanDmabufImportState) -> bool {
        import.plane_count() == 1 && self.supports_single_sample_import_extent(import)
    }
}

impl VulkanExternalMemoryDeviceFunctions {
    fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
        Self {
            image_drm_format_modifier: ext::image_drm_format_modifier::Device::new(instance, device),
            external_memory_fd: khr::external_memory_fd::Device::new(instance, device),
        }
    }
}

impl VulkanExternalSyncDeviceFunctions {
    fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
        Self {
            external_semaphore_fd: khr::external_semaphore_fd::Device::new(instance, device),
        }
    }
}

impl fmt::Debug for VulkanExternalMemoryDeviceFunctions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanExternalMemoryDeviceFunctions")
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for VulkanExternalSyncDeviceFunctions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanExternalSyncDeviceFunctions")
            .finish_non_exhaustive()
    }
}

/// Device state for the provisional Vulkan in-memory/offscreen renderer.
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
    pub(super) external_memory_fns: Option<VulkanExternalMemoryDeviceFunctions>,
    pub(super) external_sync_fns: Option<VulkanExternalSyncDeviceFunctions>,
    builtin_sampled_texture_pipelines:
        Mutex<HashMap<(vk::Format, bool), Arc<VulkanSampledTextureGraphicsPipeline>>>,
    builtin_solid_color_pipelines: Mutex<HashMap<(vk::Format, bool), Arc<VulkanSolidColorGraphicsPipeline>>>,
    single_color_render_passes:
        Mutex<HashMap<(vk::Format, VulkanSingleColorRenderPassLoadOp), Arc<VulkanRenderPass>>>,
    sampled_texture_descriptor_set_layout: Mutex<Option<Arc<VulkanDescriptorSetLayout>>>,
    sampled_texture_pipeline_layout: Mutex<Option<Arc<VulkanSampledTexturePipelineLayout>>>,
    solid_color_pipeline_layout: Mutex<Option<Arc<VulkanPipelineLayout>>>,
    pending_graphics_submissions: Mutex<Vec<VulkanSubmittedCommandBuffer>>,
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
        let external_memory = super::VulkanExternalMemoryCapabilities::discover(&physical_device);
        let external_sync = super::VulkanExternalSyncCapabilities::discover(&physical_device);
        let mut enabled_device_extensions = Vec::new();
        if external_memory.prerequisites_available {
            enabled_device_extensions.extend(
                super::VulkanExternalMemoryCapabilities::required_device_extensions(
                    physical_device.api_version(),
                ),
            );
        }
        if external_sync.prerequisites_available {
            enabled_device_extensions.extend(
                super::VulkanExternalSyncCapabilities::required_device_extensions(
                    physical_device.api_version(),
                ),
            );
        }
        let enabled_extension_pointers = enabled_device_extensions
            .iter()
            .copied()
            .map(std::ffi::CStr::as_ptr)
            .collect::<Vec<_>>();
        let mut capabilities = VulkanRendererCapabilities::for_initialized_device(&enabled_device_extensions);
        capabilities.external_memory = external_memory;
        capabilities.external_sync = external_sync;
        capabilities.formats =
            super::VulkanFormatCapabilities::discover(&physical_device, &capabilities.external_memory)?;
        capabilities.import.memory = capabilities.formats.memory_import.iter().next().is_some();
        let has_public_render_target_formats = capabilities
            .formats
            .render_target_formats()
            .iter()
            .next()
            .is_some();
        capabilities.rendering.offscreen = has_public_render_target_formats;
        let has_dmabuf_render_target_formats =
            capabilities.formats.dmabuf_render_target.iter().next().is_some();
        capabilities.rendering.dmabuf_target_development = has_dmabuf_render_target_formats;
        capabilities.export.memory = has_public_render_target_formats;

        let queue_priorities = [1.0];
        let queue_create_infos = queue_families
            .unique_indices()
            .map(|queue_family_index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&queue_priorities)
            })
            .collect::<Vec<_>>();
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&enabled_extension_pointers);

        // SAFETY: `physical_device` was enumerated from `instance`, every queue family index comes
        // from the same physical device query, and `queue_priorities`/extension-name pointers live
        // through the call. External-memory and external-sync extension names are only included
        // after device-extension discovery reports the full prerequisite set for this API version;
        // promoted core functionality is not redundantly enabled as an extension. No allocation
        // callbacks are used.
        let logical_device = unsafe {
            instance
                .handle()
                .create_device(physical_device.handle(), &create_info, None)
        }
        .map_err(VulkanError::from)?;
        let logical_device = VulkanLogicalDevice::new(logical_device, instance.clone());
        let external_memory_fns = if capabilities.external_memory.prerequisites_available {
            Some(VulkanExternalMemoryDeviceFunctions::new(
                instance.handle(),
                logical_device.handle(),
            ))
        } else {
            None
        };
        let external_sync_fns = if capabilities.external_sync.prerequisites_available {
            Some(VulkanExternalSyncDeviceFunctions::new(
                instance.handle(),
                logical_device.handle(),
            ))
        } else {
            None
        };

        let graphics_family = queue_families
            .graphics
            .ok_or(VulkanError::QueueFamilyUnsupported)?;
        let graphics_queue = VulkanQueue::new(
            unsafe { logical_device.handle().get_device_queue(graphics_family, 0) },
            graphics_family,
        );
        let transfer_queue = match queue_families.transfer {
            Some(transfer_family) if transfer_family == graphics_family => Some(graphics_queue.clone()),
            Some(transfer_family) => Some(VulkanQueue::new(
                unsafe { logical_device.handle().get_device_queue(transfer_family, 0) },
                transfer_family,
            )),
            None => None,
        };
        let queues = VulkanQueues {
            graphics: Some(graphics_queue),
            transfer: transfer_queue,
        };

        let graphics_command_pool = create_command_pool(&logical_device, graphics_family, true)?;
        let transfer_command_pool = queue_families
            .transfer
            .map(|family| create_command_pool(&logical_device, family, family == graphics_family))
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
            enabled_extensions: enabled_device_extensions
                .iter()
                .map(|extension| extension.to_string_lossy().into_owned())
                .collect(),
            external_memory_fns,
            external_sync_fns,
            builtin_sampled_texture_pipelines: Mutex::new(HashMap::new()),
            builtin_solid_color_pipelines: Mutex::new(HashMap::new()),
            single_color_render_passes: Mutex::new(HashMap::new()),
            sampled_texture_descriptor_set_layout: Mutex::new(None),
            sampled_texture_pipeline_layout: Mutex::new(None),
            solid_color_pipeline_layout: Mutex::new(None),
            pending_graphics_submissions: Mutex::new(Vec::new()),
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
            external_memory_fns: None,
            external_sync_fns: None,
            builtin_sampled_texture_pipelines: Mutex::new(HashMap::new()),
            builtin_solid_color_pipelines: Mutex::new(HashMap::new()),
            single_color_render_passes: Mutex::new(HashMap::new()),
            sampled_texture_descriptor_set_layout: Mutex::new(None),
            sampled_texture_pipeline_layout: Mutex::new(None),
            solid_color_pipeline_layout: Mutex::new(None),
            pending_graphics_submissions: Mutex::new(Vec::new()),
        }
    }

    fn collect_completed_graphics_submissions(&self) -> Result<(), VulkanError> {
        let mut pending = self
            .pending_graphics_submissions
            .lock()
            .map_err(|_| host_synchronization_failed())?;
        let mut index = 0;
        while index < pending.len() {
            if pending[index].is_complete()? {
                let submission = pending.swap_remove(index);
                submission.complete()?;
            } else {
                index += 1;
            }
        }

        Ok(())
    }

    fn retain_graphics_submission(
        &self,
        submission: VulkanSubmittedCommandBuffer,
    ) -> Result<(), VulkanError> {
        if let Err(err) = self.collect_completed_graphics_submissions() {
            if vulkan_error_invalidates_context(&err) {
                return Err(err);
            }
            tracing::warn!(?err, "failed to collect completed Vulkan graphics submissions");
        }
        match self.pending_graphics_submissions.lock() {
            Ok(mut pending) => pending.push(submission),
            Err(_) => {
                tracing::warn!(
                    "Vulkan graphics submission tracking lock poisoned; leaking pending submission"
                );
                std::mem::forget(submission);
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn allocate_graphics_command_buffer(&self) -> Result<VulkanCommandBuffer, VulkanError> {
        self.collect_completed_graphics_submissions()?;
        self.logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let command_pool = self.graphics_command_pool.as_ref().ok_or_else(|| {
            VulkanError::DeviceInitializationFailed("missing graphics command pool".to_owned())
        })?;

        allocate_command_buffer(command_pool)
    }

    #[allow(dead_code)]
    fn create_sync_file_semaphore(
        &self,
        created_for_sync_file_export: bool,
        imported_from_sync_file: bool,
    ) -> Result<VulkanSyncFileSemaphore, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();
        let sync_fd = vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD;
        let mut export_info = vk::ExportSemaphoreCreateInfo::default().handle_types(sync_fd);
        let mut semaphore_info = vk::SemaphoreCreateInfo::default();
        if created_for_sync_file_export {
            semaphore_info = semaphore_info.push_next(&mut export_info);
        }

        // SAFETY: `logical_device` is live. The optional pNext chain contains a Vulkan-defined
        // export-create struct whose stack storage outlives the call, and `SYNC_FD` export creation
        // is only requested by callers after the physical-device property query reports export
        // support. This creates a binary semaphore with no queue use yet. No allocation callbacks
        // are used.
        let handle = unsafe { logical_device.handle().create_semaphore(&semaphore_info, None) }
            .map_err(VulkanError::from)?;

        Ok(VulkanSyncFileSemaphore {
            inner: Arc::new(VulkanSyncFileSemaphoreInner {
                logical_device,
                handle,
                created_for_sync_file_export,
                imported_from_sync_file,
                host_access: Mutex::new(()),
                payload_state: Mutex::new(VulkanSyncFileSemaphorePayloadState::Unsignaled),
            }),
        })
    }

    #[allow(dead_code)]
    pub(crate) fn create_exportable_sync_file_semaphore(
        &self,
    ) -> Result<VulkanSyncFileSemaphore, VulkanError> {
        if !self.capabilities.external_sync.sync_file_exportable || self.external_sync_fns.is_none() {
            return Err(VulkanError::UnsupportedOperation("sync-file semaphore export"));
        }

        self.create_sync_file_semaphore(true, false)
    }

    pub(crate) fn can_export_sync_file(&self) -> bool {
        self.capabilities.external_sync.sync_file_exportable && self.external_sync_fns.is_some()
    }

    /// Import a Linux sync-file fd into a temporary Vulkan binary semaphore payload.
    ///
    /// # Safety
    ///
    /// If `sync_file` is [`VulkanSyncFileImport::Fd`], the fd must be a valid Linux sync-file fd
    /// suitable for `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`. Use
    /// [`VulkanSyncFileImport::AlreadySignaled`] for Vulkan's special `-1` already-signaled import.
    #[allow(dead_code)]
    pub(crate) unsafe fn import_sync_file_semaphore(
        &self,
        sync_file: VulkanSyncFileImport,
    ) -> Result<VulkanSyncFileSemaphore, VulkanError> {
        if !self.capabilities.external_sync.sync_file_importable || self.external_sync_fns.is_none() {
            return Err(VulkanError::UnsupportedOperation("sync-file semaphore import"));
        }
        let external_sync_fns = self
            .external_sync_fns
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("sync-file semaphore import"))?;
        let export_from_imported = self.capabilities.external_sync.sync_file_export_from_imported;
        let semaphore = self.create_sync_file_semaphore(export_from_imported, true)?;
        let raw_fd = sync_file.into_raw_fd();
        let already_signaled = raw_fd == -1;
        let import_info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore.handle())
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .fd(raw_fd);

        // SAFETY: `external_sync_fns` was loaded only when VK_KHR_external_semaphore_fd was enabled
        // on this live device. `semaphore` was freshly created from the same logical device and has
        // no pending queue use. `SYNC_FD` has copy transference and therefore uses TEMPORARY import.
        // `raw_fd` either comes from `OwnedFd::into_raw_fd` or is Vulkan's special `-1` already
        // signaled value. On success Vulkan owns a non-`-1` fd, and on failure a non-`-1` fd is
        // reconstructed and closed below.
        let result = unsafe {
            external_sync_fns
                .external_semaphore_fd
                .import_semaphore_fd(&import_info)
        };
        match result {
            Ok(()) => {
                semaphore.mark_sync_file_imported(already_signaled)?;
                Ok(semaphore)
            }
            Err(err) => {
                if raw_fd != -1 {
                    // SAFETY: A failed import does not take ownership of `raw_fd`; reconstructing
                    // the `OwnedFd` closes it exactly once. The semaphore wrapper is dropped
                    // afterwards and destroys the Vulkan semaphore. `-1` is not an owned fd and was
                    // handled above.
                    unsafe { drop(OwnedFd::from_raw_fd(raw_fd)) };
                }
                Err(VulkanError::from(err))
            }
        }
    }

    /// Convert a Smithay sync point into a Vulkan sync-file wait semaphore when possible.
    ///
    /// Already-signaled sync points need no Vulkan wait. Exportable sync points are imported as
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT` binary semaphore payloads. Non-exportable
    /// sync points, or exportable sync points whose export fails, are waited on by the CPU before
    /// returning without a semaphore.
    ///
    /// # Safety
    ///
    /// If this device supports sync-file import and `sync` exports a fence fd, that fd must be a
    /// valid Linux sync-file fd suitable for `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`.
    #[allow(dead_code)]
    pub(crate) unsafe fn import_sync_point_wait_semaphore(
        &self,
        sync: &SyncPoint,
    ) -> Result<Option<VulkanSyncFileSemaphore>, VulkanError> {
        if !sync.contains_fence() || sync.is_reached() {
            return Ok(None);
        }

        if sync.is_exportable()
            && self.capabilities.external_sync.sync_file_importable
            && self.external_sync_fns.is_some()
        {
            if let Some(fd) = sync.export() {
                // SAFETY: Forwarded from this method's caller.
                return unsafe { self.import_sync_file_semaphore(VulkanSyncFileImport::Fd(fd)) }.map(Some);
            }
        }

        sync.wait().map_err(|_| VulkanError::SyncInterrupted)?;
        Ok(None)
    }

    /// Export a signaled or pending-signaled Vulkan semaphore payload as a Linux sync-file fd.
    ///
    /// # Safety
    ///
    /// The caller must ensure the semaphore satisfies the `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`
    /// export valid-usage rules: there must be no queue currently waiting on it, it must be binary,
    /// and it must be signaled or have a submitted pending signal operation whose dependencies have
    /// also been submitted.
    #[allow(dead_code)]
    pub(crate) unsafe fn export_sync_file_semaphore(
        &self,
        semaphore: &VulkanSyncFileSemaphore,
    ) -> Result<Option<OwnedFd>, VulkanError> {
        if !self.capabilities.external_sync.sync_file_exportable || self.external_sync_fns.is_none() {
            return Err(VulkanError::UnsupportedOperation("sync-file semaphore export"));
        }
        if !semaphore.can_export_sync_file(self.capabilities.external_sync.sync_file_export_from_imported) {
            return Err(VulkanError::UnsupportedOperation("sync-file semaphore export"));
        }
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        if !logical_device.is_same_device(&semaphore.inner.logical_device) {
            return Err(VulkanError::UnsupportedOperation("sync-file semaphore export"));
        }
        let external_sync_fns = self
            .external_sync_fns
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("sync-file semaphore export"))?;
        let _semaphore_guard = semaphore.lock_host_access()?;
        semaphore.exportable_payload_state()?;
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore.handle())
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);

        // SAFETY: The caller upholds the sync-file semaphore export valid-usage requirements above.
        // `external_sync_fns` was loaded from the same logical device, `semaphore` belongs to that
        // device, and the host-access mutex prevents concurrent host operations on the semaphore.
        // The returned fd is newly owned by the application, except `-1` is a valid sync-file result
        // meaning the payload is already signaled and no fd needs to be closed.
        let raw_fd = unsafe {
            external_sync_fns
                .external_semaphore_fd
                .get_semaphore_fd(&get_info)
        }
        .map_err(VulkanError::from)?;
        let exported_fd = if raw_fd == -1 {
            None
        } else {
            // SAFETY: `vkGetSemaphoreFdKHR` returned a newly owned POSIX file descriptor for the
            // application to close or transfer. `-1` was handled above.
            Some(unsafe { OwnedFd::from_raw_fd(raw_fd) })
        };
        semaphore.set_payload_state(VulkanSyncFileSemaphorePayloadState::Unsignaled)?;
        Ok(exported_fd)
    }

    #[allow(dead_code)]
    pub(crate) fn dmabuf_external_image_format_properties(
        &self,
        import: &VulkanDmabufImportState,
    ) -> Result<Option<VulkanDmabufExternalImageFormatProperties>, VulkanError> {
        self.dmabuf_external_image_format_properties_for_usage(import, vk::ImageUsageFlags::SAMPLED)
    }

    #[allow(dead_code)]
    pub(crate) fn dmabuf_render_target_external_image_format_properties(
        &self,
        import: &VulkanDmabufImportState,
    ) -> Result<Option<VulkanDmabufExternalImageFormatProperties>, VulkanError> {
        self.dmabuf_external_image_format_properties_for_usage(import, vk::ImageUsageFlags::COLOR_ATTACHMENT)
    }

    #[allow(dead_code)]
    fn dmabuf_external_image_format_properties_for_usage(
        &self,
        import: &VulkanDmabufImportState,
        usage: vk::ImageUsageFlags,
    ) -> Result<Option<VulkanDmabufExternalImageFormatProperties>, VulkanError> {
        if !self.capabilities.external_memory.prerequisites_available || self.external_memory_fns.is_none() {
            return Ok(None);
        }

        let has_modifier_record = if usage == vk::ImageUsageFlags::SAMPLED {
            self.capabilities.formats.dmabuf_import_record(import).is_some()
        } else if usage == vk::ImageUsageFlags::COLOR_ATTACHMENT {
            self.capabilities
                .formats
                .dmabuf_render_target_record(import)
                .is_some()
        } else {
            false
        };
        if !has_modifier_record {
            return Ok(None);
        }

        let physical_device = self
            .physical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing physical device".to_owned()))?;
        let vk_format = get_render_vk_format(import.format())?;
        let mut external_image_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut drm_format_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(import.modifier().into())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let format_info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(vk_format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .flags(vk::ImageCreateFlags::empty())
            .push_next(&mut external_image_info)
            .push_next(&mut drm_format_info);
        let mut external_properties = vk::ExternalImageFormatProperties::default();
        let mut image_properties = vk::ImageFormatProperties2::default().push_next(&mut external_properties);

        // SAFETY: `physical_device` belongs to the retained instance. The pNext chains are built
        // from stack values that outlive the call, use Vulkan-defined structs, and request only a
        // 2D sampled or color-attachment DRM-modifier image with DMA_BUF external memory after the
        // corresponding device extensions and modifier record have been discovered.
        let result = unsafe {
            physical_device
                .instance()
                .handle()
                .get_physical_device_image_format_properties2(
                    physical_device.handle(),
                    &format_info,
                    &mut image_properties,
                )
        };

        match result {
            Ok(()) => {
                let image_format_properties = image_properties.image_format_properties;
                let _ = image_properties;
                let external_memory_properties = external_properties.external_memory_properties;
                let importable = external_memory_properties
                    .external_memory_features
                    .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
                    && external_memory_properties
                        .compatible_handle_types
                        .contains(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                Ok(Some(VulkanDmabufExternalImageFormatProperties {
                    image_format_properties,
                    external_memory_properties,
                    importable,
                    dedicated_only: external_memory_properties
                        .external_memory_features
                        .contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY),
                }))
            }
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(None),
            Err(error) => Err(VulkanError::from(error)),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn dmabuf_import_candidate(
        &self,
        import: &VulkanDmabufImportState,
    ) -> Result<Option<VulkanDmabufImportCandidate>, VulkanError> {
        let Some(properties) = self.dmabuf_external_image_format_properties(import)? else {
            return Ok(None);
        };

        if !properties.supports_sampled_import(import) {
            return Ok(None);
        }

        Ok(Some(VulkanDmabufImportCandidate {
            dedicated_only: properties.dedicated_only,
            properties,
        }))
    }

    #[allow(dead_code)]
    pub(crate) fn dmabuf_render_target_candidate(
        &self,
        import: &VulkanDmabufImportState,
    ) -> Result<Option<VulkanDmabufImportCandidate>, VulkanError> {
        let Some(properties) = self.dmabuf_render_target_external_image_format_properties(import)? else {
            return Ok(None);
        };

        if !properties.supports_render_target_import(import) {
            return Ok(None);
        }

        Ok(Some(VulkanDmabufImportCandidate {
            dedicated_only: properties.dedicated_only,
            properties,
        }))
    }

    #[allow(dead_code)]
    pub(crate) fn create_dmabuf_import_image(
        &self,
        import: &VulkanDmabufImportState,
    ) -> Result<Option<VulkanDmabufImportImage>, VulkanError> {
        self.create_dmabuf_image_for_usage(import, vk::ImageUsageFlags::SAMPLED)
    }

    #[allow(dead_code)]
    pub(crate) fn create_dmabuf_render_target_image(
        &self,
        import: &VulkanDmabufImportState,
    ) -> Result<Option<VulkanDmabufImportImage>, VulkanError> {
        self.create_dmabuf_image_for_usage(import, vk::ImageUsageFlags::COLOR_ATTACHMENT)
    }

    #[allow(dead_code)]
    fn create_dmabuf_image_for_usage(
        &self,
        import: &VulkanDmabufImportState,
        usage: vk::ImageUsageFlags,
    ) -> Result<Option<VulkanDmabufImportImage>, VulkanError> {
        let candidate = if usage == vk::ImageUsageFlags::SAMPLED {
            self.dmabuf_import_candidate(import)?
        } else if usage == vk::ImageUsageFlags::COLOR_ATTACHMENT {
            self.dmabuf_render_target_candidate(import)?
        } else {
            None
        };
        let Some(candidate) = candidate else {
            return Ok(None);
        };
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();
        let vk_format = get_render_vk_format(import.format())?;
        let extent = vk::Extent3D {
            width: import.size.w.try_into().unwrap_or_default(),
            height: import.size.h.try_into().unwrap_or_default(),
            depth: 1,
        };
        let plane_layouts = dmabuf_plane_layouts(import);
        let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(import.modifier().into())
            .plane_layouts(&plane_layouts);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory_info)
            .push_next(&mut modifier_info);

        // SAFETY: `logical_device` is live. The pNext chain contains Vulkan-defined image-create
        // extension structs whose stack storage outlives the call. The image shape matches the
        // earlier external-image-format query/candidate: 2D, one mip level, one array layer,
        // TYPE_1 samples, sampled or color-attachment usage, DRM_FORMAT_MODIFIER tiling, UNDEFINED
        // initial layout, and DMA_BUF_EXT external memory. External-memory images with nonzero
        // handle types must be created with UNDEFINED initial layout; preserving producer contents
        // must be handled by a later external acquire/synchronization path, not by PREINITIALIZED
        // image creation.
        // Plane-layout pointers are derived from validated plane count/order and nonzero stride
        // metadata, and remain alive through the call; modifier-specific layout validity is still
        // checked by the driver and may make image creation fail. No allocation callbacks are used.
        let image =
            unsafe { logical_device.handle().create_image(&image_info, None) }.map_err(VulkanError::from)?;
        let unbound = VulkanUnboundImage {
            logical_device,
            image,
            extent,
            format: vk_format,
            usage,
            external_memory_handle_type: Some(VulkanExternalMemoryHandleType::Dmabuf),
        };
        // SAFETY: `unbound.image` was just created from `unbound.logical_device`, has not been
        // destroyed, and the call only queries requirements for that image. No memory has been bound
        // yet, and the image owner remains alive for the duration of the query.
        let memory_requirements = unsafe {
            unbound
                .logical_device
                .handle()
                .get_image_memory_requirements(unbound.image)
        };

        Ok(Some(VulkanDmabufImportImage {
            image: unbound,
            memory_requirements,
            candidate,
        }))
    }

    #[allow(dead_code)]
    pub(crate) fn create_bound_dmabuf_import_image(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<Option<VulkanOwnedImage>, VulkanError> {
        self.create_bound_dmabuf_image_with_sync(
            dmabuf,
            dmabuf_import_sync_state(),
            vk::ImageUsageFlags::SAMPLED,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn create_bound_dmabuf_render_target_image(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<Option<VulkanOwnedImage>, VulkanError> {
        self.create_bound_dmabuf_image_with_sync(
            dmabuf,
            dmabuf_import_sync_state(),
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
    }

    /// Create and acquire a dmabuf image for use as a color-attachment render target.
    ///
    /// # Safety
    ///
    /// The caller must ensure the foreign producer has released ownership to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` before this acquire is submitted. If `preserve_contents` is
    /// true, the producer must have released the image in `VK_IMAGE_LAYOUT_GENERAL`. If
    /// `acquire_semaphore` is present, it must signal only after the producer's writes and ownership
    /// release complete. If it is absent, those operations must already be complete and visible to
    /// this renderer's Vulkan queue submission. If `preserve_contents` is false, this helper discards
    /// the old contents and acquires from an unknown foreign layout with `UNDEFINED` as the old
    /// layout.
    #[allow(dead_code)]
    pub(crate) unsafe fn create_acquired_dmabuf_render_target_image(
        &self,
        dmabuf: &Dmabuf,
        preserve_contents: bool,
        acquire_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<Option<VulkanOwnedImage>, VulkanError> {
        let sync = if preserve_contents {
            VulkanImageSyncState::foreign_known_general_for_dmabuf_import()
        } else {
            dmabuf_import_sync_state()
        };
        let Some(image) =
            self.create_bound_dmabuf_image_with_sync(dmabuf, sync, vk::ImageUsageFlags::COLOR_ATTACHMENT)?
        else {
            return Ok(None);
        };

        if !self.submit_dmabuf_render_target_foreign_acquire(&image, preserve_contents, acquire_semaphore)? {
            return Err(VulkanError::UnsupportedOperation("dmabuf external ownership"));
        }

        Ok(Some(image))
    }

    /// Create and acquire a dmabuf color-attachment render target using a Smithay sync point as the
    /// optional producer-completion dependency.
    ///
    /// # Safety
    ///
    /// The caller must ensure the foreign producer has released ownership to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` before this acquire is submitted. If `preserve_contents` is
    /// true, the producer must have released the image in `VK_IMAGE_LAYOUT_GENERAL`. If
    /// `acquire_sync` is present, it must represent the producer's completion dependency for that
    /// release and signal only after the producer's writes and ownership release for this dmabuf are
    /// complete. If it is absent, those operations must already be complete and visible to this
    /// renderer's Vulkan queue submission. If `acquire_sync` exports a fence fd and this device
    /// supports sync-file import, that fd must be a valid Linux sync-file fd suitable for
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`. If `preserve_contents` is false, this helper
    /// discards the old contents and acquires from an unknown foreign layout with `UNDEFINED` as the
    /// old layout.
    #[allow(dead_code)]
    pub(crate) unsafe fn create_acquired_dmabuf_render_target_image_with_sync_point(
        &self,
        dmabuf: &Dmabuf,
        preserve_contents: bool,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanOwnedImage>, VulkanError> {
        let sync = if preserve_contents {
            VulkanImageSyncState::foreign_known_general_for_dmabuf_import()
        } else {
            dmabuf_import_sync_state()
        };
        let Some(image) =
            self.create_bound_dmabuf_image_with_sync(dmabuf, sync, vk::ImageUsageFlags::COLOR_ATTACHMENT)?
        else {
            return Ok(None);
        };

        let acquire_semaphore = if let Some(acquire_sync) = acquire_sync {
            // SAFETY: Forwarded from this method's caller.
            unsafe { self.import_sync_point_wait_semaphore(acquire_sync)? }
        } else {
            None
        };

        if !self.submit_dmabuf_render_target_foreign_acquire(
            &image,
            preserve_contents,
            acquire_semaphore.as_ref(),
        )? {
            return Err(VulkanError::UnsupportedOperation("dmabuf external ownership"));
        }

        Ok(Some(image))
    }

    fn create_bound_dmabuf_import_image_with_sync(
        &self,
        dmabuf: &Dmabuf,
        sync: VulkanImageSyncState,
    ) -> Result<Option<VulkanOwnedImage>, VulkanError> {
        self.create_bound_dmabuf_image_with_sync(dmabuf, sync, vk::ImageUsageFlags::SAMPLED)
    }

    fn create_bound_dmabuf_image_with_sync(
        &self,
        dmabuf: &Dmabuf,
        sync: VulkanImageSyncState,
        usage: vk::ImageUsageFlags,
    ) -> Result<Option<VulkanOwnedImage>, VulkanError> {
        let import = VulkanDmabufImportState::from_dmabuf(dmabuf)?;
        let Some(fd) = single_plane_dmabuf_fd(dmabuf)? else {
            return Ok(None);
        };
        let Some(import_image) = self.create_dmabuf_image_for_usage(&import, usage)? else {
            return Ok(None);
        };
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();
        let memory_properties = self
            .memory_properties
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing memory properties".to_owned()))?;
        let external_memory_fns = self
            .external_memory_fns
            .as_ref()
            .ok_or(VulkanError::ExternalMemoryUnsupported)?;

        let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: `external_memory_fns` was loaded only when VK_KHR_external_memory_fd was enabled
        // on this live device. `fd` is a live duplicated dmabuf file descriptor for the duration of
        // the call, and the output pointer refers to stack storage.
        unsafe {
            external_memory_fns.external_memory_fd.get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd.as_raw_fd(),
                &mut fd_properties,
            )
        }
        .map_err(VulkanError::from)?;

        let memory_type_bits = dmabuf_import_memory_type_bits(
            import_image.memory_requirements.memory_type_bits,
            fd_properties.memory_type_bits,
        );
        let memory_type_index = find_memory_type_index(
            memory_properties,
            memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
        )?;
        let memory = allocate_imported_dmabuf_memory(
            &logical_device,
            import_image.image.image(),
            import_image.memory_requirements.size,
            memory_type_index,
            fd,
            import_image.candidate.dedicated_only,
        )?;

        if let Err(err) = unsafe {
            logical_device
                .handle()
                .bind_image_memory(import_image.image.image(), memory, 0)
        }
        .map_err(VulkanError::from)
        {
            // SAFETY: `memory` was allocated from `logical_device` above and has not been bound
            // successfully or transferred into an owner. Freeing it here prevents a leak; the
            // unbound image is destroyed when `import_image` is dropped.
            unsafe { logical_device.handle().free_memory(memory, None) };
            return Err(err);
        }

        Ok(Some(import_image.image.into_bound_image_with_sync(memory, sync)))
    }

    #[allow(dead_code)]
    pub(crate) fn create_dmabuf_sampled_image_resources(
        &self,
        dmabuf: &Dmabuf,
        min_filter: TextureFilter,
        mag_filter: TextureFilter,
    ) -> Result<Option<VulkanSampledImage>, VulkanError> {
        let Some(image) = self.create_bound_dmabuf_import_image(dmabuf)? else {
            return Ok(None);
        };
        let view = self.create_image_view(&image)?;
        let sampler = self.create_sampler(min_filter, mag_filter)?;

        Ok(Some(VulkanSampledImage { sampler, view, image }))
    }

    /// Create sampled resources for a dmabuf whose external Vulkan ownership/layout are known.
    ///
    /// # Safety
    ///
    /// The caller must ensure the dmabuf producer released the image to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`, and that `acquire_semaphore`, if
    /// present, represents the producer's completion dependency for that release. If no semaphore is
    /// supplied, the caller must ensure the producer's writes and ownership release are already
    /// complete and visible to this Vulkan queue submission.
    #[allow(dead_code)]
    pub(crate) unsafe fn create_acquired_dmabuf_sampled_image_resources_with_known_general_layout(
        &self,
        dmabuf: &Dmabuf,
        min_filter: TextureFilter,
        mag_filter: TextureFilter,
        acquire_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<Option<VulkanSampledImage>, VulkanError> {
        let Some(image) = self.create_bound_dmabuf_import_image_with_sync(
            dmabuf,
            VulkanImageSyncState::foreign_known_general_for_dmabuf_import(),
        )?
        else {
            return Ok(None);
        };
        let view = self.create_image_view(&image)?;
        let sampler = self.create_sampler(min_filter, mag_filter)?;
        if !self.submit_sampled_dmabuf_foreign_acquire(&image, acquire_semaphore)? {
            return Err(VulkanError::UnsupportedOperation("dmabuf external ownership"));
        }

        Ok(Some(VulkanSampledImage { sampler, view, image }))
    }

    /// Create sampled resources for a known-layout dmabuf, using a Smithay sync point as the
    /// optional producer-completion dependency.
    ///
    /// # Safety
    ///
    /// The caller must ensure the dmabuf producer released the image to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`. If `acquire_sync` is `Some`, it
    /// must represent the producer's completion dependency for that release and signal only after
    /// the producer's writes and ownership release for this dmabuf are complete. If `acquire_sync`
    /// is `None`, the caller must ensure those writes and ownership release are already complete and
    /// visible to this Vulkan queue submission. If `acquire_sync` exports a fence fd and this device
    /// supports sync-file import, that fd must be a valid Linux sync-file fd suitable for
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`.
    #[allow(dead_code)]
    pub(crate) unsafe fn create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point(
        &self,
        dmabuf: &Dmabuf,
        min_filter: TextureFilter,
        mag_filter: TextureFilter,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanSampledImage>, VulkanError> {
        unsafe {
            // SAFETY: Forwarded from this method's caller.
            self.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point_classified(
                dmabuf,
                min_filter,
                mag_filter,
                acquire_sync,
            )
        }
        .map_err(VulkanSampledDmabufForeignAcquireError::into_inner)
    }

    /// Create sampled resources for a known-layout dmabuf with classified acquire-submit errors.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the same external-state and acquire-sync requirements as
    /// [`VulkanDeviceState::create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point`].
    #[allow(dead_code)]
    pub(crate) unsafe fn create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point_classified(
        &self,
        dmabuf: &Dmabuf,
        min_filter: TextureFilter,
        mag_filter: TextureFilter,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanSampledImage>, VulkanSampledDmabufForeignAcquireError> {
        let Some(prepared) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            self.prepare_sampled_dmabuf_foreign_acquire_with_known_general_layout(
                dmabuf,
                min_filter,
                mag_filter,
                acquire_sync,
            )
        })?
        else {
            return Ok(None);
        };

        self.submit_prepared_sampled_dmabuf_foreign_acquire_classified(prepared)
            .map(Some)
    }

    /// Prepare sampled dmabuf resources and record the Vulkan acquire command buffer.
    ///
    /// The returned bundle owns the imported image, view, sampler, optional acquire semaphore, and
    /// recorded acquire command buffer, but no queue ownership transfer has been submitted yet.
    /// Submit validation, fence creation, host-access locking, and semaphore payload reservation still
    /// happen in [`VulkanDeviceState::submit_prepared_sampled_dmabuf_foreign_acquire_classified`].
    /// Higher layers must not consume move-only Wayland release ownership merely because this
    /// preparation succeeded; that needs a later ready-to-submit reservation token that eliminates or
    /// reserves those remaining pre-submit failure points first.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the same external-state and acquire-sync requirements as
    /// [`VulkanDeviceState::create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point`].
    #[allow(dead_code)]
    pub(crate) unsafe fn prepare_sampled_dmabuf_foreign_acquire_with_known_general_layout(
        &self,
        dmabuf: &Dmabuf,
        min_filter: TextureFilter,
        mag_filter: TextureFilter,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanPreparedSampledDmabufAcquire>, VulkanSampledDmabufForeignAcquireError> {
        let Some(image) = self
            .create_bound_dmabuf_import_image_with_sync(
                dmabuf,
                VulkanImageSyncState::foreign_known_general_for_dmabuf_import(),
            )
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?
        else {
            return Ok(None);
        };
        let view = self
            .create_image_view(&image)
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?;
        let sampler = self
            .create_sampler(min_filter, mag_filter)
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?;

        let acquire_semaphore = if let Some(acquire_sync) = acquire_sync {
            unsafe { self.import_sync_point_wait_semaphore(acquire_sync) }
                .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?
        } else {
            None
        };

        let mut command_buffer = self
            .allocate_graphics_command_buffer()
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?;
        self.begin_command_buffer(&mut command_buffer)
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?;
        if !self
            .record_sampled_dmabuf_foreign_acquire_barrier(&mut command_buffer, &image)
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?
        {
            return Err(VulkanSampledDmabufForeignAcquireError::RetrySafe(
                VulkanError::UnsupportedOperation("dmabuf external ownership"),
            ));
        }
        self.end_command_buffer(&mut command_buffer)
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?;

        Ok(Some(VulkanPreparedSampledDmabufAcquire {
            sampler,
            view,
            image,
            acquire_semaphore,
            command_buffer,
        }))
    }

    /// Submit a prepared sampled dmabuf acquire bundle.
    ///
    /// This is the first helper that may call `vkQueueSubmit`, but it still performs fallible
    /// pre-submit validation/allocation and semaphore payload reservation before that call. Callers
    /// that have consumed external move-only obligations must treat any error from this helper as
    /// non-retryable for those obligations unless a narrower ready-to-submit token has already
    /// reserved the submit state.
    #[allow(dead_code)]
    pub(crate) fn submit_prepared_sampled_dmabuf_foreign_acquire_classified(
        &self,
        prepared: VulkanPreparedSampledDmabufAcquire,
    ) -> Result<VulkanSampledImage, VulkanSampledDmabufForeignAcquireError> {
        match self.submit_prepared_sampled_dmabuf_foreign_acquire_ready(prepared, || Ok(())) {
            Ok((sampled_image, ())) => Ok(sampled_image),
            Err(VulkanSampledDmabufReadyAcquireError::RetrySafe(err))
            | Err(VulkanSampledDmabufReadyAcquireError::ReadyCallbackFailed(err))
            | Err(VulkanSampledDmabufReadyAcquireError::ReadyCallbackCommitted { err, .. }) => {
                Err(VulkanSampledDmabufForeignAcquireError::RetrySafe(err))
            }
            Err(VulkanSampledDmabufReadyAcquireError::AcquireSubmitted {
                err, sampled_image, ..
            }) => Err(VulkanSampledDmabufForeignAcquireError::AcquireSubmitted {
                err,
                sampled_image: Some(sampled_image),
            }),
        }
    }

    /// Submit a prepared sampled dmabuf acquire after running a callback at the ready-to-submit point.
    ///
    /// The callback is called only after submit validation, fence creation, host-access locking, and
    /// semaphore payload reservation have succeeded. It runs while the submit state is reserved and
    /// immediately before `vkQueueSubmit`; callers may use it for move-only obligations that must not
    /// be consumed by retry-safe setup failures. If the callback succeeds, any later error is reported
    /// with the callback's return value so the caller can complete or compensate that obligation.
    ///
    /// The callback runs while semaphore, command-pool, and queue host-access locks are held. It must
    /// be short, must not call back into Vulkan submit/import/export paths, and must not invoke
    /// arbitrary renderer code that could re-enter this device.
    #[allow(dead_code)]
    pub(crate) fn submit_prepared_sampled_dmabuf_foreign_acquire_ready<T, F>(
        &self,
        prepared: VulkanPreparedSampledDmabufAcquire,
        before_submit: F,
    ) -> Result<(VulkanSampledImage, T), VulkanSampledDmabufReadyAcquireError<T>>
    where
        F: FnOnce() -> Result<T, VulkanError>,
    {
        let VulkanPreparedSampledDmabufAcquire {
            sampler,
            view,
            image,
            acquire_semaphore,
            mut command_buffer,
        } = prepared;

        let ready_result = if let Some(acquire_semaphore) = acquire_semaphore.as_ref() {
            let synchronization = VulkanSubmitSynchronization::default()
                .wait_sync_file(acquire_semaphore, vk::PipelineStageFlags::TOP_OF_PIPE);
            // SAFETY: This helper fixes the wait stage to TOP_OF_PIPE, which is supported by every
            // graphics queue. `submit_command_buffer_and_wait` validates that the semaphore belongs
            // to this device, is not duplicated in the submit, and has a waitable tracked payload
            // before the queue operation is attempted.
            unsafe {
                self.submit_graphics_command_buffer_and_wait_with_synchronization_after_reservation(
                    &mut command_buffer,
                    &synchronization,
                    before_submit,
                )
            }
        } else {
            unsafe {
                self.submit_graphics_command_buffer_and_wait_after_reservation(
                    &mut command_buffer,
                    before_submit,
                )
            }
        };

        match ready_result {
            Ok(ready) => Ok((VulkanSampledImage { sampler, view, image }, ready)),
            Err(VulkanReadySubmitError::ReservationFailed(err)) => {
                Err(VulkanSampledDmabufReadyAcquireError::RetrySafe(err))
            }
            Err(VulkanReadySubmitError::ReadyCallbackFailed(err)) => {
                Err(VulkanSampledDmabufReadyAcquireError::ReadyCallbackFailed(err))
            }
            Err(VulkanReadySubmitError::ReadyCallbackCommitted { err, ready }) => {
                Err(VulkanSampledDmabufReadyAcquireError::ReadyCallbackCommitted { err, ready })
            }
            Err(VulkanReadySubmitError::Submitted { err, ready }) => {
                Err(VulkanSampledDmabufReadyAcquireError::AcquireSubmitted {
                    err,
                    sampled_image: VulkanSampledImage { sampler, view, image },
                    ready,
                })
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn submit_sampled_dmabuf_foreign_acquire(
        &self,
        image: &VulkanOwnedImage,
        acquire_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<bool, VulkanError> {
        self.submit_sampled_dmabuf_foreign_acquire_classified(image, acquire_semaphore)
            .map_err(VulkanSampledDmabufForeignAcquireError::into_inner)
    }

    #[allow(dead_code)]
    pub(crate) fn submit_sampled_dmabuf_foreign_acquire_classified(
        &self,
        image: &VulkanOwnedImage,
        acquire_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<bool, VulkanSampledDmabufForeignAcquireError> {
        let mut command_buffer = self
            .allocate_graphics_command_buffer()
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?;
        self.begin_command_buffer(&mut command_buffer)
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?;
        if !self
            .record_sampled_dmabuf_foreign_acquire_barrier(&mut command_buffer, image)
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?
        {
            return Ok(false);
        }
        self.end_command_buffer(&mut command_buffer)
            .map_err(VulkanSampledDmabufForeignAcquireError::RetrySafe)?;

        let acquire_result = if let Some(acquire_semaphore) = acquire_semaphore {
            let synchronization = VulkanSubmitSynchronization::default()
                .wait_sync_file(acquire_semaphore, vk::PipelineStageFlags::TOP_OF_PIPE);
            // SAFETY: This helper fixes the wait stage to TOP_OF_PIPE, which is supported by every
            // graphics queue. `submit_command_buffer_and_wait` validates that the semaphore belongs
            // to this device, is not duplicated in the submit, and has a waitable tracked payload
            // before the queue operation is attempted.
            unsafe {
                self.submit_graphics_command_buffer_and_wait_with_synchronization(
                    &mut command_buffer,
                    &synchronization,
                )
            }
        } else {
            self.submit_graphics_command_buffer_and_wait(&mut command_buffer)
        };

        acquire_result
            .map_err(|err| classify_sampled_dmabuf_acquire_submit_error(command_buffer.state, err))?;

        Ok(true)
    }

    #[allow(dead_code)]
    pub(crate) fn submit_sampled_dmabuf_foreign_release(
        &self,
        image: &VulkanOwnedImage,
        release_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<bool, VulkanError> {
        self.submit_sampled_dmabuf_foreign_release_classified(image, release_semaphore)
            .map_err(VulkanSampledDmabufForeignReleaseError::into_inner)
    }

    #[allow(dead_code)]
    fn submit_sampled_dmabuf_foreign_release_classified(
        &self,
        image: &VulkanOwnedImage,
        release_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<bool, VulkanSampledDmabufForeignReleaseError> {
        let mut command_buffer = self
            .allocate_graphics_command_buffer()
            .map_err(VulkanSampledDmabufForeignReleaseError::RetrySafe)?;
        self.begin_command_buffer(&mut command_buffer)
            .map_err(VulkanSampledDmabufForeignReleaseError::RetrySafe)?;
        if !self
            .record_sampled_dmabuf_foreign_release_barrier(&mut command_buffer, image)
            .map_err(VulkanSampledDmabufForeignReleaseError::RetrySafe)?
        {
            return Ok(false);
        }
        self.end_command_buffer(&mut command_buffer)
            .map_err(VulkanSampledDmabufForeignReleaseError::RetrySafe)?;

        let release_result = if let Some(release_semaphore) = release_semaphore {
            let synchronization = VulkanSubmitSynchronization::default().signal_sync_file(release_semaphore);
            // SAFETY: This submit has no semaphore waits, and `submit_command_buffer_and_wait`
            // validates that the signal semaphore belongs to this device, is not duplicated in the
            // submit, and has an unsignaled tracked payload before the queue operation is attempted.
            unsafe {
                self.submit_graphics_command_buffer_and_wait_with_synchronization(
                    &mut command_buffer,
                    &synchronization,
                )
            }
        } else {
            self.submit_graphics_command_buffer_and_wait(&mut command_buffer)
        };

        release_result
            .map_err(|err| classify_sampled_dmabuf_release_submit_error(command_buffer.state, err))?;

        Ok(true)
    }

    #[allow(dead_code)]
    fn submit_sampled_dmabuf_foreign_release_async(
        &self,
        image: &VulkanOwnedImage,
        release_semaphore: &VulkanSyncFileSemaphore,
    ) -> Result<Option<VulkanSubmittedCommandBuffer>, VulkanError> {
        let mut command_buffer = self.allocate_graphics_command_buffer()?;
        self.begin_command_buffer(&mut command_buffer)?;
        if !self.record_sampled_dmabuf_foreign_release_barrier(&mut command_buffer, image)? {
            return Ok(None);
        }
        self.end_command_buffer(&mut command_buffer)?;

        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let queue =
            self.queues.graphics.as_ref().ok_or_else(|| {
                VulkanError::DeviceInitializationFailed("missing graphics queue".to_owned())
            })?;
        let synchronization = VulkanSubmitSynchronization::default().signal_sync_file(release_semaphore);
        // SAFETY: This submit has no semaphore waits, and `submit_owned_command_buffer` validates
        // that the signal semaphore belongs to this device, is not duplicated in the submit, and has
        // an unsignaled tracked payload before the queue operation is attempted. The returned
        // submission retains all objects referenced by the pending queue batch.
        unsafe { submit_owned_command_buffer(logical_device, queue, command_buffer, &synchronization) }
            .map(Some)
    }

    #[allow(dead_code)]
    pub(crate) fn submit_dmabuf_render_target_foreign_acquire(
        &self,
        image: &VulkanOwnedImage,
        preserve_contents: bool,
        acquire_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<bool, VulkanError> {
        let mut command_buffer = self.allocate_graphics_command_buffer()?;
        self.begin_command_buffer(&mut command_buffer)?;
        if !self.record_dmabuf_render_target_foreign_acquire_barrier(
            &mut command_buffer,
            image,
            preserve_contents,
        )? {
            return Ok(false);
        }
        self.end_command_buffer(&mut command_buffer)?;

        if let Some(acquire_semaphore) = acquire_semaphore {
            let synchronization = VulkanSubmitSynchronization::default()
                .wait_sync_file(acquire_semaphore, vk::PipelineStageFlags::TOP_OF_PIPE);
            // SAFETY: This helper fixes the wait stage to TOP_OF_PIPE, which is supported by every
            // graphics queue. `submit_command_buffer_and_wait` validates that the semaphore belongs
            // to this device, is not duplicated in the submit, and has a waitable tracked payload
            // before the queue operation is attempted.
            unsafe {
                self.submit_graphics_command_buffer_and_wait_with_synchronization(
                    &mut command_buffer,
                    &synchronization,
                )?
            };
        } else {
            self.submit_graphics_command_buffer_and_wait(&mut command_buffer)?;
        }

        Ok(true)
    }

    #[allow(dead_code)]
    pub(crate) fn submit_dmabuf_render_target_foreign_release(
        &self,
        image: &VulkanOwnedImage,
        release_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<bool, VulkanError> {
        let mut command_buffer = self.allocate_graphics_command_buffer()?;
        self.begin_command_buffer(&mut command_buffer)?;
        if !self.record_dmabuf_render_target_foreign_release_barrier(&mut command_buffer, image)? {
            return Ok(false);
        }
        self.end_command_buffer(&mut command_buffer)?;

        if let Some(release_semaphore) = release_semaphore {
            let synchronization = VulkanSubmitSynchronization::default().signal_sync_file(release_semaphore);
            // SAFETY: This submit has no semaphore waits, and `submit_command_buffer_and_wait`
            // validates that the signal semaphore belongs to this device, is not duplicated in the
            // submit, and has an unsignaled tracked payload before the queue operation is attempted.
            unsafe {
                self.submit_graphics_command_buffer_and_wait_with_synchronization(
                    &mut command_buffer,
                    &synchronization,
                )?
            };
        } else {
            self.submit_graphics_command_buffer_and_wait(&mut command_buffer)?;
        }

        Ok(true)
    }

    #[allow(dead_code)]
    fn submit_dmabuf_render_target_foreign_release_async(
        &self,
        image: &VulkanOwnedImage,
        release_semaphore: &VulkanSyncFileSemaphore,
    ) -> Result<Option<VulkanSubmittedCommandBuffer>, VulkanError> {
        let mut command_buffer = self.allocate_graphics_command_buffer()?;
        self.begin_command_buffer(&mut command_buffer)?;
        if !self.record_dmabuf_render_target_foreign_release_barrier(&mut command_buffer, image)? {
            return Ok(None);
        }
        self.end_command_buffer(&mut command_buffer)?;

        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let queue =
            self.queues.graphics.as_ref().ok_or_else(|| {
                VulkanError::DeviceInitializationFailed("missing graphics queue".to_owned())
            })?;
        let synchronization = VulkanSubmitSynchronization::default().signal_sync_file(release_semaphore);
        // SAFETY: This submit has no semaphore waits, and `submit_owned_command_buffer` validates
        // that the signal semaphore belongs to this device, is not duplicated in the submit, and has
        // an unsignaled tracked payload before the queue operation is attempted. The returned
        // submission retains all objects referenced by the pending queue batch.
        unsafe { submit_owned_command_buffer(logical_device, queue, command_buffer, &synchronization) }
            .map(Some)
    }

    #[allow(dead_code)]
    pub(crate) fn release_dmabuf_render_target_to_foreign_general(
        &self,
        image: &VulkanOwnedImage,
        export_sync_file: bool,
    ) -> Result<(bool, Option<OwnedFd>), VulkanError> {
        ensure_dmabuf_external_image(image)?;
        let release_semaphore = if export_sync_file {
            Some(self.create_exportable_sync_file_semaphore()?)
        } else {
            None
        };
        let Some(release_semaphore) = release_semaphore else {
            if !self.submit_dmabuf_render_target_foreign_release(image, None)? {
                return Ok((false, None));
            }
            return Ok((true, None));
        };

        let Some(submission) =
            self.submit_dmabuf_render_target_foreign_release_async(image, &release_semaphore)?
        else {
            return Ok((false, None));
        };
        // SAFETY: `submit_dmabuf_render_target_foreign_release_async` submitted the signal operation
        // and returned only after `vkQueueSubmit` accepted it. SYNC_FD export permits a pending
        // signal operation whose dependencies have been submitted. On successful export, the
        // submission is retained until the batch fence completes so the command buffer and semaphore
        // object outlive queue use. On non-fatal export failure, we wait below before returning no
        // fence so callers do not observe an unfenced pending release as complete.
        let release_sync_file = match unsafe { self.export_sync_file_semaphore(&release_semaphore) } {
            Ok(sync_file) => {
                self.retain_graphics_submission(submission)?;
                sync_file
            }
            Err(err) if vulkan_error_invalidates_context(&err) => return Err(err),
            Err(err) => {
                tracing::warn!(?err, "failed to export dmabuf render-target release fence");
                submission.wait_complete()?;
                None
            }
        };

        Ok((true, release_sync_file))
    }

    #[allow(dead_code)]
    pub(crate) fn release_sampled_dmabuf_to_foreign_general(
        &self,
        image: &VulkanOwnedImage,
        export_sync_file: bool,
    ) -> Result<(bool, Option<OwnedFd>), VulkanError> {
        self.release_sampled_dmabuf_to_foreign_general_classified(image, export_sync_file)
            .map_err(VulkanSampledDmabufForeignReleaseError::into_inner)
    }

    #[allow(dead_code)]
    pub(crate) fn release_sampled_dmabuf_to_foreign_general_classified(
        &self,
        image: &VulkanOwnedImage,
        export_sync_file: bool,
    ) -> Result<(bool, Option<OwnedFd>), VulkanSampledDmabufForeignReleaseError> {
        ensure_dmabuf_external_image(image).map_err(VulkanSampledDmabufForeignReleaseError::RetrySafe)?;
        let release_semaphore = if export_sync_file {
            Some(
                self.create_exportable_sync_file_semaphore()
                    .map_err(VulkanSampledDmabufForeignReleaseError::RetrySafe)?,
            )
        } else {
            None
        };
        let Some(release_semaphore) = release_semaphore else {
            if !self.submit_sampled_dmabuf_foreign_release_classified(image, None)? {
                return Ok((false, None));
            }
            return Ok((true, None));
        };

        let Some(submission) = self
            .submit_sampled_dmabuf_foreign_release_async(image, &release_semaphore)
            .map_err(VulkanSampledDmabufForeignReleaseError::RetrySafe)?
        else {
            return Ok((false, None));
        };
        // SAFETY: `submit_sampled_dmabuf_foreign_release_async` submitted the signal operation and
        // returned only after `vkQueueSubmit` accepted it. SYNC_FD export permits a pending signal
        // operation whose dependencies have been submitted. On successful export, the submission is
        // retained until the batch fence completes so the command buffer and semaphore object outlive
        // queue use. On non-fatal export failure, we wait below before returning no fence so callers
        // do not observe an unfenced pending release as complete.
        let release_sync_file = match unsafe { self.export_sync_file_semaphore(&release_semaphore) } {
            Ok(sync_file) => {
                self.retain_graphics_submission(submission)
                    .map_err(VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted)?;
                sync_file
            }
            Err(err) if vulkan_error_invalidates_context(&err) => {
                return Err(VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(err));
            }
            Err(err) => {
                tracing::warn!(?err, "failed to export sampled dmabuf release fence");
                submission
                    .wait_complete()
                    .map_err(VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted)?;
                None
            }
        };

        Ok((true, release_sync_file))
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
        ensure_command_buffer_recording(command_buffer)?;
        let _pool_guard = command_buffer.command_pool.lock_host_access()?;

        unsafe {
            command_buffer
                .command_pool
                .logical_device
                .handle()
                .end_command_buffer(command_buffer.handle)
        }
        .map_err(VulkanError::from)?;

        command_buffer.state = VulkanCommandBufferState::Executable;
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn submit_graphics_command_buffer_and_wait(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
    ) -> Result<(), VulkanError> {
        unsafe { self.submit_graphics_command_buffer_and_wait_after_reservation(command_buffer, || Ok(())) }
            .map_err(VulkanReadySubmitError::into_inner)
    }

    #[allow(dead_code)]
    unsafe fn submit_graphics_command_buffer_and_wait_after_reservation<T, F>(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        before_submit: F,
    ) -> Result<T, VulkanReadySubmitError<T>>
    where
        F: FnOnce() -> Result<T, VulkanError>,
    {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))
            .map_err(VulkanReadySubmitError::ReservationFailed)?;
        let queue = self
            .queues
            .graphics
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing graphics queue".to_owned()))
            .map_err(VulkanReadySubmitError::ReservationFailed)?;

        unsafe {
            submit_command_buffer_and_wait_after_reservation(
                logical_device,
                queue,
                command_buffer,
                &VulkanSubmitSynchronization::default(),
                before_submit,
            )
        }
    }

    /// Submit a graphics command buffer with binary semaphore waits/signals and wait for completion.
    ///
    /// # Safety
    ///
    /// The caller must ensure all binary semaphore payload-state and stage-mask valid usage for
    /// `vkQueueSubmit`: wait semaphores must be signaled or have pending signal operations whose
    /// dependencies have been submitted, signal semaphores must be unsignaled with no conflicting
    /// pending operation, and every wait stage mask must contain only stages supported by the
    /// graphics queue and enabled device features/extensions. Semaphores must not be reused in a way
    /// that violates binary semaphore payload consumption/production rules.
    #[allow(dead_code)]
    pub(super) unsafe fn submit_graphics_command_buffer_and_wait_with_synchronization(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        synchronization: &VulkanSubmitSynchronization<'_>,
    ) -> Result<(), VulkanError> {
        unsafe {
            self.submit_graphics_command_buffer_and_wait_with_synchronization_after_reservation(
                command_buffer,
                synchronization,
                || Ok(()),
            )
        }
        .map_err(VulkanReadySubmitError::into_inner)
    }

    /// Submit a graphics command buffer with a ready callback after all pre-submit reservation.
    ///
    /// # Safety
    ///
    /// The caller must uphold the same semaphore payload-state and stage-mask valid-usage contract as
    /// [`VulkanDeviceState::submit_graphics_command_buffer_and_wait_with_synchronization`]. The
    /// callback runs while queue, command-pool, and semaphore host-access locks are held; it must be
    /// short and must not re-enter this Vulkan device submit path.
    #[allow(dead_code)]
    unsafe fn submit_graphics_command_buffer_and_wait_with_synchronization_after_reservation<T, F>(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        synchronization: &VulkanSubmitSynchronization<'_>,
        before_submit: F,
    ) -> Result<T, VulkanReadySubmitError<T>>
    where
        F: FnOnce() -> Result<T, VulkanError>,
    {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))
            .map_err(VulkanReadySubmitError::ReservationFailed)?;
        let queue = self
            .queues
            .graphics
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing graphics queue".to_owned()))
            .map_err(VulkanReadySubmitError::ReservationFailed)?;

        // SAFETY: The caller upholds the semaphore payload-state and stage-mask valid-usage
        // requirements for the supplied synchronization description. This method selected the
        // graphics queue matching the command buffer's queue-family check below.
        unsafe {
            submit_command_buffer_and_wait_after_reservation(
                logical_device,
                queue,
                command_buffer,
                synchronization,
                before_submit,
            )
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn submit_graphics_command_buffer_with_synchronization_for_tests(
        &self,
        command_buffer: VulkanCommandBuffer,
        synchronization: &VulkanSubmitSynchronization<'_>,
    ) -> Result<VulkanSubmittedCommandBuffer, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let queue =
            self.queues.graphics.as_ref().ok_or_else(|| {
                VulkanError::DeviceInitializationFailed("missing graphics queue".to_owned())
            })?;

        // SAFETY: Forwarded from this test-only helper's caller. This selects the same graphics
        // queue as the waited helper and returns an owner that retains submitted resources until the
        // test waits or drops it.
        unsafe { submit_owned_command_buffer(logical_device, queue, command_buffer, synchronization) }
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

        submit_command_buffer_and_wait_without_synchronization(logical_device, queue, command_buffer)
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
    pub(super) fn record_sampled_dmabuf_foreign_acquire_barrier(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        image: &VulkanOwnedImage,
    ) -> Result<bool, VulkanError> {
        record_sampled_dmabuf_foreign_acquire_barrier(command_buffer, image)
    }

    #[allow(dead_code)]
    pub(super) fn record_sampled_dmabuf_foreign_release_barrier(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        image: &VulkanOwnedImage,
    ) -> Result<bool, VulkanError> {
        record_sampled_dmabuf_foreign_release_barrier(command_buffer, image)
    }

    #[allow(dead_code)]
    pub(super) fn record_dmabuf_render_target_foreign_acquire_barrier(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        image: &VulkanOwnedImage,
        preserve_contents: bool,
    ) -> Result<bool, VulkanError> {
        record_dmabuf_render_target_foreign_acquire_barrier(command_buffer, image, preserve_contents)
    }

    #[allow(dead_code)]
    pub(super) fn record_dmabuf_render_target_foreign_release_barrier(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        image: &VulkanOwnedImage,
    ) -> Result<bool, VulkanError> {
        record_dmabuf_render_target_foreign_release_barrier(command_buffer, image)
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
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        ensure_logical_device_image_device(logical_device, image, "image view device")?;

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
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        ensure_logical_device_image_device(logical_device, image, "image device")?;
        ensure_image_locally_usable(image)?;

        let view = self.create_color_attachment_image_view(image)?;
        let render_pass = self.single_color_clear_render_pass(image.format())?;
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
    pub(super) fn clear_color_attachment_image_in(
        &self,
        image: &VulkanOwnedImage,
        color: vk::ClearColorValue,
        clear_areas: &[vk::Rect2D],
    ) -> Result<(), VulkanError> {
        if clear_areas.is_empty() {
            return Ok(());
        }
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        ensure_logical_device_image_device(logical_device, image, "image device")?;
        ensure_image_locally_usable(image)?;

        let view = self.create_color_attachment_image_view(image)?;
        let render_pass = self.single_color_load_render_pass(image.format())?;
        let framebuffer = create_single_color_framebuffer(&render_pass, &view, image.extent())?;
        let mut command_buffer = self.allocate_graphics_command_buffer()?;

        self.begin_command_buffer(&mut command_buffer)?;
        self.transition_image_layout(
            &mut command_buffer,
            image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;
        synchronize_color_attachment_load(&mut command_buffer, image)?;
        record_color_attachment_clear_rects(
            &mut command_buffer,
            image,
            &render_pass,
            &framebuffer,
            color,
            clear_areas,
        )?;
        self.end_command_buffer(&mut command_buffer)?;
        self.submit_graphics_command_buffer_and_wait(&mut command_buffer)
    }

    #[allow(dead_code)]
    pub(super) fn single_color_clear_render_pass(
        &self,
        format: vk::Format,
    ) -> Result<Arc<VulkanRenderPass>, VulkanError> {
        self.single_color_render_pass(format, VulkanSingleColorRenderPassLoadOp::Clear)
    }

    #[allow(dead_code)]
    pub(super) fn single_color_load_render_pass(
        &self,
        format: vk::Format,
    ) -> Result<Arc<VulkanRenderPass>, VulkanError> {
        self.single_color_render_pass(format, VulkanSingleColorRenderPassLoadOp::Load)
    }

    fn single_color_render_pass(
        &self,
        format: vk::Format,
        load_op: VulkanSingleColorRenderPassLoadOp,
    ) -> Result<Arc<VulkanRenderPass>, VulkanError> {
        let key = (format, load_op);
        let mut render_passes = self
            .single_color_render_passes
            .lock()
            .map_err(|_| host_synchronization_failed())?;
        if let Some(render_pass) = render_passes.get(&key) {
            return Ok(Arc::clone(render_pass));
        }

        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        let render_pass = Arc::new(create_single_color_render_pass_with_load_op(
            logical_device,
            format,
            load_op.to_vk(),
        )?);
        render_passes.insert(key, Arc::clone(&render_pass));
        Ok(render_pass)
    }

    #[allow(dead_code)]
    pub(super) fn create_color_attachment_image_view(
        &self,
        image: &VulkanOwnedImage,
    ) -> Result<VulkanImageView, VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        ensure_logical_device_image_device(logical_device, image, "image view device")?;

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
    pub(super) fn sampled_texture_descriptor_set_layout(
        &self,
    ) -> Result<Arc<VulkanDescriptorSetLayout>, VulkanError> {
        let mut layout = self
            .sampled_texture_descriptor_set_layout
            .lock()
            .map_err(|_| host_synchronization_failed())?;
        if let Some(layout) = layout.as_ref() {
            return Ok(Arc::clone(layout));
        }

        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();
        let cached = Arc::new(create_sampled_texture_descriptor_set_layout(&logical_device)?);
        *layout = Some(Arc::clone(&cached));
        Ok(cached)
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
        let descriptor_set_layout = self.sampled_texture_descriptor_set_layout()?;
        let pipeline_layout = create_pipeline_layout_for_descriptor_set_layout(&descriptor_set_layout)?;

        Ok(VulkanSampledTexturePipelineLayout {
            pipeline_layout,
            descriptor_set_layout,
        })
    }

    #[allow(dead_code)]
    pub(super) fn sampled_texture_pipeline_layout(
        &self,
    ) -> Result<Arc<VulkanSampledTexturePipelineLayout>, VulkanError> {
        let mut layout = self
            .sampled_texture_pipeline_layout
            .lock()
            .map_err(|_| host_synchronization_failed())?;
        if let Some(layout) = layout.as_ref() {
            return Ok(Arc::clone(layout));
        }

        let cached = Arc::new(self.create_sampled_texture_pipeline_layout()?);
        *layout = Some(Arc::clone(&cached));
        Ok(cached)
    }

    #[allow(dead_code)]
    pub(super) fn solid_color_pipeline_layout(&self) -> Result<Arc<VulkanPipelineLayout>, VulkanError> {
        let mut layout = self
            .solid_color_pipeline_layout
            .lock()
            .map_err(|_| host_synchronization_failed())?;
        if let Some(layout) = layout.as_ref() {
            return Ok(Arc::clone(layout));
        }

        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?
            .clone();
        let cached = Arc::new(create_solid_color_pipeline_layout(&logical_device)?);
        *layout = Some(Arc::clone(&cached));
        Ok(cached)
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
        blend_enabled: bool,
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
        if blend_enabled
            && !format_properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
        {
            return Err(VulkanError::UnsupportedOperation(
                "graphics pipeline blend format",
            ));
        }

        let vertex_shader = create_shader_module(&logical_device, shaders.vertex)?;
        let fragment_shader = create_shader_module(&logical_device, shaders.fragment)?;
        let sampled_texture_layout = self.sampled_texture_pipeline_layout()?;
        let render_pass = self.single_color_load_render_pass(shaders.color_format)?;
        let pipeline = create_sampled_texture_graphics_pipeline(
            &logical_device,
            &render_pass,
            sampled_texture_layout.pipeline_layout(),
            &vertex_shader,
            &fragment_shader,
            blend_enabled,
        )?;

        Ok(VulkanSampledTextureGraphicsPipeline {
            color_format: shaders.color_format,
            blend_enabled,
            render_pass,
            layout: sampled_texture_layout,
            pipeline,
        })
    }

    #[allow(dead_code)]
    pub(super) fn create_builtin_sampled_texture_graphics_pipeline(
        &self,
        color_format: vk::Format,
        blend_enabled: bool,
    ) -> Result<VulkanSampledTextureGraphicsPipeline, VulkanError> {
        // SAFETY: These built-in shader modules were generated from local GLSL by glslangValidator.
        // They contain compatible vertex/fragment `main` entry points, no non-built-in vertex
        // inputs, matching location interfaces, set 0 binding 0 as a combined image sampler, and
        // one color output compatible with the renderer's UNORM color-attachment formats. The
        // fragment shader reads a 32-byte push-constant block containing affine UV mapping axes,
        // global alpha value, and opaque-alpha flag covered by the pipeline layout.
        let shaders = unsafe {
            VulkanSampledTexturePipelineShaders::from_spirv_unchecked(
                color_format,
                BUILTIN_TEXTURED_VERTEX_SHADER_SPIRV,
                BUILTIN_TEXTURED_FRAGMENT_SHADER_SPIRV,
            )
        }?;

        self.create_sampled_texture_graphics_pipeline(shaders, blend_enabled)
    }

    #[allow(dead_code)]
    pub(super) fn builtin_sampled_texture_graphics_pipeline(
        &self,
        color_format: vk::Format,
        blend_enabled: bool,
    ) -> Result<Arc<VulkanSampledTextureGraphicsPipeline>, VulkanError> {
        let key = (color_format, blend_enabled);
        let mut pipelines = self
            .builtin_sampled_texture_pipelines
            .lock()
            .map_err(|_| host_synchronization_failed())?;
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(Arc::clone(pipeline));
        }

        let pipeline =
            Arc::new(self.create_builtin_sampled_texture_graphics_pipeline(color_format, blend_enabled)?);
        pipelines.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }

    #[allow(dead_code)]
    pub(super) fn create_builtin_solid_color_graphics_pipeline(
        &self,
        color_format: vk::Format,
        blend_enabled: bool,
    ) -> Result<VulkanSolidColorGraphicsPipeline, VulkanError> {
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
            color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )?;
        // SAFETY: `physical_device` belongs to `instance`, and querying format properties is
        // read-only with no additional extension or lifetime requirements.
        let format_properties = unsafe {
            instance
                .handle()
                .get_physical_device_format_properties(physical_device.handle(), color_format)
        };
        if blend_enabled
            && !format_properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
        {
            return Err(VulkanError::UnsupportedOperation("solid pipeline blend format"));
        }

        let vertex_shader = create_shader_module(
            &logical_device,
            // SAFETY: This built-in shader module was generated from local GLSL by
            // glslangValidator. It contains a vertex `main` entry point with no non-built-in inputs
            // and emits a fullscreen triangle suitable for dynamic viewport/scissor rendering.
            unsafe { VulkanShaderSpirv::from_words_unchecked(BUILTIN_TEXTURED_VERTEX_SHADER_SPIRV)? },
        )?;
        let fragment_shader = create_shader_module(
            &logical_device,
            // SAFETY: This built-in shader module was generated from local GLSL by
            // glslangValidator. It contains a fragment `main` entry point, writes one location-0
            // color output, and reads a 16-byte fragment push-constant block containing `vec4 color`.
            unsafe { VulkanShaderSpirv::from_words_unchecked(BUILTIN_SOLID_FRAGMENT_SHADER_SPIRV)? },
        )?;
        let layout = self.solid_color_pipeline_layout()?;
        let render_pass = self.single_color_load_render_pass(color_format)?;
        let pipeline = create_sampled_texture_graphics_pipeline(
            &logical_device,
            &render_pass,
            &layout,
            &vertex_shader,
            &fragment_shader,
            blend_enabled,
        )?;

        Ok(VulkanSolidColorGraphicsPipeline {
            color_format,
            blend_enabled,
            render_pass,
            layout,
            pipeline,
        })
    }

    #[allow(dead_code)]
    pub(super) fn builtin_solid_color_graphics_pipeline(
        &self,
        color_format: vk::Format,
        blend_enabled: bool,
    ) -> Result<Arc<VulkanSolidColorGraphicsPipeline>, VulkanError> {
        let key = (color_format, blend_enabled);
        let mut pipelines = self
            .builtin_solid_color_pipelines
            .lock()
            .map_err(|_| host_synchronization_failed())?;
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(Arc::clone(pipeline));
        }

        let pipeline =
            Arc::new(self.create_builtin_solid_color_graphics_pipeline(color_format, blend_enabled)?);
        pipelines.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
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
                scissor_area: draw_area,
                uv_origin: [0.0, 0.0],
                uv_x_axis: [1.0, 0.0],
                uv_y_axis: [0.0, 1.0],
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
    pub(super) fn render_solid_color_to_color_image_in(
        &self,
        target: &VulkanOwnedImage,
        pipeline: &VulkanSolidColorGraphicsPipeline,
        draw_constants: VulkanSolidColorDrawConstants,
    ) -> Result<(), VulkanError> {
        let logical_device = self
            .logical_device
            .as_ref()
            .ok_or_else(|| VulkanError::DeviceInitializationFailed("missing logical device".to_owned()))?;
        validate_solid_color_draw_inputs(logical_device, target, pipeline)?;
        validate_color_attachment_area(target, draw_constants.draw_area, "solid color draw area")?;
        validate_color_attachment_area(target, draw_constants.scissor_area, "solid color draw area")?;
        validate_solid_color_draw_constants(draw_constants)?;

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
        record_solid_color_draw(
            &mut command_buffer,
            target,
            pipeline,
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
        self.read_image_region_to_tightly_packed_buffer(
            image,
            vk::Offset3D { x: 0, y: 0, z: 0 },
            image.extent(),
        )
    }

    #[allow(dead_code)]
    pub(super) fn read_image_region_to_tightly_packed_buffer(
        &self,
        image: &VulkanOwnedImage,
        image_offset: vk::Offset3D,
        extent: vk::Extent3D,
    ) -> Result<Vec<u8>, VulkanError> {
        let readback_size = tightly_packed_image_size(image.format(), extent)?;
        let readback_buffer =
            self.create_host_visible_buffer(readback_size, vk::BufferUsageFlags::TRANSFER_DST)?;
        let mut command_buffer = self.allocate_graphics_command_buffer()?;

        self.begin_command_buffer(&mut command_buffer)?;
        self.transition_image_layout(&mut command_buffer, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)?;
        self.copy_image_region_to_buffer(&mut command_buffer, image, &readback_buffer, image_offset, extent)?;
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
        self.copy_image_region_to_buffer(
            command_buffer,
            image,
            buffer,
            vk::Offset3D { x: 0, y: 0, z: 0 },
            extent,
        )
    }

    #[allow(dead_code)]
    pub(super) fn copy_image_region_to_buffer(
        &self,
        command_buffer: &mut VulkanCommandBuffer,
        image: &VulkanOwnedImage,
        buffer: &VulkanHostVisibleBuffer,
        image_offset: vk::Offset3D,
        extent: vk::Extent3D,
    ) -> Result<(), VulkanError> {
        copy_image_region_to_buffer(command_buffer, image, buffer, image_offset, extent)
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
    supports_graphics: bool,
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
        queue_family_index,
        supports_graphics,
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
        state: VulkanCommandBufferState::Initial,
        pending_image_layouts: Vec::new(),
        pending_image_syncs: Vec::new(),
        referenced_buffers: Vec::new(),
        referenced_images: Vec::new(),
    })
}

fn begin_command_buffer(command_buffer: &mut VulkanCommandBuffer) -> Result<(), VulkanError> {
    if command_buffer.state != VulkanCommandBufferState::Initial {
        return Err(VulkanError::UnsupportedOperation("command buffer recording"));
    }

    let _pool_guard = command_buffer.command_pool.lock_host_access()?;
    command_buffer.pending_image_layouts.clear();
    command_buffer.pending_image_syncs.clear();
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
    .map_err(VulkanError::from)?;

    command_buffer.state = VulkanCommandBufferState::Recording;
    Ok(())
}

fn ensure_command_buffer_recording(command_buffer: &VulkanCommandBuffer) -> Result<(), VulkanError> {
    if command_buffer.is_recording() {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedOperation("command buffer recording"))
    }
}

fn ensure_command_buffer_executable(command_buffer: &VulkanCommandBuffer) -> Result<(), VulkanError> {
    if command_buffer.state == VulkanCommandBufferState::Executable {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedOperation("command buffer executable"))
    }
}

fn ensure_graphics_command_buffer(command_buffer: &VulkanCommandBuffer) -> Result<(), VulkanError> {
    if command_buffer.command_pool.supports_graphics() {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedOperation("command buffer graphics queue"))
    }
}

fn ensure_command_buffer_image_device(
    command_buffer: &VulkanCommandBuffer,
    image: &VulkanOwnedImage,
) -> Result<(), VulkanError> {
    ensure_logical_device_image_device(
        &command_buffer.command_pool.logical_device,
        image,
        "command buffer image device",
    )
}

fn ensure_logical_device_image_device(
    logical_device: &VulkanLogicalDevice,
    image: &VulkanOwnedImage,
    reason: &'static str,
) -> Result<(), VulkanError> {
    if logical_device.is_same_device(&image.inner.logical_device) {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedOperation(reason))
    }
}

fn ensure_command_buffer_buffer_device(
    command_buffer: &VulkanCommandBuffer,
    buffer: &VulkanHostVisibleBuffer,
) -> Result<(), VulkanError> {
    if command_buffer
        .command_pool
        .logical_device
        .is_same_device(&buffer.inner.logical_device)
    {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedOperation("command buffer buffer device"))
    }
}

fn ensure_image_locally_usable(image: &VulkanOwnedImage) -> Result<(), VulkanError> {
    if image.sync_state()?.is_locally_usable() {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    }
}

fn ensure_image_locally_usable_for_recorded_color_attachment_work(
    command_buffer: &VulkanCommandBuffer,
    image: &VulkanOwnedImage,
) -> Result<(), VulkanError> {
    if command_buffer
        .projected_dmabuf_render_target_local_sync(image)?
        .is_locally_usable()
    {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
    }
}

fn ensure_image_locally_usable_for_recorded_layout_transition(
    command_buffer: &VulkanCommandBuffer,
    image: &VulkanOwnedImage,
    new_layout: vk::ImageLayout,
) -> Result<(), VulkanError> {
    if image.sync_state()?.is_locally_usable() {
        return Ok(());
    }
    if new_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
        return ensure_image_locally_usable_for_recorded_color_attachment_work(command_buffer, image);
    }

    Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
}

fn vulkan_error_is_device_lost(err: &VulkanError) -> bool {
    matches!(
        err,
        VulkanError::DeviceLost | VulkanError::VulkanApi(vk::Result::ERROR_DEVICE_LOST)
    )
}

fn vulkan_error_invalidates_context(err: &VulkanError) -> bool {
    match err {
        VulkanError::DeviceLost => true,
        VulkanError::VulkanApi(result) => vulkan_api_result_invalidates_context(*result),
        VulkanError::VulkanUnavailable
        | VulkanError::MissingRequiredExtension(_)
        | VulkanError::DeviceInitializationFailed(_)
        | VulkanError::QueueFamilyUnsupported
        | VulkanError::ExternalMemoryUnsupported => true,
        VulkanError::UnsupportedOperation(_)
        | VulkanError::NotPublicAdvertised(_)
        | VulkanError::MissingCapability(_)
        | VulkanError::UnsupportedFormat(_)
        | VulkanError::UnsupportedModifier
        | VulkanError::SyncInterrupted
        | VulkanError::MemoryTypeUnsupported => false,
    }
}

fn submit_command_buffer_and_wait_without_synchronization(
    logical_device: &VulkanLogicalDevice,
    queue: &VulkanQueue,
    command_buffer: &mut VulkanCommandBuffer,
) -> Result<(), VulkanError> {
    // SAFETY: An empty synchronization description does not include any semaphore wait/signal
    // payload-state or stage-mask requirements beyond the command-buffer/queue/fence checks in the
    // submit helper.
    unsafe {
        submit_command_buffer_and_wait(
            logical_device,
            queue,
            command_buffer,
            &VulkanSubmitSynchronization::default(),
        )
    }
}

unsafe fn submit_command_buffer_and_wait(
    logical_device: &VulkanLogicalDevice,
    queue: &VulkanQueue,
    command_buffer: &mut VulkanCommandBuffer,
    synchronization: &VulkanSubmitSynchronization<'_>,
) -> Result<(), VulkanError> {
    unsafe {
        submit_command_buffer_and_wait_after_reservation(
            logical_device,
            queue,
            command_buffer,
            synchronization,
            || Ok(()),
        )
    }
    .map_err(VulkanReadySubmitError::into_inner)
}

unsafe fn submit_command_buffer_and_wait_after_reservation<T, F>(
    logical_device: &VulkanLogicalDevice,
    queue: &VulkanQueue,
    command_buffer: &mut VulkanCommandBuffer,
    synchronization: &VulkanSubmitSynchronization<'_>,
    before_submit: F,
) -> Result<T, VulkanReadySubmitError<T>>
where
    F: FnOnce() -> Result<T, VulkanError>,
{
    ensure_command_buffer_executable(command_buffer).map_err(VulkanReadySubmitError::ReservationFailed)?;
    if !logical_device.is_same_device(&command_buffer.command_pool.logical_device) {
        return Err(VulkanReadySubmitError::ReservationFailed(
            VulkanError::UnsupportedOperation("command buffer device"),
        ));
    }
    if command_buffer.queue_family_index() != queue.queue_family_index() {
        return Err(VulkanReadySubmitError::ReservationFailed(
            VulkanError::UnsupportedOperation("command buffer queue family"),
        ));
    }
    synchronization
        .validate(logical_device)
        .map_err(VulkanReadySubmitError::ReservationFailed)?;

    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe { logical_device.handle().create_fence(&fence_info, None) }
        .map_err(VulkanError::from)
        .map_err(VulkanReadySubmitError::ReservationFailed)?;
    let command_buffers = [command_buffer.handle];
    let wait_semaphores = synchronization.wait_handles();
    let wait_stage_masks = synchronization.wait_stage_masks();
    let signal_semaphores = synchronization.signal_handles();
    let submit_infos = [vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stage_masks)
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores)];
    let semaphore_guards = match synchronization.lock_host_access() {
        Ok(guards) => guards,
        Err(err) => {
            unsafe { logical_device.handle().destroy_fence(fence, None) };
            return Err(VulkanReadySubmitError::ReservationFailed(err));
        }
    };
    let pending_semaphore_payloads = match synchronization.mark_payloads_pending_submit() {
        Ok(pending) => pending,
        Err(err) => {
            drop(semaphore_guards);
            unsafe { logical_device.handle().destroy_fence(fence, None) };
            return Err(VulkanReadySubmitError::ReservationFailed(err));
        }
    };

    let command_pool = command_buffer.command_pool.clone();
    let pool_guard = match command_pool.lock_host_access() {
        Ok(guard) => guard,
        Err(err) => {
            let _ = restore_pending_semaphore_payloads(&pending_semaphore_payloads);
            let abort_result = command_buffer.abort_pending_image_syncs();
            command_buffer.state = VulkanCommandBufferState::Invalid;
            drop(semaphore_guards);
            unsafe { logical_device.handle().destroy_fence(fence, None) };
            if let Err(abort_err) = abort_result {
                return Err(VulkanReadySubmitError::ReservationFailed(abort_err));
            }
            return Err(VulkanReadySubmitError::ReservationFailed(err));
        }
    };
    let queue_guard = match queue.lock_host_access() {
        Ok(guard) => guard,
        Err(err) => {
            let _ = restore_pending_semaphore_payloads(&pending_semaphore_payloads);
            let abort_result = command_buffer.abort_pending_image_syncs();
            command_buffer.state = VulkanCommandBufferState::Invalid;
            drop(pool_guard);
            drop(semaphore_guards);
            unsafe { logical_device.handle().destroy_fence(fence, None) };
            if let Err(abort_err) = abort_result {
                return Err(VulkanReadySubmitError::ReservationFailed(abort_err));
            }
            return Err(VulkanReadySubmitError::ReservationFailed(err));
        }
    };

    let ready = match before_submit() {
        Ok(ready) => ready,
        Err(err) => {
            let _ = restore_pending_semaphore_payloads(&pending_semaphore_payloads);
            let abort_result = command_buffer.abort_pending_image_syncs();
            command_buffer.state = VulkanCommandBufferState::Invalid;
            drop(queue_guard);
            drop(pool_guard);
            drop(semaphore_guards);
            unsafe { logical_device.handle().destroy_fence(fence, None) };
            if let Err(abort_err) = abort_result {
                return Err(VulkanReadySubmitError::ReadyCallbackFailed(abort_err));
            }
            return Err(VulkanReadySubmitError::ReadyCallbackFailed(err));
        }
    };

    // SAFETY: `logical_device`, `queue.handle`, `command_buffer.handle`, `fence`, and all semaphores
    // in `submit_infos` belong to the same live device and are kept alive through the fence wait below.
    // The queue host-access mutex externally synchronizes host access to the queue. Semaphore
    // host-access mutexes prevent concurrent import/export/destroy while the submit is pending. Slice
    // storage used by `submit_infos` lives until `queue_submit` returns. The caller of this unsafe
    // helper upholds binary semaphore payload-state and wait-stage valid usage.
    let submit_result = unsafe {
        logical_device
            .handle()
            .queue_submit(queue.handle, &submit_infos, fence)
    }
    .map_err(VulkanError::from);
    drop(queue_guard);
    drop(pool_guard);

    if let Err(err) = submit_result {
        let _ = restore_pending_semaphore_payloads(&pending_semaphore_payloads);
        let abort_result = command_buffer.abort_pending_image_syncs();
        command_buffer.state = VulkanCommandBufferState::Invalid;
        drop(semaphore_guards);
        unsafe { logical_device.handle().destroy_fence(fence, None) };
        if let Err(abort_err) = abort_result {
            return Err(VulkanReadySubmitError::ReadyCallbackCommitted {
                err: abort_err,
                ready,
            });
        }
        return Err(VulkanReadySubmitError::ReadyCallbackCommitted { err, ready });
    }

    command_buffer.state = VulkanCommandBufferState::Submitted;
    let wait_result = unsafe { logical_device.handle().wait_for_fences(&[fence], true, u64::MAX) }
        .map_err(VulkanError::from);
    let mut queue_completed = wait_result.is_ok();
    let mut can_destroy_submitted_objects = wait_result.is_ok();
    let wait_error = match wait_result {
        Ok(()) => None,
        Err(wait_err) => {
            let wait_reported_device_lost = vulkan_error_is_device_lost(&wait_err);
            let queue_idle_result = queue.lock_host_access().and_then(|_queue_guard| {
                // SAFETY: `queue.handle` belongs to `logical_device`, and queue host access is
                // externally synchronized by the queue mutex. Waiting the queue after a fence wait
                // error either proves the submitted command buffer completed or reports device loss,
                // which makes the context unusable and prevents later false ownership completion.
                unsafe { logical_device.handle().queue_wait_idle(queue.handle) }.map_err(VulkanError::from)
            });
            match queue_idle_result {
                Ok(()) => {
                    queue_completed = true;
                    can_destroy_submitted_objects = true;
                    Some(wait_err)
                }
                Err(idle_err) => {
                    can_destroy_submitted_objects =
                        wait_reported_device_lost || vulkan_error_is_device_lost(&idle_err);
                    Some(if wait_reported_device_lost {
                        wait_err
                    } else {
                        idle_err
                    })
                }
            }
        }
    };
    let image_result = if queue_completed {
        Some(command_buffer.commit_pending_image_layouts_and_syncs())
    } else {
        // The queue accepted this command buffer, but neither the fence wait nor queue-idle fallback
        // proved completion. This is expected only for context-invalidating failures such as device
        // loss. Leave image ownership reservations in their non-locally-usable pending state rather
        // than falsely restoring or completing ownership that may have transferred on the GPU.
        None
    };
    let complete_result = if queue_completed {
        complete_pending_semaphore_payloads(&pending_semaphore_payloads)
    } else {
        Ok(())
    };
    drop(semaphore_guards);

    if can_destroy_submitted_objects {
        unsafe { logical_device.handle().destroy_fence(fence, None) };
    } else {
        command_buffer.state = VulkanCommandBufferState::SubmitCompletionUnknown;
        // `VulkanPendingSemaphorePayload` owns semaphore clones specifically for this path: the
        // submitted queue may still reference them, and the caller's borrowed semaphore handles may
        // be dropped after this function returns.
        std::mem::forget(pending_semaphore_payloads);
    }

    if let Some(err) = wait_error {
        if let Some(image_result) = image_result {
            if let Err(image_err) = image_result {
                return Err(VulkanReadySubmitError::Submitted {
                    err: image_err,
                    ready,
                });
            }
        }
        if let Err(complete_err) = complete_result {
            return Err(VulkanReadySubmitError::Submitted {
                err: complete_err,
                ready,
            });
        }
        return Err(VulkanReadySubmitError::Submitted { err, ready });
    }
    if let Err(image_err) = image_result.expect("queue completion proven without wait error") {
        return Err(VulkanReadySubmitError::Submitted {
            err: image_err,
            ready,
        });
    }
    if let Err(complete_err) = complete_result {
        return Err(VulkanReadySubmitError::Submitted {
            err: complete_err,
            ready,
        });
    }
    Ok(ready)
}

unsafe fn submit_owned_command_buffer(
    logical_device: &VulkanLogicalDevice,
    queue: &VulkanQueue,
    mut command_buffer: VulkanCommandBuffer,
    synchronization: &VulkanSubmitSynchronization<'_>,
) -> Result<VulkanSubmittedCommandBuffer, VulkanError> {
    ensure_command_buffer_executable(&command_buffer)?;
    if !logical_device.is_same_device(&command_buffer.command_pool.logical_device) {
        return Err(VulkanError::UnsupportedOperation("command buffer device"));
    }
    if command_buffer.queue_family_index() != queue.queue_family_index() {
        return Err(VulkanError::UnsupportedOperation("command buffer queue family"));
    }
    synchronization.validate(logical_device)?;

    let fence_info = vk::FenceCreateInfo::default();
    let fence =
        unsafe { logical_device.handle().create_fence(&fence_info, None) }.map_err(VulkanError::from)?;
    let command_buffers = [command_buffer.handle];
    let wait_semaphores = synchronization.wait_handles();
    let wait_stage_masks = synchronization.wait_stage_masks();
    let signal_semaphores = synchronization.signal_handles();
    let submit_infos = [vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stage_masks)
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores)];
    let semaphore_guards = match synchronization.lock_host_access() {
        Ok(guards) => guards,
        Err(err) => {
            unsafe { logical_device.handle().destroy_fence(fence, None) };
            return Err(err);
        }
    };
    let pending_semaphore_payloads = match synchronization.mark_payloads_pending_submit() {
        Ok(pending) => pending,
        Err(err) => {
            drop(semaphore_guards);
            unsafe { logical_device.handle().destroy_fence(fence, None) };
            return Err(err);
        }
    };

    let submit_result = command_buffer
        .command_pool
        .lock_host_access()
        .and_then(|_pool_guard| {
            queue.lock_host_access().and_then(|_queue_guard| {
                // SAFETY: `logical_device`, `queue.handle`, `command_buffer.handle`, `fence`, and
                // all semaphores in `submit_infos` belong to the same live device. The returned
                // `VulkanSubmittedCommandBuffer` retains the command buffer, fence, image/buffer
                // references, and semaphore clones until fence completion. Queue and semaphore host
                // access are externally synchronized by their mutexes. Slice storage used by
                // `submit_infos` lives until `queue_submit` returns. The caller upholds binary
                // semaphore payload-state and wait-stage valid usage.
                unsafe {
                    logical_device
                        .handle()
                        .queue_submit(queue.handle, &submit_infos, fence)
                }
                .map_err(VulkanError::from)
            })
        });

    if let Err(err) = submit_result {
        let _ = restore_pending_semaphore_payloads(&pending_semaphore_payloads);
        let abort_result = command_buffer.abort_pending_image_syncs();
        command_buffer.state = VulkanCommandBufferState::Invalid;
        drop(semaphore_guards);
        unsafe { logical_device.handle().destroy_fence(fence, None) };
        abort_result?;
        return Err(err);
    }

    command_buffer.state = VulkanCommandBufferState::Submitted;
    drop(semaphore_guards);
    Ok(VulkanSubmittedCommandBuffer::new(
        logical_device.clone(),
        command_buffer,
        fence,
        pending_semaphore_payloads,
    ))
}

fn transition_image_layout(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
    new_layout: vk::ImageLayout,
) -> Result<(), VulkanError> {
    ensure_command_buffer_recording(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_image_locally_usable_for_recorded_layout_transition(command_buffer, image, new_layout)?;
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

fn record_sampled_dmabuf_foreign_acquire_barrier(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
) -> Result<bool, VulkanError> {
    ensure_command_buffer_recording(command_buffer)?;
    ensure_graphics_command_buffer(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_dmabuf_external_image(image)?;
    let Some(barrier) =
        command_buffer.plan_sampled_dmabuf_foreign_acquire_barrier(&image.sync_state()?, image.usage())?
    else {
        return Ok(false);
    };
    let vk_barrier = barrier.to_color_image_memory_barrier(image.image());
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    image.inner.sync.begin_sampled_dmabuf_foreign_acquire()?;
    // SAFETY: `command_buffer` is in recording state and belongs to a live command pool/device.
    // `image` is a bound external-memory image retained below until command completion. The barrier
    // is produced by the sampled-dmabuf foreign acquire planner, which validates sampled usage, a
    // local destination queue family, and the known foreign GENERAL layout before selecting the
    // FOREIGN -> local queue-family transfer and GENERAL -> SHADER_READ_ONLY layout transition.
    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_pipeline_barrier(
                command_buffer.handle,
                barrier.src_stage,
                barrier.dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk_barrier],
            )
    };
    command_buffer
        .pending_image_layouts
        .push(VulkanPendingImageLayout {
            image: image.image(),
            resource: Arc::clone(&image.inner),
            new_layout: barrier.new_layout,
        });
    command_buffer.pending_image_syncs.push(VulkanPendingImageSync {
        resource: Arc::clone(&image.inner),
        operation: VulkanPendingImageSyncOperation::SampledDmabufForeignAcquire,
    });
    command_buffer.referenced_images.push(Arc::clone(&image.inner));

    Ok(true)
}

fn record_sampled_dmabuf_foreign_release_barrier(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
) -> Result<bool, VulkanError> {
    ensure_command_buffer_recording(command_buffer)?;
    ensure_graphics_command_buffer(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_dmabuf_external_image(image)?;
    let local_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    let Some(barrier) = command_buffer.plan_sampled_dmabuf_foreign_release_barrier(
        &image.sync_state()?,
        local_layout,
        image.usage(),
    )?
    else {
        return Ok(false);
    };
    let vk_barrier = barrier.to_color_image_memory_barrier(image.image());
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    image.inner.sync.begin_sampled_dmabuf_foreign_release()?;
    // SAFETY: `command_buffer` is in recording state and belongs to a live command pool/device.
    // `image` is a bound external-memory image retained below until command completion. The barrier
    // is produced by the sampled-dmabuf foreign release planner, which validates sampled usage,
    // current local SHADER_READ_ONLY layout, and a local source queue family before selecting the
    // local -> FOREIGN queue-family transfer and SHADER_READ_ONLY -> GENERAL layout transition.
    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_pipeline_barrier(
                command_buffer.handle,
                barrier.src_stage,
                barrier.dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk_barrier],
            )
    };
    command_buffer
        .pending_image_layouts
        .push(VulkanPendingImageLayout {
            image: image.image(),
            resource: Arc::clone(&image.inner),
            new_layout: barrier.new_layout,
        });
    command_buffer.pending_image_syncs.push(VulkanPendingImageSync {
        resource: Arc::clone(&image.inner),
        operation: VulkanPendingImageSyncOperation::SampledDmabufForeignRelease,
    });
    command_buffer.referenced_images.push(Arc::clone(&image.inner));

    Ok(true)
}

#[allow(dead_code)]
fn record_dmabuf_render_target_foreign_acquire_barrier(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
    preserve_contents: bool,
) -> Result<bool, VulkanError> {
    ensure_command_buffer_recording(command_buffer)?;
    ensure_graphics_command_buffer(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_dmabuf_external_image(image)?;
    let Some(_) = command_buffer.plan_dmabuf_render_target_foreign_acquire_barrier(
        &image.sync_state()?,
        image.usage(),
        preserve_contents,
    )?
    else {
        return Ok(false);
    };

    let restore = image
        .inner
        .sync
        .begin_dmabuf_render_target_foreign_acquire(preserve_contents)?;
    let external_layout = match restore.ownership() {
        VulkanExternalImageOwnership::ForeignUnknown => vk::ImageLayout::UNDEFINED,
        VulkanExternalImageOwnership::ForeignKnownGeneral => vk::ImageLayout::GENERAL,
        VulkanExternalImageOwnership::None
        | VulkanExternalImageOwnership::AcquirePending
        | VulkanExternalImageOwnership::Local
        | VulkanExternalImageOwnership::ReleasePending => {
            let err = VulkanError::UnsupportedOperation("dmabuf external ownership");
            image
                .inner
                .sync
                .abort_dmabuf_render_target_foreign_acquire(restore)?;
            return Err(err);
        }
    };
    let barrier = match dmabuf_render_target_foreign_acquire_barrier(
        external_layout,
        command_buffer.queue_family_index(),
        image.usage(),
    ) {
        Ok(barrier) => barrier,
        Err(err) => {
            image
                .inner
                .sync
                .abort_dmabuf_render_target_foreign_acquire(restore)?;
            return Err(err);
        }
    };
    let vk_barrier = barrier.to_color_image_memory_barrier(image.image());
    let pool_guard = match command_buffer.command_pool.lock_host_access() {
        Ok(guard) => guard,
        Err(err) => {
            image
                .inner
                .sync
                .abort_dmabuf_render_target_foreign_acquire(restore)?;
            return Err(err);
        }
    };

    // SAFETY: `command_buffer` is in recording state and belongs to a live graphics command
    // pool/device. `image` is a bound dmabuf external-memory image retained below until command
    // completion. The barrier is built from the restore token returned by the host-side begin step:
    // unknown discard acquires use UNDEFINED, while known-general acquires use GENERAL even when the
    // caller does not need to preserve contents. The barrier helper validates color-attachment usage
    // and a local destination queue family before selecting the FOREIGN -> local ownership transfer
    // and COLOR_ATTACHMENT layout transition.
    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_pipeline_barrier(
                command_buffer.handle,
                barrier.src_stage,
                barrier.dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk_barrier],
            )
    };
    drop(pool_guard);
    command_buffer
        .pending_image_layouts
        .push(VulkanPendingImageLayout {
            image: image.image(),
            resource: Arc::clone(&image.inner),
            new_layout: barrier.new_layout,
        });
    command_buffer.pending_image_syncs.push(VulkanPendingImageSync {
        resource: Arc::clone(&image.inner),
        operation: VulkanPendingImageSyncOperation::DmabufRenderTargetForeignAcquire { restore },
    });
    command_buffer.referenced_images.push(Arc::clone(&image.inner));

    Ok(true)
}

#[allow(dead_code)]
/// Record a dmabuf render-target release for an image locally owned by the renderer.
///
/// The release planner also projects earlier pending render-target acquires in this command buffer
/// as locally owned, which allows acquire -> release barrier ordering in one submission without
/// advertising public dmabuf render-target support yet.
fn record_dmabuf_render_target_foreign_release_barrier(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
) -> Result<bool, VulkanError> {
    ensure_command_buffer_recording(command_buffer)?;
    ensure_graphics_command_buffer(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_dmabuf_external_image(image)?;
    let local_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    let sync = command_buffer.projected_dmabuf_render_target_release_sync(image)?;
    let Some(barrier) = command_buffer.plan_dmabuf_render_target_foreign_release_barrier(
        &sync,
        local_layout,
        image.usage(),
    )?
    else {
        return Ok(false);
    };
    let vk_barrier = barrier.to_color_image_memory_barrier(image.image());
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    image.inner.sync.begin_dmabuf_render_target_foreign_release()?;
    // SAFETY: `command_buffer` is in recording state and belongs to a live graphics command
    // pool/device. `image` is a bound dmabuf external-memory image retained below until command
    // completion. The image was created through the dmabuf external-memory path, which requires the
    // foreign queue-family extension before this operation can be reached. The barrier is produced
    // by the render-target dmabuf foreign release planner, which validates color-attachment usage,
    // current local COLOR_ATTACHMENT_OPTIMAL layout, and a local source queue family before
    // selecting the local -> FOREIGN ownership transfer and COLOR_ATTACHMENT_OPTIMAL -> GENERAL
    // layout transition.
    unsafe {
        command_buffer
            .command_pool
            .logical_device
            .handle()
            .cmd_pipeline_barrier(
                command_buffer.handle,
                barrier.src_stage,
                barrier.dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk_barrier],
            )
    };
    command_buffer
        .pending_image_layouts
        .push(VulkanPendingImageLayout {
            image: image.image(),
            resource: Arc::clone(&image.inner),
            new_layout: barrier.new_layout,
        });
    command_buffer.pending_image_syncs.push(VulkanPendingImageSync {
        resource: Arc::clone(&image.inner),
        operation: VulkanPendingImageSyncOperation::DmabufRenderTargetForeignRelease,
    });
    command_buffer.referenced_images.push(Arc::clone(&image.inner));

    Ok(true)
}

fn ensure_dmabuf_external_image(image: &VulkanOwnedImage) -> Result<(), VulkanError> {
    if image.external_memory_handle_type() == Some(VulkanExternalMemoryHandleType::Dmabuf) {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedOperation("dmabuf external memory"))
    }
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
    ensure_command_buffer_recording(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_image_locally_usable(image)?;
    ensure_command_buffer_buffer_device(command_buffer, buffer)?;
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
    ensure_command_buffer_recording(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_image_locally_usable(image)?;
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
    ensure_command_buffer_recording(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_image_locally_usable_for_recorded_color_attachment_work(command_buffer, image)?;
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

fn record_color_attachment_clear_rects(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
    render_pass: &VulkanRenderPass,
    framebuffer: &VulkanFramebuffer,
    color: vk::ClearColorValue,
    clear_areas: &[vk::Rect2D],
) -> Result<(), VulkanError> {
    ensure_command_buffer_recording(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_image_locally_usable_for_recorded_color_attachment_work(command_buffer, image)?;
    if clear_areas.is_empty() {
        return Ok(());
    }
    if !image.usage().contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
    }

    let image_layout = command_buffer
        .pending_layout_for(image)?
        .unwrap_or(image.layout()?);
    if image_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
        return Err(VulkanError::UnsupportedOperation("image color attachment layout"));
    }
    for clear_area in clear_areas {
        validate_color_attachment_area(image, *clear_area, "color attachment clear area")?;
    }

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
        .render_area(render_area);
    let clear_attachments = [vk::ClearAttachment::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .color_attachment(0)
        .clear_value(vk::ClearValue { color })];
    let clear_rects = clear_areas
        .iter()
        .copied()
        .map(|rect| vk::ClearRect {
            rect,
            base_array_layer: 0,
            layer_count: 1,
        })
        .collect::<Vec<_>>();
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    // SAFETY: The image is in COLOR_ATTACHMENT_OPTIMAL, the framebuffer and render pass were
    // created for this image view and device, all clear rectangles are validated to be non-empty and
    // inside the framebuffer, and the command buffer is host synchronized by the command-pool lock.
    unsafe {
        let device = command_buffer.command_pool.logical_device.handle();
        device.cmd_begin_render_pass(command_buffer.handle, &begin_info, vk::SubpassContents::INLINE);
        device.cmd_clear_attachments(command_buffer.handle, &clear_attachments, &clear_rects);
        device.cmd_end_render_pass(command_buffer.handle);
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
    ensure_command_buffer_recording(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, target)?;
    ensure_image_locally_usable_for_recorded_color_attachment_work(command_buffer, target)?;
    ensure_image_locally_usable(descriptor_set.sampled_image().image())?;
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
    validate_sampled_texture_draw_area(target, draw_constants.scissor_area)?;
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
    let scissors = [draw_constants.scissor_area];
    let descriptor_sets = [descriptor_set.handle()];
    let draw_constant_bytes = sampled_texture_draw_constant_bytes(draw_constants);
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    // SAFETY: All bound objects were created from the same logical device by private constructors.
    // `target` is in COLOR_ATTACHMENT_OPTIMAL for the duration of the render pass, the framebuffer
    // uses the same render pass as `pipeline`, dynamic viewport/scissor are set before drawing, and
    // `descriptor_set` retains the sampled image resources it references, and the pipeline layout
    // contains a fragment-stage push-constant range covering the 32 bytes written here. The command
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

fn record_solid_color_draw(
    command_buffer: &mut VulkanCommandBuffer,
    target: &VulkanOwnedImage,
    pipeline: &VulkanSolidColorGraphicsPipeline,
    framebuffer: &VulkanFramebuffer,
    draw_constants: VulkanSolidColorDrawConstants,
) -> Result<(), VulkanError> {
    ensure_command_buffer_recording(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, target)?;
    ensure_image_locally_usable_for_recorded_color_attachment_work(command_buffer, target)?;
    if !target.usage().contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
    }
    validate_solid_color_draw_inputs(&command_buffer.command_pool.logical_device, target, pipeline)?;
    validate_color_attachment_area(target, draw_constants.draw_area, "solid color draw area")?;
    validate_color_attachment_area(target, draw_constants.scissor_area, "solid color draw area")?;
    validate_solid_color_draw_constants(draw_constants)?;

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
    let begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(pipeline.render_pass().handle())
        .framebuffer(framebuffer.handle)
        .render_area(render_area);
    let viewport = [vk::Viewport {
        x: draw_constants.draw_area.offset.x as f32,
        y: draw_constants.draw_area.offset.y as f32,
        width: draw_constants.draw_area.extent.width as f32,
        height: draw_constants.draw_area.extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [draw_constants.scissor_area];
    let draw_constant_bytes = solid_color_draw_constant_bytes(draw_constants);
    let _pool_guard = command_buffer.command_pool.lock_host_access()?;

    // SAFETY: All bound objects were created from the same logical device by private constructors.
    // `target` is in COLOR_ATTACHMENT_OPTIMAL for the duration of the render pass, the framebuffer
    // uses the same render pass as `pipeline`, dynamic viewport/scissor are set before drawing, and
    // the pipeline layout contains a fragment-stage push-constant range covering the 16 bytes
    // written here. The command buffer is host synchronized by the command-pool lock.
    unsafe {
        let device = command_buffer.command_pool.logical_device.handle();
        device.cmd_begin_render_pass(command_buffer.handle, &begin_info, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(
            command_buffer.handle,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline().handle(),
        );
        device.cmd_set_viewport(command_buffer.handle, 0, &viewport);
        device.cmd_set_scissor(command_buffer.handle, 0, &scissors);
        device.cmd_push_constants(
            command_buffer.handle,
            pipeline.layout().handle(),
            vk::ShaderStageFlags::FRAGMENT,
            0,
            &draw_constant_bytes,
        );
        device.cmd_draw(command_buffer.handle, 3, 1, 0, 0);
        device.cmd_end_render_pass(command_buffer.handle);
    }

    command_buffer.referenced_images.push(Arc::clone(&target.inner));

    Ok(())
}

fn validate_solid_color_draw_inputs(
    logical_device: &VulkanLogicalDevice,
    target: &VulkanOwnedImage,
    pipeline: &VulkanSolidColorGraphicsPipeline,
) -> Result<(), VulkanError> {
    if target.format() != pipeline.color_format() {
        return Err(VulkanError::UnsupportedOperation("solid pipeline format"));
    }
    if !logical_device.is_same_device(&target.inner.logical_device)
        || !logical_device.is_same_device(pipeline.render_pass().logical_device())
        || !logical_device.is_same_device(pipeline.layout().logical_device())
        || !logical_device.is_same_device(pipeline.pipeline().logical_device())
    {
        return Err(VulkanError::UnsupportedOperation("solid color draw device"));
    }

    Ok(())
}

fn validate_solid_color_draw_constants(
    draw_constants: VulkanSolidColorDrawConstants,
) -> Result<(), VulkanError> {
    if !draw_constants.color.iter().all(|component| component.is_finite())
        || !(0.0..=1.0).contains(&draw_constants.color[3])
    {
        return Err(VulkanError::UnsupportedOperation("solid color draw constants"));
    }

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
    validate_color_attachment_area(target, draw_area, "sampled texture draw area")
}

fn validate_color_attachment_area(
    target: &VulkanOwnedImage,
    area: vk::Rect2D,
    reason: &'static str,
) -> Result<(), VulkanError> {
    if area.extent.width == 0 || area.extent.height == 0 || area.offset.x < 0 || area.offset.y < 0 {
        return Err(VulkanError::UnsupportedOperation(reason));
    }

    let x_end = u32::try_from(area.offset.x)
        .ok()
        .and_then(|x| x.checked_add(area.extent.width))
        .ok_or(VulkanError::UnsupportedOperation(reason))?;
    let y_end = u32::try_from(area.offset.y)
        .ok()
        .and_then(|y| y.checked_add(area.extent.height))
        .ok_or(VulkanError::UnsupportedOperation(reason))?;

    if x_end > target.extent().width || y_end > target.extent().height {
        return Err(VulkanError::UnsupportedOperation(reason));
    }

    Ok(())
}

fn validate_sampled_texture_draw_constants(
    draw_constants: VulkanSampledTextureDrawConstants,
) -> Result<(), VulkanError> {
    const UV_RECT_EPSILON: f32 = 0.000_001;
    let uv_range = -UV_RECT_EPSILON..=1.0 + UV_RECT_EPSILON;
    let [u_origin, v_origin] = draw_constants.uv_origin;
    let [u_x_axis, v_x_axis] = draw_constants.uv_x_axis;
    let [u_y_axis, v_y_axis] = draw_constants.uv_y_axis;
    let uv_corners = [
        [u_origin, v_origin],
        [u_origin + u_x_axis, v_origin + v_x_axis],
        [u_origin + u_y_axis, v_origin + v_y_axis],
        [u_origin + u_x_axis + u_y_axis, v_origin + v_x_axis + v_y_axis],
    ];
    let determinant = u_x_axis * v_y_axis - u_y_axis * v_x_axis;

    if !draw_constants
        .uv_origin
        .iter()
        .all(|component| component.is_finite())
        || !draw_constants
            .uv_x_axis
            .iter()
            .all(|component| component.is_finite())
        || !draw_constants
            .uv_y_axis
            .iter()
            .all(|component| component.is_finite())
        || !uv_corners
            .iter()
            .flatten()
            .all(|component| uv_range.contains(component))
        || !draw_constants.alpha.is_finite()
        || determinant == 0.0
        || !(0.0..=1.0).contains(&draw_constants.alpha)
    {
        return Err(VulkanError::UnsupportedOperation(
            "sampled texture draw constants",
        ));
    }

    Ok(())
}

fn sampled_texture_draw_constant_bytes(draw_constants: VulkanSampledTextureDrawConstants) -> [u8; 32] {
    let [u_origin, v_origin] = draw_constants.uv_origin;
    let [u_x_axis, v_x_axis] = draw_constants.uv_x_axis;
    let [u_y_axis, v_y_axis] = draw_constants.uv_y_axis;
    let force_opaque_alpha = if draw_constants.force_opaque_alpha {
        1.0f32
    } else {
        0.0
    };
    let mut bytes = [0; 32];
    bytes[0..4].copy_from_slice(&u_origin.to_ne_bytes());
    bytes[4..8].copy_from_slice(&v_origin.to_ne_bytes());
    bytes[8..12].copy_from_slice(&u_x_axis.to_ne_bytes());
    bytes[12..16].copy_from_slice(&v_x_axis.to_ne_bytes());
    bytes[16..20].copy_from_slice(&u_y_axis.to_ne_bytes());
    bytes[20..24].copy_from_slice(&v_y_axis.to_ne_bytes());
    bytes[24..28].copy_from_slice(&draw_constants.alpha.to_ne_bytes());
    bytes[28..32].copy_from_slice(&force_opaque_alpha.to_ne_bytes());
    bytes
}

fn solid_color_draw_constant_bytes(draw_constants: VulkanSolidColorDrawConstants) -> [u8; 16] {
    let [r, g, b, a] = draw_constants.color;
    let mut bytes = [0; 16];
    bytes[0..4].copy_from_slice(&r.to_ne_bytes());
    bytes[4..8].copy_from_slice(&g.to_ne_bytes());
    bytes[8..12].copy_from_slice(&b.to_ne_bytes());
    bytes[12..16].copy_from_slice(&a.to_ne_bytes());
    bytes
}

fn synchronize_color_attachment_load(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
) -> Result<(), VulkanError> {
    ensure_command_buffer_recording(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_image_locally_usable_for_recorded_color_attachment_work(command_buffer, image)?;
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

fn copy_image_region_to_buffer(
    command_buffer: &mut VulkanCommandBuffer,
    image: &VulkanOwnedImage,
    buffer: &VulkanHostVisibleBuffer,
    image_offset: vk::Offset3D,
    extent: vk::Extent3D,
) -> Result<(), VulkanError> {
    ensure_command_buffer_recording(command_buffer)?;
    ensure_command_buffer_image_device(command_buffer, image)?;
    ensure_image_locally_usable(image)?;
    ensure_command_buffer_buffer_device(command_buffer, buffer)?;
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
    if image_offset.x < 0 || image_offset.y < 0 || image_offset.z < 0 {
        return Err(VulkanError::UnsupportedOperation("image copy offset"));
    }
    let image_extent = image.extent();
    let copy_width = image_offset.x as u32;
    let copy_height = image_offset.y as u32;
    let copy_depth = image_offset.z as u32;
    if copy_width
        .checked_add(extent.width)
        .is_none_or(|right| right > image_extent.width)
        || copy_height
            .checked_add(extent.height)
            .is_none_or(|bottom| bottom > image_extent.height)
        || copy_depth
            .checked_add(extent.depth)
            .is_none_or(|back| back > image_extent.depth)
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
        .image_offset(image_offset)
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

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VulkanExternalImageBarrier {
    pub(super) src_stage: vk::PipelineStageFlags,
    pub(super) dst_stage: vk::PipelineStageFlags,
    pub(super) src_access: vk::AccessFlags,
    pub(super) dst_access: vk::AccessFlags,
    pub(super) old_layout: vk::ImageLayout,
    pub(super) new_layout: vk::ImageLayout,
    pub(super) src_queue_family_index: u32,
    pub(super) dst_queue_family_index: u32,
}

#[allow(dead_code)]
impl VulkanExternalImageBarrier {
    pub(super) fn to_color_image_memory_barrier(self, image: vk::Image) -> vk::ImageMemoryBarrier<'static> {
        vk::ImageMemoryBarrier::default()
            .src_access_mask(self.src_access)
            .dst_access_mask(self.dst_access)
            .old_layout(self.old_layout)
            .new_layout(self.new_layout)
            .src_queue_family_index(self.src_queue_family_index)
            .dst_queue_family_index(self.dst_queue_family_index)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
    }
}

#[allow(dead_code)]
pub(super) fn sampled_dmabuf_foreign_acquire_barrier(
    external_layout: vk::ImageLayout,
    graphics_queue_family: u32,
    usage: vk::ImageUsageFlags,
) -> Result<VulkanExternalImageBarrier, VulkanError> {
    if !usage.contains(vk::ImageUsageFlags::SAMPLED) {
        return Err(VulkanError::UnsupportedOperation("image sampled usage"));
    }
    if !is_local_queue_family_index(graphics_queue_family) {
        return Err(VulkanError::UnsupportedOperation("dmabuf queue family"));
    }
    if external_layout != vk::ImageLayout::GENERAL {
        return Err(VulkanError::UnsupportedOperation("dmabuf external layout"));
    }

    Ok(VulkanExternalImageBarrier {
        src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
        dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
        src_access: vk::AccessFlags::empty(),
        dst_access: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::GENERAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_FOREIGN_EXT,
        dst_queue_family_index: graphics_queue_family,
    })
}

#[allow(dead_code)]
pub(super) fn sampled_dmabuf_foreign_release_barrier(
    graphics_queue_family: u32,
    usage: vk::ImageUsageFlags,
) -> Result<VulkanExternalImageBarrier, VulkanError> {
    if !usage.contains(vk::ImageUsageFlags::SAMPLED) {
        return Err(VulkanError::UnsupportedOperation("image sampled usage"));
    }
    if !is_local_queue_family_index(graphics_queue_family) {
        return Err(VulkanError::UnsupportedOperation("dmabuf queue family"));
    }

    Ok(VulkanExternalImageBarrier {
        src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
        dst_stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        src_access: vk::AccessFlags::SHADER_READ,
        dst_access: vk::AccessFlags::empty(),
        old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        new_layout: vk::ImageLayout::GENERAL,
        src_queue_family_index: graphics_queue_family,
        dst_queue_family_index: vk::QUEUE_FAMILY_FOREIGN_EXT,
    })
}

#[allow(dead_code)]
pub(super) fn dmabuf_render_target_foreign_acquire_barrier(
    external_layout: vk::ImageLayout,
    graphics_queue_family: u32,
    usage: vk::ImageUsageFlags,
) -> Result<VulkanExternalImageBarrier, VulkanError> {
    if !usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
    }
    if !is_local_queue_family_index(graphics_queue_family) {
        return Err(VulkanError::UnsupportedOperation("dmabuf queue family"));
    }
    if !matches!(
        external_layout,
        vk::ImageLayout::UNDEFINED | vk::ImageLayout::GENERAL
    ) {
        return Err(VulkanError::UnsupportedOperation("dmabuf external layout"));
    }

    Ok(VulkanExternalImageBarrier {
        src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
        dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access: vk::AccessFlags::empty(),
        dst_access: vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        old_layout: external_layout,
        new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_FOREIGN_EXT,
        dst_queue_family_index: graphics_queue_family,
    })
}

#[allow(dead_code)]
pub(super) fn dmabuf_render_target_foreign_release_barrier(
    graphics_queue_family: u32,
    usage: vk::ImageUsageFlags,
) -> Result<VulkanExternalImageBarrier, VulkanError> {
    if !usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        return Err(VulkanError::UnsupportedOperation("image color attachment usage"));
    }
    if !is_local_queue_family_index(graphics_queue_family) {
        return Err(VulkanError::UnsupportedOperation("dmabuf queue family"));
    }

    Ok(VulkanExternalImageBarrier {
        src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        src_access: vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_access: vk::AccessFlags::empty(),
        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        new_layout: vk::ImageLayout::GENERAL,
        src_queue_family_index: graphics_queue_family,
        dst_queue_family_index: vk::QUEUE_FAMILY_FOREIGN_EXT,
    })
}

#[allow(dead_code)]
pub(super) fn plan_sampled_dmabuf_foreign_acquire_barrier(
    sync: &VulkanImageSyncState,
    graphics_queue_family: u32,
    usage: vk::ImageUsageFlags,
) -> Result<Option<VulkanExternalImageBarrier>, VulkanError> {
    match (sync.external_ownership(), sync.external_acquire_pending()) {
        (VulkanExternalImageOwnership::ForeignKnownGeneral, true) => {
            let external_layout = sync
                .known_foreign_layout()
                .ok_or(VulkanError::UnsupportedOperation("dmabuf external layout"))?;
            sampled_dmabuf_foreign_acquire_barrier(external_layout, graphics_queue_family, usage).map(Some)
        }
        (VulkanExternalImageOwnership::None, false) | (VulkanExternalImageOwnership::Local, false) => {
            Ok(None)
        }
        (VulkanExternalImageOwnership::ForeignUnknown, _)
        | (VulkanExternalImageOwnership::ForeignKnownGeneral, false)
        | (VulkanExternalImageOwnership::AcquirePending, _)
        | (VulkanExternalImageOwnership::None, true)
        | (VulkanExternalImageOwnership::Local, true)
        | (VulkanExternalImageOwnership::ReleasePending, _) => {
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        }
    }
}

#[allow(dead_code)]
pub(super) fn plan_sampled_dmabuf_foreign_release_barrier(
    sync: &VulkanImageSyncState,
    local_layout: vk::ImageLayout,
    graphics_queue_family: u32,
    usage: vk::ImageUsageFlags,
) -> Result<Option<VulkanExternalImageBarrier>, VulkanError> {
    match (sync.external_ownership(), sync.external_acquire_pending()) {
        (VulkanExternalImageOwnership::Local, true) => {
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        }
        (VulkanExternalImageOwnership::Local, false) => {
            if local_layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
                return Err(VulkanError::UnsupportedOperation("dmabuf local layout"));
            }

            sampled_dmabuf_foreign_release_barrier(graphics_queue_family, usage).map(Some)
        }
        (VulkanExternalImageOwnership::None, false) => Ok(None),
        (VulkanExternalImageOwnership::None, true)
        | (VulkanExternalImageOwnership::ForeignUnknown, _)
        | (VulkanExternalImageOwnership::ForeignKnownGeneral, _)
        | (VulkanExternalImageOwnership::AcquirePending, _)
        | (VulkanExternalImageOwnership::ReleasePending, _) => {
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        }
    }
}

#[allow(dead_code)]
pub(super) fn plan_dmabuf_render_target_foreign_acquire_barrier(
    sync: &VulkanImageSyncState,
    graphics_queue_family: u32,
    usage: vk::ImageUsageFlags,
    preserve_contents: bool,
) -> Result<Option<VulkanExternalImageBarrier>, VulkanError> {
    match (sync.external_ownership(), sync.external_acquire_pending()) {
        (VulkanExternalImageOwnership::ForeignKnownGeneral, true) => {
            let external_layout = sync
                .known_foreign_layout()
                .ok_or(VulkanError::UnsupportedOperation("dmabuf external layout"))?;
            dmabuf_render_target_foreign_acquire_barrier(external_layout, graphics_queue_family, usage)
                .map(Some)
        }
        (VulkanExternalImageOwnership::ForeignUnknown, true) if !preserve_contents => {
            dmabuf_render_target_foreign_acquire_barrier(
                vk::ImageLayout::UNDEFINED,
                graphics_queue_family,
                usage,
            )
            .map(Some)
        }
        (VulkanExternalImageOwnership::None, false) | (VulkanExternalImageOwnership::Local, false) => {
            Ok(None)
        }
        (VulkanExternalImageOwnership::ForeignUnknown, true)
        | (VulkanExternalImageOwnership::ForeignUnknown, false)
        | (VulkanExternalImageOwnership::ForeignKnownGeneral, false)
        | (VulkanExternalImageOwnership::AcquirePending, _)
        | (VulkanExternalImageOwnership::None, true)
        | (VulkanExternalImageOwnership::Local, true)
        | (VulkanExternalImageOwnership::ReleasePending, _) => {
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        }
    }
}

#[allow(dead_code)]
pub(super) fn plan_dmabuf_render_target_foreign_release_barrier(
    sync: &VulkanImageSyncState,
    local_layout: vk::ImageLayout,
    graphics_queue_family: u32,
    usage: vk::ImageUsageFlags,
) -> Result<Option<VulkanExternalImageBarrier>, VulkanError> {
    match (sync.external_ownership(), sync.external_acquire_pending()) {
        (VulkanExternalImageOwnership::Local, true) => {
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        }
        (VulkanExternalImageOwnership::Local, false) => {
            if local_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                return Err(VulkanError::UnsupportedOperation("dmabuf local layout"));
            }

            dmabuf_render_target_foreign_release_barrier(graphics_queue_family, usage).map(Some)
        }
        (VulkanExternalImageOwnership::None, false) => Ok(None),
        (VulkanExternalImageOwnership::None, true)
        | (VulkanExternalImageOwnership::ForeignUnknown, _)
        | (VulkanExternalImageOwnership::ForeignKnownGeneral, _)
        | (VulkanExternalImageOwnership::AcquirePending, _)
        | (VulkanExternalImageOwnership::ReleasePending, _) => {
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        }
    }
}

#[allow(dead_code)]
pub(super) fn project_dmabuf_render_target_sync_after_pending_acquire(
    mut sync: VulkanImageSyncState,
) -> Result<VulkanImageSyncState, VulkanError> {
    match (
        sync.external_ownership(),
        sync.external_acquire_pending(),
        sync.external_acquire_kind,
    ) {
        (
            VulkanExternalImageOwnership::AcquirePending,
            true,
            super::image::VulkanExternalImageAcquireKind::RenderTarget,
        ) if sync.render_target_acquire_restore_token.is_some()
            && sync.external_release_kind == super::image::VulkanExternalImageReleaseKind::None
            && sync.render_target_release_restore.is_none() =>
        {
            sync.external_ownership = VulkanExternalImageOwnership::Local;
            sync.external_acquire_pending = false;
            sync.external_acquire_kind = super::image::VulkanExternalImageAcquireKind::None;
            sync.render_target_acquire_restore_token = None;
            Ok(sync)
        }
        (VulkanExternalImageOwnership::None, false, _)
        | (VulkanExternalImageOwnership::Local, false, _)
        | (VulkanExternalImageOwnership::ForeignUnknown, _, _)
        | (VulkanExternalImageOwnership::ForeignKnownGeneral, _, _)
        | (VulkanExternalImageOwnership::AcquirePending, _, _)
        | (VulkanExternalImageOwnership::None, true, _)
        | (VulkanExternalImageOwnership::Local, true, _)
        | (VulkanExternalImageOwnership::ReleasePending, _, _) => {
            Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
        }
    }
}

#[allow(dead_code)]
fn is_local_queue_family_index(queue_family: u32) -> bool {
    queue_family != vk::QUEUE_FAMILY_IGNORED
        && queue_family != vk::QUEUE_FAMILY_EXTERNAL
        && queue_family != vk::QUEUE_FAMILY_FOREIGN_EXT
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
        .anisotropy_enable(false)
        .max_anisotropy(1.0)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
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
            external_memory_handle_type: None,
            layout: Mutex::new(vk::ImageLayout::UNDEFINED),
            sync: VulkanSharedImageSyncState::new(VulkanImageSyncState::default()),
        }),
    })
}

pub(super) fn dmabuf_plane_layouts(import: &VulkanDmabufImportState) -> Vec<vk::SubresourceLayout> {
    import
        .memory
        .planes
        .iter()
        .map(|plane| vk::SubresourceLayout {
            offset: plane.offset.into(),
            size: 0,
            row_pitch: plane.stride.into(),
            array_pitch: 0,
            depth_pitch: 0,
        })
        .collect()
}

fn single_plane_dmabuf_fd(dmabuf: &Dmabuf) -> Result<Option<OwnedFd>, VulkanError> {
    if dmabuf.num_planes() != 1 {
        return Ok(None);
    }

    dmabuf
        .0
        .planes
        .first()
        .ok_or(VulkanError::UnsupportedOperation("dmabuf planes"))?
        .fd
        .as_fd()
        .try_clone_to_owned()
        .map(Some)
        .map_err(|_| VulkanError::UnsupportedOperation("dmabuf fd"))
}

pub(super) fn dmabuf_import_memory_type_bits(image_memory_type_bits: u32, fd_memory_type_bits: u32) -> u32 {
    image_memory_type_bits & fd_memory_type_bits
}

fn allocate_imported_dmabuf_memory(
    logical_device: &VulkanLogicalDevice,
    image: vk::Image,
    allocation_size: vk::DeviceSize,
    memory_type_index: u32,
    fd: OwnedFd,
    dedicated_only: bool,
) -> Result<vk::DeviceMemory, VulkanError> {
    let raw_fd = fd.into_raw_fd();
    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(raw_fd);

    let result = if dedicated_only {
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info)
            .push_next(&mut dedicated_info);

        // SAFETY: `logical_device` is live. `allocation_size`/`memory_type_index` were derived from
        // the image memory requirements and intersected fd memory type bits. The imported raw fd is
        // a duplicated dmabuf fd and remains open for the call. On success Vulkan consumes the fd;
        // on failure ownership remains with us and is reconstructed below for closing. The dedicated
        // allocation pNext references the image whose requirements were queried.
        unsafe { logical_device.handle().allocate_memory(&allocate_info, None) }
    } else {
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info);

        // SAFETY: `logical_device` is live. `allocation_size`/`memory_type_index` were derived from
        // the image memory requirements and intersected fd memory type bits. The imported raw fd is
        // a duplicated dmabuf fd and remains open for the call. On success Vulkan consumes the fd;
        // on failure ownership remains with us and is reconstructed below for closing.
        unsafe { logical_device.handle().allocate_memory(&allocate_info, None) }
    };

    match result {
        Ok(memory) => Ok(memory),
        Err(err) => {
            // SAFETY: `raw_fd` came from `OwnedFd::into_raw_fd` above. Failed Vulkan allocation does
            // not take ownership of the fd, so reconstructing an `OwnedFd` closes it exactly once.
            unsafe { drop(OwnedFd::from_raw_fd(raw_fd)) };
            Err(VulkanError::from(err))
        }
    }
}

/// Logical device owner retained by Vulkan resource wrappers.
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
    queue_family_index: u32,
    host_access: Arc<Mutex<()>>,
}

impl VulkanQueue {
    fn new(handle: vk::Queue, queue_family_index: u32) -> Self {
        Self {
            handle,
            queue_family_index,
            host_access: Arc::new(Mutex::new(())),
        }
    }

    pub(super) fn queue_family_index(&self) -> u32 {
        self.queue_family_index
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
    queue_family_index: u32,
    supports_graphics: bool,
    host_access: Mutex<()>,
}

impl VulkanCommandPool {
    pub(super) fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    fn supports_graphics(&self) -> bool {
        self.supports_graphics
    }

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
    state: VulkanCommandBufferState,
    pending_image_layouts: Vec<VulkanPendingImageLayout>,
    pending_image_syncs: Vec<VulkanPendingImageSync>,
    referenced_buffers: Vec<Arc<VulkanHostVisibleBufferInner>>,
    referenced_images: Vec<Arc<VulkanOwnedImageInner>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VulkanCommandBufferState {
    Initial,
    Recording,
    Executable,
    Submitted,
    /// Queue submission was accepted, but neither fence wait nor queue-idle fallback proved
    /// completion and device loss was not reported. The command buffer handle must not be freed.
    SubmitCompletionUnknown,
    Invalid,
}

fn classify_sampled_dmabuf_release_submit_error(
    state: VulkanCommandBufferState,
    err: VulkanError,
) -> VulkanSampledDmabufForeignReleaseError {
    match state {
        VulkanCommandBufferState::Submitted | VulkanCommandBufferState::SubmitCompletionUnknown => {
            VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(err)
        }
        VulkanCommandBufferState::Initial
        | VulkanCommandBufferState::Recording
        | VulkanCommandBufferState::Executable
        | VulkanCommandBufferState::Invalid => VulkanSampledDmabufForeignReleaseError::RetrySafe(err),
    }
}

fn classify_sampled_dmabuf_acquire_submit_error(
    state: VulkanCommandBufferState,
    err: VulkanError,
) -> VulkanSampledDmabufForeignAcquireError {
    match state {
        VulkanCommandBufferState::Submitted | VulkanCommandBufferState::SubmitCompletionUnknown => {
            VulkanSampledDmabufForeignAcquireError::AcquireSubmitted {
                err,
                sampled_image: None,
            }
        }
        VulkanCommandBufferState::Initial
        | VulkanCommandBufferState::Recording
        | VulkanCommandBufferState::Executable
        | VulkanCommandBufferState::Invalid => VulkanSampledDmabufForeignAcquireError::RetrySafe(err),
    }
}

#[cfg(test)]
pub(super) fn classify_sampled_dmabuf_release_submit_error_for_tests(
    submitted: bool,
    completion_unknown: bool,
    err: VulkanError,
) -> VulkanSampledDmabufForeignReleaseError {
    let state = match (submitted, completion_unknown) {
        (_, true) => VulkanCommandBufferState::SubmitCompletionUnknown,
        (true, false) => VulkanCommandBufferState::Submitted,
        (false, false) => VulkanCommandBufferState::Executable,
    };
    classify_sampled_dmabuf_release_submit_error(state, err)
}

#[cfg(test)]
pub(super) fn classify_sampled_dmabuf_acquire_submit_error_for_tests(
    submitted: bool,
    completion_unknown: bool,
    err: VulkanError,
) -> VulkanSampledDmabufForeignAcquireError {
    let state = match (submitted, completion_unknown) {
        (_, true) => VulkanCommandBufferState::SubmitCompletionUnknown,
        (true, false) => VulkanCommandBufferState::Submitted,
        (false, false) => VulkanCommandBufferState::Executable,
    };
    classify_sampled_dmabuf_acquire_submit_error(state, err)
}

#[allow(dead_code)]
impl VulkanCommandBuffer {
    pub(super) fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub(super) fn queue_family_index(&self) -> u32 {
        self.command_pool.queue_family_index()
    }

    pub(super) fn is_recording(&self) -> bool {
        self.state == VulkanCommandBufferState::Recording
    }

    #[cfg(test)]
    pub(super) fn is_executable_for_tests(&self) -> bool {
        self.state == VulkanCommandBufferState::Executable
    }

    #[cfg(test)]
    pub(super) fn is_submitted_for_tests(&self) -> bool {
        self.state == VulkanCommandBufferState::Submitted
    }

    #[allow(dead_code)]
    pub(super) fn plan_sampled_dmabuf_foreign_acquire_barrier(
        &self,
        sync: &VulkanImageSyncState,
        usage: vk::ImageUsageFlags,
    ) -> Result<Option<VulkanExternalImageBarrier>, VulkanError> {
        plan_sampled_dmabuf_foreign_acquire_barrier(sync, self.queue_family_index(), usage)
    }

    #[allow(dead_code)]
    pub(super) fn plan_sampled_dmabuf_foreign_release_barrier(
        &self,
        sync: &VulkanImageSyncState,
        local_layout: vk::ImageLayout,
        usage: vk::ImageUsageFlags,
    ) -> Result<Option<VulkanExternalImageBarrier>, VulkanError> {
        plan_sampled_dmabuf_foreign_release_barrier(sync, local_layout, self.queue_family_index(), usage)
    }

    #[allow(dead_code)]
    pub(super) fn plan_dmabuf_render_target_foreign_acquire_barrier(
        &self,
        sync: &VulkanImageSyncState,
        usage: vk::ImageUsageFlags,
        preserve_contents: bool,
    ) -> Result<Option<VulkanExternalImageBarrier>, VulkanError> {
        plan_dmabuf_render_target_foreign_acquire_barrier(
            sync,
            self.queue_family_index(),
            usage,
            preserve_contents,
        )
    }

    #[allow(dead_code)]
    pub(super) fn plan_dmabuf_render_target_foreign_release_barrier(
        &self,
        sync: &VulkanImageSyncState,
        local_layout: vk::ImageLayout,
        usage: vk::ImageUsageFlags,
    ) -> Result<Option<VulkanExternalImageBarrier>, VulkanError> {
        plan_dmabuf_render_target_foreign_release_barrier(
            sync,
            local_layout,
            self.queue_family_index(),
            usage,
        )
    }

    fn pending_layout_for(&self, image: &VulkanOwnedImage) -> Result<Option<vk::ImageLayout>, VulkanError> {
        Ok(self
            .pending_image_layouts
            .iter()
            .rev()
            .find(|pending| pending.image == image.image())
            .map(|pending| pending.new_layout))
    }

    fn projected_dmabuf_render_target_local_sync(
        &self,
        image: &VulkanOwnedImage,
    ) -> Result<VulkanImageSyncState, VulkanError> {
        let mut sync = image.sync_state()?;

        for pending in &self.pending_image_syncs {
            if !Arc::ptr_eq(&pending.resource, &image.inner) {
                continue;
            }

            if let VulkanPendingImageSyncOperation::DmabufRenderTargetForeignAcquire { .. } =
                &pending.operation
            {
                sync = project_dmabuf_render_target_sync_after_pending_acquire(sync)?;
            }
        }

        Ok(sync)
    }

    fn projected_dmabuf_render_target_release_sync(
        &self,
        image: &VulkanOwnedImage,
    ) -> Result<VulkanImageSyncState, VulkanError> {
        self.projected_dmabuf_render_target_local_sync(image)
    }

    fn commit_pending_image_layouts_and_syncs(&mut self) -> Result<(), VulkanError> {
        for pending in &self.pending_image_layouts {
            *pending
                .resource
                .layout
                .lock()
                .map_err(|_| host_synchronization_failed())? = pending.new_layout;
        }
        for pending in &self.pending_image_syncs {
            // SAFETY: Pending image sync entries are added only for ownership transfers whose
            // matching Vulkan barrier has been recorded into this command buffer. This method is
            // called only after the command buffer was successfully submitted and its fence waited,
            // so the recorded ownership transfer has completed before host-side sync is advanced.
            unsafe { pending.complete()? };
        }

        self.pending_image_layouts.clear();
        self.pending_image_syncs.clear();
        self.referenced_buffers.clear();
        self.referenced_images.clear();
        Ok(())
    }

    fn abort_pending_image_syncs(&mut self) -> Result<(), VulkanError> {
        while let Some(pending) = self.pending_image_syncs.pop() {
            pending.abort()?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct VulkanPendingImageLayout {
    image: vk::Image,
    resource: Arc<VulkanOwnedImageInner>,
    new_layout: vk::ImageLayout,
}

#[derive(Debug)]
#[allow(dead_code)]
struct VulkanPendingImageSync {
    resource: Arc<VulkanOwnedImageInner>,
    operation: VulkanPendingImageSyncOperation,
}

impl VulkanPendingImageSync {
    fn abort(self) -> Result<(), VulkanError> {
        match self.operation {
            VulkanPendingImageSyncOperation::SampledDmabufForeignAcquire => {
                self.resource.sync.abort_sampled_dmabuf_foreign_acquire()
            }
            VulkanPendingImageSyncOperation::DmabufRenderTargetForeignAcquire { restore } => self
                .resource
                .sync
                .abort_dmabuf_render_target_foreign_acquire(restore),
            VulkanPendingImageSyncOperation::DmabufRenderTargetForeignRelease => {
                self.resource.sync.abort_dmabuf_render_target_foreign_release()
            }
            VulkanPendingImageSyncOperation::SampledDmabufForeignRelease => {
                self.resource.sync.abort_sampled_dmabuf_foreign_release()
            }
        }
    }

    /// Complete a pending image sync transition after the matching Vulkan operation completed.
    ///
    /// # Safety
    ///
    /// The caller must ensure the Vulkan queue-family ownership transfer and layout transition that
    /// caused this pending sync operation has been submitted and completed before calling this.
    unsafe fn complete(&self) -> Result<(), VulkanError> {
        match &self.operation {
            VulkanPendingImageSyncOperation::SampledDmabufForeignAcquire => {
                // SAFETY: Upheld by this method's caller.
                unsafe { self.resource.sync.complete_sampled_dmabuf_foreign_acquire() }
            }
            VulkanPendingImageSyncOperation::DmabufRenderTargetForeignAcquire { .. } => {
                // SAFETY: Upheld by this method's caller.
                unsafe { self.resource.sync.complete_dmabuf_render_target_foreign_acquire() }
            }
            VulkanPendingImageSyncOperation::DmabufRenderTargetForeignRelease => {
                // SAFETY: Upheld by this method's caller.
                unsafe { self.resource.sync.complete_dmabuf_render_target_foreign_release() }
            }
            VulkanPendingImageSyncOperation::SampledDmabufForeignRelease => {
                // SAFETY: Upheld by this method's caller.
                unsafe { self.resource.sync.complete_sampled_dmabuf_foreign_release() }
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum VulkanPendingImageSyncOperation {
    SampledDmabufForeignAcquire,
    DmabufRenderTargetForeignAcquire {
        restore: VulkanDmabufRenderTargetAcquireRestore,
    },
    DmabufRenderTargetForeignRelease,
    SampledDmabufForeignRelease,
}

impl Drop for VulkanCommandBuffer {
    fn drop(&mut self) {
        if self.state == VulkanCommandBufferState::SubmitCompletionUnknown {
            // The queue may still reference this command buffer and Vulkan did not report device
            // loss. Leaking the command buffer handle is safer than freeing a pending command
            // buffer. Keep the command pool and referenced resources alive as well, because the
            // pending submission may still access them.
            std::mem::forget(Arc::clone(&self.command_pool));
            std::mem::forget(std::mem::take(&mut self.pending_image_layouts));
            std::mem::forget(std::mem::take(&mut self.pending_image_syncs));
            std::mem::forget(std::mem::take(&mut self.referenced_buffers));
            std::mem::forget(std::mem::take(&mut self.referenced_images));
            return;
        }

        if self.state != VulkanCommandBufferState::Submitted {
            let _ = self.abort_pending_image_syncs();
        }

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

/// Submitted command buffer whose fence has not necessarily completed yet.
#[allow(dead_code)]
#[derive(Debug)]
pub(super) struct VulkanSubmittedCommandBuffer {
    logical_device: VulkanLogicalDevice,
    command_buffer: Option<VulkanCommandBuffer>,
    fence: Option<vk::Fence>,
    pending_semaphore_payloads: Vec<VulkanPendingSemaphorePayload>,
}

impl VulkanSubmittedCommandBuffer {
    fn new(
        logical_device: VulkanLogicalDevice,
        command_buffer: VulkanCommandBuffer,
        fence: vk::Fence,
        pending_semaphore_payloads: Vec<VulkanPendingSemaphorePayload>,
    ) -> Self {
        Self {
            logical_device,
            command_buffer: Some(command_buffer),
            fence: Some(fence),
            pending_semaphore_payloads,
        }
    }

    fn is_complete(&self) -> Result<bool, VulkanError> {
        let fence = self.fence.ok_or(VulkanError::UnsupportedOperation(
            "submitted command buffer fence",
        ))?;
        // SAFETY: `fence` belongs to `self.logical_device` and remains live until `complete` or this
        // object's drop path destroys or intentionally leaks it.
        unsafe { self.logical_device.handle().get_fence_status(fence) }.map_err(VulkanError::from)
    }

    fn complete(mut self) -> Result<(), VulkanError> {
        self.complete_inner()
    }

    pub(super) fn wait_complete(mut self) -> Result<(), VulkanError> {
        let fence = self.fence.ok_or(VulkanError::UnsupportedOperation(
            "submitted command buffer fence",
        ))?;
        // SAFETY: `fence` belongs to `self.logical_device` and remains live while this submission
        // owner exists. Waiting for all fences with an infinite timeout proves that the queue batch no
        // longer references the retained command buffer, image resources, or semaphores before they
        // are completed and dropped below.
        unsafe {
            self.logical_device
                .handle()
                .wait_for_fences(&[fence], true, u64::MAX)
        }
        .map_err(VulkanError::from)?;
        self.complete_inner()
    }

    fn complete_inner(&mut self) -> Result<(), VulkanError> {
        let command_buffer = self
            .command_buffer
            .as_mut()
            .ok_or(VulkanError::UnsupportedOperation("submitted command buffer"))?;
        command_buffer.commit_pending_image_layouts_and_syncs()?;
        complete_pending_semaphore_payloads(&self.pending_semaphore_payloads)?;
        self.pending_semaphore_payloads.clear();
        if let Some(fence) = self.fence.take() {
            // SAFETY: The fence was created by this logical device for this completed submission.
            // No allocation callbacks were used.
            unsafe { self.logical_device.handle().destroy_fence(fence, None) };
        }
        self.command_buffer.take();
        Ok(())
    }

    fn leak_pending(&mut self) {
        if let Some(mut command_buffer) = self.command_buffer.take() {
            command_buffer.state = VulkanCommandBufferState::SubmitCompletionUnknown;
            std::mem::forget(command_buffer);
        }
        self.fence.take();
        std::mem::forget(std::mem::take(&mut self.pending_semaphore_payloads));
    }
}

impl Drop for VulkanSubmittedCommandBuffer {
    fn drop(&mut self) {
        let Some(fence) = self.fence else {
            return;
        };
        // SAFETY: `fence` belongs to `self.logical_device` and remains live while this owner exists.
        match unsafe { self.logical_device.handle().get_fence_status(fence) } {
            Ok(true) => {
                if self.complete_inner().is_err() {
                    self.leak_pending();
                }
            }
            Ok(false) | Err(_) => self.leak_pending(),
        }
    }
}

/// Vulkan image bound to owned device memory.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VulkanOwnedImage {
    inner: Arc<VulkanOwnedImageInner>,
}

/// Vulkan image created for an external-memory import before memory is bound.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanUnboundImage {
    logical_device: VulkanLogicalDevice,
    image: vk::Image,
    extent: vk::Extent3D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    external_memory_handle_type: Option<VulkanExternalMemoryHandleType>,
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
    external_memory_handle_type: Option<VulkanExternalMemoryHandleType>,
    layout: Mutex<vk::ImageLayout>,
    sync: VulkanSharedImageSyncState,
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

    pub(super) fn external_memory_handle_type(&self) -> Option<VulkanExternalMemoryHandleType> {
        self.inner.external_memory_handle_type
    }

    pub(super) fn layout(&self) -> Result<vk::ImageLayout, VulkanError> {
        self.inner
            .layout
            .lock()
            .map(|layout| *layout)
            .map_err(|_| host_synchronization_failed())
    }

    pub(super) fn sync_state(&self) -> Result<VulkanImageSyncState, VulkanError> {
        self.inner.sync.get()
    }

    #[cfg(test)]
    pub(super) fn set_sync_state(&self, sync: VulkanImageSyncState) -> Result<(), VulkanError> {
        self.inner.sync.set_for_tests(sync)
    }
}

/// Shared image synchronization state used by all handles for one Vulkan image.
#[derive(Debug)]
pub(super) struct VulkanSharedImageSyncState {
    state: Mutex<VulkanImageSyncState>,
}

impl VulkanSharedImageSyncState {
    pub(super) fn new(state: VulkanImageSyncState) -> Self {
        Self {
            state: Mutex::new(state),
        }
    }

    pub(super) fn get(&self) -> Result<VulkanImageSyncState, VulkanError> {
        self.state
            .lock()
            .map(|sync| *sync)
            .map_err(|_| host_synchronization_failed())
    }

    #[allow(dead_code)]
    pub(super) fn begin_sampled_dmabuf_foreign_acquire(&self) -> Result<(), VulkanError> {
        self.state
            .lock()
            .map_err(|_| host_synchronization_failed())?
            .begin_sampled_dmabuf_foreign_acquire()
    }

    pub(super) fn abort_sampled_dmabuf_foreign_acquire(&self) -> Result<(), VulkanError> {
        self.state
            .lock()
            .map_err(|_| host_synchronization_failed())?
            .abort_sampled_dmabuf_foreign_acquire()
    }

    #[allow(dead_code)]
    pub(super) fn begin_dmabuf_render_target_foreign_acquire(
        &self,
        preserve_contents: bool,
    ) -> Result<VulkanDmabufRenderTargetAcquireRestore, VulkanError> {
        self.state
            .lock()
            .map_err(|_| host_synchronization_failed())?
            .begin_dmabuf_render_target_foreign_acquire(preserve_contents)
    }

    #[allow(dead_code)]
    pub(super) fn abort_dmabuf_render_target_foreign_acquire(
        &self,
        restore: VulkanDmabufRenderTargetAcquireRestore,
    ) -> Result<(), VulkanError> {
        self.state
            .lock()
            .map_err(|_| host_synchronization_failed())?
            .abort_dmabuf_render_target_foreign_acquire(restore)
    }

    /// Complete a pending dmabuf render-target foreign acquire after the Vulkan barrier has executed.
    ///
    /// # Safety
    ///
    /// The caller must ensure the matching queue-family ownership transfer and layout transition
    /// from foreign ownership to this renderer's queue has completed before calling this.
    #[allow(dead_code)]
    unsafe fn complete_dmabuf_render_target_foreign_acquire(&self) -> Result<(), VulkanError> {
        // SAFETY: Forwarded from this method's caller.
        unsafe {
            self.state
                .lock()
                .map_err(|_| host_synchronization_failed())?
                .complete_dmabuf_render_target_foreign_acquire()
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn complete_dmabuf_render_target_foreign_acquire_for_tests(
        &self,
    ) -> Result<(), VulkanError> {
        // SAFETY: Forwarded from this test-only method's caller.
        unsafe { self.complete_dmabuf_render_target_foreign_acquire() }
    }

    /// Complete a pending sampled-dmabuf foreign acquire after the Vulkan barrier has executed.
    ///
    /// # Safety
    ///
    /// The caller must ensure the matching queue-family ownership transfer and layout transition
    /// from foreign ownership to this renderer's queue has completed before calling this.
    unsafe fn complete_sampled_dmabuf_foreign_acquire(&self) -> Result<(), VulkanError> {
        // SAFETY: Forwarded from this method's caller.
        unsafe {
            self.state
                .lock()
                .map_err(|_| host_synchronization_failed())?
                .complete_sampled_dmabuf_foreign_acquire()
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn complete_sampled_dmabuf_foreign_acquire_for_tests(&self) -> Result<(), VulkanError> {
        // SAFETY: Forwarded from this test-only method's caller.
        unsafe { self.complete_sampled_dmabuf_foreign_acquire() }
    }

    #[allow(dead_code)]
    pub(super) fn begin_sampled_dmabuf_foreign_release(&self) -> Result<(), VulkanError> {
        self.state
            .lock()
            .map_err(|_| host_synchronization_failed())?
            .begin_sampled_dmabuf_foreign_release()
    }

    pub(super) fn abort_sampled_dmabuf_foreign_release(&self) -> Result<(), VulkanError> {
        self.state
            .lock()
            .map_err(|_| host_synchronization_failed())?
            .abort_sampled_dmabuf_foreign_release()
    }

    #[allow(dead_code)]
    pub(super) fn begin_dmabuf_render_target_foreign_release(&self) -> Result<(), VulkanError> {
        self.state
            .lock()
            .map_err(|_| host_synchronization_failed())?
            .begin_dmabuf_render_target_foreign_release()
    }

    #[allow(dead_code)]
    pub(super) fn abort_dmabuf_render_target_foreign_release(&self) -> Result<(), VulkanError> {
        self.state
            .lock()
            .map_err(|_| host_synchronization_failed())?
            .abort_dmabuf_render_target_foreign_release()
    }

    /// Complete a pending dmabuf render-target foreign release after the Vulkan barrier has executed.
    ///
    /// # Safety
    ///
    /// The caller must ensure the matching queue-family ownership transfer and layout transition
    /// from this renderer's queue to foreign ownership has completed before calling this.
    #[allow(dead_code)]
    unsafe fn complete_dmabuf_render_target_foreign_release(&self) -> Result<(), VulkanError> {
        // SAFETY: Forwarded from this method's caller.
        unsafe {
            self.state
                .lock()
                .map_err(|_| host_synchronization_failed())?
                .complete_dmabuf_render_target_foreign_release()
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn complete_dmabuf_render_target_foreign_release_for_tests(
        &self,
    ) -> Result<(), VulkanError> {
        // SAFETY: Forwarded from this test-only method's caller.
        unsafe { self.complete_dmabuf_render_target_foreign_release() }
    }

    /// Complete a pending sampled-dmabuf foreign release after the Vulkan barrier has executed.
    ///
    /// # Safety
    ///
    /// The caller must ensure the matching queue-family ownership transfer and layout transition
    /// from this renderer's queue to foreign ownership has completed before calling this.
    unsafe fn complete_sampled_dmabuf_foreign_release(&self) -> Result<(), VulkanError> {
        // SAFETY: Forwarded from this method's caller.
        unsafe {
            self.state
                .lock()
                .map_err(|_| host_synchronization_failed())?
                .complete_sampled_dmabuf_foreign_release()
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn complete_sampled_dmabuf_foreign_release_for_tests(&self) -> Result<(), VulkanError> {
        // SAFETY: Forwarded from this test-only method's caller.
        unsafe { self.complete_sampled_dmabuf_foreign_release() }
    }

    #[cfg(test)]
    fn set_for_tests(&self, sync: VulkanImageSyncState) -> Result<(), VulkanError> {
        *self.state.lock().map_err(|_| host_synchronization_failed())? = sync;
        Ok(())
    }
}

#[allow(dead_code)]
impl VulkanUnboundImage {
    pub(super) fn image(&self) -> vk::Image {
        self.image
    }

    pub(super) fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    pub(super) fn format(&self) -> vk::Format {
        self.format
    }

    pub(super) fn usage(&self) -> vk::ImageUsageFlags {
        self.usage
    }

    fn into_bound_image(self, memory: vk::DeviceMemory) -> VulkanOwnedImage {
        self.into_bound_image_with_sync(memory, VulkanImageSyncState::default())
    }

    fn into_bound_image_with_sync(
        self,
        memory: vk::DeviceMemory,
        sync: VulkanImageSyncState,
    ) -> VulkanOwnedImage {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is `ManuallyDrop`, so its fields will not be dropped automatically. Moving
        // the logical device out with `ptr::read` transfers the single owning reference into the
        // bound image owner without incrementing/leaking the device `Arc`. The image handle is also
        // transferred and must no longer be destroyed by `VulkanUnboundImage`.
        let logical_device = unsafe { ptr::read(&this.logical_device) };

        VulkanOwnedImage {
            inner: Arc::new(VulkanOwnedImageInner {
                logical_device,
                image: this.image,
                memory,
                extent: this.extent,
                format: this.format,
                usage: this.usage,
                external_memory_handle_type: this.external_memory_handle_type,
                layout: Mutex::new(vk::ImageLayout::UNDEFINED),
                sync: VulkanSharedImageSyncState::new(sync),
            }),
        }
    }
}

impl Drop for VulkanUnboundImage {
    fn drop(&mut self) {
        // SAFETY: `self.image` was created from `self.logical_device` and is destroyed exactly
        // once by this owner. No memory has been bound through this scaffold type yet, so there is
        // no corresponding device memory to free. `VulkanLogicalDevice` is retained by value, so the
        // device outlives the image. No allocation callbacks are used.
        unsafe { self.logical_device.handle().destroy_image(self.image, None) };
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

    fn logical_device(&self) -> &VulkanLogicalDevice {
        &self.image.logical_device
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

/// Pair of SPIR-V modules compatible with the sampled-texture graphics pipeline.
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
    /// vertex module must not declare non-built-in vertex input attributes, because the renderer
    /// pipeline uses an empty vertex-input state. The fragment module must provide a `main` entry
    /// point with the fragment execution model. Their location interfaces must match, the fragment
    /// module must use descriptor set 0 binding 0 as a single `COMBINED_IMAGE_SAMPLER`. If the
    /// fragment module reads push constants, those reads must fit inside the first 32 bytes provided
    /// by this sampled-texture pipeline layout. Its color output must be compatible with a single
    /// `color_format` color attachment in subpass 0 of the render pass used by the renderer.
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
    0x07230203, 0x00010000, 0x0008000b, 0x00000044, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000013, 0x0000003e, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00030005, 0x00000009, 0x00007675, 0x00060005, 0x0000000a, 0x77617244, 0x736e6f43,
    0x746e6174, 0x00000073, 0x00060006, 0x0000000a, 0x00000000, 0x6f5f7675, 0x69676972, 0x0000006e,
    0x00060006, 0x0000000a, 0x00000001, 0x785f7675, 0x6978615f, 0x00000073, 0x00060006, 0x0000000a,
    0x00000002, 0x795f7675, 0x6978615f, 0x00000073, 0x00050006, 0x0000000a, 0x00000003, 0x68706c61,
    0x00000061, 0x00080006, 0x0000000a, 0x00000004, 0x63726f66, 0x706f5f65, 0x65757161, 0x706c615f,
    0x00006168, 0x00030005, 0x0000000c, 0x00006370, 0x00040005, 0x00000013, 0x76755f76, 0x00000000,
    0x00040005, 0x00000028, 0x6f6c6f63, 0x00000072, 0x00030005, 0x0000002c, 0x00786574, 0x00050005,
    0x0000003e, 0x5f74756f, 0x6f6c6f63, 0x00000072, 0x00030047, 0x0000000a, 0x00000002, 0x00050048,
    0x0000000a, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000a, 0x00000001, 0x00000023,
    0x00000008, 0x00050048, 0x0000000a, 0x00000002, 0x00000023, 0x00000010, 0x00050048, 0x0000000a,
    0x00000003, 0x00000023, 0x00000018, 0x00050048, 0x0000000a, 0x00000004, 0x00000023, 0x0000001c,
    0x00040047, 0x00000013, 0x0000001e, 0x00000000, 0x00040047, 0x0000002c, 0x00000021, 0x00000000,
    0x00040047, 0x0000002c, 0x00000022, 0x00000000, 0x00040047, 0x0000003e, 0x0000001e, 0x00000000,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000002, 0x00040020, 0x00000008, 0x00000007, 0x00000007,
    0x0007001e, 0x0000000a, 0x00000007, 0x00000007, 0x00000007, 0x00000006, 0x00000006, 0x00040020,
    0x0000000b, 0x00000009, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000009, 0x00040015,
    0x0000000d, 0x00000020, 0x00000001, 0x0004002b, 0x0000000d, 0x0000000e, 0x00000000, 0x00040020,
    0x0000000f, 0x00000009, 0x00000007, 0x00040020, 0x00000012, 0x00000001, 0x00000007, 0x0004003b,
    0x00000012, 0x00000013, 0x00000001, 0x00040015, 0x00000014, 0x00000020, 0x00000000, 0x0004002b,
    0x00000014, 0x00000015, 0x00000000, 0x00040020, 0x00000016, 0x00000001, 0x00000006, 0x0004002b,
    0x0000000d, 0x00000019, 0x00000001, 0x0004002b, 0x00000014, 0x0000001e, 0x00000001, 0x0004002b,
    0x0000000d, 0x00000021, 0x00000002, 0x00040017, 0x00000026, 0x00000006, 0x00000004, 0x00040020,
    0x00000027, 0x00000007, 0x00000026, 0x00090019, 0x00000029, 0x00000006, 0x00000001, 0x00000000,
    0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x0003001b, 0x0000002a, 0x00000029, 0x00040020,
    0x0000002b, 0x00000000, 0x0000002a, 0x0004003b, 0x0000002b, 0x0000002c, 0x00000000, 0x0004002b,
    0x0000000d, 0x00000030, 0x00000004, 0x00040020, 0x00000031, 0x00000009, 0x00000006, 0x0004002b,
    0x00000006, 0x00000034, 0x00000000, 0x00020014, 0x00000035, 0x0004002b, 0x00000006, 0x00000039,
    0x3f800000, 0x0004002b, 0x00000014, 0x0000003a, 0x00000003, 0x00040020, 0x0000003b, 0x00000007,
    0x00000006, 0x00040020, 0x0000003d, 0x00000003, 0x00000026, 0x0004003b, 0x0000003d, 0x0000003e,
    0x00000003, 0x0004002b, 0x0000000d, 0x00000040, 0x00000003, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009, 0x00000007,
    0x0004003b, 0x00000027, 0x00000028, 0x00000007, 0x00050041, 0x0000000f, 0x00000010, 0x0000000c,
    0x0000000e, 0x0004003d, 0x00000007, 0x00000011, 0x00000010, 0x00050041, 0x00000016, 0x00000017,
    0x00000013, 0x00000015, 0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x00050041, 0x0000000f,
    0x0000001a, 0x0000000c, 0x00000019, 0x0004003d, 0x00000007, 0x0000001b, 0x0000001a, 0x0005008e,
    0x00000007, 0x0000001c, 0x0000001b, 0x00000018, 0x00050081, 0x00000007, 0x0000001d, 0x00000011,
    0x0000001c, 0x00050041, 0x00000016, 0x0000001f, 0x00000013, 0x0000001e, 0x0004003d, 0x00000006,
    0x00000020, 0x0000001f, 0x00050041, 0x0000000f, 0x00000022, 0x0000000c, 0x00000021, 0x0004003d,
    0x00000007, 0x00000023, 0x00000022, 0x0005008e, 0x00000007, 0x00000024, 0x00000023, 0x00000020,
    0x00050081, 0x00000007, 0x00000025, 0x0000001d, 0x00000024, 0x0003003e, 0x00000009, 0x00000025,
    0x0004003d, 0x0000002a, 0x0000002d, 0x0000002c, 0x0004003d, 0x00000007, 0x0000002e, 0x00000009,
    0x00050057, 0x00000026, 0x0000002f, 0x0000002d, 0x0000002e, 0x0003003e, 0x00000028, 0x0000002f,
    0x00050041, 0x00000031, 0x00000032, 0x0000000c, 0x00000030, 0x0004003d, 0x00000006, 0x00000033,
    0x00000032, 0x000500b7, 0x00000035, 0x00000036, 0x00000033, 0x00000034, 0x000300f7, 0x00000038,
    0x00000000, 0x000400fa, 0x00000036, 0x00000037, 0x00000038, 0x000200f8, 0x00000037, 0x00050041,
    0x0000003b, 0x0000003c, 0x00000028, 0x0000003a, 0x0003003e, 0x0000003c, 0x00000039, 0x000200f9,
    0x00000038, 0x000200f8, 0x00000038, 0x0004003d, 0x00000026, 0x0000003f, 0x00000028, 0x00050041,
    0x00000031, 0x00000041, 0x0000000c, 0x00000040, 0x0004003d, 0x00000006, 0x00000042, 0x00000041,
    0x0005008e, 0x00000026, 0x00000043, 0x0000003f, 0x00000042, 0x0003003e, 0x0000003e, 0x00000043,
    0x000100fd, 0x00010038,
];

const BUILTIN_SOLID_FRAGMENT_SHADER_SPIRV: &[u32] = &[
    0x07230203, 0x00010000, 0x0008000b, 0x00000012, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
    0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00050005, 0x00000009, 0x5f74756f, 0x6f6c6f63, 0x00000072, 0x00060005, 0x0000000a, 0x77617244,
    0x736e6f43, 0x746e6174, 0x00000073, 0x00050006, 0x0000000a, 0x00000000, 0x6f6c6f63, 0x00000072,
    0x00030005, 0x0000000c, 0x00006370, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00030047,
    0x0000000a, 0x00000002, 0x00050048, 0x0000000a, 0x00000000, 0x00000023, 0x00000000, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
    0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b,
    0x00000008, 0x00000009, 0x00000003, 0x0003001e, 0x0000000a, 0x00000007, 0x00040020, 0x0000000b,
    0x00000009, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000009, 0x00040015, 0x0000000d,
    0x00000020, 0x00000001, 0x0004002b, 0x0000000d, 0x0000000e, 0x00000000, 0x00040020, 0x0000000f,
    0x00000009, 0x00000007, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x00050041, 0x0000000f, 0x00000010, 0x0000000c, 0x0000000e, 0x0004003d, 0x00000007,
    0x00000011, 0x00000010, 0x0003003e, 0x00000009, 0x00000011, 0x000100fd, 0x00010038,
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
        // pipeline layout. Cached/shared layouts are owned through `Arc`, so this final drop runs
        // only after there are no remaining safe references or users, satisfying host
        // synchronization for destruction.
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

fn create_solid_color_pipeline_layout(
    logical_device: &VulkanLogicalDevice,
) -> Result<VulkanPipelineLayout, VulkanError> {
    let push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(SOLID_COLOR_DRAW_CONSTANT_SIZE)];
    let create_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    // SAFETY: `logical_device` is a live Vulkan device. The push-constant range is 16 bytes, starts
    // at offset 0, is a multiple of 4, and is exposed to the fragment shader. No descriptor set
    // layouts or allocation callbacks are used.
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
        // descriptor-set layout. Cached/shared layouts are owned through `Arc`, so this final drop
        // runs only after there are no remaining safe references or users, satisfying host
        // synchronization.
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
    descriptor_set_layout: Arc<VulkanDescriptorSetLayout>,
}

#[allow(dead_code)]
impl VulkanSampledTexturePipelineLayout {
    pub(super) fn pipeline_layout(&self) -> &VulkanPipelineLayout {
        &self.pipeline_layout
    }

    pub(super) fn descriptor_set_layout(&self) -> &VulkanDescriptorSetLayout {
        self.descriptor_set_layout.as_ref()
    }
}

/// Render-pass, pipeline-layout, and graphics-pipeline bundle for future sampled-texture draws.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanSampledTextureGraphicsPipeline {
    color_format: vk::Format,
    blend_enabled: bool,
    render_pass: Arc<VulkanRenderPass>,
    layout: Arc<VulkanSampledTexturePipelineLayout>,
    pipeline: VulkanGraphicsPipeline,
}

#[allow(dead_code)]
impl VulkanSampledTextureGraphicsPipeline {
    pub(super) fn color_format(&self) -> vk::Format {
        self.color_format
    }

    pub(super) fn blend_enabled(&self) -> bool {
        self.blend_enabled
    }

    pub(super) fn render_pass(&self) -> &VulkanRenderPass {
        self.render_pass.as_ref()
    }

    #[cfg(test)]
    pub(super) fn render_pass_arc(&self) -> &Arc<VulkanRenderPass> {
        &self.render_pass
    }

    pub(super) fn layout(&self) -> &VulkanSampledTexturePipelineLayout {
        self.layout.as_ref()
    }

    #[cfg(test)]
    pub(super) fn layout_arc(&self) -> &Arc<VulkanSampledTexturePipelineLayout> {
        &self.layout
    }

    pub(super) fn pipeline(&self) -> &VulkanGraphicsPipeline {
        &self.pipeline
    }
}

/// Render-pass, pipeline-layout, and graphics-pipeline bundle for solid-color draws.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanSolidColorGraphicsPipeline {
    color_format: vk::Format,
    blend_enabled: bool,
    render_pass: Arc<VulkanRenderPass>,
    layout: Arc<VulkanPipelineLayout>,
    pipeline: VulkanGraphicsPipeline,
}

#[allow(dead_code)]
impl VulkanSolidColorGraphicsPipeline {
    pub(super) fn color_format(&self) -> vk::Format {
        self.color_format
    }

    pub(super) fn blend_enabled(&self) -> bool {
        self.blend_enabled
    }

    pub(super) fn render_pass(&self) -> &VulkanRenderPass {
        self.render_pass.as_ref()
    }

    #[cfg(test)]
    pub(super) fn render_pass_arc(&self) -> &Arc<VulkanRenderPass> {
        &self.render_pass
    }

    pub(super) fn layout(&self) -> &VulkanPipelineLayout {
        self.layout.as_ref()
    }

    #[cfg(test)]
    pub(super) fn layout_arc(&self) -> &Arc<VulkanPipelineLayout> {
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
    // same device. The push-constant range is 32 bytes, starts at offset 0, is a multiple of 4, and
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
    blend_enabled: bool,
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
        .blend_enable(blend_enabled)
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
    // pass has one color attachment at subpass 0, and the fixed-function state describes a simple
    // triangle-list pipeline with dynamic viewport/scissor and either premultiplied alpha blending
    // or blending disabled for opaque regions. All create-info slices live through the call and no
    // allocation callbacks are used.
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
    if !render_pass.logical_device().is_same_device(view.logical_device()) {
        return Err(VulkanError::UnsupportedOperation("framebuffer device"));
    }

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

/// Uploaded sampled image bundle for the Vulkan texture path.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanSampledImage {
    sampler: VulkanSampler,
    view: VulkanImageView,
    image: VulkanOwnedImage,
}

/// Sampled dmabuf resources prepared before the Vulkan foreign acquire queue submission.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanPreparedSampledDmabufAcquire {
    sampler: VulkanSampler,
    view: VulkanImageView,
    image: VulkanOwnedImage,
    acquire_semaphore: Option<VulkanSyncFileSemaphore>,
    command_buffer: VulkanCommandBuffer,
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

/// Queue family indices discovered during device initialization.
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
