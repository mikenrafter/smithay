//! Native Vulkan renderer.
//!
//! This module provides a provisional, opt-in Vulkan renderer for an explicit [`PhysicalDevice`]. It
//! can upload sampled textures from CPU memory, render to in-memory/offscreen targets, read back
//! those offscreen targets through Smithay's public [`ImportMem`], [`Bind`], [`Offscreen`],
//! [`Renderer`], and [`ExportMem`] traits, and expose validation-stage dmabuf render-target
//! development hooks. It remains intentionally incomplete and does not provide a complete compositor
//! renderer, HDR, colour-management policy, or a fully integrated presentation backend.
//!
//! Downstream compositors must not treat the presence of this module or the `renderer_vulkan`
//! feature as broad Vulkan rendering support. Real enablement must be added incrementally behind
//! explicit capability bits, with tests and stub failure paths before enabling working
//! functionality.
//!
//! Smithay's optional renderer traits are capability surfaces. The CPU-memory/offscreen path is the
//! most complete path. Explicit dmabuf render-target wrappers remain ownership-contract APIs on
//! [`VulkanRenderer::bind_dmabuf_render_target`]. Public [`Bind<Dmabuf>`] is the compositor GBM
//! scanout path: discard/full-repaint acquire, no sampled client import. Generic
//! texture `ExportMem`, `ExportDma`, broad explicit sync, blit/copy, and full presentation remain
//! unsupported until their corresponding capability bits can become true with coverage. Sampled dmabuf
//! import is validation-stage implemented for the normal `ImportDmaWl` path when callers provide
//! commit-local external-state and texture-cache lifecycle evidence; ignored runtime probes drive real
//! linux-dmabuf, drm-syncobj, `wl_surface.commit`, renderer-utils cache import, sampling, and release
//! point signaling. The safe generic `ImportDma` path still fails closed because it receives only a raw
//! [`Dmabuf`] and damage, without Wayland acquire/release points or renderer-utils cache lifecycle.
//! Public sampled `ImportDma` advertisement remains closed until a direct generic import
//! implementation exists. The explicit [`VulkanSampledDmabufImport`] path already tracks local
//! ownership until foreign-GENERAL release.
//!
//! Intended implementation order:
//!
//! 1. Vulkan instance/device initialisation
//! 2. extension discovery
//! 3. format/modifier query
//! 4. `ImportMem`
//! 5. basic texture upload
//! 6. offscreen target
//! 7. clear frame
//! 8. textured quad rendering
//! 9. alpha blending
//! 10. crop/scale/transform (output `Frame` transforms use the same `Transform::transform_rect_in`
//!     compositor-space mapping as pixman; crop/scale are dest/src rectangles)
//! 11. readback test path
//! 12. `ImportDma` (Wayland `ImportDmaWl` is validation-stage; generic `ImportDma` stays
//!     fail-closed. Public foreign-GENERAL import uses [`VulkanSampledDmabufImport`] and blocks
//!     reimport until foreign-GENERAL release.)
//! 13. dmabuf modifier handling
//! 14. export dmabuf
//! 15. KMS presentation path
//! 16. explicit sync
//! 17. blit/copy (same-device framebuffer `Blit` + sampled `ExportMem::copy_texture`)
//! 18. multi-GPU integration
//! 19. colour-capable render targets
//! 20. HDR-ready hooks
//!
//! Every future feature should follow this pattern: capability flag first, test second, stub
//! failure path third, real implementation fourth, enablement last.

use std::{
    collections::HashMap,
    os::fd::{AsFd, BorrowedFd, OwnedFd},
    sync::{Arc, Mutex, Weak},
};

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
use crate::backend::allocator::Buffer as _;
#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
use crate::backend::allocator::dmabuf::DmabufSyncFlags;
#[cfg(all(
    feature = "wayland_frontend",
    feature = "backend_egl",
    feature = "use_system_lib"
))]
use crate::backend::renderer::ImportAll;
#[cfg(feature = "wayland_frontend")]
use crate::backend::renderer::{ImportDmaWl, ImportMemWl};
#[cfg(feature = "wayland_frontend")]
use crate::utils::user_data::UserDataMap;
#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
use crate::wayland::drm_syncobj::DrmSyncPoint;

use crate::{
    backend::vulkan::{Instance, PhysicalDevice},
    backend::{
        allocator::{
            Format, Fourcc, Modifier,
            dmabuf::{Dmabuf, WeakDmabuf},
            format::FormatSet,
            vulkan::VulkanAllocatorDmabufForeignReleaseEvidence,
        },
        renderer::{
            Bind, Blit, BlitFrame, Color32F, ContextId, DebugFlags, ExportMem, ImportDma, ImportMem,
            Offscreen, RenderTargetLifecycle, Renderer, RendererSuper, SurfaceCacheTextureReleaseError,
            Texture, TextureFilter,
            sync::{Fence, Interrupted, SyncPoint},
        },
    },
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
};
use ash::vk;
#[cfg(feature = "wayland_frontend")]
use wayland_server::protocol::{wl_buffer, wl_surface::WlSurface};

mod capabilities;
mod device;
mod error;
pub mod format;
mod image;

#[cfg(feature = "backend_winit")]
pub(crate) use self::error::{HOST_FENCE_WAIT_TIMEOUT_NS, wait_for_fences};

pub use self::{
    capabilities::{
        VulkanColorCapabilities, VulkanDeviceCapabilities, VulkanExportCapabilities,
        VulkanExternalMemoryCapabilities, VulkanExternalSyncCapabilities, VulkanFormatCapabilities,
        VulkanFormatCapabilityRecord, VulkanFormatTiling, VulkanFormatUsage, VulkanImportCapabilities,
        VulkanRendererCapabilities, VulkanRenderingCapabilities, VulkanSyncCapabilities,
    },
    error::VulkanError,
    image::{VulkanFrame, VulkanMemoryMapping, VulkanRenderTarget, VulkanTexture},
};

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufExternalImageState {
    queue_owner: SampledDmabufExternalQueueOwner,
    layout: SampledDmabufExternalImageLayout,
}

#[allow(dead_code)]
impl SampledDmabufExternalImageState {
    fn foreign_general() -> Self {
        Self {
            queue_owner: SampledDmabufExternalQueueOwner::Foreign,
            layout: SampledDmabufExternalImageLayout::General,
        }
    }

    fn is_foreign_general(&self) -> bool {
        *self == Self::foreign_general()
    }

    #[cfg(test)]
    fn external_general_for_tests() -> Self {
        Self {
            queue_owner: SampledDmabufExternalQueueOwner::External,
            layout: SampledDmabufExternalImageLayout::General,
        }
    }

    #[cfg(test)]
    fn foreign_shader_read_only_for_tests() -> Self {
        Self {
            queue_owner: SampledDmabufExternalQueueOwner::Foreign,
            layout: SampledDmabufExternalImageLayout::ShaderReadOnlyOptimal,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampledDmabufExternalQueueOwner {
    Foreign,
    #[cfg(test)]
    External,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampledDmabufExternalImageLayout {
    General,
    #[cfg(test)]
    ShaderReadOnlyOptimal,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufKnownLayoutEvidence {
    dmabuf: WeakDmabuf,
    external_state: SampledDmabufExternalImageState,
}

#[allow(dead_code)]
impl SampledDmabufKnownLayoutEvidence {
    /// Create evidence that a sampled dmabuf is in foreign ownership with GENERAL layout.
    ///
    /// # Safety
    ///
    /// The caller must ensure the producer released the image to `VK_QUEUE_FAMILY_FOREIGN_EXT` in
    /// `VK_IMAGE_LAYOUT_GENERAL` before this renderer acquires it.
    unsafe fn foreign_general(dmabuf: WeakDmabuf) -> Self {
        Self {
            dmabuf,
            external_state: SampledDmabufExternalImageState::foreign_general(),
        }
    }

    #[cfg(test)]
    fn new_for_tests(dmabuf: WeakDmabuf, external_state: SampledDmabufExternalImageState) -> Self {
        Self {
            dmabuf,
            external_state,
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn has_foreign_general_state(&self) -> bool {
        self.external_state.is_foreign_general()
    }
}

/// Ordered readiness gates for public sampled dmabuf [`ImportDma`] advertisement.
///
/// This is intentionally separate from the validation-stage [`ImportDmaWl`] path. The Wayland path
/// may use commit-local evidence from Smithay's renderer-managed buffer wrapper, but public generic
/// [`ImportDma`] has only a raw [`Dmabuf`] and damage. Future work must fill in these contracts in
/// order instead of deriving Vulkan image layout, queue-family ownership, or release lifecycle from
/// linux-dmabuf metadata or syncobj points alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufPublicImportContracts {
    raw_import_capability: bool,
    advertised_formats: bool,
    public_external_state_policy: bool,
    public_import_lifecycle: bool,
    public_import_implementation: bool,
}

impl SampledDmabufPublicImportContracts {
    fn validate(self) -> Result<(), VulkanError> {
        if !self.raw_import_capability {
            return Err(VulkanError::NotPublicAdvertised("sampled dmabuf import"));
        }
        if !self.advertised_formats {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf advertised formats",
            ));
        }
        if !self.public_external_state_policy {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf public external-state policy",
            ));
        }
        if !self.public_import_lifecycle {
            return Err(VulkanError::MissingCapability("sampled dmabuf import lifecycle"));
        }
        if !self.public_import_implementation {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf public import implementation",
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct SampledDmabufWaylandForeignGeneralEvidenceSlot {
    evidence: Mutex<Option<SampledDmabufWaylandForeignGeneralEvidence>>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct SampledDmabufWaylandCommitTokenSlot {
    token: Arc<()>,
}

impl Default for SampledDmabufWaylandCommitTokenSlot {
    fn default() -> Self {
        Self { token: Arc::new(()) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum SampledDmabufWaylandExternalStateUse {
    FirstImport,
    CurrentReacquire,
}

/// Commit-local proof that a Wayland dmabuf is ready for Vulkan sampled import.
///
/// This evidence is stored on Smithay's renderer-managed Wayland buffer wrapper. It is deliberately
/// separate from linux-dmabuf metadata and explicit-sync points: those describe buffer layout data and
/// ordering, but not the Vulkan image layout or queue-family ownership needed by the sampled import
/// acquire barrier.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SampledDmabufWaylandForeignGeneralEvidence {
    dmabuf: WeakDmabuf,
    use_case: SampledDmabufWaylandExternalStateUse,
    external_state: SampledDmabufExternalImageState,
    commit_token: Option<Weak<()>>,
    release_generation: Option<u64>,
}

#[allow(dead_code)]
impl SampledDmabufWaylandForeignGeneralEvidence {
    unsafe fn new(
        dmabuf: WeakDmabuf,
        use_case: SampledDmabufWaylandExternalStateUse,
        commit_token: Weak<()>,
        release_generation: Option<u64>,
    ) -> Self {
        Self {
            dmabuf,
            use_case,
            external_state: SampledDmabufExternalImageState::foreign_general(),
            commit_token: Some(commit_token),
            release_generation,
        }
    }

    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            use_case: SampledDmabufWaylandExternalStateUse::FirstImport,
            external_state: SampledDmabufExternalImageState::foreign_general(),
            commit_token: None,
            release_generation: None,
        }
    }

    #[cfg(test)]
    fn current_reacquire_for_tests(dmabuf: &Dmabuf) -> Self {
        Self::current_reacquire_with_optional_release_generation_for_tests(dmabuf, None)
    }

    #[cfg(test)]
    fn current_reacquire_with_release_generation_for_tests(dmabuf: &Dmabuf, release_generation: u64) -> Self {
        Self::current_reacquire_with_optional_release_generation_for_tests(dmabuf, Some(release_generation))
    }

    #[cfg(test)]
    fn current_reacquire_with_optional_release_generation_for_tests(
        dmabuf: &Dmabuf,
        release_generation: Option<u64>,
    ) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            use_case: SampledDmabufWaylandExternalStateUse::CurrentReacquire,
            external_state: SampledDmabufExternalImageState::foreign_general(),
            commit_token: None,
            release_generation,
        }
    }

    #[cfg(test)]
    fn first_import_with_state_for_tests(
        dmabuf: &Dmabuf,
        external_state: SampledDmabufExternalImageState,
    ) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            use_case: SampledDmabufWaylandExternalStateUse::FirstImport,
            external_state,
            commit_token: None,
            release_generation: None,
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    #[cfg(feature = "wayland_frontend")]
    fn validate_commit_token(&self, user_data: &UserDataMap) -> Result<(), VulkanError> {
        let Some(commit_token) = self.commit_token.as_ref() else {
            return Ok(());
        };
        let Some(stored_token) = commit_token.upgrade() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state commit token",
            ));
        };
        let Some(current_slot) = user_data.get::<SampledDmabufWaylandCommitTokenSlot>() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state commit token",
            ));
        };
        if !Arc::ptr_eq(&stored_token, &current_slot.token) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state commit token",
            ));
        }

        Ok(())
    }

    fn layout_history_for_policy(&self) -> SampledDmabufWaylandLayoutHistory {
        match self.use_case {
            SampledDmabufWaylandExternalStateUse::FirstImport => {
                SampledDmabufWaylandLayoutHistory::NoRendererHistory
            }
            SampledDmabufWaylandExternalStateUse::CurrentReacquire => {
                SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral
            }
        }
    }

    fn validate_release_generation(&self, release_generation: Option<u64>) -> Result<(), VulkanError> {
        match self.use_case {
            SampledDmabufWaylandExternalStateUse::FirstImport => {
                if self.release_generation.is_some() {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland external-state release generation",
                    ));
                }
            }
            SampledDmabufWaylandExternalStateUse::CurrentReacquire => {
                match (self.release_generation, release_generation) {
                    (Some(stored), Some(current)) if stored == current => {}
                    _ => {
                        return Err(VulkanError::UnsupportedOperation(
                            "sampled dmabuf Wayland external-state release generation",
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_for_use(
        &self,
        dmabuf: &Dmabuf,
        use_case: SampledDmabufWaylandExternalStateUse,
    ) -> Result<(), VulkanError> {
        if !self.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state identity",
            ));
        }
        if self.use_case != use_case {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state use",
            ));
        }
        if !self.external_state.is_foreign_general() {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state",
            ));
        }

        Ok(())
    }

    fn first_import_layout(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<SampledDmabufWaylandFirstImportLayoutEvidence, VulkanError> {
        self.validate_for_use(dmabuf, SampledDmabufWaylandExternalStateUse::FirstImport)?;
        Ok(SampledDmabufWaylandFirstImportLayoutEvidence {
            dmabuf: self.dmabuf.clone(),
        })
    }

    fn current_reacquire_layout(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<SampledDmabufWaylandCurrentReacquireLayoutEvidence, VulkanError> {
        self.validate_for_use(dmabuf, SampledDmabufWaylandExternalStateUse::CurrentReacquire)?;
        Ok(SampledDmabufWaylandCurrentReacquireLayoutEvidence {
            dmabuf: self.dmabuf.clone(),
            release_generation: self.release_generation,
        })
    }

    fn first_import_known_foreign_general(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<SampledDmabufKnownLayoutEvidence, VulkanError> {
        self.validate_for_use(dmabuf, SampledDmabufWaylandExternalStateUse::FirstImport)?;
        Ok(unsafe {
            // SAFETY: Creating this storage token is unsafe and requires the caller to prove the
            // current Wayland buffer's dmabuf was released to FOREIGN ownership in GENERAL layout.
            SampledDmabufKnownLayoutEvidence::foreign_general(self.dmabuf.clone())
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum SampledDmabufLayoutEvidence {
    /// A normal Wayland linux-dmabuf commit. Explicit acquire sync may prove producer completion,
    /// but it does not prove Vulkan queue-family ownership or image layout.
    WaylandDmabuf,
    /// Smithay-owned Wayland/Vulkan interop policy evidence for normal linux-dmabuf commits.
    ///
    /// This token is the intended continuation point for the normal `ImportDmaWl` path once this
    /// fork defines and tests its own initial external ownership/layout policy. It must only be
    /// produced by the Wayland/Vulkan policy guard, not by copying another renderer's assumptions.
    SmithayWaylandVulkanPolicy(SampledDmabufWaylandVulkanInteropPolicy),
    /// Caller-provided proof that the producer released the image to `FOREIGN` ownership in
    /// `VK_IMAGE_LAYOUT_GENERAL`.
    KnownForeignGeneral(SampledDmabufKnownLayoutEvidence),
}

/// Opaque evidence that a normal Wayland dmabuf commit satisfies Smithay's Vulkan interop policy.
///
/// This is deliberately private and constructable only through validation-stage Smithay
/// Wayland/Vulkan evidence. The surrounding scaffold models the policy inputs separately:
/// first-import/reacquire external state, queue-family ownership, acquire synchronization, release
/// synchronization, and per-commit texture cache behavior. Keeping this as a separate evidence token
/// prevents future work from treating `linux-dmabuf` protocol metadata or explicit sync alone as a
/// Vulkan layout/ownership proof, and keeps the remaining production evidence sources and release
/// lifecycle hooks explicit.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandVulkanInteropPolicy {
    dmabuf: WeakDmabuf,
    foreign_general: SampledDmabufKnownLayoutEvidence,
}

#[allow(dead_code)]
impl SampledDmabufWaylandVulkanInteropPolicy {
    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Evidence for Smithay's chosen first-import external image state for a Wayland dmabuf.
///
/// This must eventually prove the Vulkan `oldLayout` and source queue family used when importing a
/// Wayland dmabuf that has no renderer-local history yet. It is intentionally separate from
/// acquire-sync evidence: synchronization orders producer completion, but does not identify the
/// image's current Vulkan layout or ownership.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandFirstImportLayoutEvidence {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufWaylandFirstImportLayoutEvidence {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Evidence for Smithay's first-import layout policy for a normal Wayland dmabuf.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandFirstImportLayoutPolicy {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufWaylandFirstImportLayoutPolicy {
    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Evidence for Smithay's reacquire layout policy for a previously imported Wayland dmabuf.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandReacquireLayoutPolicy {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufWaylandReacquireLayoutPolicy {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Evidence that the current Wayland producer returned a previously released dmabuf in the layout
/// and ownership expected by Smithay's Vulkan reacquire path.
///
/// Renderer-local release history proves Smithay's previous release state, not the current producer's
/// return state. This separate token keeps reacquire development-gated until the current-commit
/// contract is modeled explicitly.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandCurrentReacquireLayoutEvidence {
    dmabuf: WeakDmabuf,
    release_generation: Option<u64>,
}

#[allow(dead_code)]
impl SampledDmabufWaylandCurrentReacquireLayoutEvidence {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            release_generation: None,
        }
    }

    #[cfg(test)]
    fn new_with_release_generation_for_tests(dmabuf: &Dmabuf, release_generation: u64) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            release_generation: Some(release_generation),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn validate_release_generation(&self, release_generation: Option<u64>) -> Result<(), VulkanError> {
        match (self.release_generation, release_generation) {
            (Some(stored), Some(current)) if stored == current => Ok(()),
            _ => Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland current reacquire generation",
            )),
        }
    }
}

/// Evidence that this renderer recorded a Wayland sampled dmabuf release to foreign `GENERAL` state.
///
/// This is a test-only development token for runtime loopback probes. It is deliberately narrower
/// than a production producer-return policy: it records that Smithay's Vulkan renderer released the
/// previously imported texture, but callers must still prove the current Wayland acquire point orders
/// the relevant release call site and that no foreign producer changed the image layout/ownership
/// before reacquire.
#[cfg(all(test, feature = "wayland_frontend"))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandRendererForeignGeneralReleaseEvidence {
    dmabuf: WeakDmabuf,
    renderer_context: ContextId<VulkanTexture>,
    release_generation: u64,
}

#[cfg(all(test, feature = "wayland_frontend"))]
impl SampledDmabufWaylandRendererForeignGeneralReleaseEvidence {
    fn new(dmabuf: WeakDmabuf, renderer_context: ContextId<VulkanTexture>, release_generation: u64) -> Self {
        Self {
            dmabuf,
            renderer_context,
            release_generation,
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn is_for_renderer_context(&self, renderer_context: &ContextId<VulkanTexture>) -> bool {
        &self.renderer_context == renderer_context
    }

    fn is_current_release_generation(&self, release_generation: Option<u64>) -> bool {
        release_generation == Some(self.release_generation)
    }
}

/// Evidence for the layout/ownership contract used by a normal Wayland sampled-dmabuf commit.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum SampledDmabufWaylandLayoutPolicy {
    /// First import of a dmabuf with no renderer-local layout history.
    FirstImport(SampledDmabufWaylandFirstImportLayoutPolicy),
    /// Reacquire of a dmabuf this renderer previously released to foreign GENERAL ownership.
    Reacquire(SampledDmabufWaylandReacquireLayoutPolicy),
}

#[allow(dead_code)]
impl SampledDmabufWaylandLayoutPolicy {
    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        match self {
            SampledDmabufWaylandLayoutPolicy::FirstImport(policy) => policy.is_for_dmabuf(dmabuf),
            SampledDmabufWaylandLayoutPolicy::Reacquire(policy) => policy.is_for_dmabuf(dmabuf),
        }
    }
}

/// Evidence for queue-family ownership transfers used by Smithay's Wayland/Vulkan dmabuf policy.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandQueueFamilyPolicy {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufWaylandQueueFamilyPolicy {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Evidence that Wayland acquire sync is mapped into the Vulkan sampled-dmabuf acquire operation.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandAcquireSyncPolicy {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufWaylandAcquireSyncPolicy {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Evidence that a Wayland acquire sync point is tied to a sampled dmabuf identity.
///
/// Normal renderer-managed Wayland buffer evidence is also bound to the buffer wrapper commit token
/// so acquire/release sync evidence cannot be mixed across same-dmabuf commits. Unbound evidence is
/// kept for scaffold-only policy tests.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SampledDmabufAcquireSyncEvidence {
    dmabuf: WeakDmabuf,
    sync: SyncPoint,
    commit_token: Option<Weak<()>>,
}

#[allow(dead_code)]
impl SampledDmabufAcquireSyncEvidence {
    fn new(dmabuf: &Dmabuf, sync: SyncPoint) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            sync,
            commit_token: None,
        }
    }

    fn new_with_commit_token(dmabuf: &Dmabuf, sync: SyncPoint, commit_token: Weak<()>) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            sync,
            commit_token: Some(commit_token),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn sync(&self) -> &SyncPoint {
        &self.sync
    }

    fn same_commit_token_as(
        &self,
        release_evidence: &SampledDmabufReleaseEvidence,
    ) -> Result<(), VulkanError> {
        match (&self.commit_token, &release_evidence.commit_token) {
            (None, None) => Ok(()),
            (Some(acquire_token), Some(release_token)) => {
                let Some(acquire_token) = acquire_token.upgrade() else {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland acquire/release sync commit token",
                    ));
                };
                let Some(release_token) = release_token.upgrade() else {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland acquire/release sync commit token",
                    ));
                };
                if Arc::ptr_eq(&acquire_token, &release_token) {
                    Ok(())
                } else {
                    Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland acquire/release sync commit token",
                    ))
                }
            }
            _ => Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland acquire/release sync commit token",
            )),
        }
    }
}

/// Evidence that Vulkan sampled-dmabuf release is mapped to the Wayland release point.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandReleaseSyncPolicy {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufWaylandReleaseSyncPolicy {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Evidence that texture-cache reuse preserves per-commit Wayland/Vulkan dmabuf contracts.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandTextureCachePolicy {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufWaylandTextureCachePolicy {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Evidence that the import is on the post-retired-release surface-cache call site.
///
/// This is only validation-stage reachability evidence. The normal `import_surface` helper drains
/// retired textures before importing a replacement and exposes a temporary marker for that import
/// call under Smithay's normal serialized renderer-utils surface access. The full texture-cache
/// release lifecycle therefore remains guarded separately.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandTextureCacheReplacementReleaseReachability {
    dmabuf: WeakDmabuf,
    renderer_context: Option<ContextId<VulkanTexture>>,
}

#[allow(dead_code)]
impl SampledDmabufWaylandTextureCacheReplacementReleaseReachability {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            renderer_context: None,
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn validate_renderer_context(
        &self,
        renderer_context: &ContextId<VulkanTexture>,
    ) -> Result<(), VulkanError> {
        match self.renderer_context.as_ref() {
            Some(stored_context) if stored_context != renderer_context => {
                Err(VulkanError::UnsupportedOperation(
                    "sampled dmabuf Wayland import_surface post-retired-release renderer identity",
                ))
            }
            _ => Ok(()),
        }
    }
}

/// Evidence that the renderer surface-cache release hook can release sampled dmabuf textures.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandTextureCacheReleaseHook {
    dmabuf: WeakDmabuf,
    renderer_context: Option<ContextId<VulkanTexture>>,
}

#[allow(dead_code)]
impl SampledDmabufWaylandTextureCacheReleaseHook {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            renderer_context: None,
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn validate_renderer_context(
        &self,
        renderer_context: &ContextId<VulkanTexture>,
    ) -> Result<(), VulkanError> {
        match self.renderer_context.as_ref() {
            Some(stored_context) if stored_context != renderer_context => {
                Err(VulkanError::UnsupportedOperation(
                    "sampled dmabuf Wayland texture-cache release hook renderer identity",
                ))
            }
            _ => Ok(()),
        }
    }
}

/// Evidence that no-next-import/teardown call sites release sampled dmabuf textures before drop.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandTextureCacheReleaseLifecycle {
    dmabuf: WeakDmabuf,
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct SampledDmabufWaylandTextureCacheReleaseLifecycleSlot {
    evidence: Mutex<Option<SampledDmabufWaylandTextureCacheReleaseLifecycleEvidence>>,
}

/// Explicit proof that no-next-import/teardown call sites release sampled dmabuf textures.
///
/// This is a validation-stage marker for compositor-owned lifecycle coverage. It is tied to the
/// Vulkan renderer context because the release call sites must use the same renderer/context that
/// imported the sampled dmabuf texture from the Wayland surface cache, and to the renderer-managed
/// buffer wrapper's commit token so copied lifecycle evidence cannot satisfy another current commit.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SampledDmabufWaylandTextureCacheReleaseLifecycleEvidence {
    dmabuf: WeakDmabuf,
    renderer_context: ContextId<VulkanTexture>,
    commit_token: Weak<()>,
}

#[allow(dead_code)]
impl SampledDmabufWaylandTextureCacheReleaseLifecycle {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

#[allow(dead_code)]
impl SampledDmabufWaylandTextureCacheReleaseLifecycleEvidence {
    unsafe fn new(
        dmabuf: WeakDmabuf,
        renderer_context: ContextId<VulkanTexture>,
        commit_token: Weak<()>,
    ) -> Self {
        Self {
            dmabuf,
            renderer_context,
            commit_token,
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn is_for_renderer_context(&self, renderer_context: &ContextId<VulkanTexture>) -> bool {
        &self.renderer_context == renderer_context
    }

    #[cfg(feature = "wayland_frontend")]
    fn validate_commit_token(&self, user_data: &UserDataMap) -> Result<(), VulkanError> {
        let Some(stored_token) = self.commit_token.upgrade() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland texture-cache release lifecycle commit token",
            ));
        };
        let Some(current_slot) = user_data.get::<SampledDmabufWaylandCommitTokenSlot>() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland texture-cache release lifecycle commit token",
            ));
        };
        if !Arc::ptr_eq(&stored_token, &current_slot.token) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland texture-cache release lifecycle commit token",
            ));
        }

        Ok(())
    }

    fn release_lifecycle(
        &self,
        dmabuf: &Dmabuf,
        renderer_context: &ContextId<VulkanTexture>,
    ) -> Result<SampledDmabufWaylandTextureCacheReleaseLifecycle, VulkanError> {
        if !self.is_for_renderer_context(renderer_context) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland texture-cache release lifecycle renderer identity",
            ));
        }
        if !self.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland texture-cache release lifecycle identity",
            ));
        }
        Ok(SampledDmabufWaylandTextureCacheReleaseLifecycle {
            dmabuf: self.dmabuf.clone(),
        })
    }
}

/// Layout-policy phase available for a normal Wayland sampled-dmabuf commit.
///
/// Wayland explicit sync can order producer completion, but it does not describe Vulkan image
/// layout. Raw renderer-local history starts with this shape, but normal context construction may
/// replace it with the phase carried by attached Smithay-owned external-state evidence. That keeps
/// controlled first-import loopback evidence and controlled current-reacquire release evidence from
/// being reclassified by stale sampled history, while still treating unreleased local ownership as a
/// hard fail-closed state.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampledDmabufWaylandLayoutHistory {
    /// The renderer has no local proof of the dmabuf's previous Vulkan layout/ownership.
    NoRendererHistory,
    /// The renderer locally acquired the dmabuf and has not yet recorded a matching foreign release.
    LocallyAcquired,
    /// The renderer previously released this dmabuf to foreign ownership in GENERAL.
    ReleasedByRendererToForeignGeneral,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampledDmabufWaylandLayoutHistoryRecord {
    history: SampledDmabufWaylandLayoutHistory,
    release_generation: u64,
}

/// Validated normal-path inputs available to Smithay's Wayland/Vulkan sampled-dmabuf policy.
///
/// This deliberately separates protocol/import evidence from the opaque policy evidence tokens. The
/// fields are necessary inputs for producing those tokens, but none of them alone proves Vulkan
/// image layout, queue-family ownership, or release/cache correctness.
#[allow(dead_code)]
struct SampledDmabufWaylandVulkanInteropPolicyContext<'a> {
    dmabuf: &'a Dmabuf,
    import: &'a image::VulkanDmabufImportState,
    acquire_sync: &'a SampledDmabufAcquireSyncEvidence,
    release_evidence: &'a SampledDmabufReleaseEvidence,
    release_ownership: Option<SampledDmabufReleaseOwnershipEvidence>,
    per_commit_texture_import: bool,
    /// The policy phase selected for this current commit. In production context construction this may
    /// come from attached typed external-state evidence rather than raw renderer-local history.
    layout_history: SampledDmabufWaylandLayoutHistory,
    first_import_layout: Option<SampledDmabufWaylandFirstImportLayoutEvidence>,
    first_import_foreign_general: Option<SampledDmabufKnownLayoutEvidence>,
    current_reacquire_layout: Option<SampledDmabufWaylandCurrentReacquireLayoutEvidence>,
    current_reacquire_foreign_general: Option<SampledDmabufKnownLayoutEvidence>,
    texture_cache_replacement_release_reachability:
        Option<SampledDmabufWaylandTextureCacheReplacementReleaseReachability>,
    texture_cache_release_hook: Option<SampledDmabufWaylandTextureCacheReleaseHook>,
    texture_cache_release_lifecycle: Option<SampledDmabufWaylandTextureCacheReleaseLifecycle>,
}

#[allow(dead_code)]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandExternalStateEvidenceSources {
    first_import_layout: Option<SampledDmabufWaylandFirstImportLayoutEvidence>,
    first_import_foreign_general: Option<SampledDmabufKnownLayoutEvidence>,
    current_reacquire_layout: Option<SampledDmabufWaylandCurrentReacquireLayoutEvidence>,
    current_reacquire_foreign_general: Option<SampledDmabufKnownLayoutEvidence>,
}

#[allow(dead_code)]
impl<'a> SampledDmabufWaylandVulkanInteropPolicyContext<'a> {
    fn new(
        dmabuf: &'a Dmabuf,
        import: &'a image::VulkanDmabufImportState,
        acquire_sync: &'a SampledDmabufAcquireSyncEvidence,
        release_evidence: &'a SampledDmabufReleaseEvidence,
        per_commit_texture_import: bool,
        layout_history: SampledDmabufWaylandLayoutHistory,
    ) -> Self {
        Self {
            dmabuf,
            import,
            acquire_sync,
            release_evidence,
            release_ownership: None,
            per_commit_texture_import,
            layout_history,
            first_import_layout: None,
            first_import_foreign_general: None,
            current_reacquire_layout: None,
            current_reacquire_foreign_general: None,
            texture_cache_replacement_release_reachability: None,
            texture_cache_release_hook: None,
            texture_cache_release_lifecycle: None,
        }
    }

    fn with_external_state_sources(
        mut self,
        sources: SampledDmabufWaylandExternalStateEvidenceSources,
    ) -> Self {
        self.first_import_layout = sources.first_import_layout;
        self.first_import_foreign_general = sources.first_import_foreign_general;
        self.current_reacquire_layout = sources.current_reacquire_layout;
        self.current_reacquire_foreign_general = sources.current_reacquire_foreign_general;
        self
    }

    fn with_texture_cache_replacement_release_reachability(
        mut self,
        reachability: SampledDmabufWaylandTextureCacheReplacementReleaseReachability,
    ) -> Self {
        self.texture_cache_replacement_release_reachability = Some(reachability);
        self
    }

    fn with_texture_cache_release_hook(mut self, hook: SampledDmabufWaylandTextureCacheReleaseHook) -> Self {
        self.texture_cache_release_hook = Some(hook);
        self
    }

    fn with_texture_cache_release_lifecycle(
        mut self,
        lifecycle: Option<SampledDmabufWaylandTextureCacheReleaseLifecycle>,
    ) -> Self {
        self.texture_cache_release_lifecycle = lifecycle;
        self
    }

    fn with_release_ownership(mut self, release_ownership: SampledDmabufReleaseOwnershipEvidence) -> Self {
        self.release_ownership = Some(release_ownership);
        self
    }
}

/// Module-private typed context for validation-stage sampled dmabuf imports.
///
/// Raw [`ImportDma`] only receives a [`Dmabuf`] and damage, which is insufficient for Vulkan: it
/// cannot carry acquire sync, release ownership, current external image state, or renderer cache
/// lifecycle. This context is the internal Smithay-shaped contract that gathers those inputs before
/// the Vulkan import core runs. It is intentionally not a public advertisement surface; public raw
/// [`ImportDma`] remains fail-closed until an equivalent contract exists for arbitrary callers.
#[allow(dead_code)]
struct SampledDmabufImportContext<'a> {
    dmabuf: &'a Dmabuf,
    import: image::VulkanDmabufImportState,
    acquire_sync: SampledDmabufAcquireSyncEvidence,
    release_evidence: SampledDmabufReleaseEvidence,
    release_ownership: SampledDmabufReleaseOwnershipEvidence,
    per_commit_texture_import: bool,
    layout_history: SampledDmabufWaylandLayoutHistory,
    external_state_sources: SampledDmabufWaylandExternalStateEvidenceSources,
    texture_cache_replacement_release_reachability:
        SampledDmabufWaylandTextureCacheReplacementReleaseReachability,
    texture_cache_release_hook: SampledDmabufWaylandTextureCacheReleaseHook,
    texture_cache_release_lifecycle: Option<SampledDmabufWaylandTextureCacheReleaseLifecycle>,
}

#[allow(dead_code)]
impl<'a> SampledDmabufImportContext<'a> {
    fn wayland_policy_context(&self) -> SampledDmabufWaylandVulkanInteropPolicyContext<'_> {
        SampledDmabufWaylandVulkanInteropPolicyContext::new(
            self.dmabuf,
            &self.import,
            &self.acquire_sync,
            &self.release_evidence,
            self.per_commit_texture_import,
            self.layout_history,
        )
        .with_external_state_sources(self.external_state_sources.clone())
        .with_texture_cache_replacement_release_reachability(
            self.texture_cache_replacement_release_reachability.clone(),
        )
        .with_texture_cache_release_hook(self.texture_cache_release_hook.clone())
        .with_texture_cache_release_lifecycle(self.texture_cache_release_lifecycle.clone())
        .with_release_ownership(self.release_ownership.clone())
    }
}

/// Validation evidence for Smithay's normal Wayland dmabuf -> Vulkan sampled-image policy.
///
/// Each field names one contract that must be backed by implementation and tests before the normal
/// `ImportDmaWl` path may construct [`SampledDmabufWaylandVulkanInteropPolicy`]. The default value
/// is deliberately all-`None` so production remains fail-closed at the first missing policy step.
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
struct SampledDmabufWaylandVulkanInteropPolicyContracts {
    /// Defines the external ownership and Vulkan image layout used for this commit, either through
    /// first-import policy or renderer-local reacquire history.
    layout: Option<SampledDmabufWaylandLayoutPolicy>,
    /// Proves the policy's layout path is specifically foreign ownership with GENERAL image layout,
    /// matching the validation-stage unsafe import helper's precondition.
    foreign_general: Option<SampledDmabufKnownLayoutEvidence>,
    /// Defines the queue-family ownership transfer to and from this renderer's Vulkan queue.
    queue_family_transfer: Option<SampledDmabufWaylandQueueFamilyPolicy>,
    /// Defines how the Wayland acquire point is converted into a Vulkan wait dependency for the
    /// import/acquire transition.
    acquire_sync: Option<SampledDmabufWaylandAcquireSyncPolicy>,
    /// Defines how Vulkan sampling completion and foreign release are transferred to the Wayland
    /// release point.
    release_sync: Option<SampledDmabufWaylandReleaseSyncPolicy>,
    /// Defines how texture-cache reuse observes per-commit acquire/release obligations and avoids
    /// reusing stale host-side layout evidence.
    texture_cache_reuse: Option<SampledDmabufWaylandTextureCachePolicy>,
    /// Proves the renderer owns the release point before a sampled dmabuf texture can be created.
    release_ownership: Option<SampledDmabufReleaseOwnershipEvidence>,
}

/// Evidence that a Wayland release point exists for a sampled dmabuf.
///
/// This is only protocol-handle evidence, optionally bound to the renderer-managed buffer wrapper's
/// commit token. It does not prove Vulkan has finished sampling, released the image back to foreign
/// ownership, or signaled/satisfied the Wayland release point.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SampledDmabufReleaseEvidence {
    dmabuf: WeakDmabuf,
    commit_token: Option<Weak<()>>,
}

#[allow(dead_code)]
impl SampledDmabufReleaseEvidence {
    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn same_commit_token_as(
        &self,
        release_ownership: &SampledDmabufReleaseOwnershipEvidence,
    ) -> Result<(), VulkanError> {
        match (&self.commit_token, &release_ownership.commit_token) {
            (None, None) => Ok(()),
            (Some(release_token), Some(ownership_token)) => {
                let Some(release_token) = release_token.upgrade() else {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland release ownership commit token",
                    ));
                };
                let Some(ownership_token) = ownership_token.upgrade() else {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland release ownership commit token",
                    ));
                };
                if Arc::ptr_eq(&release_token, &ownership_token) {
                    Ok(())
                } else {
                    Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland release ownership commit token",
                    ))
                }
            }
            _ => Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland release ownership commit token",
            )),
        }
    }
}

/// Evidence that the Wayland release point is still available for renderer ownership transfer.
///
/// This cloneable token is a precondition marker, not the moved release obligation. The actual
/// `Buffer::take_release_point_for_renderer()` transfer happens in the sampled acquire
/// ready-to-submit callback, after validation-stage policy guards and retry-safe Vulkan setup have
/// accepted the import path but before `vkQueueSubmit` can consume the producer acquire dependency.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SampledDmabufReleaseOwnershipEvidence {
    dmabuf: WeakDmabuf,
    commit_token: Option<Weak<()>>,
}

#[allow(dead_code)]
impl SampledDmabufReleaseOwnershipEvidence {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            commit_token: None,
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Renderer-owned sampled dmabuf release obligation.
#[allow(dead_code)]
#[derive(Debug)]
struct SampledDmabufReleaseOwnership {
    dmabuf: WeakDmabuf,
    release: image::VulkanSampledDmabufRelease,
}

#[allow(dead_code)]
#[derive(Debug)]
enum PendingSampledDmabufImportObligation {
    /// Release ownership was moved but acquire submission did not happen; only the Wayland release
    /// point remains to be satisfied.
    ReleaseOnly(SampledDmabufReleaseOwnership),
    /// Acquire side effects may still leave the image locally owned by Vulkan; retry or teardown must
    /// not allow another import of the same dmabuf first.
    AcquiredTexture(VulkanTexture),
    /// A release submission was accepted but completion could not be proven. This is a fail-closed
    /// device/context-invalid state for this dmabuf: the renderer must not retry it as locally owned
    /// or import the same dmabuf again, and it must not signal the Wayland release point because that
    /// could race still-pending GPU work.
    ReleaseCompletionUnknownTexture(VulkanTexture),
}

impl PendingSampledDmabufImportObligation {
    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        match self {
            PendingSampledDmabufImportObligation::ReleaseOnly(release_ownership) => {
                release_ownership.is_for_dmabuf(dmabuf)
            }
            PendingSampledDmabufImportObligation::AcquiredTexture(texture)
            | PendingSampledDmabufImportObligation::ReleaseCompletionUnknownTexture(texture) => {
                texture
                    .sampled_dmabuf
                    .as_ref()
                    .and_then(WeakDmabuf::upgrade)
                    .as_ref()
                    == Some(dmabuf)
            }
        }
    }

    fn may_still_own_image_locally(&self) -> bool {
        matches!(
            self,
            PendingSampledDmabufImportObligation::AcquiredTexture(_)
                | PendingSampledDmabufImportObligation::ReleaseCompletionUnknownTexture(_)
        )
    }
}

#[allow(dead_code)]
impl SampledDmabufReleaseOwnership {
    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            release: image::VulkanSampledDmabufRelease::validation_stage_without_wayland_point(),
        }
    }

    #[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
    fn wayland_syncobj(dmabuf: &Dmabuf, release_point: DrmSyncPoint) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            release: image::VulkanSampledDmabufRelease::wayland_syncobj(release_point),
        }
    }

    fn cache_only(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            release: image::VulkanSampledDmabufRelease::validation_stage_without_wayland_point(),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn into_release(self) -> image::VulkanSampledDmabufRelease {
        self.release
    }

    fn signal_wayland_release_once(&self) -> Result<(), VulkanError> {
        self.release.signal_wayland_release_once()
    }
}

/// Acquire options for binding a foreign dmabuf as a Vulkan render target.
///
/// This is the development-fork API for the Vulkan-specific external target contract. It is explicit
/// because a plain [`Dmabuf`] does not encode Vulkan queue-family ownership, image layout, or acquire
/// synchronization.
#[derive(Debug, Clone, Copy)]
pub struct VulkanDmabufRenderTargetAcquire<'a> {
    /// Preserve previous target contents during acquire.
    ///
    /// If this is `true`, the foreign side must have released the image in
    /// `VK_IMAGE_LAYOUT_GENERAL`. If this is `false`, previous contents are discarded and the
    /// renderer will force an effective target age of zero through the generic bridge.
    pub preserve_contents: bool,
    /// Optional producer-completion dependency for the foreign release into Vulkan ownership.
    pub acquire_sync: Option<&'a SyncPoint>,
}

impl<'a> VulkanDmabufRenderTargetAcquire<'a> {
    /// Acquire for a full repaint, discarding previous contents.
    pub fn discard() -> Self {
        Self {
            preserve_contents: false,
            acquire_sync: None,
        }
    }

    /// Acquire while preserving previous contents after `acquire_sync` is satisfied.
    pub fn preserve(acquire_sync: Option<&'a SyncPoint>) -> Self {
        Self {
            preserve_contents: true,
            acquire_sync,
        }
    }
}

impl Default for VulkanDmabufRenderTargetAcquire<'_> {
    fn default() -> Self {
        Self::discard()
    }
}

/// Acquire options for importing a foreign dmabuf as a sampled Vulkan texture.
///
/// A plain [`Dmabuf`] does not encode Vulkan queue-family ownership or image layout. Generic
/// [`ImportDma`] therefore stays fail-closed. Callers that can prove foreign `GENERAL` release
/// construct [`VulkanSampledDmabufImport`] instead.
#[derive(Debug, Clone, Copy)]
pub struct VulkanSampledDmabufAcquire<'a> {
    /// Optional producer-completion dependency for the foreign release into Vulkan ownership.
    pub acquire_sync: Option<&'a SyncPoint>,
}

impl<'a> VulkanSampledDmabufAcquire<'a> {
    /// Acquire after `acquire_sync` is satisfied, or immediately if it is `None`.
    pub fn with_sync(acquire_sync: Option<&'a SyncPoint>) -> Self {
        Self { acquire_sync }
    }
}

/// Typed wrapper for sampled dmabuf import through Vulkan's explicit foreign-GENERAL contract.
///
/// Creating this wrapper is unsafe because the caller must prove the same external-memory ownership
/// and layout requirements as [`VulkanRenderer::import_dmabuf_texture_with_known_general_layout`].
/// Once constructed, [`VulkanRenderer::import_sampled_dmabuf`] can import without smuggling those
/// requirements through generic [`ImportDma`].
#[derive(Debug)]
pub struct VulkanSampledDmabufImport<'a, 'sync> {
    dmabuf: &'a Dmabuf,
    acquire: VulkanSampledDmabufAcquire<'sync>,
}

impl<'a, 'sync> VulkanSampledDmabufImport<'a, 'sync> {
    /// Creates a sampled-import wrapper for a dmabuf released to foreign ownership in `GENERAL`.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the safety requirements documented on
    /// [`VulkanRenderer::import_dmabuf_texture_with_known_general_layout`] for `dmabuf` and
    /// `acquire.acquire_sync`.
    pub unsafe fn foreign_general(dmabuf: &'a Dmabuf, acquire: VulkanSampledDmabufAcquire<'sync>) -> Self {
        Self { dmabuf, acquire }
    }
}

/// Typed target wrapper for binding a foreign dmabuf through Vulkan's explicit acquire path.
///
/// This wrapper is the safe [`Bind`] target for Vulkan dmabuf render-target development. Creating it
/// is unsafe because the caller must prove the same external-memory ownership, layout, and acquire
/// synchronization requirements as [`VulkanRenderer::bind_dmabuf_render_target`]. Those requirements
/// are not a one-time construction check: they must be true each time the wrapper is bound. Once
/// constructed, generic render paths can bind it without smuggling those requirements through a plain
/// [`Dmabuf`].
#[derive(Debug)]
pub struct VulkanDmabufRenderTarget<'target, 'sync> {
    dmabuf: &'target mut Dmabuf,
    acquire: VulkanDmabufRenderTargetAcquire<'sync>,
}

impl<'target, 'sync> VulkanDmabufRenderTarget<'target, 'sync> {
    /// Creates a Vulkan dmabuf render-target wrapper with explicit acquire options.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the safety requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] for `dmabuf` and `acquire` before every later
    /// safe bind of this wrapper. The same underlying dmabuf storage must not be bound through
    /// another alias while this wrapper, or a framebuffer created from it, is live.
    pub unsafe fn new(dmabuf: &'target mut Dmabuf, acquire: VulkanDmabufRenderTargetAcquire<'sync>) -> Self {
        Self { dmabuf, acquire }
    }

    /// Creates a discard/full-repaint target wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the discard-acquire requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind of this wrapper.
    pub unsafe fn discard(dmabuf: &'target mut Dmabuf) -> Self {
        // SAFETY: Forwarded to this constructor's caller.
        unsafe { Self::new(dmabuf, VulkanDmabufRenderTargetAcquire::discard()) }
    }

    /// Creates a preserve-content target wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the preserve-acquire requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind of this wrapper,
    /// including the foreign release in `VK_IMAGE_LAYOUT_GENERAL`.
    pub unsafe fn preserve(dmabuf: &'target mut Dmabuf, acquire_sync: Option<&'sync SyncPoint>) -> Self {
        // SAFETY: Forwarded to this constructor's caller.
        unsafe { Self::new(dmabuf, VulkanDmabufRenderTargetAcquire::preserve(acquire_sync)) }
    }

    /// Returns whether this acquire preserves previous contents.
    pub fn preserve_contents(&self) -> bool {
        self.acquire.preserve_contents
    }

    /// Returns the acquire options carried by this wrapper.
    pub fn acquire(&self) -> VulkanDmabufRenderTargetAcquire<'sync> {
        self.acquire
    }
}

/// Owned typed target wrapper for binding a foreign dmabuf through Vulkan's explicit acquire path.
///
/// This is useful for generic render paths that receive an owned [`Dmabuf`] handle from an allocator
/// or swapchain before constructing the renderer target. Creating it is unsafe for the same reason as
/// [`VulkanDmabufRenderTarget`]: the caller must prove the Vulkan external-memory ownership, layout,
/// and acquire-synchronization requirements documented on
/// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind.
#[derive(Debug)]
pub struct VulkanOwnedDmabufRenderTarget<'sync> {
    dmabuf: Dmabuf,
    acquire: VulkanDmabufRenderTargetAcquire<'sync>,
}

impl<'sync> VulkanOwnedDmabufRenderTarget<'sync> {
    /// Creates an owned Vulkan dmabuf render-target wrapper with explicit acquire options.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the safety requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] for `dmabuf` and `acquire` before every later
    /// safe bind of this wrapper. The same underlying dmabuf storage must not be bound through
    /// another alias while this wrapper, or a framebuffer created from it, is live.
    pub unsafe fn new(dmabuf: Dmabuf, acquire: VulkanDmabufRenderTargetAcquire<'sync>) -> Self {
        Self { dmabuf, acquire }
    }

    /// Creates an owned discard/full-repaint target wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the discard-acquire requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind of this wrapper.
    pub unsafe fn discard(dmabuf: Dmabuf) -> Self {
        // SAFETY: Forwarded to this constructor's caller.
        unsafe { Self::new(dmabuf, VulkanDmabufRenderTargetAcquire::discard()) }
    }

    /// Creates an owned preserve-content target wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the preserve-acquire requirements documented on
    /// [`VulkanRenderer::bind_dmabuf_render_target`] before every later safe bind of this wrapper,
    /// including the foreign release in `VK_IMAGE_LAYOUT_GENERAL`.
    pub unsafe fn preserve(dmabuf: Dmabuf, acquire_sync: Option<&'sync SyncPoint>) -> Self {
        // SAFETY: Forwarded to this constructor's caller.
        unsafe { Self::new(dmabuf, VulkanDmabufRenderTargetAcquire::preserve(acquire_sync)) }
    }

    /// Returns whether this acquire preserves previous contents.
    pub fn preserve_contents(&self) -> bool {
        self.acquire.preserve_contents
    }

    /// Returns the acquire options carried by this wrapper.
    pub fn acquire(&self) -> VulkanDmabufRenderTargetAcquire<'sync> {
        self.acquire
    }

    /// Returns the underlying dmabuf handle.
    pub fn dmabuf(&self) -> &Dmabuf {
        &self.dmabuf
    }
}

/// Controlled allocator contract for binding a Vulkan allocator-owned dmabuf render target.
///
/// This token ties a mutable dmabuf wrapper to the allocator evidence that released the same Vulkan
/// image to `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`. It is a validation-stage
/// bridge for the normal DRM/GBM swapchain path; it does not advertise generic Vulkan dmabuf target
/// support or prove that the image is safe for KMS scanout.
#[derive(Debug)]
pub(crate) struct VulkanAllocatorDmabufRenderTargetContract<'target> {
    dmabuf: &'target mut Dmabuf,
    foreign_general_release: Option<VulkanAllocatorDmabufForeignReleaseEvidence>,
}

impl<'target> VulkanAllocatorDmabufRenderTargetContract<'target> {
    /// Construct the render-target contract from allocator FOREIGN/GENERAL release evidence.
    pub(crate) fn from_allocator_release(
        dmabuf: &'target mut Dmabuf,
        foreign_general_release: VulkanAllocatorDmabufForeignReleaseEvidence,
    ) -> Result<Self, VulkanError> {
        if !foreign_general_release.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "allocator dmabuf release evidence",
            ));
        }

        Ok(Self {
            dmabuf,
            foreign_general_release: Some(foreign_general_release),
        })
    }

    fn take_release_evidence(&mut self) -> Result<VulkanAllocatorDmabufForeignReleaseEvidence, VulkanError> {
        self.foreign_general_release
            .take()
            .ok_or(VulkanError::UnsupportedOperation(
                "allocator dmabuf render target contract",
            ))
    }

    fn into_parts(
        mut self,
    ) -> Result<(&'target mut Dmabuf, VulkanAllocatorDmabufForeignReleaseEvidence), VulkanError> {
        let evidence = self.take_release_evidence()?;
        Ok((self.dmabuf, evidence))
    }
}

/// Opaque evidence that a Vulkan dmabuf producer released an image for Wayland sampled import.
///
/// This token is produced by Smithay's Vulkan renderer only after releasing an acquired dmabuf render
/// target to `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`. It is tied to the
/// originating [`Dmabuf`] identity and carries the release dependency that a Wayland acquire point must
/// order before `ImportDmaWl` may sample the committed buffer. This is the explicit producer-facing
/// external-state contract for the controlled Vulkan producer path; it does not infer Vulkan layout or
/// queue-family ownership from linux-dmabuf metadata, and it does not public-advertise generic
/// [`ImportDma`] support.
#[derive(Debug)]
pub struct VulkanWaylandDmabufProducerRelease {
    dmabuf: WeakDmabuf,
    acquire_sync: SyncPoint,
    foreign_general: SampledDmabufKnownLayoutEvidence,
}

/// Internal compatibility name for the original Smithay-controlled loopback producer-release token.
///
/// New controlled producer `ImportDmaWl` code should prefer [`VulkanWaylandDmabufProducerRelease`].
pub(crate) type VulkanDmabufLoopbackImportEvidence = VulkanWaylandDmabufProducerRelease;

impl VulkanWaylandDmabufProducerRelease {
    unsafe fn new(dmabuf: WeakDmabuf, acquire_sync: SyncPoint) -> Self {
        Self {
            dmabuf: dmabuf.clone(),
            acquire_sync,
            foreign_general: unsafe {
                // SAFETY: Forwarded from this constructor's caller.
                SampledDmabufKnownLayoutEvidence::foreign_general(dmabuf)
            },
        }
    }

    /// Returns the release dependency that must be satisfied before sampled reacquire uses this dmabuf.
    pub fn acquire_sync(&self) -> &SyncPoint {
        &self.acquire_sync
    }

    /// Returns whether this evidence is tied to `dmabuf`'s Smithay identity.
    pub fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Explicit producer contract for admitting a current Wayland dmabuf commit to Vulkan sampled import.
///
/// A compositor obtains this token from a controlled Vulkan producer release, imports or otherwise
/// orders [`VulkanWaylandDmabufProducerRelease::acquire_sync`] through the Wayland acquire point for
/// the current committed buffer, then calls
/// [`VulkanRenderer::admit_wayland_dmabuf_current_commit_from_vulkan_producer_release_for_sampled_import`]
/// before drawing that surface through the normal renderer-utils [`ImportDmaWl`] path.
///
/// This contract is intentionally narrower than arbitrary linux-dmabuf import: it admits only commits
/// whose producer release token explicitly proves `FOREIGN + GENERAL`, whose protocol-created dmabuf
/// preserves the expected view, and whose renderer-utils cache lifecycle/release ownership remain
/// available for this exact current buffer wrapper.
#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[allow(dead_code)]
#[derive(Debug)]
pub struct VulkanWaylandDmabufSampledImportProducerContract<'a> {
    producer_dmabuf: &'a Dmabuf,
    producer_release: VulkanWaylandDmabufProducerRelease,
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
impl<'a> VulkanWaylandDmabufSampledImportProducerContract<'a> {
    /// Construct the controlled Vulkan producer contract for a Wayland sampled import.
    pub fn from_vulkan_producer_release(
        producer_dmabuf: &'a Dmabuf,
        producer_release: VulkanWaylandDmabufProducerRelease,
    ) -> Result<Self, VulkanError> {
        if !producer_release.is_for_dmabuf(producer_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer evidence",
            ));
        }

        Ok(Self {
            producer_dmabuf,
            producer_release,
        })
    }

    /// Returns the release dependency that must be ordered by the Wayland acquire point.
    pub fn acquire_sync(&self) -> &SyncPoint {
        self.producer_release.acquire_sync()
    }

    /// Import the producer release dependency into a Wayland DRM syncobj acquire point.
    ///
    /// This helper wires the controlled Vulkan producer release into the normal
    /// linux-drm-syncobj-v1 acquire path. The sync-file import only orders the producer release; the
    /// caller must still pass this contract to one of the `admit_wayland_*_from_vulkan_producer_release`
    /// methods for the matching current commit before drawing through `ImportDmaWl`.
    ///
    /// If the producer release dependency is already reached, this signals the acquire point directly.
    /// Otherwise it imports an exported Linux sync-file. Returns [`VulkanError::MissingCapability`] if a
    /// pending producer release dependency cannot be exported as a Linux sync-file, and
    /// [`VulkanError::UnsupportedOperation`] if the acquire point cannot be signaled or cannot import the
    /// sync-file.
    #[cfg(feature = "backend_drm")]
    pub fn import_release_sync_into_acquire_point(
        &self,
        acquire_point: &crate::wayland::drm_syncobj::DrmSyncPoint,
    ) -> Result<(), VulkanError> {
        if self.acquire_sync().is_reached() {
            return acquire_point.signal().map_err(|_| {
                VulkanError::UnsupportedOperation("sampled dmabuf producer acquire sync signal")
            });
        }

        let sync_file = self
            .acquire_sync()
            .export()
            .ok_or(VulkanError::MissingCapability(
                "sampled dmabuf producer release sync-file export",
            ))?;
        acquire_point.import_sync_file(sync_file.as_fd()).map_err(|_| {
            VulkanError::UnsupportedOperation("sampled dmabuf producer acquire sync-file import")
        })
    }
}

/// Validation-stage admission for importing a current Wayland dmabuf commit as a sampled Vulkan image.
///
/// This keeps producer/compositor policy separate from raw linux-dmabuf metadata. The producer evidence
/// proves a controlled Smithay Vulkan producer released the source dmabuf to `FOREIGN + GENERAL`; the
/// remaining proofs are explicit compositor assertions about the committed Wayland buffer and its normal
/// renderer-utils lifecycle. The admission helper validates those assertions against the current
/// renderer-managed surface buffer before recording the wrapper-local evidence consumed by [`ImportDmaWl`].
///
/// This crate-internal type does not public-advertise generic sampled [`ImportDma`] support and must
/// not be used as proof for arbitrary client dmabufs unless a real producer contract establishes the
/// same Vulkan external-state and synchronization guarantees.
#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VulkanWaylandDmabufSampledImportAdmission<'a> {
    producer_dmabuf: &'a Dmabuf,
    producer_evidence: &'a VulkanDmabufLoopbackImportEvidence,
    imported_syncable: Option<VulkanWaylandDmabufSampledImportSyncableProof>,
    imported_view: Option<VulkanWaylandDmabufSampledImportViewProof>,
    acquire_ordering: Option<VulkanWaylandDmabufSampledImportAcquireOrderingProof<'a>>,
    renderer_utils_lifecycle: Option<VulkanWaylandDmabufSampledImportRendererUtilsLifecycleProof>,
}

/// Validation-stage proof that an imported Wayland dmabuf supports dma-buf synchronization ioctls.
///
/// This proves only that the current imported Smithay dmabuf wrapper is backed by fds accepting
/// `DMA_BUF_SYNC` for every plane. It does not prove the dmabuf is the same kernel object as a producer
/// dmabuf, and it does not prove Vulkan image layout, queue-family ownership, or acquire ordering.
#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[derive(Debug, Clone)]
pub(crate) struct VulkanWaylandDmabufSampledImportSyncableProof {
    imported_dmabuf: WeakDmabuf,
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[allow(dead_code)]
impl VulkanWaylandDmabufSampledImportSyncableProof {
    pub(crate) fn new(imported_dmabuf: &Dmabuf) -> Result<Self, VulkanError> {
        if !Self::dmabuf_is_syncable(imported_dmabuf) {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf imported syncable dmabuf",
            ));
        }

        Ok(Self {
            imported_dmabuf: imported_dmabuf.weak(),
        })
    }

    #[cfg(test)]
    unsafe fn assume_for_imported_dmabuf(imported_dmabuf: &Dmabuf) -> Self {
        Self {
            imported_dmabuf: imported_dmabuf.weak(),
        }
    }

    fn is_for(&self, imported_dmabuf: &Dmabuf) -> bool {
        self.imported_dmabuf.upgrade().as_ref() == Some(imported_dmabuf)
    }

    fn dmabuf_is_syncable(imported_dmabuf: &Dmabuf) -> bool {
        imported_dmabuf.num_planes() > 0
            && (0..imported_dmabuf.num_planes()).all(|idx| {
                imported_dmabuf
                    .sync_plane(idx, DmabufSyncFlags::READ | DmabufSyncFlags::START)
                    .is_ok()
                    && imported_dmabuf
                        .sync_plane(idx, DmabufSyncFlags::READ | DmabufSyncFlags::END)
                        .is_ok()
            })
    }
}

/// Validation-stage proof that a Wayland acquire point orders the controlled producer release.
///
/// This is an explicit compositor/producer assertion bound to the producer evidence and imported
/// Smithay dmabuf wrapper. It does not infer ordering from the presence of a drm-syncobj acquire point
/// alone, and it does not prove Vulkan image layout or queue-family ownership. Admission separately
/// verifies that the current renderer-managed buffer still carries an acquire point before recording
/// sampled-import evidence.
#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[derive(Debug, Clone)]
pub(crate) struct VulkanWaylandDmabufSampledImportAcquireOrderingProof<'a> {
    producer_dmabuf: WeakDmabuf,
    producer_evidence: &'a VulkanDmabufLoopbackImportEvidence,
    imported_dmabuf: WeakDmabuf,
    commit_token: Weak<()>,
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[allow(dead_code)]
impl<'a> VulkanWaylandDmabufSampledImportAcquireOrderingProof<'a> {
    /// Assert that `buffer`'s Wayland acquire point orders `producer_evidence.acquire_sync()`.
    ///
    /// # Safety
    ///
    /// The caller must prove that `buffer` belongs to the same current Wayland commit as
    /// `imported_dmabuf`, and that waiting for its acquire point orders the release dependency exposed by
    /// `producer_evidence.acquire_sync()` for `producer_dmabuf`. This constructor does not inspect or
    /// transfer syncobj timelines and does not derive Vulkan external state from Wayland protocol data.
    pub(crate) unsafe fn assume_wayland_acquire_orders_loopback_release(
        producer_dmabuf: &Dmabuf,
        producer_evidence: &'a VulkanDmabufLoopbackImportEvidence,
        imported_dmabuf: &Dmabuf,
        buffer: &super::utils::Buffer,
    ) -> Result<Self, VulkanError> {
        if !producer_evidence.is_for_dmabuf(producer_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer acquire ordering evidence",
            ));
        }
        let current_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer).map_err(|_| {
            VulkanError::UnsupportedOperation("sampled dmabuf producer acquire ordering buffer")
        })?;
        if current_dmabuf != imported_dmabuf {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer acquire ordering buffer identity",
            ));
        }
        if buffer.acquire_point().is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf producer acquire point",
            ));
        }
        let commit_token_slot = buffer
            .user_data()
            .get_or_insert_threadsafe(SampledDmabufWaylandCommitTokenSlot::default);

        Ok(Self {
            producer_dmabuf: producer_dmabuf.weak(),
            producer_evidence,
            imported_dmabuf: imported_dmabuf.weak(),
            commit_token: Arc::downgrade(&commit_token_slot.token),
        })
    }

    fn validate_for(
        &self,
        producer_dmabuf: &Dmabuf,
        producer_evidence: &VulkanDmabufLoopbackImportEvidence,
        imported_dmabuf: &Dmabuf,
        buffer: &super::utils::Buffer,
    ) -> Result<(), VulkanError> {
        if self.producer_dmabuf.upgrade().as_ref() != Some(producer_dmabuf)
            || self.imported_dmabuf.upgrade().as_ref() != Some(imported_dmabuf)
            || !std::ptr::eq(self.producer_evidence, producer_evidence)
            || !producer_evidence.is_for_dmabuf(producer_dmabuf)
        {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer acquire ordering identity",
            ));
        }
        if buffer.acquire_point().is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf producer acquire point",
            ));
        }
        let Some(stored_token) = self.commit_token.upgrade() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer acquire ordering commit token",
            ));
        };
        let Some(current_slot) = buffer.user_data().get::<SampledDmabufWaylandCommitTokenSlot>() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer acquire ordering commit token",
            ));
        };
        if !Arc::ptr_eq(&stored_token, &current_slot.token) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer acquire ordering commit token",
            ));
        }

        Ok(())
    }
}

/// Validation-stage proof that renderer-utils release lifecycle is covered for this current commit.
///
/// This is an explicit compositor assertion bound to the importing renderer context, current imported
/// dmabuf wrapper, and renderer-managed buffer token. It does not by itself release anything; admission
/// validates the token against the current surface buffer before recording the lifecycle evidence later
/// consumed by the normal [`ImportDmaWl`] path.
#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[derive(Debug, Clone)]
pub(crate) struct VulkanWaylandDmabufSampledImportRendererUtilsLifecycleProof {
    renderer_context: ContextId<VulkanTexture>,
    imported_dmabuf: WeakDmabuf,
    commit_token: Weak<()>,
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[allow(dead_code)]
impl VulkanWaylandDmabufSampledImportRendererUtilsLifecycleProof {
    /// Assert that renderer-utils release lifecycle call sites cover this current buffer commit.
    ///
    /// # Safety
    ///
    /// The caller must prove that imports of `buffer` for `renderer` go through Smithay's normal
    /// renderer-utils surface cache, and that all no-next-import/reset/drop/teardown paths for this
    /// surface and renderer satisfy the lifecycle contract documented on
    /// [`VulkanRenderer::mark_wayland_dmabuf_texture_cache_release_lifecycle_for_sampled_import`].
    pub(crate) unsafe fn assume_renderer_utils_lifecycle(
        renderer: &VulkanRenderer,
        buffer: &super::utils::Buffer,
        imported_dmabuf: &Dmabuf,
    ) -> Result<Self, VulkanError> {
        let current_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
            .map_err(|_| VulkanError::UnsupportedOperation("sampled dmabuf producer lifecycle buffer"))?;
        if current_dmabuf != imported_dmabuf {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer lifecycle buffer identity",
            ));
        }
        let commit_token_slot = buffer
            .user_data()
            .get_or_insert_threadsafe(SampledDmabufWaylandCommitTokenSlot::default);

        Ok(Self {
            renderer_context: renderer.context_id.clone(),
            imported_dmabuf: imported_dmabuf.weak(),
            commit_token: Arc::downgrade(&commit_token_slot.token),
        })
    }

    fn validate_for(
        &self,
        renderer: &VulkanRenderer,
        imported_dmabuf: &Dmabuf,
        buffer: &super::utils::Buffer,
    ) -> Result<(), VulkanError> {
        if self.renderer_context != renderer.context_id {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer lifecycle renderer identity",
            ));
        }
        if self.imported_dmabuf.upgrade().as_ref() != Some(imported_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer lifecycle identity",
            ));
        }
        let Some(stored_token) = self.commit_token.upgrade() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer lifecycle commit token",
            ));
        };
        let Some(current_slot) = buffer.user_data().get::<SampledDmabufWaylandCommitTokenSlot>() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer lifecycle commit token",
            ));
        };
        if !Arc::ptr_eq(&stored_token, &current_slot.token) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer lifecycle commit token",
            ));
        }

        Ok(())
    }
}

/// Validation-stage proof that a protocol-created dmabuf preserves the producer's import view.
///
/// This is intentionally a *view* proof, not a Vulkan external-state or portable kernel storage proof.
/// It binds the producer and imported Smithay dmabuf wrappers and verifies that the protocol-created
/// dmabuf has the same size, FourCC/modifier, flags, plane count, offsets, and strides expected by the
/// controlled producer. It does not prove `VK_QUEUE_FAMILY_FOREIGN_EXT`, `VK_IMAGE_LAYOUT_GENERAL`, or
/// acquire ordering; those remain separate admission requirements.
#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[derive(Debug, Clone)]
pub(crate) struct VulkanWaylandDmabufSampledImportViewProof {
    producer_dmabuf: WeakDmabuf,
    imported_dmabuf: WeakDmabuf,
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[allow(dead_code)]
impl VulkanWaylandDmabufSampledImportViewProof {
    pub(crate) fn new(producer_dmabuf: &Dmabuf, imported_dmabuf: &Dmabuf) -> Result<Self, VulkanError> {
        if !Self::same_import_view(producer_dmabuf, imported_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer imported view",
            ));
        }

        Ok(Self {
            producer_dmabuf: producer_dmabuf.weak(),
            imported_dmabuf: imported_dmabuf.weak(),
        })
    }

    fn is_for(&self, producer_dmabuf: &Dmabuf, imported_dmabuf: &Dmabuf) -> bool {
        self.producer_dmabuf.upgrade().as_ref() == Some(producer_dmabuf)
            && self.imported_dmabuf.upgrade().as_ref() == Some(imported_dmabuf)
    }

    fn same_import_view(producer_dmabuf: &Dmabuf, imported_dmabuf: &Dmabuf) -> bool {
        producer_dmabuf.size() == imported_dmabuf.size()
            && producer_dmabuf.format() == imported_dmabuf.format()
            && producer_dmabuf.0.flags == imported_dmabuf.0.flags
            && producer_dmabuf.num_planes() == imported_dmabuf.num_planes()
            && producer_dmabuf.offsets().eq(imported_dmabuf.offsets())
            && producer_dmabuf.strides().eq(imported_dmabuf.strides())
    }
}

#[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
#[allow(dead_code)]
impl<'a> VulkanWaylandDmabufSampledImportAdmission<'a> {
    /// Create an admission policy tied to controlled Smithay Vulkan producer evidence.
    pub(crate) fn from_loopback_evidence(
        producer_dmabuf: &'a Dmabuf,
        producer_evidence: &'a VulkanDmabufLoopbackImportEvidence,
    ) -> Self {
        Self {
            producer_dmabuf,
            producer_evidence,
            imported_syncable: None,
            imported_view: None,
            acquire_ordering: None,
            renderer_utils_lifecycle: None,
        }
    }

    /// Attach proof that the imported Wayland dmabuf is backed by kernel-syncable dma-buf fds.
    pub(crate) fn with_imported_syncable(
        mut self,
        imported_syncable: VulkanWaylandDmabufSampledImportSyncableProof,
    ) -> Self {
        self.imported_syncable = Some(imported_syncable);
        self
    }

    /// Attach proof that protocol import preserved the producer dmabuf view expected by the caller.
    pub(crate) fn with_imported_view(
        mut self,
        imported_view: VulkanWaylandDmabufSampledImportViewProof,
    ) -> Self {
        self.imported_view = Some(imported_view);
        self
    }

    /// Attach proof that Wayland acquire synchronization orders `producer_evidence.acquire_sync()`.
    pub(crate) fn with_acquire_ordering(
        mut self,
        acquire_ordering: VulkanWaylandDmabufSampledImportAcquireOrderingProof<'a>,
    ) -> Self {
        self.acquire_ordering = Some(acquire_ordering);
        self
    }

    /// Attach proof that renderer-utils release lifecycle is covered for this surface cache.
    pub(crate) fn with_renderer_utils_lifecycle(
        mut self,
        renderer_utils_lifecycle: VulkanWaylandDmabufSampledImportRendererUtilsLifecycleProof,
    ) -> Self {
        self.renderer_utils_lifecycle = Some(renderer_utils_lifecycle);
        self
    }

    /// Admit the current renderer-managed dmabuf commit for validation-stage sampled import.
    ///
    /// The helper verifies producer evidence identity, caller-supplied protocol/sync/lifecycle
    /// declarations, current surface state, current dmabuf identity, and explicit acquire/release
    /// points before recording external-state and lifecycle evidence on the current buffer wrapper.
    ///
    /// # Safety
    ///
    /// The caller must prove that `producer_evidence` describes the image state of this exact current
    /// Wayland commit, that no intervening access changed the image's external state, and that the
    /// acquire point on `surface` orders the producer release to `VK_QUEUE_FAMILY_FOREIGN_EXT` in
    /// `VK_IMAGE_LAYOUT_GENERAL`. The renderer-utils lifecycle declaration must cover replacement,
    /// no-next-import, surface reset/drop, and renderer teardown release paths for this renderer.
    pub(crate) unsafe fn admit_current_surface_commit(
        &self,
        renderer: &VulkanRenderer,
        surface: &WlSurface,
        committed_dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        if !self.producer_evidence.is_for_dmabuf(self.producer_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer evidence",
            ));
        }
        let imported_syncable = self
            .imported_syncable
            .as_ref()
            .ok_or(VulkanError::MissingCapability(
                "sampled dmabuf imported syncable dmabuf",
            ))?;
        if !imported_syncable.is_for(committed_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf imported syncable dmabuf identity",
            ));
        }
        let imported_view = self.imported_view.as_ref().ok_or(VulkanError::MissingCapability(
            "sampled dmabuf producer imported view",
        ))?;
        if !imported_view.is_for(self.producer_dmabuf, committed_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer imported view identity",
            ));
        }
        let acquire_ordering = self
            .acquire_ordering
            .as_ref()
            .ok_or(VulkanError::MissingCapability(
                "sampled dmabuf producer acquire ordering",
            ))?;
        let renderer_utils_lifecycle =
            self.renderer_utils_lifecycle
                .as_ref()
                .ok_or(VulkanError::MissingCapability(
                    "sampled dmabuf producer lifecycle policy",
                ))?;

        super::utils::with_renderer_surface_state(surface, |state| {
            let buffer = state.buffer().ok_or(VulkanError::MissingCapability(
                "sampled dmabuf producer current buffer",
            ))?;
            let current_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
                .map_err(|_| VulkanError::UnsupportedOperation("sampled dmabuf producer current buffer"))?;
            if current_dmabuf != committed_dmabuf {
                return Err(VulkanError::UnsupportedOperation(
                    "sampled dmabuf producer current buffer identity",
                ));
            }
            acquire_ordering.validate_for(
                self.producer_dmabuf,
                self.producer_evidence,
                committed_dmabuf,
                buffer,
            )?;
            renderer_utils_lifecycle.validate_for(renderer, committed_dmabuf, buffer)?;
            if buffer.release_point().is_none() {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf producer release point",
                ));
            }

            let (use_case, release_generation) =
                match renderer.sampled_dmabuf_layout_history_snapshot(committed_dmabuf) {
                    SampledDmabufWaylandLayoutHistory::NoRendererHistory => {
                        (SampledDmabufWaylandExternalStateUse::FirstImport, None)
                    }
                    SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => (
                        SampledDmabufWaylandExternalStateUse::CurrentReacquire,
                        renderer.sampled_dmabuf_release_generation_snapshot(committed_dmabuf),
                    ),
                    SampledDmabufWaylandLayoutHistory::LocallyAcquired => {
                        return Err(VulkanError::MissingCapability(
                            "sampled dmabuf Wayland Vulkan unreleased local acquire",
                        ));
                    }
                };

            unsafe {
                // SAFETY: Forwarded from this admission method's caller after validating the exact
                // renderer-managed buffer, current dmabuf identity, producer evidence, acquire ordering,
                // release ownership availability, and renderer-utils lifecycle proof above. Recording
                // on `buffer.user_data()` avoids a second surface lookup after validation.
                VulkanRenderer::mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
                    buffer.user_data(),
                    committed_dmabuf,
                    use_case,
                    release_generation,
                )?;
                renderer.mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
                    buffer.user_data(),
                    committed_dmabuf,
                )?;
            }

            Ok(())
        })
        .unwrap_or(Err(VulkanError::MissingCapability(
            "sampled dmabuf producer renderer surface state",
        )))?;

        Ok(())
    }
}

#[cfg(all(test, feature = "wayland_frontend", feature = "backend_drm"))]
#[derive(Debug, Clone)]
struct VulkanWaylandDmabufSampledReacquireAcquireOrderingProof<'a> {
    renderer_context: ContextId<VulkanTexture>,
    renderer_release_evidence: &'a SampledDmabufWaylandRendererForeignGeneralReleaseEvidence,
    imported_dmabuf: WeakDmabuf,
    commit_token: Weak<()>,
}

#[cfg(all(test, feature = "wayland_frontend", feature = "backend_drm"))]
impl<'a> VulkanWaylandDmabufSampledReacquireAcquireOrderingProof<'a> {
    /// Assert that `buffer`'s Wayland acquire point orders this renderer's previous sampled release.
    ///
    /// # Safety
    ///
    /// The caller must prove that `buffer` belongs to the same current Wayland commit as
    /// `imported_dmabuf`, that waiting for its acquire point orders the renderer release represented by
    /// `renderer_release_evidence`, and that no foreign producer changed the dmabuf's Vulkan external
    /// state after that release and before this reacquire commit.
    unsafe fn assume_wayland_acquire_orders_renderer_release(
        renderer: &VulkanRenderer,
        renderer_release_evidence: &'a SampledDmabufWaylandRendererForeignGeneralReleaseEvidence,
        imported_dmabuf: &Dmabuf,
        buffer: &super::utils::Buffer,
    ) -> Result<Self, VulkanError> {
        if !renderer_release_evidence.is_for_renderer_context(&renderer.context_id) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire evidence renderer identity",
            ));
        }
        if !renderer_release_evidence.is_for_dmabuf(imported_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire evidence identity",
            ));
        }
        if !renderer_release_evidence.is_current_release_generation(
            renderer.sampled_dmabuf_release_generation_snapshot(imported_dmabuf),
        ) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire evidence generation",
            ));
        }
        let current_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer).map_err(|_| {
            VulkanError::UnsupportedOperation("sampled dmabuf renderer-release reacquire buffer")
        })?;
        if current_dmabuf != imported_dmabuf {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire buffer identity",
            ));
        }
        if buffer.acquire_point().is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf renderer-release reacquire acquire point",
            ));
        }
        let commit_token_slot = buffer
            .user_data()
            .get_or_insert_threadsafe(SampledDmabufWaylandCommitTokenSlot::default);

        Ok(Self {
            renderer_context: renderer.context_id.clone(),
            renderer_release_evidence,
            imported_dmabuf: imported_dmabuf.weak(),
            commit_token: Arc::downgrade(&commit_token_slot.token),
        })
    }

    fn validate_for(
        &self,
        renderer: &VulkanRenderer,
        renderer_release_evidence: &SampledDmabufWaylandRendererForeignGeneralReleaseEvidence,
        imported_dmabuf: &Dmabuf,
        buffer: &super::utils::Buffer,
    ) -> Result<(), VulkanError> {
        if self.renderer_context != renderer.context_id
            || !std::ptr::eq(self.renderer_release_evidence, renderer_release_evidence)
            || !renderer_release_evidence.is_for_renderer_context(&renderer.context_id)
            || !renderer_release_evidence.is_for_dmabuf(imported_dmabuf)
            || self.imported_dmabuf.upgrade().as_ref() != Some(imported_dmabuf)
        {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire ordering identity",
            ));
        }
        if !renderer_release_evidence.is_current_release_generation(
            renderer.sampled_dmabuf_release_generation_snapshot(imported_dmabuf),
        ) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire ordering generation",
            ));
        }
        if buffer.acquire_point().is_none() {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf renderer-release reacquire acquire point",
            ));
        }
        let Some(stored_token) = self.commit_token.upgrade() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire commit token",
            ));
        };
        let Some(current_slot) = buffer.user_data().get::<SampledDmabufWaylandCommitTokenSlot>() else {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire commit token",
            ));
        };
        if !Arc::ptr_eq(&stored_token, &current_slot.token) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire commit token",
            ));
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "wayland_frontend", feature = "backend_drm"))]
#[derive(Debug)]
struct VulkanWaylandDmabufSampledReacquireAdmission<'a> {
    renderer_release_evidence: &'a SampledDmabufWaylandRendererForeignGeneralReleaseEvidence,
    imported_syncable: Option<VulkanWaylandDmabufSampledImportSyncableProof>,
    imported_view: Option<VulkanWaylandDmabufSampledImportViewProof>,
    acquire_ordering: Option<VulkanWaylandDmabufSampledReacquireAcquireOrderingProof<'a>>,
    renderer_utils_lifecycle: Option<VulkanWaylandDmabufSampledImportRendererUtilsLifecycleProof>,
}

#[cfg(all(test, feature = "wayland_frontend", feature = "backend_drm"))]
impl<'a> VulkanWaylandDmabufSampledReacquireAdmission<'a> {
    fn from_renderer_release_evidence(
        renderer_release_evidence: &'a SampledDmabufWaylandRendererForeignGeneralReleaseEvidence,
    ) -> Self {
        Self {
            renderer_release_evidence,
            imported_syncable: None,
            imported_view: None,
            acquire_ordering: None,
            renderer_utils_lifecycle: None,
        }
    }

    fn with_imported_syncable(
        mut self,
        imported_syncable: VulkanWaylandDmabufSampledImportSyncableProof,
    ) -> Self {
        self.imported_syncable = Some(imported_syncable);
        self
    }

    fn with_imported_view(mut self, imported_view: VulkanWaylandDmabufSampledImportViewProof) -> Self {
        self.imported_view = Some(imported_view);
        self
    }

    fn with_acquire_ordering(
        mut self,
        acquire_ordering: VulkanWaylandDmabufSampledReacquireAcquireOrderingProof<'a>,
    ) -> Self {
        self.acquire_ordering = Some(acquire_ordering);
        self
    }

    fn with_renderer_utils_lifecycle(
        mut self,
        renderer_utils_lifecycle: VulkanWaylandDmabufSampledImportRendererUtilsLifecycleProof,
    ) -> Self {
        self.renderer_utils_lifecycle = Some(renderer_utils_lifecycle);
        self
    }

    /// Admit a same-dmabuf current commit using this renderer's previous sampled release evidence.
    ///
    /// # Safety
    ///
    /// The caller must prove that `renderer_release_evidence` is the release event ordered by this
    /// commit's Wayland acquire point, that no foreign producer changed the dmabuf's queue-family
    /// ownership or image layout after that release, and that the renderer-utils lifecycle proof covers
    /// all no-next-import/reset/drop/teardown paths for this current buffer commit. This helper records
    /// validation-stage evidence only; the actual acquire wait and release-point transfer still happen
    /// through the following normal `ImportDmaWl` import.
    unsafe fn admit_current_surface_commit(
        &self,
        renderer: &VulkanRenderer,
        surface: &WlSurface,
        committed_dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        if !self
            .renderer_release_evidence
            .is_for_renderer_context(&renderer.context_id)
            || !self.renderer_release_evidence.is_for_dmabuf(committed_dmabuf)
        {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire evidence identity",
            ));
        }
        if !self.renderer_release_evidence.is_current_release_generation(
            renderer.sampled_dmabuf_release_generation_snapshot(committed_dmabuf),
        ) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf renderer-release reacquire evidence generation",
            ));
        }
        let imported_syncable = self
            .imported_syncable
            .as_ref()
            .ok_or(VulkanError::MissingCapability(
                "sampled dmabuf imported syncable dmabuf",
            ))?;
        if !imported_syncable.is_for(committed_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf imported syncable dmabuf identity",
            ));
        }
        let imported_view = self.imported_view.as_ref().ok_or(VulkanError::MissingCapability(
            "sampled dmabuf producer imported view",
        ))?;
        if !imported_view.is_for(committed_dmabuf, committed_dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf producer imported view identity",
            ));
        }
        let acquire_ordering = self
            .acquire_ordering
            .as_ref()
            .ok_or(VulkanError::MissingCapability(
                "sampled dmabuf renderer-release reacquire acquire ordering",
            ))?;
        let renderer_utils_lifecycle =
            self.renderer_utils_lifecycle
                .as_ref()
                .ok_or(VulkanError::MissingCapability(
                    "sampled dmabuf producer lifecycle policy",
                ))?;

        super::utils::with_renderer_surface_state(surface, |state| {
            let buffer = state.buffer().ok_or(VulkanError::MissingCapability(
                "sampled dmabuf renderer-release reacquire current buffer",
            ))?;
            let current_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer).map_err(|_| {
                VulkanError::UnsupportedOperation("sampled dmabuf renderer-release reacquire current buffer")
            })?;
            if current_dmabuf != committed_dmabuf {
                return Err(VulkanError::UnsupportedOperation(
                    "sampled dmabuf renderer-release reacquire current buffer identity",
                ));
            }
            acquire_ordering.validate_for(
                renderer,
                self.renderer_release_evidence,
                committed_dmabuf,
                buffer,
            )?;
            renderer_utils_lifecycle.validate_for(renderer, committed_dmabuf, buffer)?;
            if buffer.release_point().is_none() {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf renderer-release reacquire release point",
                ));
            }

            unsafe {
                // SAFETY: Forwarded from this admission method's caller after validation of this exact
                // renderer-managed buffer, renderer release evidence, sync, and lifecycle proof tokens.
                // Recording on `buffer.user_data()` avoids re-looking up a different same-dmabuf commit
                // after token validation.
                VulkanRenderer::mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
                    buffer.user_data(),
                    committed_dmabuf,
                    SampledDmabufWaylandExternalStateUse::CurrentReacquire,
                    Some(self.renderer_release_evidence.release_generation),
                )?;
                renderer.mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
                    buffer.user_data(),
                    committed_dmabuf,
                )?;
            }

            Ok(())
        })
        .unwrap_or(Err(VulkanError::MissingCapability(
            "sampled dmabuf renderer-release reacquire renderer surface state",
        )))?;

        Ok(())
    }
}

use self::{
    device::{
        VulkanDeviceState, VulkanDmabufRenderTargetForeignReleaseError,
        VulkanSampledDmabufForeignReleaseError, VulkanSyncFileSemaphore, image_copy_buffer_offset,
        tightly_packed_image_size,
    },
    format::{get_format_info, get_render_vk_format},
};

fn sampled_dmabuf_cache_device_release_error(
    err: VulkanSampledDmabufForeignReleaseError,
) -> SurfaceCacheTextureReleaseError<VulkanError> {
    match err {
        VulkanSampledDmabufForeignReleaseError::RetrySafe(err) => {
            SurfaceCacheTextureReleaseError::RetrySafe(err)
        }
        VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(err) => {
            SurfaceCacheTextureReleaseError::ReleaseSideEffectsCommitted(err)
        }
    }
}

#[derive(Debug)]
struct VulkanSyncFileFence {
    fd: OwnedFd,
}

impl VulkanSyncFileFence {
    fn new(fd: OwnedFd) -> Self {
        Self { fd }
    }
}

impl Fence for VulkanSyncFileFence {
    fn is_signaled(&self) -> bool {
        let mut poll_fd = [rustix::event::PollFd::new(&self.fd, rustix::event::PollFlags::IN)];
        matches!(
            rustix::event::poll(
                &mut poll_fd,
                Some(&rustix::time::Timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                }),
            ),
            Ok(ready) if ready > 0
        )
    }

    fn wait(&self) -> Result<(), Interrupted> {
        let mut poll_fd = [rustix::event::PollFd::new(&self.fd, rustix::event::PollFlags::IN)];
        loop {
            match rustix::event::poll(&mut poll_fd, None) {
                Ok(ready) if ready > 0 => return Ok(()),
                Ok(_) => continue,
                Err(_) => return Err(Interrupted),
            }
        }
    }

    fn is_exportable(&self) -> bool {
        true
    }

    fn export(&self) -> Option<OwnedFd> {
        self.fd.as_fd().try_clone_to_owned().ok()
    }
}

fn sync_point_from_sync_file(sync_file: Option<OwnedFd>) -> SyncPoint {
    match sync_file {
        Some(fd) => SyncPoint::from(VulkanSyncFileFence::new(fd)),
        None => SyncPoint::signaled(),
    }
}

/// Provisional native Vulkan renderer for explicit-device in-memory/offscreen rendering.
#[derive(Debug)]
pub struct VulkanRenderer {
    context_id: ContextId<VulkanTexture>,
    debug_flags: DebugFlags,
    downscale_filter: TextureFilter,
    upscale_filter: TextureFilter,
    capabilities: VulkanRendererCapabilities,
    device: Option<VulkanDeviceState>,
    sampled_dmabuf_layout_history: HashMap<WeakDmabuf, SampledDmabufWaylandLayoutHistoryRecord>,
    pending_sampled_dmabuf_import_obligations: Vec<PendingSampledDmabufImportObligation>,
    wayland_linux_dmabuf_interop: bool,
}

/// Builder for explicit Vulkan renderer initialization.
///
/// The builder initializes a logical Vulkan device from an explicitly provided [`PhysicalDevice`].
/// Public operations remain limited to the capability bits advertised by the initialized renderer.
#[derive(Debug, Default, Clone)]
pub struct VulkanRendererBuilder {
    physical_device: Option<PhysicalDevice>,
    extra_device_extensions: Vec<&'static std::ffi::CStr>,
}

impl VulkanRendererBuilder {
    /// Creates an empty Vulkan renderer builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the Vulkan physical device to use for renderer initialization.
    ///
    /// The renderer derives its Vulkan instance from this [`PhysicalDevice`], matching Smithay's
    /// backend ownership model and avoiding mismatched instance/device pairs.
    pub fn with_physical_device(mut self, physical_device: PhysicalDevice) -> Self {
        self.physical_device = Some(physical_device);
        self
    }

    /// Enables additional logical-device extensions when the renderer device is created.
    ///
    /// Window-system integration (for example `VK_KHR_swapchain`) is not implied by the offscreen
    /// renderer. Callers that present to a `VkSurfaceKHR` must request the swapchain extension here
    /// after confirming the physical device supports it.
    pub fn with_device_extensions(mut self, extensions: &[&'static std::ffi::CStr]) -> Self {
        self.extra_device_extensions.extend_from_slice(extensions);
        self
    }

    /// Builds a Vulkan renderer.
    ///
    /// This returns [`VulkanError::VulkanUnavailable`] unless a [`PhysicalDevice`] was provided.
    pub fn build(self) -> Result<VulkanRenderer, VulkanError> {
        let physical_device = self.physical_device.ok_or(VulkanError::VulkanUnavailable)?;
        let device = VulkanDeviceState::new(physical_device, &self.extra_device_extensions)?;
        let capabilities = device.capabilities.clone();

        Ok(VulkanRenderer {
            context_id: ContextId::new(),
            debug_flags: DebugFlags::empty(),
            downscale_filter: TextureFilter::Linear,
            upscale_filter: TextureFilter::Linear,
            capabilities,
            device: Some(device),
            sampled_dmabuf_layout_history: HashMap::new(),
            pending_sampled_dmabuf_import_obligations: Vec::new(),
            wayland_linux_dmabuf_interop: false,
        })
    }
}

impl VulkanRenderer {
    /// Vulkan instance this renderer was created from.
    pub fn instance(&self) -> Option<&Instance> {
        self.device.as_ref()?.instance.as_ref()
    }

    /// Physical device this renderer was created from.
    pub fn physical_device(&self) -> Option<&PhysicalDevice> {
        self.device.as_ref()?.physical_device.as_ref()
    }

    /// Enable or disable the opt-in Linux dma-buf external interop policy for Wayland sampled import.
    ///
    /// Default is off. When enabled, [`ImportDmaWl`] may record the existing FOREIGN/`GENERAL` and
    /// texture-cache lifecycle marks for commits that have not already been admitted by the
    /// controlled Vulkan producer path. That is the Linux EGL/Vulkan dma-buf convention after
    /// acquire is satisfied, not a proof from linux-dmabuf metadata. Generic [`ImportDma`] stays
    /// fail-closed.
    pub fn set_wayland_linux_dmabuf_interop(&mut self, enabled: bool) {
        self.wayland_linux_dmabuf_interop = enabled;
    }

    /// Returns whether the Linux dma-buf external interop policy is enabled.
    pub fn wayland_linux_dmabuf_interop(&self) -> bool {
        self.wayland_linux_dmabuf_interop
    }

    /// Formats the Wayland linux-dmabuf global may advertise for [`ImportDmaWl`].
    ///
    /// Empty unless Linux dma-buf interop is enabled. This is not [`ImportDma::dmabuf_formats`].
    pub fn wayland_sampled_dmabuf_formats(&self) -> FormatSet {
        if self.wayland_linux_dmabuf_interop {
            self.capabilities.formats.dmabuf_import.clone()
        } else {
            FormatSet::default()
        }
    }

    /// Wraps a swapchain image the renderer must not destroy.
    ///
    /// Layout tracking starts at `UNDEFINED`. That is the WSI discard path: this backend does not
    /// preserve prior contents (`buffer_age` is 0), so `oldLayout=UNDEFINED` is valid even after
    /// the image was previously `PRESENT_SRC_KHR`.
    pub(crate) fn wrap_swapchain_image(
        &self,
        image: ash::vk::Image,
        size: crate::utils::Size<i32, crate::utils::Buffer>,
        format: Fourcc,
    ) -> Result<VulkanRenderTarget<'static>, VulkanError> {
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let logical_device = device
            .logical_device
            .as_ref()
            .ok_or(VulkanError::VulkanUnavailable)?
            .clone();
        let extent = vk::Extent3D {
            width: size.w.max(0) as u32,
            height: size.h.max(0) as u32,
            depth: 1,
        };
        let vk_format = get_render_vk_format(format)?;
        let color_image =
            device::VulkanOwnedImage::from_unowned_swapchain_image(logical_device, image, extent, vk_format);
        Ok(image::VulkanRenderTarget::from_swapchain_image(
            self.context_id.clone(),
            size,
            format,
            color_image,
        ))
    }

    pub(crate) fn transition_swapchain_target(
        &self,
        target: &VulkanRenderTarget<'_>,
        new_layout: vk::ImageLayout,
    ) -> Result<(), VulkanError> {
        if target.image.source != image::VulkanImageSource::Swapchain {
            return Err(VulkanError::UnsupportedOperation("swapchain target"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("swapchain target image"))?;
        let mut command_buffer = device.allocate_graphics_command_buffer()?;
        device.begin_command_buffer(&mut command_buffer)?;
        device.transition_image_layout(&mut command_buffer, color_image, new_layout)?;
        device.end_command_buffer(&mut command_buffer)?;
        device.submit_graphics_command_buffer_and_wait(&mut command_buffer)?;
        Ok(())
    }

    pub(crate) fn graphics_queue(&self) -> Result<(vk::Queue, u32), VulkanError> {
        let queue = self
            .device
            .as_ref()
            .ok_or(VulkanError::VulkanUnavailable)?
            .queues
            .graphics
            .as_ref()
            .ok_or(VulkanError::QueueFamilyUnsupported)?;
        Ok((queue.handle(), queue.queue_family_index()))
    }

    /// Runs `f` with the graphics queue while holding its host-access lock.
    ///
    /// Vulkan requires host access to a `VkQueue` to be externally synchronized, including
    /// `vkQueueSubmit` and `vkQueuePresentKHR`. Renderer submits already take this lock; window
    /// present must use this helper instead of the raw handle from [`Self::graphics_queue`].
    pub(crate) fn with_locked_graphics_queue<T, F>(&self, f: F) -> Result<T, VulkanError>
    where
        F: FnOnce(vk::Queue) -> Result<T, VulkanError>,
    {
        let queue = self
            .device
            .as_ref()
            .ok_or(VulkanError::VulkanUnavailable)?
            .queues
            .graphics
            .as_ref()
            .ok_or(VulkanError::QueueFamilyUnsupported)?;
        let _guard = queue.lock_host_access()?;
        f(queue.handle())
    }

    pub(crate) fn logical_device(&self) -> Result<&ash::Device, VulkanError> {
        Ok(self
            .device
            .as_ref()
            .ok_or(VulkanError::VulkanUnavailable)?
            .logical_device
            .as_ref()
            .ok_or(VulkanError::VulkanUnavailable)?
            .handle())
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        let obligations = std::mem::take(&mut self.pending_sampled_dmabuf_import_obligations);
        for obligation in obligations {
            match obligation {
                PendingSampledDmabufImportObligation::ReleaseOnly(release_ownership) => {
                    if let Err(err) = release_ownership.signal_wayland_release_once() {
                        tracing::warn!(?err, "failed to satisfy pending sampled dmabuf release point");
                    }
                }
                PendingSampledDmabufImportObligation::AcquiredTexture(texture) => {
                    match self.release_imported_dmabuf_texture_to_foreign_general_classified(&texture, false)
                    {
                        Ok((true, _)) => {}
                        Ok((false, _)) => {
                            tracing::warn!("pending sampled dmabuf import texture was not released on drop");
                        }
                        Err(device::VulkanSampledDmabufForeignReleaseError::RetrySafe(err)) => {
                            tracing::warn!(?err, "failed to release pending sampled dmabuf import texture");
                        }
                        Err(device::VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(err)) => {
                            tracing::warn!(
                                ?err,
                                "sampled dmabuf import texture release submission failed during drop"
                            );
                        }
                    }
                }
                PendingSampledDmabufImportObligation::ReleaseCompletionUnknownTexture(_texture) => {
                    tracing::warn!(
                        "dropping sampled dmabuf import texture with unknown submitted release state; \
                         any attached release point remains unsatisfied because Vulkan release \
                         completion could not be proven"
                    );
                }
            }
        }
    }
}

impl VulkanRenderer {
    /// Attempts to create a Vulkan renderer.
    ///
    /// This returns [`VulkanError::VulkanUnavailable`] until a default device-selection policy exists.
    /// Use [`VulkanRenderer::builder`] with an explicit [`PhysicalDevice`] to initialize device state.
    pub fn new() -> Result<Self, VulkanError> {
        Self::builder().build()
    }

    /// Creates a builder for explicit Vulkan renderer initialization.
    pub fn builder() -> VulkanRendererBuilder {
        VulkanRendererBuilder::new()
    }

    /// Mark a renderer-managed Wayland dmabuf buffer as released to Vulkan `FOREIGN + GENERAL`.
    ///
    /// This is a development-stage evidence API for the normal [`ImportDmaWl`] path. It does not
    /// public-advertise sampled [`ImportDma`] support and does not derive evidence from Wayland
    /// protocol metadata. The marker is stored on Smithay's current renderer-managed buffer wrapper,
    /// so explicit-sync same-buffer commits receive a fresh evidence slot when
    /// `on_commit_buffer_handler` refreshes the wrapper.
    ///
    /// # Safety
    ///
    /// The caller must prove that the current commit represented by `buffer` and `dmabuf` has been
    /// released by its producer to `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`, and
    /// that the buffer's acquire synchronization orders the producer writes and ownership release for
    /// this exact dmabuf. This proof must be current for this commit; linux-dmabuf format/plane
    /// metadata and linux-drm-syncobj acquire/release points are not sufficient by themselves.
    #[cfg(all(test, feature = "wayland_frontend"))]
    unsafe fn mark_wayland_dmabuf_foreign_general_for_sampled_import(
        buffer: &super::utils::Buffer,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Forwarded from this test-only unsafe evidence-marking helper's caller.
            Self::mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
                buffer.user_data(),
                dmabuf,
                SampledDmabufWaylandExternalStateUse::FirstImport,
                None,
            )
        }
    }

    /// Admit a Wayland dmabuf commit under the opt-in Linux dma-buf external interop policy.
    ///
    /// This writes the same buffer-local FOREIGN/`GENERAL` and texture-cache lifecycle marks as
    /// [`VulkanRenderer::assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import`]
    /// and
    /// [`VulkanRenderer::mark_wayland_dmabuf_texture_cache_release_lifecycle_for_sampled_import`].
    /// It does not replace the controlled Vulkan producer admission path. If those marks are already
    /// present, this is a no-op.
    ///
    /// Acquire must already be representable as a fence-bearing [`SyncPoint`]: either a drm-syncobj
    /// acquire point or a dma-buf exported read fence. This helper does not infer layout from
    /// format/modifier metadata.
    #[cfg(feature = "wayland_frontend")]
    fn admit_wayland_linux_dmabuf_interop_for_sampled_import(
        &self,
        buffer: &super::utils::Buffer,
    ) -> Result<(), VulkanError> {
        if !self.wayland_linux_dmabuf_interop {
            return Ok(());
        }

        let dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
            .map_err(|_| VulkanError::UnsupportedOperation("sampled dmabuf linux interop buffer"))?;
        if self
            .sampled_dmabuf_wayland_buffer_foreign_general_evidence(buffer, dmabuf)?
            .is_some()
        {
            return Ok(());
        }

        self.sampled_dmabuf_wayland_acquire_sync_evidence(dmabuf, buffer)?;

        let (use_case, release_generation) = match self.sampled_dmabuf_layout_history_snapshot(dmabuf) {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => {
                (SampledDmabufWaylandExternalStateUse::FirstImport, None)
            }
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => (
                SampledDmabufWaylandExternalStateUse::CurrentReacquire,
                self.sampled_dmabuf_release_generation_snapshot(dmabuf),
            ),
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland Vulkan unreleased local acquire",
                ));
            }
        };

        unsafe {
            // SAFETY: The Linux-interop flag is an explicit compositor opt-in that acquire has been
            // satisfied and the imported dma-buf follows FOREIGN + GENERAL external layout. This
            // records the existing commit-local marks consumed by ImportDmaWl; it does not advertise
            // generic ImportDma.
            Self::mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
                buffer.user_data(),
                dmabuf,
                use_case,
                release_generation,
            )?;
            self.mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
                buffer.user_data(),
                dmabuf,
            )?;
        }

        Ok(())
    }

    /// Assume a renderer-managed Wayland dmabuf commit's external state for validation-stage sampled import.
    ///
    /// This is the remaining development contract for the normal [`ImportDmaWl`] path. It stores the
    /// current-commit `FOREIGN + GENERAL` Vulkan external image state that cannot be inferred from
    /// Wayland protocol metadata. Renderer-utils texture-cache release lifecycle coverage is a
    /// separate explicit marker, so this function cannot be used to smuggle no-next-import or teardown
    /// release assumptions. It does not consume the Wayland release point, create a Vulkan image, or
    /// public-advertise generic sampled [`ImportDma`] support.
    ///
    /// # Safety
    ///
    /// The caller must prove that this same `buffer`/`dmabuf` pair is currently in `FOREIGN + GENERAL`
    /// ownership/layout for this commit. The compositor must still use the normal renderer-utils
    /// lifecycle hooks for cache release; that lifecycle evidence is validated separately.
    #[cfg(feature = "wayland_frontend")]
    pub unsafe fn assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import(
        &self,
        buffer: &super::utils::Buffer,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        let (use_case, release_generation) = match self.sampled_dmabuf_layout_history_snapshot(dmabuf) {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => {
                (SampledDmabufWaylandExternalStateUse::FirstImport, None)
            }
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => (
                SampledDmabufWaylandExternalStateUse::CurrentReacquire,
                self.sampled_dmabuf_release_generation_snapshot(dmabuf),
            ),
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland Vulkan unreleased local acquire",
                ));
            }
        };
        unsafe {
            // SAFETY: Forwarded from this validation contract's caller.
            Self::mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
                buffer.user_data(),
                dmabuf,
                use_case,
                release_generation,
            )
        }
    }

    /// Mark a renderer-managed Wayland dmabuf commit's external state for validation-stage sampled import.
    ///
    /// Prefer
    /// [`assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import`](Self::assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import)
    /// so callers explicitly acknowledge the unsafe `FOREIGN + GENERAL` external-state assumption.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the same safety contract as
    /// [`assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import`](Self::assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import).
    #[cfg(feature = "wayland_frontend")]
    #[deprecated(
        note = "use assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import to make the unsafe external-state assumption explicit"
    )]
    pub unsafe fn mark_wayland_dmabuf_current_commit_for_sampled_import(
        &self,
        buffer: &super::utils::Buffer,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Forwarded from this compatibility wrapper's caller.
            self.assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import(buffer, dmabuf)
        }
    }

    /// Assume the current renderer-managed dmabuf commit on a [`WlSurface`] for validation-stage
    /// sampled import.
    ///
    /// This is a convenience wrapper around
    /// [`assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import`](Self::assume_wayland_dmabuf_current_commit_foreign_general_for_sampled_import)
    /// for compositors that use Smithay's normal [`super::utils::on_commit_buffer_handler`] path. It
    /// looks up the current renderer-managed buffer stored for `surface`, verifies that it is the same
    /// dmabuf identity as `dmabuf`, and then records the current-commit external-state contract on
    /// that buffer wrapper. It does not consume the Wayland release point,
    /// create a Vulkan image, or public-advertise generic sampled [`ImportDma`] support.
    ///
    /// # Safety
    ///
    /// The caller must prove that the surface's current renderer-managed buffer is the current commit
    /// represented by `dmabuf`, that the producer released that image to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`, and that the buffer's acquire
    /// synchronization orders the producer writes and ownership release for this exact commit. This
    /// marker does not record renderer-utils texture-cache release lifecycle evidence.
    #[cfg(feature = "wayland_frontend")]
    pub unsafe fn assume_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import(
        &self,
        surface: &WlSurface,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        let (use_case, release_generation) = match self.sampled_dmabuf_layout_history_snapshot(dmabuf) {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => {
                (SampledDmabufWaylandExternalStateUse::FirstImport, None)
            }
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => (
                SampledDmabufWaylandExternalStateUse::CurrentReacquire,
                self.sampled_dmabuf_release_generation_snapshot(dmabuf),
            ),
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland Vulkan unreleased local acquire",
                ));
            }
        };
        unsafe {
            // SAFETY: Forwarded from this surface-level validation contract's caller.
            self.mark_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import_use(
                surface,
                dmabuf,
                use_case,
                release_generation,
            )
        }
    }

    /// Admit a current Wayland dmabuf commit from a controlled Vulkan producer release.
    ///
    /// This is the concrete `ImportDmaWl` producer contract for Smithay-controlled Vulkan producers:
    /// the producer supplies [`VulkanWaylandDmabufProducerRelease`] proving that it released the
    /// source dmabuf to `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`, while this method
    /// validates that the current renderer-managed Wayland buffer is the expected committed dmabuf,
    /// that the protocol import preserved the producer view, that an explicit acquire point is present,
    /// that a release point remains available for renderer ownership, and that renderer-utils cache
    /// lifecycle evidence is tied to this renderer/current-buffer token. On success it records the
    /// wrapper-local evidence consumed by the normal [`ImportDmaWl`] surface path.
    ///
    /// This method does not public-advertise generic [`ImportDma`] support and must not be used for
    /// arbitrary linux-dmabuf producers unless the caller has an equivalent producer release token and
    /// acquire-ordering proof for the exact current commit.
    ///
    /// # Safety
    ///
    /// The caller must prove that `buffer` belongs to the same current commit on `surface` as
    /// `committed_dmabuf`, that the committed Wayland dmabuf is the same underlying storage/image that
    /// was released by `contract`, that the producer release is still the current external state for
    /// this commit with no intervening access, acquire, release, image-layout transition, or
    /// queue-family transfer, that `contract.acquire_sync()` is imported into or otherwise ordered by
    /// `buffer`'s Wayland acquire point before `ImportDmaWl` can import the buffer, that no concurrent
    /// surface replacement can occur for the duration of this call, and that all imports of this surface
    /// with `self` use the normal renderer-utils cache release paths before replacement,
    /// no-next-import, reset, teardown, or renderer drop.
    #[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
    pub unsafe fn admit_wayland_dmabuf_current_commit_from_vulkan_producer_release_for_sampled_import(
        &self,
        contract: VulkanWaylandDmabufSampledImportProducerContract<'_>,
        surface: &WlSurface,
        buffer: &super::utils::Buffer,
        committed_dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        let producer_release = &contract.producer_release;
        let imported_syncable = VulkanWaylandDmabufSampledImportSyncableProof::new(committed_dmabuf)?;
        let imported_view =
            VulkanWaylandDmabufSampledImportViewProof::new(contract.producer_dmabuf, committed_dmabuf)?;
        let acquire_ordering = unsafe {
            // SAFETY: Forwarded from this method's caller. This proof is bound to the current buffer
            // token and validates the producer evidence/dmabuf identities below.
            VulkanWaylandDmabufSampledImportAcquireOrderingProof::assume_wayland_acquire_orders_loopback_release(
                contract.producer_dmabuf,
                producer_release,
                committed_dmabuf,
                buffer,
            )?
        };
        let renderer_utils_lifecycle = unsafe {
            // SAFETY: Forwarded from this method's caller. This proof is bound to this renderer
            // context and the current buffer token.
            VulkanWaylandDmabufSampledImportRendererUtilsLifecycleProof::assume_renderer_utils_lifecycle(
                self,
                buffer,
                committed_dmabuf,
            )?
        };
        let admission = VulkanWaylandDmabufSampledImportAdmission::from_loopback_evidence(
            contract.producer_dmabuf,
            producer_release,
        )
        .with_imported_syncable(imported_syncable)
        .with_imported_view(imported_view)
        .with_acquire_ordering(acquire_ordering)
        .with_renderer_utils_lifecycle(renderer_utils_lifecycle);

        unsafe {
            // SAFETY: The public contract method's checks and proofs have been assembled above; the
            // remaining external-state/acquire/lifecycle obligations are forwarded from this method's
            // safety contract.
            admission.admit_current_surface_commit(self, surface, committed_dmabuf)
        }
    }

    /// Admit the current dmabuf commit on `surface` from a controlled Vulkan producer release.
    ///
    /// This convenience wrapper derives the current renderer-managed buffer and committed dmabuf from
    /// renderer-utils state, then delegates to
    /// [`VulkanRenderer::admit_wayland_dmabuf_current_commit_from_vulkan_producer_release_for_sampled_import`].
    /// Prefer this method when compositor code only has the surface after the Wayland commit has been
    /// promoted into renderer-utils state. It still keeps sampled dmabuf support scoped to the explicit
    /// controlled-producer `ImportDmaWl` contract and does not public-advertise generic [`ImportDma`].
    ///
    /// # Safety
    ///
    /// The caller must satisfy the same storage identity, no-intervening-transition, acquire-ordering,
    /// surface synchronization, and renderer-utils lifecycle requirements as
    /// [`VulkanRenderer::admit_wayland_dmabuf_current_commit_from_vulkan_producer_release_for_sampled_import`].
    #[cfg(all(feature = "wayland_frontend", feature = "backend_drm"))]
    pub unsafe fn admit_wayland_surface_current_dmabuf_from_vulkan_producer_release_for_sampled_import(
        &self,
        contract: VulkanWaylandDmabufSampledImportProducerContract<'_>,
        surface: &WlSurface,
    ) -> Result<(), VulkanError> {
        let (buffer, committed_dmabuf) = super::utils::with_renderer_surface_state(surface, |state| {
            let buffer = state.buffer().ok_or(VulkanError::MissingCapability(
                "sampled dmabuf producer current buffer",
            ))?;
            let committed_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
                .map_err(|_| VulkanError::UnsupportedOperation("sampled dmabuf producer current buffer"))?;

            Ok((buffer.clone(), committed_dmabuf.clone()))
        })
        .unwrap_or(Err(VulkanError::MissingCapability(
            "sampled dmabuf producer renderer surface state",
        )))?;

        unsafe {
            // SAFETY: This surface-level helper forwards the caller's contract to the stricter
            // buffer-level admission method after deriving the current renderer-managed buffer and
            // committed dmabuf from renderer-utils state.
            self.admit_wayland_dmabuf_current_commit_from_vulkan_producer_release_for_sampled_import(
                contract,
                surface,
                &buffer,
                &committed_dmabuf,
            )
        }
    }

    #[cfg(feature = "wayland_frontend")]
    unsafe fn mark_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import_use(
        &self,
        surface: &WlSurface,
        dmabuf: &Dmabuf,
        use_case: SampledDmabufWaylandExternalStateUse,
        release_generation: Option<u64>,
    ) -> Result<(), VulkanError> {
        Self::with_wayland_surface_current_dmabuf_buffer(surface, dmabuf, |buffer| {
            unsafe {
                // SAFETY: Forwarded from this helper's caller after verifying that the current
                // renderer-managed buffer is the requested dmabuf.
                Self::mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
                    buffer.user_data(),
                    dmabuf,
                    use_case,
                    release_generation,
                )
            }
        })
    }

    #[cfg(feature = "wayland_frontend")]
    fn with_wayland_surface_current_dmabuf_buffer<F, T>(
        surface: &WlSurface,
        dmabuf: &Dmabuf,
        f: F,
    ) -> Result<T, VulkanError>
    where
        F: FnOnce(&super::utils::Buffer) -> Result<T, VulkanError>,
    {
        super::utils::with_renderer_surface_state(surface, |state| {
            let buffer = state.buffer().ok_or(VulkanError::MissingCapability(
                "sampled dmabuf Wayland current buffer",
            ))?;
            let current_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
                .map_err(|_| VulkanError::UnsupportedOperation("sampled dmabuf Wayland current buffer"))?;

            if current_dmabuf != dmabuf {
                return Err(VulkanError::UnsupportedOperation(
                    "sampled dmabuf Wayland current buffer identity",
                ));
            }

            f(buffer)
        })
        .unwrap_or(Err(VulkanError::MissingCapability(
            "sampled dmabuf Wayland renderer surface state",
        )))
    }

    /// Mark the current renderer-managed dmabuf commit on a [`WlSurface`] for validation-stage sampled import.
    ///
    /// Prefer
    /// [`assume_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import`](Self::assume_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import)
    /// so callers explicitly acknowledge the unsafe `FOREIGN + GENERAL` external-state assumption.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the same safety contract as
    /// [`assume_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import`](Self::assume_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import).
    #[cfg(feature = "wayland_frontend")]
    #[deprecated(
        note = "use assume_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import to make the unsafe external-state assumption explicit"
    )]
    pub unsafe fn mark_wayland_surface_current_dmabuf_commit_for_sampled_import(
        &self,
        surface: &WlSurface,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Forwarded from this compatibility wrapper's caller.
            self.assume_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import(
                surface, dmabuf,
            )
        }
    }

    /// Mark a renderer-managed Wayland dmabuf commit using Smithay loopback release evidence.
    ///
    /// This test-only helper keeps the runtime `ImportDmaWl` loopback probes tied to evidence that was
    /// produced by [`release_dmabuf_render_target_for_sampled_loopback`](Self::release_dmabuf_render_target_for_sampled_loopback)
    /// for the same Smithay dmabuf identity. It records typed first-import `FOREIGN + GENERAL`
    /// evidence directly from that controlled release, rather than reclassifying the commit from
    /// renderer history, and it still does not public-advertise generic sampled [`ImportDma`] support.
    ///
    /// # Safety
    ///
    /// The caller must prove there was no intervening access, acquire, release, or layout/ownership
    /// transition of `dmabuf` after `evidence` was produced, and that the Wayland acquire point for
    /// this current commit waits for or otherwise orders `evidence.acquire_sync()` before
    /// `import_surface` can import the buffer.
    #[cfg(all(test, feature = "wayland_frontend"))]
    unsafe fn mark_wayland_surface_current_dmabuf_commit_from_loopback_evidence_for_sampled_import(
        &self,
        surface: &WlSurface,
        dmabuf: &Dmabuf,
        evidence: &VulkanDmabufLoopbackImportEvidence,
    ) -> Result<(), VulkanError> {
        self.validate_dmabuf_loopback_import_evidence(dmabuf, evidence)?;
        unsafe {
            // SAFETY: Forwarded from this test-only helper's caller. The evidence identity check
            // above is only an additional guard; the caller still proves current external state,
            // no-intervening-use, and acquire ordering for this Wayland commit.
            self.mark_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import_use(
                surface,
                dmabuf,
                SampledDmabufWaylandExternalStateUse::FirstImport,
                None,
            )
        }
    }

    /// Produce test-only evidence from this renderer's sampled-dmabuf release history.
    ///
    /// This helper is intentionally limited to tests because renderer-local history only proves what
    /// this renderer released; it does not prove what an arbitrary Wayland producer did before the
    /// next commit. Runtime loopback probes use the returned token together with an explicitly ordered
    /// Wayland reacquire point to replace raw current-commit marking for same-dmabuf reacquire.
    #[cfg(all(test, feature = "wayland_frontend"))]
    fn sampled_dmabuf_wayland_renderer_foreign_general_release_evidence_for_tests(
        &mut self,
        dmabuf: &Dmabuf,
    ) -> Result<SampledDmabufWaylandRendererForeignGeneralReleaseEvidence, VulkanError> {
        match self.sampled_dmabuf_layout_history(dmabuf) {
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => {
                let release_generation = self.sampled_dmabuf_release_generation_snapshot(dmabuf).ok_or(
                    VulkanError::MissingCapability(
                        "sampled dmabuf Wayland Vulkan renderer release generation",
                    ),
                )?;
                Ok(SampledDmabufWaylandRendererForeignGeneralReleaseEvidence::new(
                    dmabuf.weak(),
                    self.context_id.clone(),
                    release_generation,
                ))
            }
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan unreleased local acquire",
            )),
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan renderer release evidence",
            )),
        }
    }

    /// Mark a renderer-managed Wayland dmabuf commit from this renderer's release-history evidence.
    ///
    /// This test-only helper lets the same-dmabuf reacquire loopback probe use release evidence that
    /// the probe obtained immediately after the renderer-utils cache release hook, instead of stale
    /// render-target loopback evidence. It records typed current-reacquire `FOREIGN + GENERAL`
    /// evidence directly from that release-history token and does not public-advertise generic sampled
    /// [`ImportDma`] support.
    ///
    /// # Safety
    ///
    /// The caller must prove the current Wayland acquire point is ordered after the release that
    /// produced `evidence`, and that no foreign producer access changed the dmabuf's queue-family
    /// ownership or image layout before this current commit is imported.
    #[cfg(all(test, feature = "wayland_frontend"))]
    unsafe fn mark_wayland_surface_current_dmabuf_commit_from_renderer_release_evidence_for_sampled_import(
        &self,
        surface: &WlSurface,
        dmabuf: &Dmabuf,
        evidence: &SampledDmabufWaylandRendererForeignGeneralReleaseEvidence,
    ) -> Result<(), VulkanError> {
        if !evidence.is_for_renderer_context(&self.context_id) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland renderer release evidence renderer identity",
            ));
        }
        if !evidence.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland renderer release evidence identity",
            ));
        }
        if !evidence.is_current_release_generation(self.sampled_dmabuf_release_generation_snapshot(dmabuf)) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland renderer release evidence generation",
            ));
        }

        unsafe {
            // SAFETY: Forwarded from this test-only helper's caller. The renderer release evidence
            // proves only this renderer's prior release and identity; the caller still proves current
            // acquire ordering and no intervening external-state change for this Wayland commit.
            self.mark_wayland_surface_current_dmabuf_commit_foreign_general_for_sampled_import_use(
                surface,
                dmabuf,
                SampledDmabufWaylandExternalStateUse::CurrentReacquire,
                Some(evidence.release_generation),
            )
        }
    }

    /// Mark a renderer-managed Wayland dmabuf buffer as covered by texture-cache release call sites.
    ///
    /// This is a development-stage lifecycle evidence API for the normal [`ImportDmaWl`] path. It is
    /// intentionally separate from the external-state marker because the normal `import_surface`
    /// replacement-release point does not by itself prove no-next-import, reset, unmap, destruction, or
    /// renderer teardown release call sites.
    ///
    /// # Safety
    ///
    /// The caller must prove that imports of `buffer` for this exact renderer context go through the
    /// normal renderer-utils surface cache, and that before any no-next-import, reset, unmap, surface
    /// destruction, renderer teardown, or compositor decision to stop importing the surface with this
    /// renderer, the compositor will call
    /// [`super::utils::retire_and_release_surface_textures`] or
    /// [`super::utils::retire_and_release_surface_tree_textures`] while this renderer is still
    /// available. If renderer-utils has already retired the texture for this renderer context, such as
    /// after a removed-buffer commit or replacement update, the compositor may instead call
    /// [`super::utils::release_retired_surface_textures`] before the retired texture is dropped. If
    /// release returns a retry-safe error, the caller must not reset/drop the affected surface state
    /// until the obligation is retried or otherwise preserved.
    #[cfg(feature = "wayland_frontend")]
    pub unsafe fn mark_wayland_dmabuf_texture_cache_release_lifecycle_for_sampled_import(
        &self,
        buffer: &super::utils::Buffer,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Forwarded from this explicit lifecycle-marking helper's caller.
            self.mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
                buffer.user_data(),
                dmabuf,
            )
        }
    }

    /// Mark the current renderer-managed dmabuf commit on a [`WlSurface`] as covered by texture-cache
    /// release call sites.
    ///
    /// This is the surface-scoped companion to
    /// [`mark_wayland_dmabuf_texture_cache_release_lifecycle_for_sampled_import`](Self::mark_wayland_dmabuf_texture_cache_release_lifecycle_for_sampled_import)
    /// for compositors using Smithay's normal [`super::utils::on_commit_buffer_handler`] path. It
    /// looks up the surface's current renderer-managed buffer, verifies that it is the same dmabuf
    /// identity as `dmabuf`, and records lifecycle evidence on that current buffer wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the same lifecycle contract as
    /// [`mark_wayland_dmabuf_texture_cache_release_lifecycle_for_sampled_import`](Self::mark_wayland_dmabuf_texture_cache_release_lifecycle_for_sampled_import)
    /// for the surface's current renderer-managed buffer.
    #[cfg(feature = "wayland_frontend")]
    pub unsafe fn mark_wayland_surface_current_dmabuf_commit_texture_cache_release_lifecycle_for_sampled_import(
        &self,
        surface: &WlSurface,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        Self::with_wayland_surface_current_dmabuf_buffer(surface, dmabuf, |buffer| unsafe {
            // SAFETY: Forwarded from this surface-scoped lifecycle helper's caller after verifying the
            // current renderer-managed buffer is the requested dmabuf.
            self.mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
                buffer.user_data(),
                dmabuf,
            )
        })
    }

    #[cfg(feature = "wayland_frontend")]
    unsafe fn mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
        user_data: &UserDataMap,
        dmabuf: &Dmabuf,
        use_case: SampledDmabufWaylandExternalStateUse,
        release_generation: Option<u64>,
    ) -> Result<(), VulkanError> {
        let slot =
            user_data.get_or_insert_threadsafe(SampledDmabufWaylandForeignGeneralEvidenceSlot::default);
        let commit_token_slot =
            user_data.get_or_insert_threadsafe(SampledDmabufWaylandCommitTokenSlot::default);
        let mut evidence = slot.evidence.lock().map_err(|_| {
            VulkanError::UnsupportedOperation("sampled dmabuf Wayland external-state evidence")
        })?;
        *evidence = Some(unsafe {
            // SAFETY: Forwarded from this unsafe evidence-marking helper's caller.
            SampledDmabufWaylandForeignGeneralEvidence::new(
                dmabuf.weak(),
                use_case,
                Arc::downgrade(&commit_token_slot.token),
                release_generation,
            )
        });
        Ok(())
    }

    #[cfg(feature = "wayland_frontend")]
    unsafe fn mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
        &self,
        user_data: &UserDataMap,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        let slot =
            user_data.get_or_insert_threadsafe(SampledDmabufWaylandTextureCacheReleaseLifecycleSlot::default);
        let commit_token_slot =
            user_data.get_or_insert_threadsafe(SampledDmabufWaylandCommitTokenSlot::default);
        let mut evidence = slot.evidence.lock().map_err(|_| {
            VulkanError::UnsupportedOperation("sampled dmabuf Wayland texture-cache release lifecycle")
        })?;
        *evidence = Some(unsafe {
            // SAFETY: Forwarded from this unsafe lifecycle-marking helper's caller.
            SampledDmabufWaylandTextureCacheReleaseLifecycleEvidence::new(
                dmabuf.weak(),
                self.context_id.clone(),
                Arc::downgrade(&commit_token_slot.token),
            )
        });
        Ok(())
    }

    /// Returns the uninitialized/default capabilities without constructing a renderer.
    ///
    /// All capability bits are false and all format/extension sets are empty until a renderer is
    /// initialized with an explicit [`PhysicalDevice`].
    pub fn scaffold_capabilities() -> VulkanRendererCapabilities {
        VulkanRendererCapabilities::default()
    }

    #[cfg(test)]
    fn new_scaffold_for_tests() -> Self {
        Self {
            context_id: ContextId::new(),
            debug_flags: DebugFlags::empty(),
            downscale_filter: TextureFilter::Linear,
            upscale_filter: TextureFilter::Linear,
            capabilities: VulkanRendererCapabilities::default(),
            device: None,
            sampled_dmabuf_layout_history: HashMap::new(),
            pending_sampled_dmabuf_import_obligations: Vec::new(),
            wayland_linux_dmabuf_interop: false,
        }
    }

    /// Returns the discovered Vulkan renderer capabilities.
    ///
    /// Uninitialized test renderers report all capability bits as false and all format/extension
    /// sets as empty. Renderers built with an explicit [`PhysicalDevice`] report discovered device
    /// and format capabilities.
    pub fn capabilities(&self) -> &VulkanRendererCapabilities {
        &self.capabilities
    }

    /// Returns whether Vulkan device state was initialized.
    ///
    /// This does not imply that rendering, import, export, or presentation operations are supported.
    pub fn is_device_initialized(&self) -> bool {
        self.device.is_some() && self.capabilities.device.available
    }

    fn render_target_formats(&self) -> FormatSet {
        self.capabilities.formats.render_target_formats()
    }

    fn development_gated_dmabuf_render_target_formats(&self) -> FormatSet {
        if self.capabilities.rendering.dmabuf_target_development {
            self.capabilities.formats.dmabuf_render_target.clone()
        } else {
            FormatSet::default()
        }
    }

    fn public_dmabuf_render_target_formats(&self) -> FormatSet {
        if self.capabilities.rendering.dmabuf_targets {
            self.capabilities.formats.dmabuf_render_target.clone()
        } else {
            FormatSet::default()
        }
    }

    fn public_dmabuf_import_formats(&self) -> FormatSet {
        if self
            .validate_sampled_dmabuf_public_advertisement_contract()
            .is_ok()
        {
            self.capabilities.formats.dmabuf_import.clone()
        } else {
            FormatSet::default()
        }
    }

    #[allow(dead_code)]
    fn prune_sampled_dmabuf_layout_history(&mut self) {
        self.sampled_dmabuf_layout_history
            .retain(|dmabuf, _| !dmabuf.is_gone());
    }

    #[allow(dead_code)]
    fn sampled_dmabuf_layout_history(&mut self, dmabuf: &Dmabuf) -> SampledDmabufWaylandLayoutHistory {
        self.prune_sampled_dmabuf_layout_history();
        self.sampled_dmabuf_layout_history_snapshot(dmabuf)
    }

    fn sampled_dmabuf_layout_history_snapshot(&self, dmabuf: &Dmabuf) -> SampledDmabufWaylandLayoutHistory {
        if self
            .pending_sampled_dmabuf_import_obligations
            .iter()
            .any(|obligation| obligation.is_for_dmabuf(dmabuf) && obligation.may_still_own_image_locally())
        {
            return SampledDmabufWaylandLayoutHistory::LocallyAcquired;
        }
        self.sampled_dmabuf_layout_history
            .get(&dmabuf.weak())
            .map(|record| record.history)
            .unwrap_or(SampledDmabufWaylandLayoutHistory::NoRendererHistory)
    }

    #[allow(dead_code)]
    fn sampled_dmabuf_release_generation_snapshot(&self, dmabuf: &Dmabuf) -> Option<u64> {
        if self.sampled_dmabuf_layout_history_snapshot(dmabuf)
            != SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral
        {
            return None;
        }
        self.sampled_dmabuf_layout_history
            .get(&dmabuf.weak())
            .and_then(|record| {
                (record.history == SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral)
                    .then_some(record.release_generation)
            })
    }

    #[allow(dead_code)]
    fn validate_no_pending_sampled_dmabuf_import_obligation(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        if self
            .pending_sampled_dmabuf_import_obligations
            .iter()
            .any(|obligation| obligation.is_for_dmabuf(dmabuf))
        {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf pending import obligation",
            ));
        }

        Ok(())
    }

    /// Public sampled-import lifecycle: pending obligations and outstanding local ownership both
    /// block another acquire of the same dmabuf.
    fn validate_sampled_dmabuf_public_import_lifecycle(&self, dmabuf: &Dmabuf) -> Result<(), VulkanError> {
        self.validate_no_pending_sampled_dmabuf_import_obligation(dmabuf)?;
        if self.sampled_dmabuf_layout_history_snapshot(dmabuf)
            == SampledDmabufWaylandLayoutHistory::LocallyAcquired
        {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf locally acquired",
            ));
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn retain_pending_sampled_dmabuf_import_obligation(
        &mut self,
        obligation: PendingSampledDmabufImportObligation,
    ) {
        if let PendingSampledDmabufImportObligation::AcquiredTexture(texture)
        | PendingSampledDmabufImportObligation::ReleaseCompletionUnknownTexture(texture) = &obligation
        {
            if let Some(dmabuf) = texture.sampled_dmabuf.as_ref().and_then(WeakDmabuf::upgrade) {
                self.record_sampled_dmabuf_locally_acquired(&dmabuf);
            }
        }

        self.pending_sampled_dmabuf_import_obligations.push(obligation);
    }

    #[allow(dead_code)]
    fn cleanup_pending_sampled_dmabuf_import_obligations(&mut self) -> Result<(), VulkanError> {
        let mut obligations = std::mem::take(&mut self.pending_sampled_dmabuf_import_obligations).into_iter();
        while let Some(obligation) = obligations.next() {
            match obligation {
                PendingSampledDmabufImportObligation::ReleaseOnly(release_ownership) => {
                    if let Err(err) = release_ownership.signal_wayland_release_once() {
                        self.retain_pending_sampled_dmabuf_import_obligation(
                            PendingSampledDmabufImportObligation::ReleaseOnly(release_ownership),
                        );
                        self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                        return Err(err);
                    }
                }
                PendingSampledDmabufImportObligation::AcquiredTexture(mut texture) => {
                    if texture.context_id != self.context_id {
                        self.retain_pending_sampled_dmabuf_import_obligation(
                            PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                        );
                        self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                        return Err(VulkanError::UnsupportedOperation("foreign dmabuf texture"));
                    }
                    if texture.image.source != image::VulkanImageSource::DmabufImport {
                        self.retain_pending_sampled_dmabuf_import_obligation(
                            PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                        );
                        self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                        return Err(VulkanError::UnsupportedOperation("dmabuf texture"));
                    }
                    let Some(dmabuf) = texture.sampled_dmabuf.as_ref().and_then(WeakDmabuf::upgrade) else {
                        self.retain_pending_sampled_dmabuf_import_obligation(
                            PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                        );
                        self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                        return Err(VulkanError::UnsupportedOperation("sampled dmabuf identity"));
                    };
                    if texture.sampled_image.is_none() {
                        self.retain_pending_sampled_dmabuf_import_obligation(
                            PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                        );
                        self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                        return Err(VulkanError::UnsupportedOperation("dmabuf texture sampled image"));
                    }
                    let cleanup = {
                        let Some(device) = self.device.as_ref() else {
                            self.retain_pending_sampled_dmabuf_import_obligation(
                                PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                            );
                            self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                            return Err(VulkanError::VulkanUnavailable);
                        };
                        let sampled_image = texture
                            .sampled_image
                            .as_ref()
                            .expect("sampled image checked before pending cleanup release");
                        device.release_sampled_dmabuf_to_foreign_general_classified(
                            sampled_image.image(),
                            false,
                        )
                    };

                    match cleanup {
                        Ok((true, _)) => {}
                        Ok((false, _)) => {
                            self.retain_pending_sampled_dmabuf_import_obligation(
                                PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                            );
                            self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                            return Err(VulkanError::UnsupportedOperation(
                                "sampled dmabuf pending import cleanup release",
                            ));
                        }
                        Err(device::VulkanSampledDmabufForeignReleaseError::RetrySafe(err)) => {
                            self.retain_pending_sampled_dmabuf_import_obligation(
                                PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                            );
                            self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                            return Err(err);
                        }
                        Err(device::VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(err)) => {
                            self.retain_pending_sampled_dmabuf_import_obligation(
                                PendingSampledDmabufImportObligation::ReleaseCompletionUnknownTexture(
                                    texture,
                                ),
                            );
                            self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                            return Err(err);
                        }
                    }
                    if let Err(err) = texture.signal_sampled_dmabuf_release_point(None) {
                        if let Some(release) = texture.sampled_dmabuf_release.take() {
                            self.retain_pending_sampled_dmabuf_import_obligation(
                                PendingSampledDmabufImportObligation::ReleaseOnly(
                                    SampledDmabufReleaseOwnership {
                                        dmabuf: dmabuf.weak(),
                                        release,
                                    },
                                ),
                            );
                        }
                        self.record_sampled_dmabuf_released_to_foreign_general(&dmabuf);
                        self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                        return Err(err);
                    }
                    self.record_sampled_dmabuf_released_to_foreign_general(&dmabuf);
                }
                PendingSampledDmabufImportObligation::ReleaseCompletionUnknownTexture(texture) => {
                    self.retain_pending_sampled_dmabuf_import_obligation(
                        PendingSampledDmabufImportObligation::ReleaseCompletionUnknownTexture(texture),
                    );
                    self.pending_sampled_dmabuf_import_obligations.extend(obligations);
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf release completion unknown",
                    ));
                }
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn record_sampled_dmabuf_released_to_foreign_general(&mut self, dmabuf: &Dmabuf) {
        self.prune_sampled_dmabuf_layout_history();
        let release_generation = self
            .sampled_dmabuf_layout_history
            .get(&dmabuf.weak())
            .map(|record| {
                record
                    .release_generation
                    .checked_add(1)
                    .expect("sampled dmabuf release generation overflow")
            })
            .unwrap_or(1);
        self.sampled_dmabuf_layout_history.insert(
            dmabuf.weak(),
            SampledDmabufWaylandLayoutHistoryRecord {
                history: SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
                release_generation,
            },
        );
    }

    #[allow(dead_code)]
    fn record_sampled_dmabuf_locally_acquired(&mut self, dmabuf: &Dmabuf) {
        self.prune_sampled_dmabuf_layout_history();
        let release_generation = self
            .sampled_dmabuf_layout_history
            .get(&dmabuf.weak())
            .map(|record| record.release_generation)
            .unwrap_or(0);
        self.sampled_dmabuf_layout_history.insert(
            dmabuf.weak(),
            SampledDmabufWaylandLayoutHistoryRecord {
                history: SampledDmabufWaylandLayoutHistory::LocallyAcquired,
                release_generation,
            },
        );
    }

    #[allow(dead_code)]
    fn sampled_dmabuf_wayland_texture_cache_release_hook(
        &self,
        dmabuf: &Dmabuf,
    ) -> SampledDmabufWaylandTextureCacheReleaseHook {
        SampledDmabufWaylandTextureCacheReleaseHook {
            dmabuf: dmabuf.weak(),
            renderer_context: Some(self.context_id.clone()),
        }
    }

    #[allow(dead_code)]
    fn validate_dmabuf_loopback_import_evidence(
        &self,
        dmabuf: &Dmabuf,
        evidence: &VulkanDmabufLoopbackImportEvidence,
    ) -> Result<SampledDmabufKnownLayoutEvidence, VulkanError> {
        if !evidence.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation("dmabuf loopback evidence"));
        }

        Ok(evidence.foreign_general.clone())
    }

    /// Check whether the sampled dmabuf path may be public-advertised through [`ImportDma`].
    ///
    /// This is intentionally stricter than raw Vulkan probing. Future implementation work should
    /// make this pass only after metadata validation, acquire synchronization, ownership/layout
    /// transitions, sampled rendering, release synchronization, and tests all pass through the normal
    /// Smithay renderer path.
    fn validate_sampled_dmabuf_public_advertisement_contract(&self) -> Result<(), VulkanError> {
        self.sampled_dmabuf_public_import_contracts().validate()
    }

    /// Gather public sampled-dmabuf [`ImportDma`] readiness without advertising it.
    ///
    /// Raw Vulkan probing may populate format records used by validation tests and development paths.
    /// The public external-state policy is the explicit [`VulkanSampledDmabufImport`] wrapper: a raw
    /// [`Dmabuf`] still cannot prove layout or ownership. The public lifecycle is local ownership
    /// until foreign-GENERAL release; the same dmabuf cannot be imported again while outstanding.
    /// Generic [`ImportDma`] stays fail-closed until a direct generic implementation exists.
    fn sampled_dmabuf_public_import_contracts(&self) -> SampledDmabufPublicImportContracts {
        SampledDmabufPublicImportContracts {
            raw_import_capability: self.capabilities.import.dmabuf,
            advertised_formats: self.capabilities.formats.dmabuf_import.iter().next().is_some(),
            public_external_state_policy: true,
            public_import_lifecycle: true,
            public_import_implementation: false,
        }
    }

    /// Check whether the normal sampled-dmabuf external-state policy is ready for public import.
    ///
    /// The policy is the explicit [`VulkanSampledDmabufImport`] wrapper. Raw probed formats still
    /// cannot prove layout or ownership for generic [`ImportDma`]. The implementation gate remains
    /// closed.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_public_external_state_contract(&self) -> Result<(), VulkanError> {
        if self
            .sampled_dmabuf_public_import_contracts()
            .public_external_state_policy
        {
            Ok(())
        } else {
            Err(VulkanError::MissingCapability(
                "sampled dmabuf public external-state policy",
            ))
        }
    }

    /// Check whether the public sampled-dmabuf import lifecycle is ready.
    ///
    /// After [`VulkanRenderer::import_sampled_dmabuf`], this renderer owns the image locally until
    /// [`VulkanRenderer::release_imported_dmabuf_texture_to_foreign_general_sync_point`]. The same
    /// dmabuf cannot be imported again while that ownership is outstanding. Generic [`ImportDma`]
    /// still has no place to put that pair.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_public_import_lifecycle_contract(&self) -> Result<(), VulkanError> {
        if self
            .sampled_dmabuf_public_import_contracts()
            .public_import_lifecycle
        {
            Ok(())
        } else {
            Err(VulkanError::MissingCapability("sampled dmabuf import lifecycle"))
        }
    }

    /// Convert Wayland explicit-sync state into the renderer sync-point contract used by Vulkan.
    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_wayland_acquire_sync_evidence(
        &self,
        dmabuf: &Dmabuf,
        #[cfg(feature = "backend_drm")] buffer: &super::utils::Buffer,
        #[cfg(not(feature = "backend_drm"))] _buffer: &super::utils::Buffer,
    ) -> Result<SampledDmabufAcquireSyncEvidence, VulkanError> {
        #[cfg(feature = "backend_drm")]
        {
            let current_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
                .map_err(|_| VulkanError::UnsupportedOperation("sampled dmabuf acquire sync buffer"))?;
            if current_dmabuf != dmabuf {
                return Err(VulkanError::UnsupportedOperation(
                    "sampled dmabuf acquire sync buffer identity",
                ));
            }
            let acquire_sync = match buffer.acquire_point().cloned().map(SyncPoint::from) {
                Some(sync) => Some(sync),
                None => match dmabuf.export_sync_file(0, DmabufSyncFlags::READ) {
                    Ok(fd) => Some(sync_point_from_sync_file(Some(fd))),
                    Err(_) => None,
                },
            };
            self.validate_sampled_dmabuf_wayland_acquire_sync_contract(acquire_sync.as_ref())?;
            let commit_token_slot = buffer
                .user_data()
                .get_or_insert_threadsafe(SampledDmabufWaylandCommitTokenSlot::default);
            acquire_sync
                .map(|sync| {
                    SampledDmabufAcquireSyncEvidence::new_with_commit_token(
                        dmabuf,
                        sync,
                        Arc::downgrade(&commit_token_slot.token),
                    )
                })
                .ok_or(VulkanError::NotPublicAdvertised("sampled dmabuf implicit sync"))
        }

        #[cfg(not(feature = "backend_drm"))]
        {
            let _ = dmabuf;
            Err(VulkanError::MissingCapability(
                "sampled dmabuf explicit sync contract",
            ))
        }
    }

    /// Extract the Wayland release point needed for the sampled-dmabuf release lifecycle.
    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_wayland_release_evidence(
        &self,
        dmabuf: &Dmabuf,
        #[cfg(feature = "backend_drm")] buffer: &super::utils::Buffer,
        #[cfg(not(feature = "backend_drm"))] _buffer: &super::utils::Buffer,
    ) -> Result<SampledDmabufReleaseEvidence, VulkanError> {
        #[cfg(feature = "backend_drm")]
        {
            let current_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
                .map_err(|_| VulkanError::UnsupportedOperation("sampled dmabuf release point buffer"))?;
            if current_dmabuf != dmabuf {
                return Err(VulkanError::UnsupportedOperation(
                    "sampled dmabuf release point buffer identity",
                ));
            }
            if buffer.release_point().is_none() && !self.wayland_linux_dmabuf_interop {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf release point contract",
                ));
            }
            let commit_token_slot = buffer
                .user_data()
                .get_or_insert_threadsafe(SampledDmabufWaylandCommitTokenSlot::default);

            Ok(SampledDmabufReleaseEvidence {
                dmabuf: dmabuf.weak(),
                commit_token: Some(Arc::downgrade(&commit_token_slot.token)),
            })
        }

        #[cfg(not(feature = "backend_drm"))]
        {
            let _ = dmabuf;
            Err(VulkanError::MissingCapability(
                "sampled dmabuf release point contract",
            ))
        }
    }

    /// Validate the development-stage sampled dmabuf import subset before any Vulkan object work.
    ///
    /// This is intentionally separate from public [`ImportDma`] advertisement. It models the next
    /// intended path in small fail-closed steps so future work has a clear continuation point:
    ///
    /// 1. validate dmabuf metadata and raw Vulkan modifier capability,
    /// 2. consume Wayland explicit acquire synchronization from the surface-state buffer,
    /// 3. require/provide the missing external ownership and image-layout contract,
    /// 4. release the sampled image back to foreign ownership only after Vulkan sampling is done,
    /// 5. public-advertise [`ImportDma`] formats only after the whole lifecycle is covered.
    fn validate_sampled_dmabuf_import_metadata(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<image::VulkanDmabufImportState, VulkanError> {
        let import = image::VulkanDmabufImportState::from_dmabuf(dmabuf)?;

        if import.plane_count() != 1 {
            return Err(VulkanError::UnsupportedOperation("sampled dmabuf planes"));
        }
        if import.modifier() == Modifier::Invalid {
            return Err(VulkanError::MissingCapability("sampled dmabuf explicit modifier"));
        }
        get_format_info(import.format())?;

        if !self
            .capabilities
            .formats
            .has_sampled_dmabuf_modifier_record(&import)
        {
            return Err(VulkanError::MissingCapability("sampled dmabuf format/modifier"));
        }

        Ok(import)
    }

    /// Validate the Wayland acquire synchronization part of sampled dmabuf import.
    ///
    /// `linux-dmabuf` alone implies implicit synchronization. The Wayland acquire-evidence helper
    /// may export a dma-buf read fence as a [`SyncPoint`] in that case. This contract still requires
    /// a fence-bearing [`SyncPoint`]; a missing fence is not treated as ready.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_acquire_sync_contract(
        &self,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<(), VulkanError> {
        if acquire_sync.is_some_and(SyncPoint::contains_fence) {
            Ok(())
        } else {
            Err(VulkanError::NotPublicAdvertised("sampled dmabuf implicit sync"))
        }
    }

    /// Validate that the Wayland commit provided a release point for sampled dmabuf import.
    ///
    /// The presence of a release point is only evidence that the compositor has a protocol object to
    /// satisfy later. It does not by itself make the sampled dmabuf release lifecycle complete.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_release_point_contract(
        &self,
        dmabuf: &Dmabuf,
        has_release_point: bool,
    ) -> Result<SampledDmabufReleaseEvidence, VulkanError> {
        if has_release_point {
            Ok(SampledDmabufReleaseEvidence {
                dmabuf: dmabuf.weak(),
                commit_token: None,
            })
        } else {
            Err(VulkanError::MissingCapability(
                "sampled dmabuf release point contract",
            ))
        }
    }

    /// Validate Smithay's own Wayland/Vulkan sampled-dmabuf interop policy.
    ///
    /// `linux-dmabuf` metadata plus explicit acquire/release sync is enough to identify memory,
    /// format/modifier, and producer/compositor ordering, but it is not by itself a Vulkan
    /// queue-family ownership or image-layout contract. This guard is the named development-gated
    /// continuation point for the normal `ImportDmaWl` path. Future implementation must replace
    /// this fail-closed marker with a Smithay-owned policy that defines how initial import,
    /// subsequent reacquire, sampling, release, and texture-cache reuse map onto Vulkan external
    /// image state.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_vulkan_interop_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufLayoutEvidence, VulkanError> {
        let contracts = self.sampled_dmabuf_wayland_vulkan_interop_policy_contracts(context)?;
        self.validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(context.dmabuf, &contracts)
            .map(SampledDmabufLayoutEvidence::SmithayWaylandVulkanPolicy)
    }

    /// Assemble all currently modeled Smithay Wayland/Vulkan sampled-dmabuf policy contracts.
    ///
    /// This helper deliberately preserves the staged fail-closed order used by the normal
    /// `ImportDmaWl` path. It does not construct public `ImportDma` support; it only gathers the
    /// validation-stage contract tokens needed before the aggregate policy evidence can be produced.
    #[allow(dead_code)]
    fn sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandVulkanInteropPolicyContracts, VulkanError> {
        self.validate_sampled_dmabuf_wayland_context_metadata(context)?;
        Ok(SampledDmabufWaylandVulkanInteropPolicyContracts {
            layout: Some(self.validate_sampled_dmabuf_wayland_layout_policy(context)?),
            foreign_general: Some(self.validate_sampled_dmabuf_wayland_foreign_general_policy(context)?),
            queue_family_transfer: Some(self.validate_sampled_dmabuf_wayland_queue_family_policy(context)?),
            acquire_sync: Some(self.validate_sampled_dmabuf_wayland_acquire_sync_policy(context)?),
            release_sync: Some(self.validate_sampled_dmabuf_wayland_release_sync_policy(context)?),
            texture_cache_reuse: Some(self.validate_sampled_dmabuf_wayland_texture_cache_policy(context)?),
            release_ownership: Some(self.validate_sampled_dmabuf_wayland_release_ownership_policy(context)?),
        })
    }

    /// Validate that the policy context's parsed import metadata still describes the current dmabuf.
    ///
    /// The policy context intentionally carries both the Smithay dmabuf identity and the parsed Vulkan
    /// import metadata. Future policy producers must not reuse metadata across commits or dmabuf
    /// identities while constructing layout/ownership evidence.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_context_metadata(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<(), VulkanError> {
        let current_import = image::VulkanDmabufImportState::from_dmabuf(context.dmabuf)?;
        if &current_import != context.import {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland import metadata",
            ));
        }

        Ok(())
    }

    /// Validate the external image layout policy for this normal Wayland dmabuf commit.
    ///
    /// First imports and reacquires are alternative states. A dmabuf with no renderer-local history
    /// must satisfy the first-import policy; a dmabuf this renderer previously released to foreign
    /// GENERAL ownership may satisfy the narrower reacquire policy instead.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_layout_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandLayoutPolicy, VulkanError> {
        match context.layout_history {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => self
                .validate_sampled_dmabuf_wayland_first_import_layout_policy(context)
                .map(SampledDmabufWaylandLayoutPolicy::FirstImport),
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan unreleased local acquire",
            )),
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => self
                .validate_sampled_dmabuf_wayland_reacquire_layout_policy(context)
                .map(SampledDmabufWaylandLayoutPolicy::Reacquire),
        }
    }

    /// Validate that Smithay's Wayland/Vulkan policy has specifically established foreign ownership
    /// and `VK_IMAGE_LAYOUT_GENERAL` for the current sampled-dmabuf acquire.
    ///
    /// This is intentionally separate from the higher-level layout-path token. The validation-stage
    /// import helper uses a `FOREIGN` -> local acquire barrier with `GENERAL` as `oldLayout`; a future
    /// policy that selects a different initial layout must use a matching helper instead of satisfying
    /// this guard.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_foreign_general_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufKnownLayoutEvidence, VulkanError> {
        match context.layout_history {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => {
                self.validate_sampled_dmabuf_wayland_first_import_layout_policy(context)?;
                let evidence =
                    context
                        .first_import_foreign_general
                        .as_ref()
                        .ok_or(VulkanError::MissingCapability(
                            "sampled dmabuf Wayland Vulkan foreign GENERAL policy",
                        ))?;
                if !evidence.is_for_dmabuf(context.dmabuf) {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland foreign GENERAL identity",
                    ));
                }
                if !evidence.has_foreign_general_state() {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland foreign GENERAL state",
                    ));
                }

                Ok(evidence.clone())
            }
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan unreleased local acquire",
            )),
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => {
                self.validate_sampled_dmabuf_wayland_current_reacquire_layout_policy(context)?;
                let evidence = context.current_reacquire_foreign_general.as_ref().ok_or(
                    VulkanError::MissingCapability("sampled dmabuf Wayland Vulkan foreign GENERAL policy"),
                )?;
                if !evidence.is_for_dmabuf(context.dmabuf) {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland foreign GENERAL identity",
                    ));
                }
                if !evidence.has_foreign_general_state() {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland foreign GENERAL state",
                    ));
                }

                Ok(evidence.clone())
            }
        }
    }

    /// Validate the first-import external image layout policy for a normal Wayland dmabuf.
    ///
    /// This is the first Smithay-owned policy item that must be implemented before the normal
    /// `ImportDmaWl` path can acquire an arbitrary client dmabuf as a sampled Vulkan image.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_first_import_layout_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandFirstImportLayoutPolicy, VulkanError> {
        match context.layout_history {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => {}
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland Vulkan unreleased local acquire",
                ));
            }
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland Vulkan first-import layout history",
                ));
            }
        }

        let evidence = context
            .first_import_layout
            .as_ref()
            .ok_or(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan first-import layout policy",
            ))?;
        if !evidence.is_for_dmabuf(context.dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland first-import identity",
            ));
        }

        Ok(SampledDmabufWaylandFirstImportLayoutPolicy {
            dmabuf: context.dmabuf.weak(),
        })
    }

    /// Locate first-import layout/ownership evidence for a normal Wayland dmabuf.
    ///
    /// A dmabuf with no renderer-local history needs a Smithay-owned policy for the imported image's
    /// initial external ownership and layout before Vulkan can acquire it. `linux-dmabuf` metadata and
    /// explicit acquire sync do not provide that proof, so production `ImportDmaWl` remains
    /// development-gated at this evidence source for first imports.
    #[allow(dead_code)]
    fn sampled_dmabuf_wayland_first_import_layout_evidence(
        &self,
        dmabuf: &Dmabuf,
        layout_history: SampledDmabufWaylandLayoutHistory,
        external_state: Option<&SampledDmabufWaylandForeignGeneralEvidence>,
    ) -> Result<Option<SampledDmabufWaylandFirstImportLayoutEvidence>, VulkanError> {
        match layout_history {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => external_state
                .map(|evidence| evidence.first_import_layout(dmabuf))
                .transpose()?
                .ok_or(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland Vulkan first-import layout policy",
                ))
                .map(Some),
            SampledDmabufWaylandLayoutHistory::LocallyAcquired
            | SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => Ok(None),
        }
    }

    /// Locate first-import proof that the imported dmabuf is specifically in foreign ownership with
    /// `VK_IMAGE_LAYOUT_GENERAL`.
    ///
    /// The validation-stage sampled import helper uses `FOREIGN` + `GENERAL` as the known external
    /// state. A future first-import policy may either provide this evidence or use a different helper
    /// matching a different external state. Production remains development-gated here for first
    /// imports; the guard prevents treating first-import layout evidence as enough by itself.
    #[allow(dead_code)]
    fn sampled_dmabuf_wayland_first_import_foreign_general_evidence(
        &self,
        dmabuf: &Dmabuf,
        layout_history: SampledDmabufWaylandLayoutHistory,
        external_state: Option<&SampledDmabufWaylandForeignGeneralEvidence>,
    ) -> Result<Option<SampledDmabufKnownLayoutEvidence>, VulkanError> {
        match layout_history {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => external_state
                .map(|evidence| evidence.first_import_known_foreign_general(dmabuf))
                .transpose()?
                .ok_or(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland Vulkan foreign GENERAL policy",
                ))
                .map(Some),
            SampledDmabufWaylandLayoutHistory::LocallyAcquired
            | SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => Ok(None),
        }
    }

    /// Validate that the current Wayland producer returned a reacquired dmabuf in the layout and
    /// ownership required by the known-GENERAL Vulkan acquire path.
    ///
    /// Renderer-local release history is only prior-state evidence. The current commit needs its own
    /// Smithay-owned contract before reacquire may use the same `FOREIGN` + `GENERAL` unsafe helper.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_current_reacquire_layout_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandCurrentReacquireLayoutEvidence, VulkanError> {
        let evidence = context
            .current_reacquire_layout
            .as_ref()
            .ok_or(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan current reacquire layout policy",
            ))?;
        if !evidence.is_for_dmabuf(context.dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland current reacquire identity",
            ));
        }
        evidence
            .validate_release_generation(self.sampled_dmabuf_release_generation_snapshot(context.dmabuf))?;

        Ok(evidence.clone())
    }

    /// Locate current-commit producer-return evidence for reacquiring a normal Wayland dmabuf.
    ///
    /// Renderer-local history records what this renderer previously released; it does not prove what
    /// the Wayland producer returned on this commit. Until Smithay has a concrete current-commit
    /// layout/ownership contract, production `ImportDmaWl` keeps the intended reacquire path
    /// development-gated at this exact evidence source.
    #[allow(dead_code)]
    fn sampled_dmabuf_wayland_current_reacquire_layout_evidence(
        &self,
        dmabuf: &Dmabuf,
        layout_history: SampledDmabufWaylandLayoutHistory,
        external_state: Option<&SampledDmabufWaylandForeignGeneralEvidence>,
    ) -> Result<Option<SampledDmabufWaylandCurrentReacquireLayoutEvidence>, VulkanError> {
        match layout_history {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => Ok(None),
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan unreleased local acquire",
            )),
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => external_state
                .map(|evidence| {
                    evidence.validate_release_generation(
                        self.sampled_dmabuf_release_generation_snapshot(dmabuf),
                    )?;
                    evidence.current_reacquire_layout(dmabuf)
                })
                .transpose()?
                .ok_or(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland Vulkan current reacquire layout policy",
                ))
                .map(Some),
        }
    }

    /// Locate reacquire proof that the current Wayland producer returned the dmabuf in foreign
    /// ownership with `VK_IMAGE_LAYOUT_GENERAL`.
    ///
    /// This is the known-state source used by the validation-stage sampled import helper for
    /// reacquires. It is intentionally separate from renderer-local release history: the current
    /// commit must carry same-dmabuf current-return evidence before this helper can produce the
    /// known-layout token consumed by the Vulkan acquire path.
    #[allow(dead_code)]
    fn sampled_dmabuf_wayland_current_reacquire_foreign_general_evidence(
        &self,
        dmabuf: &Dmabuf,
        layout_history: SampledDmabufWaylandLayoutHistory,
        current_reacquire_layout: Option<&SampledDmabufWaylandCurrentReacquireLayoutEvidence>,
    ) -> Result<Option<SampledDmabufKnownLayoutEvidence>, VulkanError> {
        match layout_history {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => Ok(None),
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan unreleased local acquire",
            )),
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => {
                let evidence = current_reacquire_layout.ok_or(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland Vulkan current reacquire layout policy",
                ))?;
                if !evidence.is_for_dmabuf(dmabuf) {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland current reacquire identity",
                    ));
                }
                evidence
                    .validate_release_generation(self.sampled_dmabuf_release_generation_snapshot(dmabuf))?;

                Ok(Some(unsafe {
                    // SAFETY: The same-dmabuf current-reacquire layout evidence is the
                    // validation-stage contract that the current producer returned this dmabuf to
                    // FOREIGN ownership in GENERAL layout after this renderer's prior release.
                    SampledDmabufKnownLayoutEvidence::foreign_general(dmabuf.weak())
                }))
            }
        }
    }

    /// Read commit-local Wayland/Vulkan external-state evidence from the renderer buffer wrapper.
    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_wayland_buffer_foreign_general_evidence(
        &self,
        buffer: &super::utils::Buffer,
        dmabuf: &Dmabuf,
    ) -> Result<Option<SampledDmabufWaylandForeignGeneralEvidence>, VulkanError> {
        if let Some(evidence) =
            self.sampled_dmabuf_wayland_user_data_foreign_general_evidence(buffer.user_data(), dmabuf)?
        {
            return Ok(Some(evidence));
        }

        Ok(None)
    }

    /// Read explicit texture-cache release lifecycle evidence from the renderer buffer wrapper.
    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_wayland_buffer_texture_cache_release_lifecycle(
        &self,
        buffer: &super::utils::Buffer,
        dmabuf: &Dmabuf,
    ) -> Result<Option<SampledDmabufWaylandTextureCacheReleaseLifecycle>, VulkanError> {
        self.sampled_dmabuf_wayland_user_data_texture_cache_release_lifecycle(buffer.user_data(), dmabuf)
    }

    /// Read commit-local Wayland/Vulkan external-state evidence from wrapper-local user data.
    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_wayland_user_data_foreign_general_evidence(
        &self,
        user_data: &UserDataMap,
        dmabuf: &Dmabuf,
    ) -> Result<Option<SampledDmabufWaylandForeignGeneralEvidence>, VulkanError> {
        let Some(slot) = user_data.get::<SampledDmabufWaylandForeignGeneralEvidenceSlot>() else {
            return Ok(None);
        };
        let evidence = slot.evidence.lock().map_err(|_| {
            VulkanError::UnsupportedOperation("sampled dmabuf Wayland external-state evidence")
        })?;
        let Some(evidence) = evidence.as_ref() else {
            return Ok(None);
        };
        if !evidence.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state identity",
            ));
        }
        evidence.validate_commit_token(user_data)?;
        evidence.validate_release_generation(self.sampled_dmabuf_release_generation_snapshot(dmabuf))?;
        Ok(Some(evidence.clone()))
    }

    /// Read compositor-provided texture-cache release lifecycle evidence from wrapper-local user data.
    #[cfg(feature = "wayland_frontend")]
    #[allow(dead_code)]
    fn sampled_dmabuf_wayland_user_data_texture_cache_release_lifecycle(
        &self,
        user_data: &UserDataMap,
        dmabuf: &Dmabuf,
    ) -> Result<Option<SampledDmabufWaylandTextureCacheReleaseLifecycle>, VulkanError> {
        let Some(slot) = user_data.get::<SampledDmabufWaylandTextureCacheReleaseLifecycleSlot>() else {
            return Ok(None);
        };
        let evidence = slot.evidence.lock().map_err(|_| {
            VulkanError::UnsupportedOperation("sampled dmabuf Wayland texture-cache release lifecycle")
        })?;
        let Some(evidence) = evidence.as_ref() else {
            return Ok(None);
        };

        evidence.validate_commit_token(user_data)?;
        evidence.release_lifecycle(dmabuf, &self.context_id).map(Some)
    }

    /// Gather the external-state evidence sources used by the normal Wayland sampled-dmabuf path.
    ///
    /// This preserves the production fail-closed order before sync evidence is considered: first the
    /// first-import layout and known-state sources, then the current-reacquire layout and known-state
    /// sources. The helper is intentionally still validation-stage: caller-provided
    /// `FOREIGN + GENERAL` evidence may now satisfy the external-state source, while public
    /// advertisement remains closed until the full generic sampled-dmabuf contract is implemented.
    #[allow(dead_code)]
    fn sampled_dmabuf_wayland_external_state_evidence_sources(
        &self,
        dmabuf: &Dmabuf,
        layout_history: SampledDmabufWaylandLayoutHistory,
        external_state: Option<&SampledDmabufWaylandForeignGeneralEvidence>,
    ) -> Result<SampledDmabufWaylandExternalStateEvidenceSources, VulkanError> {
        let first_import_layout =
            self.sampled_dmabuf_wayland_first_import_layout_evidence(dmabuf, layout_history, external_state)?;
        let first_import_foreign_general = self
            .sampled_dmabuf_wayland_first_import_foreign_general_evidence(
                dmabuf,
                layout_history,
                external_state,
            )?;
        let current_reacquire_layout = self.sampled_dmabuf_wayland_current_reacquire_layout_evidence(
            dmabuf,
            layout_history,
            external_state,
        )?;
        let current_reacquire_foreign_general = self
            .sampled_dmabuf_wayland_current_reacquire_foreign_general_evidence(
                dmabuf,
                layout_history,
                current_reacquire_layout.as_ref(),
            )?;

        Ok(SampledDmabufWaylandExternalStateEvidenceSources {
            first_import_layout,
            first_import_foreign_general,
            current_reacquire_layout,
            current_reacquire_foreign_general,
        })
    }

    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_wayland_policy_layout_history(
        &self,
        dmabuf: &Dmabuf,
        renderer_history: SampledDmabufWaylandLayoutHistory,
        external_state: Option<&SampledDmabufWaylandForeignGeneralEvidence>,
    ) -> Result<SampledDmabufWaylandLayoutHistory, VulkanError> {
        if renderer_history == SampledDmabufWaylandLayoutHistory::LocallyAcquired {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan unreleased local acquire",
            ));
        }

        if let Some(evidence) = external_state {
            evidence.validate_release_generation(self.sampled_dmabuf_release_generation_snapshot(dmabuf))?;
            Ok(evidence.layout_history_for_policy())
        } else {
            Ok(renderer_history)
        }
    }

    /// Validate the reacquire external image layout policy for a normal Wayland dmabuf.
    ///
    /// This token is only produced from renderer-local history plus current-commit producer-return
    /// evidence. Prior release history alone is not sufficient because Wayland metadata and explicit
    /// sync do not prove the producer returned the image in `FOREIGN` ownership and `GENERAL` layout.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_reacquire_layout_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandReacquireLayoutPolicy, VulkanError> {
        match context.layout_history {
            SampledDmabufWaylandLayoutHistory::NoRendererHistory => Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan reacquire layout history",
            )),
            SampledDmabufWaylandLayoutHistory::LocallyAcquired => Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan unreleased local acquire",
            )),
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral => {
                self.validate_sampled_dmabuf_wayland_current_reacquire_layout_policy(context)?;
                Ok(SampledDmabufWaylandReacquireLayoutPolicy {
                    dmabuf: context.dmabuf.weak(),
                })
            }
        }
    }

    /// Validate queue-family ownership transfer policy for a normal Wayland dmabuf.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_queue_family_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandQueueFamilyPolicy, VulkanError> {
        if !self.capabilities.external_memory.foreign_queue_family {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan queue-family capability",
            ));
        }

        Ok(SampledDmabufWaylandQueueFamilyPolicy {
            dmabuf: context.dmabuf.weak(),
        })
    }

    /// Validate acquire-sync import/wait policy for a normal Wayland dmabuf.
    ///
    /// The normal Wayland path requires an explicit acquire fence. The known-layout import helper
    /// then maps that [`SyncPoint`] into the Vulkan acquire submission by importing a sync-file wait
    /// semaphore when the device supports it and the sync point exports a suitable sync-file fd, or
    /// by waiting on the CPU before submitting the acquire barrier. This token does not prove image
    /// layout or queue-family ownership; those remain separate policy contracts.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_acquire_sync_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandAcquireSyncPolicy, VulkanError> {
        if !context.acquire_sync.is_for_dmabuf(context.dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf acquire sync identity",
            ));
        }
        context
            .acquire_sync
            .same_commit_token_as(context.release_evidence)?;
        self.validate_sampled_dmabuf_wayland_acquire_sync_contract(Some(context.acquire_sync.sync()))?;
        Ok(SampledDmabufWaylandAcquireSyncPolicy {
            dmabuf: context.dmabuf.weak(),
        })
    }

    /// Validate release-sync export/transfer policy for a normal Wayland dmabuf.
    ///
    /// The release evidence proves that a Wayland release point exists for this dmabuf. It is not an
    /// ownership transfer by itself; the ownership-transfer policy is deliberately validated later,
    /// after cache release lifecycle evidence, so the eventual `take_release_point_for_renderer()`
    /// operation happens only after all earlier guards pass and retry-safe Vulkan acquire setup reaches
    /// the ready-to-submit callback.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_release_sync_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandReleaseSyncPolicy, VulkanError> {
        if !context.release_evidence.is_for_dmabuf(context.dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf release evidence identity",
            ));
        }
        Ok(SampledDmabufWaylandReleaseSyncPolicy {
            dmabuf: context.dmabuf.weak(),
        })
    }

    /// Validate surface-cache reachability for replacement import.
    ///
    /// `RendererSurfaceState::update_buffer` retires cached textures on new buffer commits, and
    /// `import_surface` drains those retired textures with the renderer release hook before importing
    /// the replacement texture on Smithay's normal surface-cache path. Under the normal serialized
    /// renderer-utils surface access model, this marker proves that the current import call is past
    /// that retired-texture release point; the full no-next-import, reset, and destruction release
    /// lifecycle remains guarded below.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_texture_cache_replacement_reachability_contract(
        &self,
        dmabuf: &Dmabuf,
        post_retired_release_import: bool,
    ) -> Result<SampledDmabufWaylandTextureCacheReplacementReleaseReachability, VulkanError> {
        if post_retired_release_import {
            Ok(SampledDmabufWaylandTextureCacheReplacementReleaseReachability {
                dmabuf: dmabuf.weak(),
                renderer_context: Some(self.context_id.clone()),
            })
        } else {
            Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan import_surface post-retired-release call site",
            ))
        }
    }

    /// Validate renderer ownership of the Wayland release point for a normal Wayland dmabuf.
    ///
    /// This is the final validation-stage ownership-availability guard before the intended path may
    /// create a sampled dmabuf texture with a release obligation. Production supplies this token from
    /// the current Wayland buffer's release-point state, but the actual
    /// `Buffer::take_release_point_for_renderer()` transfer remains a separate fallible move performed
    /// by the ready-to-submit callback, after submit validation, fence creation, host locking, and
    /// semaphore payload reservation have all succeeded.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_release_ownership_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufReleaseOwnershipEvidence, VulkanError> {
        let Some(release_ownership) = context.release_ownership.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland release ownership transfer",
            ));
        };
        if !release_ownership.is_for_dmabuf(context.dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland release ownership identity",
            ));
        }
        context.release_evidence.same_commit_token_as(release_ownership)?;
        Ok(release_ownership.clone())
    }

    /// Validate that the current Wayland buffer still has a release point available for the renderer.
    ///
    /// This deliberately does not take the release point. It only proves that a later late-bound
    /// ownership transfer can be attempted after layout, sync, and texture-cache lifecycle guards have
    /// accepted the import. The move-only release obligation is validated separately once the point is
    /// actually taken.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_release_ownership_availability_contract(
        &self,
        dmabuf: &Dmabuf,
        has_release_point: bool,
    ) -> Result<SampledDmabufReleaseOwnershipEvidence, VulkanError> {
        if has_release_point {
            Ok(SampledDmabufReleaseOwnershipEvidence {
                dmabuf: dmabuf.weak(),
                commit_token: None,
            })
        } else {
            Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland release ownership transfer",
            ))
        }
    }

    /// Locate production evidence that the Wayland release point is available for renderer ownership.
    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_wayland_release_ownership_evidence(
        &self,
        dmabuf: &Dmabuf,
        #[cfg(feature = "backend_drm")] buffer: &super::utils::Buffer,
        #[cfg(not(feature = "backend_drm"))] _buffer: &super::utils::Buffer,
    ) -> Result<SampledDmabufReleaseOwnershipEvidence, VulkanError> {
        #[cfg(feature = "backend_drm")]
        {
            let current_dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer).map_err(|_| {
                VulkanError::UnsupportedOperation("sampled dmabuf Wayland release ownership buffer")
            })?;
            if current_dmabuf != dmabuf {
                return Err(VulkanError::UnsupportedOperation(
                    "sampled dmabuf Wayland release ownership buffer identity",
                ));
            }
            if buffer.release_point().is_none() && !self.wayland_linux_dmabuf_interop {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf Wayland release ownership transfer",
                ));
            }
            let commit_token_slot = buffer
                .user_data()
                .get_or_insert_threadsafe(SampledDmabufWaylandCommitTokenSlot::default);
            Ok(SampledDmabufReleaseOwnershipEvidence {
                dmabuf: dmabuf.weak(),
                commit_token: Some(Arc::downgrade(&commit_token_slot.token)),
            })
        }

        #[cfg(not(feature = "backend_drm"))]
        {
            let _ = dmabuf;
            Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland release ownership transfer",
            ))
        }
    }

    #[cfg(feature = "wayland_frontend")]
    fn sampled_dmabuf_take_wayland_release_ownership(
        dmabuf: &Dmabuf,
        #[cfg(feature = "backend_drm")] buffer: &super::utils::Buffer,
        #[cfg(not(feature = "backend_drm"))] _buffer: &super::utils::Buffer,
        linux_interop: bool,
    ) -> Result<SampledDmabufReleaseOwnership, VulkanError> {
        #[cfg(feature = "backend_drm")]
        {
            if buffer.release_point().is_none() {
                if !linux_interop {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland release ownership transfer",
                    ));
                }
                return Ok(SampledDmabufReleaseOwnership::cache_only(dmabuf));
            }
            let release_point =
                buffer
                    .take_release_point_for_renderer()
                    .ok_or(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland release ownership transfer",
                    ))?;
            Ok(SampledDmabufReleaseOwnership::wayland_syncobj(
                dmabuf,
                release_point,
            ))
        }

        #[cfg(not(feature = "backend_drm"))]
        {
            let _ = dmabuf;
            let _ = linux_interop;
            Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland release ownership transfer",
            ))
        }
    }

    /// Validate texture-cache reuse policy for a normal Wayland dmabuf.
    ///
    /// The validation-stage normal path imports a fresh sampled dmabuf texture for each explicit-sync
    /// commit instead of reusing renderer-local image state across commit-specific acquire/release
    /// points. That is not sufficient by itself: eviction, reset, and destruction paths must also be
    /// able to release imported Vulkan sampled dmabuf textures before drop. The renderer hook token
    /// only proves that a renderer-available cache release can call into Vulkan; it does not prove
    /// that compositors invoke the helper at every no-next-import/teardown point. Keep the call-site
    /// lifecycle as a separate token so the generic cache hook cannot be mistaken for a completed
    /// Vulkan release path.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_texture_cache_policy(
        &self,
        context: &SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
    ) -> Result<SampledDmabufWaylandTextureCachePolicy, VulkanError> {
        if !context.per_commit_texture_import {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan texture-cache policy",
            ));
        }

        let Some(replacement_release_reachability) =
            context.texture_cache_replacement_release_reachability.as_ref()
        else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan import_surface post-retired-release call site",
            ));
        };
        if !replacement_release_reachability.is_for_dmabuf(context.dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland import_surface post-retired-release call site identity",
            ));
        }
        replacement_release_reachability.validate_renderer_context(&self.context_id)?;

        let Some(release_hook) = context.texture_cache_release_hook.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan texture-cache release hook",
            ));
        };
        if !release_hook.is_for_dmabuf(context.dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland texture-cache release hook identity",
            ));
        }
        release_hook.validate_renderer_context(&self.context_id)?;

        let Some(release_lifecycle) = context.texture_cache_release_lifecycle.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan texture-cache release call sites",
            ));
        };
        if !release_lifecycle.is_for_dmabuf(context.dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland texture-cache release lifecycle identity",
            ));
        }

        Ok(SampledDmabufWaylandTextureCachePolicy {
            dmabuf: context.dmabuf.weak(),
        })
    }

    /// Validate each named part of Smithay's Wayland/Vulkan sampled-dmabuf policy.
    ///
    /// The fully satisfied path is only a validation-stage scaffold until each field is replaced or
    /// backed by a concrete implementation predicate and test. Future work must produce evidence for
    /// each contract instead of setting these markers from protocol metadata or another renderer's
    /// assumptions.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_wayland_vulkan_interop_policy_contracts(
        &self,
        dmabuf: &Dmabuf,
        contracts: &SampledDmabufWaylandVulkanInteropPolicyContracts,
    ) -> Result<SampledDmabufWaylandVulkanInteropPolicy, VulkanError> {
        let Some(layout) = contracts.layout.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan layout policy",
            ));
        };
        if !layout.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland layout identity",
            ));
        }
        let Some(foreign_general) = contracts.foreign_general.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan foreign GENERAL policy",
            ));
        };
        if !foreign_general.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland foreign GENERAL identity",
            ));
        }
        let Some(queue_family_transfer) = contracts.queue_family_transfer.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan queue-family policy",
            ));
        };
        if !queue_family_transfer.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland queue-family identity",
            ));
        }
        let Some(acquire_sync) = contracts.acquire_sync.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan acquire sync policy",
            ));
        };
        if !acquire_sync.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland acquire sync identity",
            ));
        }
        let Some(release_sync) = contracts.release_sync.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan release sync policy",
            ));
        };
        if !release_sync.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland release sync identity",
            ));
        }
        let Some(texture_cache_reuse) = contracts.texture_cache_reuse.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland Vulkan texture-cache policy",
            ));
        };
        if !texture_cache_reuse.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland texture-cache identity",
            ));
        }
        let Some(release_ownership) = contracts.release_ownership.as_ref() else {
            return Err(VulkanError::MissingCapability(
                "sampled dmabuf Wayland release ownership transfer",
            ));
        };
        if !release_ownership.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland release ownership identity",
            ));
        }

        Ok(SampledDmabufWaylandVulkanInteropPolicy {
            dmabuf: dmabuf.weak(),
            foreign_general: foreign_general.clone(),
        })
    }

    /// Validate the external ownership and image-layout contract for sampled dmabuf import.
    ///
    /// A Wayland acquire point proves producer completion, but not Vulkan queue-family ownership or
    /// image layout. Explicit known-layout evidence may pass directly. The normal Wayland path may
    /// pass only after Smithay's own Wayland/Vulkan interop policy evidence exists.
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_known_layout_contract(
        &self,
        dmabuf: &Dmabuf,
        evidence: SampledDmabufLayoutEvidence,
    ) -> Result<SampledDmabufKnownLayoutEvidence, VulkanError> {
        match evidence {
            SampledDmabufLayoutEvidence::KnownForeignGeneral(evidence) => {
                if !evidence.is_for_dmabuf(dmabuf) {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf known-layout identity",
                    ));
                }
                if !evidence.has_foreign_general_state() {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf known external state",
                    ));
                }
                Ok(evidence)
            }
            SampledDmabufLayoutEvidence::SmithayWaylandVulkanPolicy(policy) => {
                if !policy.is_for_dmabuf(dmabuf) {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland policy identity",
                    ));
                }
                if !policy.foreign_general.is_for_dmabuf(dmabuf) {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland foreign GENERAL identity",
                    ));
                }
                if !policy.foreign_general.has_foreign_general_state() {
                    return Err(VulkanError::UnsupportedOperation(
                        "sampled dmabuf Wayland foreign GENERAL state",
                    ));
                }
                Ok(policy.foreign_general)
            }
            SampledDmabufLayoutEvidence::WaylandDmabuf => Err(VulkanError::MissingCapability(
                "sampled dmabuf known-layout contract",
            )),
        }
    }

    /// Validate a move-only renderer-owned release obligation for a sampled dmabuf import.
    ///
    /// The ownership transfer itself is intentionally separate from cloneable release-point evidence.
    /// This validator only checks that the already-owned obligation belongs to the same dmabuf before
    /// the successful import attaches it to [`VulkanTexture`].
    #[allow(dead_code)]
    fn validate_sampled_dmabuf_release_lifecycle_contract(
        &self,
        dmabuf: &Dmabuf,
        evidence: &SampledDmabufReleaseOwnership,
    ) -> Result<(), VulkanError> {
        if !evidence.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf release ownership identity",
            ));
        }

        Ok(())
    }

    fn render_target_format_supported(&self, format: Fourcc) -> bool {
        self.render_target_formats().contains(&Format {
            code: format,
            modifier: Modifier::Invalid,
        })
    }

    #[allow(dead_code)]
    fn create_offscreen_render_target(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<VulkanRenderTarget<'static>, VulkanError> {
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let vk_format = get_render_vk_format(format)?;
        let extent = extent_from_size(size, "offscreen render target size")?;
        let color_image = device.create_offscreen_color_image(extent, vk_format)?;

        Ok(VulkanRenderTarget::from_offscreen_image(
            self.context_id.clone(),
            size,
            format,
            color_image,
        ))
    }

    #[allow(dead_code)]
    fn create_imported_dmabuf_texture(
        &mut self,
        dmabuf: &Dmabuf,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let Some(sampled_image) = device.create_dmabuf_sampled_image_resources(
            dmabuf,
            self.downscale_filter,
            self.upscale_filter,
        )?
        else {
            return Ok(None);
        };

        Ok(Some(VulkanTexture::from_dmabuf_sampled_image(
            self.context_id.clone(),
            dmabuf,
            &import,
            sampled_image,
        )))
    }

    /// Import a dmabuf as a sampled texture when the producer's Vulkan external state is known.
    ///
    /// # Safety
    ///
    /// The caller must ensure the dmabuf producer released the image to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`, and that `acquire_semaphore`, if
    /// present, represents the producer's completion dependency for that release. If no semaphore is
    /// supplied, the caller must ensure the producer's writes and ownership release are already
    /// complete and visible to this renderer's Vulkan queue submission.
    #[allow(dead_code)]
    unsafe fn create_imported_dmabuf_texture_with_known_general_layout(
        &mut self,
        dmabuf: &Dmabuf,
        foreign_general: SampledDmabufKnownLayoutEvidence,
        acquire_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        self.validate_sampled_dmabuf_known_layout_contract(
            dmabuf,
            SampledDmabufLayoutEvidence::KnownForeignGeneral(foreign_general),
        )?;
        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let Some(sampled_image) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            device.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout(
                dmabuf,
                self.downscale_filter,
                self.upscale_filter,
                acquire_semaphore,
            )
        })?
        else {
            return Ok(None);
        };

        Ok(Some(VulkanTexture::from_acquired_dmabuf_sampled_image(
            self.context_id.clone(),
            dmabuf,
            &import,
            sampled_image,
        )))
    }

    /// Import a known-layout dmabuf using a Smithay sync point as the optional acquire dependency.
    ///
    /// # Safety
    ///
    /// The caller must ensure the dmabuf producer released the image to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`. If `acquire_sync` is `Some`, it
    /// must represent the producer's completion dependency for that release and signal only after
    /// the producer's writes and ownership release for this dmabuf are complete. If `acquire_sync`
    /// is `None`, the caller must ensure those writes and ownership release are already complete and
    /// visible to this renderer's Vulkan queue submission. If `acquire_sync` exports a fence fd and
    /// this device supports sync-file import, that fd must be a valid Linux sync-file fd suitable for
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`.
    #[allow(dead_code)]
    unsafe fn create_imported_dmabuf_texture_with_known_general_layout_and_sync_point(
        &mut self,
        dmabuf: &Dmabuf,
        foreign_general: SampledDmabufKnownLayoutEvidence,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        self.validate_sampled_dmabuf_known_layout_contract(
            dmabuf,
            SampledDmabufLayoutEvidence::KnownForeignGeneral(foreign_general),
        )?;
        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;
        let acquire = {
            let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
            unsafe {
                // SAFETY: Forwarded from this method's caller.
                device.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point_classified(
                    dmabuf,
                    self.downscale_filter,
                    self.upscale_filter,
                    acquire_sync,
                )
            }
        };
        let sampled_image = match acquire {
            Ok(Some(sampled_image)) => sampled_image,
            Ok(None) => {
                return Ok(None);
            }
            Err(device::VulkanSampledDmabufForeignAcquireError::RetrySafe(err)) => return Err(err),
            Err(device::VulkanSampledDmabufForeignAcquireError::AcquireSubmitted {
                err,
                sampled_image: Some(sampled_image),
            }) => {
                return Err(
                    self.sampled_dmabuf_import_error_after_submitted_acquire_without_release(
                        dmabuf,
                        &import,
                        err,
                        sampled_image,
                    ),
                );
            }
            Err(device::VulkanSampledDmabufForeignAcquireError::AcquireSubmitted {
                err,
                sampled_image: None,
            }) => return Err(err),
        };

        Ok(Some(VulkanTexture::from_acquired_dmabuf_sampled_image(
            self.context_id.clone(),
            dmabuf,
            &import,
            sampled_image,
        )))
    }

    /// Import a known-layout dmabuf and attach the Wayland release obligation to the texture.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the same known-layout and acquire-sync requirements as
    /// [`VulkanRenderer::create_imported_dmabuf_texture_with_known_general_layout_and_sync_point`].
    #[allow(dead_code)]
    unsafe fn create_imported_dmabuf_texture_with_known_general_layout_release_and_sync_point<F>(
        &mut self,
        dmabuf: &Dmabuf,
        foreign_general: SampledDmabufKnownLayoutEvidence,
        acquire_sync: Option<&SyncPoint>,
        release_ownership: F,
    ) -> Result<Option<VulkanTexture>, VulkanError>
    where
        F: FnOnce() -> Result<SampledDmabufReleaseOwnership, VulkanError>,
    {
        self.validate_sampled_dmabuf_known_layout_contract(
            dmabuf,
            SampledDmabufLayoutEvidence::KnownForeignGeneral(foreign_general),
        )?;
        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;
        let Some(prepared) = ({
            let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
            unsafe {
                // SAFETY: Forwarded from this method's caller. This prepares image/view/sampler,
                // acquire sync, and the acquire command buffer without submitting it or consuming the
                // move-only Wayland release point.
                device
                    .prepare_sampled_dmabuf_foreign_acquire_with_known_general_layout(
                        dmabuf,
                        self.downscale_filter,
                        self.upscale_filter,
                        acquire_sync,
                    )
                    .map_err(device::VulkanSampledDmabufForeignAcquireError::into_inner)?
            }
        }) else {
            return Ok(None);
        };

        let acquire = {
            let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
            device.submit_prepared_sampled_dmabuf_foreign_acquire_ready(prepared, release_ownership)
        };

        let (sampled_image, release_ownership) = match acquire {
            Ok((sampled_image, release_ownership)) => (sampled_image, release_ownership),
            Err(device::VulkanSampledDmabufReadyAcquireError::RetrySafe(err))
            | Err(device::VulkanSampledDmabufReadyAcquireError::ReadyCallbackFailed(err)) => {
                return Err(err);
            }
            Err(device::VulkanSampledDmabufReadyAcquireError::ReadyCallbackCommitted { err, ready }) => {
                return Err(self.sampled_dmabuf_release_only_error_after_ready_callback(err, ready));
            }
            Err(device::VulkanSampledDmabufReadyAcquireError::AcquireSubmitted {
                err,
                sampled_image,
                ready,
            }) => {
                return Err(self.sampled_dmabuf_import_error_after_submitted_acquire(
                    dmabuf,
                    &import,
                    err,
                    sampled_image,
                    ready,
                ));
            }
        };
        if let Err(err) = self.validate_sampled_dmabuf_release_lifecycle_contract(dmabuf, &release_ownership)
        {
            return Err(self.sampled_dmabuf_import_error_after_submitted_acquire(
                dmabuf,
                &import,
                err,
                sampled_image,
                release_ownership,
            ));
        }
        let release = release_ownership.into_release();

        Ok(Some(
            VulkanTexture::from_acquired_dmabuf_sampled_image_with_release(
                self.context_id.clone(),
                dmabuf,
                &import,
                sampled_image,
                release,
            ),
        ))
    }

    fn sampled_dmabuf_release_only_error_after_ready_callback(
        &mut self,
        submit_err: VulkanError,
        release_ownership: SampledDmabufReleaseOwnership,
    ) -> VulkanError {
        match release_ownership.signal_wayland_release_once() {
            Ok(()) => submit_err,
            Err(release_err) => {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::ReleaseOnly(release_ownership),
                );
                release_err
            }
        }
    }

    fn sampled_dmabuf_import_error_after_submitted_acquire_without_release(
        &mut self,
        dmabuf: &Dmabuf,
        import: &image::VulkanDmabufImportState,
        acquire_err: VulkanError,
        sampled_image: device::VulkanSampledImage,
    ) -> VulkanError {
        self.record_sampled_dmabuf_locally_acquired(dmabuf);
        let texture = VulkanTexture::from_acquired_dmabuf_sampled_image(
            self.context_id.clone(),
            dmabuf,
            import,
            sampled_image,
        );

        let cleanup = {
            let Some(sampled_image) = texture.sampled_image.as_ref() else {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                );
                return VulkanError::UnsupportedOperation("dmabuf texture sampled image");
            };
            let Some(device) = self.device.as_ref() else {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                );
                return VulkanError::VulkanUnavailable;
            };

            device.release_sampled_dmabuf_to_foreign_general_classified(sampled_image.image(), false)
        };

        match cleanup {
            Ok((true, _)) => {
                self.record_sampled_dmabuf_released_to_foreign_general(dmabuf);
                acquire_err
            }
            Ok((false, _)) => {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                );
                VulkanError::UnsupportedOperation("sampled dmabuf acquire cleanup release")
            }
            Err(device::VulkanSampledDmabufForeignReleaseError::RetrySafe(cleanup_err)) => {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                );
                tracing::warn!(
                    ?acquire_err,
                    ?cleanup_err,
                    "retained sampled dmabuf texture after failed cleanup of submitted loopback acquire"
                );
                acquire_err
            }
            Err(device::VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(cleanup_err)) => {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::ReleaseCompletionUnknownTexture(texture),
                );
                tracing::warn!(
                    ?acquire_err,
                    ?cleanup_err,
                    "retained sampled dmabuf texture after loopback cleanup release completion became unknown"
                );
                acquire_err
            }
        }
    }

    fn sampled_dmabuf_import_error_after_submitted_acquire(
        &mut self,
        dmabuf: &Dmabuf,
        import: &image::VulkanDmabufImportState,
        acquire_err: VulkanError,
        sampled_image: device::VulkanSampledImage,
        release_ownership: SampledDmabufReleaseOwnership,
    ) -> VulkanError {
        self.record_sampled_dmabuf_locally_acquired(dmabuf);
        let mut texture = VulkanTexture::from_acquired_dmabuf_sampled_image_with_release(
            self.context_id.clone(),
            dmabuf,
            import,
            sampled_image,
            release_ownership.into_release(),
        );

        let cleanup = {
            if texture.sampled_image.is_none() {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                );
                return VulkanError::UnsupportedOperation("dmabuf texture sampled image");
            }
            let Some(device) = self.device.as_ref() else {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                );
                return VulkanError::VulkanUnavailable;
            };
            let sampled_image = texture
                .sampled_image
                .as_ref()
                .expect("sampled image checked before cleanup release");

            device.release_sampled_dmabuf_to_foreign_general_classified(sampled_image.image(), false)
        };

        match cleanup {
            Ok((true, release_sync_file)) => {
                if let Err(release_err) = texture
                    .signal_sampled_dmabuf_release_point(release_sync_file.as_ref().map(OwnedFd::as_fd))
                {
                    if let Some(release) = texture.sampled_dmabuf_release.take() {
                        self.retain_pending_sampled_dmabuf_import_obligation(
                            PendingSampledDmabufImportObligation::ReleaseOnly(
                                SampledDmabufReleaseOwnership {
                                    dmabuf: dmabuf.weak(),
                                    release,
                                },
                            ),
                        );
                    }
                    self.record_sampled_dmabuf_released_to_foreign_general(dmabuf);
                    return release_err;
                }
                self.record_sampled_dmabuf_released_to_foreign_general(dmabuf);
                acquire_err
            }
            Ok((false, _)) => {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                );
                VulkanError::UnsupportedOperation("sampled dmabuf acquire cleanup release")
            }
            Err(device::VulkanSampledDmabufForeignReleaseError::RetrySafe(cleanup_err)) => {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::AcquiredTexture(texture),
                );
                tracing::warn!(
                    ?acquire_err,
                    ?cleanup_err,
                    "retained sampled dmabuf texture after failed cleanup of submitted acquire"
                );
                acquire_err
            }
            Err(device::VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted(cleanup_err)) => {
                self.retain_pending_sampled_dmabuf_import_obligation(
                    PendingSampledDmabufImportObligation::ReleaseCompletionUnknownTexture(texture),
                );
                tracing::warn!(
                    ?acquire_err,
                    ?cleanup_err,
                    "retained sampled dmabuf texture after cleanup release completion became unknown"
                );
                acquire_err
            }
        }
    }

    #[allow(dead_code)]
    fn sampled_dmabuf_release_ownership_error_after_acquire_cleanup(
        &mut self,
        dmabuf: &Dmabuf,
        ownership_err: VulkanError,
        cleanup: Result<(bool, Option<OwnedFd>), device::VulkanSampledDmabufForeignReleaseError>,
    ) -> VulkanError {
        match cleanup {
            Ok((true, None)) => {
                self.record_sampled_dmabuf_released_to_foreign_general(dmabuf);
                ownership_err
            }
            Ok((true, Some(_))) => {
                VulkanError::UnsupportedOperation("sampled dmabuf acquire cleanup sync file")
            }
            Ok((false, _)) => VulkanError::UnsupportedOperation("sampled dmabuf acquire cleanup release"),
            Err(cleanup_err) => cleanup_err.into_inner(),
        }
    }

    /// Import a dmabuf as a render target and acquire it for color-attachment rendering.
    ///
    /// This is the implementation helper behind the explicit public dmabuf render-target API.
    ///
    /// # Safety
    ///
    /// The caller must ensure the foreign producer has released ownership to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` before this acquire is submitted. If `preserve_contents` is
    /// true, the producer must have released the image in `VK_IMAGE_LAYOUT_GENERAL`. If
    /// `acquire_semaphore` is present, it must signal only after the producer's writes and ownership
    /// release complete. If it is absent, those operations must already be complete and visible to
    /// this renderer's Vulkan queue submission. If `preserve_contents` is false, previous contents
    /// are discarded.
    #[allow(dead_code)]
    pub(crate) unsafe fn create_acquired_dmabuf_render_target(
        &mut self,
        dmabuf: &Dmabuf,
        preserve_contents: bool,
        acquire_semaphore: Option<&VulkanSyncFileSemaphore>,
    ) -> Result<Option<VulkanRenderTarget<'static>>, VulkanError> {
        self.validate_no_pending_sampled_dmabuf_import_obligation(dmabuf)?;
        let import = validate_dmabuf_render_target_metadata(dmabuf)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let Some(color_image) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            device.create_acquired_dmabuf_render_target_image(dmabuf, preserve_contents, acquire_semaphore)
        })?
        else {
            return Ok(None);
        };

        Ok(Some(VulkanRenderTarget::from_acquired_dmabuf_render_target(
            self.context_id.clone(),
            dmabuf,
            &import,
            color_image,
        )))
    }

    /// Import a dmabuf as an internal render target and acquire it using a Smithay sync point as the
    /// optional producer-completion dependency.
    ///
    /// This is a sync-point convenience wrapper for the acquired dmabuf render-target path used by
    /// the explicit public dmabuf render-target API.
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
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`. If `preserve_contents` is false, previous
    /// contents are discarded.
    #[allow(dead_code)]
    pub(crate) unsafe fn create_acquired_dmabuf_render_target_with_sync_point<'target>(
        &mut self,
        dmabuf: &'target Dmabuf,
        preserve_contents: bool,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanRenderTarget<'target>>, VulkanError> {
        self.validate_no_pending_sampled_dmabuf_import_obligation(dmabuf)?;
        let import = validate_dmabuf_render_target_metadata(dmabuf)?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let Some(color_image) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            device.create_acquired_dmabuf_render_target_image_with_sync_point(
                dmabuf,
                preserve_contents,
                acquire_sync,
            )
        })?
        else {
            return Ok(None);
        };

        Ok(Some(VulkanRenderTarget::from_acquired_dmabuf_render_target(
            self.context_id.clone(),
            dmabuf,
            &import,
            color_image,
        )))
    }

    /// Bind a foreign dmabuf as a Vulkan render target using explicit Vulkan acquire semantics.
    ///
    /// This method is intentionally public in this development fork. It is the correct Vulkan-shaped
    /// entry point for callers that can reason about external-memory ownership, layout, and acquire
    /// synchronization. Generic [`Bind<Dmabuf>`] delegates here with discard/full-repaint acquire
    /// policy; callers that use this method directly are opting into the explicit unsafe
    /// external-memory contract and may provide preserve/acquire-sync policy themselves.
    ///
    /// # Safety
    ///
    /// The caller must ensure the foreign producer has released ownership to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` before this acquire is submitted. If
    /// `acquire.preserve_contents` is true, the producer must have released the image in
    /// `VK_IMAGE_LAYOUT_GENERAL`. If `acquire.acquire_sync` is present, it must represent the
    /// producer's completion dependency for that release and signal only after the producer's writes
    /// and ownership release for this dmabuf are complete. If it is absent, those operations must
    /// already be complete and visible to this renderer's Vulkan queue submission. If the sync point
    /// exports a fence fd and this device supports sync-file import, that fd must be a valid Linux
    /// sync-file fd suitable for `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`. If
    /// `preserve_contents` is false, previous contents are discarded. The caller must not bind the
    /// same underlying dmabuf storage through another alias while the returned target is acquired.
    /// After a successful acquire, renderers must either finish a frame for the returned target or
    /// call [`VulkanRenderer::release_dmabuf_render_target_after_render_error`] before reusing the
    /// dmabuf externally; dropping the target alone does not release Vulkan ownership back to the
    /// foreign queue family.
    pub unsafe fn bind_dmabuf_render_target<'target>(
        &mut self,
        dmabuf: &'target mut Dmabuf,
        acquire: VulkanDmabufRenderTargetAcquire<'_>,
    ) -> Result<Option<VulkanRenderTarget<'target>>, VulkanError> {
        // SAFETY: Forwarded from this public unsafe method's caller.
        unsafe {
            self.create_acquired_dmabuf_render_target_with_sync_point(
                dmabuf,
                acquire.preserve_contents,
                acquire.acquire_sync,
            )
        }
    }

    /// Bind a dmabuf render target after consuming allocator-owned FOREIGN/GENERAL release evidence.
    ///
    /// This is a validation-stage bridge from the Vulkan allocator export/release path to the normal
    /// Vulkan dmabuf render-target acquire path. It does not advertise generic dmabuf target support
    /// beyond the existing development-gated capability surface.
    ///
    /// # Safety
    ///
    /// The caller must ensure there was no intervening access, acquire, release, or layout/ownership
    /// transition of the dmabuf after `evidence` was produced by the Vulkan allocator. This helper
    /// consumes the evidence, checks it is tied to this Smithay dmabuf identity, and acquires from the
    /// known foreign `GENERAL` layout.
    #[allow(dead_code)]
    pub(crate) unsafe fn bind_allocator_released_dmabuf_render_target<'target>(
        &mut self,
        dmabuf: &'target mut Dmabuf,
        evidence: VulkanAllocatorDmabufForeignReleaseEvidence,
    ) -> Result<Option<VulkanRenderTarget<'target>>, VulkanError> {
        let contract = VulkanAllocatorDmabufRenderTargetContract::from_allocator_release(dmabuf, evidence)?;

        // SAFETY: Forwarded from this helper's caller and backed by the consumed allocator evidence.
        unsafe { self.bind_allocator_dmabuf_render_target_contract(contract) }
    }

    /// Bind a dmabuf render target after consuming the allocator render-target contract.
    ///
    /// This keeps the allocator-owned `FOREIGN + GENERAL` proof in a named normal-path shape before
    /// the renderer reaches the unsafe Vulkan acquire. It still uses the same development-gated dmabuf
    /// target implementation and does not enable generic KMS presentation or scanout advertisement.
    ///
    /// # Safety
    ///
    /// The caller must ensure there was no intervening access, acquire, release, or layout/ownership
    /// transition of the dmabuf after `contract` was constructed from allocator release evidence.
    #[allow(dead_code)]
    pub(crate) unsafe fn bind_allocator_dmabuf_render_target_contract<'target>(
        &mut self,
        contract: VulkanAllocatorDmabufRenderTargetContract<'target>,
    ) -> Result<Option<VulkanRenderTarget<'target>>, VulkanError> {
        let (dmabuf, evidence) = contract.into_parts()?;
        if !evidence.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "allocator dmabuf release evidence",
            ));
        }

        // SAFETY: Forwarded from this helper's caller and backed by the consumed allocator evidence.
        unsafe { self.bind_dmabuf_render_target(dmabuf, VulkanDmabufRenderTargetAcquire::preserve(None)) }
    }

    /// Returns whether this dmabuf's format, modifier, and plane count can be sampled.
    ///
    /// This is protocol-admission metadata only. It does not import the image, acquire ownership, or
    /// advertise generic [`ImportDma`].
    pub fn sampled_dmabuf_import_supported(&self, dmabuf: &Dmabuf) -> bool {
        self.validate_sampled_dmabuf_import_metadata(dmabuf).is_ok()
    }

    /// Import a sampled dmabuf through the explicit foreign-GENERAL contract.
    ///
    /// On success this renderer owns the image locally until
    /// [`VulkanRenderer::release_imported_dmabuf_texture_to_foreign_general_sync_point`]. The same
    /// dmabuf cannot be imported again while that ownership is outstanding. This does not enable
    /// generic [`ImportDma`]. Callers that cannot construct [`VulkanSampledDmabufImport`] still
    /// cannot import.
    pub fn import_sampled_dmabuf(
        &mut self,
        import: VulkanSampledDmabufImport<'_, '_>,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        unsafe {
            // SAFETY: Constructing `VulkanSampledDmabufImport` requires the same proof as
            // `import_dmabuf_texture_with_known_general_layout`.
            self.import_dmabuf_texture_with_known_general_layout(import.dmabuf, import.acquire.acquire_sync)
        }
    }

    /// Import a dmabuf as a sampled texture when the producer's Vulkan external state is known.
    ///
    /// This is a validation-stage development helper for the intended sampled dmabuf path. It does
    /// not make generic [`ImportDma`] public-advertised, and `dmabuf_formats()` remains gated by the
    /// public sampled-dmabuf readiness contract until arbitrary client-buffer acquire/layout/sync and
    /// release-lifecycle contracts are implemented and tested.
    ///
    /// Use [`VulkanRenderer::release_imported_dmabuf_texture_to_foreign_general_sync_point`] before
    /// handing the dmabuf back to a foreign Vulkan producer/consumer.
    ///
    /// # Safety
    ///
    /// The caller must ensure the dmabuf producer released the image to
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`. If `acquire_sync` is `Some`, it
    /// must represent the producer's completion dependency for that release and signal only after
    /// the producer's writes and ownership release for this dmabuf are complete. If `acquire_sync`
    /// is `None`, the caller must ensure those writes and ownership release are already complete and
    /// visible to this renderer's Vulkan queue submission. If `acquire_sync` exports a fence fd and
    /// this device supports sync-file import, that fd must be a valid Linux sync-file fd suitable for
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`.
    pub unsafe fn import_dmabuf_texture_with_known_general_layout(
        &mut self,
        dmabuf: &Dmabuf,
        acquire_sync: Option<&SyncPoint>,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        self.validate_sampled_dmabuf_public_import_lifecycle(dmabuf)?;
        let foreign_general = unsafe {
            // SAFETY: This public unsafe method requires its caller to prove the producer released
            // the dmabuf to FOREIGN ownership in GENERAL layout.
            SampledDmabufKnownLayoutEvidence::foreign_general(dmabuf.weak())
        };
        let texture = unsafe {
            // SAFETY: Forwarded from this public unsafe method's caller.
            self.create_imported_dmabuf_texture_with_known_general_layout_and_sync_point(
                dmabuf,
                foreign_general,
                acquire_sync,
            )?
        };
        if texture.is_some() {
            self.record_sampled_dmabuf_locally_acquired(dmabuf);
        }
        Ok(texture)
    }

    /// Release an acquired dmabuf texture back to foreign ownership in `VK_IMAGE_LAYOUT_GENERAL`.
    ///
    /// This is the release counterpart to
    /// [`VulkanRenderer::import_dmabuf_texture_with_known_general_layout`]. It does not make public
    /// dmabuf import/export supported; callers must only pass textures created by the acquired
    /// dmabuf import path for this renderer.
    #[allow(dead_code)]
    fn release_imported_dmabuf_texture_to_foreign_general(
        &mut self,
        texture: &VulkanTexture,
        export_sync_file: bool,
    ) -> Result<(bool, Option<OwnedFd>), VulkanError> {
        self.release_imported_dmabuf_texture_to_foreign_general_classified(texture, export_sync_file)
            .map_err(device::VulkanSampledDmabufForeignReleaseError::into_inner)
    }

    #[allow(dead_code)]
    fn release_imported_dmabuf_texture_to_foreign_general_classified(
        &mut self,
        texture: &VulkanTexture,
        export_sync_file: bool,
    ) -> Result<(bool, Option<OwnedFd>), device::VulkanSampledDmabufForeignReleaseError> {
        if texture.context_id != self.context_id {
            return Err(device::VulkanSampledDmabufForeignReleaseError::RetrySafe(
                VulkanError::UnsupportedOperation("foreign dmabuf texture"),
            ));
        }
        if texture.image.source != image::VulkanImageSource::DmabufImport {
            return Err(device::VulkanSampledDmabufForeignReleaseError::RetrySafe(
                VulkanError::UnsupportedOperation("dmabuf texture"),
            ));
        }
        let sampled_image = texture
            .sampled_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf texture sampled image"))
            .map_err(device::VulkanSampledDmabufForeignReleaseError::RetrySafe)?;
        let device = self
            .device
            .as_ref()
            .ok_or(VulkanError::VulkanUnavailable)
            .map_err(device::VulkanSampledDmabufForeignReleaseError::RetrySafe)?;

        let (released, release_sync_file) = device
            .release_sampled_dmabuf_to_foreign_general_classified(sampled_image.image(), export_sync_file)?;
        if released {
            texture
                .signal_sampled_dmabuf_release_point(release_sync_file.as_ref().map(OwnedFd::as_fd))
                .map_err(device::VulkanSampledDmabufForeignReleaseError::ReleaseSubmitted)?;
            if let Some(dmabuf) = texture.sampled_dmabuf.as_ref().and_then(WeakDmabuf::upgrade) {
                self.record_sampled_dmabuf_released_to_foreign_general(&dmabuf);
            }
        }
        Ok((released, release_sync_file))
    }

    /// Complete sampled-dmabuf cache release after Vulkan image release side effects occurred.
    ///
    /// Once the device release helper reports that the sampled image has been released to foreign
    /// ownership in `GENERAL`, failures while satisfying the Wayland release point are no longer
    /// retry-safe for the same texture. The generic surface cache must drop that retired texture and
    /// retain only later unprocessed entries.
    #[allow(dead_code)]
    fn complete_sampled_dmabuf_cache_release_after_device_release(
        &mut self,
        texture: &VulkanTexture,
        release_sync_file: Option<BorrowedFd<'_>>,
    ) -> Result<(), SurfaceCacheTextureReleaseError<VulkanError>> {
        if let Some(dmabuf) = texture.sampled_dmabuf.as_ref().and_then(WeakDmabuf::upgrade) {
            if let Err(release_err) = texture.signal_sampled_dmabuf_release_point(release_sync_file) {
                if let Some(release) = texture.sampled_dmabuf_release.clone() {
                    self.retain_pending_sampled_dmabuf_import_obligation(
                        PendingSampledDmabufImportObligation::ReleaseOnly(SampledDmabufReleaseOwnership {
                            dmabuf: dmabuf.weak(),
                            release,
                        }),
                    );
                }
                self.record_sampled_dmabuf_released_to_foreign_general(&dmabuf);
                return Err(SurfaceCacheTextureReleaseError::ReleaseSideEffectsCommitted(
                    release_err,
                ));
            }
            self.record_sampled_dmabuf_released_to_foreign_general(&dmabuf);
        } else {
            texture
                .signal_sampled_dmabuf_release_point(release_sync_file)
                .map_err(SurfaceCacheTextureReleaseError::ReleaseSideEffectsCommitted)?;
        }

        Ok(())
    }

    /// Release a cached Wayland texture before renderer surface-state drops it.
    ///
    /// Ordinary textures do not carry a sampled-dmabuf release obligation and can be dropped by the
    /// generic cache. Textures imported through the intended Wayland sampled-dmabuf path remain
    /// development-gated by the same image ownership/layout preconditions as the explicit sampled
    /// dmabuf release helper. Queue-release side effects are mapped into the generic surface-cache
    /// release outcome contract so retry-safe failures retain the texture while accepted release
    /// submissions are not retried as locally owned.
    #[allow(dead_code)]
    fn release_retired_wayland_texture_for_cache(
        &mut self,
        texture: &VulkanTexture,
    ) -> Result<(), SurfaceCacheTextureReleaseError<VulkanError>> {
        if texture.sampled_dmabuf_release.is_none() {
            return Ok(());
        }

        if texture.context_id != self.context_id {
            return Err(SurfaceCacheTextureReleaseError::RetrySafe(
                VulkanError::UnsupportedOperation("foreign dmabuf texture"),
            ));
        }
        if texture.image.source != image::VulkanImageSource::DmabufImport {
            return Err(SurfaceCacheTextureReleaseError::RetrySafe(
                VulkanError::UnsupportedOperation("dmabuf texture"),
            ));
        }
        let sampled_image =
            texture
                .sampled_image
                .as_ref()
                .ok_or(SurfaceCacheTextureReleaseError::RetrySafe(
                    VulkanError::UnsupportedOperation("dmabuf texture sampled image"),
                ))?;
        let device = self
            .device
            .as_ref()
            .ok_or(SurfaceCacheTextureReleaseError::RetrySafe(
                VulkanError::VulkanUnavailable,
            ))?;

        let (released, release_sync_file) = device
            .release_sampled_dmabuf_to_foreign_general_classified(sampled_image.image(), false)
            .map_err(sampled_dmabuf_cache_device_release_error)?;
        if !released {
            return Err(SurfaceCacheTextureReleaseError::RetrySafe(
                VulkanError::MissingCapability("sampled dmabuf Wayland Vulkan texture-cache release state"),
            ));
        }

        self.complete_sampled_dmabuf_cache_release_after_device_release(
            texture,
            release_sync_file.as_ref().map(OwnedFd::as_fd),
        )
    }

    /// Release an acquired dmabuf texture and return the exported release fence as a [`SyncPoint`]
    /// when available.
    ///
    /// This is a validation-stage development helper for explicit known-layout sampled dmabuf
    /// imports. It does not make generic [`ImportDma`] public-advertised. When `released` is true,
    /// the texture's dmabuf image has been returned to foreign ownership and must not be sampled by
    /// this renderer again until it is reacquired. The returned [`SyncPoint`] is the release
    /// dependency for the foreign producer or consumer.
    pub fn release_imported_dmabuf_texture_to_foreign_general_sync_point(
        &mut self,
        texture: &VulkanTexture,
        export_sync_file: bool,
    ) -> Result<(bool, SyncPoint), VulkanError> {
        let (released, sync_file) =
            self.release_imported_dmabuf_texture_to_foreign_general(texture, export_sync_file)?;
        Ok((released, sync_point_from_sync_file(sync_file)))
    }

    /// Build the typed sampled-dmabuf import context from renderer-managed Wayland buffer state.
    ///
    /// This is the internal boundary between Smithay's Wayland protocol/renderer-utils evidence and
    /// Vulkan's explicit import requirements. The returned context owns validation evidence that can
    /// be checked without consuming the move-only Wayland release point; release ownership is taken
    /// later by the Vulkan acquire ready-to-submit callback.
    #[cfg(feature = "wayland_frontend")]
    fn wayland_sampled_dmabuf_import_context<'a>(
        &mut self,
        buffer: &'a super::utils::Buffer,
        surface: Option<&crate::wayland::compositor::SurfaceData>,
    ) -> Result<SampledDmabufImportContext<'a>, VulkanError> {
        let dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
            .expect("wayland_sampled_dmabuf_import_context without checking buffer type?");
        self.validate_no_pending_sampled_dmabuf_import_obligation(dmabuf)?;

        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;
        let renderer_layout_history = self.sampled_dmabuf_layout_history(dmabuf);
        let wayland_external_state =
            self.sampled_dmabuf_wayland_buffer_foreign_general_evidence(buffer, dmabuf)?;
        let layout_history = self.sampled_dmabuf_wayland_policy_layout_history(
            dmabuf,
            renderer_layout_history,
            wayland_external_state.as_ref(),
        )?;
        let external_state_sources = self.sampled_dmabuf_wayland_external_state_evidence_sources(
            dmabuf,
            layout_history,
            wayland_external_state.as_ref(),
        )?;
        let acquire_sync = self.sampled_dmabuf_wayland_acquire_sync_evidence(dmabuf, buffer)?;
        let release_evidence = self.sampled_dmabuf_wayland_release_evidence(dmabuf, buffer)?;
        let release_ownership = self.sampled_dmabuf_wayland_release_ownership_evidence(dmabuf, buffer)?;
        let post_retired_release_import = surface
            .map(super::utils::surface_import_after_retired_release)
            .unwrap_or(false);
        let texture_cache_replacement_release_reachability = self
            .validate_sampled_dmabuf_wayland_texture_cache_replacement_reachability_contract(
                dmabuf,
                post_retired_release_import,
            )?;
        let texture_cache_release_hook = self.sampled_dmabuf_wayland_texture_cache_release_hook(dmabuf);
        let texture_cache_release_lifecycle =
            self.sampled_dmabuf_wayland_buffer_texture_cache_release_lifecycle(buffer, dmabuf)?;

        Ok(SampledDmabufImportContext {
            dmabuf,
            import,
            acquire_sync,
            release_evidence,
            release_ownership,
            per_commit_texture_import: true,
            layout_history,
            external_state_sources,
            texture_cache_replacement_release_reachability,
            texture_cache_release_hook,
            texture_cache_release_lifecycle,
        })
    }

    /// Complete the normal Wayland sampled-dmabuf import after evidence collection.
    ///
    /// This is the shared implementation core for [`ImportDmaWl`]. It deliberately receives a
    /// release-ownership transfer closure so the renderer-managed Wayland buffer wrapper remains the
    /// production owner of the move-only release point until the Vulkan acquire helper reaches its
    /// ready-to-submit callback.
    /// Extracting this helper keeps the intended Wayland path testable while generic `ImportDma`
    /// remains fail-closed and without inventing a side-channel advertisement surface.
    #[cfg(feature = "wayland_frontend")]
    #[allow(dead_code)]
    fn import_wayland_dmabuf_with_policy_context<F>(
        &mut self,
        context: SampledDmabufWaylandVulkanInteropPolicyContext<'_>,
        release_ownership: F,
    ) -> Result<VulkanTexture, VulkanError>
    where
        F: FnOnce() -> Result<SampledDmabufReleaseOwnership, VulkanError>,
    {
        let dmabuf = context.dmabuf;
        let layout_evidence = self.validate_sampled_dmabuf_wayland_vulkan_interop_policy(&context)?;
        let foreign_general = self.validate_sampled_dmabuf_known_layout_contract(dmabuf, layout_evidence)?;

        let texture = unsafe {
            // SAFETY: The validation-stage Wayland policy above is the only production source of
            // layout evidence for this helper. The release-ownership closure is consumed only from
            // the Vulkan acquire ready-to-submit callback, after retry-safe setup has succeeded and
            // immediately before queue submission.
            self.create_imported_dmabuf_texture_with_known_general_layout_release_and_sync_point(
                dmabuf,
                foreign_general,
                Some(context.acquire_sync.sync()),
                release_ownership,
            )?
        };

        if let Some(texture) = texture {
            self.record_sampled_dmabuf_locally_acquired(dmabuf);
            Ok(texture)
        } else {
            Err(VulkanError::MissingCapability("sampled dmabuf texture import"))
        }
    }

    /// Complete a contextual sampled-dmabuf import from the typed validation-stage contract.
    ///
    /// Keeping this as a distinct step makes the intended future API shape explicit: callers first
    /// produce a context carrying sync, external-state, and lifecycle evidence; the renderer then
    /// validates that context and takes move-only release ownership at the Vulkan acquire
    /// ready-to-submit boundary. The normal Wayland path binds that take to the same renderer-managed
    /// buffer that produced the context, rather than accepting an arbitrary release-ownership source.
    #[cfg(feature = "wayland_frontend")]
    fn import_wayland_dmabuf_with_context(
        &mut self,
        context: SampledDmabufImportContext<'_>,
        buffer: &super::utils::Buffer,
    ) -> Result<VulkanTexture, VulkanError> {
        let dmabuf = context.dmabuf;
        let policy_context = context.wayland_policy_context();
        let linux_interop = self.wayland_linux_dmabuf_interop;
        self.import_wayland_dmabuf_with_policy_context(policy_context, || {
            Self::sampled_dmabuf_take_wayland_release_ownership(dmabuf, buffer, linux_interop)
        })
    }

    /// Release an acquired dmabuf render target back to foreign ownership in `GENERAL` layout.
    ///
    /// This is the release counterpart to the acquired dmabuf render-target helper used by public
    /// dmabuf render-target binding.
    #[allow(dead_code)]
    pub(crate) fn release_acquired_dmabuf_render_target_to_foreign_general(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        export_sync_file: bool,
    ) -> Result<(bool, Option<OwnedFd>), VulkanError> {
        self.release_acquired_dmabuf_render_target_to_foreign_general_classified(target, export_sync_file)
            .map_err(VulkanDmabufRenderTargetForeignReleaseError::into_inner)
    }

    #[allow(dead_code)]
    fn release_acquired_dmabuf_render_target_to_foreign_general_classified(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        export_sync_file: bool,
    ) -> Result<(bool, Option<OwnedFd>), VulkanDmabufRenderTargetForeignReleaseError> {
        if target.context_id != self.context_id {
            return Err(VulkanDmabufRenderTargetForeignReleaseError::RetrySafe(
                VulkanError::UnsupportedOperation("foreign dmabuf render target"),
            ));
        }
        if target.image.source != image::VulkanImageSource::RenderTarget {
            return Err(VulkanDmabufRenderTargetForeignReleaseError::RetrySafe(
                VulkanError::UnsupportedOperation("dmabuf render target"),
            ));
        }
        if !target.image.sync.is_locally_usable() {
            return Err(VulkanDmabufRenderTargetForeignReleaseError::RetrySafe(
                VulkanError::UnsupportedOperation("dmabuf external ownership"),
            ));
        }
        let color_image =
            target
                .color_image
                .as_ref()
                .ok_or(VulkanDmabufRenderTargetForeignReleaseError::RetrySafe(
                    VulkanError::UnsupportedOperation("dmabuf render target image"),
                ))?;
        let device = self
            .device
            .as_ref()
            .ok_or(VulkanDmabufRenderTargetForeignReleaseError::RetrySafe(
                VulkanError::VulkanUnavailable,
            ))?;
        let release = match device
            .release_dmabuf_render_target_to_foreign_general_classified(color_image, export_sync_file)
        {
            Ok(release) => release,
            Err(VulkanDmabufRenderTargetForeignReleaseError::ReleaseSubmitted(err)) => {
                target.image.layout = image::VulkanImageLayoutState::Undefined;
                target.image.sync = image::dmabuf_import_sync_state();
                return Err(VulkanDmabufRenderTargetForeignReleaseError::ReleaseSubmitted(err));
            }
            Err(err) => return Err(err),
        };
        if release.0 {
            // The foreign side now owns the image in GENERAL. Keep the renderer-facing layout
            // unusable until a later explicit acquire restores local color-attachment ownership.
            target.image.layout = image::VulkanImageLayoutState::Undefined;
            target.image.sync = image::VulkanImageSyncState::foreign_known_general_for_dmabuf_import();
        }

        Ok(release)
    }

    /// Release an acquired dmabuf render target and return the exported release fence as a
    /// [`SyncPoint`] when available.
    ///
    /// This is a crate-private helper for public dmabuf render-target binding and DRM error
    /// cleanup.
    #[allow(dead_code)]
    pub(crate) fn release_acquired_dmabuf_render_target_to_foreign_general_sync_point(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        export_sync_file: bool,
    ) -> Result<(bool, SyncPoint), VulkanError> {
        self.release_acquired_dmabuf_render_target_to_foreign_general_sync_point_classified(
            target,
            export_sync_file,
        )
        .map_err(VulkanDmabufRenderTargetForeignReleaseError::into_inner)
    }

    #[allow(dead_code)]
    fn release_acquired_dmabuf_render_target_to_foreign_general_sync_point_classified(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        export_sync_file: bool,
    ) -> Result<(bool, SyncPoint), VulkanDmabufRenderTargetForeignReleaseError> {
        let (released, sync_file) = self
            .release_acquired_dmabuf_render_target_to_foreign_general_classified(target, export_sync_file)?;
        Ok((released, sync_point_from_sync_file(sync_file)))
    }

    /// Release an acquired dmabuf render target and produce evidence for Smithay-controlled sampled loopback.
    ///
    /// This validation-stage helper is the bridge from the normal Vulkan dmabuf render-target path to
    /// the known-layout sampled-import path. It returns evidence only after the render target was
    /// released to foreign ownership in `VK_IMAGE_LAYOUT_GENERAL`, and the evidence is tied to the
    /// original dmabuf identity recorded when the target was acquired. It does not make arbitrary
    /// Wayland dmabufs public-advertised or supported through generic [`ImportDma`].
    #[allow(dead_code)]
    fn release_dmabuf_render_target_for_sampled_loopback(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        export_sync_file: bool,
    ) -> Result<Option<VulkanDmabufLoopbackImportEvidence>, VulkanError> {
        let dmabuf = target
            .dmabuf
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf loopback render target"))?
            .clone();
        let (released, acquire_sync) = self
            .release_acquired_dmabuf_render_target_to_foreign_general_sync_point(target, export_sync_file)?;
        if released {
            Ok(Some(unsafe {
                // SAFETY: `release_acquired_dmabuf_render_target_to_foreign_general_sync_point`
                // returned `released == true`, so this renderer submitted/completed the release of
                // the matching dmabuf render target to FOREIGN ownership in GENERAL layout. The
                // returned sync point carries the release dependency for later sampled reacquire.
                VulkanDmabufLoopbackImportEvidence::new(dmabuf, acquire_sync)
            }))
        } else {
            Ok(None)
        }
    }

    /// Release an acquired dmabuf render target for a later Wayland sampled-dmabuf import.
    ///
    /// This is the producer side of Smithay's controlled Vulkan-producer `ImportDmaWl` contract. On
    /// success it submits the release of `target` to `VK_QUEUE_FAMILY_FOREIGN_EXT` in
    /// `VK_IMAGE_LAYOUT_GENERAL` and returns an opaque token that the consumer renderer can later pass
    /// to
    /// `VulkanRenderer::admit_wayland_dmabuf_current_commit_from_vulkan_producer_release_for_sampled_import`
    /// after the dmabuf is committed through the normal Wayland linux-dmabuf/drm-syncobj path. If
    /// `export_sync_file` is true, the returned token's acquire sync should be imported into the
    /// Wayland acquire point for the committed buffer; if it is false, the caller must still provide an
    /// equivalent ordering guarantee before admission.
    ///
    /// This does not make arbitrary producer dmabufs or generic [`ImportDma`] public-advertised.
    #[allow(dead_code)]
    pub fn release_dmabuf_render_target_for_wayland_sampled_import(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        export_sync_file: bool,
    ) -> Result<Option<VulkanWaylandDmabufProducerRelease>, VulkanError> {
        self.release_dmabuf_render_target_for_sampled_loopback(target, export_sync_file)
    }

    /// Import a Smithay-controlled released dmabuf render target as a sampled texture.
    ///
    /// The consumed evidence must have been produced by
    /// [`VulkanRenderer::release_dmabuf_render_target_for_sampled_loopback`] for the same Smithay
    /// [`Dmabuf`] identity. This is a validation-stage loopback helper and does not advertise generic
    /// sampled dmabuf import.
    ///
    /// # Safety
    ///
    /// The caller must ensure there was no intervening access, acquire, release, or layout/ownership
    /// transition of the dmabuf after the evidence was produced. The dmabuf must still be in
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` ownership and `VK_IMAGE_LAYOUT_GENERAL`, with `evidence`'s
    /// acquire sync point representing the release dependency for this sampled import.
    #[allow(dead_code)]
    unsafe fn import_dmabuf_texture_from_loopback(
        &mut self,
        dmabuf: &Dmabuf,
        evidence: VulkanDmabufLoopbackImportEvidence,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        self.validate_sampled_dmabuf_public_import_lifecycle(dmabuf)?;
        let foreign_general = self.validate_dmabuf_loopback_import_evidence(dmabuf, &evidence)?;
        let acquire_sync = evidence.acquire_sync;
        let texture = unsafe {
            // SAFETY: Forwarded from this method's caller and validated against the consumed
            // loopback evidence's dmabuf identity above.
            self.create_imported_dmabuf_texture_with_known_general_layout_and_sync_point(
                dmabuf,
                foreign_general,
                Some(&acquire_sync),
            )?
        };
        if texture.is_some() {
            self.record_sampled_dmabuf_locally_acquired(dmabuf);
        }
        Ok(texture)
    }

    /// Release an acquired dmabuf render target and immediately import it as a sampled texture.
    ///
    /// This validation-stage loopback helper keeps the render-target release evidence internal to a
    /// single renderer call: it prevalidates the sampled import contract, releases `target` to foreign
    /// ownership in `VK_IMAGE_LAYOUT_GENERAL`, removes and drops the render-target object only after
    /// release evidence exists, waits any exported release fence internally, and consumes the resulting
    /// evidence for the sampled import without handing a raw known-layout token back to the caller. If
    /// release fails or produces no evidence, `target` remains available for ordinary error cleanup. If
    /// the release was submitted but later completion/export became an error, `target` is consumed
    /// because local ownership can no longer be retried safely. The target must still have been acquired
    /// through the explicit Vulkan dmabuf render-target path, so this does not make generic
    /// [`ImportDma`] or arbitrary producer dmabufs public-advertised.
    #[allow(dead_code)]
    pub(crate) fn release_dmabuf_render_target_and_import_sampled_loopback(
        &mut self,
        target: &mut Option<VulkanRenderTarget<'_>>,
        export_sync_file: bool,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        let target_ref = target
            .as_mut()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf loopback render target"))?;
        let weak_dmabuf = target_ref
            .dmabuf
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf loopback render target"))?
            .clone();
        let dmabuf = weak_dmabuf
            .upgrade()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf loopback render target"))?;
        self.validate_sampled_dmabuf_public_import_lifecycle(&dmabuf)?;
        let _ = self.validate_sampled_dmabuf_import_metadata(&dmabuf)?;
        let _ = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let (released, acquire_sync) = match self
            .release_acquired_dmabuf_render_target_to_foreign_general_sync_point_classified(
                target_ref,
                export_sync_file,
            ) {
            Ok(release) => release,
            Err(VulkanDmabufRenderTargetForeignReleaseError::RetrySafe(err)) => return Err(err),
            Err(VulkanDmabufRenderTargetForeignReleaseError::ReleaseSubmitted(err)) => {
                let released_target = target
                    .take()
                    .ok_or(VulkanError::UnsupportedOperation("dmabuf loopback render target"))?;
                drop(released_target);
                return Err(err);
            }
        };
        if !released {
            return Ok(None);
        }
        let released_target = target
            .take()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf loopback render target"))?;
        drop(released_target);
        let acquire_sync = if acquire_sync.contains_fence() {
            while let Err(err) = acquire_sync.wait() {
                tracing::warn!(
                    ?err,
                    "interrupted while waiting for dmabuf loopback release fence before sampled import"
                );
                std::thread::yield_now();
            }
            SyncPoint::signaled()
        } else {
            acquire_sync
        };
        let evidence = unsafe {
            // SAFETY: The classified release returned `released == true`, so this renderer submitted
            // the matching render-target release to FOREIGN ownership in GENERAL layout. Any exported
            // release fence was waited above, so no release dependency is handed out or stranded by
            // post-release sampled-import failure.
            VulkanDmabufLoopbackImportEvidence::new(weak_dmabuf, acquire_sync)
        };

        unsafe {
            // SAFETY: The evidence was just produced by this renderer from the consumed render
            // target, and no caller code can run between the release and this sampled acquire.
            self.import_dmabuf_texture_from_loopback(&dmabuf, evidence)
        }
    }

    /// Import a Smithay-controlled loopback dmabuf as a sampled texture with a release obligation.
    ///
    /// This test-only helper lets ignored runtime probes exercise the same Vulkan release hook used by
    /// renderer-utils surface-cache retirement without constructing a real Wayland `wl_buffer` and DRM
    /// syncobj release point. It remains validation-stage evidence: public `ImportDma` advertisement
    /// is unchanged, and production `ImportDmaWl` still obtains release ownership from the
    /// renderer-managed Wayland buffer wrapper.
    ///
    /// # Safety
    ///
    /// The caller must satisfy the same no-intervening-use and known `FOREIGN + GENERAL` external
    /// state requirements as [`VulkanRenderer::import_dmabuf_texture_from_loopback`].
    #[cfg(test)]
    unsafe fn import_dmabuf_texture_from_loopback_with_release_for_tests(
        &mut self,
        dmabuf: &Dmabuf,
        evidence: VulkanDmabufLoopbackImportEvidence,
        release_ownership: SampledDmabufReleaseOwnership,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        self.validate_sampled_dmabuf_public_import_lifecycle(dmabuf)?;
        let foreign_general = self.validate_dmabuf_loopback_import_evidence(dmabuf, &evidence)?;
        let acquire_sync = evidence.acquire_sync;
        let texture = unsafe {
            // SAFETY: Forwarded from this test-only helper's caller and validated against the
            // consumed loopback evidence's dmabuf identity above. The release obligation remains a
            // renderer-owned texture obligation and is validated against the same dmabuf before it is
            // attached.
            self.create_imported_dmabuf_texture_with_known_general_layout_release_and_sync_point(
                dmabuf,
                foreign_general,
                Some(&acquire_sync),
                || Ok(release_ownership),
            )?
        };
        if texture.is_some() {
            self.record_sampled_dmabuf_locally_acquired(dmabuf);
        }
        Ok(texture)
    }

    /// Release an explicitly acquired dmabuf render target after rendering failed before frame finish.
    ///
    /// This is the public error-cleanup counterpart to
    /// [`VulkanRenderer::bind_dmabuf_render_target`]. It returns the target to foreign ownership in
    /// `VK_IMAGE_LAYOUT_GENERAL` without exporting a release fence. Call this before reusing the
    /// dmabuf externally when rendering fails before
    /// [`Frame::finish`](crate::backend::renderer::Frame::finish).
    pub fn release_dmabuf_render_target_after_render_error(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
    ) -> Result<(), VulkanError> {
        self.release_acquired_dmabuf_render_target_to_foreign_general_sync_point(target, false)
            .map(|_| ())
    }

    #[allow(dead_code)]
    fn clear_offscreen_render_target(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
        color: Color32F,
    ) -> Result<(), VulkanError> {
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign offscreen target"));
        }
        if target.image.source != image::VulkanImageSource::Offscreen {
            return Err(VulkanError::UnsupportedOperation("offscreen target"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("offscreen target image"))?;
        let format = target
            .image
            .format
            .ok_or(VulkanError::UnsupportedOperation("offscreen target format"))?;

        device.clear_offscreen_color_image(color_image, clear_color_value_for_format(format, color)?)?;
        target.image.layout = image::VulkanImageLayoutState::TransferDst;
        Ok(())
    }

    #[allow(dead_code)]
    fn read_offscreen_render_target(
        &mut self,
        target: &mut VulkanRenderTarget<'_>,
    ) -> Result<Vec<u8>, VulkanError> {
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign offscreen target"));
        }
        if target.image.source != image::VulkanImageSource::Offscreen {
            return Err(VulkanError::UnsupportedOperation("offscreen target"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("offscreen target image"))?;

        let data = device.read_image_to_tightly_packed_buffer(color_image)?;
        target.image.layout = image::VulkanImageLayoutState::TransferSrc;
        Ok(data)
    }
}

impl RendererSuper for VulkanRenderer {
    type Error = VulkanError;
    type TextureId = VulkanTexture;
    type Framebuffer<'buffer> = VulkanRenderTarget<'buffer>;
    type Frame<'frame, 'buffer>
        = VulkanFrame<'frame, 'buffer>
    where
        'buffer: 'frame,
        Self: 'frame;
}

impl Renderer for VulkanRenderer {
    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.context_id.clone()
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.downscale_filter = filter;
        Ok(())
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.upscale_filter = filter;
        Ok(())
    }

    fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.debug_flags = flags;
    }

    fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    fn render<'frame, 'buffer>(
        &'frame mut self,
        framebuffer: &'frame mut Self::Framebuffer<'buffer>,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        if framebuffer.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign render target"));
        }
        if !matches!(
            framebuffer.image.source,
            image::VulkanImageSource::Offscreen
                | image::VulkanImageSource::RenderTarget
                | image::VulkanImageSource::Swapchain
        ) {
            return Err(VulkanError::UnsupportedOperation("render target"));
        }
        if output_size.w <= 0 || output_size.h <= 0 {
            return Err(VulkanError::UnsupportedOperation("frame size"));
        }
        if framebuffer.image.size.w != output_size.w || framebuffer.image.size.h != output_size.h {
            return Err(VulkanError::UnsupportedOperation("frame size"));
        }
        if framebuffer.image.source == image::VulkanImageSource::RenderTarget
            && !framebuffer.image.sync.is_locally_usable()
        {
            return Err(VulkanError::UnsupportedOperation("dmabuf external ownership"));
        }
        if framebuffer.color_image.is_none() {
            return Err(VulkanError::UnsupportedOperation("render target image"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;

        Ok(VulkanFrame {
            context_id: self.context_id.clone(),
            output_size,
            transform: dst_transform,
            device: Some(device),
            target: Some(framebuffer),
            _renderer: std::marker::PhantomData,
        })
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        sync.wait().map_err(|_| VulkanError::SyncInterrupted)
    }

    fn release_imported_texture_for_surface_cache(
        &mut self,
        texture: &Self::TextureId,
    ) -> Result<(), SurfaceCacheTextureReleaseError<Self::Error>> {
        self.release_retired_wayland_texture_for_cache(texture)
    }

    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        self.cleanup_pending_sampled_dmabuf_import_obligations()
    }
}

impl<'target> Bind<VulkanRenderTarget<'target>> for VulkanRenderer {
    fn bind<'a>(
        &mut self,
        target: &'a mut VulkanRenderTarget<'target>,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign render target"));
        }
        if !matches!(
            target.image.source,
            image::VulkanImageSource::Offscreen | image::VulkanImageSource::Swapchain
        ) {
            return Err(VulkanError::UnsupportedOperation("render target"));
        }
        if target.color_image.is_none() {
            return Err(VulkanError::UnsupportedOperation("render target image"));
        }

        Ok(VulkanRenderTarget {
            context_id: target.context_id.clone(),
            image: target.image.clone(),
            color_image: target.color_image.clone(),
            dmabuf: target.dmabuf.clone(),
            _target: std::marker::PhantomData,
        })
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.render_target_formats())
    }
}

impl<'target, 'sync> Bind<VulkanDmabufRenderTarget<'target, 'sync>> for VulkanRenderer {
    fn bind<'a>(
        &mut self,
        target: &'a mut VulkanDmabufRenderTarget<'target, 'sync>,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        unsafe {
            // SAFETY: `VulkanDmabufRenderTarget` can only be constructed by callers that accepted
            // and upheld the explicit Vulkan external-memory acquire contract. Reborrow the wrapped
            // dmabuf for the lifetime of the returned framebuffer so Rust prevents rebinding through
            // this wrapper while the framebuffer is live.
            self.bind_dmabuf_render_target(&mut *target.dmabuf, target.acquire)
        }?
        .ok_or(VulkanError::MissingCapability(
            "dmabuf render target format/modifier",
        ))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.development_gated_dmabuf_render_target_formats())
    }
}

impl<'target, 'sync> RenderTargetLifecycle<VulkanDmabufRenderTarget<'target, 'sync>> for VulkanRenderer {
    fn target_age(&self, target: &VulkanDmabufRenderTarget<'target, 'sync>, age: usize) -> usize {
        if target.preserve_contents() { age } else { 0 }
    }

    fn release_after_render_error(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }

    fn release_after_no_render(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }
}

impl<'sync> Bind<VulkanOwnedDmabufRenderTarget<'sync>> for VulkanRenderer {
    fn bind<'a>(
        &mut self,
        target: &'a mut VulkanOwnedDmabufRenderTarget<'sync>,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        unsafe {
            // SAFETY: `VulkanOwnedDmabufRenderTarget` can only be constructed by callers that
            // accepted and upheld the explicit Vulkan external-memory acquire contract. Borrow the
            // owned dmabuf for the lifetime of the returned framebuffer so Rust prevents rebinding
            // through this wrapper while the framebuffer is live.
            self.bind_dmabuf_render_target(&mut target.dmabuf, target.acquire)
        }?
        .ok_or(VulkanError::MissingCapability(
            "dmabuf render target format/modifier",
        ))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.development_gated_dmabuf_render_target_formats())
    }
}

impl<'sync> RenderTargetLifecycle<VulkanOwnedDmabufRenderTarget<'sync>> for VulkanRenderer {
    fn target_age(&self, target: &VulkanOwnedDmabufRenderTarget<'sync>, age: usize) -> usize {
        if target.preserve_contents() { age } else { 0 }
    }

    fn release_after_render_error(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }

    fn release_after_no_render(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }
}

impl<'target> Bind<VulkanAllocatorDmabufRenderTargetContract<'target>> for VulkanRenderer {
    fn bind<'a>(
        &mut self,
        target: &'a mut VulkanAllocatorDmabufRenderTargetContract<'target>,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        let evidence = target.take_release_evidence()?;
        if !evidence.is_for_dmabuf(target.dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "allocator dmabuf release evidence",
            ));
        }

        unsafe {
            // SAFETY: `VulkanAllocatorDmabufRenderTargetContract` can only be constructed from
            // allocator release evidence for this dmabuf. The caller of the future DRM integration
            // point must still ensure no intervening access, ownership transfer, or layout transition
            // occurs between contract construction and this bind.
            self.bind_dmabuf_render_target(
                &mut *target.dmabuf,
                VulkanDmabufRenderTargetAcquire::preserve(None),
            )
        }?
        .ok_or(VulkanError::MissingCapability(
            "dmabuf render target format/modifier",
        ))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.development_gated_dmabuf_render_target_formats())
    }
}

impl<'target> RenderTargetLifecycle<VulkanAllocatorDmabufRenderTargetContract<'target>> for VulkanRenderer {
    fn target_age(&self, _target: &VulkanAllocatorDmabufRenderTargetContract<'target>, age: usize) -> usize {
        age
    }

    fn release_after_render_error(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }

    fn release_after_no_render(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        if !self.capabilities.rendering.dmabuf_targets {
            return Err(VulkanError::NotPublicAdvertised("dmabuf render target"));
        }

        unsafe {
            // SAFETY: Public `Bind<Dmabuf>` is the compositor GBM scanout contract: the caller
            // owns the buffer for this frame, previous contents are discarded, and no acquire
            // fence is required. This is not sampled client-buffer import. Successful frames
            // release the image for KMS from `Frame::finish`; failed or skipped renders are
            // handled by `RenderTargetLifecycle<Dmabuf>` below.
            self.bind_dmabuf_render_target(target, VulkanDmabufRenderTargetAcquire::discard())
        }?
        .ok_or(VulkanError::MissingCapability(
            "dmabuf render target format/modifier",
        ))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.public_dmabuf_render_target_formats())
    }
}

impl RenderTargetLifecycle<Dmabuf> for VulkanRenderer {
    fn target_age(&self, _target: &Dmabuf, _age: usize) -> usize {
        // The generic dmabuf binding currently uses discard acquire, so preserved contents are not
        // part of the contract. Force full repaint until a future preserve/acquire-sync policy is
        // modeled in the standard path.
        0
    }

    fn release_after_render_error(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }

    fn release_after_no_render(&mut self, target: &mut Self::Framebuffer<'_>) -> Result<(), Self::Error> {
        self.release_dmabuf_render_target_after_render_error(target)
    }
}

fn validate_dmabuf_render_target_metadata(
    target: &Dmabuf,
) -> Result<image::VulkanDmabufImportState, VulkanError> {
    let import = image::VulkanDmabufImportState::from_dmabuf(target)?;
    if import.plane_count() != 1 {
        return Err(VulkanError::UnsupportedOperation("dmabuf render target planes"));
    }

    Ok(import)
}

impl Offscreen<VulkanRenderTarget<'static>> for VulkanRenderer {
    fn create_buffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<VulkanRenderTarget<'static>, Self::Error> {
        if !self.render_target_format_supported(format) {
            return Err(VulkanError::UnsupportedFormat(format));
        }

        self.create_offscreen_render_target(format, size)
    }
}

impl ImportDma for VulkanRenderer {
    fn dmabuf_formats(&self) -> FormatSet {
        self.public_dmabuf_import_formats()
    }

    fn import_dmabuf(
        &mut self,
        _dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        Err(VulkanError::MissingCapability(
            "sampled dmabuf generic ImportDma external-state contract",
        ))
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportDmaWl for VulkanRenderer {
    fn import_dma_buffer_from_surface_state(
        &mut self,
        buffer: &super::utils::Buffer,
        surface: Option<&crate::wayland::compositor::SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        self.admit_wayland_linux_dmabuf_interop_for_sampled_import(buffer)?;
        let context = self.wayland_sampled_dmabuf_import_context(buffer, surface)?;
        self.import_wayland_dmabuf_with_context(context, buffer)
    }
}

#[cfg(all(
    feature = "wayland_frontend",
    feature = "backend_egl",
    feature = "use_system_lib"
))]
impl ImportAll for VulkanRenderer {
    fn import_buffer(
        &mut self,
        buffer: &wl_buffer::WlBuffer,
        surface: Option<&crate::wayland::compositor::SurfaceData>,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Option<Result<Self::TextureId, Self::Error>> {
        super::import_shm_dmabuf_buffer(self, buffer, surface, damage)
    }

    fn import_buffer_from_surface_state(
        &mut self,
        buffer: &super::utils::Buffer,
        surface: Option<&crate::wayland::compositor::SurfaceData>,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Option<Result<Self::TextureId, Self::Error>> {
        super::import_shm_dmabuf_buffer_from_surface_state(self, buffer, surface, damage)
    }
}

impl ImportMem for VulkanRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let memory_format = Format {
            code: format,
            modifier: Modifier::Invalid,
        };
        if !self.capabilities.formats.memory_import.contains(&memory_format) {
            return Err(VulkanError::UnsupportedFormat(format));
        }

        let vk_format = get_render_vk_format(format)?;
        let extent = extent_from_size(size, "memory import size")?;
        let required_size = tightly_packed_image_size(vk_format, extent)?;
        if (data.len() as u64) < required_size {
            return Err(VulkanError::UnsupportedOperation("memory import data"));
        }
        let required_len = usize::try_from(required_size)
            .map_err(|_| VulkanError::UnsupportedOperation("memory import data size"))?;

        let sampled_image = device.create_uploaded_sampled_image(
            extent,
            vk_format,
            &data[..required_len],
            self.downscale_filter,
            self.upscale_filter,
        )?;

        Ok(VulkanTexture::from_sampled_image(
            self.context_id.clone(),
            size,
            format,
            sampled_image,
            flipped,
        ))
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        data: &[u8],
        region: Rectangle<i32, BufferCoord>,
    ) -> Result<(), Self::Error> {
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign memory texture"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        if texture.image.source != image::VulkanImageSource::MemoryUpload {
            return Err(VulkanError::UnsupportedOperation("memory texture"));
        }
        let sampled_image = texture
            .sampled_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("memory texture"))?;
        let format = texture
            .image
            .format
            .ok_or(VulkanError::UnsupportedOperation("memory texture format"))?;
        let vk_format = get_render_vk_format(format)?;
        let texture_extent = extent_from_size(texture.image.size, "memory update size")?;
        let (image_offset, region_extent) = update_region_to_vk(texture.image.size, region)?;
        let required_size = tightly_packed_image_size(vk_format, texture_extent)?;
        if (data.len() as u64) < required_size {
            return Err(VulkanError::UnsupportedOperation("memory update data"));
        }
        let required_len = usize::try_from(required_size)
            .map_err(|_| VulkanError::UnsupportedOperation("memory update data size"))?;
        let buffer_offset = image_copy_buffer_offset(
            vk_format,
            texture_extent.width,
            image_offset.x as u32,
            image_offset.y as u32,
        )?;

        device.update_uploaded_image_region(
            sampled_image.image(),
            &data[..required_len],
            buffer_offset,
            texture_extent.width,
            texture_extent.height,
            image_offset,
            region_extent,
        )
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        Box::new(
            self.capabilities
                .formats
                .memory_import
                .iter()
                .map(|format| format.code)
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportMemWl for VulkanRenderer {
    fn import_shm_buffer(
        &mut self,
        buffer: &wl_buffer::WlBuffer,
        _surface: Option<&crate::wayland::compositor::SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        let texture = crate::wayland::shm::with_buffer_contents(buffer, |ptr, len, data| {
            let fourcc = crate::wayland::shm::shm_format_to_fourcc(data.format)
                .ok_or(VulkanError::UnsupportedOperation("wl_shm format"))?;
            let bits_per_pixel = crate::backend::allocator::format::get_bpp(fourcc)
                .ok_or(VulkanError::UnsupportedFormat(fourcc))?;
            if bits_per_pixel % 8 != 0 {
                return Err(VulkanError::UnsupportedFormat(fourcc));
            }
            let bytes_per_pixel = bits_per_pixel / 8;
            let packed = copy_shm_buffer_to_tightly_packed(
                data.offset,
                data.width,
                data.height,
                data.stride,
                bytes_per_pixel,
                ptr,
                len,
            )?;

            self.import_memory(&packed, fourcc, (data.width, data.height).into(), false)
        })
        .map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer"))??;

        Ok(texture)
    }
}

impl ExportMem for VulkanRenderer {
    type TextureMapping = VulkanMemoryMapping;

    fn copy_framebuffer(
        &mut self,
        target: &Self::Framebuffer<'_>,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign render target"));
        }
        if target.image.source != image::VulkanImageSource::Offscreen {
            return Err(VulkanError::UnsupportedOperation("render target"));
        }
        let target_format = target
            .format()
            .ok_or(VulkanError::UnsupportedOperation("render target format"))?;
        if format != target_format {
            return Err(VulkanError::UnsupportedFormat(format));
        }

        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("render target image"))?;
        let (image_offset, extent) = image_region_to_vk(target.size(), region, "framebuffer copy region")?;
        let data = device.read_image_region_to_tightly_packed_buffer(color_image, image_offset, extent)?;

        Ok(VulkanMemoryMapping {
            data,
            size: region.size,
            format,
            flipped: false,
        })
    }

    fn copy_texture(
        &mut self,
        texture: &Self::TextureId,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign memory texture"));
        }
        let sampled = texture
            .sampled_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("texture memory export"))?;
        let image = sampled.image();
        if !image.usage().contains(vk::ImageUsageFlags::TRANSFER_SRC) {
            return Err(VulkanError::UnsupportedOperation("texture transfer source"));
        }
        let texture_format = texture
            .format()
            .ok_or(VulkanError::UnsupportedOperation("texture format"))?;
        if format != texture_format {
            return Err(VulkanError::UnsupportedFormat(format));
        }

        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let (image_offset, extent) = image_region_to_vk(texture.size(), region, "texture copy region")?;
        let data = device.read_image_region_to_tightly_packed_buffer(image, image_offset, extent)?;

        Ok(VulkanMemoryMapping {
            data,
            size: region.size,
            format,
            flipped: texture.y_inverted,
        })
    }

    fn can_read_texture(&mut self, texture: &Self::TextureId) -> Result<bool, Self::Error> {
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign memory texture"));
        }

        Ok(texture.sampled_image.as_ref().is_some_and(|sampled| {
            sampled
                .image()
                .usage()
                .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        }))
    }

    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        Ok(&texture_mapping.data)
    }
}

fn extent_from_size(size: Size<i32, BufferCoord>, error: &'static str) -> Result<vk::Extent3D, VulkanError> {
    if size.w <= 0 || size.h <= 0 {
        return Err(VulkanError::UnsupportedOperation(error));
    }

    Ok(vk::Extent3D {
        width: size
            .w
            .try_into()
            .map_err(|_| VulkanError::UnsupportedOperation(error))?,
        height: size
            .h
            .try_into()
            .map_err(|_| VulkanError::UnsupportedOperation(error))?,
        depth: 1,
    })
}

fn physical_rect_to_buffer(rect: Rectangle<i32, Physical>) -> Rectangle<i32, BufferCoord> {
    Rectangle::new((rect.loc.x, rect.loc.y).into(), (rect.size.w, rect.size.h).into())
}

fn texture_filter_to_vk(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Nearest => vk::Filter::NEAREST,
        TextureFilter::Linear => vk::Filter::LINEAR,
    }
}

fn blit_render_targets(
    device: &device::VulkanDeviceState,
    from: &VulkanRenderTarget<'_>,
    to: &mut VulkanRenderTarget<'_>,
    src: Rectangle<i32, Physical>,
    dst: Rectangle<i32, Physical>,
    filter: TextureFilter,
) -> Result<SyncPoint, VulkanError> {
    if from.context_id != to.context_id {
        return Err(VulkanError::UnsupportedOperation("foreign render target"));
    }
    let src_image = from
        .color_image
        .as_ref()
        .ok_or(VulkanError::UnsupportedOperation("blit source image"))?;
    let dst_image = to
        .color_image
        .as_ref()
        .ok_or(VulkanError::UnsupportedOperation("blit destination image"))?;
    if src_image.image() == dst_image.image() {
        return Err(VulkanError::UnsupportedOperation("blit same target"));
    }

    let (src_offset, src_extent) =
        image_region_to_vk(from.size(), physical_rect_to_buffer(src), "blit source region")?;
    let (dst_offset, dst_extent) =
        image_region_to_vk(to.size(), physical_rect_to_buffer(dst), "blit destination region")?;

    device.blit_owned_images(
        src_image,
        dst_image,
        src_offset,
        src_extent,
        dst_offset,
        dst_extent,
        texture_filter_to_vk(filter),
    )?;

    Ok(SyncPoint::signaled())
}

impl Blit for VulkanRenderer {
    fn blit(
        &mut self,
        from: &Self::Framebuffer<'_>,
        to: &mut Self::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        if !self.capabilities.rendering.blit {
            return Err(VulkanError::MissingCapability("blit"));
        }
        if from.context_id != self.context_id || to.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign render target"));
        }
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        blit_render_targets(device, from, to, src, dst, filter)
    }
}

impl<'buffer> BlitFrame<VulkanRenderTarget<'buffer>> for VulkanFrame<'_, 'buffer> {
    fn blit_to(
        &mut self,
        to: &mut VulkanRenderTarget<'buffer>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        let device = self.device.ok_or(VulkanError::VulkanUnavailable)?;
        let from = self
            .target
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("blit frame target"))?;
        blit_render_targets(device, from, to, src, dst, filter)
    }

    fn blit_from(
        &mut self,
        from: &VulkanRenderTarget<'buffer>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        let device = self.device.ok_or(VulkanError::VulkanUnavailable)?;
        let to = self
            .target
            .as_mut()
            .ok_or(VulkanError::UnsupportedOperation("blit frame target"))?;
        blit_render_targets(device, from, to, src, dst, filter)
    }
}

fn image_region_to_vk(
    image_size: Size<i32, BufferCoord>,
    region: Rectangle<i32, BufferCoord>,
    error: &'static str,
) -> Result<(vk::Offset3D, vk::Extent3D), VulkanError> {
    if region.loc.x < 0 || region.loc.y < 0 || region.size.w <= 0 || region.size.h <= 0 {
        return Err(VulkanError::UnsupportedOperation(error));
    }
    if region
        .loc
        .x
        .checked_add(region.size.w)
        .is_none_or(|right| right > image_size.w)
        || region
            .loc
            .y
            .checked_add(region.size.h)
            .is_none_or(|bottom| bottom > image_size.h)
    {
        return Err(VulkanError::UnsupportedOperation(error));
    }

    Ok((
        vk::Offset3D {
            x: region.loc.x,
            y: region.loc.y,
            z: 0,
        },
        vk::Extent3D {
            width: region
                .size
                .w
                .try_into()
                .map_err(|_| VulkanError::UnsupportedOperation(error))?,
            height: region
                .size
                .h
                .try_into()
                .map_err(|_| VulkanError::UnsupportedOperation(error))?,
            depth: 1,
        },
    ))
}

#[allow(dead_code)]
fn copy_shm_buffer_to_tightly_packed(
    offset: i32,
    width: i32,
    height: i32,
    stride: i32,
    bytes_per_pixel: usize,
    ptr: *const u8,
    len: usize,
) -> Result<Vec<u8>, VulkanError> {
    if offset < 0 || width <= 0 || height <= 0 || stride < 0 || bytes_per_pixel == 0 {
        return Err(VulkanError::UnsupportedOperation("wl_shm buffer layout"));
    }
    let offset =
        usize::try_from(offset).map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let width =
        usize::try_from(width).map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let height =
        usize::try_from(height).map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let stride =
        usize::try_from(stride).map_err(|_| VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let row_len = width
        .checked_mul(bytes_per_pixel)
        .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    if stride < row_len {
        return Err(VulkanError::UnsupportedOperation("wl_shm buffer layout"));
    }
    let last_row = height
        .checked_sub(1)
        .and_then(|row| row.checked_mul(stride))
        .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let required_len = offset
        .checked_add(last_row)
        .and_then(|start| start.checked_add(row_len))
        .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    if required_len > len {
        return Err(VulkanError::UnsupportedOperation("wl_shm buffer length"));
    }
    if ptr.is_null() {
        return Err(VulkanError::UnsupportedOperation("wl_shm buffer"));
    }

    let packed_len = row_len
        .checked_mul(height)
        .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
    let mut packed = vec![0; packed_len];
    for row in 0..height {
        let row_start = offset
            .checked_add(
                row.checked_mul(stride)
                    .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?,
            )
            .ok_or(VulkanError::UnsupportedOperation("wl_shm buffer layout"))?;
        // SAFETY: `with_buffer_contents` provides a raw pointer valid for `len` bytes for the
        // duration of the callback. Bounds above prove `row_start..row_start + row_len` is inside
        // that range, and `packed` is an owned allocation large enough for this row. Use a raw
        // pointer copy instead of creating a Rust slice/reference into client-controlled shared
        // memory, which may be mutated by the client while this copy runs.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.add(row_start),
                packed.as_mut_ptr().add(row * row_len),
                row_len,
            );
        }
    }

    Ok(packed)
}

fn update_region_to_vk(
    texture_size: Size<i32, BufferCoord>,
    region: Rectangle<i32, BufferCoord>,
) -> Result<(vk::Offset3D, vk::Extent3D), VulkanError> {
    if region.loc.x < 0 || region.loc.y < 0 || region.size.w <= 0 || region.size.h <= 0 {
        return Err(VulkanError::UnsupportedOperation("memory update region"));
    }
    if region
        .loc
        .x
        .checked_add(region.size.w)
        .is_none_or(|right| right > texture_size.w)
        || region
            .loc
            .y
            .checked_add(region.size.h)
            .is_none_or(|bottom| bottom > texture_size.h)
    {
        return Err(VulkanError::UnsupportedOperation("memory update region"));
    }

    Ok((
        vk::Offset3D {
            x: region.loc.x,
            y: region.loc.y,
            z: 0,
        },
        vk::Extent3D {
            width: region
                .size
                .w
                .try_into()
                .map_err(|_| VulkanError::UnsupportedOperation("memory update region"))?,
            height: region
                .size
                .h
                .try_into()
                .map_err(|_| VulkanError::UnsupportedOperation("memory update region"))?,
            depth: 1,
        },
    ))
}

fn clear_color_value_for_format(format: Fourcc, color: Color32F) -> Result<vk::ClearColorValue, VulkanError> {
    let mut components = color.components();
    if get_format_info(format)?.opaque_alpha {
        components[3] = 1.0;
    }

    Ok(vk::ClearColorValue { float32: components })
}

#[cfg(test)]
mod tests;
