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
//! most complete path. Dmabuf render-target support is development-gated in this fork, and its Vulkan
//! external-ownership and synchronization preconditions are explicit on
//! [`VulkanRenderer::bind_dmabuf_render_target`]. Generic [`Bind<Dmabuf>`] uses that path with a
//! conservative discard/full-repaint acquire policy so DRM/GBM compositor rendering follows the same
//! target abstraction as the other renderers once the validation-stage gate is true. Generic
//! texture `ExportMem`, `ExportDma`, broad explicit sync, blit/copy, and full presentation remain
//! unsupported until their corresponding capability bits can become true with coverage. Sampled dmabuf
//! import is validation-reachable through the explicit known-layout development helper, a local-only
//! experimental generic `ImportDma` path that assumes `FOREIGN + GENERAL` external state, and the
//! normal `ImportDmaWl` path's staged guards. The normal Wayland path now models the policy context,
//! explicit acquire/release sync evidence, first-import vs. reacquire history, known-layout evidence
//! identity, and renderer cache-release hook evidence. Public sampled `ImportDma` advertisement still
//! remains closed until production external-state evidence, acquire/release lifecycle, teardown cache
//! release, and direct import implementation contracts are proven by tests.
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
//! 10. crop/scale/transform
//! 11. readback test path
//! 12. `ImportDma`
//! 13. dmabuf modifier handling
//! 14. export dmabuf
//! 15. KMS presentation path
//! 16. explicit sync
//! 17. blit/copy
//! 18. multi-GPU integration
//! 19. colour-capable render targets
//! 20. HDR-ready hooks
//!
//! Every future feature should follow this pattern: capability flag first, test second, stub
//! failure path third, real implementation fourth, enablement last.

use std::{
    collections::HashMap,
    os::fd::{AsFd, BorrowedFd, OwnedFd},
    sync::Mutex,
};

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
    backend::vulkan::PhysicalDevice,
    backend::{
        allocator::{
            Format, Fourcc, Modifier,
            dmabuf::{Dmabuf, WeakDmabuf},
            format::FormatSet,
            vulkan::VulkanAllocatorDmabufForeignReleaseEvidence,
        },
        renderer::{
            Bind, Color32F, ContextId, DebugFlags, ExportMem, ImportDma, ImportMem, Offscreen,
            RenderTargetLifecycle, Renderer, RendererSuper, SurfaceCacheTextureReleaseError, Texture,
            TextureFilter,
            sync::{Fence, Interrupted, SyncPoint},
        },
    },
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
};
use ash::vk;
#[cfg(feature = "wayland_frontend")]
use wayland_server::protocol::wl_buffer;

mod capabilities;
mod device;
mod error;
pub mod format;
mod image;

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
struct SampledDmabufKnownLayoutEvidence {
    dmabuf: WeakDmabuf,
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
        Self { dmabuf }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
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

/// Commit-local proof that a Wayland dmabuf is ready for Vulkan sampled import.
///
/// This evidence is stored on Smithay's renderer-managed Wayland buffer wrapper. It is deliberately
/// separate from linux-dmabuf metadata and explicit-sync points: those describe buffer layout data and
/// ordering, but not the Vulkan image layout or queue-family ownership needed by the sampled import
/// acquire barrier.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
struct SampledDmabufWaylandForeignGeneralEvidence {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufWaylandForeignGeneralEvidence {
    unsafe fn new(dmabuf: WeakDmabuf) -> Self {
        Self { dmabuf }
    }

    #[cfg(test)]
    fn new_for_tests(dmabuf: &Dmabuf) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn first_import_layout(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<SampledDmabufWaylandFirstImportLayoutEvidence, VulkanError> {
        if !self.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state identity",
            ));
        }
        Ok(SampledDmabufWaylandFirstImportLayoutEvidence {
            dmabuf: self.dmabuf.clone(),
        })
    }

    fn current_reacquire_layout(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<SampledDmabufWaylandCurrentReacquireLayoutEvidence, VulkanError> {
        if !self.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state identity",
            ));
        }
        Ok(SampledDmabufWaylandCurrentReacquireLayoutEvidence {
            dmabuf: self.dmabuf.clone(),
        })
    }

    fn known_foreign_general(
        &self,
        dmabuf: &Dmabuf,
    ) -> Result<SampledDmabufKnownLayoutEvidence, VulkanError> {
        if !self.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "sampled dmabuf Wayland external-state identity",
            ));
        }
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
/// This is deliberately private and currently unconstructable in production code. The surrounding
/// scaffold already models the policy inputs separately: first-import/reacquire external state,
/// queue-family ownership, acquire synchronization, release synchronization, and per-commit texture
/// cache behavior. Keeping this as a separate evidence token prevents future work from treating
/// `linux-dmabuf` protocol metadata or explicit sync alone as a Vulkan layout/ownership proof, and
/// keeps the remaining production evidence sources and release lifecycle hooks explicit.
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
}

#[allow(dead_code)]
impl SampledDmabufWaylandCurrentReacquireLayoutEvidence {
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
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SampledDmabufAcquireSyncEvidence {
    dmabuf: WeakDmabuf,
    sync: SyncPoint,
}

#[allow(dead_code)]
impl SampledDmabufAcquireSyncEvidence {
    fn new(dmabuf: &Dmabuf, sync: SyncPoint) -> Self {
        Self {
            dmabuf: dmabuf.weak(),
            sync,
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn sync(&self) -> &SyncPoint {
        &self.sync
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
}

#[allow(dead_code)]
impl SampledDmabufWaylandTextureCacheReplacementReleaseReachability {
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

/// Evidence that the renderer surface-cache release hook can release sampled dmabuf textures.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct SampledDmabufWaylandTextureCacheReleaseHook {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufWaylandTextureCacheReleaseHook {
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
/// imported the sampled dmabuf texture from the Wayland surface cache.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
struct SampledDmabufWaylandTextureCacheReleaseLifecycleEvidence {
    dmabuf: WeakDmabuf,
    renderer_context: ContextId<VulkanTexture>,
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
    unsafe fn new(dmabuf: WeakDmabuf, renderer_context: ContextId<VulkanTexture>) -> Self {
        Self {
            dmabuf,
            renderer_context,
        }
    }

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn is_for_renderer_context(&self, renderer_context: &ContextId<VulkanTexture>) -> bool {
        &self.renderer_context == renderer_context
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

/// Renderer-local layout history available for a normal Wayland sampled-dmabuf commit.
///
/// Wayland explicit sync can order producer completion, but it does not describe Vulkan image
/// layout. This history is intentionally separate from sync evidence so future reacquire work must
/// prove when the renderer is relying on its own prior foreign release rather than first-import
/// protocol metadata.
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
/// This is only protocol-handle evidence. It does not prove Vulkan has finished sampling, released
/// the image back to foreign ownership, or signaled/satisfied the Wayland release point.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SampledDmabufReleaseEvidence {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufReleaseEvidence {
    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }
}

/// Evidence that the Wayland release point is still available for renderer ownership transfer.
///
/// This cloneable token is a precondition marker, not the moved release obligation. The actual
/// `Buffer::take_release_point_for_renderer()` transfer stays at the late texture-construction
/// boundary, after earlier policy guards have accepted the import path.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SampledDmabufReleaseOwnershipEvidence {
    dmabuf: WeakDmabuf,
}

#[allow(dead_code)]
impl SampledDmabufReleaseOwnershipEvidence {
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

/// Renderer-owned sampled dmabuf release obligation.
#[allow(dead_code)]
#[derive(Debug)]
struct SampledDmabufReleaseOwnership {
    dmabuf: WeakDmabuf,
    release: image::VulkanSampledDmabufRelease,
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

    fn is_for_dmabuf(&self, dmabuf: &Dmabuf) -> bool {
        self.dmabuf.upgrade().as_ref() == Some(dmabuf)
    }

    fn into_release(self) -> image::VulkanSampledDmabufRelease {
        self.release
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

/// Opaque evidence that a Smithay-controlled Vulkan dmabuf render target was released for sampled import.
///
/// This token is produced only after this renderer releases an acquired dmabuf render target to
/// `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`. It is tied to the originating
/// [`Dmabuf`] identity and is consumed by the validation-stage loopback sampled-import helper. This
/// keeps the Smithay-controlled loopback path separate from arbitrary Wayland client dmabufs and does
/// not public-advertise generic [`ImportDma`] support.
#[derive(Debug)]
pub struct VulkanDmabufLoopbackImportEvidence {
    dmabuf: WeakDmabuf,
    acquire_sync: SyncPoint,
    foreign_general: SampledDmabufKnownLayoutEvidence,
}

impl VulkanDmabufLoopbackImportEvidence {
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

use self::{
    device::{
        VulkanDeviceState, VulkanSampledDmabufForeignReleaseError, VulkanSyncFileSemaphore,
        image_copy_buffer_offset, tightly_packed_image_size,
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
    sampled_dmabuf_layout_history: HashMap<WeakDmabuf, SampledDmabufWaylandLayoutHistory>,
}

/// Builder for explicit Vulkan renderer initialization.
///
/// The builder initializes a logical Vulkan device from an explicitly provided [`PhysicalDevice`].
/// Public operations remain limited to the capability bits advertised by the initialized renderer.
#[derive(Debug, Default, Clone)]
pub struct VulkanRendererBuilder {
    physical_device: Option<PhysicalDevice>,
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

    /// Builds a Vulkan renderer.
    ///
    /// This returns [`VulkanError::VulkanUnavailable`] unless a [`PhysicalDevice`] was provided.
    pub fn build(self) -> Result<VulkanRenderer, VulkanError> {
        let physical_device = self.physical_device.ok_or(VulkanError::VulkanUnavailable)?;
        let device = VulkanDeviceState::new(physical_device)?;
        let capabilities = device.capabilities.clone();

        Ok(VulkanRenderer {
            context_id: ContextId::new(),
            debug_flags: DebugFlags::empty(),
            downscale_filter: TextureFilter::Linear,
            upscale_filter: TextureFilter::Linear,
            capabilities,
            device: Some(device),
            sampled_dmabuf_layout_history: HashMap::new(),
        })
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
            Self::mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(buffer.user_data(), dmabuf)
        }
    }

    /// Mark a renderer-managed Wayland dmabuf commit for validation-stage sampled import.
    ///
    /// This is the intended development contract for the normal [`ImportDmaWl`] path. It stores both
    /// pieces of compositor-provided evidence that cannot be inferred from Wayland protocol metadata:
    /// current-commit `FOREIGN + GENERAL` Vulkan external image state and renderer-utils
    /// texture-cache release lifecycle coverage for this renderer context. It does not consume the
    /// Wayland release point, create a Vulkan image, or public-advertise generic sampled [`ImportDma`]
    /// support.
    ///
    /// # Safety
    ///
    /// The caller must satisfy both lower-level contracts for this same `buffer`/`dmabuf` pair and
    /// this renderer context: current-commit external-state evidence and renderer-utils
    /// texture-cache release lifecycle coverage. In particular, the proof of `FOREIGN + GENERAL`
    /// ownership/layout must be current for this commit, and the compositor must preserve or retry any
    /// cache-release obligation before dropping/resetting the surface state.
    #[cfg(feature = "wayland_frontend")]
    pub unsafe fn mark_wayland_dmabuf_current_commit_for_sampled_import(
        &self,
        buffer: &super::utils::Buffer,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Forwarded from this combined validation contract's caller.
            Self::mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
                buffer.user_data(),
                dmabuf,
            )?;
            // SAFETY: Forwarded from this combined validation contract's caller.
            self.mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
                buffer.user_data(),
                dmabuf,
            )
        }
    }

    #[cfg(feature = "wayland_frontend")]
    unsafe fn mark_wayland_dmabuf_user_data_foreign_general_for_sampled_import(
        user_data: &UserDataMap,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        let slot =
            user_data.get_or_insert_threadsafe(SampledDmabufWaylandForeignGeneralEvidenceSlot::default);
        let mut evidence = slot.evidence.lock().map_err(|_| {
            VulkanError::UnsupportedOperation("sampled dmabuf Wayland external-state evidence")
        })?;
        *evidence = Some(unsafe {
            // SAFETY: Forwarded from this unsafe evidence-marking helper's caller.
            SampledDmabufWaylandForeignGeneralEvidence::new(dmabuf.weak())
        });
        Ok(())
    }

    /// Mark a renderer-managed Wayland dmabuf buffer as covered by texture-cache release call sites.
    ///
    /// This is a development-stage lifecycle evidence API for the normal [`ImportDmaWl`] path. It
    /// does not public-advertise sampled [`ImportDma`] support and is not inferred from Wayland
    /// protocol state. This marker proves only that the compositor has wired the no-next-import and
    /// teardown release call sites for this renderer/context; it does not prove external image state,
    /// acquire synchronization, release synchronization, or public import support.
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
    /// until the obligation is retried or otherwise preserved. No active frame, render element, clone,
    /// or external user may sample the cached texture after the teardown release helper has retired and
    /// released it.
    #[cfg(all(test, feature = "wayland_frontend"))]
    unsafe fn mark_wayland_dmabuf_texture_cache_release_lifecycle_for_sampled_import(
        &self,
        buffer: &super::utils::Buffer,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        unsafe {
            // SAFETY: Forwarded from this test-only unsafe lifecycle-marking helper's caller.
            self.mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
                buffer.user_data(),
                dmabuf,
            )
        }
    }

    #[cfg(feature = "wayland_frontend")]
    unsafe fn mark_wayland_dmabuf_user_data_texture_cache_release_lifecycle_for_sampled_import(
        &self,
        user_data: &UserDataMap,
        dmabuf: &Dmabuf,
    ) -> Result<(), VulkanError> {
        let slot =
            user_data.get_or_insert_threadsafe(SampledDmabufWaylandTextureCacheReleaseLifecycleSlot::default);
        let mut evidence = slot.evidence.lock().map_err(|_| {
            VulkanError::UnsupportedOperation("sampled dmabuf Wayland texture-cache release lifecycle")
        })?;
        *evidence = Some(unsafe {
            // SAFETY: Forwarded from this unsafe lifecycle-marking helper's caller.
            SampledDmabufWaylandTextureCacheReleaseLifecycleEvidence::new(
                dmabuf.weak(),
                self.context_id.clone(),
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
        self.sampled_dmabuf_layout_history
            .get(&dmabuf.weak())
            .copied()
            .unwrap_or(SampledDmabufWaylandLayoutHistory::NoRendererHistory)
    }

    #[allow(dead_code)]
    fn record_sampled_dmabuf_released_to_foreign_general(&mut self, dmabuf: &Dmabuf) {
        self.prune_sampled_dmabuf_layout_history();
        self.sampled_dmabuf_layout_history.insert(
            dmabuf.weak(),
            SampledDmabufWaylandLayoutHistory::ReleasedByRendererToForeignGeneral,
        );
    }

    #[allow(dead_code)]
    fn record_sampled_dmabuf_locally_acquired(&mut self, dmabuf: &Dmabuf) {
        self.prune_sampled_dmabuf_layout_history();
        self.sampled_dmabuf_layout_history
            .insert(dmabuf.weak(), SampledDmabufWaylandLayoutHistory::LocallyAcquired);
    }

    #[allow(dead_code)]
    fn sampled_dmabuf_wayland_texture_cache_release_hook(
        &self,
        dmabuf: &Dmabuf,
    ) -> SampledDmabufWaylandTextureCacheReleaseHook {
        SampledDmabufWaylandTextureCacheReleaseHook {
            dmabuf: dmabuf.weak(),
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
    /// Raw Vulkan probing may populate format records used by validation tests and development paths,
    /// but the generic public import trait still lacks an external-state and lifecycle contract for
    /// arbitrary dmabufs, and the direct generic import implementation is still pending. Keep those
    /// fields false until real Smithay-facing contracts replace the current fail-closed markers.
    fn sampled_dmabuf_public_import_contracts(&self) -> SampledDmabufPublicImportContracts {
        SampledDmabufPublicImportContracts {
            raw_import_capability: self.capabilities.import.dmabuf,
            advertised_formats: self.capabilities.formats.dmabuf_import.iter().next().is_some(),
            public_external_state_policy: false,
            public_import_lifecycle: false,
            public_import_implementation: false,
        }
    }

    /// Check whether the normal sampled-dmabuf external-state policy is ready for public import.
    ///
    /// Raw probed Vulkan formats are not enough for public [`ImportDma`] advertisement. The normal
    /// path must also have Smithay-owned first-import and reacquire layout/ownership evidence sources
    /// that can prove the external image state before the Vulkan acquire helper runs. Keep this as a
    /// separate guard so enabling raw dmabuf import flags cannot skip the currently missing contracts.
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
            let acquire_sync = buffer.acquire_point().cloned().map(SyncPoint::from);
            self.validate_sampled_dmabuf_wayland_acquire_sync_contract(acquire_sync.as_ref())?;
            acquire_sync
                .map(|sync| SampledDmabufAcquireSyncEvidence::new(dmabuf, sync))
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
            if buffer.release_point().is_none() {
                return Err(VulkanError::MissingCapability(
                    "sampled dmabuf release point contract",
                ));
            }

            Ok(SampledDmabufReleaseEvidence {
                dmabuf: dmabuf.weak(),
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
    /// `linux-dmabuf` alone implies implicit synchronization. Vulkan sampled import remains
    /// not public-advertised for that case until a tested implicit-sync policy exists. The current
    /// validation-stage path accepts only commits that carry explicit acquire synchronization through
    /// Smithay's renderer-managed surface-state buffer.
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
                .map(|evidence| evidence.known_foreign_general(dmabuf))
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
                .map(|evidence| evidence.current_reacquire_layout(dmabuf))
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

    /// Read compositor-provided texture-cache release lifecycle evidence from the renderer buffer wrapper.
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

        evidence.release_lifecycle(dmabuf, &self.context_id).map(Some)
    }

    /// Gather the external-state evidence sources used by the normal Wayland sampled-dmabuf path.
    ///
    /// This preserves the production fail-closed order before sync evidence is considered: first the
    /// first-import layout and known-state sources, then the current-reacquire layout and known-state
    /// sources. The helper is intentionally still validation-stage: caller-provided
    /// `FOREIGN + GENERAL` evidence may now satisfy the external-state source, but public
    /// advertisement and the release lifecycle remain development-gated at their exact missing
    /// contracts.
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
    /// operation happens only after all earlier guards pass.
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
    /// `Buffer::take_release_point_for_renderer()` transfer remains a separate fallible move at the
    /// late texture-construction boundary.
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
            self.validate_sampled_dmabuf_wayland_release_ownership_availability_contract(
                dmabuf,
                buffer.release_point().is_some(),
            )
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
    ) -> Result<SampledDmabufReleaseOwnership, VulkanError> {
        #[cfg(feature = "backend_drm")]
        {
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
    /// texture construction moves it into [`VulkanTexture`].
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
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let Some(sampled_image) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            device.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point(
                dmabuf,
                self.downscale_filter,
                self.upscale_filter,
                acquire_sync,
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

    /// Development-only direct sampled dmabuf import experiment.
    ///
    /// This intentionally does not make sampled [`ImportDma`] public-advertised: `dmabuf_formats()`
    /// still uses the ordered public readiness contract. The purpose of this path is to make direct
    /// calls reach the Vulkan sampled import implementation in this local fork so validation/runtime
    /// probes can expose the next concrete failure instead of stopping forever at abstract public
    /// external-state guards.
    ///
    /// # Safety assumption
    ///
    /// This path assumes, for development discovery only, that the producer has already released the
    /// dmabuf to `VK_QUEUE_FAMILY_FOREIGN_EXT` in `VK_IMAGE_LAYOUT_GENERAL`, and that producer writes
    /// and ownership release are already complete and visible to this renderer. There is no acquire
    /// sync input on generic [`ImportDma`], so this must not be treated as downstream/public support.
    fn experimental_import_dmabuf_assume_foreign_general(
        &mut self,
        dmabuf: &Dmabuf,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
        tracing::warn!(
            "experimentally importing sampled dmabuf by assuming FOREIGN ownership and GENERAL layout"
        );
        let foreign_general = unsafe {
            // SAFETY: This is the explicit local-development assumption documented on this helper.
            SampledDmabufKnownLayoutEvidence::foreign_general(dmabuf.weak())
        };

        unsafe {
            // SAFETY: This is the explicit local-development assumption documented on this helper.
            self.create_imported_dmabuf_texture_with_known_general_layout_and_sync_point(
                dmabuf,
                foreign_general,
                None,
            )
        }
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
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let Some(sampled_image) = (unsafe {
            // SAFETY: Forwarded from this method's caller.
            device.create_acquired_dmabuf_sampled_image_resources_with_known_general_layout_and_sync_point(
                dmabuf,
                self.downscale_filter,
                self.upscale_filter,
                acquire_sync,
            )
        })?
        else {
            return Ok(None);
        };
        let release_ownership = release_ownership()?;
        self.validate_sampled_dmabuf_release_lifecycle_contract(dmabuf, &release_ownership)?;
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
        if !evidence.is_for_dmabuf(dmabuf) {
            return Err(VulkanError::UnsupportedOperation(
                "allocator dmabuf release evidence",
            ));
        }

        // SAFETY: Forwarded from this helper's caller and backed by the consumed allocator evidence.
        unsafe { self.bind_dmabuf_render_target(dmabuf, VulkanDmabufRenderTargetAcquire::preserve(None)) }
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
        let foreign_general = unsafe {
            // SAFETY: This public unsafe method requires its caller to prove the producer released
            // the dmabuf to FOREIGN ownership in GENERAL layout.
            SampledDmabufKnownLayoutEvidence::foreign_general(dmabuf.weak())
        };
        // SAFETY: Forwarded from this public unsafe method's caller.
        unsafe {
            self.create_imported_dmabuf_texture_with_known_general_layout_and_sync_point(
                dmabuf,
                foreign_general,
                acquire_sync,
            )
        }
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
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign dmabuf texture"));
        }
        if texture.image.source != image::VulkanImageSource::DmabufImport {
            return Err(VulkanError::UnsupportedOperation("dmabuf texture"));
        }
        let sampled_image = texture
            .sampled_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf texture sampled image"))?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;

        let (released, release_sync_file) =
            device.release_sampled_dmabuf_to_foreign_general(sampled_image.image(), export_sync_file)?;
        if released {
            texture.signal_sampled_dmabuf_release_point(release_sync_file.as_ref().map(OwnedFd::as_fd))?;
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
        texture
            .signal_sampled_dmabuf_release_point(release_sync_file)
            .map_err(SurfaceCacheTextureReleaseError::ReleaseSideEffectsCommitted)?;
        if let Some(dmabuf) = texture.sampled_dmabuf.as_ref().and_then(WeakDmabuf::upgrade) {
            self.record_sampled_dmabuf_released_to_foreign_general(&dmabuf);
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

    /// Complete the normal Wayland sampled-dmabuf import after evidence collection.
    ///
    /// This is the shared implementation core for [`ImportDmaWl`]. It deliberately still requires a
    /// late-bound release-ownership transfer closure so the renderer-managed Wayland buffer wrapper
    /// remains the production owner of the move-only release point until all policy guards have passed.
    /// Extracting this helper keeps the intended Wayland path testable without turning the direct
    /// `ImportDma` experiment into public support or inventing a side-channel advertisement surface.
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
            // layout evidence for this helper. The release-ownership closure is late-bound so the
            // move-only Wayland release point is not taken until all policy guards have accepted the
            // import and texture construction is ready to attach the release obligation.
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
        if target.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign dmabuf render target"));
        }
        if target.image.source != image::VulkanImageSource::RenderTarget {
            return Err(VulkanError::UnsupportedOperation("dmabuf render target"));
        }
        if !target.image.sync.is_locally_usable() {
            return Err(VulkanError::UnsupportedOperation("dmabuf external ownership"));
        }
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("dmabuf render target image"))?;
        let device = self.device.as_ref().ok_or(VulkanError::VulkanUnavailable)?;
        let release =
            device.release_dmabuf_render_target_to_foreign_general(color_image, export_sync_file)?;
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
        let (released, sync_file) =
            self.release_acquired_dmabuf_render_target_to_foreign_general(target, export_sync_file)?;
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
    pub fn release_dmabuf_render_target_for_sampled_loopback(
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
    pub unsafe fn import_dmabuf_texture_from_loopback(
        &mut self,
        dmabuf: &Dmabuf,
        evidence: VulkanDmabufLoopbackImportEvidence,
    ) -> Result<Option<VulkanTexture>, VulkanError> {
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
            image::VulkanImageSource::Offscreen | image::VulkanImageSource::RenderTarget
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
        Ok(())
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
        if target.image.source != image::VulkanImageSource::Offscreen {
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

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        if !self.capabilities.rendering.dmabuf_target_development {
            return Err(VulkanError::NotPublicAdvertised("dmabuf render target"));
        }

        unsafe {
            // SAFETY: `Bind<Dmabuf>` follows Smithay's renderer target contract. For externally
            // shared targets, that contract requires callers to ensure no concurrent foreign access
            // and to satisfy renderer-specific external ownership/layout requirements before
            // binding. The Vulkan dmabuf target path uses a discard/full-repaint acquire policy here,
            // so previous contents are not preserved and no acquire fence is required by this
            // binding. Successful frames release the image for foreign/KMS use from `Frame::finish`;
            // failed or skipped renders are handled by `RenderTargetLifecycle<Dmabuf>` below.
            self.bind_dmabuf_render_target(target, VulkanDmabufRenderTargetAcquire::discard())
        }?
        .ok_or(VulkanError::MissingCapability(
            "dmabuf render target format/modifier",
        ))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(self.development_gated_dmabuf_render_target_formats())
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
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        let texture = self.experimental_import_dmabuf_assume_foreign_general(dmabuf)?;

        if let Some(texture) = texture {
            self.record_sampled_dmabuf_locally_acquired(dmabuf);
            Ok(texture)
        } else {
            Err(VulkanError::MissingCapability("sampled dmabuf texture import"))
        }
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
        let dmabuf = crate::wayland::dmabuf::get_dmabuf(buffer)
            .expect("import_dma_buffer_from_surface_state without checking buffer type?");

        let import = self.validate_sampled_dmabuf_import_metadata(dmabuf)?;

        let layout_history = self.sampled_dmabuf_layout_history(dmabuf);
        let wayland_external_state =
            self.sampled_dmabuf_wayland_buffer_foreign_general_evidence(buffer, dmabuf)?;
        let external_state_sources = self.sampled_dmabuf_wayland_external_state_evidence_sources(
            dmabuf,
            layout_history,
            wayland_external_state.as_ref(),
        )?;
        let acquire_sync = self.sampled_dmabuf_wayland_acquire_sync_evidence(dmabuf, buffer)?;
        let release_evidence = self.sampled_dmabuf_wayland_release_evidence(dmabuf, buffer)?;
        let release_ownership = self.sampled_dmabuf_wayland_release_ownership_evidence(dmabuf, buffer)?;
        let replacement_release_reachability = self
            .validate_sampled_dmabuf_wayland_texture_cache_replacement_reachability_contract(
                dmabuf,
                surface
                    .map(super::utils::surface_import_after_retired_release)
                    .unwrap_or(false),
            )?;
        let texture_cache_release_hook = self.sampled_dmabuf_wayland_texture_cache_release_hook(dmabuf);
        let texture_cache_release_lifecycle =
            self.sampled_dmabuf_wayland_buffer_texture_cache_release_lifecycle(buffer, dmabuf)?;
        let policy_context = SampledDmabufWaylandVulkanInteropPolicyContext::new(
            dmabuf,
            &import,
            &acquire_sync,
            &release_evidence,
            true,
            layout_history,
        )
        .with_external_state_sources(external_state_sources)
        .with_texture_cache_replacement_release_reachability(replacement_release_reachability)
        .with_texture_cache_release_hook(texture_cache_release_hook)
        .with_texture_cache_release_lifecycle(texture_cache_release_lifecycle)
        .with_release_ownership(release_ownership);
        self.import_wayland_dmabuf_with_policy_context(policy_context, || {
            Self::sampled_dmabuf_take_wayland_release_ownership(dmabuf, buffer)
        })
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
        _region: Rectangle<i32, BufferCoord>,
        _format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign memory texture"));
        }

        Err(VulkanError::UnsupportedOperation("texture memory export"))
    }

    fn can_read_texture(&mut self, texture: &Self::TextureId) -> Result<bool, Self::Error> {
        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign memory texture"));
        }

        Ok(false)
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
