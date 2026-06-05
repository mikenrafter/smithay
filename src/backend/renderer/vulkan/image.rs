use std::{marker::PhantomData, sync::Arc};

use crate::{
    backend::{
        allocator::{Fourcc, Modifier},
        renderer::{Color32F, ContextId, Frame, Texture, sync::SyncPoint},
    },
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
};

use super::{VulkanError, VulkanRenderer, device::VulkanSampledImage};

/// Vulkan frame scaffold.
#[derive(Debug)]
pub struct VulkanFrame<'renderer, 'buffer> {
    pub(super) context_id: ContextId<VulkanTexture>,
    pub(super) output_size: Size<i32, Physical>,
    pub(super) transform: Transform,
    pub(super) _renderer: PhantomData<&'renderer mut VulkanRenderer>,
    pub(super) _target: PhantomData<&'renderer mut VulkanRenderTarget<'buffer>>,
}

/// Vulkan texture scaffold.
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    pub(super) context_id: ContextId<VulkanTexture>,
    pub(super) image: VulkanImageState,
    pub(super) sampled_image: Option<Arc<VulkanSampledImage>>,
    pub(super) y_inverted: bool,
}

/// Vulkan render target scaffold.
#[derive(Debug)]
pub struct VulkanRenderTarget<'buffer> {
    pub(super) image: VulkanImageState,
    pub(super) _target: PhantomData<&'buffer mut ()>,
}

/// Vulkan image state placeholder shared by textures and render targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VulkanImageState {
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) format: Option<Fourcc>,
    pub(super) source: VulkanImageSource,
    pub(super) usage: VulkanImageUsage,
    pub(super) layout: VulkanImageLayoutState,
    pub(super) sync: VulkanImageSyncState,
}

impl VulkanImageState {
    fn width(&self) -> u32 {
        self.size.w.try_into().unwrap_or_default()
    }

    fn height(&self) -> u32 {
        self.size.h.try_into().unwrap_or_default()
    }

    fn size(&self) -> Size<i32, BufferCoord> {
        self.size
    }

    fn format(&self) -> Option<Fourcc> {
        self.format
    }

    #[cfg(test)]
    pub(super) fn new_for_tests(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> Self {
        Self {
            size,
            format,
            source: VulkanImageSource::Uninitialized,
            usage: VulkanImageUsage::default(),
            layout: VulkanImageLayoutState::Undefined,
            sync: VulkanImageSyncState::default(),
        }
    }
}

impl VulkanTexture {
    pub(crate) fn from_sampled_image(
        context_id: ContextId<VulkanTexture>,
        size: Size<i32, BufferCoord>,
        format: Fourcc,
        sampled_image: VulkanSampledImage,
        flipped: bool,
    ) -> Self {
        Self {
            context_id,
            image: VulkanImageState {
                size,
                format: Some(format),
                source: VulkanImageSource::MemoryUpload,
                usage: VulkanImageUsage {
                    sampled: true,
                    transfer_dst: true,
                    ..VulkanImageUsage::default()
                },
                layout: VulkanImageLayoutState::ShaderReadOnly,
                sync: VulkanImageSyncState::default(),
            },
            sampled_image: Some(Arc::new(sampled_image)),
            y_inverted: flipped,
        }
    }

    #[cfg(test)]
    pub(super) fn has_sampled_image_for_tests(&self) -> bool {
        self.sampled_image.is_some()
    }

    #[cfg(test)]
    pub(super) fn is_y_inverted_for_tests(&self) -> bool {
        self.y_inverted
    }
}

/// Provenance of a Vulkan image managed by the renderer.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VulkanImageSource {
    Uninitialized,
    MemoryUpload,
    DmabufImport,
    Offscreen,
    RenderTarget,
    Swapchain,
}

/// Image usage bits tracked by the renderer scaffold.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VulkanImageUsage {
    pub(super) sampled: bool,
    pub(super) color_attachment: bool,
    pub(super) transfer_src: bool,
    pub(super) transfer_dst: bool,
    pub(super) exportable: bool,
    pub(super) host_visible: bool,
}

/// Coarse image layout state tracked by the renderer scaffold.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VulkanImageLayoutState {
    Undefined,
    ShaderReadOnly,
    ColorAttachment,
    TransferSrc,
    TransferDst,
    Present,
}

/// Synchronization state tracked by the renderer scaffold.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VulkanImageSyncState {
    pub(super) pending_write: bool,
    pub(super) exportable_sync: bool,
}

/// External-memory metadata placeholder for future dmabuf import/export support.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VulkanExternalMemoryState {
    pub(super) handle_type: VulkanExternalMemoryHandleType,
    pub(super) format: Fourcc,
    pub(super) modifier: Modifier,
    pub(super) planes: Vec<VulkanDmabufPlane>,
    pub(super) disjoint: bool,
}

/// External memory handle families the renderer is expected to support incrementally.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VulkanExternalMemoryHandleType {
    Dmabuf,
}

/// Dmabuf plane metadata needed before importing/exporting a Vulkan image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VulkanDmabufPlane {
    pub(super) plane_idx: u32,
    pub(super) offset: u32,
    pub(super) stride: u32,
}

/// Import metadata placeholder for a future `ImportDma` implementation.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VulkanDmabufImportState {
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) memory: VulkanExternalMemoryState,
}

impl VulkanRenderTarget<'_> {
    #[cfg(test)]
    pub(super) fn new_for_tests(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> Self {
        Self {
            image: VulkanImageState::new_for_tests(size, format),
            _target: PhantomData,
        }
    }
}

impl Texture for VulkanTexture {
    fn width(&self) -> u32 {
        self.image.width()
    }

    fn height(&self) -> u32 {
        self.image.height()
    }

    fn size(&self) -> Size<i32, BufferCoord> {
        self.image.size()
    }

    fn format(&self) -> Option<Fourcc> {
        self.image.format()
    }
}

impl Texture for VulkanRenderTarget<'_> {
    fn width(&self) -> u32 {
        self.image.width()
    }

    fn height(&self) -> u32 {
        self.image.height()
    }

    fn size(&self) -> Size<i32, BufferCoord> {
        self.image.size()
    }

    fn format(&self) -> Option<Fourcc> {
        self.image.format()
    }
}

impl Frame for VulkanFrame<'_, '_> {
    type Error = VulkanError;
    type TextureId = VulkanTexture;

    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.context_id.clone()
    }

    fn clear(&mut self, _color: Color32F, _at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        if _at.is_empty() {
            return Ok(());
        }

        Err(VulkanError::UnsupportedOperation("clear"))
    }

    fn draw_solid(
        &mut self,
        _dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _color: Color32F,
    ) -> Result<(), Self::Error> {
        if damage.is_empty() {
            return Ok(());
        }

        Err(VulkanError::UnsupportedOperation("draw solid"))
    }

    fn render_texture_from_to(
        &mut self,
        _texture: &Self::TextureId,
        _src: Rectangle<f64, BufferCoord>,
        _dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        _src_transform: Transform,
        _alpha: f32,
    ) -> Result<(), Self::Error> {
        if damage.is_empty() {
            return Ok(());
        }

        Err(VulkanError::UnsupportedOperation("render texture"))
    }

    fn transformation(&self) -> Transform {
        self.transform
    }

    fn output_size(&self) -> Size<i32, Physical> {
        self.output_size
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        sync.wait().map_err(|_| VulkanError::SyncInterrupted)
    }

    fn finish(self) -> Result<SyncPoint, Self::Error> {
        Ok(SyncPoint::signaled())
    }
}
