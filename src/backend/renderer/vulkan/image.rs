use std::{marker::PhantomData, sync::Arc};

use ash::vk;

use crate::{
    backend::{
        allocator::{Fourcc, Modifier},
        renderer::{Color32F, ContextId, Frame, Texture, sync::SyncPoint},
    },
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
};

use super::{
    VulkanError, VulkanRenderer, clear_color_value_for_format,
    device::{VulkanDeviceState, VulkanOwnedImage, VulkanSampledImage, VulkanSampledTextureDrawConstants},
    format::{get_format_info, get_render_vk_format},
};

/// Vulkan frame scaffold.
#[derive(Debug)]
pub struct VulkanFrame<'renderer, 'buffer> {
    pub(super) context_id: ContextId<VulkanTexture>,
    pub(super) output_size: Size<i32, Physical>,
    pub(super) transform: Transform,
    pub(super) device: Option<&'renderer VulkanDeviceState>,
    pub(super) target: Option<&'renderer mut VulkanRenderTarget<'buffer>>,
    pub(super) _renderer: PhantomData<&'renderer mut VulkanRenderer>,
}

/// Vulkan texture scaffold.
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    pub(super) context_id: ContextId<VulkanTexture>,
    pub(super) image: VulkanImageState,
    pub(super) sampled_image: Option<Arc<VulkanSampledImage>>,
    #[allow(dead_code)]
    pub(super) y_inverted: bool,
}

/// Vulkan render target scaffold.
#[allow(dead_code)]
#[derive(Debug)]
pub struct VulkanRenderTarget<'buffer> {
    pub(super) context_id: ContextId<VulkanTexture>,
    pub(super) image: VulkanImageState,
    pub(super) color_image: Option<VulkanOwnedImage>,
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
    #[allow(dead_code)]
    pub(crate) fn from_offscreen_image(
        context_id: ContextId<VulkanTexture>,
        size: Size<i32, BufferCoord>,
        format: Fourcc,
        color_image: VulkanOwnedImage,
    ) -> Self {
        Self {
            context_id,
            image: VulkanImageState {
                size,
                format: Some(format),
                source: VulkanImageSource::Offscreen,
                usage: VulkanImageUsage {
                    color_attachment: true,
                    transfer_src: true,
                    transfer_dst: true,
                    ..VulkanImageUsage::default()
                },
                layout: VulkanImageLayoutState::Undefined,
                sync: VulkanImageSyncState::default(),
            },
            color_image: Some(color_image),
            _target: PhantomData,
        }
    }

    #[cfg(test)]
    pub(super) fn new_for_tests(size: Size<i32, BufferCoord>, format: Option<Fourcc>) -> Self {
        Self {
            context_id: ContextId::new(),
            image: VulkanImageState::new_for_tests(size, format),
            color_image: None,
            _target: PhantomData,
        }
    }

    #[cfg(test)]
    pub(super) fn has_color_image_for_tests(&self) -> bool {
        self.color_image.is_some()
    }

    #[cfg(test)]
    pub(super) fn context_id_for_tests(&self) -> ContextId<VulkanTexture> {
        self.context_id.clone()
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

    fn clear(&mut self, color: Color32F, at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        if at.is_empty() {
            return Ok(());
        }

        if !is_full_target_damage(self.output_size, at) {
            return Err(VulkanError::UnsupportedOperation("partial clear"));
        }

        let device = self
            .device
            .ok_or(VulkanError::UnsupportedOperation("clear device"))?;
        let target = self
            .target
            .as_deref_mut()
            .ok_or(VulkanError::UnsupportedOperation("clear target"))?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("offscreen target image"))?;
        let format = target
            .image
            .format
            .ok_or(VulkanError::UnsupportedOperation("offscreen target format"))?;

        device.clear_color_attachment_image(color_image, clear_color_value_for_format(format, color)?)?;
        target.image.layout = VulkanImageLayoutState::ColorAttachment;
        Ok(())
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
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        if damage.is_empty() {
            return Ok(());
        }

        if texture.context_id != self.context_id {
            return Err(VulkanError::UnsupportedOperation("foreign render texture"));
        }
        if !is_full_target_damage(self.output_size, damage) {
            return Err(VulkanError::UnsupportedOperation("render texture damage"));
        }
        if self.transform != Transform::Normal || !is_axis_aligned_sampled_texture_transform(src_transform) {
            return Err(VulkanError::UnsupportedOperation("render texture transform"));
        }
        let uv_rect = source_to_uv_rect(texture.image.size, src, texture.y_inverted, src_transform)
            .ok_or(VulkanError::UnsupportedOperation("render texture source"))?;
        let draw_area = output_destination_to_vk_rect(self.output_size, dst)
            .ok_or(VulkanError::UnsupportedOperation("render texture destination"))?;
        if draw_area.extent.width == 0 || draw_area.extent.height == 0 {
            return Err(VulkanError::UnsupportedOperation("render texture destination"));
        }
        if !opaque_regions.is_empty() {
            return Err(VulkanError::UnsupportedOperation("render texture opaque regions"));
        }
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(VulkanError::UnsupportedOperation("render texture alpha"));
        }
        let device = self
            .device
            .ok_or(VulkanError::UnsupportedOperation("render texture device"))?;
        let target = self
            .target
            .as_deref_mut()
            .ok_or(VulkanError::UnsupportedOperation("render texture target"))?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("offscreen target image"))?;
        let target_format = target
            .image
            .format
            .ok_or(VulkanError::UnsupportedOperation("offscreen target format"))?;
        let sampled_image = texture
            .sampled_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("render texture image"))?;
        let texture_format = texture
            .image
            .format
            .ok_or(VulkanError::UnsupportedOperation("render texture format"))?;
        let force_opaque_alpha = get_format_info(texture_format)?.opaque_alpha;

        let pipeline =
            device.create_builtin_sampled_texture_graphics_pipeline(get_render_vk_format(target_format)?)?;
        let descriptor_pool = device.create_sampled_texture_descriptor_pool(1)?;
        let descriptor_set = device.create_sampled_texture_descriptor_set(
            &descriptor_pool,
            pipeline.layout().descriptor_set_layout(),
            Arc::clone(sampled_image),
        )?;

        device.render_sampled_texture_to_color_image_in(
            color_image,
            &descriptor_set,
            &pipeline,
            VulkanSampledTextureDrawConstants {
                draw_area,
                uv_rect,
                alpha,
                force_opaque_alpha,
            },
        )?;
        target.image.layout = VulkanImageLayoutState::ColorAttachment;
        Ok(())
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

fn is_full_target_damage(output_size: Size<i32, Physical>, damage: &[Rectangle<i32, Physical>]) -> bool {
    damage.len() == 1 && damage[0] == Rectangle::from_size(output_size)
}

fn is_axis_aligned_sampled_texture_transform(transform: Transform) -> bool {
    matches!(
        transform,
        Transform::Normal | Transform::_180 | Transform::Flipped | Transform::Flipped180
    )
}

pub(super) fn source_to_uv_rect(
    texture_size: Size<i32, BufferCoord>,
    src: Rectangle<f64, BufferCoord>,
    y_inverted: bool,
    src_transform: Transform,
) -> Option<[f32; 4]> {
    if texture_size.w <= 0
        || texture_size.h <= 0
        || !src.loc.x.is_finite()
        || !src.loc.y.is_finite()
        || !src.size.w.is_finite()
        || !src.size.h.is_finite()
        || src.loc.x < 0.0
        || src.loc.y < 0.0
        || src.size.w <= 0.0
        || src.size.h <= 0.0
    {
        return None;
    }

    let x_end = src.loc.x + src.size.w;
    let y_end = src.loc.y + src.size.h;
    if x_end > f64::from(texture_size.w) || y_end > f64::from(texture_size.h) {
        return None;
    }

    let (origin, x_axis, y_axis) = match src_transform {
        Transform::Normal => ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        Transform::_180 => ((1.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
        Transform::Flipped => ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)),
        Transform::Flipped180 => ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0)),
        Transform::_90 | Transform::_270 | Transform::Flipped90 | Transform::Flipped270 => return None,
    };
    let uv_at = |x: f64, y: f64| {
        let u = (src.loc.x + x * src.size.w) / f64::from(texture_size.w);
        let v = (src.loc.y + y * src.size.h) / f64::from(texture_size.h);

        (u as f32, if y_inverted { (1.0 - v) as f32 } else { v as f32 })
    };
    let offset = uv_at(origin.0, origin.1);
    let x_end = uv_at(origin.0 + x_axis.0, origin.1 + x_axis.1);
    let y_end = uv_at(origin.0 + y_axis.0, origin.1 + y_axis.1);

    Some([offset.0, offset.1, x_end.0 - offset.0, y_end.1 - offset.1])
}

fn output_destination_to_vk_rect(
    output_size: Size<i32, Physical>,
    dst: Rectangle<i32, Physical>,
) -> Option<vk::Rect2D> {
    if dst.loc.x < 0 || dst.loc.y < 0 || dst.size.w <= 0 || dst.size.h <= 0 {
        return None;
    }
    let x_end = dst.loc.x.checked_add(dst.size.w)?;
    let y_end = dst.loc.y.checked_add(dst.size.h)?;
    if x_end > output_size.w || y_end > output_size.h {
        return None;
    }

    Some(vk::Rect2D {
        offset: vk::Offset2D {
            x: dst.loc.x,
            y: dst.loc.y,
        },
        extent: vk::Extent2D {
            width: u32::try_from(dst.size.w).ok()?,
            height: u32::try_from(dst.size.h).ok()?,
        },
    })
}
