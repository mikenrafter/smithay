use std::{marker::PhantomData, sync::Arc};

use ash::vk;

use crate::{
    backend::{
        allocator::{Buffer, Fourcc, Modifier, dmabuf::Dmabuf},
        renderer::{Color32F, ContextId, Frame, Texture, TextureMapping, sync::SyncPoint},
    },
    utils::{Buffer as BufferCoord, Physical, Rectangle, Size, Transform},
};

use super::{
    VulkanError, VulkanRenderer, clear_color_value_for_format,
    device::{
        VulkanDeviceState, VulkanOwnedImage, VulkanSampledImage, VulkanSampledTextureDrawConstants,
        VulkanSolidColorDrawConstants,
    },
    format::{get_format_info, get_render_vk_format},
};

/// Vulkan frame for the provisional in-memory/offscreen renderer.
#[derive(Debug)]
pub struct VulkanFrame<'renderer, 'buffer> {
    pub(super) context_id: ContextId<VulkanTexture>,
    pub(super) output_size: Size<i32, Physical>,
    pub(super) transform: Transform,
    pub(super) device: Option<&'renderer VulkanDeviceState>,
    pub(super) target: Option<&'renderer mut VulkanRenderTarget<'buffer>>,
    pub(super) _renderer: PhantomData<&'renderer mut VulkanRenderer>,
}

/// Vulkan texture tracked by the renderer.
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    pub(super) context_id: ContextId<VulkanTexture>,
    pub(super) image: VulkanImageState,
    pub(super) sampled_image: Option<Arc<VulkanSampledImage>>,
    #[allow(dead_code)]
    pub(super) y_inverted: bool,
}

/// Vulkan render target tracked by the renderer.
#[allow(dead_code)]
#[derive(Debug)]
pub struct VulkanRenderTarget<'buffer> {
    pub(super) context_id: ContextId<VulkanTexture>,
    pub(super) image: VulkanImageState,
    pub(super) color_image: Option<VulkanOwnedImage>,
    pub(super) _target: PhantomData<&'buffer mut ()>,
}

/// CPU-memory readback mapping produced by the Vulkan renderer.
#[derive(Debug, Clone)]
pub struct VulkanMemoryMapping {
    pub(super) data: Vec<u8>,
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) format: Fourcc,
    pub(super) flipped: bool,
}

/// Vulkan image state shared by textures and render targets.
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

    #[allow(dead_code)]
    pub(crate) fn from_dmabuf_sampled_image(
        context_id: ContextId<VulkanTexture>,
        import: &VulkanDmabufImportState,
        sampled_image: VulkanSampledImage,
    ) -> Self {
        Self {
            context_id,
            image: dmabuf_import_image_state(import),
            sampled_image: Some(Arc::new(sampled_image)),
            y_inverted: import.y_inverted,
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

#[allow(dead_code)]
pub(super) fn dmabuf_import_image_state(import: &VulkanDmabufImportState) -> VulkanImageState {
    VulkanImageState {
        size: import.size,
        format: Some(import.format()),
        source: VulkanImageSource::DmabufImport,
        usage: VulkanImageUsage {
            sampled: true,
            ..VulkanImageUsage::default()
        },
        layout: VulkanImageLayoutState::Undefined,
        sync: VulkanImageSyncState {
            external_acquire_pending: true,
            external_ownership: VulkanExternalImageOwnership::ForeignUnknown,
            ..VulkanImageSyncState::default()
        },
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

/// Image usage bits tracked by the renderer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VulkanImageUsage {
    pub(super) sampled: bool,
    pub(super) color_attachment: bool,
    pub(super) transfer_src: bool,
    pub(super) transfer_dst: bool,
    pub(super) exportable: bool,
    pub(super) host_visible: bool,
}

/// Coarse image layout state tracked by the renderer.
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

/// Synchronization state tracked by the renderer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VulkanImageSyncState {
    pub(super) pending_write: bool,
    pub(super) exportable_sync: bool,
    pub(super) external_acquire_pending: bool,
    pub(super) external_ownership: VulkanExternalImageOwnership,
}

impl VulkanImageSyncState {
    #[allow(dead_code)]
    pub(super) fn foreign_known_general_for_dmabuf_import() -> Self {
        Self {
            external_acquire_pending: true,
            external_ownership: VulkanExternalImageOwnership::ForeignKnownGeneral,
            ..Self::default()
        }
    }

    #[allow(dead_code)]
    pub(crate) fn external_acquire_pending(&self) -> bool {
        self.external_acquire_pending
    }

    #[allow(dead_code)]
    pub(crate) fn external_ownership(&self) -> VulkanExternalImageOwnership {
        self.external_ownership
    }

    #[allow(dead_code)]
    pub(crate) fn known_foreign_layout(&self) -> Option<vk::ImageLayout> {
        match self.external_ownership {
            VulkanExternalImageOwnership::ForeignKnownGeneral => Some(vk::ImageLayout::GENERAL),
            VulkanExternalImageOwnership::None
            | VulkanExternalImageOwnership::ForeignUnknown
            | VulkanExternalImageOwnership::Local => None,
        }
    }

    #[allow(dead_code)]
    pub(super) fn complete_sampled_dmabuf_foreign_acquire(&mut self) -> Result<(), VulkanError> {
        match (self.external_ownership, self.external_acquire_pending) {
            (VulkanExternalImageOwnership::ForeignKnownGeneral, true) => {
                self.external_ownership = VulkanExternalImageOwnership::Local;
                self.external_acquire_pending = false;
                Ok(())
            }
            (VulkanExternalImageOwnership::ForeignUnknown, _)
            | (VulkanExternalImageOwnership::ForeignKnownGeneral, false)
            | (VulkanExternalImageOwnership::None, _)
            | (VulkanExternalImageOwnership::Local, _) => {
                Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
            }
        }
    }

    #[allow(dead_code)]
    pub(super) fn complete_sampled_dmabuf_foreign_release(&mut self) -> Result<(), VulkanError> {
        match (self.external_ownership, self.external_acquire_pending) {
            (VulkanExternalImageOwnership::Local, false) => {
                self.external_ownership = VulkanExternalImageOwnership::ForeignKnownGeneral;
                self.external_acquire_pending = true;
                Ok(())
            }
            (VulkanExternalImageOwnership::None, false)
            | (VulkanExternalImageOwnership::None, true)
            | (VulkanExternalImageOwnership::ForeignUnknown, _)
            | (VulkanExternalImageOwnership::ForeignKnownGeneral, _)
            | (VulkanExternalImageOwnership::Local, true) => {
                Err(VulkanError::UnsupportedOperation("dmabuf external ownership"))
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VulkanExternalImageOwnership {
    #[default]
    None,
    ForeignUnknown,
    ForeignKnownGeneral,
    Local,
}

/// External-memory metadata reserved for future dmabuf import/export support.
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

/// Import metadata reserved for a future `ImportDma` implementation.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VulkanDmabufImportState {
    pub(super) size: Size<i32, BufferCoord>,
    pub(super) memory: VulkanExternalMemoryState,
    pub(super) y_inverted: bool,
}

impl VulkanDmabufImportState {
    #[allow(dead_code)]
    pub(crate) fn from_dmabuf(dmabuf: &Dmabuf) -> Result<Self, VulkanError> {
        let size = dmabuf.size();
        if size.w <= 0 || size.h <= 0 {
            return Err(VulkanError::UnsupportedOperation("dmabuf size"));
        }

        if dmabuf.num_planes() == 0 {
            return Err(VulkanError::UnsupportedOperation("dmabuf planes"));
        }

        let mut planes = Vec::with_capacity(dmabuf.num_planes());
        for (expected_idx, plane) in dmabuf.0.planes.iter().enumerate() {
            if plane.plane_idx != expected_idx as u32 {
                return Err(VulkanError::UnsupportedOperation("dmabuf plane index"));
            }
            if plane.stride == 0 {
                return Err(VulkanError::UnsupportedOperation("dmabuf stride"));
            }

            planes.push(VulkanDmabufPlane {
                plane_idx: plane.plane_idx,
                offset: plane.offset,
                stride: plane.stride,
            });
        }

        let format = dmabuf.format();

        Ok(Self {
            size,
            memory: VulkanExternalMemoryState {
                handle_type: VulkanExternalMemoryHandleType::Dmabuf,
                format: format.code,
                modifier: format.modifier,
                disjoint: planes.len() > 1,
                planes,
            },
            y_inverted: dmabuf.y_inverted(),
        })
    }

    pub(crate) fn format(&self) -> Fourcc {
        self.memory.format
    }

    pub(crate) fn modifier(&self) -> Modifier {
        self.memory.modifier
    }

    pub(crate) fn plane_count(&self) -> usize {
        self.memory.planes.len()
    }
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

impl Texture for VulkanMemoryMapping {
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
        Some(self.format)
    }
}

impl TextureMapping for VulkanMemoryMapping {
    fn flipped(&self) -> bool {
        self.flipped
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

        let full_target_clear = is_full_target_damage(self.output_size, at);
        let clear_areas = if full_target_clear {
            None
        } else {
            if self.transform != Transform::Normal {
                return Err(VulkanError::UnsupportedOperation("clear transform"));
            }
            let clear_areas = clear_damage_to_clear_areas(self.output_size, at)
                .ok_or(VulkanError::UnsupportedOperation("clear damage"))?;
            if clear_areas.is_empty() {
                return Ok(());
            }
            Some(clear_areas)
        };

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

        let color = clear_color_value_for_format(format, color)?;
        if full_target_clear {
            device.clear_color_attachment_image(color_image, color)?;
        } else if let Some(clear_areas) = clear_areas {
            device.clear_color_attachment_image_in(color_image, color, clear_areas.as_slice())?;
        }
        target.image.layout = VulkanImageLayoutState::ColorAttachment;
        Ok(())
    }

    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        if damage.is_empty() {
            return Ok(());
        }

        if self.transform != Transform::Normal {
            return Err(VulkanError::UnsupportedOperation("draw solid transform"));
        }
        let clear_areas = draw_solid_damage_to_clear_areas(self.output_size, dst, damage)
            .ok_or(VulkanError::UnsupportedOperation("draw solid damage"))?;
        if clear_areas.is_empty() {
            return Ok(());
        }
        if !color.is_opaque() && rects_overlap(&clear_areas) {
            return Err(VulkanError::UnsupportedOperation("draw solid damage"));
        }
        let draw_region = Rectangle::from_size(self.output_size)
            .intersection(dst)
            .ok_or(VulkanError::UnsupportedOperation("draw solid destination"))?;
        let draw_area = output_destination_to_vk_rect(self.output_size, draw_region)
            .ok_or(VulkanError::UnsupportedOperation("draw solid destination"))?;

        let device = self
            .device
            .ok_or(VulkanError::UnsupportedOperation("draw solid device"))?;
        let target = self
            .target
            .as_deref_mut()
            .ok_or(VulkanError::UnsupportedOperation("draw solid target"))?;
        let color_image = target
            .color_image
            .as_ref()
            .ok_or(VulkanError::UnsupportedOperation("offscreen target image"))?;
        let format = target
            .image
            .format
            .ok_or(VulkanError::UnsupportedOperation("offscreen target format"))?;

        if color.is_opaque() {
            device.clear_color_attachment_image_in(
                color_image,
                clear_color_value_for_format(format, color)?,
                &clear_areas,
            )?;
        } else {
            let pipeline =
                device.builtin_solid_color_graphics_pipeline(get_render_vk_format(format)?, true)?;
            for scissor_area in clear_areas {
                device.render_solid_color_to_color_image_in(
                    color_image,
                    &pipeline,
                    VulkanSolidColorDrawConstants {
                        draw_area,
                        scissor_area,
                        color: color.components(),
                    },
                )?;
            }
        }
        target.image.layout = VulkanImageLayoutState::ColorAttachment;
        Ok(())
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
        if texture.image.sync.external_acquire_pending {
            return Err(VulkanError::UnsupportedOperation("dmabuf import synchronization"));
        }
        if self.transform != Transform::Normal {
            return Err(VulkanError::UnsupportedOperation("render texture transform"));
        }
        let (uv_origin, uv_x_axis, uv_y_axis) =
            source_to_uv_rect(texture.image.size, src, texture.y_inverted, src_transform)
                .ok_or(VulkanError::UnsupportedOperation("render texture source"))?;
        let draw_area = output_destination_to_vk_rect(self.output_size, dst)
            .ok_or(VulkanError::UnsupportedOperation("render texture destination"))?;
        if draw_area.extent.width == 0 || draw_area.extent.height == 0 {
            return Err(VulkanError::UnsupportedOperation("render texture destination"));
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
        let (non_opaque_scissor_areas, opaque_scissor_areas) = render_texture_damage_to_scissor_areas(
            self.output_size,
            dst,
            damage,
            opaque_regions,
            force_opaque_alpha && alpha == 1.0,
            alpha,
        )
        .ok_or(VulkanError::UnsupportedOperation("render texture damage"))?;
        if non_opaque_scissor_areas.is_empty() && opaque_scissor_areas.is_empty() {
            return Ok(());
        }

        let target_vk_format = get_render_vk_format(target_format)?;
        let descriptor_set_count = if non_opaque_scissor_areas.is_empty() { 0 } else { 1 }
            + if opaque_scissor_areas.is_empty() { 0 } else { 1 };
        let descriptor_pool = device.create_sampled_texture_descriptor_pool(descriptor_set_count)?;

        if !non_opaque_scissor_areas.is_empty() {
            let pipeline = device.builtin_sampled_texture_graphics_pipeline(target_vk_format, true)?;
            let descriptor_set = device.create_sampled_texture_descriptor_set(
                &descriptor_pool,
                pipeline.layout().descriptor_set_layout(),
                Arc::clone(sampled_image),
            )?;

            for scissor_area in non_opaque_scissor_areas {
                device.render_sampled_texture_to_color_image_in(
                    color_image,
                    &descriptor_set,
                    &pipeline,
                    VulkanSampledTextureDrawConstants {
                        draw_area,
                        scissor_area,
                        uv_origin,
                        uv_x_axis,
                        uv_y_axis,
                        alpha,
                        force_opaque_alpha,
                    },
                )?;
            }
        }

        if !opaque_scissor_areas.is_empty() {
            let pipeline = device.builtin_sampled_texture_graphics_pipeline(target_vk_format, false)?;
            let descriptor_set = device.create_sampled_texture_descriptor_set(
                &descriptor_pool,
                pipeline.layout().descriptor_set_layout(),
                Arc::clone(sampled_image),
            )?;

            for scissor_area in opaque_scissor_areas {
                device.render_sampled_texture_to_color_image_in(
                    color_image,
                    &descriptor_set,
                    &pipeline,
                    VulkanSampledTextureDrawConstants {
                        draw_area,
                        scissor_area,
                        uv_origin,
                        uv_x_axis,
                        uv_y_axis,
                        alpha,
                        force_opaque_alpha,
                    },
                )?;
            }
        }
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

pub(super) fn clear_damage_to_clear_areas(
    output_size: Size<i32, Physical>,
    damage: &[Rectangle<i32, Physical>],
) -> Option<Vec<vk::Rect2D>> {
    draw_solid_damage_to_clear_areas(output_size, Rectangle::from_size(output_size), damage)
}

pub(super) fn damage_to_scissor_areas(
    output_size: Size<i32, Physical>,
    dst: Rectangle<i32, Physical>,
    damage: &[Rectangle<i32, Physical>],
) -> Option<Vec<vk::Rect2D>> {
    if output_size.w <= 0 || output_size.h <= 0 || dst.size.w <= 0 || dst.size.h <= 0 {
        return None;
    }

    let output = Rectangle::from_size(output_size);
    let draw_region = output.intersection(dst)?;
    let mut scissor_rects = Vec::new();

    for damage in damage {
        if damage.size.w <= 0 || damage.size.h <= 0 {
            continue;
        }
        let translated_damage = Rectangle::new(
            (
                dst.loc.x.checked_add(damage.loc.x)?,
                dst.loc.y.checked_add(damage.loc.y)?,
            )
                .into(),
            damage.size,
        );
        let Some(scissor) = output
            .intersection(translated_damage)
            .and_then(|damage| draw_region.intersection(damage))
        else {
            continue;
        };

        if scissor.size.w <= 0 || scissor.size.h <= 0 {
            continue;
        }
        if scissor_rects.iter().any(|existing| scissor.overlaps(*existing)) {
            return None;
        }
        scissor_rects.push(scissor);
    }

    scissor_rects
        .into_iter()
        .map(|scissor| output_destination_to_vk_rect(output_size, scissor))
        .collect()
}

pub(super) fn draw_solid_damage_to_clear_areas(
    output_size: Size<i32, Physical>,
    dst: Rectangle<i32, Physical>,
    damage: &[Rectangle<i32, Physical>],
) -> Option<Vec<vk::Rect2D>> {
    if output_size.w <= 0 || output_size.h <= 0 || dst.size.w <= 0 || dst.size.h <= 0 {
        return None;
    }

    let output = Rectangle::from_size(output_size);
    let Some(draw_region) = output.intersection(dst) else {
        return Some(Vec::new());
    };
    let mut clear_rects = Vec::new();

    for damage in damage {
        if damage.size.w <= 0 || damage.size.h <= 0 {
            continue;
        }
        let translated_damage = Rectangle::new(
            (
                dst.loc.x.checked_add(damage.loc.x)?,
                dst.loc.y.checked_add(damage.loc.y)?,
            )
                .into(),
            damage.size,
        );
        let Some(clear) = output
            .intersection(translated_damage)
            .and_then(|damage| draw_region.intersection(damage))
        else {
            continue;
        };

        if clear.size.w <= 0 || clear.size.h <= 0 {
            continue;
        }
        clear_rects.push(clear);
    }

    clear_rects
        .into_iter()
        .map(|clear| output_destination_to_vk_rect(output_size, clear))
        .collect()
}

fn rects_overlap(rects: &[vk::Rect2D]) -> bool {
    rects.iter().enumerate().any(|(index, rect)| {
        rects[index + 1..]
            .iter()
            .any(|other| vk_rects_overlap(*rect, *other))
    })
}

fn vk_rects_overlap(a: vk::Rect2D, b: vk::Rect2D) -> bool {
    let a_x = i64::from(a.offset.x);
    let a_y = i64::from(a.offset.y);
    let b_x = i64::from(b.offset.x);
    let b_y = i64::from(b.offset.y);
    let a_x_end = a_x + i64::from(a.extent.width);
    let a_y_end = a_y + i64::from(a.extent.height);
    let b_x_end = b_x + i64::from(b.extent.width);
    let b_y_end = b_y + i64::from(b.extent.height);

    a_x < b_x_end && b_x < a_x_end && a_y < b_y_end && b_y < a_y_end
}

pub(super) fn render_texture_damage_to_scissor_areas(
    output_size: Size<i32, Physical>,
    dst: Rectangle<i32, Physical>,
    damage: &[Rectangle<i32, Physical>],
    opaque_regions: &[Rectangle<i32, Physical>],
    is_implicit_opaque: bool,
    alpha: f32,
) -> Option<(Vec<vk::Rect2D>, Vec<vk::Rect2D>)> {
    let mut non_opaque_damage = Vec::new();
    let mut opaque_damage = Vec::new();

    if is_implicit_opaque {
        opaque_damage.extend_from_slice(damage);
    } else if alpha != 1.0 || opaque_regions.is_empty() {
        non_opaque_damage.extend_from_slice(damage);
    } else {
        non_opaque_damage.extend_from_slice(damage);
        opaque_damage.extend_from_slice(damage);
        non_opaque_damage =
            Rectangle::subtract_rects_many_in_place(non_opaque_damage, opaque_regions.iter().copied());
        opaque_damage =
            Rectangle::subtract_rects_many_in_place(opaque_damage, non_opaque_damage.iter().copied());
    }

    Some((
        damage_to_scissor_areas(output_size, dst, &non_opaque_damage)?,
        damage_to_scissor_areas(output_size, dst, &opaque_damage)?,
    ))
}

pub(super) fn source_to_uv_rect(
    texture_size: Size<i32, BufferCoord>,
    src: Rectangle<f64, BufferCoord>,
    y_inverted: bool,
    src_transform: Transform,
) -> Option<([f32; 2], [f32; 2], [f32; 2])> {
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
        Transform::_90 => ((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
        Transform::_270 => ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
        Transform::Flipped90 => ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
        Transform::Flipped180 => ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0)),
        Transform::Flipped270 => ((1.0, 1.0), (0.0, -1.0), (-1.0, 0.0)),
    };
    let uv_at = |x: f64, y: f64| {
        let u = (src.loc.x + x * src.size.w) / f64::from(texture_size.w);
        let v = (src.loc.y + y * src.size.h) / f64::from(texture_size.h);

        (u as f32, if y_inverted { (1.0 - v) as f32 } else { v as f32 })
    };
    let offset = uv_at(origin.0, origin.1);
    let x_end = uv_at(origin.0 + x_axis.0, origin.1 + x_axis.1);
    let y_end = uv_at(origin.0 + y_axis.0, origin.1 + y_axis.1);

    Some((
        [offset.0, offset.1],
        [x_end.0 - offset.0, x_end.1 - offset.1],
        [y_end.0 - offset.0, y_end.1 - offset.1],
    ))
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
