//! Format mapping helpers for the Vulkan renderer.
//!
//! These helpers are table-driven and do not require a live Vulkan device. They only describe
//! renderer-side DRM fourcc to Vulkan format mappings and basic format semantics; actual device
//! support must be queried later before enabling import, export, or rendering capability bits.
//!
//! This module deliberately uses UNORM renderer formats for 8-bit colour formats. The Vulkan
//! allocator has a compatibility table that currently maps some FourCC formats to sRGB Vulkan
//! formats for external-memory compatibility; that table is not renderer colour policy.

use ash::vk;

use crate::backend::allocator::Fourcc;

use super::VulkanError;

/// Vulkan renderer format information for a DRM fourcc.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanFormatInfo {
    /// DRM format.
    pub fourcc: Fourcc,
    /// Vulkan format used by the renderer.
    pub vk_format: vk::Format,
    /// Whether the DRM format carries alpha.
    pub has_alpha: bool,
    /// Whether the Vulkan format has an alpha component that must be forced opaque for this DRM format.
    pub opaque_alpha: bool,
    /// Whether the DRM format is a 10-bit format.
    pub is_10bit: bool,
}

#[derive(Debug, Clone, Copy)]
struct FormatSemantics {
    fourcc: Fourcc,
    vk_format: vk::Format,
    has_alpha: bool,
    opaque_alpha: bool,
    is_10bit: bool,
}

const FORMAT_TABLE: &[FormatSemantics] = &[
    FormatSemantics {
        fourcc: Fourcc::Xrgb8888,
        vk_format: vk::Format::B8G8R8A8_UNORM,
        has_alpha: false,
        opaque_alpha: true,
        is_10bit: false,
    },
    FormatSemantics {
        fourcc: Fourcc::Argb8888,
        vk_format: vk::Format::B8G8R8A8_UNORM,
        has_alpha: true,
        opaque_alpha: false,
        is_10bit: false,
    },
    FormatSemantics {
        fourcc: Fourcc::Xbgr8888,
        vk_format: vk::Format::R8G8B8A8_UNORM,
        has_alpha: false,
        opaque_alpha: true,
        is_10bit: false,
    },
    FormatSemantics {
        fourcc: Fourcc::Abgr8888,
        vk_format: vk::Format::R8G8B8A8_UNORM,
        has_alpha: true,
        opaque_alpha: false,
        is_10bit: false,
    },
    #[cfg(target_endian = "little")]
    FormatSemantics {
        fourcc: Fourcc::Rgbx8888,
        vk_format: vk::Format::A8B8G8R8_UNORM_PACK32,
        has_alpha: false,
        opaque_alpha: true,
        is_10bit: false,
    },
    #[cfg(target_endian = "little")]
    FormatSemantics {
        fourcc: Fourcc::Rgba8888,
        vk_format: vk::Format::A8B8G8R8_UNORM_PACK32,
        has_alpha: true,
        opaque_alpha: false,
        is_10bit: false,
    },
    #[cfg(target_endian = "little")]
    FormatSemantics {
        fourcc: Fourcc::Rgb565,
        vk_format: vk::Format::R5G6B5_UNORM_PACK16,
        has_alpha: false,
        opaque_alpha: false,
        is_10bit: false,
    },
    #[cfg(target_endian = "little")]
    FormatSemantics {
        fourcc: Fourcc::Xrgb2101010,
        vk_format: vk::Format::A2R10G10B10_UNORM_PACK32,
        has_alpha: false,
        opaque_alpha: true,
        is_10bit: true,
    },
    #[cfg(target_endian = "little")]
    FormatSemantics {
        fourcc: Fourcc::Argb2101010,
        vk_format: vk::Format::A2R10G10B10_UNORM_PACK32,
        has_alpha: true,
        opaque_alpha: false,
        is_10bit: true,
    },
    #[cfg(target_endian = "little")]
    FormatSemantics {
        fourcc: Fourcc::Xbgr2101010,
        vk_format: vk::Format::A2B10G10R10_UNORM_PACK32,
        has_alpha: false,
        opaque_alpha: true,
        is_10bit: true,
    },
    #[cfg(target_endian = "little")]
    FormatSemantics {
        fourcc: Fourcc::Abgr2101010,
        vk_format: vk::Format::A2B10G10R10_UNORM_PACK32,
        has_alpha: true,
        opaque_alpha: false,
        is_10bit: true,
    },
];

/// Returns static format information for a DRM fourcc.
pub fn get_format_info(fourcc: Fourcc) -> Result<VulkanFormatInfo, VulkanError> {
    let semantics = get_format_semantics(fourcc)?;

    Ok(VulkanFormatInfo {
        fourcc,
        vk_format: semantics.vk_format,
        has_alpha: semantics.has_alpha,
        opaque_alpha: semantics.opaque_alpha,
        is_10bit: semantics.is_10bit,
    })
}

pub(crate) fn renderer_format_infos() -> impl Iterator<Item = VulkanFormatInfo> {
    FORMAT_TABLE.iter().copied().map(|semantics| VulkanFormatInfo {
        fourcc: semantics.fourcc,
        vk_format: semantics.vk_format,
        has_alpha: semantics.has_alpha,
        opaque_alpha: semantics.opaque_alpha,
        is_10bit: semantics.is_10bit,
    })
}

/// Converts a DRM fourcc to the Vulkan renderer's static format mapping.
///
/// This does not probe device support and is not a renderer render-target or colour-management
/// policy. Capability discovery must validate this format for each intended image usage.
pub fn get_render_vk_format(fourcc: Fourcc) -> Result<vk::Format, VulkanError> {
    get_format_semantics(fourcc).map(|semantics| semantics.vk_format)
}

fn get_format_semantics(fourcc: Fourcc) -> Result<FormatSemantics, VulkanError> {
    FORMAT_TABLE
        .iter()
        .copied()
        .find(|info| info.fourcc == fourcc)
        .ok_or(VulkanError::UnsupportedFormat(fourcc))
}

/// Returns whether the DRM fourcc carries alpha.
pub fn has_alpha(fourcc: Fourcc) -> Result<bool, VulkanError> {
    get_format_info(fourcc).map(|info| info.has_alpha)
}

/// Returns whether the DRM fourcc is a 10-bit format.
pub fn is_10bit(fourcc: Fourcc) -> Result<bool, VulkanError> {
    get_format_info(fourcc).map(|info| info.is_10bit)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::backend::allocator::Fourcc;
    use ash::vk;

    use super::*;

    #[test]
    fn all_renderer_formats_have_table_mapping_and_semantics() {
        for info in FORMAT_TABLE.iter().copied() {
            let format_info = get_format_info(info.fourcc).unwrap();
            assert_eq!(format_info.vk_format, info.vk_format);
            assert_eq!(format_info.has_alpha, info.has_alpha);
            assert_eq!(format_info.opaque_alpha, info.opaque_alpha);
            assert_eq!(format_info.is_10bit, info.is_10bit);
        }
    }

    #[test]
    fn renderer_format_table_has_no_duplicate_fourcc_entries() {
        let mut seen = HashSet::new();

        for info in FORMAT_TABLE.iter().copied() {
            assert!(
                seen.insert(info.fourcc),
                "duplicate Vulkan format entry for {:?}",
                info.fourcc
            );
        }
    }

    #[test]
    fn renderer_formats_do_not_use_srgb_without_colour_policy() {
        for info in FORMAT_TABLE.iter().copied() {
            assert!(
                !matches!(
                    info.vk_format,
                    vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32
                ),
                "renderer format for {:?} must not imply sRGB policy",
                info.fourcc
            );
        }
    }

    #[test]
    fn opaque_alpha_only_applies_to_non_alpha_formats() {
        for info in FORMAT_TABLE.iter().copied() {
            if info.opaque_alpha {
                assert!(
                    !info.has_alpha,
                    "{:?} cannot both carry and force alpha",
                    info.fourcc
                );
            }
        }
    }

    #[test]
    fn xrgb8888_maps_to_expected_vk_format() {
        assert_eq!(
            get_render_vk_format(Fourcc::Xrgb8888).unwrap(),
            vk::Format::B8G8R8A8_UNORM
        );
    }

    #[test]
    fn argb8888_preserves_alpha_semantics() {
        assert_eq!(
            get_render_vk_format(Fourcc::Argb8888).unwrap(),
            vk::Format::B8G8R8A8_UNORM
        );
        assert!(matches!(has_alpha(Fourcc::Argb8888), Ok(true)));
        assert!(matches!(has_alpha(Fourcc::Xrgb8888), Ok(false)));
    }

    #[test]
    fn abgr8888_maps_correctly() {
        assert_eq!(
            get_render_vk_format(Fourcc::Abgr8888).unwrap(),
            vk::Format::R8G8B8A8_UNORM
        );
        assert!(matches!(has_alpha(Fourcc::Abgr8888), Ok(true)));
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn rgba8888_and_rgbx8888_map_to_unorm_pack32() {
        assert_eq!(
            get_render_vk_format(Fourcc::Rgba8888).unwrap(),
            vk::Format::A8B8G8R8_UNORM_PACK32
        );
        assert_eq!(
            get_render_vk_format(Fourcc::Rgbx8888).unwrap(),
            vk::Format::A8B8G8R8_UNORM_PACK32
        );
        assert!(matches!(has_alpha(Fourcc::Rgba8888), Ok(true)));
        assert!(matches!(has_alpha(Fourcc::Rgbx8888), Ok(false)));
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn rgb565_maps_to_r5g6b5_unorm_on_little_endian() {
        assert_eq!(
            get_render_vk_format(Fourcc::Rgb565).unwrap(),
            vk::Format::R5G6B5_UNORM_PACK16
        );
    }

    #[test]
    #[cfg(not(target_endian = "little"))]
    fn rgb565_is_not_implied_on_big_endian() {
        assert!(matches!(
            get_format_info(Fourcc::Rgb565),
            Err(VulkanError::UnsupportedFormat(Fourcc::Rgb565))
        ));
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn xrgb2101010_is_identified_as_10bit() {
        assert_eq!(
            get_render_vk_format(Fourcc::Xrgb2101010).unwrap(),
            vk::Format::A2R10G10B10_UNORM_PACK32
        );
        assert!(matches!(is_10bit(Fourcc::Xrgb2101010), Ok(true)));
        assert!(matches!(is_10bit(Fourcc::Xrgb8888), Ok(false)));
    }

    #[test]
    #[cfg(not(target_endian = "little"))]
    fn xrgb2101010_semantics_do_not_imply_mapping_on_big_endian() {
        assert!(matches!(
            get_format_info(Fourcc::Xrgb2101010),
            Err(VulkanError::UnsupportedFormat(Fourcc::Xrgb2101010))
        ));
    }

    #[test]
    fn formats_outside_renderer_table_return_unsupported() {
        assert!(matches!(
            get_format_info(Fourcc::Bgra8888),
            Err(VulkanError::UnsupportedFormat(Fourcc::Bgra8888))
        ));
    }

    #[test]
    fn unknown_fourcc_returns_unsupported() {
        assert!(matches!(
            get_render_vk_format(Fourcc::Yuyv),
            Err(VulkanError::UnsupportedFormat(Fourcc::Yuyv))
        ));
    }

    #[test]
    fn opaque_drm_formats_require_alpha_override() {
        assert!(get_format_info(Fourcc::Xrgb8888).unwrap().opaque_alpha);
        assert!(get_format_info(Fourcc::Xbgr8888).unwrap().opaque_alpha);
        assert!(!get_format_info(Fourcc::Argb8888).unwrap().opaque_alpha);
        assert!(!get_format_info(Fourcc::Abgr8888).unwrap().opaque_alpha);
    }
}
