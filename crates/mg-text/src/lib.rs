//! Text rendering for the immersive scene: one line of Unicode text -> an 8-bit coverage bitmap.
//!
//! Windows DirectWrite does the hard parts that a hand-written glyph atlas cannot: complex-script shaping
//! (Bangla, Devanagari, Arabic, Thai ...), bidirectional runs, and system font fallback. The bitmap holds
//! glyph COVERAGE only (white text, alpha = coverage); the GPU tints it with the album palette, so one
//! bitmap serves every colour theme. No network, no files, nothing persisted, nothing logged.
//!
//! The input text is untrusted (it comes from a media session): callers should sanitise it first
//! (`mg_media::sanitize_text`), and this crate additionally bounds length, size and the bitmap area.

#![deny(unsafe_code)]

#[cfg(windows)]
#[allow(unsafe_code)] // all COM/FFI lives in this one module; see its safety notes
mod dwrite;
#[cfg(windows)]
pub use dwrite::TextRenderer;

/// Largest text accepted per line (characters); longer input is cut before layout.
pub const MAX_CHARS: usize = 300;
/// Largest bitmap side produced (pixels).
pub const MAX_SIDE: u32 = 4096;

/// How to render one line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineStyle {
    /// Font size in pixels (device pixels: the caller scales for the window/DPI).
    pub px: f32,
    /// 400 regular .. 700 bold.
    pub weight: u16,
    /// Maximum text width in pixels; longer text is trimmed with an ellipsis.
    pub max_width: u32,
}

/// Coverage bitmap: `coverage[y * width + x]`, 0 = transparent, 255 = fully covered. Includes `pad` pixels
/// of empty margin on every side so GPU shadows/halos never clip at the edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextBitmap {
    pub width: u32,
    pub height: u32,
    pub pad: u32,
    pub coverage: Vec<u8>,
}

impl TextBitmap {
    /// Number of pixels with visible coverage (used by tests and to detect "nothing drawn").
    pub fn ink(&self) -> usize {
        self.coverage.iter().filter(|&&c| c > 24).count()
    }
}

/// Cuts text to `MAX_CHARS` characters (never in the middle of a character).
pub fn bound_text(s: &str) -> &str {
    match s.char_indices().nth(MAX_CHARS) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_text_is_bounded_on_a_character_boundary() {
        let s = "আ".repeat(1000);
        assert_eq!(bound_text(&s).chars().count(), MAX_CHARS);
        assert_eq!(bound_text("short"), "short");
    }
}
