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

/// Which optical cut of the UI face to use (Segoe UI Variable Display for large type, Text for small type;
/// both fall back to Segoe UI where the variable family is missing, e.g. Windows 10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Cut {
    #[default]
    Text,
    Display,
}

/// How to render one text block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineStyle {
    /// Font size in pixels (device pixels: the caller scales for the window/DPI).
    pub px: f32,
    /// 400 regular .. 700 bold.
    pub weight: u16,
    /// Maximum text width in pixels; longer text is trimmed with an ellipsis.
    pub max_width: u32,
    /// 1 = a single unwrapped line trimmed per character (the default). 2..=4 = word wrap (emergency break
    /// for runs without spaces) up to this many lines, then a word-granularity ellipsis.
    pub max_lines: u8,
    /// Letter spacing in em (e.g. -0.015). Applied only to Latin/Greek/Cyrillic text (`tracking_safe`):
    /// spacing breaks joining/shaping in Arabic, Bangla and similar scripts.
    pub tracking_em: f32,
    /// Line advance as a multiple of `px` for wrapped text (0 = 1.2).
    pub line_height: f32,
    pub cut: Cut,
    /// Alignment of the lines of a wrapped block inside the bitmap (the widest line sets its width).
    pub align: Align,
}

/// Paragraph alignment in the reading direction: `Start` is left for left-to-right text, right for right-to-left.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
    #[default]
    Start,
    Center,
    End,
}

impl Default for LineStyle {
    fn default() -> Self {
        Self {
            px: 16.0,
            weight: 400,
            max_width: 1000,
            max_lines: 1,
            tracking_em: 0.0,
            line_height: 0.0,
            cut: Cut::Text,
            align: Align::Start,
        }
    }
}

/// True when the first strongly directional character is right-to-left (Hebrew, Arabic, Syriac, Thaana,
/// NKo, and their presentation forms). Callers align such text to the column's start edge (the right).
pub fn is_rtl(s: &str) -> bool {
    for c in s.chars() {
        let u = c as u32;
        if matches!(u, 0x0590..=0x08FF | 0xFB1D..=0xFDFF | 0xFE70..=0xFEFF | 0x10800..=0x10FFF | 0x1E800..=0x1EFFF)
        {
            return true;
        }
        if c.is_alphabetic() {
            return false;
        }
    }
    false
}

/// True when every letter is Latin, Greek or Cyrillic, so letter-spacing cannot break shaping.
pub fn tracking_safe(s: &str) -> bool {
    s.chars()
        .filter(|c| c.is_alphabetic())
        .all(|c| (c as u32) < 0x0530 || matches!(c as u32, 0x1D00..=0x1FFF))
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
    fn direction_and_tracking_follow_the_script() {
        assert!(is_rtl("مرحبا بالعالم"));
        assert!(is_rtl("  «שלום» world"));
        assert!(!is_rtl("Hello مرحبا"));
        assert!(!is_rtl("123 আমার"));
        assert!(!is_rtl(""));
        assert!(tracking_safe("Ünïcödé Ελλάδα Привет 123 ✓"));
        assert!(!tracking_safe("Hello আমার"));
        assert!(!tracking_safe("歌曲"));
        assert!(!tracking_safe("مرحبا"));
    }

    #[test]
    fn long_text_is_bounded_on_a_character_boundary() {
        let s = "আ".repeat(1000);
        assert_eq!(bound_text(&s).chars().count(), MAX_CHARS);
        assert_eq!(bound_text("short"), "short");
    }
}
