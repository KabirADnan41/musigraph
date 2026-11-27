//! DirectWrite + Direct2D rendering into a WIC memory bitmap. Windows only.
//!
//! Safety contract: every COM interface is owned by a `windows` smart pointer (released on drop); raw
//! pointers are used only for the duration of one call or one bitmap lock, with lengths taken from the
//! API (stride x height checked against the buffer size). COM is initialised per thread (MTA) and released when
//! the renderer is dropped (after its interfaces); keep the renderer on the thread that created it.

use crate::{Align, Cut, LineStyle, MAX_SIDE, TextBitmap, bound_text, is_rtl, tracking_safe};
use windows::Win32::Graphics::Direct2D::Common::{
    D2D1_ALPHA_MODE_PREMULTIPLIED, D2D1_COLOR_F, D2D1_PIXEL_FORMAT,
};
use windows::Win32::Graphics::Direct2D::{
    D2D1_DRAW_TEXT_OPTIONS_CLIP, D2D1_FACTORY_TYPE_SINGLE_THREADED, D2D1_FEATURE_LEVEL_DEFAULT,
    D2D1_RENDER_TARGET_PROPERTIES, D2D1_RENDER_TARGET_TYPE_DEFAULT, D2D1_RENDER_TARGET_USAGE_NONE,
    D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE, D2D1CreateFactory, ID2D1Factory,
};
use windows::Win32::Graphics::DirectWrite::{
    DWRITE_FACTORY_TYPE_SHARED, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL,
    DWRITE_FONT_WEIGHT, DWRITE_LINE_SPACING_METHOD_UNIFORM, DWRITE_READING_DIRECTION_RIGHT_TO_LEFT,
    DWRITE_TEXT_ALIGNMENT_CENTER, DWRITE_TEXT_ALIGNMENT_TRAILING, DWRITE_TEXT_METRICS,
    DWRITE_TEXT_RANGE, DWRITE_TRIMMING, DWRITE_TRIMMING_GRANULARITY_CHARACTER,
    DWRITE_TRIMMING_GRANULARITY_WORD, DWRITE_WORD_WRAPPING_EMERGENCY_BREAK,
    DWRITE_WORD_WRAPPING_NO_WRAP, DWRITE_WORD_WRAPPING_WRAP, DWriteCreateFactory, IDWriteFactory,
    IDWriteFontCollection, IDWriteTextLayout1,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;
use windows::Win32::Graphics::Imaging::{
    CLSID_WICImagingFactory, GUID_WICPixelFormat32bppPBGRA, IWICImagingFactory,
    WICBitmapCacheOnLoad, WICBitmapLockRead, WICRect,
};
use windows::Win32::System::Com::{
    CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize,
};
use windows::core::Interface;
use windows::core::{BOOL, PCWSTR, w};

const FALLBACK: PCWSTR = w!("Segoe UI");
const VAR_DISPLAY: PCWSTR = w!("Segoe UI Variable Display");
const VAR_TEXT: PCWSTR = w!("Segoe UI Variable Text");

/// `name` if the system font collection has that family, else the Segoe UI fallback.
///
/// # Safety
/// `dw` must be a live factory; `name` a static NUL-terminated string.
unsafe fn probe_family(dw: &IDWriteFactory, name: PCWSTR) -> PCWSTR {
    let mut coll: Option<IDWriteFontCollection> = None;
    let (mut idx, mut exists) = (0u32, BOOL(0));
    // SAFETY: out-pointers are locals that outlive the calls.
    let found = unsafe {
        dw.GetSystemFontCollection(&mut coll, false).is_ok()
            && coll
                .as_ref()
                .is_some_and(|c| c.FindFamilyName(name, &mut idx, &mut exists).is_ok())
    };
    if found && exists.as_bool() {
        name
    } else {
        FALLBACK
    }
}

/// Uninitialises COM on drop; owned by `TextRenderer` as its last field so it runs after the interfaces.
struct ComGuard(bool);

impl Drop for ComGuard {
    fn drop(&mut self) {
        if self.0 {
            // SAFETY: balances the successful CoInitializeEx on this thread.
            unsafe { CoUninitialize() };
        }
    }
}

pub struct TextRenderer {
    d2d: ID2D1Factory,
    dwrite: IDWriteFactory,
    wic: IWICImagingFactory,
    /// Resolved once at startup: Segoe UI Variable Display/Text, or Segoe UI where missing.
    display: PCWSTR,
    text: PCWSTR,
    /// Declared last so COM is uninitialised only after every interface above has been released.
    _com: ComGuard,
}

impl TextRenderer {
    /// Creates the factories on the current thread. `None` if DirectWrite/Direct2D/WIC are unavailable.
    pub fn new() -> Option<Self> {
        // SAFETY: plain apartment initialisation; an Err (already initialised differently) is not balanced.
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        let guard = ComGuard(hr.is_ok());
        // SAFETY: factory creation with valid, constant arguments; results are owned smart pointers.
        let (d2d, dwrite, wic) = unsafe {
            let d2d: ID2D1Factory =
                D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None).ok()?;
            let dwrite: IDWriteFactory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED).ok()?;
            let wic: IWICImagingFactory =
                CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER).ok()?;
            (d2d, dwrite, wic)
        };
        // SAFETY: live factory, static names.
        let (display, text) = unsafe {
            (
                probe_family(&dwrite, VAR_DISPLAY),
                probe_family(&dwrite, VAR_TEXT),
            )
        };
        Some(Self {
            d2d,
            dwrite,
            wic,
            display,
            text,
            _com: guard,
        })
    }

    /// True when the Segoe UI Variable cuts were found (false = Segoe UI fallback, e.g. Windows 10).
    pub fn has_variable_cuts(&self) -> bool {
        self.display != FALLBACK && self.text != FALLBACK
    }

    /// Renders one line (or, with `max_lines > 1`, a wrapped block) of text. `None` for empty/whitespace-only
    /// text or on any API failure. RTL text (see `is_rtl`) is laid out right-to-left and right-aligned in the
    /// bitmap; the bitmap is cropped to the ink box so callers align it to their column's start edge.
    pub fn render_line(&self, text: &str, style: &LineStyle) -> Option<TextBitmap> {
        let text = bound_text(text);
        if text.trim().is_empty() || !style.px.is_finite() || style.px < 4.0 || style.px > 400.0 {
            return None;
        }
        let pad = (style.px * 0.3).ceil() as u32;
        let max_text_w = style.max_width.clamp(8, MAX_SIDE - 2 * pad);
        let lines = style.max_lines.clamp(1, 4);
        let lh = if style.line_height.is_finite() && style.line_height >= 0.8 {
            style.line_height.min(2.0)
        } else {
            1.2
        };
        let wide: Vec<u16> = text.encode_utf16().collect();
        let family = match style.cut {
            Cut::Display => self.display,
            Cut::Text => self.text,
        };
        // SAFETY: all calls below use interfaces owned by `self`/locals with valid arguments; the pixel
        // buffer is read only while the WIC lock is alive and only within the length the API reports.
        unsafe {
            let fmt = self
                .dwrite
                .CreateTextFormat(
                    family,
                    None,
                    DWRITE_FONT_WEIGHT(i32::from(style.weight.clamp(100, 900))),
                    DWRITE_FONT_STYLE_NORMAL,
                    DWRITE_FONT_STRETCH_NORMAL,
                    style.px,
                    w!(""),
                )
                .ok()?;
            if is_rtl(text) {
                fmt.SetReadingDirection(DWRITE_READING_DIRECTION_RIGHT_TO_LEFT)
                    .ok()?;
            }
            let granularity = if lines == 1 {
                fmt.SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP).ok()?;
                DWRITE_TRIMMING_GRANULARITY_CHARACTER
            } else {
                // emergency break keeps space-less runs (CJK, URLs) inside the column; older systems: wrap
                if fmt
                    .SetWordWrapping(DWRITE_WORD_WRAPPING_EMERGENCY_BREAK)
                    .is_err()
                {
                    fmt.SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP).ok()?;
                }
                fmt.SetLineSpacing(
                    DWRITE_LINE_SPACING_METHOD_UNIFORM,
                    style.px * lh,
                    style.px * lh * 0.8,
                )
                .ok()?;
                DWRITE_TRIMMING_GRANULARITY_WORD
            };
            let sign = self.dwrite.CreateEllipsisTrimmingSign(&fmt).ok()?;
            let trimming = DWRITE_TRIMMING {
                granularity,
                delimiter: 0,
                delimiterCount: 0,
            };
            fmt.SetTrimming(&trimming, &sign).ok()?;
            let box_h = if lines == 1 {
                style.px * 3.0
            } else {
                style.px * lh * f32::from(lines)
            };
            let layout = self
                .dwrite
                .CreateTextLayout(&wide, &fmt, max_text_w as f32, box_h)
                .ok()?;
            if style.tracking_em != 0.0 && style.tracking_em.is_finite() && tracking_safe(text) {
                let sp = style.tracking_em.clamp(-0.05, 0.2) * style.px;
                let range = DWRITE_TEXT_RANGE {
                    startPosition: 0,
                    length: wide.len() as u32,
                };
                if let Ok(l1) = layout.cast::<IDWriteTextLayout1>() {
                    l1.SetCharacterSpacing(sp * 0.5, sp * 0.5, 0.0, range)
                        .ok()?;
                }
            }
            let mut m = DWRITE_TEXT_METRICS::default();
            layout.GetMetrics(&mut m).ok()?;
            let ink_w = (m
                .widthIncludingTrailingWhitespace
                .min(max_text_w as f32)
                .ceil() as u32)
                .max(1);
            // draw in a box exactly as wide as the text so RTL/right-aligned lines land inside the bitmap
            layout.SetMaxWidth(ink_w as f32).ok()?;
            match style.align {
                Align::Start => {}
                Align::Center => layout.SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER).ok()?,
                Align::End => layout
                    .SetTextAlignment(DWRITE_TEXT_ALIGNMENT_TRAILING)
                    .ok()?,
            }
            layout.GetMetrics(&mut m).ok()?;
            let w = ink_w + 2 * pad;
            let h = (m.height.min(box_h).ceil() as u32).max(1) + 2 * pad;
            if w > MAX_SIDE || h > MAX_SIDE {
                return None;
            }

            let bitmap = self
                .wic
                .CreateBitmap(w, h, &GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad)
                .ok()?;
            let props = D2D1_RENDER_TARGET_PROPERTIES {
                r#type: D2D1_RENDER_TARGET_TYPE_DEFAULT,
                pixelFormat: D2D1_PIXEL_FORMAT {
                    format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED,
                },
                dpiX: 96.0,
                dpiY: 96.0,
                usage: D2D1_RENDER_TARGET_USAGE_NONE,
                minLevel: D2D1_FEATURE_LEVEL_DEFAULT,
            };
            let rt = self.d2d.CreateWicBitmapRenderTarget(&bitmap, &props).ok()?;
            rt.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
            let white = D2D1_COLOR_F {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            };
            let brush = rt.CreateSolidColorBrush(&white, None).ok()?;
            rt.BeginDraw();
            rt.Clear(Some(&D2D1_COLOR_F {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            }));
            // The origin type (windows_numerics::Vector2, repr(C) {X: f32, Y: f32}) is not re-exported by the
            // `windows` crate; build it from two f32s rather than adding a direct dependency.
            #[allow(clippy::missing_transmute_annotations)]
            // the target type cannot be named, see above
            let origin = std::mem::transmute::<[f32; 2], _>([pad as f32, pad as f32]);
            rt.DrawTextLayout(origin, &layout, &brush, D2D1_DRAW_TEXT_OPTIONS_CLIP);
            rt.EndDraw(None, None).ok()?;

            let rect = WICRect {
                X: 0,
                Y: 0,
                Width: w as i32,
                Height: h as i32,
            };
            let lock = bitmap.Lock(&rect, WICBitmapLockRead.0 as u32).ok()?;
            let stride = lock.GetStride().ok()? as usize;
            let mut len = 0u32;
            let mut ptr = std::ptr::null_mut();
            lock.GetDataPointer(&mut len, &mut ptr).ok()?;
            let need = stride * (h as usize - 1) + w as usize * 4;
            if ptr.is_null() || stride < w as usize * 4 || (len as usize) < need {
                return None;
            }
            let data = std::slice::from_raw_parts(ptr, len as usize);
            let mut coverage = Vec::with_capacity(w as usize * h as usize);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    coverage.push(data[y * stride + x * 4 + 3]); // premultiplied BGRA: alpha = coverage
                }
            }
            Some(TextBitmap {
                width: w,
                height: h,
                pad,
                coverage,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style(max_width: u32) -> LineStyle {
        LineStyle {
            px: 40.0,
            weight: 600,
            max_width,
            ..LineStyle::default()
        }
    }

    /// Left edge (px from the bitmap's left) of the ink in the lowest `rows` rows of the ink box.
    fn bottom_left_ink(b: &TextBitmap, rows: u32) -> u32 {
        let (w, h) = (b.width as usize, b.height as usize);
        let last = (0..h)
            .rev()
            .find(|&y| b.coverage[y * w..(y + 1) * w].iter().any(|&c| c > 100))
            .expect("ink");
        let first = last.saturating_sub(rows as usize);
        (0..w)
            .find(|&x| (first..=last).any(|y| b.coverage[y * w + x] > 100))
            .expect("ink") as u32
    }

    #[test]
    fn paragraph_alignment_moves_the_short_last_line_and_keeps_the_default() {
        let r = TextRenderer::new().expect("directwrite available");
        let two = |align| LineStyle {
            max_lines: 2,
            line_height: 1.1,
            align,
            ..style(330)
        };
        let text = "Hold the light a little longer tonight";
        let start = r.render_line(text, &two(Align::Start)).expect("start");
        let centre = r.render_line(text, &two(Align::Center)).expect("centre");
        let end = r.render_line(text, &two(Align::End)).expect("end");
        assert_eq!(start.height, centre.height, "same two lines");
        let (s, c, e) = (
            bottom_left_ink(&start, 12),
            bottom_left_ink(&centre, 12),
            bottom_left_ink(&end, 12),
        );
        assert!(
            s <= start.pad + 6,
            "start-aligned text starts at the pad ({s})"
        );
        assert!(
            c > s + 6 && e > c + 6,
            "centre {c} and end {e} are further right than start {s}"
        );
        // the default style is unchanged (Start)
        assert_eq!(LineStyle::default().align, Align::Start);
        assert_eq!(
            r.render_line(
                text,
                &LineStyle {
                    align: Align::Start,
                    ..two(Align::Start)
                }
            )
            .unwrap(),
            start
        );
    }

    #[test]
    fn latin_bangla_arabic_and_cjk_all_draw_ink() {
        let r = TextRenderer::new().expect("directwrite available");
        let mut widths = Vec::new();
        for s in [
            "Hello, world",
            "আমার সোনার বাংলা",
            "مرحبا بالعالم",
            "歌曲の名前",
            "Ünïcödé ✓ 🎵",
        ] {
            let b = r
                .render_line(s, &style(2000))
                .unwrap_or_else(|| panic!("no bitmap for {s}"));
            assert!(b.ink() > 200, "{s}: ink {}", b.ink());
            assert_eq!(b.coverage.len(), (b.width * b.height) as usize);
            widths.push(b.width);
        }
        assert!(
            widths.windows(2).any(|w| w[0] != w[1]),
            "different scripts shape to different widths"
        );
    }

    #[test]
    fn margins_stay_empty_and_long_text_is_trimmed_to_the_width() {
        let r = TextRenderer::new().unwrap();
        let long =
            "A very long song title that cannot possibly fit in the available width ".repeat(3);
        let b = r.render_line(&long, &style(500)).unwrap();
        assert!(b.width <= 500 + 2 * b.pad, "width {}", b.width);
        let (pad, w) = (b.pad as usize, b.width as usize);
        for y in 0..b.height as usize {
            assert!(
                b.coverage[y * w..y * w + pad].iter().all(|&c| c == 0),
                "left margin row {y}"
            );
        }
        // trimmed text still ends with visible ink near the right edge (the ellipsis)
        let right = (0..b.height as usize).any(|y| {
            b.coverage[y * w + w - pad - 6..y * w + w - pad]
                .iter()
                .any(|&c| c > 24)
        });
        assert!(right, "ellipsis expected at the trimmed end");
    }

    fn ink_cols(b: &TextBitmap) -> (usize, usize) {
        let w = b.width as usize;
        let cols: Vec<usize> = (0..w)
            .filter(|&x| (0..b.height as usize).any(|y| b.coverage[y * w + x] > 24))
            .collect();
        (cols[0], *cols.last().unwrap())
    }

    #[test]
    fn wrapping_is_opt_in_bounded_by_lines_and_width() {
        let r = TextRenderer::new().unwrap();
        let title = "An extremely long song title that will certainly need two lines here and more";
        let one = r.render_line(title, &style(600)).unwrap();
        let two = r
            .render_line(
                title,
                &LineStyle {
                    max_lines: 2,
                    line_height: 1.1,
                    ..style(600)
                },
            )
            .unwrap();
        let adv = 40.0 * 1.1;
        let body = |b: &TextBitmap| (b.height - 2 * b.pad) as f32;
        assert!(body(&one) < adv * 1.6, "default stays one line");
        assert!(
            body(&two) > adv * 1.6 && body(&two) <= adv * 2.0 + 1.0,
            "two lines, never three: {}",
            body(&two)
        );
        assert!(two.width <= 600 + 2 * two.pad);
        // a CJK run without spaces still wraps inside the column (emergency break / CJK line breaking)
        let cjk = r
            .render_line(
                &"歌曲の名前".repeat(30),
                &LineStyle {
                    max_lines: 2,
                    ..style(500)
                },
            )
            .unwrap();
        assert!(cjk.width <= 500 + 2 * cjk.pad && body(&cjk) > 40.0 * 1.6);
    }

    #[test]
    fn tracking_applies_to_latin_only_and_rtl_ink_stays_inside() {
        let r = TextRenderer::new().unwrap();
        let tracked = |s: &str, em: f32| {
            r.render_line(
                s,
                &LineStyle {
                    tracking_em: em,
                    ..style(2000)
                },
            )
            .unwrap()
            .width
        };
        assert!(tracked("Midnight City", -0.04) < tracked("Midnight City", 0.0));
        assert_eq!(
            tracked("مرحبا بالعالم", -0.04),
            tracked("مرحبا بالعالم", 0.0)
        );
        for s in ["مرحبا بالعالم", "שלום עולם"] {
            let b = r
                .render_line(
                    s,
                    &LineStyle {
                        max_lines: 2,
                        ..style(1200)
                    },
                )
                .unwrap();
            let (l, rt) = ink_cols(&b);
            assert!(
                l >= b.pad as usize - 2 && rt <= (b.width - b.pad) as usize + 2,
                "{s}: ink {l}..{rt} in width {} pad {}",
                b.width,
                b.pad
            );
        }
    }

    #[test]
    fn display_and_text_cuts_both_render() {
        let r = TextRenderer::new().unwrap();
        for cut in [Cut::Display, Cut::Text] {
            let b = r
                .render_line("Album Focus", &LineStyle { cut, ..style(2000) })
                .unwrap();
            assert!(b.ink() > 200);
        }
        eprintln!("variable cuts available: {}", r.has_variable_cuts());
    }

    #[test]
    fn empty_and_absurd_requests_are_refused() {
        let r = TextRenderer::new().unwrap();
        assert!(r.render_line("", &style(100)).is_none());
        assert!(r.render_line("   \t ", &style(100)).is_none());
        let bad = |px| LineStyle {
            px,
            weight: 400,
            max_width: 100,
            ..LineStyle::default()
        };
        assert!(r.render_line("x", &bad(f32::NAN)).is_none());
        assert!(r.render_line("x", &bad(5000.0)).is_none());
    }
}
