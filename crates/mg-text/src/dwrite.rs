//! DirectWrite + Direct2D rendering into a WIC memory bitmap. Windows only.
//!
//! Safety contract: every COM interface is owned by a `windows` smart pointer (released on drop); raw
//! pointers are used only for the duration of one call or one bitmap lock, with lengths taken from the
//! API (stride x height checked against the buffer size). COM is initialised per thread (MTA); keep the
//! renderer and its guard on one worker thread and drop the renderer before the guard.

use crate::{LineStyle, MAX_SIDE, TextBitmap, bound_text};
use windows::Win32::Graphics::Direct2D::Common::{
    D2D_RECT_F, D2D1_ALPHA_MODE_PREMULTIPLIED, D2D1_COLOR_F, D2D1_PIXEL_FORMAT,
};
use windows::Win32::Graphics::Direct2D::{
    D2D1_DRAW_TEXT_OPTIONS_NONE, D2D1_FACTORY_TYPE_SINGLE_THREADED, D2D1_FEATURE_LEVEL_DEFAULT,
    D2D1_RENDER_TARGET_PROPERTIES, D2D1_RENDER_TARGET_TYPE_DEFAULT, D2D1_RENDER_TARGET_USAGE_NONE,
    D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE, D2D1CreateFactory, ID2D1Factory,
};
use windows::Win32::Graphics::DirectWrite::{
    DWRITE_FACTORY_TYPE_SHARED, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL,
    DWRITE_FONT_WEIGHT, DWRITE_MEASURING_MODE_NATURAL, DWRITE_TEXT_METRICS, DWRITE_TRIMMING,
    DWRITE_TRIMMING_GRANULARITY_CHARACTER, DWRITE_WORD_WRAPPING_NO_WRAP, DWriteCreateFactory,
    IDWriteFactory,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;
use windows::Win32::Graphics::Imaging::{
    CLSID_WICImagingFactory, GUID_WICPixelFormat32bppPBGRA, IWICImagingFactory,
    WICBitmapCacheOnLoad, WICBitmapLockRead, WICRect,
};
use windows::Win32::System::Com::{
    CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize,
};
use windows::core::{PCWSTR, w};

const FAMILY: PCWSTR = w!("Segoe UI");

/// Uninitialises COM on drop. Keep it alive as long as the renderer and drop the renderer first.
pub struct ComGuard(bool);

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
}

impl TextRenderer {
    /// Creates the factories on the current thread. `None` if DirectWrite/Direct2D/WIC are unavailable.
    /// The returned guard must outlive the renderer (declare it first, so it drops last).
    pub fn new() -> Option<(ComGuard, Self)> {
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
        Some((guard, Self { d2d, dwrite, wic }))
    }

    /// Renders one line of text. `None` for empty/whitespace-only text or on any API failure.
    pub fn render_line(&self, text: &str, style: &LineStyle) -> Option<TextBitmap> {
        let text = bound_text(text);
        if text.trim().is_empty() || !style.px.is_finite() || style.px < 4.0 || style.px > 400.0 {
            return None;
        }
        let pad = (style.px * 0.3).ceil() as u32;
        let max_text_w = style.max_width.clamp(8, MAX_SIDE - 2 * pad);
        let wide: Vec<u16> = text.encode_utf16().collect();
        // SAFETY: all calls below use interfaces owned by `self`/locals with valid arguments; the pixel
        // buffer is read only while the WIC lock is alive and only within the length the API reports.
        unsafe {
            let fmt = self
                .dwrite
                .CreateTextFormat(
                    FAMILY,
                    None,
                    DWRITE_FONT_WEIGHT(i32::from(style.weight.clamp(100, 900))),
                    DWRITE_FONT_STYLE_NORMAL,
                    DWRITE_FONT_STRETCH_NORMAL,
                    style.px,
                    w!(""),
                )
                .ok()?;
            fmt.SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP).ok()?;
            let sign = self.dwrite.CreateEllipsisTrimmingSign(&fmt).ok()?;
            let trimming = DWRITE_TRIMMING {
                granularity: DWRITE_TRIMMING_GRANULARITY_CHARACTER,
                delimiter: 0,
                delimiterCount: 0,
            };
            fmt.SetTrimming(&trimming, &sign).ok()?;
            let layout = self
                .dwrite
                .CreateTextLayout(&wide, &fmt, max_text_w as f32, style.px * 3.0)
                .ok()?;
            let mut m = DWRITE_TEXT_METRICS::default();
            layout.GetMetrics(&mut m).ok()?;
            let w = (m
                .widthIncludingTrailingWhitespace
                .min(max_text_w as f32)
                .ceil() as u32)
                .max(1)
                + 2 * pad;
            let h = (m.height.ceil() as u32).max(1) + 2 * pad;
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
            let area = D2D_RECT_F {
                left: pad as f32,
                top: pad as f32,
                right: (pad + max_text_w) as f32,
                bottom: h as f32,
            };
            rt.DrawText(
                &wide,
                &fmt,
                &area,
                &brush,
                D2D1_DRAW_TEXT_OPTIONS_NONE,
                DWRITE_MEASURING_MODE_NATURAL,
            );
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
        }
    }

    #[test]
    fn latin_bangla_arabic_and_cjk_all_draw_ink() {
        let (_g, r) = TextRenderer::new().expect("directwrite available");
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
        let (_g, r) = TextRenderer::new().unwrap();
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

    #[test]
    fn empty_and_absurd_requests_are_refused() {
        let (_g, r) = TextRenderer::new().unwrap();
        assert!(r.render_line("", &style(100)).is_none());
        assert!(r.render_line("   \t ", &style(100)).is_none());
        let bad = |px| LineStyle {
            px,
            weight: 400,
            max_width: 100,
        };
        assert!(r.render_line("x", &bad(f32::NAN)).is_none());
        assert!(r.render_line("x", &bad(5000.0)).is_none());
    }
}
