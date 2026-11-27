//! Stress-shader renderer (P3.3): a fullscreen fragment shader whose cost approximates the planned
//! scene (haze noise, 64 bars, reflection, glow). It uses synthetic bands and exists only for
//! benchmarks; it is never used for live music (ADR-0008).

use crate::context::{FrameStatus, Gpu};
use crate::fullscreen::FullscreenRenderer;
pub use crate::fullscreen::scaled_size;

const UNIFORM_BYTES: usize = 16 + 16 * 16;

pub struct StressRenderer {
    inner: FullscreenRenderer,
    /// Time to create the shader modules and pipelines (the driver may compile later, on first use).
    pub compile_ms: f64,
}

impl StressRenderer {
    pub fn new(gpu: &Gpu, scale: f32) -> Self {
        let inner = FullscreenRenderer::new(
            &gpu.device,
            gpu.config.format,
            scale,
            "stress",
            include_str!("stress.wgsl"),
            UNIFORM_BYTES,
        );
        let compile_ms = inner.compile_ms;
        Self { inner, compile_ms }
    }

    /// Internal render resolution for the given surface size.
    pub fn render_size(&self, surface: (u32, u32)) -> (u32, u32) {
        self.inner.render_size(surface)
    }

    /// Draws one frame with synthetic, deterministic "bands" derived from `time`.
    pub fn render(&mut self, gpu: &mut Gpu, time: f32) -> Result<FrameStatus, String> {
        let (rw, rh) = self.inner.current_render_size(gpu);
        let bytes = uniform_bytes(rw as f32, rh as f32, time);
        self.inner.render(gpu, &bytes)
    }
}
/// Packs the shader uniform (res, time, level, 64 synthetic bands) in the WGSL `U` layout (272 bytes).
pub fn uniform_bytes(w: f32, h: f32, time: f32) -> [u8; UNIFORM_BYTES] {
    let mut bytes = [0u8; UNIFORM_BYTES];
    let level = 0.5 + 0.5 * (time * 2.1).sin();
    for (i, v) in [w, h, time, level].iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
    }
    for b in 0..64usize {
        let x = b as f32 / 64.0;
        let v = (0.55 + 0.45 * (time * (1.5 + x * 4.0) + b as f32 * 0.7).sin())
            * (1.0 - 0.6 * x)
            * (0.6 + 0.4 * level);
        bytes[16 + b * 4..16 + b * 4 + 4].copy_from_slice(&v.clamp(0.02, 1.0).to_ne_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(b: &[u8], off: usize) -> f32 {
        f32::from_ne_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }

    #[test]
    fn uniform_layout_matches_wgsl() {
        let b = uniform_bytes(1920.0, 1080.0, 1.25);
        assert_eq!(b.len(), 272);
        assert_eq!(f32_at(&b, 0), 1920.0);
        assert_eq!(f32_at(&b, 4), 1080.0);
        assert_eq!(f32_at(&b, 8), 1.25);
        for i in 0..64 {
            let v = f32_at(&b, 16 + i * 4);
            assert!(v.is_finite() && (0.02..=1.0).contains(&v), "band {i} = {v}");
        }
    }

    #[test]
    fn shader_source_declares_matching_struct() {
        let s = include_str!("stress.wgsl");
        assert!(s.contains("bands: array<vec4<f32>, 16>"));
        assert!(s.contains("res: vec2<f32>"));
    }

    #[test]
    fn scaled_size_is_exact_and_safe() {
        assert_eq!(scaled_size((1920, 1080), 1.0), (1920, 1080));
        assert_eq!(scaled_size((1920, 1080), 0.75), (1440, 810));
        assert_eq!(scaled_size((1920, 1080), 0.5), (960, 540));
        assert_eq!(scaled_size((1, 1), 0.25), (1, 1));
        assert_eq!(scaled_size((1920, 1080), f32::NAN), (1920, 1080));
    }
}
