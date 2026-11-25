//! Scene 1 "Cinematic Spectrum": the GPU-side parameter block and renderer.
//!
//! `SceneParams` is plain data produced by the music-to-motion mapper (mg-engine). The shader draws
//! exactly what it is given; it never invents audio activity. Idle motion (camera drift, horizon
//! breathing) is time-based and uncorrelated with audio. Quality (render scale) only changes cost.

use crate::context::{FrameStatus, Gpu};
use crate::fullscreen::FullscreenRenderer;

/// Mirrored bar slots across the screen (bass in the centre, treble at the edges).
pub const SLOTS: usize = 96;
const VEC4S: usize = SLOTS / 4;
/// Bytes of the packed uniform block (must equal the WGSL `U` struct size).
pub const UNIFORM_BYTES: usize = 16 + 16 + 16 + 16 + 64 + 3 * VEC4S * 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneMode {
    /// Real system audio.
    Live,
    /// Explicitly labelled synthetic test signal (marker drawn in the corner).
    Synthetic,
    /// No usable audio (capture failed or stale): dashed amber horizon line, no music activity.
    NoAudio,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SceneParams {
    pub time: f32,
    pub energy: f32,
    pub glow: f32,
    pub haze: f32,
    pub warp: f32,
    pub swell: f32,
    pub tilt: f32,
    pub scroll: [f32; 2],
    /// Dust fall phase in 0..1 (one full period of the wrapped dust grid), integrated on the CPU so
    /// changing energy never makes the field jump and the hash input stays small.
    pub dust: f32,
    /// Two impulse rings: (age in seconds, amplitude).
    pub rings: [[f32; 2]; 2],
    /// Linear RGB roles: haze/base, bar body, accent/tips, secondary (haze tint and curtain).
    pub palette: [[f32; 3]; 4],
    pub mode: SceneMode,
    pub bars: [f32; SLOTS],
    pub peaks: [f32; SLOTS],
    pub bright: [f32; SLOTS],
}

impl Default for SceneParams {
    fn default() -> Self {
        Self {
            time: 0.0,
            energy: 0.0,
            glow: 0.25,
            haze: 0.3,
            warp: 0.15,
            swell: 0.0,
            tilt: 0.0,
            scroll: [0.0, 0.0],
            dust: 0.0,
            rings: [[100.0, 0.0]; 2],
            palette: [
                [0.08, 0.16, 0.36],
                [0.10, 0.45, 0.60],
                [0.55, 0.85, 0.95],
                [0.32, 0.65, 0.78],
            ],
            mode: SceneMode::NoAudio,
            bars: [0.0; SLOTS],
            peaks: [0.0; SLOTS],
            bright: [0.0; SLOTS],
        }
    }
}

fn put(out: &mut Vec<u8>, vals: &[f32]) {
    for v in vals {
        // Never let a NaN/inf reach the shader.
        let v = if v.is_finite() { *v } else { 0.0 };
        out.extend_from_slice(&v.to_ne_bytes());
    }
}

impl SceneParams {
    /// Packs the WGSL `U` layout for a render target of `res` pixels.
    pub fn pack(&self, res: (u32, u32)) -> Vec<u8> {
        let mut o = Vec::with_capacity(UNIFORM_BYTES);
        put(
            &mut o,
            &[res.0 as f32, res.1 as f32, self.time, self.energy],
        );
        put(&mut o, &[self.glow, self.haze, self.warp, self.swell]);
        let mode = match self.mode {
            SceneMode::Live => 0.0,
            SceneMode::Synthetic => 1.0,
            SceneMode::NoAudio => 2.0,
        };
        put(&mut o, &[self.tilt, self.scroll[0], self.scroll[1], mode]);
        put(
            &mut o,
            &[
                self.rings[0][0],
                self.rings[0][1],
                self.rings[1][0],
                self.rings[1][1],
            ],
        );
        for (i, c) in self.palette.iter().enumerate() {
            put(
                &mut o,
                &[c[0], c[1], c[2], if i == 0 { self.dust } else { 0.0 }],
            );
        }
        for arr in [&self.bars, &self.peaks, &self.bright] {
            put(&mut o, &arr[..]);
        }
        debug_assert_eq!(o.len(), UNIFORM_BYTES);
        o
    }
}

pub struct SpectrumScene {
    inner: FullscreenRenderer,
}

impl SpectrumScene {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, scale: f32) -> Self {
        Self {
            inner: FullscreenRenderer::new(
                device,
                format,
                scale,
                "spectrum",
                include_str!("spectrum.wgsl"),
                UNIFORM_BYTES,
            ),
        }
    }

    /// Scene for `context::init_headless` devices: fixed RGBA8 target so tests can read pixels as RGBA.
    pub fn new_headless(device: &wgpu::Device) -> Self {
        Self::new(device, wgpu::TextureFormat::Rgba8Unorm, 1.0)
    }

    pub fn compile_ms(&self) -> f64 {
        self.inner.compile_ms
    }

    pub fn render_size(&self, surface: (u32, u32)) -> (u32, u32) {
        self.inner.render_size(surface)
    }

    pub fn render(&mut self, gpu: &mut Gpu, p: &SceneParams) -> Result<FrameStatus, String> {
        let res = self.inner.current_render_size(gpu);
        self.inner.render(gpu, &p.pack(res))
    }

    /// Renders the scene, then `post` records overlay passes on the surface in the same submission.
    pub fn render_with<F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView)>(
        &mut self,
        gpu: &mut Gpu,
        p: &SceneParams,
        post: F,
    ) -> Result<FrameStatus, String> {
        let res = self.inner.current_render_size(gpu);
        self.inner.render_with(gpu, &p.pack(res), post)
    }

    /// Headless render plus overlay hook (tests, `--scene-selftest`).
    pub fn render_offscreen_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: (u32, u32),
        p: &SceneParams,
        post: &dyn Fn(&mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) -> Result<Vec<u8>, String> {
        self.inner
            .render_offscreen_with(device, queue, res.0, res.1, &p.pack(res), post)
    }

    /// Headless render for tests and measurements: tightly packed 4-byte pixels (texture format order).
    pub fn render_offscreen(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: (u32, u32),
        p: &SceneParams,
    ) -> Result<Vec<u8>, String> {
        self.inner
            .render_offscreen(device, queue, res.0, res.1, &p.pack(res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::init_headless;

    fn f32_at(b: &[u8], off: usize) -> f32 {
        f32::from_ne_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }

    #[test]
    fn uniform_layout_matches_wgsl() {
        let mut p = SceneParams::default();
        p.bars[0] = 0.5;
        p.bars[95] = 0.75;
        p.peaks[1] = 0.25;
        p.glow = 0.7;
        p.mode = SceneMode::Live;
        p.dust = 0.25;
        let b = p.pack((1280, 720));
        assert_eq!(b.len(), UNIFORM_BYTES);
        assert_eq!(UNIFORM_BYTES, 1280);
        assert_eq!(f32_at(&b, 0), 1280.0);
        assert_eq!(f32_at(&b, 16), 0.7); // glow
        assert_eq!(f32_at(&b, 44), 0.0); // mode live
        assert_eq!(f32_at(&b, 76), 0.25); // dust phase (pal0.w)
        let bars = 128;
        assert_eq!(f32_at(&b, bars), 0.5);
        assert_eq!(f32_at(&b, bars + 95 * 4), 0.75);
        assert_eq!(f32_at(&b, bars + SLOTS * 4 + 4), 0.25); // peaks[1]
        let src = include_str!("spectrum.wgsl");
        assert!(src.contains("array<vec4<f32>, 24>") && src.contains("pal2: vec4<f32>"));
    }

    #[test]
    fn non_finite_values_never_reach_the_shader() {
        let mut p = SceneParams {
            glow: f32::NAN,
            ..Default::default()
        };
        p.bars[3] = f32::INFINITY;
        let b = p.pack((10, 10));
        assert!(
            b.chunks_exact(4)
                .all(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]).is_finite())
        );
    }

    /// Mean luminance of a tightly packed RGBA/BGRA8 image region (rows y0..y1, columns x0..x1).
    fn mean_lum(img: &[u8], w: u32, x0: u32, x1: u32, y0: u32, y1: u32) -> f32 {
        let mut s = 0.0f64;
        let mut n = 0.0f64;
        for y in y0..y1 {
            for x in x0..x1 {
                let i = ((y * w + x) * 4) as usize;
                s += (f64::from(img[i]) + f64::from(img[i + 1]) + f64::from(img[i + 2])) / 3.0;
                n += 1.0;
            }
        }
        (s / n / 255.0) as f32
    }

    fn headless() -> Option<(crate::context::Headless, SpectrumScene)> {
        match init_headless() {
            Ok(h) => {
                let s = SpectrumScene::new(&h.device, wgpu::TextureFormat::Rgba8Unorm, 1.0);
                Some((h, s))
            }
            Err(e) => {
                eprintln!("SKIPPED (no hardware GPU in this environment): {e}");
                None
            }
        }
    }

    #[test]
    fn renders_and_reacts_to_bar_heights_headless() {
        let Some((h, scene)) = headless() else { return };
        let res = (480u32, 270u32);
        let quiet = SceneParams {
            mode: SceneMode::Live,
            ..Default::default()
        };
        let mut loud = quiet.clone();
        for i in 0..SLOTS {
            loud.bars[i] = 0.8;
            loud.peaks[i] = 0.85;
            loud.bright[i] = 1.0;
        }
        let a = scene
            .render_offscreen(&h.device, &h.queue, res, &quiet)
            .unwrap();
        let b = scene
            .render_offscreen(&h.device, &h.queue, res, &loud)
            .unwrap();
        assert_eq!(a.len(), (res.0 * res.1 * 4) as usize);
        // The band of the screen above the horizon where tall bars live must get clearly brighter.
        let (x0, x1, y0, y1) = (40, 440, 90, 150);
        let (la, lb) = (
            mean_lum(&a, res.0, x0, x1, y0, y1),
            mean_lum(&b, res.0, x0, x1, y0, y1),
        );
        assert!(lb > la + 0.05, "bars region {la} -> {lb}");
        // A quiet scene is calm: dark overall (no full-screen brightness).
        assert!(mean_lum(&a, res.0, 0, res.0, 0, res.1) < 0.35);
    }

    /// The dust phase wraps at 1.0; every layer must be periodic over one phase unit or the sparkles re-roll.
    #[test]
    fn dust_phase_wrap_is_seamless_headless() {
        let Some((h, scene)) = headless() else { return };
        let res = (480u32, 270u32);
        let mk = |dust| SceneParams {
            mode: SceneMode::Live,
            dust,
            ..Default::default()
        };
        let a = scene
            .render_offscreen(&h.device, &h.queue, res, &mk(0.0))
            .unwrap();
        let b = scene
            .render_offscreen(&h.device, &h.queue, res, &mk(1.0))
            .unwrap();
        let max_diff = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
        assert!(max_diff <= 1, "dust phase 0 and 1 differ by {max_diff}");
        let c = scene
            .render_offscreen(&h.device, &h.queue, res, &mk(0.5))
            .unwrap();
        assert!(a != c, "dust phase must move the sparkles");
    }

    #[test]
    fn frequency_specific_response_is_spatial_headless() {
        let Some((h, scene)) = headless() else { return };
        let res = (480u32, 270u32);
        let mut centre = SceneParams {
            mode: SceneMode::Live,
            ..Default::default()
        };
        let mut edge = centre.clone();
        for i in 44..52 {
            centre.bars[i] = 0.85; // bass: centre slots
            centre.bright[i] = 1.0;
        }
        for i in (0..8).chain(88..96) {
            edge.bars[i] = 0.85; // treble: outer slots
            edge.bright[i] = 1.0;
        }
        let c = scene
            .render_offscreen(&h.device, &h.queue, res, &centre)
            .unwrap();
        let e = scene
            .render_offscreen(&h.device, &h.queue, res, &edge)
            .unwrap();
        let region = |img: &[u8], x0, x1| mean_lum(img, res.0, x0, x1, 90, 160);
        assert!(
            region(&c, 200, 280) > region(&c, 40, 120) + 0.04,
            "centre bars light the centre"
        );
        assert!(
            region(&e, 40, 120) > region(&e, 200, 280) + 0.03,
            "edge bars light the edges"
        );
    }

    #[test]
    fn composition_is_resolution_independent_headless() {
        let Some((h, scene)) = headless() else { return };
        let mut p = SceneParams {
            mode: SceneMode::Live,
            ..Default::default()
        };
        for i in 0..SLOTS {
            p.bars[i] = 0.2 + 0.6 * ((i as f32 * 0.37).sin() * 0.5 + 0.5);
            p.bright[i] = 0.7;
        }
        let small = scene
            .render_offscreen(&h.device, &h.queue, (320, 180), &p)
            .unwrap();
        let big = scene
            .render_offscreen(&h.device, &h.queue, (960, 540), &p)
            .unwrap();
        let (ls, lb) = (
            mean_lum(&small, 320, 0, 320, 0, 180),
            mean_lum(&big, 960, 0, 960, 0, 540),
        );
        assert!((ls - lb).abs() < 0.03, "mean luminance {ls} vs {lb}");
        // Ultrawide aspect still renders (no assumption of 16:9).
        let wide = scene
            .render_offscreen(&h.device, &h.queue, (840, 360), &p)
            .unwrap();
        assert!(mean_lum(&wide, 840, 0, 840, 0, 360) > 0.0);
    }
}
