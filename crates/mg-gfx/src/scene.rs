//! Scene 1 "Cinematic Spectrum": the GPU-side parameter block and renderer.
//!
//! `SceneParams` is plain data produced by the music-to-motion mapper (mg-engine). The shader draws
//! exactly what it is given; it never invents audio activity. Idle motion (camera drift, horizon
//! breathing) is time-based and uncorrelated with audio. Quality (render scale) only changes cost.

use crate::context::{FrameStatus, Gpu};
use crate::fluid::FluidSim;
use crate::fullscreen::FullscreenRenderer;

/// Mirrored bar slots across the screen (bass in the centre, treble at the edges).
pub const SLOTS: usize = 96;
const VEC4S: usize = SLOTS / 4;
/// Bytes of the packed uniform block (must equal the WGSL `U` struct size).
pub const UNIFORM_BYTES: usize = 16 + 16 + 16 + 16 + 64 + 3 * VEC4S * 16 + 64 + 32;

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
    /// Aurora: ribbon phases (cycles, integrated on the CPU) and band levels (bass to treble, 0..1).
    pub aurora_phase: [f32; 4],
    pub aurora_band: [f32; 4],
    /// Album Focus: artwork centre (uv x, uv y) and half size in screen heights; `focus_active` 0 when there is no artwork.
    pub focus: [f32; 4],
    pub focus_rot: f32,
    pub focus_active: f32,
    /// Aurora onset front origin (0..1 across the screen), changes with every onset.
    pub front_x0: f32,
    /// Orb: (flow phase, rotation phase) in cycles, integrated on the CPU, then the impact azimuths (0..1) of the two onset ripples whose (age, amplitude) live in `rings`.
    pub orb_flow: [f32; 4],
    /// Orb: smoothed bass, mid, treble (0..1) and the slow glow breath (0..1).
    pub orb_band: [f32; 4],
    /// Fluid: emitters, flow and memory for the dye simulation. Its own uniform block (`fluid::FluidParams::pack`),
    /// never part of `pack()`, so it cannot collide with the ex0..ex5 fields of the other scenes.
    pub fluid: crate::fluid::FluidParams,
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
            aurora_phase: [0.0; 4],
            aurora_band: [0.0; 4],
            focus: [0.5, 0.4, 0.16, 0.16],
            focus_rot: 0.0,
            focus_active: 0.0,
            front_x0: 0.5,
            orb_flow: [0.0; 4],
            orb_band: [0.0; 4],
            fluid: crate::fluid::FluidParams::default(),
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
        put(&mut o, &self.aurora_phase);
        put(&mut o, &self.aurora_band);
        put(&mut o, &self.focus);
        put(
            &mut o,
            &[self.focus_rot, self.front_x0, self.focus_active, 0.0],
        );
        put(&mut o, &self.orb_flow);
        put(&mut o, &self.orb_band);
        debug_assert_eq!(o.len(), UNIFORM_BYTES);
        o
    }
}

/// How far a scene has come. Technical verification (builds, tests, measured pixels and frame time) is not
/// artistic acceptance: `Accepted` is only ever set by the owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    Experimental,
    TechnicallyVerified,
    EvaluationPending,
    Accepted,
}

/// The visual styles. Each has its own composition and animation algorithm; all read the same `SceneParams`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    /// Mirrored needle spectrum over a water horizon.
    Spectrum,
    /// Layered light ribbons over a ridge.
    Aurora,
    /// Atmosphere radiating from the album artwork.
    Focus,
    /// One softly lit, audio-displaced sphere.
    Orb,
    /// Liquid dye field: a feedback simulation (`fluid.rs`) drawn as glossy ink.
    Fluid,
}

impl SceneKind {
    /// Every scene, in pipeline order (`index()` is the pipeline index: never reorder).
    pub const ALL: [SceneKind; 5] = [
        SceneKind::Spectrum,
        SceneKind::Aurora,
        SceneKind::Focus,
        SceneKind::Orb,
        SceneKind::Fluid,
    ];

    /// Lifecycle stage: the single source of truth for what the scene selector offers.
    pub fn lifecycle(self) -> Lifecycle {
        match self {
            SceneKind::Spectrum | SceneKind::Focus => Lifecycle::TechnicallyVerified,
            SceneKind::Aurora | SceneKind::Fluid => Lifecycle::Experimental,
            SceneKind::Orb => Lifecycle::TechnicallyVerified,
        }
    }

    /// Offered by the scene selector (keys, cycling, `--auto-scenes`): everything past `Experimental`.
    pub fn in_selector(self) -> bool {
        self.lifecycle() != Lifecycle::Experimental
    }

    /// Selector scenes in stable declared order.
    pub fn selector() -> impl Iterator<Item = SceneKind> {
        Self::ALL.into_iter().filter(|k| k.in_selector())
    }

    pub fn index(self) -> usize {
        self as usize
    }

    /// Next selector scene (an experimental scene continues at the first selector scene).
    pub fn next(self) -> Self {
        let v: Vec<_> = Self::selector().collect();
        v[v.iter()
            .position(|k| *k == self)
            .map_or(0, |i| (i + 1) % v.len())]
    }

    pub fn prev(self) -> Self {
        let v: Vec<_> = Self::selector().collect();
        let n = v.len();
        v[v.iter()
            .position(|k| *k == self)
            .map_or(n - 1, |i| (i + n - 1) % n)]
    }

    pub fn name(self) -> &'static str {
        match self {
            SceneKind::Spectrum => "spectrum",
            SceneKind::Aurora => "aurora",
            SceneKind::Focus => "focus",
            SceneKind::Orb => "orb",
            SceneKind::Fluid => "fluid",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == s)
    }
}

const SPECTRUM_WGSL: &str = concat!(include_str!("common.wgsl"), include_str!("spectrum.wgsl"));
const AURORA_WGSL: &str = concat!(include_str!("common.wgsl"), include_str!("aurora.wgsl"));
const FOCUS_WGSL: &str = concat!(include_str!("common.wgsl"), include_str!("focus.wgsl"));
const ORB_WGSL: &str = concat!(include_str!("common.wgsl"), include_str!("orb.wgsl"));
const FLUID_WGSL: &str = concat!(include_str!("common.wgsl"), include_str!("fluid.wgsl"));

/// All scene pipelines, created up front (a scene switch never compiles), sharing one uniform block.
/// Fluid additionally owns a feedback simulation (`fluid.rs`) that runs only while Fluid is on screen.
pub struct SceneSet {
    inner: FullscreenRenderer,
    fluid: FluidSim,
}

impl SceneSet {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, scale: f32) -> Self {
        let mut inner = FullscreenRenderer::new_multi(
            device,
            format,
            scale,
            "scenes",
            &[SPECTRUM_WGSL, AURORA_WGSL, FOCUS_WGSL, ORB_WGSL],
            UNIFORM_BYTES,
        );
        let fluid = FluidSim::new(device, FLUID_WGSL);
        let i = inner.add_pipeline(device, "fluid", FLUID_WGSL, fluid.bind_group_layout());
        assert_eq!(i, SceneKind::Fluid.index(), "pipeline order");
        Self { inner, fluid }
    }

    /// Fluid quality tier 0..=3 (grid 256x144 .. 768x432 at 16:9); a change restarts the field.
    pub fn set_fluid_tier(&mut self, tier: u8) {
        self.fluid.set_tier(tier);
    }

    /// Draws `scene` (or a crossfade `scene -> to` at blend `t`), then `post` records overlays.
    /// The Fluid simulation advances (fixed steps, from `p.time`) only while Fluid is one of the two scenes;
    /// otherwise its textures are released, so the next entry starts from a clear field.
    pub fn render_scenes<F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView)>(
        &mut self,
        gpu: &mut Gpu,
        p: &SceneParams,
        scene: SceneKind,
        xfade: Option<(SceneKind, f32)>,
        post: F,
    ) -> Result<FrameStatus, String> {
        let res = self.inner.current_render_size(gpu);
        if scene == SceneKind::Fluid || xfade.is_some_and(|(k, _)| k == SceneKind::Fluid) {
            self.fluid
                .advance(&gpu.device, &gpu.queue, res, &p.fluid, p.time);
        } else {
            self.fluid.release();
        }
        self.inner.render_scenes(
            gpu,
            &p.pack(res),
            scene.index(),
            xfade.map(|(k, t)| (k.index(), t)),
            Some(self.fluid.render_group()),
            post,
        )
    }

    /// Headless counterpart of the live loop's Fluid step: advances the simulation to `p.time` at render size
    /// `res` (fixed steps, at most three per call). Returns the steps run.
    pub fn fluid_advance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: (u32, u32),
        p: &SceneParams,
    ) -> u32 {
        self.fluid.advance(device, queue, res, &p.fluid, p.time)
    }

    /// Runs exactly `n` fixed Fluid steps with `p` (soak tests: no clock, no rendering).
    pub fn fluid_step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: (u32, u32),
        p: &SceneParams,
        n: u32,
    ) {
        self.fluid.step(device, queue, res, &p.fluid, n);
    }

    /// Drops the Fluid textures (what the live loop does when Fluid leaves the screen).
    pub fn fluid_release(&mut self) {
        self.fluid.release();
    }

    /// Current Fluid grid, None while released.
    pub fn fluid_grid(&self) -> Option<(u32, u32)> {
        self.fluid.grid()
    }

    /// Diagnostic (headless): GPU ms per Fluid sim step and per Fluid scene draw at `res` (no readback), with the
    /// field as it is now. The grid follows `res` and the current tier.
    pub fn fluid_gpu_ms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: (u32, u32),
        p: &SceneParams,
        reps: u32,
    ) -> Result<(f64, f64), String> {
        let wait = || {
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .map_err(|e| format!("timing wait failed: {e}"))
        };
        self.fluid.step(device, queue, res, &p.fluid, 1);
        wait()?;
        let t0 = std::time::Instant::now();
        self.fluid.step(device, queue, res, &p.fluid, reps.max(1));
        wait()?;
        let sim = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(reps.max(1));
        let draw = self.inner.time_offscreen(
            device,
            queue,
            res,
            &p.pack(res),
            SceneKind::Fluid.index(),
            Some(self.fluid.render_group()),
            reps,
        )?;
        Ok((sim, draw))
    }

    /// Current dye texture as RGBA f32 (x/y/z dye, w heat); tests and diagnostics.
    pub fn fluid_dye(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(u32, u32, Vec<f32>), String> {
        self.fluid.read_dye(device, queue)
    }

    /// Headless render of any scene or crossfade, then `post`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_scenes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: (u32, u32),
        p: &SceneParams,
        scene: SceneKind,
        xfade: Option<(SceneKind, f32)>,
        post: &dyn Fn(&mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) -> Result<Vec<u8>, String> {
        self.inner.render_offscreen_scenes(
            device,
            queue,
            res.0,
            res.1,
            &p.pack(res),
            scene.index(),
            xfade.map(|(k, t)| (k.index(), t)),
            Some(self.fluid.render_group()),
            post,
        )
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
        assert_eq!(UNIFORM_BYTES, 1376);
        assert_eq!(f32_at(&b, 0), 1280.0);
        assert_eq!(f32_at(&b, 16), 0.7); // glow
        assert_eq!(f32_at(&b, 44), 0.0); // mode live
        assert_eq!(f32_at(&b, 76), 0.25); // dust phase (pal0.w)
        let bars = 128;
        assert_eq!(f32_at(&b, bars), 0.5);
        assert_eq!(f32_at(&b, bars + 95 * 4), 0.75);
        assert_eq!(f32_at(&b, bars + SLOTS * 4 + 4), 0.25); // peaks[1]
        let src = include_str!("common.wgsl");
        assert!(
            src.contains("array<vec4<f32>, 24>")
                && src.contains("pal3: vec4<f32>")
                && src.contains("ex3: vec4<f32>")
                && src.contains("ex5: vec4<f32>")
        );
        // extras sit after the three slot arrays: 128 (bars) + 3 * 96 * 4 = 1280
        assert_eq!(f32_at(&b, 1280), 0.0);
        let q = SceneParams {
            orb_flow: [1.0, 2.0, 3.0, 4.0],
            orb_band: [5.0, 6.0, 7.0, 8.0],
            ..Default::default()
        };
        let b = q.pack((1, 1));
        assert_eq!((f32_at(&b, 1344), f32_at(&b, 1356)), (1.0, 4.0)); // ex4
        assert_eq!((f32_at(&b, 1360), f32_at(&b, 1372)), (5.0, 8.0)); // ex5, last float
        assert_eq!(b.len(), 1376);
    }

    #[test]
    fn selector_order_skips_experimental_scenes_and_cycles_stably() {
        use SceneKind::*;
        assert_eq!(
            SceneKind::selector().collect::<Vec<_>>(),
            [Spectrum, Focus, Orb]
        );
        assert_eq!(Aurora.lifecycle(), Lifecycle::Experimental);
        assert!(!Aurora.in_selector());
        assert_eq!(Fluid.lifecycle(), Lifecycle::Experimental);
        assert!(!Fluid.in_selector());
        assert_eq!((Fluid.next(), Fluid.prev()), (Spectrum, Orb));
        assert_eq!(SceneKind::parse("fluid"), Some(Fluid));
        for k in SceneKind::ALL {
            assert!(k.next() != Fluid && k.prev() != Fluid, "{k:?}");
        }
        assert!(
            SceneKind::ALL
                .iter()
                .all(|k| k.lifecycle() != Lifecycle::Accepted),
            "only the owner accepts"
        );
        for k in SceneKind::ALL {
            assert!(k.next() != Aurora && k.prev() != Aurora, "{k:?}");
        }
        assert_eq!(
            (Spectrum.next(), Focus.next(), Orb.next()),
            (Focus, Orb, Spectrum)
        );
        assert_eq!(
            (Spectrum.prev(), Focus.prev(), Orb.prev()),
            (Orb, Spectrum, Focus)
        );
        assert_eq!((Aurora.next(), Aurora.prev()), (Spectrum, Orb));
        assert_eq!(SceneKind::parse("orb"), Some(Orb));
        assert_eq!(SceneKind::parse("aurora"), Some(Aurora));
        for (i, k) in SceneKind::ALL.iter().enumerate() {
            assert_eq!(k.index(), i, "pipeline order");
        }
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

    fn headless() -> Option<(crate::context::Headless, SceneSet)> {
        match init_headless() {
            Ok(h) => {
                let s = SceneSet::new(&h.device, wgpu::TextureFormat::Rgba8Unorm, 1.0);
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

    #[test]
    fn every_scene_renders_and_a_crossfade_blends_between_two_scenes_headless() {
        let Some((h, scene)) = headless() else { return };
        let res = (320u32, 180u32);
        let mut p = SceneParams {
            mode: SceneMode::Live,
            energy: 0.6,
            glow: 0.6,
            orb_band: [0.6, 0.3, 0.5, 0.4],
            ..Default::default()
        };
        for i in 0..SLOTS {
            p.bars[i] = 0.2 + 0.6 * ((i as f32 * 0.37).sin() * 0.5 + 0.5);
            p.bright[i] = 0.7;
        }
        p.aurora_band = [0.5, 0.6, 0.4, 0.3];
        p.aurora_phase = [0.1, 0.4, 0.7, 0.9];
        let render = |k: SceneKind, x: Option<(SceneKind, f32)>| {
            scene
                .render_offscreen_scenes(&h.device, &h.queue, res, &p, k, x, &|_, _| {})
                .unwrap()
        };
        let a = render(SceneKind::Spectrum, None);
        let b = render(SceneKind::Aurora, None);
        let c = render(SceneKind::Focus, None);
        assert!(a != b && b != c && a != c, "scenes must look different");
        let diff = |x: &[u8], y: &[u8]| {
            x.iter()
                .zip(y)
                .map(|(p, q)| p.abs_diff(*q) as u64)
                .sum::<u64>() as f64
                / x.len() as f64
        };
        assert!(
            diff(&a, &b) > 3.0 && diff(&b, &c) > 3.0 && diff(&a, &c) > 3.0,
            "scenes differ clearly"
        );
        let x0 = render(SceneKind::Spectrum, Some((SceneKind::Aurora, 0.0)));
        let x1 = render(SceneKind::Spectrum, Some((SceneKind::Aurora, 1.0)));
        let xm = render(SceneKind::Spectrum, Some((SceneKind::Aurora, 0.5)));
        let maxd = |x: &[u8], y: &[u8]| x.iter().zip(y).map(|(p, q)| p.abs_diff(*q)).max().unwrap();
        assert!(
            maxd(&x0, &a) <= 2,
            "t=0 shows the first scene ({})",
            maxd(&x0, &a)
        );
        assert!(
            maxd(&x1, &b) <= 2,
            "t=1 shows the second scene ({})",
            maxd(&x1, &b)
        );
        let mid: Vec<u8> = a
            .iter()
            .zip(&b)
            .map(|(p, q)| ((u16::from(*p) + u16::from(*q)) / 2) as u8)
            .collect();
        assert!(
            maxd(&xm, &mid) <= 3,
            "t=0.5 is the average ({})",
            maxd(&xm, &mid)
        );
        // same scene on both sides is just that scene
        assert_eq!(render(SceneKind::Aurora, Some((SceneKind::Aurora, 0.5))), b);
        // a fourth scene crosses over like the others
        let o = render(SceneKind::Orb, None);
        assert!(
            diff(&o, &a) > 3.0 && diff(&o, &b) > 3.0 && diff(&o, &c) > 3.0,
            "orb differs"
        );
        let ox = render(SceneKind::Orb, Some((SceneKind::Aurora, 1.0)));
        assert!(maxd(&ox, &b) <= 2, "orb -> aurora t=1 shows aurora");
        orb_is_a_lit_sphere_that_answers_to_bass_treble_and_onsets(&h, &scene);
        fluid_renders_and_crossfades(&h, p);
    }

    /// Fluid joins the crossfade like the others: released it shows its dark base, a stepped field differs from
    /// every other scene, and a crossfade into it ends on exactly the Fluid frame.
    fn fluid_renders_and_crossfades(h: &crate::context::Headless, mut p: SceneParams) {
        let res = (320u32, 180u32);
        let mut scene = SceneSet::new(&h.device, wgpu::TextureFormat::Rgba8Unorm, 1.0);
        let render = |s: &SceneSet, p: &SceneParams, k: SceneKind, x: Option<(SceneKind, f32)>| {
            s.render_offscreen_scenes(&h.device, &h.queue, res, p, k, x, &|_, _| {})
                .unwrap()
        };
        let idle = render(&scene, &p, SceneKind::Fluid, None);
        let rgb = mean_lum(&idle, res.0, 0, res.0, 0, res.1);
        assert!(rgb > 0.005 && rgb < 0.3, "dark base {rgb}");
        for e in &mut p.fluid.emit {
            e[2] = 1.0;
        }
        p.fluid.flow[0] = 0.08;
        scene.fluid_step(&h.device, &h.queue, res, &p, 240);
        assert_eq!(scene.fluid_grid(), Some((384, 216)), "default tier T1");
        let f = render(&scene, &p, SceneKind::Fluid, None);
        assert!(mad(&f, &idle) > 1.0, "dye shows: {}", mad(&f, &idle));
        for k in [
            SceneKind::Spectrum,
            SceneKind::Aurora,
            SceneKind::Focus,
            SceneKind::Orb,
        ] {
            assert!(
                mad(&f, &render(&scene, &p, k, None)) > 3.0,
                "{k:?} vs fluid"
            );
        }
        let maxd = |x: &[u8], y: &[u8]| x.iter().zip(y).map(|(p, q)| p.abs_diff(*q)).max().unwrap();
        let o = render(&scene, &p, SceneKind::Orb, None);
        assert!(
            maxd(
                &render(&scene, &p, SceneKind::Orb, Some((SceneKind::Fluid, 1.0))),
                &f
            ) <= 2
        );
        assert!(
            maxd(
                &render(&scene, &p, SceneKind::Fluid, Some((SceneKind::Orb, 1.0))),
                &o
            ) <= 2
        );
        let m = render(&scene, &p, SceneKind::Orb, Some((SceneKind::Fluid, 0.5)));
        assert!(
            mad(&m, &f) > 0.5 && mad(&m, &o) > 0.5,
            "half way is a blend"
        );
        scene.fluid_release();
        assert_eq!(scene.fluid_grid(), None);
        assert!(
            maxd(&render(&scene, &p, SceneKind::Fluid, None), &idle) <= 1,
            "released = clear"
        );
    }

    #[test]
    fn aurora_phase_wrap_is_seamless_headless() {
        let Some((h, scene)) = headless() else { return };
        let res = (320u32, 180u32);
        let mk = |ph: f32| SceneParams {
            mode: SceneMode::Live,
            energy: 0.6,
            glow: 0.6,
            aurora_band: [0.5, 0.6, 0.4, 0.3],
            aurora_phase: [ph; 4],
            ..Default::default()
        };
        let render = |p: &SceneParams| {
            scene
                .render_offscreen_scenes(
                    &h.device,
                    &h.queue,
                    res,
                    p,
                    SceneKind::Aurora,
                    None,
                    &|_, _| {},
                )
                .unwrap()
        };
        let a = render(&mk(0.0));
        let b = render(&mk(64.0));
        let max = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
        assert!(max <= 3, "phase 0 and 64 cycles differ by {max} levels");
        assert!(a != render(&mk(0.3)), "the phase must move the ribbons");
    }
    /// Mean absolute per-byte difference between two frames (0..255).
    fn mad(x: &[u8], y: &[u8]) -> f64 {
        x.iter()
            .zip(y)
            .map(|(p, q)| p.abs_diff(*q) as u64)
            .sum::<u64>() as f64
            / x.len() as f64
    }

    /// Orb checks share the crossfade test's device: parallel headless devices can run out of GPU memory.
    fn orb_is_a_lit_sphere_that_answers_to_bass_treble_and_onsets(
        h: &crate::context::Headless,
        scene: &SceneSet,
    ) {
        let res = (320u32, 180u32);
        let base = SceneParams {
            mode: SceneMode::Live,
            energy: 0.1,
            glow: 0.25,
            haze: 0.3,
            ..Default::default()
        };
        let loud = |band: [f32; 4]| SceneParams {
            energy: 0.7,
            glow: 0.7,
            orb_band: band,
            orb_flow: [3.3, 1.7, 0.0, 0.0],
            ..base.clone()
        };
        let render = |p: &SceneParams| {
            let img = scene
                .render_offscreen_scenes(
                    &h.device,
                    &h.queue,
                    res,
                    p,
                    SceneKind::Orb,
                    None,
                    &|_, _| {},
                )
                .unwrap();
            assert_eq!(img.len(), (res.0 * res.1 * 4) as usize);
            img
        };
        let silence = render(&base);
        let bass = render(&loud([0.9, 0.2, 0.1, 0.8]));
        let treble = render(&loud([0.1, 0.2, 0.9, 0.1]));
        let mut kick = loud([0.5, 0.2, 0.1, 0.4]);
        kick.rings = [[0.25, 0.15], [100.0, 0.0]];
        kick.orb_flow[2] = 0.3;
        let kick = render(&kick);
        // the orb region (centre 0.57,0.44, radius 0.22 h = 40 px) is brighter than the corners
        let (cx, cy) = (182u32, 79u32);
        for img in [&silence, &bass, &treble, &kick] {
            let orb = mean_lum(img, res.0, cx - 20, cx + 20, cy - 20, cy + 20);
            let corners = (mean_lum(img, res.0, 0, 30, 0, 30)
                + mean_lum(img, res.0, 290, 320, 150, 180))
                / 2.0;
            assert!(
                orb > 0.04 && orb > corners + 0.03,
                "orb {orb} corners {corners}"
            );
        }
        // silence is dim; bass lifts and reshapes the sphere; treble and a ripple each change the frame clearly
        let lum = |img: &[u8]| mean_lum(img, res.0, cx - 40, cx + 40, cy - 40, cy + 40);
        assert!(
            lum(&bass) > lum(&silence) * 1.15,
            "silence {} bass {}",
            lum(&silence),
            lum(&bass)
        );
        assert!(
            mad(&silence, &bass) > 1.0,
            "silence vs bass {}",
            mad(&silence, &bass)
        );
        assert!(
            mad(&silence, &treble) > 1.0,
            "silence vs treble {}",
            mad(&silence, &treble)
        );
        assert!(
            mad(&bass, &treble) > 0.5,
            "bass vs treble {}",
            mad(&bass, &treble)
        );
        let mut no_kick = loud([0.5, 0.2, 0.1, 0.4]);
        no_kick.orb_flow[2] = 0.3;
        assert!(
            mad(&render(&no_kick), &kick) > 0.1,
            "a ripple must be visible"
        );
        // no full-screen flash: the ripple frame stays close to the same frame without it
        assert!(mad(&render(&no_kick), &kick) < 6.0);
    }
}
