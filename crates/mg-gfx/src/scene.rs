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

/// Scene-transition look (all zero = the scene exactly as drawn without a transition). Veils run 0 visible .. 1
/// hidden: per layer (atmosphere, geometry, particles) for the layered scenes, and a global one (last) for the
/// scenes without separable layers. Packed into the free `ex2` and `ex3.xzw` slots; see `common.wgsl`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Fx {
    pub veil: [f32; 4],
    /// Relative luminance of the palette bridge the veil fades towards.
    pub bridge_luma: f32,
    /// Bridge breathing amplitude (lightness fraction) and iris strength (0 or 1).
    pub breathing: f32,
    pub iris: f32,
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
    /// Aurora onset front origin (0..1 across the screen), changes with every onset.
    pub front_x0: f32,
    /// Orb: (flow phase, rotation phase) in cycles, integrated on the CPU, then the impact azimuths (0..1) of the two onset ripples whose (age, amplitude) live in `rings`.
    pub orb_flow: [f32; 4],
    /// Orb: smoothed bass, mid, treble (0..1) and the slow glow breath (0..1).
    pub orb_band: [f32; 4],
    /// Fluid: emitters, flow and memory for the dye simulation. Its own uniform block (`fluid::FluidParams::pack`),
    /// never part of `pack()`, so it cannot collide with the ex0..ex5 fields of the other scenes.
    pub fluid: crate::fluid::FluidParams,
    /// Album Focus environment: its own block (`FocusParams::pack`, `@group(1)` of focus.wgsl), never part of `pack()`.
    pub album: FocusParams,
    pub mode: SceneMode,
    pub fx: Fx,
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
            front_x0: 0.5,
            orb_flow: [0.0; 4],
            orb_band: [0.0; 4],
            fluid: crate::fluid::FluidParams::default(),
            album: FocusParams::default(),
            mode: SceneMode::NoAudio,
            fx: Fx::default(),
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
        // ex2 = transition veils, ex3.xzw = bridge luma, breathing, iris (Album Focus reads its own block)
        put(&mut o, &self.fx.veil);
        put(
            &mut o,
            &[
                self.fx.bridge_luma,
                self.front_x0,
                self.fx.breathing,
                self.fx.iris,
            ],
        );
        put(&mut o, &self.orb_flow);
        put(&mut o, &self.orb_band);
        debug_assert_eq!(o.len(), UNIFORM_BYTES);
        o
    }
}

/// Bytes of `FocusParams::pack` (the WGSL `FU` struct in focus.wgsl).
pub const FOCUS_BYTES: usize = 10 * 16;
/// Edge-colour grid side (the cover's 8x8 box mip) of `ArtLight`.
pub const ART_GRID: usize = 8;
const ART_SLOT_VEC4S: usize = ART_GRID * ART_GRID + 1;

/// Album Focus environment ("a room lit by the album art", focus.wgsl). Geometry is in screen heights from the
/// top-left corner (independent of the render scale); colours come from the cover slots (`ArtLight`) and the
/// palette in the shared block. The mapper fills the music fields, the runner the geometry and the weights.
#[derive(Clone, Debug, PartialEq)]
pub struct FocusParams {
    /// Light source: centre x, centre y, half width, half height (the cover, or the empty cover slot of the
    /// text-led composition). A zero size means a soft default source mid-screen: `pack` then replaces the centre
    /// by (0.5 x aspect, 0.5), so the shader never sees a point at the top-left corner.
    pub rect: [f32; 4],
    /// Spill intensity I and radius scale k.
    pub spill: f32,
    pub radius: f32,
    /// Weights of the current and the previous cover's colours (their drawn alphas, the previous one under the
    /// current); the remainder is palette light.
    pub art_w: [f32; 2],
    /// Haze flow phase (noise units, integrated on the CPU, never reset), warp W, dust twinkle depth 0..1.
    pub flow: [f32; 2],
    pub warp: f32,
    pub twinkle: f32,
    /// Haze hue offset (radians, OKLCH).
    pub hue: f32,
    /// Control-bar visibility 0..1 (floor dim: environment x0.8 below 0.80 H).
    pub bar: f32,
    /// Text ink rectangles (x0, y0, x1, y1) whose environment luminance is capped at `ink_cap`, with strength
    /// `ink_on` 0..1 (0 = unused).
    pub ink: [[f32; 4]; 4],
    pub ink_cap: [f32; 4],
    pub ink_on: [f32; 4],
}

impl Default for FocusParams {
    fn default() -> Self {
        Self {
            rect: [0.0; 4],
            spill: 0.5,
            radius: 1.0,
            art_w: [0.0; 2],
            flow: [0.0; 2],
            warp: 3.0,
            twinkle: 0.0,
            hue: 0.0,
            bar: 0.0,
            ink: [[0.0; 4]; 4],
            ink_cap: [1.0; 4],
            ink_on: [0.0; 4],
        }
    }
}

impl FocusParams {
    /// Packs the WGSL `FU` layout for a render target of `res` pixels (finite values only), without allocating.
    /// `res` sets the dust cell (about 147 motes at any aspect) and the mid-screen centre of a zero-size `rect`.
    pub fn pack(&self, res: (u32, u32)) -> [u8; FOCUS_BYTES] {
        let aspect = res.0.max(1) as f32 / res.1.max(1) as f32;
        let inv_cell = 1.0 / (0.11 * (aspect / (16.0 / 9.0)).sqrt());
        let mut rect = self.rect;
        if !(rect[2] > 0.0 || rect[3] > 0.0) {
            (rect[0], rect[1]) = (0.5 * aspect, 0.5);
        }
        let mut vals = [0.0f32; FOCUS_BYTES / 4];
        let mut n = 0;
        let mut put = |v: &[f32]| {
            for x in v {
                // Never let a NaN/inf reach the shader.
                vals[n] = if x.is_finite() { *x } else { 0.0 };
                n += 1;
            }
        };
        put(&rect);
        put(&[self.spill, self.radius, self.art_w[0], self.art_w[1]]);
        put(&[self.flow[0], self.flow[1], self.warp, self.twinkle]);
        put(&[self.hue, self.bar, 0.0, inv_cell]);
        for r in &self.ink {
            put(r);
        }
        put(&self.ink_cap);
        put(&self.ink_on);
        debug_assert_eq!(n, vals.len());
        let mut o = [0u8; FOCUS_BYTES];
        for (c, v) in o.chunks_exact_mut(4).zip(vals) {
            c.copy_from_slice(&v.to_ne_bytes());
        }
        o
    }
}

/// The colours a cover throws into the room (linear RGB), computed once per cover off the render thread: the
/// 8x8 box-filtered grid (row-major, sampled bilinearly at the nearest point of the cover) and the mean, with the
/// mean's relative luminance Y.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtLight {
    pub grid: [[f32; 3]; ART_GRID * ART_GRID],
    pub mean: [f32; 3],
    pub mean_y: f32,
}

/// Focus's `@group(1)`: the per-frame block and the two cover-colour slots (current, previous), which are
/// written only when a cover changes (mirroring the overlay's `set_art` / `shift_art` / `clear_prev_art`).
/// Size of the low-resolution room-light texture (`fs_room`); keep equal to ROOM_W / ROOM_H in focus.wgsl. About
/// 1/16 of the pixels of a 1080p frame: the room (gradient, spill, haze) has no detail finer than ~0.15 H.
pub const ROOM_SIZE: (u32, u32) = (480, 270);
const ROOM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct FocusRes {
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    art: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Same buffers with a 1x1 stand-in for the room texture, bound while the room pass itself draws (and by the
    /// scene while the room is released).
    pre_group: wgpu::BindGroup,
    /// The room texture and the group that samples it; None while Focus is off screen (about 1 MB released).
    room: Option<(wgpu::TextureView, wgpu::BindGroup)>,
    slots: [Option<ArtLight>; 2],
}

fn room_texture(
    device: &wgpu::Device,
    label: &str,
    (w, h): (u32, u32),
    usage: wgpu::TextureUsages,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ROOM_FORMAT,
            usage,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl FocusRes {
    fn new(device: &wgpu::Device) -> Self {
        let entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("focus"),
            entries: &[
                entry(0),
                entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let buf = |label, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniform = buf("focus", FOCUS_BYTES);
        let art = buf("focus-art", 2 * ART_SLOT_VEC4S * 16);
        let stand_in = room_texture(
            device,
            "focus-room-1x1",
            (1, 1),
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("focus-room"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut me = Self {
            pre_group: Self::bind(
                device,
                &bgl,
                &uniform,
                &art,
                &sampler,
                "focus-room-pass",
                &stand_in,
            ),
            bgl,
            uniform,
            art,
            sampler,
            room: None,
            slots: [None, None],
        };
        me.acquire(device);
        me
    }

    fn bind(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        art: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        label: &str,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: art.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Creates the room texture when it is not there; returns its view when it was just created.
    fn acquire(&mut self, device: &wgpu::Device) -> Option<wgpu::TextureView> {
        if self.room.is_some() {
            return None;
        }
        let view = room_texture(
            device,
            "focus-room",
            ROOM_SIZE,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let group = Self::bind(
            device,
            &self.bgl,
            &self.uniform,
            &self.art,
            &self.sampler,
            "focus",
            &view,
        );
        self.room = Some((view.clone(), group));
        Some(view)
    }

    /// The group the Focus scene draws with: the room texture, or the stand-in while released.
    fn group(&self) -> &wgpu::BindGroup {
        self.room.as_ref().map_or(&self.pre_group, |r| &r.1)
    }

    /// Uploads both slots (2 KB, only when a cover changes). An empty slot is black; its weight is 0 anyway.
    fn write_art(&self, queue: &wgpu::Queue) {
        let mut o = Vec::with_capacity(2 * ART_SLOT_VEC4S * 16);
        for s in &self.slots {
            match s {
                Some(a) => {
                    for c in &a.grid {
                        put(&mut o, &[c[0], c[1], c[2], 0.0]);
                    }
                    put(&mut o, &[a.mean[0], a.mean[1], a.mean[2], a.mean_y]);
                }
                None => put(&mut o, &[0.0; ART_SLOT_VEC4S * 4]),
            }
        }
        queue.write_buffer(&self.art, 0, &o);
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

/// Q-010 placeholder (OWNER-QUESTIONS.md): whether Experimental scenes appear in the scene panel at all.
pub const EXPERIMENTAL_SELECTABLE: bool = true;

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

    /// Display name for the scene panel (`name()` stays the CLI id).
    pub fn label(self) -> &'static str {
        match self {
            SceneKind::Spectrum => "Spectrum",
            SceneKind::Aurora => "Aurora",
            SceneKind::Focus => "Focus",
            SceneKind::Orb => "Orb",
            SceneKind::Fluid => "Fluid",
        }
    }

    /// The panel's main group: the selector scenes in declared order.
    pub fn panel_main() -> impl Iterator<Item = SceneKind> {
        Self::selector()
    }

    /// The panel's Experimental group; empty unless `EXPERIMENTAL_SELECTABLE`. Mouse and Enter only:
    /// keys, cycling and `--auto-scenes` never reach these.
    pub fn panel_experimental() -> impl Iterator<Item = SceneKind> {
        Self::ALL
            .into_iter()
            .filter(|k| EXPERIMENTAL_SELECTABLE && k.lifecycle() == Lifecycle::Experimental)
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
    focus: FocusRes,
}

impl SceneSet {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, scale: f32) -> Self {
        let mut inner = FullscreenRenderer::new_multi(
            device,
            format,
            scale,
            "scenes",
            &[SPECTRUM_WGSL, AURORA_WGSL],
            UNIFORM_BYTES,
        );
        let focus = FocusRes::new(device);
        let fluid = FluidSim::new(device, FLUID_WGSL);
        let focus_pipe = inner.add_pipeline(device, "focus", FOCUS_WGSL, Some(&focus.bgl));
        let order = [
            focus_pipe,
            inner.add_pipeline(device, "orb", ORB_WGSL, None),
            inner.add_pipeline(device, "fluid", FLUID_WGSL, Some(fluid.bind_group_layout())),
        ];
        assert_eq!(
            order,
            [
                SceneKind::Focus.index(),
                SceneKind::Orb.index(),
                SceneKind::Fluid.index()
            ],
            "pipeline order"
        );
        // the room pass draws into the small texture the Focus pass samples; it runs only when Focus is drawn
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("focus-room"),
            source: wgpu::ShaderSource::Wgsl(FOCUS_WGSL.into()),
        });
        let room_pipe = crate::fullscreen::make_pipeline(
            device,
            &module,
            &[Some(inner.shared_layout()), Some(&focus.bgl)],
            "focus-room",
            ROOM_FORMAT,
            "fs_room",
        );
        inner.set_prepass(
            SceneKind::Focus.index(),
            crate::fullscreen::PrePass {
                pipeline: room_pipe,
                target: focus.room.as_ref().map(|r| r.0.clone()),
                group1: focus.pre_group.clone(),
            },
        );
        // new buffers are zero-filled: both cover slots start empty (weight 0)
        Self {
            inner,
            fluid,
            focus,
        }
    }

    /// The `@group(1)` bind groups by scene index, after writing Focus's per-frame block.
    fn groups<'a>(
        focus: &'a FocusRes,
        fluid: &'a FluidSim,
        queue: &wgpu::Queue,
        p: &SceneParams,
        res: (u32, u32),
    ) -> [(usize, &'a wgpu::BindGroup); 2] {
        queue.write_buffer(&focus.uniform, 0, &p.album.pack(res));
        [
            (SceneKind::Focus.index(), focus.group()),
            (SceneKind::Fluid.index(), fluid.render_group()),
        ]
    }

    /// Album Focus: the current cover's colours change (None = no cover). Mirrors `Overlay::set_art`/`clear_art`.
    pub fn set_focus_art(&mut self, queue: &wgpu::Queue, light: Option<ArtLight>) {
        self.focus.slots[0] = light;
        self.focus.write_art(queue);
    }

    /// Album Focus: the current cover becomes the previous one (crossfade source). Mirrors `Overlay::shift_art`.
    pub fn shift_focus_art(&mut self, queue: &wgpu::Queue) {
        self.focus.slots[1] = self.focus.slots[0].take();
        self.focus.write_art(queue);
    }

    /// Album Focus: the previous cover is gone. Mirrors `Overlay::clear_prev_art`.
    pub fn clear_prev_focus_art(&mut self, queue: &wgpu::Queue) {
        self.focus.slots[1] = None;
        self.focus.write_art(queue);
    }

    /// Which cover slots hold colours (current, previous); tests and diagnostics.
    pub fn focus_art_slots(&self) -> [bool; 2] {
        [self.focus.slots[0].is_some(), self.focus.slots[1].is_some()]
    }

    /// Diagnostic (headless): GPU ms per Focus scene draw at `res`, submit-to-completion with no readback.
    pub fn focus_gpu_ms(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: (u32, u32),
        p: &SceneParams,
        reps: u32,
    ) -> Result<f64, String> {
        let groups = Self::groups(&self.focus, &self.fluid, queue, p, res);
        self.inner.time_offscreen(
            device,
            queue,
            res,
            &p.pack(res),
            SceneKind::Focus.index(),
            &groups,
            reps,
        )
    }

    /// Creates the Focus room texture if it was released (what the live loop does when Focus enters the screen).
    pub fn focus_acquire(&mut self, device: &wgpu::Device) {
        if let Some(view) = self.focus.acquire(device) {
            self.inner
                .set_prepass_target(SceneKind::Focus.index(), Some(view));
        }
    }

    /// Drops the Focus room texture (what the live loop does when Focus is neither shown nor crossfading); the
    /// cover colours and the buffers stay. Acquire it again before drawing Focus.
    pub fn focus_release(&mut self) {
        self.focus.room = None;
        self.inner
            .set_prepass_target(SceneKind::Focus.index(), None);
    }

    /// Whether the Focus room texture exists.
    pub fn focus_room_alive(&self) -> bool {
        self.focus.room.is_some()
    }

    /// Fluid quality tier 0..=3 (grid 256x144 .. 768x432 at 16:9); a change restarts the field.
    pub fn set_fluid_tier(&mut self, tier: u8) {
        self.fluid.set_tier(tier);
    }

    /// Draws `scene` (or a crossfade `scene -> to` at blend `t`), then `post` records overlays.
    /// The Fluid simulation advances (fixed steps, from `p.time`) only while Fluid is one of the two scenes;
    /// otherwise its textures are released, so the next entry starts from a clear field. Focus's room texture
    /// likewise exists only while Focus is one of the two scenes (it is redrawn every frame, so nothing is lost).
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
        if scene == SceneKind::Focus || xfade.is_some_and(|(k, _)| k == SceneKind::Focus) {
            self.focus_acquire(&gpu.device);
        } else {
            self.focus_release();
        }
        let queue = gpu.queue.clone();
        self.inner.render_scenes(
            gpu,
            &p.pack(res),
            scene.index(),
            xfade.map(|(k, t)| (k.index(), t)),
            &Self::groups(&self.focus, &self.fluid, &queue, p, res),
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
            &Self::groups(&self.focus, &self.fluid, queue, p, res),
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
            &Self::groups(&self.focus, &self.fluid, queue, p, res),
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

    fn light(left: [f32; 3], right: [f32; 3]) -> ArtLight {
        let grid = std::array::from_fn(|i| {
            if i % ART_GRID < ART_GRID / 2 {
                left
            } else {
                right
            }
        });
        let mean = [0, 1, 2].map(|c| 0.5 * (left[c] + right[c]));
        ArtLight {
            grid,
            mean,
            mean_y: 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2],
        }
    }

    /// Mean linear luminance (0..1) and mean rgb of a pixel box of an RGBA8 image.
    fn box_rgb(img: &[u8], w: u32, x0: u32, x1: u32, y0: u32, y1: u32) -> [f32; 3] {
        let mut s = [0.0f32; 3];
        for y in y0..y1 {
            for x in x0..x1 {
                for c in 0..3 {
                    s[c] += f32::from(img[((y * w + x) * 4) as usize + c]) / 255.0;
                }
            }
        }
        s.map(|v| v / ((x1 - x0) * (y1 - y0)) as f32)
    }

    #[test]
    fn focus_block_packs_to_its_declared_size_and_stays_finite() {
        let mut f = FocusParams {
            rect: [f32::NAN, 0.5, f32::INFINITY, 0.2],
            ..Default::default()
        };
        f.ink[3] = [0.1, 0.2, 0.3, 0.4];
        f.ink_cap[3] = 0.05;
        let b = f.pack((1920, 1080));
        assert_eq!(b.len(), FOCUS_BYTES);
        assert!(
            b.chunks_exact(4)
                .all(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]).is_finite())
        );
        // ink rect 3 at vec4 index 7 (after rect, a, b, c, ink0..2), its cap at 8 .z... check by offsets
        assert_eq!(f32_at(&b, (4 + 3) * 16), 0.1);
        assert_eq!(f32_at(&b, 8 * 16 + 3 * 4), 0.05);
        // the dust cell follows the aspect: 21:9 has a bigger cell count per height, so a smaller cell, in 1/cell
        let wide = f32_at(&f.pack((2560, 1080)), 3 * 16 + 12);
        let std = f32_at(&b, 3 * 16 + 12);
        assert!(
            (std - 1.0 / 0.11).abs() < 0.01 && wide < std,
            "{std} {wide}"
        );
        let src = include_str!("focus.wgsl");
        assert!(
            src.contains("ink: array<vec4<f32>, 4>") && src.contains("fa: array<vec4<f32>, 130>")
        );
    }

    /// Like Fluid's textures, Focus's room texture is released while Focus is off screen and recreated on re-entry,
    /// drawing exactly the same room again (the room pass renders it from scratch every frame).
    #[test]
    fn focus_room_texture_is_released_off_screen_and_returns_identical_headless() {
        let Some((h, mut scene)) = headless() else {
            return;
        };
        let res = (320u32, 180u32);
        let mut p = SceneParams {
            mode: SceneMode::Live,
            ..Default::default()
        };
        p.album.rect = [0.5 * 320.0 / 180.0, 0.46, 0.25, 0.25];
        p.album.spill = 0.9;
        p.album.art_w = [1.0, 0.0];
        scene.set_focus_art(&h.queue, Some(light([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])));
        let draw = |s: &SceneSet| {
            s.render_offscreen_scenes(
                &h.device,
                &h.queue,
                res,
                &p,
                SceneKind::Focus,
                None,
                &|_, _| {},
            )
            .unwrap()
        };
        let maxd = |x: &[u8], y: &[u8]| x.iter().zip(y).map(|(p, q)| p.abs_diff(*q)).max().unwrap();
        assert!(scene.focus_room_alive());
        let before = draw(&scene);
        scene.focus_release();
        scene.focus_release();
        assert!(!scene.focus_room_alive());
        assert!(
            maxd(&draw(&scene), &before) > 8,
            "a released room is not drawn"
        );
        scene.focus_acquire(&h.device);
        scene.focus_acquire(&h.device);
        assert!(scene.focus_room_alive());
        assert!(maxd(&draw(&scene), &before) <= 1, "back as it was");
        assert_eq!(scene.focus_art_slots(), [true, false], "cover colours stay");
    }

    /// A zero-size source rectangle is the documented soft default mid-screen, not a point at the top-left.
    #[test]
    fn a_zero_size_focus_rect_packs_as_a_mid_screen_source() {
        let d = FocusParams::default();
        let b = d.pack((1280, 720));
        assert_eq!((f32_at(&b, 0), f32_at(&b, 4)), (0.5 * 1280.0 / 720.0, 0.5));
        let real = FocusParams {
            rect: [0.3, 0.4, 0.2, 0.1],
            ..d
        };
        let b = real.pack((1280, 720));
        assert_eq!(
            [f32_at(&b, 0), f32_at(&b, 4), f32_at(&b, 8), f32_at(&b, 12)],
            [0.3, 0.4, 0.2, 0.1]
        );
    }

    #[test]
    fn focus_room_is_lit_by_the_cover_it_follows_the_slots_and_never_shows_a_slot_headless() {
        let Some((h, mut scene)) = headless() else {
            return;
        };
        let res = (640u32, 360u32);
        let mut p = SceneParams {
            mode: SceneMode::Live,
            ..Default::default()
        };
        p.album.rect = [0.5 * 640.0 / 360.0, 0.46, 0.25, 0.25];
        p.album.spill = 0.9;
        let render = |s: &SceneSet, p: &SceneParams| {
            s.render_offscreen_scenes(
                &h.device,
                &h.queue,
                res,
                p,
                SceneKind::Focus,
                None,
                &|_, _| {},
            )
            .unwrap()
        };
        // no cover: the empty slot is a soft light, never darker than its surroundings
        p.album.art_w = [0.0, 0.0];
        let none = render(&scene, &p);
        assert_eq!(scene.focus_art_slots(), [false, false]);
        let c = box_rgb(&none, res.0, 290, 350, 130, 190);
        let edge = box_rgb(&none, res.0, 10, 70, 130, 190);
        assert!(
            c[0] + c[1] + c[2] > 1.2 * (edge[0] + edge[1] + edge[2]),
            "{c:?} vs {edge:?}"
        );
        // a red-left / blue-right cover lights its left red and its right blue
        scene.set_focus_art(&h.queue, Some(light([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])));
        assert_eq!(scene.focus_art_slots(), [true, false]);
        p.album.art_w = [1.0, 0.0];
        let lit = render(&scene, &p);
        let (l, r) = (
            box_rgb(&lit, res.0, 190, 225, 150, 210),
            box_rgb(&lit, res.0, 415, 450, 150, 210),
        );
        assert!(
            l[0] > 1.5 * l[2] && r[2] > 1.5 * r[0],
            "left {l:?} right {r:?}"
        );
        // the previous cover is mirrored into the second slot and weighs in with its own weight
        scene.shift_focus_art(&h.queue);
        scene.set_focus_art(&h.queue, Some(light([0.0, 1.0, 0.0], [0.0, 1.0, 0.0])));
        assert_eq!(scene.focus_art_slots(), [true, true]);
        p.album.art_w = [0.0, 1.0];
        let prev = render(&scene, &p);
        let g = box_rgb(&prev, res.0, 190, 225, 150, 210);
        // slot 1 holds the previous (red-left) cover, the current slot 0 is green: only the previous cover may show
        assert!(
            g[0] > 4.0 * g[1],
            "previous slot lights the room (red, not the current green): {g:?}"
        );
        scene.clear_prev_focus_art(&h.queue);
        scene.set_focus_art(&h.queue, None);
        assert_eq!(scene.focus_art_slots(), [false, false]);
        // the art-brightest cap: next to a white cover the room stays at most 0.6 x its mean Y (0.6), and the pocket
        // caps what is under a text rectangle; the floor dims with the bar
        scene.set_focus_art(&h.queue, Some(light([1.0; 3], [1.0; 3])));
        p.album.art_w = [1.0, 0.0];
        p.album.ink[0] = [1.15, 0.30, 1.55, 0.45];
        p.album.ink_cap[0] = 0.03;
        p.album.ink_on[0] = 1.0;
        let capped = render(&scene, &p);
        let y = |b: [f32; 3]| 0.2126 * b[0] + 0.7152 * b[1] + 0.0722 * b[2];
        let near = box_rgb(&capped, res.0, 415, 445, 170, 240); // just right of the cover (0.04 H)
        assert!(
            y(near) <= 0.6 + 0.01,
            "art stays the brightest: {}",
            y(near)
        );
        let under = box_rgb(&capped, res.0, 420, 540, 120, 160); // inside the pocket rectangle
        assert!(y(under) <= 0.03 + 0.004, "pocket cap: {}", y(under));
        let (below_before, above_before) = (
            box_rgb(&capped, res.0, 20, 140, 310, 350),
            box_rgb(&capped, res.0, 20, 140, 20, 60),
        );
        p.album.bar = 1.0;
        let dimmed = render(&scene, &p);
        let (below, above) = (
            box_rgb(&dimmed, res.0, 20, 140, 310, 350),
            box_rgb(&dimmed, res.0, 20, 140, 20, 60),
        );
        assert!(
            y(below) < 0.86 * y(below_before),
            "floor dims: {} -> {}",
            y(below_before),
            y(below)
        );
        assert!(
            (y(above) - y(above_before)).abs() < 0.004,
            "the rest does not"
        );
    }

    #[test]
    fn panel_groups_derive_from_lifecycle_and_partition_all_scenes() {
        use SceneKind::*;
        assert_eq!(
            SceneKind::panel_main().collect::<Vec<_>>(),
            [Spectrum, Focus, Orb]
        );
        assert_eq!(
            SceneKind::panel_experimental().collect::<Vec<_>>(),
            [Aurora, Fluid]
        );
        let mut all: Vec<_> = SceneKind::panel_main()
            .chain(SceneKind::panel_experimental())
            .collect();
        all.sort_by_key(|k| k.index());
        assert_eq!(all, SceneKind::ALL, "every scene is in exactly one group");
        for k in SceneKind::ALL {
            assert!(!k.label().is_empty());
            assert!(SceneKind::parse(k.name()) == Some(k));
        }
        assert_eq!(Fluid.label(), "Fluid");
        // experimental scenes never enter keys / cycling / auto rotation
        assert!(SceneKind::panel_experimental().all(|k| !k.in_selector()));
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
    fn transition_fx_packs_into_the_free_ex2_and_ex3_slots_and_zero_changes_nothing() {
        let p = SceneParams {
            front_x0: 0.7,
            fx: Fx {
                veil: [0.1, 0.2, 0.3, 0.4],
                bridge_luma: 0.16,
                breathing: 0.02,
                iris: 1.0,
            },
            ..Default::default()
        };
        let (z, b) = (SceneParams::default().pack((1, 1)), p.pack((1, 1)));
        let f = |b: &[u8], o: usize| f32::from_ne_bytes(b[o..o + 4].try_into().unwrap());
        // ex2 at 1312, ex3 at 1328 (ex4 is at 1344)
        assert_eq!(
            [f(&b, 1312), f(&b, 1316), f(&b, 1320), f(&b, 1324)],
            [0.1, 0.2, 0.3, 0.4]
        );
        assert_eq!(
            [f(&b, 1328), f(&b, 1332), f(&b, 1336), f(&b, 1340)],
            [0.16, 0.7, 0.02, 1.0]
        );
        assert!(z[1312..1328].iter().all(|&x| x == 0) && f(&z, 1328) == 0.0 && f(&z, 1340) == 0.0);
        // everything else is byte-identical
        assert_eq!(
            z[..1312],
            SceneParams {
                front_x0: 0.5,
                ..p.clone()
            }
            .pack((1, 1))[..1312]
        );
        assert_eq!(b.len(), UNIFORM_BYTES);
    }

    #[test]
    fn transition_fx_fades_every_scene_to_one_flat_palette_bridge_headless() {
        let Some((h, scene)) = headless() else { return };
        let res = (320u32, 180u32);
        let mut base = SceneParams {
            mode: SceneMode::Live,
            energy: 0.6,
            glow: 0.6,
            orb_band: [0.6, 0.3, 0.5, 0.4],
            aurora_band: [0.5, 0.6, 0.4, 0.3],
            aurora_phase: [0.1, 0.4, 0.7, 0.9],
            ..Default::default()
        };
        for i in 0..SLOTS {
            base.bars[i] = 0.2 + 0.6 * ((i as f32 * 0.37).sin() * 0.5 + 0.5);
            base.bright[i] = 0.7;
        }
        let render = |k: SceneKind, fx: Fx| {
            let p = SceneParams { fx, ..base.clone() };
            scene
                .render_offscreen_scenes(&h.device, &h.queue, res, &p, k, None, &|_, _| {})
                .unwrap()
        };
        let lum = |i: &[u8], x0, x1, y0, y1| mean_lum(i, res.0, x0, x1, y0, y1);
        let bridge = Fx {
            veil: [1.0; 4],
            bridge_luma: 0.16,
            ..Fx::default()
        };
        let flat = render(SceneKind::Spectrum, bridge);
        // the bridge: a quiet gradient, not black, near the requested luminance
        let y = lum(&flat, 0, res.0, 0, res.1);
        assert!((0.08..0.30).contains(&y), "bridge luminance {y}");
        assert!(
            lum(&flat, 140, 180, 70, 110) > lum(&flat, 0, 40, 0, 40) * 0.9,
            "radial lift, no hole"
        );
        for k in SceneKind::ALL {
            let idle = render(k, Fx::default());
            // veils at zero with the other fields set: the scene is untouched (early outs and exact multiplies)
            let inert = Fx {
                veil: [0.0; 4],
                bridge_luma: 0.16,
                breathing: 0.02,
                iris: 1.0,
            };
            assert_eq!(render(k, inert), idle, "{k:?}: zero veil is bit-identical");
            // fully veiled: every scene shows the very same bridge (no scene content leaks)
            assert_eq!(render(k, bridge), flat, "{k:?}: fully veiled == bridge");
            // half veiled differs from both ends
            let half = render(
                k,
                Fx {
                    veil: [0.5; 4],
                    bridge_luma: 0.16,
                    ..Fx::default()
                },
            );
            assert!(
                half != idle && half != flat,
                "{k:?}: half veil is a different picture"
            );
        }
        // layered scenes: one layer's veil removes only that layer; monolithic scenes ignore the layer veils
        for k in [SceneKind::Spectrum, SceneKind::Aurora, SceneKind::Orb] {
            let idle = render(k, Fx::default());
            let l_all = lum(&idle, 0, res.0, 0, res.1);
            let geom = render(
                k,
                Fx {
                    veil: [0.0, 1.0, 0.0, 0.0],
                    bridge_luma: 0.16,
                    ..Fx::default()
                },
            );
            assert!(
                geom != idle && lum(&geom, 0, res.0, 0, res.1) < l_all,
                "{k:?}: geometry veil removes the bars/ribbons/body"
            );
            let atmo = render(
                k,
                Fx {
                    veil: [0.6, 0.0, 0.0, 0.0],
                    bridge_luma: 0.16,
                    ..Fx::default()
                },
            );
            assert!(atmo != idle, "{k:?}: atmosphere veil");
        }
        for k in [SceneKind::Focus, SceneKind::Fluid] {
            let idle = render(k, Fx::default());
            let layers = Fx {
                veil: [1.0, 1.0, 1.0, 0.0],
                bridge_luma: 0.16,
                ..Fx::default()
            };
            assert_eq!(
                render(k, layers),
                idle,
                "{k:?}: layer veils do nothing here"
            );
        }
        // iris: closes from the corners inward, so the middle stays visible longer than with a plain fade
        let v = |iris: f32| {
            render(
                SceneKind::Focus,
                Fx {
                    veil: [0.0, 0.0, 0.0, 0.5],
                    bridge_luma: 0.16,
                    iris,
                    ..Fx::default()
                },
            )
        };
        let dist =
            |a: &[u8], x0, x1, y0, y1| (lum(a, x0, x1, y0, y1) - lum(&flat, x0, x1, y0, y1)).abs();
        let (with, without) = (v(1.0), v(0.0));
        assert!(with != without, "iris changes the picture");
        let (cw, co) = (
            dist(&with, 140, 180, 70, 110),
            dist(&without, 140, 180, 70, 110),
        );
        assert!(cw >= co - 1e-4, "the middle closes last: {cw} vs {co}");
        let (kw, ko) = (dist(&with, 0, 40, 0, 40), dist(&without, 0, 40, 0, 40));
        assert!(
            (kw - ko).abs() < 0.02,
            "corners are already fully on the veil curve: {kw} vs {ko}"
        );
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
