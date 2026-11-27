//! Scene "Fluid" (F1, Experimental): the GPU feedback simulation behind the liquid dye field.
//!
//! * Purpose: advance a dye-only simulation (two ping-pong RGBA16Float textures, x/y/z = dye for the palette
//!   roles pal1/pal2/pal3, w = heat) at a FIXED 1/60 s step, and hand the scene shader the current texture.
//! * Input: `FluidParams`, produced and smoothed by the CPU mapper (mg-engine `scene_map.rs`); the shader
//!   (`fluid.wgsl`) only integrates. Velocity is analytic and divergence-free, so there is no pressure solve.
//! * Frame-rate independence: `StepClock` turns the mapper's scene time into whole 1/60 s steps (at most 3 per
//!   frame; an over-budget remainder is dropped so a slow frame never snowballs).
//! * Lifetime: textures exist only while Fluid is shown or crossfading (`release` drops them, so a re-entry
//!   starts from a clear field); while released the scene reads a 1x1 zero texture. Nothing is allocated per
//!   frame: textures are recreated only on first use, tier change or re-entry, and for an aspect change only
//!   once it exceeds `REGRID_TOLERANCE` and has settled (`RegridPolicy`); the field is then resampled into the
//!   new grid (`fs_copy`), never cleared, so a fullscreen toggle or a window drag keeps the picture.
//! * Must not: read audio, invent activity, or inject anything into the bottom 38% of the screen (shader).

/// Grid heights of the quality tiers T0..T3 (width follows the render aspect).
pub const TIER_HEIGHTS: [u32; 4] = [144, 216, 288, 432];
/// Default tier (T1; the GT 730 reference). Auto-tier is a TODO (needs GPU timestamps; see the design addendum).
pub const DEFAULT_TIER: u8 = 1;
/// Fixed simulation step (seconds).
pub const STEP_S: f32 = 1.0 / 60.0;
pub const MAX_STEPS_PER_FRAME: u32 = 3;
/// Bytes of the packed `FU` block in `fluid.wgsl` (10 x vec4).
pub const FLUID_UNIFORM_BYTES: usize = 160;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Relative grid-width change ignored (window chrome, odd sizes): the sim keeps its grid, the render stretches
/// it by at most this much.
pub const REGRID_TOLERANCE: f32 = 0.03;
/// Calls (frames) a new grid must hold before the field is resampled into it: a window drag never reallocates.
pub const REGRID_SETTLE: u32 = 12;

/// What the CPU mapper hands the Fluid simulation each frame (see `fluid.wgsl` `FU` for units).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidParams {
    /// Bass/mid/treble emitters (dye x/y/z): uv x, uv y (down), strength (0..~1.3, already gated), sigma (sh).
    pub emit: [[f32; 4]; 3],
    /// Swirl per emitter (1/s, signed) and the bass push (upward jet, screen heights/s).
    pub swirl: [f32; 4],
    /// Onset vortices: uv x, uv y, peak swirl speed (sh/s, signed), age (s; >= 100 none).
    pub vort: [[f32; 4]; 2],
    /// Noise flow speed (sh/s), noise frequency (1/sh), evolution phase (lattice units, 0..256), buoyancy (sh/s).
    pub flow: [f32; 4],
    /// Dye memory tau (s), heat rate (1/s), exposure, jet width (sh; <= 0.03 means twice the bass splat).
    pub look: [f32; 4],
    /// Vortex dye hue (0 bass .. 1 treble) for slots 0 and 1, flow detail (0 laminar .. 1 turbulent: weight of
    /// the fine noise octave), treble spread (0..1: the treble splat widens up to 3x horizontally).
    pub misc: [f32; 4],
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            emit: [
                [0.35, 0.30, 0.0, 0.05],
                [0.55, 0.40, 0.0, 0.045],
                [0.70, 0.25, 0.0, 0.035],
            ],
            swirl: [0.0; 4],
            vort: [[0.5, 0.3, 0.0, 100.0]; 2],
            flow: [0.008, 3.0, 0.0, 0.012],
            look: [3.0, 0.0, 1.0, 0.0],
            misc: [0.0, 0.0, 1.0, 0.0],
        }
    }
}

impl FluidParams {
    /// Packs the WGSL `FU` block (non-finite values become 0). `round_trip` enables the step's round-trip
    /// error correction; `grid` is the simulation grid (the resample pass reads it).
    pub fn pack(&self, round_trip: bool, grid: (u32, u32)) -> [u8; FLUID_UNIFORM_BYTES] {
        let sim = [
            if round_trip { 1.0 } else { 0.0 },
            grid.0 as f32,
            grid.1 as f32,
            0.0,
        ];
        let mut out = [0u8; FLUID_UNIFORM_BYTES];
        let vals = self
            .emit
            .iter()
            .chain([&self.swirl])
            .chain(&self.vort)
            .chain([&self.flow, &self.look, &self.misc, &sim])
            .flatten();
        for (dst, x) in out.chunks_exact_mut(4).zip(vals) {
            dst.copy_from_slice(&(if x.is_finite() { *x } else { 0.0 }).to_ne_bytes());
        }
        out
    }
}

/// Whether a grid change from `cur` to `want` is worth a reallocation (tier change or a width change beyond
/// `REGRID_TOLERANCE`).
fn beyond_tolerance(cur: (u32, u32), want: (u32, u32)) -> bool {
    cur.1 != want.1 || (want.0 as f32 - cur.0 as f32).abs() > REGRID_TOLERANCE * cur.0 as f32
}

/// Debounces grid changes: a new grid is adopted only after the same target held for `REGRID_SETTLE` calls.
#[derive(Default, Clone, Debug)]
struct RegridPolicy {
    pending: Option<((u32, u32), u32)>,
}

impl RegridPolicy {
    /// True when the sim should move from `cur` to `want` now.
    fn update(&mut self, cur: (u32, u32), want: (u32, u32)) -> bool {
        if !beyond_tolerance(cur, want) {
            self.pending = None;
            return false;
        }
        let held = match self.pending {
            Some((g, n)) if g == want => n + 1,
            _ => 1,
        };
        self.pending = Some((want, held));
        if held >= REGRID_SETTLE {
            self.pending = None;
            return true;
        }
        false
    }
}

/// Simulation grid for `tier` (clamped to 0..=3) at a render size `res`: height from the tier, width follows the aspect.
pub fn grid_for(tier: u8, res: (u32, u32)) -> (u32, u32) {
    let h = TIER_HEIGHTS[usize::from(tier.min(3))];
    let aspect = res.0.max(1) as f32 / res.1.max(1) as f32;
    (((h as f32 * aspect).round() as u32).clamp(16, 4096), h)
}

/// Fixed-step accumulator driven by the mapper's scene time (seconds).
#[derive(Default, Clone, Debug)]
pub struct StepClock {
    acc: f32,
    last: Option<f32>,
}

impl StepClock {
    /// Whole 1/60 s steps owed at scene time `time` (0 on the first call; <= `MAX_STEPS_PER_FRAME`).
    pub fn steps(&mut self, time: f32) -> u32 {
        if !time.is_finite() {
            return 0;
        }
        let dt = self.last.map_or(0.0, |l| (time - l).clamp(0.0, 0.25));
        self.last = Some(time);
        self.acc += dt;
        let n = (((self.acc + 1e-5) / STEP_S) as u32).min(MAX_STEPS_PER_FRAME);
        self.acc = (self.acc - n as f32 * STEP_S).max(0.0);
        if self.acc >= STEP_S {
            self.acc = 0.0; // over budget: drop the remainder (the sim slows down instead of spiralling)
        }
        n
    }
}

struct Buffers {
    textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    /// `groups[i]` reads `views[i]`.
    groups: [wgpu::BindGroup; 2],
    cur: usize,
    grid: (u32, u32),
}

pub struct FluidSim {
    pipeline: wgpu::RenderPipeline,
    /// `fs_copy`: resamples the current field into a new grid.
    copy: wgpu::RenderPipeline,
    regrid: RegridPolicy,
    bgl: wgpu::BindGroupLayout,
    empty_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Bound while released: a 1x1 zero texture (the scene then shows only its base).
    idle_group: wgpu::BindGroup,
    bufs: Option<Buffers>,
    tier: u8,
    clock: StepClock,
}

impl FluidSim {
    /// `wgsl` is the full Fluid module (common.wgsl + fluid.wgsl); its `fs_sim` entry is the step.
    pub fn new(device: &wgpu::Device, wgsl: &str) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(FLUID_UNIFORM_BYTES as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // group 0 of the step is empty: `fs_sim` never touches the scene uniform `u`
        let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid-empty"),
            entries: &[],
        });
        let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid-empty"),
            layout: &empty_bgl,
            entries: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fluid-sim"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = crate::fullscreen::make_pipeline(
            device,
            &shader,
            &[Some(&empty_bgl), Some(&bgl)],
            "fluid-sim",
            FORMAT,
            "fs_sim",
        );
        let copy = crate::fullscreen::make_pipeline(
            device,
            &shader,
            &[Some(&empty_bgl), Some(&bgl)],
            "fluid-copy",
            FORMAT,
            "fs_copy",
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-uniform"),
            size: FLUID_UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fluid-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (_, idle_view) = dye_texture(device, (1, 1));
        let idle_group = make_group(device, &bgl, &uniform, &idle_view, &sampler);
        Self {
            pipeline,
            copy,
            regrid: RegridPolicy::default(),
            bgl,
            empty_group,
            uniform,
            sampler,
            idle_group,
            bufs: None,
            tier: DEFAULT_TIER,
            clock: StepClock::default(),
        }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl
    }

    /// Selects tier 0..=3 (clamped); a change releases the buffers (the field restarts clear).
    pub fn set_tier(&mut self, tier: u8) {
        let tier = tier.min(3);
        if tier != self.tier {
            self.tier = tier;
            self.release();
        }
    }

    pub fn tier(&self) -> u8 {
        self.tier
    }

    /// Simulation grid currently allocated (None while released).
    pub fn grid(&self) -> Option<(u32, u32)> {
        self.bufs.as_ref().map(|b| b.grid)
    }

    /// Drops the textures and the step clock: the next use starts from a clear field.
    pub fn release(&mut self) {
        self.bufs = None;
        self.clock = StepClock::default();
        self.regrid = RegridPolicy::default();
    }

    /// The bind group the scene shader reads (current dye, or the idle 1x1 zero texture).
    pub fn render_group(&self) -> &wgpu::BindGroup {
        self.bufs
            .as_ref()
            .map_or(&self.idle_group, |b| &b.groups[b.cur])
    }

    /// Advances to scene time `time` (seconds, from the mapper) in whole fixed steps and submits them.
    /// Returns the number of steps run. Always uploads `p` (the scene shader reads exposure from it).
    pub fn advance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: (u32, u32),
        p: &FluidParams,
        time: f32,
    ) -> u32 {
        let n = self.clock.steps(time);
        self.step(device, queue, res, p, n);
        n
    }

    /// Runs exactly `n` fixed steps with `p` (tests, soak runs and `advance`).
    pub fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        res: (u32, u32),
        p: &FluidParams,
        n: u32,
    ) {
        let want = grid_for(self.tier, res);
        let old = match self.bufs.take() {
            None => None,
            Some(b) if self.regrid.update(b.grid, want) => Some(b),
            keep => {
                self.bufs = keep;
                None
            }
        };
        let b = self.bufs.get_or_insert_with(|| {
            make_buffers(device, &self.bgl, &self.uniform, &self.sampler, want)
        });
        // written before the commands below run: the copy pass reads the new grid from it
        queue.write_buffer(&self.uniform, 0, &p.pack(self.tier > 0, b.grid));
        if old.is_none() && n == 0 {
            return;
        }
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fluid-steps"),
        });
        if let Some(o) = &old {
            // the new pair starts as the old field resampled (cur = 0), never as a clear field
            let mut pass = begin_pass(
                &mut enc,
                &b.views[0],
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            );
            pass.set_pipeline(&self.copy);
            pass.set_bind_group(0, &self.empty_group, &[]);
            pass.set_bind_group(1, &o.groups[o.cur], &[]);
            pass.draw(0..3, 0..1);
        }
        for _ in 0..n {
            let mut pass = begin_pass(&mut enc, &b.views[1 - b.cur], wgpu::LoadOp::Load);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.empty_group, &[]);
            pass.set_bind_group(1, &b.groups[b.cur], &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            b.cur = 1 - b.cur;
        }
        queue.submit(Some(enc.finish()));
    }

    /// Reads the current dye texture back as RGBA f32 (tests and diagnostics; waits for the GPU).
    pub fn read_dye(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(u32, u32, Vec<f32>), String> {
        let b = self.bufs.as_ref().ok_or("fluid sim is released")?;
        let (w, h) = b.grid;
        let row = (w * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-readback"),
            size: u64::from(row) * u64::from(h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fluid-readback"),
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &b.textures[b.cur],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let wait = std::time::Duration::from_secs(10);
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(wait),
            })
            .map_err(|e| format!("fluid readback wait failed: {e}"))?;
        rx.recv_timeout(wait)
            .map_err(|e| format!("fluid readback timed out: {e}"))?
            .map_err(|e| format!("fluid readback map failed: {e}"))?;
        let data = buf
            .slice(..)
            .get_mapped_range()
            .map_err(|e| format!("fluid readback range failed: {e}"))?;
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            let r = &data[y * row as usize..y * row as usize + (w * 8) as usize];
            out.extend(
                r.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_ne_bytes([c[0], c[1]]))),
            );
        }
        drop(data);
        buf.unmap();
        Ok((w, h, out))
    }
}

fn make_buffers(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    grid: (u32, u32),
) -> Buffers {
    let (t0, v0) = dye_texture(device, grid);
    let (t1, v1) = dye_texture(device, grid);
    let groups = [
        make_group(device, bgl, uniform, &v0, sampler),
        make_group(device, bgl, uniform, &v1, sampler),
    ];
    Buffers {
        textures: [t0, t1],
        views: [v0, v1],
        groups,
        cur: 0,
        grid,
    }
}

fn dye_texture(device: &wgpu::Device, size: (u32, u32)) -> (wgpu::Texture, wgpu::TextureView) {
    // wgpu zero-initialises every new texture: a fresh pair is a clear field
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fluid-dye"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let v = t.create_view(&wgpu::TextureViewDescriptor::default());
    (t, v)
}

fn make_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fluid-group"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn begin_pass<'e>(
    enc: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("fluid-step"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// IEEE half -> f32 (readback only; no extra dependency for one conversion).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_decode() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7e00).is_nan());
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }

    #[test]
    fn tiers_follow_the_aspect_and_clamp() {
        assert_eq!(grid_for(0, (1920, 1080)), (256, 144));
        assert_eq!(grid_for(1, (1920, 1080)), (384, 216));
        assert_eq!(grid_for(2, (1920, 1080)), (512, 288));
        assert_eq!(grid_for(3, (1920, 1080)), (768, 432));
        assert_eq!(grid_for(9, (1920, 1080)), (768, 432));
        assert_eq!(grid_for(1, (2560, 1080)).1, 216);
        assert_eq!(grid_for(1, (2560, 1080)).0, 512);
        assert_eq!(grid_for(0, (0, 0)), (144, 144));
    }

    #[test]
    fn step_clock_is_frame_rate_independent_and_capped() {
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            let mut c = StepClock::default();
            let frames = (10.0 * hz) as u32;
            let total: u32 = (0..=frames).map(|i| c.steps(i as f32 / hz)).sum();
            assert!(
                (599..=601).contains(&total),
                "{hz} Hz: {total} steps in 10 s"
            );
        }
        let mut c = StepClock::default();
        assert_eq!(c.steps(0.0), 0);
        assert_eq!(
            c.steps(1.0),
            MAX_STEPS_PER_FRAME,
            "a 250 ms hitch runs 3 steps"
        );
        assert_eq!(c.steps(1.0 + STEP_S), 1, "and the remainder was dropped");
        assert_eq!(c.steps(0.5), 0, "time going backwards runs nothing");
        assert_eq!(c.steps(f32::NAN), 0);
    }

    #[test]
    fn fluid_uniform_layout_matches_wgsl() {
        let p = FluidParams {
            emit: [[1.0, 2.0, 3.0, 4.0], [5.0; 4], [6.0; 4]],
            swirl: [7.0, 0.0, 0.0, 8.0],
            vort: [[9.0, 0.0, 0.0, 10.0], [11.0; 4]],
            flow: [12.0, 0.0, 0.0, f32::NAN],
            look: [13.0; 4],
            misc: [14.0, 0.0, 0.0, 15.0],
        };
        let b = p.pack(true, (384, 216));
        let at = |i: usize| f32::from_ne_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        assert_eq!(b.len(), FLUID_UNIFORM_BYTES);
        assert_eq!((at(0), at(12), at(48), at(60)), (1.0, 4.0, 7.0, 8.0));
        assert_eq!((at(64), at(76), at(80)), (9.0, 10.0, 11.0));
        assert_eq!(
            (at(96), at(108)),
            (12.0, 0.0),
            "NaN never reaches the shader"
        );
        assert_eq!(
            (at(112), at(128), at(140), at(144)),
            (13.0, 14.0, 15.0, 1.0)
        );
        assert_eq!(at(144), 1.0, "round-trip correction flag");
        assert_eq!((at(148), at(152), at(156)), (384.0, 216.0, 0.0), "grid");
        assert_eq!(
            FluidParams::default().pack(false, (1, 1))[144..148],
            0f32.to_ne_bytes()
        );
        let src = include_str!("fluid.wgsl");
        assert!(src.contains("emit: array<vec4<f32>, 3>") && src.contains("sim: vec4<f32>"));
    }

    #[test]
    fn small_aspect_changes_never_regrid_and_large_ones_wait_to_settle() {
        let mut r = RegridPolicy::default();
        let cur = grid_for(1, (1920, 1080));
        // window chrome / odd sizes within the tolerance: never a reallocation
        for res in [(1920, 1050), (1900, 1080), (1960, 1080)] {
            for _ in 0..100 {
                assert!(!r.update(cur, grid_for(1, res)), "{res:?}");
            }
        }
        // a window drag: the target changes every frame, so nothing is reallocated while it moves
        for w in 0..200u32 {
            assert!(!r.update(cur, grid_for(1, (1400 + 2 * w, 1080))));
        }
        // once it holds, exactly one regrid after REGRID_SETTLE frames
        let want = grid_for(1, (1440, 1080));
        let fired: Vec<bool> = (0..REGRID_SETTLE + 5)
            .map(|_| r.update(cur, want))
            .collect();
        assert_eq!(fired.iter().filter(|&&f| f).count(), 1);
        assert!(fired[REGRID_SETTLE as usize - 1]);
    }

    #[test]
    fn a_grid_change_resamples_the_field_instead_of_clearing_it_headless() {
        let Ok(h) = crate::context::init_headless() else {
            eprintln!("SKIPPED (no hardware GPU in this environment)");
            return;
        };
        let wgsl = concat!(include_str!("common.wgsl"), include_str!("fluid.wgsl"));
        let mut sim = FluidSim::new(&h.device, wgsl);
        let mut p = FluidParams::default();
        for e in &mut p.emit {
            e[2] = 0.8;
        }
        let mean = |sim: &FluidSim| {
            let (w, hh, d) = sim.read_dye(&h.device, &h.queue).expect("dye");
            (
                (w, hh),
                d.chunks_exact(4)
                    .map(|c| f64::from(c[0] + c[1] + c[2]))
                    .sum::<f64>()
                    / f64::from(w * hh),
            )
        };
        sim.step(&h.device, &h.queue, (1920, 1080), &p, 120);
        let (g0, m0) = mean(&sim);
        assert!(m0 > 0.01, "the emitters painted: {m0}");
        // small aspect change: same textures, untouched field
        let q = FluidParams {
            emit: p.emit.map(|e| [e[0], e[1], 0.0, e[3]]),
            ..p
        };
        for _ in 0..30 {
            sim.step(&h.device, &h.queue, (1920, 1050), &q, 0);
        }
        assert_eq!(mean(&sim), (g0, m0), "no reallocation within the tolerance");
        // fullscreen-like toggle to 4:3: after it settles, a new grid with the same field resampled
        for _ in 0..REGRID_SETTLE {
            sim.step(&h.device, &h.queue, (1440, 1080), &q, 0);
        }
        let (g1, m1) = mean(&sim);
        assert_eq!(g1, grid_for(DEFAULT_TIER, (1440, 1080)));
        eprintln!("FLUID regrid {g0:?} -> {g1:?}: mean dye {m0:.4} -> {m1:.4}");
        assert!((m1 - m0).abs() <= 0.05 * m0, "mean dye {m0} -> {m1}");
    }
}
