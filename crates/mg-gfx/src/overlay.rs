//! Overlay pass: album artwork and song text drawn over the scene at native resolution.
//!
//! Resource policy: textures are created only when new content arrives (a track change), sized exactly to
//! the content (artwork is already bounded by the decoder; text bitmaps by mg-text), and replaced or
//! dropped immediately when the content goes away. Nothing is allocated or uploaded per frame; a frame
//! only rewrites one 128-byte uniform and records one draw.

const UNIFORM_BYTES: usize = 12 * 16;
const CONTROLS_BYTES: usize = 34 * 16;
/// Text slots: 0 title, 1 artist, 2..=4 lyric lines, 5 elapsed time, 6 duration (control bar).
pub const TEXT_SLOTS: usize = 7;
/// First text slot of the control bar: `CLOCK_SLOT` is the elapsed time, `CLOCK_SLOT + 1` the duration.
pub const CLOCK_SLOT: usize = 5;
/// Largest texture side accepted (defence in depth; callers already bound their content).
const MAX_TEXTURE_SIDE: u32 = 4096;

/// Artwork placement in surface pixels. `(cx, cy)` is the centre; `hw`/`hh` the half extents (the caller
/// preserves the image aspect ratio when computing them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtDraw {
    pub cx: f32,
    pub cy: f32,
    pub hw: f32,
    pub hh: f32,
    pub alpha: f32,
    pub radius: f32,
    /// Halo strength (0..1) and spread in pixels: the music-reactive glow that leaves the artwork.
    pub halo: f32,
    pub spread: f32,
}

/// Text placement: top-left corner and size in surface pixels, plus opacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextDraw {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub alpha: f32,
}

/// Icon ids understood by `overlay.wgsl` (`icon_cov`).
pub mod icon {
    pub const PREV: u32 = 0;
    pub const PLAY: u32 = 1;
    pub const PAUSE: u32 = 2;
    pub const NEXT: u32 = 3;
    pub const SCENE: u32 = 4;
    pub const LYRICS: u32 = 5;
    pub const LAYOUT: u32 = 6;
    pub const FULLSCREEN: u32 = 7;
    pub const EXIT: u32 = 8;
}

/// Number of buttons the control bar shader draws.
pub const CONTROL_BUTTONS: usize = 8;

/// One button, final surface pixels: centre, half size of the square hit target, icon id and states (0..1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ButtonDraw {
    pub cx: f32,
    pub cy: f32,
    pub half: f32,
    pub icon: u32,
    /// 1 enabled, ~0.35 disabled.
    pub opacity: f32,
    pub hover: f32,
    pub toggled: f32,
    pub focus: bool,
    pub press: f32,
}

/// Progress line: track x range, centre y, half thickness, played fraction and hover (knob) amount.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProgressDraw {
    pub x0: f32,
    pub x1: f32,
    pub y: f32,
    pub half_thick: f32,
    pub frac: f32,
    pub hover: f32,
    /// Keyboard focus ring around the track.
    pub focus: bool,
}

/// Opacity of the unplayed and the played part of the progress track (mirrors `fs_controls`; the control bar
/// checks their contrast over the backing).
pub const TRACK_ALPHA: f32 = 0.12;
pub const PLAYED_ALPHA: f32 = 0.95;

/// The temporary on-screen control bar (drawn as a second, scissored draw of the overlay pass).
#[derive(Clone, Debug, PartialEq)]
pub struct ControlsDraw {
    /// Master alpha 0..1 (the draw is skipped at 0).
    pub alpha: f32,
    /// Pixels per design pixel.
    pub scale: f32,
    /// Pill centre x, centre y, half width, half height.
    pub bar: [f32; 4],
    /// Divider x, centre y, half height.
    pub sep: [f32; 3],
    pub buttons: [ButtonDraw; CONTROL_BUTTONS],
    pub progress: Option<ProgressDraw>,
    pub elapsed: Option<TextDraw>,
    pub duration: Option<TextDraw>,
    pub time_alpha: f32,
    /// Linear RGB: pill fill, icons/text, accent.
    pub bg: [f32; 3],
    pub bg_alpha: f32,
    pub fg: [f32; 3],
    pub accent: [f32; 3],
}

impl ControlsDraw {
    /// Packs the WGSL `C` layout for a surface of `surface` pixels (public for tests).
    pub fn pack(&self, surface: (u32, u32)) -> Vec<u8> {
        let mut o = Vec::with_capacity(CONTROLS_BYTES);
        put(
            &mut o,
            &[
                surface.0 as f32,
                surface.1 as f32,
                self.alpha.clamp(0.0, 1.0),
                self.scale.max(0.01),
            ],
        );
        put(&mut o, &self.bar);
        put(
            &mut o,
            &[
                self.bg[0],
                self.bg[1],
                self.bg[2],
                self.bg_alpha.clamp(0.0, 1.0),
            ],
        );
        put(&mut o, &[self.fg[0], self.fg[1], self.fg[2], 0.0]);
        put(
            &mut o,
            &[self.accent[0], self.accent[1], self.accent[2], 0.0],
        );
        match &self.progress {
            Some(p) => {
                put(&mut o, &[p.x0, p.x1, p.y, p.half_thick.max(0.0)]);
                put(
                    &mut o,
                    &[
                        p.frac.clamp(0.0, 1.0),
                        p.hover.clamp(0.0, 1.0),
                        self.time_alpha.clamp(0.0, 1.0),
                        if p.focus { 1.0 } else { 0.0 },
                    ],
                );
            }
            None => put(&mut o, &[0.0; 8]),
        }
        put(
            &mut o,
            &[self.sep[0], self.sep[1], self.sep[2].max(0.0), 0.0],
        );
        let text = |t: &Option<TextDraw>| match t {
            Some(t) => [t.x, t.y, t.w.max(0.0), t.h.max(0.0)],
            None => [0.0; 4],
        };
        put(&mut o, &text(&self.elapsed));
        put(&mut o, &text(&self.duration));
        for b in &self.buttons {
            put(&mut o, &[b.cx, b.cy, b.half.max(0.0), b.icon as f32]);
            put(
                &mut o,
                &[
                    b.opacity.clamp(0.0, 1.0),
                    b.hover.clamp(0.0, 1.0),
                    b.toggled.clamp(0.0, 1.0),
                    if b.focus { 1.0 } else { 0.0 },
                ],
            );
            put(&mut o, &[b.press.clamp(0.0, 1.0), 0.0, 0.0, 0.0]);
        }
        debug_assert_eq!(o.len(), CONTROLS_BYTES);
        o
    }

    /// Scissor rectangle `(x, y, w, h)` covering the pill, its shadow, the progress line and the times,
    /// clamped inside a `surface` (always at least 1x1).
    pub fn scissor(&self, surface: (u32, u32)) -> (u32, u32, u32, u32) {
        let s = self.scale.max(0.01);
        let m = 28.0 * s;
        let mut x0 = self.bar[0] - self.bar[2] - m;
        let mut x1 = self.bar[0] + self.bar[2] + m;
        let mut y0 = self.bar[1] - self.bar[3] - m;
        let y1 = self.bar[1] + self.bar[3] + m;
        if let Some(p) = &self.progress {
            x0 = x0.min(p.x0 - 12.0 * s); // focus ring
            x1 = x1.max(p.x1 + 12.0 * s);
            y0 = y0.min(p.y - 14.0 * s);
        }
        for t in [&self.elapsed, &self.duration].into_iter().flatten() {
            x0 = x0.min(t.x - 4.0 * s);
            x1 = x1.max(t.x + t.w + 4.0 * s);
            y0 = y0.min(t.y - 4.0 * s);
        }
        let (sw, sh) = (surface.0.max(1), surface.1.max(1));
        let f = |v: f32, hi: u32| {
            if v.is_finite() {
                v.clamp(0.0, hi as f32)
            } else {
                0.0
            }
        };
        let x0 = (f(x0, sw - 1).floor() as u32).min(sw - 1);
        let y0 = (f(y0, sh - 1).floor() as u32).min(sh - 1);
        let x1 = (f(x1, sw).ceil() as u32).clamp(x0 + 1, sw);
        let y1 = (f(y1, sh).ceil() as u32).clamp(y0 + 1, sh);
        (x0, y0, x1 - x0, y1 - y0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverlayParams {
    pub res: (u32, u32),
    pub time: f32,
    pub energy: f32,
    pub art: Option<ArtDraw>,
    pub title: Option<TextDraw>,
    pub artist: Option<TextDraw>,
    /// Lyric lines (slot 2, 3, 4 of the text textures) and their slow glow boost (0..1).
    pub lyrics: [Option<TextDraw>; 3],
    pub lyric_glow: f32,
    /// Shadow strength under the text (0..1).
    pub shadow: f32,
    /// Linear RGB.
    pub text_color: [f32; 3],
    pub halo_color: [f32; 3],
    /// The control bar when visible (None or alpha 0 = nothing drawn, zero cost).
    pub controls: Option<ControlsDraw>,
}

fn put(out: &mut Vec<u8>, vals: &[f32]) {
    for v in vals {
        let v = if v.is_finite() { *v } else { 0.0 };
        out.extend_from_slice(&v.to_ne_bytes());
    }
}

impl OverlayParams {
    /// Packs the WGSL `O` layout (public for tests).
    pub fn pack(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(UNIFORM_BYTES);
        put(
            &mut o,
            &[self.res.0 as f32, self.res.1 as f32, self.time, self.energy],
        );
        match &self.art {
            Some(a) => {
                put(&mut o, &[a.cx, a.cy, a.hw.max(0.0), a.hh.max(0.0)]);
                put(
                    &mut o,
                    &[
                        a.alpha.clamp(0.0, 1.0),
                        a.radius.max(0.0),
                        a.halo.clamp(0.0, 1.0),
                        a.spread.max(1.0),
                    ],
                );
            }
            None => put(&mut o, &[0.0; 8]),
        }
        let text = |t: &Option<TextDraw>| match t {
            Some(t) => [t.x, t.y, t.w.max(0.0), t.h.max(0.0)],
            None => [0.0; 4],
        };
        put(&mut o, &text(&self.title));
        put(&mut o, &text(&self.artist));
        let al = |t: &Option<TextDraw>| t.map_or(0.0, |t| t.alpha.clamp(0.0, 1.0));
        put(
            &mut o,
            &[
                al(&self.title),
                al(&self.artist),
                self.shadow.clamp(0.0, 1.0),
                0.0,
            ],
        );
        put(
            &mut o,
            &[
                self.text_color[0],
                self.text_color[1],
                self.text_color[2],
                0.0,
            ],
        );
        put(
            &mut o,
            &[
                self.halo_color[0],
                self.halo_color[1],
                self.halo_color[2],
                0.0,
            ],
        );
        for l in &self.lyrics {
            put(&mut o, &text(l));
        }
        put(
            &mut o,
            &[
                al(&self.lyrics[0]),
                al(&self.lyrics[1]),
                al(&self.lyrics[2]),
                self.lyric_glow.clamp(0.0, 1.0),
            ],
        );
        debug_assert_eq!(o.len(), UNIFORM_BYTES);
        o
    }

    /// The control bar, when it would change pixels.
    fn visible_controls(&self) -> Option<&ControlsDraw> {
        self.controls.as_ref().filter(|c| c.alpha > 0.002)
    }

    /// True when the artwork/text part would change no pixel (nothing present or fully transparent).
    pub fn is_empty(&self) -> bool {
        let vis = |a: f32| a > 0.002;
        !(self.art.is_some_and(|a| vis(a.alpha))
            || self.title.is_some_and(|t| vis(t.alpha))
            || self.artist.is_some_and(|t| vis(t.alpha))
            || self.lyrics.iter().any(|l| l.is_some_and(|t| vis(t.alpha))))
    }
}

pub struct Overlay {
    pipeline: wgpu::RenderPipeline,
    controls_pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    controls_uniform: wgpu::Buffer,
    art: Option<wgpu::TextureView>,
    texts: [Option<wgpu::TextureView>; TEXT_SLOTS],
    /// 1x1 stand-ins so the bind group is always complete.
    blank_rgba: wgpu::TextureView,
    blank_r8: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

fn make_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    format: wgpu::TextureFormat,
    (w, h): (u32, u32),
    bytes_per_px: u32,
    data: &[u8],
) -> Option<wgpu::TextureView> {
    if w == 0
        || h == 0
        || w > MAX_TEXTURE_SIDE
        || h > MAX_TEXTURE_SIDE
        || data.len() != (w * h * bytes_per_px) as usize
    {
        return None;
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * bytes_per_px),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    Some(tex.create_view(&wgpu::TextureViewDescriptor::default()))
}

impl Overlay {
    /// Overlay for `context::init_headless` devices (RGBA8 target), for tests and previews.
    pub fn new_headless(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::new(device, queue, wgpu::TextureFormat::Rgba8Unorm)
    }

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(UNIFORM_BYTES as u64),
                    },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                tex_entry(5),
                tex_entry(6),
                tex_entry(7),
                tex_entry(8),
                tex_entry(9),
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(CONTROLS_BYTES as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let make_pipeline = |label: &str, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = make_pipeline("overlay", "fs");
        let controls_pipeline = make_pipeline("overlay controls", "fs_controls");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay"),
            size: UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let controls_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay controls"),
            size: CONTROLS_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blank_rgba = make_texture(
            device,
            queue,
            "blank",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            (1, 1),
            4,
            &[0; 4],
        )
        .expect("1x1 texture");
        let blank_r8 = make_texture(
            device,
            queue,
            "blank",
            wgpu::TextureFormat::R8Unorm,
            (1, 1),
            1,
            &[0],
        )
        .expect("1x1 texture");
        let mut s = Self {
            bind_group: Self::group(
                device,
                &bgl,
                [&uniform, &controls_uniform],
                &blank_rgba,
                [&blank_r8; TEXT_SLOTS],
                &sampler,
            ),
            pipeline,
            controls_pipeline,
            bgl,
            sampler,
            uniform,
            controls_uniform,
            art: None,
            texts: Default::default(),
            blank_rgba,
            blank_r8,
        };
        s.rebuild(device);
        s
    }

    fn group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniforms: [&wgpu::Buffer; 2],
        art: &wgpu::TextureView,
        texts: [&wgpu::TextureView; TEXT_SLOTS],
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: uniforms[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(texts[5]),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(texts[6]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(art),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(texts[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(texts[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(texts[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(texts[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(texts[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    fn rebuild(&mut self, device: &wgpu::Device) {
        self.bind_group = Self::group(
            device,
            &self.bgl,
            [&self.uniform, &self.controls_uniform],
            self.art.as_ref().unwrap_or(&self.blank_rgba),
            std::array::from_fn(|i| self.texts[i].as_ref().unwrap_or(&self.blank_r8)),
            &self.sampler,
        );
    }

    /// Replaces the artwork (straight RGBA8, `w x h`). Returns false (and clears the artwork) when the data
    /// is malformed or too large.
    pub fn set_art(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        rgba: &[u8],
    ) -> bool {
        self.art = make_texture(
            device,
            queue,
            "art",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            size,
            4,
            rgba,
        );
        self.rebuild(device);
        self.art.is_some()
    }

    pub fn clear_art(&mut self, device: &wgpu::Device) {
        self.art = None;
        self.rebuild(device);
    }

    /// Replaces text slot `which` (0 title, 1 artist, 2..=4 lyric lines) with a coverage bitmap (8-bit, `w x h`).
    pub fn set_text(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        which: usize,
        size: (u32, u32),
        coverage: &[u8],
    ) -> bool {
        let t = make_texture(
            device,
            queue,
            "text",
            wgpu::TextureFormat::R8Unorm,
            size,
            1,
            coverage,
        );
        let ok = t.is_some();
        self.texts[which.min(TEXT_SLOTS - 1)] = t;
        self.rebuild(device);
        ok
    }

    pub fn clear_text(&mut self, device: &wgpu::Device, which: usize) {
        self.texts[which.min(TEXT_SLOTS - 1)] = None;
        self.rebuild(device);
    }

    /// Records the overlay draw on top of whatever `view` already holds. No-op when nothing is visible.
    pub fn draw(
        &self,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        p: &OverlayParams,
    ) {
        let main = !p.is_empty();
        let controls = p.visible_controls();
        if !main && controls.is_none() {
            return;
        }
        if main {
            queue.write_buffer(&self.uniform, 0, &p.pack());
        }
        if let Some(c) = controls {
            queue.write_buffer(&self.controls_uniform, 0, &c.pack(p.res));
        }
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        if main {
            pass.set_pipeline(&self.pipeline);
            pass.draw(0..3, 0..1);
        }
        if let Some(c) = controls {
            // Only the bar area is shaded: a hidden bar costs nothing, a visible one a small scissored draw.
            let (x, y, w, h) = c.scissor(p.res);
            pass.set_scissor_rect(x, y, w, h);
            pass.set_pipeline(&self.controls_pipeline);
            pass.draw(0..3, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::init_headless;
    use crate::scene::{SceneParams, SceneSet};

    fn px(img: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * w + x) * 4) as usize;
        [img[i], img[i + 1], img[i + 2], img[i + 3]]
    }

    fn params(res: (u32, u32)) -> OverlayParams {
        OverlayParams {
            res,
            time: 1.0,
            energy: 0.5,
            art: Some(ArtDraw {
                cx: 160.0,
                cy: 90.0,
                hw: 40.0,
                hh: 20.0,
                alpha: 1.0,
                radius: 4.0,
                halo: 0.0,
                spread: 10.0,
            }),
            title: None,
            artist: None,
            lyrics: [None; 3],
            lyric_glow: 0.0,
            shadow: 0.5,
            text_color: [1.0, 1.0, 1.0],
            halo_color: [0.2, 0.4, 1.0],
            controls: None,
        }
    }

    fn bar(res: (u32, u32)) -> ControlsDraw {
        let s = res.1 as f32 / 1080.0;
        let (cx, cy) = (res.0 as f32 / 2.0, res.1 as f32 - 60.0 * s);
        let ids = [
            icon::PREV,
            icon::PLAY,
            icon::NEXT,
            icon::SCENE,
            icon::LYRICS,
            icon::LAYOUT,
            icon::FULLSCREEN,
            icon::EXIT,
        ];
        ControlsDraw {
            alpha: 1.0,
            scale: s,
            bar: [cx, cy, 280.0 * s, 32.0 * s],
            sep: [cx - 4.0 * s, cy, 18.0 * s],
            buttons: std::array::from_fn(|i| ButtonDraw {
                cx: cx - 230.0 * s + 66.0 * s * i as f32,
                cy,
                half: 22.0 * s,
                icon: ids[i],
                opacity: 1.0,
                hover: 0.0,
                toggled: 0.0,
                focus: false,
                press: 0.0,
            }),
            progress: Some(ProgressDraw {
                x0: cx - 224.0 * s,
                x1: cx + 224.0 * s,
                y: cy - 52.0 * s,
                half_thick: 1.5 * s,
                frac: 0.4,
                hover: 0.0,
                focus: false,
            }),
            elapsed: None,
            duration: None,
            time_alpha: 1.0,
            bg: [0.01, 0.012, 0.02],
            bg_alpha: 0.62,
            fg: [0.9, 0.9, 0.9],
            accent: [0.2, 0.5, 1.0],
        }
    }

    #[test]
    fn controls_pack_matches_wgsl_scissor_is_inside_and_hostile_values_are_finite() {
        let res = (1920u32, 1080u32);
        let mut c = bar(res);
        let b = c.pack(res);
        assert_eq!(b.len(), CONTROLS_BYTES);
        let src = include_str!("overlay.wgsl");
        assert!(src.contains("b: array<vec4<f32>, 24>") && src.contains("fn fs_controls"));
        c.alpha = f32::NAN;
        c.bar[0] = f32::INFINITY;
        c.buttons[3].hover = f32::NAN;
        assert!(
            c.pack(res)
                .chunks_exact(4)
                .all(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]).is_finite())
        );
        for res in [(1920u32, 1080u32), (320, 180), (1, 1), (100, 4000)] {
            let (x, y, w, h) = bar(res).scissor(res);
            assert!(w >= 1 && h >= 1 && x + w <= res.0.max(1) && y + h <= res.1.max(1));
        }
        let (x, y, w, h) = c.scissor(res); // NaN/inf geometry still yields a legal rectangle
        assert!(w >= 1 && h >= 1 && x + w <= res.0 && y + h <= res.1);
    }

    #[test]
    fn controls_draw_enabled_disabled_hover_and_focus_states_headless() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        let res = (960u32, 540u32);
        let scene = SceneSet::new_headless(&h.device);
        let ov = Overlay::new_headless(&h.device, &h.queue);
        let sp = SceneParams::default();
        let bare = scene
            .render_offscreen(&h.device, &h.queue, res, &sp)
            .unwrap();
        let render = |c: Option<ControlsDraw>| {
            let mut p = params(res);
            p.art = None;
            p.controls = c;
            scene
                .render_offscreen_with(&h.device, &h.queue, res, &sp, &|enc, view| {
                    ov.draw(&h.queue, enc, view, &p)
                })
                .unwrap()
        };
        let diff = |a: &[u8], b: &[u8], x0: u32, x1: u32, y0: u32, y1: u32| {
            let mut n = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = ((y * res.0 + x) * 4) as usize;
                    n += (0..3)
                        .map(|k| u64::from(a[i + k].abs_diff(b[i + k])))
                        .sum::<u64>();
                }
            }
            n
        };
        let on = bar(res);
        let img = render(Some(on.clone()));
        let b = on.buttons[1]; // play
        let reg = |r: &[u8]| {
            diff(
                r,
                &bare,
                (b.cx - b.half) as u32,
                (b.cx + b.half) as u32,
                (b.cy - b.half) as u32,
                (b.cy + b.half) as u32,
            )
        };
        assert!(reg(&img) > 5000, "bar and icon drawn");
        // alpha 0 / None leave the scene untouched (zero cost path)
        let mut hidden = on.clone();
        hidden.alpha = 0.0;
        assert_eq!(render(Some(hidden)), bare);
        assert_eq!(render(None), bare);
        // far from the bar nothing changes (scissor + coverage)
        assert_eq!(diff(&img, &bare, 0, res.0, 0, 200), 0);
        // disabled icon is dimmer than the enabled one, hover and focus change the pixels
        let mut dis = on.clone();
        dis.buttons[1].opacity = 0.35;
        let mut hov = on.clone();
        hov.buttons[1].hover = 1.0;
        let mut foc = on.clone();
        foc.buttons[1].focus = true;
        let mut tog = on.clone();
        tog.buttons[1].toggled = 1.0;
        let (i_dis, i_hov, i_foc, i_tog) = (
            render(Some(dis)),
            render(Some(hov)),
            render(Some(foc)),
            render(Some(tog)),
        );
        // regions that must differ from the plain enabled state
        let around = |r: &[u8], pad: f32| {
            diff(
                r,
                &img,
                (b.cx - b.half - pad) as u32,
                (b.cx + b.half + pad) as u32,
                (b.cy - b.half - pad) as u32,
                (b.cy + b.half + pad) as u32,
            )
        };
        assert!(around(&i_dis, 0.0) > 800, "disabled differs");
        assert!(reg(&i_dis) < reg(&img), "disabled is dimmer");
        assert!(around(&i_hov, 0.0) > 800, "hover plate differs");
        assert!(
            around(&i_foc, 8.0) > around(&i_foc, 0.0) + 100,
            "focus ring is drawn outside the hit target"
        );
        assert!(around(&i_tog, 0.0) > 500, "toggled differs");
        // nothing is NaN-black: the pill interior is a translucent dark tint, not pure black or transparent
        let px = |r: &[u8], x: u32, y: u32| {
            let i = ((y * res.0 + x) * 4) as usize;
            [r[i], r[i + 1], r[i + 2], r[i + 3]]
        };
        let (bx, by) = (on.bar[0] as u32 + 60, on.bar[1] as u32 + 20);
        let (a0, a1) = (px(&bare, bx, by), px(&img, bx, by));
        assert!(a1 != a0, "pill tints the scene: {a0:?} -> {a1:?}");
        assert!(a1[3] >= 250, "output stays opaque over the opaque scene");
        let sum: u64 = img.iter().map(|&v| u64::from(v)).sum();
        assert!(sum > 0);
    }

    #[test]
    fn layout_matches_wgsl_and_hostile_values_are_finite() {
        let mut p = params((320, 180));
        p.energy = f32::NAN;
        p.art.as_mut().unwrap().hw = f32::INFINITY;
        let b = p.pack();
        assert_eq!(b.len(), UNIFORM_BYTES);
        assert!(
            b.chunks_exact(4)
                .all(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]).is_finite())
        );
        let src = include_str!("overlay.wgsl");
        assert!(src.contains("col_halo: vec4<f32>") && src.contains("text_p: vec4<f32>"));
        assert!(
            OverlayParams {
                art: None,
                ..params((1, 1))
            }
            .is_empty()
        );
        let mut hidden = params((1, 1));
        hidden.art.as_mut().unwrap().alpha = 0.0;
        assert!(hidden.is_empty());
    }

    #[test]
    fn artwork_and_text_are_drawn_and_nothing_is_drawn_when_empty_headless() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        let res = (320u32, 180u32);
        let scene = SceneSet::new_headless(&h.device);
        let mut ov = Overlay::new(&h.device, &h.queue, wgpu::TextureFormat::Rgba8Unorm);
        // artwork: left half pure red, right half pure green (2:1 image shown at 80x40 px)
        let (aw, ah) = (8u32, 4u32);
        let art: Vec<u8> = (0..ah)
            .flat_map(|_| {
                (0..aw).flat_map(|x| {
                    if x < aw / 2 {
                        [255, 0, 0, 255]
                    } else {
                        [0, 255, 0, 255]
                    }
                })
            })
            .collect();
        assert!(ov.set_art(&h.device, &h.queue, (aw, ah), &art));
        assert!(
            !ov.set_art(&h.device, &h.queue, (aw, ah), &art[..10]),
            "malformed data is refused"
        );
        assert!(ov.set_art(&h.device, &h.queue, (aw, ah), &art));
        // title: a solid coverage block 100x20
        assert!(ov.set_text(&h.device, &h.queue, 0, (100, 20), &vec![255u8; 2000]));
        let sp = SceneParams::default();
        let render = |ov: &Overlay, p: &OverlayParams| {
            scene
                .render_offscreen_with(&h.device, &h.queue, res, &sp, &|enc, view| {
                    ov.draw(&h.queue, enc, view, p)
                })
                .unwrap()
        };
        let base = scene
            .render_offscreen(&h.device, &h.queue, res, &sp)
            .unwrap();
        let mut p = params(res);
        p.title = Some(TextDraw {
            x: 20.0,
            y: 140.0,
            w: 100.0,
            h: 20.0,
            alpha: 1.0,
        });
        let img = render(&ov, &p);
        let left = px(&img, res.0, 140, 90);
        let right = px(&img, res.0, 180, 90);
        assert!(left[0] > 230 && left[1] < 25, "left half red: {left:?}");
        assert!(
            right[1] > 230 && right[0] < 25,
            "right half green: {right:?}"
        );
        // aspect ratio preserved: well above the 2:1 rect (half height 20) the pixel is the plain scene
        assert_eq!(
            px(&img, res.0, 160, 60),
            px(&base, res.0, 160, 60),
            "outside the artwork rect"
        );
        let t = px(&img, res.0, 70, 150);
        assert!(
            t[0] > 230 && t[1] > 230 && t[2] > 230,
            "title block drawn white: {t:?}"
        );
        // no content or fully transparent: identical to the bare scene
        let mut none = params(res);
        none.art = None;
        assert_eq!(render(&ov, &none), base);
        let mut faded = p.clone();
        faded.art.as_mut().unwrap().alpha = 0.0;
        faded.title.as_mut().unwrap().alpha = 0.0;
        assert_eq!(render(&ov, &faded), base);
        // cleared content draws nothing even when placed
        ov.clear_art(&h.device);
        ov.clear_text(&h.device, 0);
        let img2 = render(&ov, &p);
        assert!(
            px(&img2, res.0, 140, 90)[0] < 100,
            "cleared artwork leaves the scene"
        );
    }

    #[test]
    fn lyric_lines_draw_in_their_slots_and_fade_independently_headless() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        let res = (320u32, 180u32);
        let scene = SceneSet::new_headless(&h.device);
        let mut ov = Overlay::new_headless(&h.device, &h.queue);
        assert!(ov.set_text(&h.device, &h.queue, 2, (60, 12), &vec![255u8; 720]));
        assert!(ov.set_text(&h.device, &h.queue, 4, (60, 12), &vec![255u8; 720]));
        let sp = SceneParams::default();
        let bare = scene
            .render_offscreen(&h.device, &h.queue, res, &sp)
            .unwrap();
        let mut p = params(res);
        p.art = None;
        p.lyrics = [
            Some(TextDraw {
                x: 20.0,
                y: 40.0,
                w: 60.0,
                h: 12.0,
                alpha: 1.0,
            }),
            None,
            Some(TextDraw {
                x: 20.0,
                y: 120.0,
                w: 60.0,
                h: 12.0,
                alpha: 0.4,
            }),
        ];
        let img = scene
            .render_offscreen_with(&h.device, &h.queue, res, &sp, &|enc, view| {
                ov.draw(&h.queue, enc, view, &p)
            })
            .unwrap();
        let a = px(&img, res.0, 50, 46);
        let b = px(&img, res.0, 50, 126);
        assert!(a[0] > 230 && a[1] > 230, "full-alpha line is bright: {a:?}");
        assert!(b[0] > 60 && b[0] < 200, "40% line is dimmer: {b:?}");
        assert_eq!(
            px(&img, res.0, 200, 90),
            px(&bare, res.0, 200, 90),
            "elsewhere untouched"
        );
        // an absent slot draws nothing even if its texture exists
        assert_eq!(px(&img, res.0, 50, 86), px(&bare, res.0, 50, 86));
    }
}
