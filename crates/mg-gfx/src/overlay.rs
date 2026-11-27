//! Overlay pass: album artwork and song text drawn over the scene at native resolution.
//!
//! Resource policy: textures are created only when new content arrives (a track change), sized exactly to
//! the content (artwork is already bounded by the decoder; text bitmaps by mg-text), and replaced or
//! dropped immediately when the content goes away. Nothing is allocated or uploaded per frame; a frame
//! only rewrites one 128-byte uniform and records one draw.

const UNIFORM_BYTES: usize = 21 * 16;
const CONTROLS_BYTES: usize = 34 * 16;
/// Text slots: 0 title, 1 artist, 2..=4 lyric lines, 5 elapsed time, 6 duration (control bar), 7 meta line.
pub const TEXT_SLOTS: usize = 8;
/// First text slot of the control bar: `CLOCK_SLOT` is the elapsed time, `CLOCK_SLOT + 1` the duration.
pub const CLOCK_SLOT: usize = 5;
/// Text slot of the meta line (version / remaster note under the artist).
pub const META_SLOT: usize = 7;
/// Largest texture side accepted (defence in depth; callers already bound their content).
const MAX_TEXTURE_SIDE: u32 = 4096;

/// Scene-panel atlas layout (binding 13, one Rgba8UnormSrgb texture): thumbnail cells (3 columns, 2 rows of
/// `THUMB_SIZE`, cell = `SceneKind::index()`) above label cells (3 columns, 3 rows of `LABEL_SIZE`, white with
/// the glyph coverage in alpha).
pub const THUMB_SIZE: (u32, u32) = (256, 144);
pub const THUMB_CELLS: usize = 5;
pub const LABEL_SIZE: (u32, u32) = (256, 48);
pub const LABEL_CELLS: usize = 8;
pub const LABEL_ORIGIN_Y: u32 = 2 * THUMB_SIZE.1;
pub const ATLAS_SIZE: (u32, u32) = (3 * THUMB_SIZE.0, LABEL_ORIGIN_Y + 3 * LABEL_SIZE.1);

/// Artwork placement in surface pixels. `(cx, cy)` is the centre; `hw`/`hh` the half extents (the caller
/// preserves the image aspect ratio when computing them). The plate (rounded corners, 1 px inner stroke, contact
/// and ambient shadow) is drawn by the shader; nothing about it reacts to the music.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtDraw {
    pub cx: f32,
    pub cy: f32,
    pub hw: f32,
    pub hh: f32,
    pub alpha: f32,
    pub radius: f32,
    /// Shadow strength 0..1 (1 = the spec's contact 0.55 + ambient 0.45 alphas).
    pub shadow: f32,
    /// Sampling LOD of the blur bridge during a crossfade (0 = sharp; the cover texture carries the mips it needs).
    pub lod: f32,
}

/// Linear-RGB colour per text role (album-tinted on the CPU) plus the ambient shadow tint.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TextColors {
    pub title: [f32; 3],
    pub artist: [f32; 3],
    pub meta: [f32; 3],
    /// Current lyric line.
    pub lyric: [f32; 3],
    /// Context lyric lines (next / previous).
    pub context: [f32; 3],
    /// Ambient artwork shadow.
    pub shade: [f32; 3],
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
            // the focused backing: hover thickness 3s + pad 7s + 3rw (rw = max(1, s), see `fs_controls`), +1 px
            let reach = 10.0 * s + 3.0 * s.max(1.0) + 1.0;
            x0 = x0.min(p.x0 - reach);
            x1 = x1.max(p.x1 + reach);
            y0 = y0.min(p.y - reach);
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

/// Tiles the scene panel can draw (one per scene).
pub const PANEL_TILES: usize = THUMB_CELLS;
/// Label cells with a fixed meaning in the atlas (cells 0..4 are the scene labels, `SceneKind::index()`).
pub const LABEL_CURRENT: usize = 5;
pub const LABEL_EXPERIMENTAL_PILL: usize = 6;
pub const LABEL_EXPERIMENTAL_HEADING: usize = 7;
const PANEL_BYTES: usize = (5 + PANEL_TILES * 6) * 16;

/// One scene tile, final surface pixels. `(cx, cy, hw, hh)` is the thumbnail rectangle after the CPU applied
/// the hover scale and the rise; text rectangles are integer-aligned and drawn 1:1 from the atlas label cells.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PanelTileDraw {
    pub rect: [f32; 4],
    pub alpha: f32,
    /// 0..1 hover amount (brightness 0.80 -> 1.0, stroke 0.12 -> 0.35).
    pub hover: f32,
    /// This is the current scene: accent outline and the "Current" pill.
    pub selected: bool,
    /// Keyboard focus ring (white ring with a black keyline, so it reads on any scene).
    pub focus: bool,
    pub experimental: bool,
    /// The thumbnail exists in the atlas (cell = scene index); false draws a clean placeholder.
    pub has_thumb: bool,
    /// Atlas cell of the thumbnail and of the scene label (`SceneKind::index()`).
    pub cell: usize,
    /// Label text rectangle (left, top, w, h); w 0 = no label yet.
    pub label: [f32; 4],
    /// Pill rectangles (cx, cy, half w, half h): "Current" top-left, "EXPERIMENTAL" top-right.
    pub cur_pill: [f32; 4],
    pub exp_pill: [f32; 4],
}

/// The scene panel (a third, full-surface draw of the overlay pass; every pixel is computed from this).
#[derive(Clone, Debug, PartialEq)]
pub struct PanelDraw {
    /// Scrim opacity multiplier 0..1 (the draw is skipped at 0).
    pub alpha: f32,
    /// Pixels per design pixel.
    pub scale: f32,
    /// Album/theme accent (linear RGB) of the selection outline and the "Current" pill.
    pub accent: [f32; 3],
    pub count: usize,
    pub tiles: [PanelTileDraw; PANEL_TILES],
    /// "EXPERIMENTAL" heading text rectangle (left, top, w, h); w 0 = none.
    pub heading: [f32; 4],
    pub heading_alpha: f32,
    /// Bitmap sizes (w, h) of the pill texts, centred in their pills by the shader.
    pub cur_text: [f32; 2],
    pub exp_text: [f32; 2],
}

impl PanelDraw {
    /// Packs the WGSL `P` layout (public for tests).
    pub fn pack(&self, surface: (u32, u32)) -> Vec<u8> {
        let mut o = Vec::with_capacity(PANEL_BYTES);
        put(
            &mut o,
            &[
                surface.0 as f32,
                surface.1 as f32,
                self.alpha.clamp(0.0, 1.0),
                self.scale.max(0.01),
            ],
        );
        put(
            &mut o,
            &[
                self.accent[0],
                self.accent[1],
                self.accent[2],
                self.heading_alpha.clamp(0.0, 1.0),
            ],
        );
        put(
            &mut o,
            &[
                self.cur_text[0].max(0.0),
                self.cur_text[1].max(0.0),
                self.exp_text[0].max(0.0),
                self.exp_text[1].max(0.0),
            ],
        );
        put(&mut o, &self.heading.map(|v| v.max(0.0)));
        put(&mut o, &[self.count.min(PANEL_TILES) as f32, 0.0, 0.0, 0.0]);
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        for t in &self.tiles {
            put(&mut o, &t.rect.map(|v| v.max(0.0)));
            put(
                &mut o,
                &[
                    t.alpha.clamp(0.0, 1.0),
                    t.hover.clamp(0.0, 1.0),
                    flag(t.selected),
                    flag(t.focus),
                ],
            );
            put(
                &mut o,
                &[
                    flag(t.experimental),
                    flag(t.has_thumb),
                    t.cell.min(THUMB_CELLS - 1) as f32,
                    0.0,
                ],
            );
            put(&mut o, &t.label.map(|v| v.max(0.0)));
            put(&mut o, &t.cur_pill.map(|v| v.max(0.0)));
            put(&mut o, &t.exp_pill.map(|v| v.max(0.0)));
        }
        debug_assert_eq!(o.len(), PANEL_BYTES);
        o
    }

    /// The panel dims the whole picture, so its scissor is the whole surface (never empty).
    pub fn scissor(&self, surface: (u32, u32)) -> (u32, u32, u32, u32) {
        (0, 0, surface.0.max(1), surface.1.max(1))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OverlayParams {
    pub res: (u32, u32),
    pub time: f32,
    pub energy: f32,
    /// Current artwork, and the previous one while the track-change crossfade runs.
    pub art: Option<ArtDraw>,
    pub art_prev: Option<ArtDraw>,
    pub title: Option<TextDraw>,
    pub artist: Option<TextDraw>,
    pub meta: Option<TextDraw>,
    /// Lyric lines (slot 2, 3, 4 of the text textures) and their slow glow boost (0..1).
    pub lyrics: [Option<TextDraw>; 3],
    /// Per lyric slot: 0 = context colour, 1 = current-line colour.
    pub lyric_emph: [f32; 3],
    /// Lyric column mask: top, bottom, feather (px). Ink outside fades to nothing.
    pub lyric_clip: Option<[f32; 3]>,
    pub lyric_glow: f32,
    /// Shadow strength under the text (0..1).
    pub shadow: f32,
    pub colors: TextColors,
    /// Album accent (linear RGB) for the control bar.
    pub accent: [f32; 3],
    /// The control bar when visible (None or alpha 0 = nothing drawn, zero cost).
    pub controls: Option<ControlsDraw>,
    /// The scene panel when open or fading (None or alpha 0 = nothing drawn, zero cost).
    pub panel: Option<PanelDraw>,
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
        for art in [&self.art, &self.art_prev] {
            match art {
                Some(a) => {
                    put(&mut o, &[a.cx, a.cy, a.hw.max(0.0), a.hh.max(0.0)]);
                    put(
                        &mut o,
                        &[
                            a.alpha.clamp(0.0, 1.0),
                            a.radius.max(0.0),
                            a.shadow.clamp(0.0, 1.0),
                            a.lod.clamp(0.0, 8.0),
                        ],
                    );
                }
                None => put(&mut o, &[0.0; 8]),
            }
        }
        let rgb = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
        put(&mut o, &rgb(self.colors.shade, 0.0));
        let text = |t: &Option<TextDraw>| match t {
            Some(t) => [t.x, t.y, t.w.max(0.0), t.h.max(0.0)],
            None => [0.0; 4],
        };
        put(&mut o, &text(&self.title));
        put(&mut o, &text(&self.artist));
        put(&mut o, &text(&self.meta));
        let al = |t: &Option<TextDraw>| t.map_or(0.0, |t| t.alpha.clamp(0.0, 1.0));
        put(
            &mut o,
            &[
                al(&self.title),
                al(&self.artist),
                al(&self.meta),
                self.shadow.clamp(0.0, 1.0),
            ],
        );
        put(&mut o, &rgb(self.colors.title, 0.0));
        put(&mut o, &rgb(self.colors.artist, 0.0));
        put(&mut o, &rgb(self.colors.meta, 0.0));
        put(
            &mut o,
            &rgb(self.colors.lyric, self.lyric_glow.clamp(0.0, 1.0)),
        );
        put(&mut o, &rgb(self.colors.context, 0.0));
        for l in &self.lyrics {
            put(&mut o, &text(l));
        }
        put(
            &mut o,
            &[
                al(&self.lyrics[0]),
                al(&self.lyrics[1]),
                al(&self.lyrics[2]),
                0.0,
            ],
        );
        let e = self.lyric_emph.map(|v| v.clamp(0.0, 1.0));
        put(&mut o, &[e[0], e[1], e[2], 0.0]);
        match self.lyric_clip {
            Some([y0, y1, f]) => put(&mut o, &[y0, y1, f.max(0.0), 1.0]),
            None => put(&mut o, &[0.0; 4]),
        }
        debug_assert_eq!(o.len(), UNIFORM_BYTES);
        o
    }

    /// The control bar, when it would change pixels.
    fn visible_controls(&self) -> Option<&ControlsDraw> {
        self.controls.as_ref().filter(|c| c.alpha > 0.002)
    }

    /// The scene panel, when it would change pixels.
    fn visible_panel(&self) -> Option<&PanelDraw> {
        self.panel.as_ref().filter(|p| p.alpha > 0.002)
    }

    /// True when the artwork/text part would change no pixel (nothing present or fully transparent).
    pub fn is_empty(&self) -> bool {
        let vis = |a: f32| a > 0.002;
        !([self.art, self.art_prev]
            .iter()
            .any(|a| a.is_some_and(|a| vis(a.alpha)))
            || [self.title, self.artist, self.meta]
                .iter()
                .chain(&self.lyrics)
                .any(|l| l.is_some_and(|t| vis(t.alpha))))
    }
}

pub struct Overlay {
    pipeline: wgpu::RenderPipeline,
    controls_pipeline: wgpu::RenderPipeline,
    panel_pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    controls_uniform: wgpu::Buffer,
    panel_uniform: wgpu::Buffer,
    /// Current artwork and the previous one (crossfade source), see `shift_art`.
    art: Option<wgpu::TextureView>,
    art_prev: Option<wgpu::TextureView>,
    texts: [Option<wgpu::TextureView>; TEXT_SLOTS],
    /// Scene-panel atlas (thumbnails above, label coverage in alpha below), created on first upload and kept.
    atlas: Option<wgpu::Texture>,
    atlas_view: Option<wgpu::TextureView>,
    atlas_creations: u32,
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
    size: (u32, u32),
    bytes_per_px: u32,
    data: &[u8],
) -> Option<wgpu::TextureView> {
    make_texture_mips(device, queue, label, format, size, bytes_per_px, data, &[])
}

/// `make_texture` with extra mip levels 1.. (each `max(1, w >> level)` x `max(1, h >> level)`, same format). Any
/// level of the wrong length refuses the whole texture (nothing half-uploaded).
#[allow(clippy::too_many_arguments)]
fn make_texture_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    format: wgpu::TextureFormat,
    (w, h): (u32, u32),
    bytes_per_px: u32,
    data: &[u8],
    mips: &[Vec<u8>],
) -> Option<wgpu::TextureView> {
    let level_size = |l: u32| ((w >> l).max(1), (h >> l).max(1));
    if w == 0
        || h == 0
        || w > MAX_TEXTURE_SIDE
        || h > MAX_TEXTURE_SIDE
        || data.len() != (w * h * bytes_per_px) as usize
        || mips.len() >= 16
        || (w.max(h) >> mips.len()) == 0
        || mips.iter().enumerate().any(|(i, m)| {
            let (mw, mh) = level_size(i as u32 + 1);
            m.len() != (mw * mh * bytes_per_px) as usize
        })
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
        mip_level_count: 1 + mips.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, bytes) in std::iter::once(data)
        .chain(mips.iter().map(|m| m.as_slice()))
        .enumerate()
    {
        let (lw, lh) = level_size(level as u32);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(lw * bytes_per_px),
                rows_per_image: Some(lh),
            },
            wgpu::Extent3d {
                width: lw,
                height: lh,
                depth_or_array_layers: 1,
            },
        );
    }
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
                tex_entry(11),
                tex_entry(12),
                tex_entry(13),
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PANEL_BYTES as u64),
                    },
                    count: None,
                },
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
        let panel_pipeline = make_pipeline("overlay panel", "fs_panel");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // trilinear: only the cover (the sole texture with mips) reacts, for the blur bridge
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
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
        let panel_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay panel"),
            size: PANEL_BYTES as u64,
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
                [&uniform, &controls_uniform, &panel_uniform],
                [&blank_rgba, &blank_rgba],
                [&blank_r8; TEXT_SLOTS],
                &blank_rgba,
                &sampler,
            ),
            pipeline,
            controls_pipeline,
            panel_pipeline,
            bgl,
            sampler,
            uniform,
            controls_uniform,
            panel_uniform,
            art: None,
            art_prev: None,
            texts: Default::default(),
            atlas: None,
            atlas_view: None,
            atlas_creations: 0,
            blank_rgba,
            blank_r8,
        };
        s.rebuild(device);
        s
    }

    fn group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniforms: [&wgpu::Buffer; 3],
        art: [&wgpu::TextureView; 2],
        texts: [&wgpu::TextureView; TEXT_SLOTS],
        atlas: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(atlas),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(texts[META_SLOT]),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(art[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: uniforms[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: uniforms[2].as_entire_binding(),
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
                    resource: wgpu::BindingResource::TextureView(art[0]),
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
            [&self.uniform, &self.controls_uniform, &self.panel_uniform],
            [
                self.art.as_ref().unwrap_or(&self.blank_rgba),
                self.art_prev.as_ref().unwrap_or(&self.blank_rgba),
            ],
            std::array::from_fn(|i| self.texts[i].as_ref().unwrap_or(&self.blank_r8)),
            self.atlas_view.as_ref().unwrap_or(&self.blank_rgba),
            &self.sampler,
        );
    }

    /// Crossfade bookkeeping: the current artwork becomes the previous one (the old previous one is dropped) and
    /// the current slot is empty until `set_art`.
    pub fn shift_art(&mut self, device: &wgpu::Device) {
        self.art_prev = self.art.take();
        self.rebuild(device);
    }

    /// Drops the previous artwork (the crossfade finished).
    pub fn clear_prev_art(&mut self, device: &wgpu::Device) {
        self.art_prev = None;
        self.rebuild(device);
    }

    /// Replaces the current artwork (straight RGBA8, `w x h`). Returns false (and clears it) when the data
    /// is malformed or too large.
    pub fn set_art(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        rgba: &[u8],
    ) -> bool {
        self.set_art_mips(device, queue, size, rgba, &[])
    }

    /// `set_art` with mip levels 1.. (CPU box-filtered by the presentation worker, uploaded together with the
    /// picture, once per cover). The texture lives until the cover is cleared or replaced. A malformed level
    /// refuses the cover.
    pub fn set_art_mips(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        rgba: &[u8],
        mips: &[Vec<u8>],
    ) -> bool {
        self.art = make_texture_mips(
            device,
            queue,
            "art",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            size,
            4,
            rgba,
            mips,
        );
        self.rebuild(device);
        self.art.is_some()
    }

    pub fn clear_art(&mut self, device: &wgpu::Device) {
        self.art = None;
        self.rebuild(device);
    }

    /// Creates the panel atlas on first use (one `ATLAS_SIZE` texture for the life of the overlay; later uploads
    /// write sub-rectangles).
    fn ensure_atlas(&mut self, device: &wgpu::Device) -> &wgpu::Texture {
        if self.atlas.is_none() {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("panel atlas"),
                size: wgpu::Extent3d {
                    width: ATLAS_SIZE.0,
                    height: ATLAS_SIZE.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.atlas_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.atlas = Some(tex);
            self.atlas_creations += 1;
            self.rebuild(device);
        }
        self.atlas.as_ref().expect("created above")
    }

    /// Times the panel atlas texture was created (1 after the first upload, never more; tests).
    pub fn atlas_creations(&self) -> u32 {
        self.atlas_creations
    }

    fn write_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        origin: (u32, u32),
        size: (u32, u32),
        rgba: &[u8],
    ) {
        let tex = self.ensure_atlas(device);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: origin.0,
                    y: origin.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.0 * 4),
                rows_per_image: Some(size.1),
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Writes the thumbnail of scene `index` (`SceneKind::index()`; straight sRGB RGBA8, `THUMB_SIZE`) into its
    /// atlas cell. False (nothing written) for a wrong index or length.
    pub fn set_panel_thumb(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        index: usize,
        rgba: &[u8],
    ) -> bool {
        if index >= THUMB_CELLS || rgba.len() != (THUMB_SIZE.0 * THUMB_SIZE.1 * 4) as usize {
            return false;
        }
        let o = (
            (index as u32 % 3) * THUMB_SIZE.0,
            (index as u32 / 3) * THUMB_SIZE.1,
        );
        self.write_atlas(device, queue, o, THUMB_SIZE, rgba);
        true
    }

    /// Writes label `cell` (< `LABEL_CELLS`) from a coverage bitmap (`w x h`, at most `LABEL_SIZE`) into the
    /// atlas: white with the coverage in alpha (alpha is not sRGB-decoded). The rest of the cell is cleared.
    pub fn set_panel_label(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cell: usize,
        size: (u32, u32),
        coverage: &[u8],
    ) -> bool {
        if cell >= LABEL_CELLS
            || size.0 == 0
            || size.1 == 0
            || size.0 > LABEL_SIZE.0
            || size.1 > LABEL_SIZE.1
            || coverage.len() != (size.0 * size.1) as usize
        {
            return false;
        }
        let mut px = vec![0u8; (LABEL_SIZE.0 * LABEL_SIZE.1 * 4) as usize];
        for y in 0..size.1 {
            for x in 0..size.0 {
                let o = ((y * LABEL_SIZE.0 + x) * 4) as usize;
                px[o..o + 3].fill(255);
                px[o + 3] = coverage[(y * size.0 + x) as usize];
            }
        }
        let o = (
            (cell as u32 % 3) * LABEL_SIZE.0,
            LABEL_ORIGIN_Y + (cell as u32 / 3) * LABEL_SIZE.1,
        );
        self.write_atlas(device, queue, o, LABEL_SIZE, &px);
        true
    }

    /// Replaces text slot `which` (see `TEXT_SLOTS`) with a coverage bitmap (8-bit, `w x h`).
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
        let panel = p.visible_panel();
        if !main && controls.is_none() && panel.is_none() {
            return;
        }
        if main {
            queue.write_buffer(&self.uniform, 0, &p.pack());
        }
        if let Some(pn) = panel {
            queue.write_buffer(&self.panel_uniform, 0, &pn.pack(p.res));
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
        if let Some(pn) = panel {
            // the panel dims the whole picture: a full-surface draw, under the control bar
            let (x, y, w, h) = pn.scissor(p.res);
            pass.set_scissor_rect(x, y, w, h);
            pass.set_pipeline(&self.panel_pipeline);
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
                shadow: 0.0,
                lod: 0.0,
            }),
            lyric_emph: [1.0; 3],
            shadow: 0.5,
            colors: TextColors {
                title: [1.0; 3],
                artist: [1.0; 3],
                meta: [1.0; 3],
                lyric: [1.0; 3],
                context: [0.5; 3],
                shade: [0.0; 3],
            },
            accent: [0.2, 0.4, 1.0],
            ..Default::default()
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

    fn panel_tile(i: usize, selected: bool) -> PanelTileDraw {
        let cx = 90.0 + 100.0 * i as f32;
        PanelTileDraw {
            rect: [cx, 140.0, 40.0, 22.5],
            alpha: 1.0,
            selected,
            has_thumb: true,
            cell: i,
            label: [cx - 20.0, 175.0, 40.0, 16.0],
            cur_pill: [cx - 25.0, 125.0, 14.0, 6.0],
            exp_pill: [cx + 20.0, 125.0, 14.0, 6.0],
            ..Default::default()
        }
    }

    fn panel_draw(alpha: f32) -> PanelDraw {
        let mut tiles = [PanelTileDraw::default(); PANEL_TILES];
        for (i, t) in tiles.iter_mut().enumerate() {
            *t = panel_tile(i, i == 1);
        }
        tiles[3].experimental = true;
        tiles[4].has_thumb = false;
        PanelDraw {
            alpha,
            scale: 0.5,
            accent: [0.9, 0.3, 0.1],
            count: PANEL_TILES,
            tiles,
            heading: [20.0, 205.0, 40.0, 10.0],
            heading_alpha: 1.0,
            cur_text: [20.0, 8.0],
            exp_text: [20.0, 8.0],
        }
    }

    fn srgb_to_linear(c: u8) -> f32 {
        let c = f32::from(c) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    #[test]
    fn panel_pack_matches_wgsl_and_the_atlas_constants_are_shared() {
        let d = panel_draw(1.0);
        assert_eq!(d.pack((640, 360)).len(), PANEL_BYTES);
        let src = include_str!("overlay.wgsl");
        assert!(src.contains("t: array<vec4<f32>, 30>") && src.contains("fn fs_panel"));
        assert!(src.contains(&format!(
            "const ATLAS_SIZE = vec2<f32>({}.0, {}.0);",
            ATLAS_SIZE.0, ATLAS_SIZE.1
        )));
        assert!(src.contains(&format!(
            "const THUMB_SIZE = vec2<f32>({}.0, {}.0);",
            THUMB_SIZE.0, THUMB_SIZE.1
        )));
        assert!(src.contains(&format!("const LABEL_ORIGIN_Y = {LABEL_ORIGIN_Y}.0;")));
        assert_eq!((LABEL_SIZE.0, LABEL_SIZE.1), (256, 48), "shader cell size");
        assert_eq!(PANEL_BYTES, (5 + 30) * 16);
        assert_eq!(d.scissor((640, 360)), (0, 0, 640, 360));
        assert_eq!(d.scissor((0, 0)), (0, 0, 1, 1));
        let mut h = d.clone();
        h.alpha = f32::NAN;
        h.tiles[0].rect = [f32::INFINITY; 4];
        assert!(
            h.pack((640, 360))
                .chunks_exact(4)
                .all(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]).is_finite())
        );
    }

    #[test]
    fn the_panel_dims_the_picture_shows_tiles_and_labels_and_alpha_zero_draws_nothing_headless() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        let res = (640u32, 360u32);
        let scene = SceneSet::new_headless(&h.device);
        let mut ov = Overlay::new_headless(&h.device, &h.queue);
        let colors: [[u8; 3]; 5] = [
            [200, 60, 40],
            [40, 200, 60],
            [40, 60, 200],
            [200, 200, 40],
            [200, 40, 200],
        ];
        for (i, c) in colors.iter().enumerate() {
            let px: Vec<u8> = (0..THUMB_SIZE.0 * THUMB_SIZE.1)
                .flat_map(|_| [c[0], c[1], c[2], 255])
                .collect();
            assert!(ov.set_panel_thumb(&h.device, &h.queue, i, &px));
        }
        for cell in 0..LABEL_CELLS {
            assert!(ov.set_panel_label(&h.device, &h.queue, cell, (40, 16), &[255; 40 * 16]));
        }
        let p = SceneParams::default();
        let render = |panel: Option<PanelDraw>| {
            let op = OverlayParams {
                res,
                panel,
                ..Default::default()
            };
            scene
                .render_offscreen_with(&h.device, &h.queue, res, &p, &|enc, view| {
                    ov.draw(&h.queue, enc, view, &op)
                })
                .expect("render")
        };
        let bare = render(None);
        assert_eq!(render(Some(panel_draw(0.0))), bare, "alpha 0 draws nothing");
        let img = render(Some(panel_draw(1.0)));
        // scrim: black 0.45 at the top, so the linear value falls to 0.55 of the scene
        let (b, a) = (px(&bare, res.0, 5, 5), px(&img, res.0, 5, 5));
        for k in 0..3 {
            let want = f32::from(b[k]) * 0.55;
            assert!(
                (f32::from(a[k]) - want).abs() <= 3.0,
                "scrim channel {k}: {} vs {want}",
                a[k]
            );
        }
        // tile 0 centre: its thumbnail at rest brightness 0.80 (linear value, the headless target is linear)
        let t = px(&img, res.0, 90, 140);
        for k in 0..3 {
            let want = srgb_to_linear(colors[0][k]) * 0.80 * 255.0;
            assert!(
                (f32::from(t[k]) - want).abs() <= 4.0,
                "tile channel {k}: {} vs {want}",
                t[k]
            );
        }
        // the selected tile (1) shows a 2 px accent outline just inside its edge
        let e = px(&img, res.0, 190 - 40, 140);
        assert!(e[0] > 150 && e[0] > e[2] * 2, "accent outline, got {e:?}");
        // the missing thumbnail (tile 4) is the neutral placeholder, not another tile's colour
        let m = px(&img, res.0, 490, 140);
        assert!(
            m.iter().take(3).all(|v| *v < 60) && m[0].abs_diff(m[2]) < 12,
            "placeholder, got {m:?}"
        );
        // a label (white coverage box) is drawn below tile 0
        let l = px(&img, res.0, 90, 183);
        assert!(l[0] > 150 && l[1] > 150 && l[2] > 150, "label, got {l:?}");
        // far from every tile only the scrim changed (the draw stays where it belongs)
        let far = px(&img, res.0, 600, 20);
        let far_b = px(&bare, res.0, 600, 20);
        assert!((f32::from(far[0]) - f32::from(far_b[0]) * 0.55).abs() <= 3.0);
        // keyboard focus: a white ring 3 px outside the tile that the unfocused draw does not have
        let mut d = panel_draw(1.0);
        d.tiles[0].focus = true;
        let f = render(Some(d));
        let ring = px(&f, res.0, 90 - 40 - 2, 140);
        assert!(
            f != img && ring != px(&img, res.0, 90 - 40 - 2, 140),
            "focus ring drawn"
        );
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
    fn panel_atlas_is_created_once_and_refuses_malformed_uploads_headless() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        let mut ov = Overlay::new_headless(&h.device, &h.queue);
        assert_eq!(
            ov.atlas_creations(),
            0,
            "nothing allocated until the first upload"
        );
        let thumb = vec![200u8; (THUMB_SIZE.0 * THUMB_SIZE.1 * 4) as usize];
        assert!(!ov.set_panel_thumb(&h.device, &h.queue, THUMB_CELLS, &thumb));
        assert!(!ov.set_panel_thumb(&h.device, &h.queue, 0, &thumb[..8]));
        assert_eq!(ov.atlas_creations(), 0, "refused uploads allocate nothing");
        let cov = vec![255u8; 40 * 20];
        assert!(!ov.set_panel_label(&h.device, &h.queue, LABEL_CELLS, (40, 20), &cov));
        assert!(!ov.set_panel_label(&h.device, &h.queue, 0, (LABEL_SIZE.0 + 1, 1), &cov));
        assert!(!ov.set_panel_label(&h.device, &h.queue, 0, (40, 20), &cov[..5]));
        for _ in 0..50 {
            for i in 0..THUMB_CELLS {
                assert!(ov.set_panel_thumb(&h.device, &h.queue, i, &thumb));
            }
            for c in 0..LABEL_CELLS {
                assert!(ov.set_panel_label(&h.device, &h.queue, c, (40, 20), &cov));
            }
        }
        assert_eq!(
            ov.atlas_creations(),
            1,
            "one atlas for the life of the overlay"
        );
    }

    /// The scissor covers the focused progress backing (track half thickness at hover + pad 7s + 3rw, rw = max(1, s))
    /// at every UI scale, including below 1 where the ring's 1 px floor outgrows multiples of s.
    #[test]
    fn the_scissor_covers_the_focused_progress_backing_at_every_scale() {
        for s in [0.5f32, 0.75, 1.0, 1.5, 2.0] {
            let res = (1920u32, 1080u32);
            let mut c = bar(res);
            c.scale = s;
            c.bar = [960.0, 900.0, 280.0 * s, 32.0 * s];
            c.progress = Some(ProgressDraw {
                x0: 700.0,
                x1: 1220.0,
                y: 800.0,
                half_thick: 3.0 * s,
                frac: 0.4,
                hover: 1.0,
                focus: true,
            });
            let (x, y, w, _) = c.scissor(res);
            let reach = 3.0 * s + 7.0 * s + 3.0 * s.max(1.0); // beyond the track centre line / the track end
            assert!(
                y as f32 <= 800.0 - reach,
                "scale {s}: top {y} vs {}",
                800.0 - reach
            );
            assert!(x as f32 <= 700.0 - (reach - 3.0 * s), "scale {s}: left {x}");
            assert!(
                (x + w) as f32 >= 1220.0 + (reach - 3.0 * s),
                "scale {s}: right {}",
                x + w
            );
        }
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
        // the WGSL struct O: first and last field, and no halo left anywhere in the shader
        assert!(src.contains("res: vec4<f32>,       // surface w, h, time, energy"));
        assert!(src.contains("lyr_clip: vec4<f32>,") && src.contains("text_a: vec4<f32>"));
        assert!(!src.contains("halo"));
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
    fn crossfade_draws_the_new_cover_over_the_held_previous_one_and_meta_has_its_slot_headless() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        let res = (320u32, 180u32);
        let scene = SceneSet::new_headless(&h.device);
        let mut ov = Overlay::new_headless(&h.device, &h.queue);
        let solid = |c: [u8; 4]| (0..16).flat_map(|_| c).collect::<Vec<u8>>();
        assert!(ov.set_art(&h.device, &h.queue, (4, 4), &solid([255, 0, 0, 255])));
        ov.shift_art(&h.device); // red becomes the previous cover
        assert!(ov.set_art(&h.device, &h.queue, (4, 4), &solid([0, 0, 255, 255])));
        assert!(ov.set_text(&h.device, &h.queue, META_SLOT, (40, 10), &[255u8; 400]));
        let sp = SceneParams::default();
        let render = |ov: &Overlay, p: &OverlayParams| {
            scene
                .render_offscreen_with(&h.device, &h.queue, res, &sp, &|enc, view| {
                    ov.draw(&h.queue, enc, view, p)
                })
                .unwrap()
        };
        let mut p = params(res);
        p.art.as_mut().unwrap().hh = 40.0;
        p.art.as_mut().unwrap().alpha = 0.5;
        p.art_prev = p.art.map(|a| ArtDraw { alpha: 1.0, ..a });
        p.meta = Some(TextDraw {
            x: 10.0,
            y: 10.0,
            w: 40.0,
            h: 10.0,
            alpha: 1.0,
        });
        let img = render(&ov, &p);
        let mid = px(&img, res.0, 160, 90);
        assert!(
            mid[0] > 90 && mid[2] > 90 && mid[1] < 40,
            "half blue over full red, no dip to the scene: {mid:?}"
        );
        assert!(px(&img, res.0, 30, 15)[0] > 230, "meta line drawn");
        ov.clear_prev_art(&h.device);
        let img = render(&ov, &p);
        assert!(px(&img, res.0, 160, 90)[0] < 60, "previous cover dropped");
    }

    #[test]
    fn cover_mips_are_uploaded_with_the_picture_and_sampled_by_the_blur_bridge_headless() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        let res = (320u32, 180u32);
        let scene = SceneSet::new_headless(&h.device);
        let mut ov = Overlay::new_headless(&h.device, &h.queue);
        // 8x8 black/white checker, its 4x4 and 2x2 levels flat grey
        let checker: Vec<u8> = (0..64)
            .flat_map(|i| {
                let v = if (i % 8 + i / 8) % 2 == 0 { 255 } else { 0 };
                [v, v, v, 255]
            })
            .collect();
        let grey = |n: usize| {
            (0..n)
                .flat_map(|_| [188u8, 188, 188, 255])
                .collect::<Vec<u8>>()
        };
        let mips = [grey(16), grey(4)];
        // a malformed level refuses the cover instead of uploading half of it
        assert!(!ov.set_art_mips(&h.device, &h.queue, (8, 8), &checker, &[grey(15)]));
        assert!(!ov.set_art_mips(
            &h.device,
            &h.queue,
            (8, 8),
            &checker,
            &[grey(16), grey(4), grey(1), grey(1)]
        ));
        assert!(ov.set_art_mips(&h.device, &h.queue, (8, 8), &checker, &mips));
        let sp = SceneParams::default();
        let render = |ov: &Overlay, lod: f32| {
            let mut p = params(res);
            let a = p.art.as_mut().unwrap();
            (a.hw, a.hh, a.lod) = (32.0, 32.0, lod);
            scene
                .render_offscreen_with(&h.device, &h.queue, res, &sp, &|enc, view| {
                    ov.draw(&h.queue, enc, view, &p)
                })
                .unwrap()
        };
        // two neighbouring texel centres of the 64 px cover (8 px per texel)
        let (sharp, soft) = (render(&ov, 0.0), render(&ov, 2.0));
        let (a, b) = (px(&sharp, res.0, 132, 70)[0], px(&sharp, res.0, 140, 70)[0]);
        assert!(a.abs_diff(b) > 150, "lod 0 is the picture: {a} vs {b}");
        let (a, b) = (px(&soft, res.0, 132, 70)[0], px(&soft, res.0, 140, 70)[0]);
        assert!(
            a.abs_diff(b) < 4 && a.abs_diff(128) < 6,
            "lod 2 is the mip: {a} vs {b}"
        );
        // a cover without mips ignores the lod (nothing to bridge with, nothing broken)
        assert!(ov.set_art(&h.device, &h.queue, (8, 8), &checker));
        let flat = render(&ov, 2.0);
        assert!(px(&flat, res.0, 132, 70)[0].abs_diff(px(&flat, res.0, 140, 70)[0]) > 150);
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
