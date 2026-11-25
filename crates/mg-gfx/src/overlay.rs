//! Overlay pass: album artwork and song text drawn over the scene at native resolution.
//!
//! Resource policy: textures are created only when new content arrives (a track change), sized exactly to
//! the content (artwork is already bounded by the decoder; text bitmaps by mg-text), and replaced or
//! dropped immediately when the content goes away. Nothing is allocated or uploaded per frame; a frame
//! only rewrites one 128-byte uniform and records one draw.

const UNIFORM_BYTES: usize = 8 * 16;
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

#[derive(Clone, Debug, PartialEq)]
pub struct OverlayParams {
    pub res: (u32, u32),
    pub time: f32,
    pub energy: f32,
    pub art: Option<ArtDraw>,
    pub title: Option<TextDraw>,
    pub artist: Option<TextDraw>,
    /// Shadow strength under the text (0..1).
    pub shadow: f32,
    /// Linear RGB.
    pub text_color: [f32; 3],
    pub halo_color: [f32; 3],
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
        debug_assert_eq!(o.len(), UNIFORM_BYTES);
        o
    }

    /// True when drawing would change no pixel (nothing present or everything fully transparent).
    pub fn is_empty(&self) -> bool {
        let vis = |a: f32| a > 0.002;
        !(self.art.is_some_and(|a| vis(a.alpha))
            || self.title.is_some_and(|t| vis(t.alpha))
            || self.artist.is_some_and(|t| vis(t.alpha)))
    }
}

pub struct Overlay {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    art: Option<wgpu::TextureView>,
    title: Option<wgpu::TextureView>,
    artist: Option<wgpu::TextureView>,
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
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
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
        });
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
                &uniform,
                &blank_rgba,
                &blank_r8,
                &blank_r8,
                &sampler,
            ),
            pipeline,
            bgl,
            sampler,
            uniform,
            art: None,
            title: None,
            artist: None,
            blank_rgba,
            blank_r8,
        };
        s.rebuild(device);
        s
    }

    fn group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        art: &wgpu::TextureView,
        title: &wgpu::TextureView,
        artist: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(art),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(title),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(artist),
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
            &self.uniform,
            self.art.as_ref().unwrap_or(&self.blank_rgba),
            self.title.as_ref().unwrap_or(&self.blank_r8),
            self.artist.as_ref().unwrap_or(&self.blank_r8),
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

    /// Replaces the title (`which == 0`) or artist (`1`) coverage bitmap (8-bit, `w x h`).
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
        if which == 0 {
            self.title = t;
        } else {
            self.artist = t;
        }
        self.rebuild(device);
        ok
    }

    pub fn clear_text(&mut self, device: &wgpu::Device, which: usize) {
        if which == 0 {
            self.title = None;
        } else {
            self.artist = None;
        }
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
        if p.is_empty() {
            return;
        }
        queue.write_buffer(&self.uniform, 0, &p.pack());
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::init_headless;
    use crate::scene::{SceneParams, SpectrumScene};

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
            shadow: 0.5,
            text_color: [1.0, 1.0, 1.0],
            halo_color: [0.2, 0.4, 1.0],
        }
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
        let scene = SpectrumScene::new_headless(&h.device);
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
}
