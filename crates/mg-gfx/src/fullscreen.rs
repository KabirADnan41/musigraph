//! Generic fullscreen-triangle fragment-shader renderer used by every scene (stress benchmark, spectrum).
//!
//! * One uniform buffer (caller-defined layout, fixed size) bound at group 0 / binding 0.
//! * Render scale: the swap chain always matches the window (some backends reject a smaller surface).
//!   At scale < 1.0 the scene is drawn into an offscreen target of `scale x` the surface size and then
//!   upscaled with a bilinear blit. Scale is a pure cost knob: it never changes scene behaviour.
//! * `render_offscreen` renders to a texture and reads the pixels back (headless tests/measurements).

use crate::context::{FrameStatus, Gpu};
use std::time::{Duration, Instant};

struct Target {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    w: u32,
    h: u32,
    for_surface: (u32, u32),
}

pub struct FullscreenRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    uniform_len: usize,
    bind_group: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    scale: f32,
    target: Option<Target>,
    /// Time to create the shader modules and pipelines (the driver may compile later, on first use).
    pub compile_ms: f64,
}

impl FullscreenRenderer {
    /// `wgsl` must define entry points `vs` and `fs` and a uniform at `@group(0) @binding(0)`
    /// of exactly `uniform_len` bytes.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        scale: f32,
        label: &str,
        wgsl: &str,
        uniform_len: usize,
    ) -> Self {
        let t0 = Instant::now();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: uniform_len as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let pipeline = make_pipeline(device, &shader, &bgl, label, format);

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_pipeline = make_pipeline(device, &blit_shader, &blit_bgl, "blit", format);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            uniform,
            uniform_len,
            bind_group,
            blit_pipeline,
            blit_bgl,
            sampler,
            format,
            scale,
            target: None,
            compile_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Internal render resolution for the given surface size.
    pub fn render_size(&self, surface: (u32, u32)) -> (u32, u32) {
        scaled_size(surface, self.scale)
    }

    fn ensure_target(&mut self, device: &wgpu::Device, surface: (u32, u32)) {
        if self.scale >= 1.0
            || self
                .target
                .as_ref()
                .is_some_and(|t| t.for_surface == surface)
        {
            return;
        }
        let (w, h) = self.render_size(surface);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit-bg"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.target = Some(Target {
            view,
            bind_group,
            w,
            h,
            for_surface: surface,
        });
    }

    /// Size the shader's `res` uniform must describe for the current surface (the internal render size).
    pub fn current_render_size(&mut self, gpu: &Gpu) -> (u32, u32) {
        let surface = (gpu.config.width, gpu.config.height);
        self.ensure_target(&gpu.device, surface);
        match &self.target {
            Some(t) => (t.w, t.h),
            None => surface,
        }
    }

    /// Draws one frame with the caller-packed uniform (`uniform.len()` must equal the declared size).
    pub fn render(&mut self, gpu: &mut Gpu, uniform: &[u8]) -> Result<FrameStatus, String> {
        self.render_with(gpu, uniform, |_, _| {})
    }

    /// Like `render`, then lets `post` record more passes on the surface view (overlays) in the same submission.
    pub fn render_with<F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView)>(
        &mut self,
        gpu: &mut Gpu,
        uniform: &[u8],
        post: F,
    ) -> Result<FrameStatus, String> {
        if uniform.len() != self.uniform_len {
            return Err(format!(
                "uniform is {} bytes, shader expects {}",
                uniform.len(),
                self.uniform_len
            ));
        }
        self.ensure_target(&gpu.device, (gpu.config.width, gpu.config.height));
        gpu.queue.write_buffer(&self.uniform, 0, uniform);
        let this = &*self;
        gpu.render(|enc, surface_view| {
            let scene_view = this.target.as_ref().map_or(surface_view, |t| &t.view);
            let mut pass = begin_color_pass(enc, "scene", scene_view);
            pass.set_pipeline(&this.pipeline);
            pass.set_bind_group(0, &this.bind_group, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            if let Some(t) = &this.target {
                let mut blit = begin_color_pass(enc, "blit", surface_view);
                blit.set_pipeline(&this.blit_pipeline);
                blit.set_bind_group(0, &t.bind_group, &[]);
                blit.draw(0..3, 0..1);
            }
            post(enc, surface_view);
        })
    }

    /// Renders one frame into an offscreen texture of `w x h` and returns the tightly packed pixels
    /// (4 bytes per pixel in this renderer's texture format). For headless tests and measurements.
    pub fn render_offscreen(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        uniform: &[u8],
    ) -> Result<Vec<u8>, String> {
        self.render_offscreen_with(device, queue, w, h, uniform, &|_, _| {})
    }

    /// `render_offscreen` plus a `post` hook that records extra passes (overlays) before the readback.
    pub fn render_offscreen_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        uniform: &[u8],
        post: &dyn Fn(&mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) -> Result<Vec<u8>, String> {
        if uniform.len() != self.uniform_len {
            return Err("uniform size mismatch".into());
        }
        queue.write_buffer(&self.uniform, 0, uniform);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let row = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(row) * u64::from(h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen"),
        });
        {
            let mut pass = begin_color_pass(&mut enc, "offscreen", &view);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        post(&mut enc, &view);
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
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
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(10)),
            })
            .map_err(|e| format!("readback wait failed: {e}"))?;
        rx.recv_timeout(Duration::from_secs(10))
            .map_err(|e| format!("readback timed out: {e}"))?
            .map_err(|e| format!("readback map failed: {e}"))?;
        let data = buf
            .slice(..)
            .get_mapped_range()
            .map_err(|e| format!("readback range failed: {e}"))?;
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            let start = y * row as usize;
            out.extend_from_slice(&data[start..start + (w * 4) as usize]);
        }
        drop(data);
        buf.unmap();
        Ok(out)
    }
}

fn begin_color_pass<'e>(
    enc: &'e mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn make_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    label: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Internal render size for a surface size and scale (never zero).
pub fn scaled_size(surface: (u32, u32), scale: f32) -> (u32, u32) {
    let s = if scale.is_finite() {
        scale.clamp(0.25, 1.0)
    } else {
        1.0
    };
    (
        ((surface.0 as f32 * s).round() as u32).max(1),
        ((surface.1 as f32 * s).round() as u32).max(1),
    )
}
