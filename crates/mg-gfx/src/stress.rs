//! Stress-shader renderer (P3.3): a fullscreen fragment shader whose cost approximates the planned
//! P4 scene (haze noise, 64 bars, reflection, glow). It is not simplified to pass benchmarks.
//!
//! Render scale: the swap chain always matches the window (some backends reject a smaller surface).
//! At scale < 1.0 the scene is drawn into an offscreen target of `scale x` the surface size and then
//! upscaled to the surface with a bilinear blit, which is also how adaptive resolution will work in P4.

use crate::context::{FrameStatus, Gpu};
use std::time::Instant;

const UNIFORM_BYTES: usize = 16 + 16 * 16;

struct Target {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    w: u32,
    h: u32,
    for_surface: (u32, u32),
}

pub struct StressRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
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

impl StressRenderer {
    pub fn new(gpu: &Gpu, scale: f32) -> Self {
        let t0 = Instant::now();
        let format = gpu.config.format;
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("stress"),
                source: wgpu::ShaderSource::Wgsl(include_str!("stress.wgsl").into()),
            });
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stress-uniform"),
            size: UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("stress-bgl"),
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
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stress-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let pipeline = make_pipeline(gpu, &shader, &bgl, "stress", "vs", "fs", format);

        let blit_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("blit"),
                source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
            });
        let blit_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let blit_pipeline = make_pipeline(gpu, &blit_shader, &blit_bgl, "blit", "vs", "fs", format);
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            uniform,
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

    fn ensure_target(&mut self, gpu: &Gpu) {
        let surface = (gpu.config.width, gpu.config.height);
        if self.scale >= 1.0
            || self
                .target
                .as_ref()
                .is_some_and(|t| t.for_surface == surface)
        {
            return;
        }
        let (w, h) = self.render_size(surface);
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stress-target"),
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
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

    /// Draws one frame with synthetic, deterministic "bands" derived from `time`.
    pub fn render(&mut self, gpu: &mut Gpu, time: f32) -> Result<FrameStatus, String> {
        self.ensure_target(gpu);
        let (rw, rh) = match &self.target {
            Some(t) => (t.w, t.h),
            None => (gpu.config.width, gpu.config.height),
        };
        let bytes = uniform_bytes(rw as f32, rh as f32, time);
        gpu.queue.write_buffer(&self.uniform, 0, &bytes);
        let this = &*self;
        gpu.render(|enc, surface_view| {
            let scene_view = this.target.as_ref().map_or(surface_view, |t| &t.view);
            let mut pass = begin_color_pass(enc, "stress", scene_view);
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
        })
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
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    label: &str,
    vs: &str,
    fs: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(bgl)],
            immediate_size: 0,
        });
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vs),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fs),
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
