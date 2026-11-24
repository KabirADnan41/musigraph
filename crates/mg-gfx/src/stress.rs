//! Stress-shader renderer (P3.3): a fullscreen fragment shader whose cost approximates the planned
//! P4 scene (haze noise, 64 bars, reflection, glow). It is not simplified to pass benchmarks.

use crate::context::Gpu;
use std::time::Instant;

const UNIFORM_BYTES: usize = 16 + 16 * 16;

pub struct StressRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Time to create the shader module and pipeline (the driver may compile later, on first use).
    pub compile_ms: f64,
}

impl StressRenderer {
    pub fn new(gpu: &Gpu) -> Self {
        let t0 = Instant::now();
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
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("stress-pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("stress"),
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
                    targets: &[Some(gpu.config.format.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        Self {
            pipeline,
            uniform,
            bind_group,
            compile_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Draws one frame with synthetic, deterministic "bands" derived from `time`.
    pub fn render(&self, gpu: &mut Gpu, time: f32) -> Result<(), String> {
        let (w, h) = (gpu.config.width as f32, gpu.config.height as f32);
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
        gpu.queue.write_buffer(&self.uniform, 0, &bytes);
        gpu.render(|enc, view| {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stress"),
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
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        })
    }
}
