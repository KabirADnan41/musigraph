//! GPU context (P3.2): hardware adapter selection, device/queue, surface configuration and a
//! clear-colour frame. Software (CPU) adapters are rejected: never reported as hardware.

use crate::report::block_on;

/// Backend requested by the caller. Explicit on purpose: no silent fallback between backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    Dx12,
    Vulkan,
    Gl,
}

impl BackendChoice {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "dx12" => Some(Self::Dx12),
            "vulkan" => Some(Self::Vulkan),
            "gl" => Some(Self::Gl),
            _ => None,
        }
    }

    fn backends(self) -> wgpu::Backends {
        match self {
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Gl => wgpu::Backends::GL,
        }
    }
}

/// A live GPU context bound to one surface.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub info: wgpu::AdapterInfo,
    pub device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Creates device, queue and a configured surface for `target` (for example an `Arc<winit::window::Window>`).
/// Fails (never falls back to software) if only a CPU adapter is available.
pub fn init(
    backend: BackendChoice,
    target: impl Into<wgpu::SurfaceTarget<'static>>,
    width: u32,
    height: u32,
    vsync: bool,
) -> Result<Gpu, String> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backend.backends();
    let instance = wgpu::Instance::new(desc);
    let surface = instance
        .create_surface(target)
        .map_err(|e| format!("create_surface failed: {e}"))?;
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
        compatible_surface: Some(&surface),
    }))
    .map_err(|e| format!("request_adapter failed on {backend:?}: {e}"))?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!(
            "only a software adapter ({:?}) is available on {backend:?}; refusing to report it as hardware",
            info.name
        ));
    }
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("musigraph"),
        ..Default::default()
    }))
    .map_err(|e| format!("request_device failed: {e}"))?;
    let device_lost = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = device_lost.clone();
    device.set_device_lost_callback(move |reason, message| {
        eprintln!("GPU device lost: {reason:?}: {message}");
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(wgpu::TextureFormat::is_srgb)
        .or_else(|| caps.formats.first().copied())
        .ok_or("surface reports no formats")?;
    let want = if vsync {
        wgpu::PresentMode::Fifo
    } else {
        wgpu::PresentMode::Immediate
    };
    let present_mode = if caps.present_modes.contains(&want) {
        want
    } else {
        wgpu::PresentMode::Fifo
    };
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width.max(1),
        height: height.max(1),
        present_mode,
        desired_maximum_frame_latency: 2,
        alpha_mode: caps
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto),
        view_formats: vec![],
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    surface.configure(&device, &config);
    Ok(Gpu {
        device,
        queue,
        surface,
        config,
        info,
        device_lost,
    })
}

impl Gpu {
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return; // minimised: keep the old configuration
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Acquires the next frame, lets `f` record commands into it, submits and presents.
    /// Recoverable surface states (outdated, lost, occluded) skip the frame and return Ok.
    pub fn render<F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView)>(
        &mut self,
        f: F,
    ) -> Result<(), String> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            other => return Err(format!("surface error: {other:?}")),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        f(&mut enc, &view);
        self.queue.submit(Some(enc.finish()));
        self.queue.present(frame);
        Ok(())
    }

    /// Renders one clear-colour frame.
    pub fn render_clear(&mut self, rgb: [f64; 3]) -> Result<(), String> {
        self.render(|enc, view| {
            let _pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: rgb[0],
                            g: rgb[1],
                            b: rgb[2],
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        })
    }
}
