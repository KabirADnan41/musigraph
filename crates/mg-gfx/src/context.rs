//! GPU context (P3.2): hardware adapter selection, device/queue, surface configuration and frames.
//! Software adapters are rejected: they must never be reported as hardware compatibility.

use crate::report::block_on;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub use wgpu::{Device, PresentMode, Queue, TextureFormat};

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

/// True for CPU/software adapters. The GL backend always reports `DeviceType::Other`, so the
/// vendor id (Microsoft 0x1414) and well-known software renderer names are checked as well.
pub fn is_software_adapter(info: &wgpu::AdapterInfo) -> bool {
    if info.device_type == wgpu::DeviceType::Cpu || info.vendor == 0x1414 {
        return true;
    }
    let n = info.name.to_ascii_lowercase();
    [
        "llvmpipe",
        "softpipe",
        "swrast",
        "swiftshader",
        "warp",
        "basic render",
        "gdi generic",
        "software",
    ]
    .iter()
    .any(|t| n.contains(t))
}

/// Outcome of one `render` call. Skipped frames must never be counted as rendered frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameStatus {
    Presented,
    Skipped,
}

/// A live GPU context bound to one surface.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub info: wgpu::AdapterInfo,
    pub device_lost: Arc<AtomicBool>,
    /// Diagnostic: time each frame's GPU work by waiting for completion (see `render`).
    pub measure_gpu: bool,
    /// Submit-to-completion time of the last frame when `measure_gpu` is set.
    pub last_gpu_ms: Option<f64>,
    /// First uncaptured wgpu error (validation, out of memory, ...), if any. Callers must exit on it.
    pub error: Arc<Mutex<Option<String>>>,
    /// Present modes the surface supports on this adapter.
    pub present_modes: Vec<wgpu::PresentMode>,
    max_dim: u32,
}

/// Default automatic preference order. It is a starting point only (measured on one reference
/// machine, ADR-0009), not a claim about other hardware: the first backend that creates a real
/// hardware device is used, and `--backend` always overrides it.
pub const AUTO_ORDER: [BackendChoice; 3] = [
    BackendChoice::Vulkan,
    BackendChoice::Dx12,
    BackendChoice::Gl,
];

/// Tries each backend in `AUTO_ORDER` and returns the first that initialises on real hardware, plus a
/// diagnostic line for every backend that failed. Never falls back to a software adapter.
pub fn init_auto<T>(
    target: T,
    width: u32,
    height: u32,
    vsync: bool,
) -> Result<(Gpu, Vec<String>), String>
where
    T: Into<wgpu::SurfaceTarget<'static>> + Clone,
{
    let mut failures = Vec::new();
    for b in AUTO_ORDER {
        match init(b, target.clone(), width, height, vsync) {
            Ok(g) => return Ok((g, failures)),
            Err(e) => failures.push(format!("{b:?}: {e}")),
        }
    }
    Err(format!(
        "no supported hardware graphics backend: {}",
        failures.join(" | ")
    ))
}

/// A device without a window, for headless rendering measurements and tests.
pub struct Headless {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    /// Held for the life of the device: see `HEADLESS_LOCK`. Declared last so it is released after the device.
    _exclusive: std::sync::MutexGuard<'static, ()>,
}

/// Only one headless device may exist at a time. Several full-scene devices created in parallel (tests run on
/// all cores) exhaust the video memory of small GPUs (2 GB class) and fail with "Out of Memory"; this is a
/// resource limit of the machine, not a defect to hide, so the devices are serialised instead.
static HEADLESS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Creates a headless hardware device using the first backend of `AUTO_ORDER` that works.
/// Software adapters are rejected, so tests can never silently measure a CPU renderer.
pub fn init_headless() -> Result<Headless, String> {
    init_headless_order(&AUTO_ORDER)
}

/// Like `init_headless` but tries only the given backends, in order (used to verify one backend).
pub fn init_headless_order(order: &[BackendChoice]) -> Result<Headless, String> {
    let exclusive = HEADLESS_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut failures = Vec::new();
    for &b in order {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = b.backends();
        let instance = wgpu::Instance::new(desc);
        let adapter = match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
            compatible_surface: None,
        })) {
            Ok(a) => a,
            Err(e) => {
                failures.push(format!("{b:?}: {e}"));
                continue;
            }
        };
        let info = adapter.get_info();
        if is_software_adapter(&info) {
            failures.push(format!("{b:?}: software adapter {:?}", info.name));
            continue;
        }
        match block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("musigraph-headless"),
            ..Default::default()
        })) {
            Ok((device, queue)) => {
                return Ok(Headless {
                    device,
                    queue,
                    info,
                    _exclusive: exclusive,
                });
            }
            Err(e) => failures.push(format!("{b:?}: {e}")),
        }
    }
    Err(format!("no hardware device: {}", failures.join(" | ")))
}

/// Creates device, queue and a configured surface for `target` (for example an `Arc<winit::window::Window>`).
/// Fails (never falls back to software) if only a software adapter is available.
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
    if is_software_adapter(&info) {
        return Err(format!(
            "only a software adapter ({:?}, type {:?}) is available on {backend:?}; refusing to report it as hardware",
            info.name, info.device_type
        ));
    }
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("musigraph"),
        ..Default::default()
    }))
    .map_err(|e| format!("request_device failed: {e}"))?;
    let device_lost = Arc::new(AtomicBool::new(false));
    let lost = device_lost.clone();
    device.set_device_lost_callback(move |reason, message| {
        if reason != wgpu::DeviceLostReason::Destroyed {
            eprintln!("GPU device lost: {reason:?}: {message}");
            lost.store(true, Ordering::SeqCst);
        }
    });
    let error = Arc::new(Mutex::new(None));
    let slot = error.clone();
    device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
        eprintln!("wgpu error: {e}");
        if let Ok(mut s) = slot.lock() {
            s.get_or_insert_with(|| e.to_string());
        }
    }));
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
    let max_dim = device.limits().max_texture_dimension_2d;
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width.clamp(1, max_dim),
        height: height.clamp(1, max_dim),
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
        measure_gpu: false,
        last_gpu_ms: None,
        error,
        present_modes: caps.present_modes.clone(),
        max_dim,
    })
}

impl Gpu {
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return; // minimised: keep the old configuration
        }
        self.config.width = width.min(self.max_dim);
        self.config.height = height.min(self.max_dim);
        self.surface.configure(&self.device, &self.config);
    }

    /// Switches the present mode if the surface supports it; returns whether it was applied.
    pub fn set_present_mode(&mut self, mode: wgpu::PresentMode) -> bool {
        if !self.present_modes.contains(&mode) {
            return false;
        }
        self.config.present_mode = mode;
        self.surface.configure(&self.device, &self.config);
        true
    }

    /// First recorded wgpu error, if any.
    pub fn take_error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|g| g.clone())
    }

    /// Acquires the next frame, lets `f` record commands into it, submits and presents.
    /// Timeout/Occluded/Outdated give `Skipped` (Outdated reconfigures first); a lost surface is an error.
    pub fn render<F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView)>(
        &mut self,
        f: F,
    ) -> Result<FrameStatus, String> {
        let (frame, suboptimal) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => (t, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => (t, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(FrameStatus::Skipped);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(FrameStatus::Skipped);
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
        let t0 = std::time::Instant::now();
        self.queue.submit(Some(enc.finish()));
        if self.measure_gpu {
            // Diagnostic only: waits for the GPU to finish this frame (serialises CPU and GPU) to time the
            // GPU work independent of the present path. Never enabled in normal runs.
            // Bounded: a stalled GPU must never hang the (fullscreen) process; a timeout is reported as an error.
            let wait = wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(2)),
            };
            match self.device.poll(wait) {
                Ok(_) => self.last_gpu_ms = Some(t0.elapsed().as_secs_f64() * 1000.0),
                Err(e) => {
                    self.last_gpu_ms = None;
                    return Err(format!("GPU wait failed or timed out: {e}"));
                }
            }
        }
        self.queue.present(frame);
        if suboptimal {
            self.surface.configure(&self.device, &self.config);
        }
        Ok(FrameStatus::Presented)
    }

    /// Renders one clear-colour frame.
    pub fn render_clear(&mut self, rgb: [f64; 3]) -> Result<FrameStatus, String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, vendor: u32, t: wgpu::DeviceType) -> wgpu::AdapterInfo {
        let mut i = wgpu::AdapterInfo::new(t, wgpu::Backend::Gl);
        i.name = name.to_string();
        i.vendor = vendor;
        i
    }

    #[test]
    fn software_adapters_are_detected() {
        assert!(is_software_adapter(&info(
            "anything",
            0x10de,
            wgpu::DeviceType::Cpu
        )));
        assert!(is_software_adapter(&info(
            "Microsoft Basic Render Driver",
            0x1414,
            wgpu::DeviceType::Other
        )));
        assert!(is_software_adapter(&info(
            "llvmpipe (LLVM 15)",
            0x10005,
            wgpu::DeviceType::Other
        )));
        assert!(is_software_adapter(&info(
            "GDI Generic",
            0x1414,
            wgpu::DeviceType::Other
        )));
        assert!(is_software_adapter(&info(
            "SwiftShader Device",
            0x1ae0,
            wgpu::DeviceType::Other
        )));
    }

    #[test]
    fn real_gpus_are_not_flagged() {
        assert!(!is_software_adapter(&info(
            "NVIDIA GeForce GT 730",
            0x10de,
            wgpu::DeviceType::DiscreteGpu
        )));
        assert!(!is_software_adapter(&info(
            "NVIDIA GeForce GT 730/PCIe/SSE2",
            0x10de,
            wgpu::DeviceType::Other
        )));
    }

    #[test]
    fn backend_parse() {
        assert_eq!(BackendChoice::parse("dx12"), Some(BackendChoice::Dx12));
        assert_eq!(BackendChoice::parse("vulkan"), Some(BackendChoice::Vulkan));
        assert_eq!(BackendChoice::parse("gl"), Some(BackendChoice::Gl));
        assert_eq!(BackendChoice::parse("metal"), None);
        assert_eq!(BackendChoice::parse(""), None);
    }
}
