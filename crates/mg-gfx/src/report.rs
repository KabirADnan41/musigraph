//! GPU adapter report (P3.1): lists what each wgpu backend really enumerates on this machine.
//! Output is text for humans; it never claims compatibility, only what was observed.

use std::fmt::Write as _;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Minimal executor for wgpu's async calls (replaces a `pollster` dependency).
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}

/// Enumerates adapters separately per backend (DX12, Vulkan, GL) and returns a text report.
pub fn adapter_report() -> String {
    let mut out = String::new();
    for (label, backends) in [
        ("DX12", wgpu::Backends::DX12),
        ("Vulkan", wgpu::Backends::VULKAN),
        ("GL", wgpu::Backends::GL),
    ] {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        let instance = wgpu::Instance::new(desc);
        let adapters = block_on(instance.enumerate_adapters(backends));
        let _ = writeln!(out, "== backend {label}: {} adapter(s)", adapters.len());
        for a in adapters {
            let i = a.get_info();
            let l = a.limits();
            let _ = writeln!(
                out,
                "  name={:?} vendor=0x{:04x} device=0x{:04x} type={:?} backend={:?} driver={:?} driver_info={:?}",
                i.name, i.vendor, i.device, i.device_type, i.backend, i.driver, i.driver_info
            );
            let _ = writeln!(
                out,
                "    max_texture_2d={} max_bind_groups={} max_uniform_buffer={} features={:?}",
                l.max_texture_dimension_2d,
                l.max_bind_groups,
                l.max_uniform_buffer_binding_size,
                a.features()
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::block_on;

    #[test]
    fn block_on_runs_ready_future() {
        assert_eq!(block_on(async { 7 }), 7);
    }
}
