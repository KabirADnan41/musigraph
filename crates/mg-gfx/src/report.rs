//! GPU adapter report (P3.1): lists what each wgpu backend enumerates on this machine.
//! ENUMERATION ONLY: no device, surface, shader or frame is created, so nothing here proves
//! that rendering works. CPU/software adapters are tagged so they are never mistaken for hardware.

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
pub(crate) fn block_on<F: Future>(fut: F) -> F::Output {
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
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let _ = writeln!(
        out,
        "Musigraph GPU adapter report ({profile} build). ENUMERATION ONLY: no device, surface, shader or render was verified."
    );
    let (mut hardware, mut software) = (0usize, 0usize);
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
            let is_software = i.device_type == wgpu::DeviceType::Cpu;
            if is_software {
                software += 1;
            } else {
                hardware += 1;
            }
            let tag = if is_software {
                " SOFTWARE (not hardware)"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "  name={:?} vendor=0x{:04x} device=0x{:04x} type={:?} backend={:?} driver={:?} driver_info={:?}{tag}",
                i.name, i.vendor, i.device, i.device_type, i.backend, i.driver, i.driver_info
            );
            let _ = writeln!(
                out,
                "    max_texture_2d={} max_bind_groups={} max_uniform_buffer={}",
                l.max_texture_dimension_2d, l.max_bind_groups, l.max_uniform_buffer_binding_size
            );
        }
    }
    let _ = writeln!(
        out,
        "summary: {hardware} hardware adapter listing(s), {software} software adapter listing(s)"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::{adapter_report, block_on};
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    #[test]
    fn block_on_runs_ready_future() {
        assert_eq!(block_on(async { 7 }), 7);
    }

    struct PendingOnce(bool);

    impl Future for PendingOnce {
        type Output = u32;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
            if self.0 {
                Poll::Ready(9)
            } else {
                self.0 = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    #[test]
    fn block_on_handles_self_wake() {
        assert_eq!(block_on(PendingOnce(false)), 9);
    }

    #[test]
    fn block_on_handles_cross_thread_wake() {
        struct Remote(Option<std::thread::JoinHandle<()>>, bool);
        impl Future for Remote {
            type Output = u32;
            fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
                if self.1 {
                    return Poll::Ready(5);
                }
                self.1 = true;
                let w = cx.waker().clone();
                self.0 = Some(std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(30));
                    w.wake();
                }));
                Poll::Pending
            }
        }
        assert_eq!(block_on(Remote(None, false)), 5);
    }

    #[test]
    fn report_has_all_backend_sections_and_disclaimer() {
        let r = adapter_report();
        assert!(r.contains("ENUMERATION ONLY"));
        for b in ["DX12", "Vulkan", "GL"] {
            assert!(r.contains(&format!("== backend {b}:")), "missing {b}");
        }
        assert!(r.contains("summary:"));
    }
}
