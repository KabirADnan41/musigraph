//! Musigraph native fullscreen visualizer.
//! `--gpu-report` (P3.1); `--gpu-test` (P3.2 clear colour); `--gpu-bench` (P3.3 stress shader).
//! Options: `--backend dx12|vulkan|gl` `--seconds N` `--fullscreen` `--vsync on|off` `--scale F` (bench).

use std::io::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mg_gfx::context::{BackendChoice, FrameStatus, Gpu};
use mg_gfx::stress::StressRenderer;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

/// Seconds rendered before measurement starts (fullscreen settle, shader/driver warm-up).
const WARMUP_S: f64 = 1.5;
/// Refresh interval used to count missed vblanks: the panel runs 1920x1080 @ 60.000 Hz (dxdiag, 2026-11-25).
const VBLANK_MS: f64 = 1000.0 / 60.0;
/// More consecutive skipped frames than this ends the run as a failure (occluded/stalled surface).
const MAX_CONSECUTIVE_SKIPS: u32 = 120;

#[derive(Debug, PartialEq)]
struct Opts {
    backend: BackendChoice,
    seconds: u64,
    fullscreen: bool,
    vsync: bool,
    scale: f32,
    bench: bool,
}

/// Field order matters: `stress` and `gpu` (surface) must drop before `window`.
/// The window is destroyed on every exit path (Esc, close, init failure, unwind); winit then restores the
/// cursor. A hard process kill is covered by the OS only.
struct App {
    opts: Opts,
    stress: Option<StressRenderer>,
    gpu: Option<Gpu>,
    window: Option<Arc<Window>>,
    start: Instant,
    last_frame: Option<Instant>,
    intervals_ms: Vec<f64>,
    presented: u64,
    skipped: u64,
    consecutive_skips: u32,
    errors: u64,
    compile_ms: f64,
    timed_out: bool,
    result: Result<(), String>,
}

impl App {
    fn total_s(&self) -> f64 {
        self.opts.seconds as f64 + if self.opts.bench { WARMUP_S } else { 0.0 }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Musigraph GPU test (Esc to exit)")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = match el.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.result = Err(format!("create_window failed: {e}"));
                el.exit();
                return;
            }
        };
        if self.opts.fullscreen {
            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
            window.set_cursor_visible(false);
        }
        let size = window.inner_size();
        match mg_gfx::context::init(
            self.opts.backend,
            window.clone(),
            size.width,
            size.height,
            self.opts.vsync,
        ) {
            Ok(gpu) => {
                println!(
                    "adapter={:?} type={:?} backend={:?} driver={:?} {:?} surface_format={:?} present_mode={:?}",
                    gpu.info.name,
                    gpu.info.device_type,
                    gpu.info.backend,
                    gpu.info.driver,
                    gpu.info.driver_info,
                    gpu.config.format,
                    gpu.config.present_mode
                );
                if self.opts.bench {
                    let s = StressRenderer::new(&gpu, self.opts.scale);
                    self.compile_ms = s.compile_ms;
                    self.stress = Some(s);
                }
                self.gpu = Some(gpu);
            }
            Err(e) => {
                self.result = Err(e);
                el.exit();
            }
        }
        self.window = Some(window);
        self.start = Instant::now();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => el.exit(),
            WindowEvent::Resized(s) => {
                if let Some(g) = self.gpu.as_mut() {
                    g.resize(s.width, s.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let t = self.start.elapsed().as_secs_f64();
                if t >= self.total_s() {
                    self.timed_out = true;
                    return el.exit(); // bounded duration: automatic exit
                }
                let Some(g) = self.gpu.as_mut() else { return };
                if g.device_lost.load(std::sync::atomic::Ordering::SeqCst) {
                    self.result = Err("GPU device lost".into());
                    return el.exit();
                }
                if let Some(e) = g.take_error() {
                    self.result = Err(format!("wgpu error: {e}"));
                    return el.exit();
                }
                let r = if let Some(s) = self.stress.as_mut() {
                    s.render(g, t as f32)
                } else {
                    g.render_clear([
                        0.5 + 0.5 * (t * 1.3).sin(),
                        0.5 + 0.5 * (t * 0.9 + 2.0).sin(),
                        0.5 + 0.5 * (t * 0.7 + 4.0).sin(),
                    ])
                };
                match r {
                    Ok(FrameStatus::Presented) => {
                        let now = Instant::now();
                        if self.opts.bench && t >= WARMUP_S {
                            if let Some(prev) = self.last_frame {
                                self.intervals_ms.push((now - prev).as_secs_f64() * 1000.0);
                            }
                        }
                        self.last_frame = Some(now);
                        self.presented += 1;
                        self.consecutive_skips = 0;
                    }
                    Ok(FrameStatus::Skipped) => {
                        self.skipped += 1;
                        self.consecutive_skips += 1;
                        self.last_frame = None; // a skipped frame must not pollute the next interval
                        if self.consecutive_skips > MAX_CONSECUTIVE_SKIPS {
                            self.result = Err(format!(
                                "{} consecutive skipped frames (surface occluded or stalled)",
                                self.consecutive_skips
                            ));
                            el.exit();
                        }
                    }
                    Err(e) => {
                        self.errors += 1;
                        self.result = Err(e);
                        el.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // The deadline must not depend on redraw events (minimised or stalled window).
        if self.window.is_some() && self.start.elapsed().as_secs_f64() >= self.total_s() + 2.0 {
            self.timed_out = true;
            el.exit();
            return;
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn parse_opts(args: &[String], bench: bool) -> Result<Opts, String> {
    let mut o = Opts {
        backend: BackendChoice::Dx12,
        seconds: if bench { 10 } else { 5 },
        fullscreen: bench,
        vsync: true,
        scale: 1.0,
        bench,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--fullscreen" => o.fullscreen = true,
            "--windowed" if bench => o.fullscreen = false,
            "--backend" => {
                o.backend = it
                    .next()
                    .and_then(|v| BackendChoice::parse(v))
                    .ok_or("--backend needs dx12|vulkan|gl")?;
            }
            "--seconds" => {
                o.seconds = it
                    .next()
                    .and_then(|v| v.parse::<u64>().ok())
                    .filter(|s| (1..=120).contains(s))
                    .ok_or("--seconds needs 1..=120")?;
            }
            "--scale" if bench => {
                o.scale = it
                    .next()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|s| s.is_finite() && (0.25..=1.0).contains(s))
                    .ok_or("--scale needs 0.25..=1.0")?;
            }
            "--vsync" => {
                o.vsync = match it.next().map(String::as_str) {
                    Some("on") => true,
                    Some("off") => false,
                    _ => return Err("--vsync needs on|off".into()),
                }
            }
            other => return Err(format!("unsupported argument {other:?}")),
        }
    }
    Ok(o)
}

/// Percentile of an ascending-sorted slice (nearest rank); 0.0 for an empty slice.
fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Number of refresh intervals lost: each interval of `d` ms counts `round(d / vblank) - 1` missed vblanks.
fn missed_vblanks(intervals_ms: &[f64], vblank_ms: f64) -> f64 {
    intervals_ms
        .iter()
        .map(|d| ((d / vblank_ms).round() - 1.0).max(0.0))
        .sum()
}

fn run(opts: Opts) -> i32 {
    let event_loop = match EventLoop::new() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("event loop failed: {e}");
            return 1;
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);
    let (bench, vsync, scale) = (opts.bench, opts.vsync, opts.scale);
    let mut app = App {
        opts,
        stress: None,
        gpu: None,
        window: None,
        start: Instant::now(),
        last_frame: None,
        intervals_ms: Vec::new(),
        presented: 0,
        skipped: 0,
        consecutive_skips: 0,
        errors: 0,
        compile_ms: 0.0,
        timed_out: false,
        result: Ok(()),
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("event loop error: {e}");
        return 1;
    }
    let secs = app
        .start
        .elapsed()
        .max(Duration::from_millis(1))
        .as_secs_f64();
    println!(
        "presented={} skipped={} errors={} seconds={:.2} completed={} (aborted_early={})",
        app.presented, app.skipped, app.errors, secs, app.timed_out, !app.timed_out
    );
    if bench && !app.intervals_ms.is_empty() {
        let n = app.intervals_ms.len() as f64;
        let sum: f64 = app.intervals_ms.iter().sum();
        let mut s = app.intervals_ms.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let missed = missed_vblanks(&app.intervals_ms, VBLANK_MS);
        // Missed vblanks are only meaningful when presentation is synchronised (vsync on).
        let missed_txt = if vsync {
            format!(
                "missed_vblanks={missed} missed_pct={:.2}",
                100.0 * missed / (n + missed)
            )
        } else {
            "missed_vblanks=n/a missed_pct=n/a".to_string()
        };
        let res = app
            .gpu
            .as_ref()
            .map(|g| (g.config.width, g.config.height))
            .unwrap_or((0, 0));
        let render = app.stress.as_ref().map_or(res, |s| s.render_size(res));
        println!(
            "BENCH scale={scale} output_res={}x{} render_res={}x{} vsync={vsync} compile_ms={:.1} measured_frames={} fps={:.1} avg_ms={:.3} median_ms={:.3} p95_ms={:.3} p99_ms={:.3} max_ms={:.3} {missed_txt} method=frame_interval_cpu_side",
            res.0,
            res.1,
            render.0,
            render.1,
            app.compile_ms,
            app.intervals_ms.len(),
            1000.0 * n / sum,
            sum / n,
            pct(&s, 0.5),
            pct(&s, 0.95),
            pct(&s, 0.99),
            s[s.len() - 1]
        );
    }
    match app.result {
        Ok(()) if app.presented > 0 => {
            println!(
                "RESULT: ok ({} frames presented on the selected adapter; software adapters are rejected)",
                app.presented
            );
            0
        }
        Ok(()) => {
            eprintln!("RESULT: FAILED: no frame was presented");
            1
        }
        Err(e) => {
            eprintln!("RESULT: FAILED: {e}");
            1
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--gpu-report") if args.len() == 1 => {
            // A closed pipe must not panic the process.
            let _ = std::io::stdout().write_all(mg_gfx::report::adapter_report().as_bytes());
        }
        Some(flag @ ("--gpu-test" | "--gpu-bench")) => {
            match parse_opts(&args[1..], flag == "--gpu-bench") {
                Ok(o) => std::process::exit(run(o)),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(2);
                }
            }
        }
        None => {
            println!("musigraph engine (P3.3 skeleton): --gpu-report | --gpu-test | --gpu-bench")
        }
        _ => {
            eprintln!(
                "unsupported arguments; supported: --gpu-report | --gpu-test | --gpu-bench (see source header)"
            );
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parse_defaults_and_ranges() {
        let o = parse_opts(&a(&[]), true).unwrap();
        assert_eq!(
            (o.seconds, o.fullscreen, o.vsync, o.scale),
            (10, true, true, 1.0)
        );
        let t = parse_opts(&a(&[]), false).unwrap();
        assert_eq!((t.seconds, t.fullscreen), (5, false));
        assert_eq!(
            parse_opts(&a(&["--scale", "0.5"]), true).unwrap().scale,
            0.5
        );
        assert_eq!(
            parse_opts(&a(&["--backend", "vulkan", "--vsync", "off"]), true)
                .unwrap()
                .backend,
            BackendChoice::Vulkan
        );
    }

    #[test]
    fn parse_rejects_bad_input() {
        for bad in [
            &["--scale", "0.1"][..],
            &["--scale", "1.5"],
            &["--scale", "NaN"],
            &["--scale", "inf"],
            &["--scale"],
            &["--seconds", "0"],
            &["--seconds", "121"],
            &["--seconds", "x"],
            &["--vsync", "maybe"],
            &["--backend", "metal"],
            &["--bogus"],
        ] {
            assert!(parse_opts(&a(bad), true).is_err(), "{bad:?}");
        }
        assert!(parse_opts(&a(&["--windowed"]), false).is_err());
        assert!(parse_opts(&a(&["--scale", "0.5"]), false).is_err());
    }

    #[test]
    fn percentiles() {
        assert_eq!(pct(&[], 0.5), 0.0);
        assert_eq!(pct(&[7.0], 0.99), 7.0);
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(pct(&v, 0.5), 51.0);
        assert_eq!(pct(&v, 0.99), 99.0);
        assert_eq!(pct(&v, 1.0), 100.0);
    }

    #[test]
    fn missed_vblank_counting() {
        let v = VBLANK_MS;
        assert_eq!(missed_vblanks(&[], v), 0.0);
        assert_eq!(missed_vblanks(&[v, v, v], v), 0.0);
        assert_eq!(missed_vblanks(&[2.0 * v], v), 1.0);
        assert_eq!(missed_vblanks(&[3.0 * v, v], v), 2.0);
        assert_eq!(missed_vblanks(&[0.4 * v], v), 0.0); // faster than refresh never counts
    }
}
