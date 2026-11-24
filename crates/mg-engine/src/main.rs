//! Musigraph native fullscreen visualizer.
//! `--gpu-report` (P3.1); `--gpu-test` (P3.2 clear colour); `--gpu-bench` (P3.3 stress shader).
//! Options: `--backend dx12|vulkan|gl` `--seconds N` `--fullscreen` `--vsync on|off` `--scale F` (bench).

use std::io::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mg_gfx::context::{BackendChoice, Gpu};
use mg_gfx::stress::StressRenderer;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

/// Seconds rendered before measurement starts (fullscreen settle, shader/driver warm-up).
const WARMUP_S: f64 = 1.5;
/// Nominal refresh interval used to count missed vblanks (60 Hz panel; verified by dxdiag).
const VBLANK_MS: f64 = 1000.0 / 60.0;

struct Opts {
    backend: BackendChoice,
    seconds: u64,
    fullscreen: bool,
    vsync: bool,
    scale: f32,
    bench: bool,
}

/// Field order matters: `gpu` (surface) must drop before `window`.
struct App {
    opts: Opts,
    stress: Option<StressRenderer>,
    gpu: Option<Gpu>,
    window: Option<Arc<Window>>,
    start: Instant,
    last_frame: Option<Instant>,
    intervals_ms: Vec<f64>,
    frames: u64,
    errors: u64,
    compile_ms: f64,
    result: Result<(), String>,
}

impl App {
    fn scaled(&self, w: u32, h: u32) -> (u32, u32) {
        (
            ((w as f32 * self.opts.scale).round() as u32).max(1),
            ((h as f32 * self.opts.scale).round() as u32).max(1),
        )
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
        let (w, h) = self.scaled(size.width, size.height);
        match mg_gfx::context::init(self.opts.backend, window.clone(), w, h, self.opts.vsync) {
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
                    let s = StressRenderer::new(&gpu);
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
                let (w, h) = self.scaled(s.width, s.height);
                if let Some(g) = self.gpu.as_mut() {
                    g.resize(w, h);
                }
            }
            WindowEvent::RedrawRequested => {
                let t = self.start.elapsed().as_secs_f64();
                let total = self.opts.seconds as f64 + if self.opts.bench { WARMUP_S } else { 0.0 };
                if t >= total {
                    return el.exit(); // bounded duration: automatic exit
                }
                let Some(g) = self.gpu.as_mut() else { return };
                if g.device_lost.load(std::sync::atomic::Ordering::SeqCst) {
                    self.result = Err("GPU device lost".into());
                    return el.exit();
                }
                let r = if let Some(s) = &self.stress {
                    s.render(g, t as f32)
                } else {
                    g.render_clear([
                        0.5 + 0.5 * (t * 1.3).sin(),
                        0.5 + 0.5 * (t * 0.9 + 2.0).sin(),
                        0.5 + 0.5 * (t * 0.7 + 4.0).sin(),
                    ])
                };
                match r {
                    Ok(()) => {
                        let now = Instant::now();
                        if self.opts.bench && t >= WARMUP_S {
                            if let Some(prev) = self.last_frame {
                                self.intervals_ms.push((now - prev).as_secs_f64() * 1000.0);
                            }
                        }
                        self.last_frame = Some(now);
                        self.frames += 1;
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

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
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

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
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
    let bench = opts.bench;
    let vsync = opts.vsync;
    let scale = opts.scale;
    let mut app = App {
        opts,
        stress: None,
        gpu: None,
        window: None,
        start: Instant::now(),
        last_frame: None,
        intervals_ms: Vec::new(),
        frames: 0,
        errors: 0,
        compile_ms: 0.0,
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
        "frames={} errors={} seconds={:.2}",
        app.frames, app.errors, secs
    );
    if bench && !app.intervals_ms.is_empty() {
        let n = app.intervals_ms.len() as f64;
        let sum: f64 = app.intervals_ms.iter().sum();
        let mut s = app.intervals_ms.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let missed: f64 = app
            .intervals_ms
            .iter()
            .map(|d| ((d / VBLANK_MS).round() - 1.0).max(0.0))
            .sum();
        let res = app
            .gpu
            .as_ref()
            .map(|g| (g.config.width, g.config.height))
            .unwrap_or((0, 0));
        println!(
            "BENCH scale={scale} render_res={}x{} vsync={vsync} compile_ms={:.1} measured_frames={} fps={:.1} avg_ms={:.3} median_ms={:.3} p95_ms={:.3} p99_ms={:.3} max_ms={:.3} missed_vblanks={} missed_pct={:.2}",
            res.0,
            res.1,
            app.compile_ms,
            app.intervals_ms.len(),
            1000.0 * n / sum,
            sum / n,
            pct(&s, 0.5),
            pct(&s, 0.95),
            pct(&s, 0.99),
            s[s.len() - 1],
            missed,
            100.0 * missed / (n + missed)
        );
    }
    match app.result {
        Ok(()) if app.frames > 0 => {
            println!("RESULT: ok (hardware adapter rendered frames)");
            0
        }
        Ok(()) => {
            eprintln!("RESULT: FAILED: no frame was rendered");
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
