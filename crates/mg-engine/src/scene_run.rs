//! `--scene` (LIVE system audio) and `--scene-synth NAME` (explicitly labelled SYNTHETIC fixture):
//! the fullscreen Cinematic Spectrum scene.
//!
//! Live path: WASAPI loopback -> capture thread -> central `Analyzer` -> shared latest frame ->
//! `SceneMapper` -> `SpectrumScene`. Nothing here invents musical activity: if no fresh analysis frame
//! arrives (capture failed or stalled) the scene decays to silence and shows the NO AUDIO state, and the
//! window title says so. Samples are never stored or logged (ADR-0003).

use crate::scene_map::{Input, MapTuning, SceneMapper};
use mg_audio::capture::Loopback;
use mg_dsp::{AnalysisParams, Analyzer, AudioFrame};
use mg_gfx::context::{BackendChoice, FrameStatus, Gpu};
use mg_gfx::scene::{SceneMode, SpectrumScene};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

/// A minimised/occluded window is normal for a visualizer; only a very long stall is treated as failure.
const MAX_SKIP_TIME: Duration = Duration::from_secs(30);

#[derive(Debug, PartialEq)]
pub enum Source {
    Live,
    Synth(String),
}

#[derive(Debug, PartialEq)]
pub struct SceneOpts {
    pub backend: Option<BackendChoice>,
    pub seconds: u64,
    pub fullscreen: bool,
    pub vsync: bool,
    pub scale: f32,
    pub source: Source,
}

pub fn parse(args: &[String], source: Source) -> Result<SceneOpts, String> {
    let mut o = SceneOpts {
        backend: None,
        seconds: 600,
        fullscreen: true,
        vsync: true,
        scale: 1.0,
        source,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--windowed" => o.fullscreen = false,
            "--fullscreen" => o.fullscreen = true,
            "--backend" => {
                o.backend = match it.next().map(String::as_str) {
                    Some("auto") => None,
                    Some(v) => {
                        Some(BackendChoice::parse(v).ok_or("--backend needs auto|dx12|vulkan|gl")?)
                    }
                    None => return Err("--backend needs auto|dx12|vulkan|gl".into()),
                }
            }
            "--seconds" => {
                o.seconds = it
                    .next()
                    .and_then(|v| v.parse::<u64>().ok())
                    .filter(|s| (1..=3600).contains(s))
                    .ok_or("--seconds needs 1..=3600")?;
            }
            "--vsync" => {
                o.vsync = match it.next().map(String::as_str) {
                    Some("on") => true,
                    Some("off") => false,
                    _ => return Err("--vsync needs on|off".into()),
                }
            }
            "--scale" => {
                o.scale = it
                    .next()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|s| s.is_finite() && (0.25..=1.0).contains(s))
                    .ok_or("--scale needs 0.25..=1.0")?;
            }
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(o)
}

/// Latest analysis state shared between the audio side and the render loop.
#[derive(Default)]
struct Shared {
    frame: Option<AudioFrame>,
    seq: u64,
    onsets: u32,
    onset_strength: f32,
}

fn publish(sh: &Mutex<Shared>, f: &AudioFrame) {
    if let Ok(mut s) = sh.lock() {
        s.frame = Some(*f);
        s.seq += 1;
        if f.onset_event {
            s.onsets += 1;
            s.onset_strength = s.onset_strength.max(f.onset);
        }
    }
}

/// Real-time playback of a synthetic fixture through the SAME analyzer, in the render thread.
struct SynthSource {
    analyzer: Analyzer,
    samples: Vec<f32>,
    pos: usize,
    sr: u32,
    carry: f64,
    done: u64,
}

impl SynthSource {
    fn advance(&mut self, dt: f64, sh: &Mutex<Shared>) {
        self.carry += dt * f64::from(self.sr);
        let n = self.carry.floor() as usize;
        self.carry -= n as f64;
        let mut left = n;
        while left > 0 {
            let take = left.min(self.samples.len() - self.pos);
            let chunk = &self.samples[self.pos..self.pos + take];
            self.pos = (self.pos + take) % self.samples.len();
            self.done += take as u64;
            let t = self.done * 1_000_000_000 / u64::from(self.sr);
            self.analyzer.push(chunk, t, |f| publish(sh, f));
            left -= take;
        }
    }
}

struct App {
    opts: SceneOpts,
    shared: Arc<Mutex<Shared>>,
    capture: Option<Loopback>,
    synth: Option<SynthSource>,
    mapper: SceneMapper,
    stale_after: Duration,
    scene: Option<SpectrumScene>,
    gpu: Option<Gpu>,
    window: Option<Arc<Window>>,
    start: Instant,
    last_tick: Instant,
    last_seq: u64,
    last_change: Instant,
    title: String,
    last_frame: Option<Instant>,
    intervals_ms: Vec<f64>,
    presented: u64,
    skipped: u64,
    skip_since: Option<Instant>,
    stale_ticks: u64,
    timed_out: bool,
    result: Result<(), String>,
}

impl App {
    fn label(&self, fresh: bool) -> String {
        match (&self.opts.source, fresh) {
            (Source::Live, true) => "Musigraph - LIVE system audio (Esc to exit)".into(),
            (Source::Live, false) => {
                let err = self
                    .capture
                    .as_ref()
                    .and_then(|c| c.status().last_error)
                    .unwrap_or_else(|| "waiting for audio".into());
                format!("Musigraph - NO AUDIO ({err}) (Esc to exit)")
            }
            (Source::Synth(n), _) => {
                format!("Musigraph - SYNTHETIC TEST SIGNAL '{n}' - not music (Esc to exit)")
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Musigraph")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = match el.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.result = Err(format!("create_window failed: {e}"));
                return el.exit();
            }
        };
        if self.opts.fullscreen {
            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
            window.set_cursor_visible(false);
        }
        let size = window.inner_size();
        let r = match self.opts.backend {
            Some(b) => {
                mg_gfx::context::init(b, window.clone(), size.width, size.height, self.opts.vsync)
            }
            None => {
                mg_gfx::context::init_auto(window.clone(), size.width, size.height, self.opts.vsync)
                    .map(|(g, failed)| {
                        for f in &failed {
                            eprintln!("backend skipped: {f}");
                        }
                        g
                    })
            }
        };
        match r {
            Ok(gpu) => {
                println!(
                    "adapter={:?} backend={:?} present_mode={:?} surface_format={:?}",
                    gpu.info.name, gpu.info.backend, gpu.config.present_mode, gpu.config.format
                );
                self.scene = Some(SpectrumScene::new(
                    &gpu.device,
                    gpu.config.format,
                    self.opts.scale,
                ));
                self.gpu = Some(gpu);
            }
            Err(e) => {
                self.result = Err(e);
                el.exit();
            }
        }
        self.window = Some(window);
        self.start = Instant::now();
        self.last_tick = self.start;
        self.last_change = self.start;
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
            WindowEvent::RedrawRequested => self.frame(el),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() && self.start.elapsed().as_secs() >= self.opts.seconds + 2 {
            self.timed_out = true;
            return el.exit();
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl App {
    fn frame(&mut self, el: &ActiveEventLoop) {
        let now = Instant::now();
        if self.start.elapsed().as_secs() >= self.opts.seconds {
            self.timed_out = true;
            return el.exit();
        }
        let dt = now.duration_since(self.last_tick).as_secs_f64().min(0.25);
        self.last_tick = now;
        if let Some(s) = self.synth.as_mut() {
            s.advance(dt, &self.shared);
        }
        // Gather the newest analysis state (and onset events since the previous frame).
        let (frame, seq, onsets, strength) = match self.shared.lock() {
            Ok(mut s) => {
                let r = (s.frame, s.seq, s.onsets, s.onset_strength);
                s.onsets = 0;
                s.onset_strength = 0.0;
                r
            }
            Err(_) => (None, 0, 0, 0.0),
        };
        if seq != self.last_seq {
            self.last_seq = seq;
            self.last_change = now;
        }
        let fresh = frame.is_some() && now.duration_since(self.last_change) < self.stale_after;
        if !fresh {
            self.stale_ticks += 1;
        }
        let mode = match self.opts.source {
            Source::Live => SceneMode::Live,
            Source::Synth(_) => SceneMode::Synthetic,
        };
        let default_frame = AudioFrame::default();
        let params = self.mapper.update(
            &Input {
                frame: frame.as_ref().unwrap_or(&default_frame),
                onsets,
                onset_strength: strength,
                fresh,
                mode,
            },
            dt as f32,
        );
        let label = self.label(fresh);
        if label != self.title {
            if let Some(w) = &self.window {
                w.set_title(&label);
            }
            self.title = label;
        }
        let (Some(g), Some(scene)) = (self.gpu.as_mut(), self.scene.as_mut()) else {
            return;
        };
        if g.device_lost.load(std::sync::atomic::Ordering::SeqCst) {
            self.result = Err("GPU device lost".into());
            return el.exit();
        }
        if let Some(e) = g.take_error() {
            self.result = Err(format!("wgpu error: {e}"));
            return el.exit();
        }
        match scene.render(g, &params) {
            Ok(FrameStatus::Presented) => {
                if let Some(prev) = self.last_frame {
                    self.intervals_ms
                        .push(now.duration_since(prev).as_secs_f64() * 1000.0);
                }
                self.last_frame = Some(now);
                self.presented += 1;
                self.skip_since = None;
            }
            Ok(FrameStatus::Skipped) => {
                self.skipped += 1;
                std::thread::sleep(Duration::from_millis(16)); // occluded/minimised: do not spin a core
                self.last_frame = None;
                let for_s = self.skip_since.get_or_insert_with(Instant::now).elapsed();
                if for_s > MAX_SKIP_TIME {
                    self.result = Err(format!(
                        "no frame presented for {:.1} s",
                        for_s.as_secs_f64()
                    ));
                    el.exit();
                }
            }
            Err(e) => {
                self.result = Err(e);
                el.exit();
            }
        }
    }
}

/// Runs the scene; returns the process exit code.
pub fn run(opts: SceneOpts) -> i32 {
    crate::spawn_watchdog(Duration::from_secs(opts.seconds + 30));
    let shared = Arc::new(Mutex::new(Shared::default()));
    let mut capture = None;
    let mut synth = None;
    match &opts.source {
        Source::Live => {
            let sh = shared.clone();
            let mut analyzer: Option<Analyzer> = None;
            match Loopback::start(move |block, rate, t_end| {
                if analyzer.as_ref().map(Analyzer::sample_rate) != Some(rate) {
                    analyzer = Analyzer::new(rate, AnalysisParams::default()).ok();
                }
                if let Some(a) = analyzer.as_mut() {
                    a.push(block, t_end, |f| publish(&sh, f));
                }
            }) {
                Ok(c) => capture = Some(c),
                Err(e) => {
                    eprintln!("RESULT: FAILED: {e}");
                    return 1;
                }
            }
            println!(
                "LIVE system audio: WASAPI loopback of the default output; derived values only, nothing recorded."
            );
        }
        Source::Synth(name) => {
            let sr = 48_000;
            let Some(samples) = crate::sim::fixture(name, sr) else {
                eprintln!(
                    "unknown fixture '{name}'; available: {}",
                    crate::sim::FIXTURES.join(", ")
                );
                return 2;
            };
            synth = Some(SynthSource {
                analyzer: Analyzer::new(sr, AnalysisParams::default()).expect("valid rate"),
                samples,
                pos: 0,
                sr,
                carry: 0.0,
                done: 0,
            });
            println!("SYNTHETIC test signal '{name}' (not music; for tests/previews only).");
        }
    }
    let event_loop = match EventLoop::new() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("event loop failed: {e}");
            return 1;
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);
    let tune = MapTuning::default();
    let stale_after = Duration::from_secs_f32(tune.stale_after_s);
    let mut app = App {
        opts,
        shared,
        capture,
        synth,
        mapper: SceneMapper::new(tune),
        stale_after,
        scene: None,
        gpu: None,
        window: None,
        start: Instant::now(),
        last_tick: Instant::now(),
        last_seq: 0,
        last_change: Instant::now(),
        title: String::new(),
        last_frame: None,
        intervals_ms: Vec::new(),
        presented: 0,
        skipped: 0,
        skip_since: None,
        stale_ticks: 0,
        timed_out: false,
        result: Ok(()),
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("event loop error: {e}");
        return 1;
    }
    if let Some(c) = app.capture.as_mut() {
        let st = c.status();
        c.stop();
        println!(
            "audio: opens={} discontinuities={} synthesized_silence_blocks={} last_error={}",
            st.opens,
            st.discontinuities,
            st.synthesized_silence_blocks,
            st.last_error.as_deref().unwrap_or("-")
        );
    }
    let mut s = app.intervals_ms.clone();
    s.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| {
        s.get(((s.len() as f64 - 1.0) * p).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    println!(
        "SCENE presented={} skipped={} stale_ticks={} completed={} median_ms={:.2} p95_ms={:.2} p99_ms={:.2} max_ms={:.2}",
        app.presented,
        app.skipped,
        app.stale_ticks,
        app.timed_out,
        pct(0.5),
        pct(0.95),
        pct(0.99),
        s.last().copied().unwrap_or(0.0)
    );
    match app.result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("RESULT: FAILED: {e}");
            1
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
    fn parses_defaults_and_rejects_bad_values() {
        let o = parse(&a(&[]), Source::Live).unwrap();
        assert_eq!(
            (o.seconds, o.fullscreen, o.vsync, o.scale, o.backend),
            (600, true, true, 1.0, None)
        );
        let o = parse(
            &a(&[
                "--windowed",
                "--seconds",
                "30",
                "--scale",
                "0.75",
                "--backend",
                "vulkan",
            ]),
            Source::Live,
        )
        .unwrap();
        assert_eq!(
            (o.seconds, o.fullscreen, o.backend),
            (30, false, Some(BackendChoice::Vulkan))
        );
        for bad in [
            &["--seconds", "0"][..],
            &["--seconds", "99999"],
            &["--scale", "2"],
            &["--scale", "NaN"],
            &["--backend", "metal"],
            &["--bogus"],
        ] {
            assert!(parse(&a(bad), Source::Live).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn every_fixture_is_finite_bounded_and_nonempty() {
        for name in crate::sim::FIXTURES {
            let x = crate::sim::fixture(name, 48_000).unwrap();
            assert!(
                !x.is_empty() && x.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
                "{name}"
            );
        }
        assert!(crate::sim::fixture("nope", 48_000).is_none());
    }
}
