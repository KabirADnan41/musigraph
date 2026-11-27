//! `--scene` (LIVE system audio) and `--scene-synth NAME` (explicitly labelled SYNTHETIC fixture):
//! the fullscreen Cinematic Spectrum scene.
//!
//! Keys (see `key_action`): Space play/pause, Left/Right previous/next TRACK (they seek -/+5 s, Shift 15 s,
//! while the bar's progress line has keyboard focus; Home/End then jump to the start/end), `[` `]` previous/next scene,
//! 1-9 scene by selector position, L lyrics, A artwork layout, F fullscreen, Esc exit.
//!
//! Live path: WASAPI loopback -> capture thread -> central `Analyzer` -> shared latest frame ->
//! `SceneMapper` -> `SceneSet`. Nothing here invents musical activity: if no fresh analysis frame
//! arrives (capture failed or stalled) the scene decays to silence and shows the NO AUDIO state, and the
//! window title says so. Samples are never stored or logged (ADR-0003).

use crate::controls::{
    Availability, ButtonId, Click, ControlsUi, FrameInput, SEEK_STEP_LONG_S, SEEK_STEP_S, Toggles,
    key_seek_seconds, seek_seconds,
};
use crate::lyric_view::LyricSource;
use crate::media_feed::MediaFeed;
use crate::music_summary::MusicSummary;
use crate::presentation::{GpuOp, Layout, Presentation, Ready, Worker, apply_album};
use crate::scene_map::{Input, MapTuning, SceneMapper};
use crate::scene_panel::{PanelEvent, PanelKey, ScenePanel};
use crate::scene_switch::{FULL, Phase, SceneSwitch, scene_luma};
use mg_audio::capture::Loopback;
use mg_dsp::features::{MusicFeatures, MusicTracker};
use mg_dsp::{AnalysisParams, Analyzer, AudioFrame};
use mg_gfx::context::{BackendChoice, FrameStatus, Gpu};
use mg_gfx::overlay::Overlay;
use mg_gfx::scene::{SceneKind, SceneMode, SceneSet};
use mg_media::{Dispatch, MediaCommand};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

/// A minimised/occluded window is normal for a visualizer; only a very long stall is treated as failure.
/// The F key toggles fullscreen only on its own: Ctrl+F, Alt+F and Win+F belong to the user/system.
/// (Shift is allowed: it does not form a shortcut.)
fn toggle_allowed(m: ModifiersState) -> bool {
    !(m.control_key() || m.alt_key() || m.super_key())
}

/// Everything a key can do. One owner of the key map (`key_action`); behaviour never depends on any UI.
#[derive(Debug, PartialEq, Clone, Copy)]
enum KeyAction {
    Media(MediaCommand),
    PrevScene,
    NextScene,
    /// Zero-based position in `SceneKind::selector()`.
    SceneAt(usize),
    /// Open or close the scene panel (the Scene button).
    TogglePanel,
    ToggleLyrics,
    CycleArtwork,
    ToggleFullscreen,
    Exit,
}

/// Owner-approved key map. Physical keys; Esc always exits; every other key ignores key repeat and any
/// Ctrl/Alt/Win combination (Shift alone is fine): Space = play/pause, Left/Right = previous/next TRACK,
/// `[` `]` = previous/next scene, 1-9 = scene by selector position, L lyrics, A artwork layout, F fullscreen.
fn key_action(code: KeyCode, mods: ModifiersState, repeat: bool) -> Option<KeyAction> {
    if code == KeyCode::Escape {
        return Some(KeyAction::Exit);
    }
    if repeat || !toggle_allowed(mods) {
        return None;
    }
    let digit = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ]
    .iter()
    .position(|d| *d == code);
    if let Some(i) = digit {
        return Some(KeyAction::SceneAt(i));
    }
    Some(match code {
        KeyCode::Space => KeyAction::Media(MediaCommand::TogglePlayPause),
        KeyCode::ArrowLeft => KeyAction::Media(MediaCommand::Previous),
        KeyCode::ArrowRight => KeyAction::Media(MediaCommand::Next),
        KeyCode::BracketLeft => KeyAction::PrevScene,
        KeyCode::BracketRight => KeyAction::NextScene,
        KeyCode::KeyL => KeyAction::ToggleLyrics,
        KeyCode::KeyA => KeyAction::CycleArtwork,
        KeyCode::KeyF => KeyAction::ToggleFullscreen,
        _ => return None,
    })
}

/// A control-bar button runs exactly the action of its key (one implementation, `run_action`).
fn button_action(b: ButtonId) -> KeyAction {
    match b {
        ButtonId::Previous => KeyAction::Media(MediaCommand::Previous),
        ButtonId::PlayPause => KeyAction::Media(MediaCommand::TogglePlayPause),
        ButtonId::Next => KeyAction::Media(MediaCommand::Next),
        ButtonId::Scene => KeyAction::TogglePanel,
        ButtonId::Lyrics => KeyAction::ToggleLyrics,
        ButtonId::Artwork => KeyAction::CycleArtwork,
        ButtonId::Fullscreen => KeyAction::ToggleFullscreen,
        ButtonId::Exit => KeyAction::Exit,
    }
}

/// The keys the open scene panel takes before the global key map (so Esc closes the panel, not the app, and the
/// arrows move the panel focus instead of changing track). Enter ignores key repeat; Space, `[` `]`, the digits,
/// L, A and F stay global.
/// Where an Esc press goes: an open panel closes (and the key is then "held": its auto-repeat is swallowed until it
/// is released); a panel still fading out swallows it too, so a second Esc cannot exit the app by accident.
#[derive(Debug, PartialEq)]
enum EscRoute {
    Panel,
    Swallow,
    App,
}

/// `repeat` is the OS auto-repeat flag; `held_after_close` is true from the press that closed the panel until that
/// key is released. A NEW press (not a repeat, not held) always reaches the app, even during the fade-out.
fn route_esc(
    panel_open: bool,
    panel_visible: bool,
    repeat: bool,
    held_after_close: bool,
) -> EscRoute {
    if panel_open {
        EscRoute::Panel
    } else if held_after_close || (repeat && panel_visible) {
        EscRoute::Swallow
    } else {
        EscRoute::App
    }
}

/// The Esc key's state across events: `held` from the press that closed the panel until that key is released
/// (or the window loses focus, when no release event will come). The real key handler drives this.
#[derive(Debug, Default)]
struct EscState {
    held: bool,
}

impl EscState {
    /// An Esc press event (`repeat` = OS auto-repeat): where it goes; the press that closes the panel is held.
    fn press(&mut self, panel_open: bool, panel_visible: bool, repeat: bool) -> EscRoute {
        let r = route_esc(panel_open, panel_visible, repeat, self.held);
        if r == EscRoute::Panel {
            self.held = true;
        }
        r
    }

    fn release(&mut self) {
        self.held = false;
    }

    fn focus_lost(&mut self) {
        self.held = false;
    }
}

/// The `--auto-scenes` tour clock, pure so it can be tested with a fake clock: while the panel is up the timer is
/// reset to `now` every frame (nothing is due, nothing accumulates), so closing it neither switches at once nor
/// catches up. Returns (new last-switch time, a switch is due).
fn tour_step(last: Instant, now: Instant, every_s: u64, panel_up: bool) -> (Instant, bool) {
    if panel_up {
        (now, false)
    } else if now.duration_since(last).as_secs() >= every_s {
        (now, true)
    } else {
        (last, false)
    }
}

fn panel_key(code: KeyCode, shift: bool, repeat: bool) -> Option<PanelKey> {
    Some(match code {
        KeyCode::Escape => PanelKey::Escape,
        KeyCode::Tab if shift => PanelKey::BackTab,
        KeyCode::Tab => PanelKey::Tab,
        KeyCode::ArrowUp => PanelKey::Up,
        KeyCode::ArrowDown => PanelKey::Down,
        KeyCode::ArrowLeft => PanelKey::Left,
        KeyCode::ArrowRight => PanelKey::Right,
        KeyCode::Home => PanelKey::Home,
        KeyCode::End => PanelKey::End,
        KeyCode::Enter | KeyCode::NumpadEnter if !repeat => PanelKey::Enter,
        _ => return None,
    })
}

/// Seek delta (seconds) for a key while the progress line has keyboard focus: Left/Right step 5 s back/forward
/// (Shift: 15 s), Home/End jump to the start/end (infinite deltas, clamped by `key_seek_seconds`). Only then do
/// these keys seek; otherwise Left/Right keep their global meaning (previous/next TRACK, `key_action`).
fn progress_key_delta(code: KeyCode, shift: bool) -> Option<f64> {
    let step = if shift { SEEK_STEP_LONG_S } else { SEEK_STEP_S };
    match code {
        KeyCode::ArrowLeft => Some(-step),
        KeyCode::ArrowRight => Some(step),
        KeyCode::Home => Some(f64::NEG_INFINITY),
        KeyCode::End => Some(f64::INFINITY),
        _ => None,
    }
}

/// At most one "command unsupported" stderr note per this interval.
const NOTE_EVERY: Duration = Duration::from_secs(5);

const MAX_SKIP_TIME: Duration = Duration::from_secs(30);
/// Frames longer than this are recorded with a per-section CPU time breakdown.
const HITCH_MS: f64 = 40.0;

#[derive(Debug, PartialEq)]
pub enum Source {
    Live,
    Synth(String),
}

/// Where song information comes from.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum MediaChoice {
    /// Live runs read the Windows media session; synthetic runs show nothing.
    Auto,
    Off,
    /// Original demo tracks with generated artwork (previews and tests).
    Demo,
}

#[derive(Debug, PartialEq)]
pub struct SceneOpts {
    pub backend: Option<BackendChoice>,
    pub seconds: u64,
    pub fullscreen: bool,
    pub vsync: bool,
    pub scale: f32,
    pub source: Source,
    pub media: MediaChoice,
    pub layout: Layout,
    pub style: SceneKind,
    /// Switch to the next scene every N seconds (a hands-free tour; keys still work).
    pub auto_scenes: Option<u64>,
    /// Folder of user-supplied `.lrc` files (synchronised lyrics). Default: `Music\Musigraph Lyrics` in the
    /// user profile, only if it exists.
    pub lyrics_dir: Option<std::path::PathBuf>,
    /// Control bar visibility: `Auto` (idle timer) or `Always` (developer flag for screenshots and draw-cost measurement).
    pub controls: ControlsMode,
    /// Fluid quality tier 0..=3 (developer flag `--fluid-tier`; auto-tier is a TODO).
    pub fluid_tier: u8,
    /// reduced motion for the overlay (`--reduced-motion`). The Windows setting
    /// (SPI_GETCLIENTAREAANIMATION) is not reachable with the dependencies and `windows` features enabled today.
    pub reduced_motion: bool,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ControlsMode {
    Auto,
    Always,
}

pub fn parse(args: &[String], source: Source) -> Result<SceneOpts, String> {
    let mut o = SceneOpts {
        backend: None,
        seconds: 600,
        fullscreen: true,
        vsync: true,
        scale: 1.0,
        source,
        media: MediaChoice::Auto,
        layout: Layout::Corner,
        style: SceneKind::Spectrum,
        auto_scenes: None,
        lyrics_dir: None,
        controls: ControlsMode::Auto,
        fluid_tier: mg_gfx::fluid::DEFAULT_TIER,
        reduced_motion: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--windowed" => o.fullscreen = false,
            "--fullscreen" => o.fullscreen = true,
            "--style" => {
                o.style = it
                    .next()
                    .and_then(|v| SceneKind::parse(v))
                    .ok_or("--style needs spectrum|focus|orb|aurora|fluid")?;
            }
            "--auto-scenes" => {
                o.auto_scenes = Some(
                    it.next()
                        .and_then(|v| v.parse::<u64>().ok())
                        .filter(|s| (3..=3600).contains(s))
                        .ok_or("--auto-scenes needs 3..=3600 seconds")?,
                );
            }
            "--lyrics-dir" => {
                o.lyrics_dir = Some(
                    it.next()
                        .filter(|p| !p.is_empty())
                        .ok_or("--lyrics-dir needs a folder")?
                        .into(),
                );
            }
            "--controls" => {
                o.controls = match it.next().map(String::as_str) {
                    Some("auto") => ControlsMode::Auto,
                    Some("always") => ControlsMode::Always,
                    _ => return Err("--controls needs auto|always".into()),
                }
            }
            "--reduced-motion" => o.reduced_motion = true,
            "--no-media" => o.media = MediaChoice::Off,
            "--demo-media" => o.media = MediaChoice::Demo,
            "--layout" => {
                o.layout = match it.next().map(String::as_str) {
                    Some("corner") => Layout::Corner,
                    Some("center") => Layout::Center,
                    Some("off") => Layout::Off,
                    _ => return Err("--layout needs corner|center|off".into()),
                }
            }
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
            "--fluid-tier" => {
                o.fluid_tier = it
                    .next()
                    .and_then(|v| v.parse::<u8>().ok())
                    .filter(|t| *t <= 3)
                    .ok_or("--fluid-tier needs 0..=3")?;
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
    /// Latest music descriptors (plain Copy data, overwritten in place: no queue, no allocation).
    music: MusicFeatures,
}

fn publish(sh: &Mutex<Shared>, tracker: &mut MusicTracker, f: &AudioFrame) {
    let music = tracker.step(f);
    if let Ok(mut s) = sh.lock() {
        s.frame = Some(*f);
        s.music = music;
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
    tracker: MusicTracker,
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
            let tracker = &mut self.tracker;
            self.analyzer.push(chunk, t, |f| publish(sh, tracker, f));
            left -= take;
        }
    }
}

struct App {
    /// Reduced motion = `--reduced-motion` OR the Windows animation setting (polled once a second; one cheap read-only call).
    motion_last: Option<Instant>,
    motion_reduced: bool,
    opts: SceneOpts,
    shared: Arc<Mutex<Shared>>,
    capture: Option<Loopback>,
    synth: Option<SynthSource>,
    mapper: SceneMapper,
    stale_after: Duration,
    scene: Option<SceneSet>,
    overlay: Option<Overlay>,
    media: MediaFeed,
    pres: Presentation,
    worker: Worker,
    last_media_poll: Option<Instant>,
    last_np: Option<mg_media::NowPlaying>,
    switch: SceneSwitch,
    last_auto_switch: Instant,
    gpu: Option<Gpu>,
    window: Option<Arc<Window>>,
    start: Instant,
    last_tick: Instant,
    last_seq: u64,
    last_change: Instant,
    ctl: ControlsUi,
    panel: ScenePanel,
    /// Pointer position in surface pixels (None while outside the window).
    pointer: Option<(f32, f32)>,
    /// The last input was the keyboard (the panel shows its focus ring only then).
    keyboard_input: bool,
    /// Esc closed the panel and is still held: its auto-repeat must not reach the app (cleared on release).
    esc: EscState,
    cursor_hidden: bool,
    mods: ModifiersState,
    last_unsupported_note: Option<Instant>,
    title: String,
    /// Start of the last presented frame with its section times (poll+jobs, gpu_ops, render ms) and cross-fade
    /// flag: a HITCH interval ends at the next frame's start, so it is attributed to these.
    last_frame: Option<(Instant, [f64; 3], bool)>,
    intervals_ms: Vec<f64>,
    /// Attribution lines for rare long frames (see HITCH_MS); bounded.
    hitches: Vec<String>,
    presented: u64,
    skipped: u64,
    skip_since: Option<Instant>,
    stale_ticks: u64,
    music_summary: MusicSummary,
    timed_out: bool,
    result: Result<(), String>,
    exit_reason: &'static str,
    /// The window has keyboard focus (an unfocused window never hides the cursor).
    focused: bool,
}

/// Immersive mode hides the cursor together with the bar, but never while the window is in the background.
fn hide_cursor(fullscreen: bool, focused: bool, bar_wanted: bool) -> bool {
    fullscreen && focused && !bar_wanted
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
            window.set_fullscreen(Some(Fullscreen::Borderless(window.current_monitor())));
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
                let mut scene = SceneSet::new(&gpu.device, gpu.config.format, self.opts.scale);
                scene.set_fluid_tier(self.opts.fluid_tier);
                self.scene = Some(scene);
                self.overlay = Some(Overlay::new(&gpu.device, &gpu.queue, gpu.config.format));
                // Scene-panel thumbnails: rendered once, here, before the first frame is presented (a throwaway
                // scene set, error-scoped, released at once). Failure leaves placeholders; nothing runs later.
                self.panel.on_thumbs(crate::scene_thumbs::generate_guarded(
                    &gpu.device,
                    &gpu.queue,
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
            WindowEvent::CloseRequested => {
                self.exit_reason = "window close requested";
                el.exit()
            }
            WindowEvent::ModifiersChanged(m) => self.mods = m.state(),
            WindowEvent::Focused(focused) => {
                self.focused = focused;
                if !focused {
                    // alt-tab while a button is held: no release event will come, so nothing may stay pressed
                    self.mods = ModifiersState::empty();
                    self.ctl.focus_lost();
                    self.esc.focus_lost();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Released,
                        ..
                    },
                ..
            } => self.esc.release(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        repeat,
                        ..
                    },
                ..
            } => self.on_key(el, code, repeat),
            WindowEvent::CursorMoved { position, .. } => {
                self.sync_view();
                self.pointer = Some((position.x as f32, position.y as f32));
                self.keyboard_input = false;
                self.panel
                    .pointer_moved((position.x as f32, position.y as f32));
                self.ctl
                    .pointer_moved((position.x as f32, position.y as f32));
            }
            WindowEvent::CursorLeft { .. } => {
                self.pointer = None;
                self.panel.pointer_left();
                self.ctl.pointer_left()
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.on_left_button(el, state == ElementState::Pressed),
            WindowEvent::MouseInput { .. } => self.ctl.key_activity(),
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
        // Immersive mode: the cursor hides together with the control bar (one idle timer, `Visibility`).
        let hide = hide_cursor(
            self.opts.fullscreen,
            self.focused,
            self.ctl.vis.wants_visible(),
        );
        if hide != self.cursor_hidden {
            if let Some(w) = &self.window {
                w.set_cursor_visible(!hide);
                self.cursor_hidden = hide;
            }
        }
        if self.window.is_some() && self.start.elapsed().as_secs() >= self.opts.seconds + 2 {
            self.timed_out = true;
            self.exit_reason = "time limit reached (--seconds, deadline)";
            return el.exit();
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl App {
    fn run_action(&mut self, el: &ActiveEventLoop, action: KeyAction) {
        match action {
            KeyAction::Exit => {
                self.exit_reason = "Esc pressed";
                el.exit();
            }
            KeyAction::ToggleFullscreen => {
                self.opts.fullscreen = !self.opts.fullscreen;
                self.apply_fullscreen();
            }
            KeyAction::ToggleLyrics => {
                self.pres.toggle_lyrics();
            }
            KeyAction::CycleArtwork => {
                let surface = self
                    .gpu
                    .as_ref()
                    .map_or((0, 0), |g| (g.config.width, g.config.height));
                self.pres.set_layout(self.pres.layout().next(), surface);
            }
            KeyAction::TogglePanel => {
                let keyboard = self.keyboard_input;
                self.panel.toggle(self.switch.target(), keyboard);
            }
            KeyAction::PrevScene | KeyAction::NextScene | KeyAction::SceneAt(_) => {
                let from = self.switch.target();
                let to = match action {
                    KeyAction::PrevScene => Some(from.prev()),
                    KeyAction::NextScene => Some(from.next()),
                    KeyAction::SceneAt(i) => SceneKind::selector().nth(i),
                    _ => None,
                };
                if let Some(to) = to {
                    self.switch.request(to);
                    self.panel.follow(self.switch.target()); // no-op while the panel is closed
                    self.last_auto_switch = Instant::now(); // a manual choice restarts the tour timer
                }
            }
            KeyAction::Media(cmd) => {
                let now = Instant::now();
                // Busy is silently dropped; Unsupported gets one rate-limited note (never titles).
                if self.media.send(cmd, now) == Dispatch::Unsupported
                    && self
                        .last_unsupported_note
                        .is_none_or(|t| now.duration_since(t) >= NOTE_EVERY)
                {
                    self.last_unsupported_note = Some(now);
                    eprintln!("media control: command not supported by the current source");
                }
            }
        }
    }

    fn surface(&self) -> (u32, u32) {
        self.gpu
            .as_ref()
            .map_or((0, 0), |g| (g.config.width, g.config.height))
    }

    /// Gives the control bar the current surface and DPI factor (before any pointer or key handling).
    fn sync_view(&mut self) {
        let dpi = self
            .window
            .as_ref()
            .map_or(1.0, |w| w.scale_factor() as f32);
        let s = self.surface();
        self.ctl.set_view(s, dpi);
        self.panel.set_view(s, dpi);
    }

    fn availability(&self) -> Availability {
        Availability::new(self.media.caps(), self.last_np.as_ref())
    }

    /// Key press. Tab / Shift+Tab move the bar's focus ring; Enter and Space activate the focused button (and
    /// only then: with no keyboard focus Space stays the global play/pause shortcut); everything else is the
    /// approved key map, identical whether the bar is visible or not.
    fn on_key(&mut self, el: &ActiveEventLoop, code: KeyCode, repeat: bool) {
        self.sync_view();
        self.ctl.key_activity();
        self.keyboard_input = true;
        if code == KeyCode::Escape {
            match self
                .esc
                .press(self.panel.is_open(), self.panel.visible(), repeat)
            {
                EscRoute::Panel => {
                    let ev = self.panel.key(PanelKey::Escape);
                    return self.on_panel_event(ev);
                }
                EscRoute::Swallow => return,
                EscRoute::App => {}
            }
        }
        // an open scene panel takes its keys first (Esc closes only the panel)
        if self.panel.is_open()
            && toggle_allowed(self.mods)
            && let Some(k) = panel_key(code, self.mods.shift_key(), repeat)
        {
            let ev = self.panel.key(k);
            return self.on_panel_event(ev);
        }
        // the focused progress line takes Left/Right/Home/End (key repeat allowed: holding an arrow keeps seeking)
        if toggle_allowed(self.mods)
            && self.ctl.progress_focused(&self.availability())
            && let Some(delta) = progress_key_delta(code, self.mods.shift_key())
        {
            let now = Instant::now();
            let target = self.last_np.as_ref().and_then(|n| {
                key_seek_seconds(
                    n.position_now(now).unwrap_or(f64::NAN),
                    delta,
                    n.duration_s?,
                )
            });
            if let Some(t) = target {
                self.media.send(MediaCommand::SeekTo(t), now);
            }
            return;
        }
        if !repeat && toggle_allowed(self.mods) {
            let av = self.availability();
            match code {
                KeyCode::Tab => return self.ctl.tab(self.mods.shift_key(), &av),
                KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::Space => {
                    if let Some(b) = self.ctl.activate_focused(&av) {
                        return self.run_button(el, b);
                    }
                }
                _ => {}
            }
        }
        if let Some(action) = key_action(code, self.mods, repeat) {
            self.run_action(el, action);
        }
    }

    /// What the panel decided: a selection is a scene request (the transition and its latest-wins rule live in
    /// `SceneSwitch`); a manual choice restarts the tour timer like the keys do.
    fn on_panel_event(&mut self, ev: PanelEvent) {
        if let PanelEvent::Select(k) = ev {
            self.switch.request(k);
            self.last_auto_switch = Instant::now();
        }
    }

    /// A bar button (click or keyboard activation): the same action as its key, but its own exit reason.
    fn run_button(&mut self, el: &ActiveEventLoop, b: ButtonId) {
        self.run_action(el, button_action(b));
        if b == ButtonId::Exit {
            self.exit_reason = "Exit button";
        }
    }

    fn on_left_button(&mut self, el: &ActiveEventLoop, down: bool) {
        self.sync_view();
        self.keyboard_input = false;
        // a click outside the control bar belongs to the open panel (a tile selects, empty space closes)
        if down
            && self.panel.is_open()
            && let Some(p) = self.pointer
            && !self.ctl.geometry().area.contains(p)
        {
            self.ctl.key_activity();
            let ev = self.panel.click(p);
            return self.on_panel_event(ev);
        }
        let av = self.availability();
        match self.ctl.mouse_left(down, &av) {
            Some(Click::Button(b)) => self.run_button(el, b),
            Some(Click::Seek(f)) => {
                if let Some(d) = self.last_np.as_ref().and_then(|n| n.duration_s) {
                    // same command path as the keys: capability-gated, rate-limited, never blocks
                    self.media
                        .send(MediaCommand::SeekTo(seek_seconds(f, d)), Instant::now());
                }
            }
            None => {}
        }
    }

    /// Borderless fullscreen on the monitor the window is on (never a maximised decorated window);
    /// leaving restores the normal window, and Windows restores the taskbar by itself.
    fn apply_fullscreen(&mut self) {
        let Some(w) = &self.window else { return };
        if self.opts.fullscreen {
            w.set_fullscreen(Some(Fullscreen::Borderless(w.current_monitor())));
        } else {
            w.set_fullscreen(None); // the cursor comes back through the idle-timer sync in about_to_wait
        }
    }

    fn frame(&mut self, el: &ActiveEventLoop) {
        let now = Instant::now();
        if self.start.elapsed().as_secs() >= self.opts.seconds {
            self.timed_out = true;
            self.exit_reason = "time limit reached (--seconds)";
            return el.exit();
        }
        let dt = now.duration_since(self.last_tick).as_secs_f64().min(0.25);
        self.last_tick = now;
        if let Some(s) = self.synth.as_mut() {
            s.advance(dt, &self.shared);
        }
        // Gather the newest analysis state (and onset events since the previous frame).
        let (frame, seq, onsets, strength, music) = match self.shared.lock() {
            Ok(mut s) => {
                let r = (s.frame, s.seq, s.onsets, s.onset_strength, s.music);
                s.onsets = 0;
                s.onset_strength = 0.0;
                r
            }
            Err(_) => (None, 0, 0, 0.0, MusicFeatures::default()),
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
        let mut params = self.mapper.update(
            &Input {
                frame: frame.as_ref().unwrap_or(&default_frame),
                onsets,
                onset_strength: strength,
                fresh,
                mode,
            },
            dt as f32,
        );
        self.mapper.observe_music(&music, fresh);
        if fresh {
            self.music_summary.add(self.mapper.music());
        }
        // Song information: polled a few times a second, decoded off-thread, applied when ready.
        let surface = self
            .gpu
            .as_ref()
            .map_or((0, 0), |g| (g.config.width, g.config.height));
        if self
            .motion_last
            .is_none_or(|t| now.duration_since(t) >= Duration::from_secs(1))
        {
            self.motion_last = Some(now);
            let r = crate::motion_pref::reduced_motion(self.opts.reduced_motion);
            if r != self.motion_reduced {
                self.motion_reduced = r;
                self.mapper.set_reduced_motion(r);
                self.pres.set_reduced_motion(r);
                self.switch.set_reduced(r);
                self.panel.set_reduced(r);
            }
        }
        let t_poll = Instant::now();
        if self
            .last_media_poll
            .is_none_or(|t| now.duration_since(t) >= Duration::from_millis(250))
        {
            self.last_media_poll = Some(now);
            self.last_np = self.media.poll(now);
            self.pres.on_now_playing(self.last_np.as_ref(), surface);
        }
        for job in self.pres.take_jobs() {
            self.worker.submit(job);
        }
        while let Some(r) = self.worker.try_recv() {
            match r {
                Ready::Clock { .. } => self.ctl.on_ready(r),
                Ready::PanelLabels { .. } => self.panel.on_ready(r),
                r => self.pres.on_ready(r),
            }
        }
        let poll_ms = t_poll.elapsed().as_secs_f64() * 1000.0;
        self.pres
            .set_position(self.last_np.as_ref().and_then(|n| n.position_now(now)));
        self.sync_view();
        let anchor = match self.switch.target() {
            SceneKind::Spectrum => 0.70,
            SceneKind::Aurora => 0.72,
            SceneKind::Orb | SceneKind::Fluid => 0.72,
            SceneKind::Focus if self.pres.layout() == Layout::Center => 0.80,
            SceneKind::Focus => 0.72,
        };
        // While the bar is visible the lyrics ride above it. The bar advances first so the lift follows this
        // frame's bar alpha (applied after the lyric easing: no lag, no overlap with a visible bar).
        // the scene panel: its "current" follows the switch's destination; while open it holds the one idle timer
        self.panel.set_current(self.switch.target());
        self.panel.advance(dt as f32);
        self.ctl.vis.set_hold(self.panel.is_open());
        self.ctl.advance(dt as f32);
        self.pres.set_lyric_anchor(anchor);
        self.pres.set_lyric_lift(self.ctl.lyric_lift(anchor));
        // Album Focus lays out above the bar zone; the Corner block rises over it while the bar shows. The bar's
        // visibility (0..1) is the ratio of its current lift to the full lift (always > 0).
        let dpi = self
            .window
            .as_ref()
            .map_or(1.0, |w| w.scale_factor() as f32);
        let full_lift = crate::controls::lyric_lift(anchor, surface, dpi);
        let bar_vis = (self.ctl.lyric_lift(anchor) / full_lift.max(1e-3)).clamp(0.0, 1.0);
        self.pres.set_bar(
            surface.1 as f32 - 128.0 * crate::controls::ui_scale(surface, dpi),
            bar_vis,
        );
        self.pres
            .set_album_scene(self.switch.target() == SceneKind::Focus, surface);
        let mut overlay_params = self
            .pres
            .update(dt as f32, surface, params.energy, params.swell);
        overlay_params.time = params.time;
        let av = self.availability();
        overlay_params.controls = self.ctl.frame(&FrameInput {
            dt: dt as f32,
            av,
            toggles: Toggles {
                lyrics: self.pres.lyrics_on(),
                artwork: self.pres.layout() != Layout::Off,
                fullscreen: self.opts.fullscreen,
            },
            position: self.last_np.as_ref().and_then(|n| n.position_now(now)),
            duration: self.last_np.as_ref().and_then(|n| n.duration_s),
            fg: overlay_params.colors.title,
            tint: overlay_params.accent,
        });
        overlay_params.panel = self.panel.frame(overlay_params.accent);
        for job in self.ctl.take_jobs() {
            self.worker.submit(job);
        }
        for job in self.panel.take_jobs() {
            self.worker.submit(job);
        }
        // Album Focus room light: the presentation owns the light rectangle (one source of truth with the cover
        // layout), the cover weights and the text pocket; the mapper owns the music fields.
        self.pres.merge_focus(&mut params.album);
        params.album.bar = bar_vis;
        if let Some(every) = self.opts.auto_scenes {
            let (last, due) = tour_step(self.last_auto_switch, now, every, self.panel.visible());
            self.last_auto_switch = last;
            if due {
                self.switch.request_with(self.switch.target().next(), FULL);
            }
        }
        let (album, album_w) = self.pres.album();
        apply_album(
            &mut params.palette,
            &album,
            album_w,
            0.85 + 0.30 * params.energy,
        );
        let tr = self
            .switch
            .update(dt as f32, |k| scene_luma(k, &params.palette));
        params.fx = tr.fx();
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
        let t_ops = Instant::now();
        if let Some(ov) = self.overlay.as_mut() {
            let ops: Vec<GpuOp> = self
                .pres
                .take_ops()
                .into_iter()
                .chain(self.ctl.take_ops())
                .chain(self.panel.take_ops())
                .collect();
            for op in ops {
                op.apply(ov, scene, &g.device, &g.queue);
            }
        }
        let ops_ms = t_ops.elapsed().as_secs_f64() * 1000.0;
        let queue = g.queue.clone();
        let overlay = self.overlay.as_ref();
        let t_render = Instant::now();
        match scene.render_scenes(g, &params, tr.scene, |enc, view| {
            if let Some(ov) = overlay {
                ov.draw(&queue, enc, view, &overlay_params);
            }
        }) {
            Ok(FrameStatus::Presented) => {
                let render_ms = t_render.elapsed().as_secs_f64() * 1000.0;
                if let Some((prev, [p_poll, p_ops, p_render], p_xfade)) = self.last_frame {
                    let interval = now.duration_since(prev).as_secs_f64() * 1000.0;
                    self.intervals_ms.push(interval);
                    // Rare long frames: [prev start, this start) is the previous frame's sections (CPU time; no
                    // track data is recorded) plus the event-loop wait before this frame.
                    if interval > HITCH_MS && self.hitches.len() < 24 {
                        self.hitches.push(format!(
                            "HITCH t={:.1}s interval={interval:.0}ms prev: poll+jobs={p_poll:.1} gpu_ops={p_ops:.1} render={p_render:.1} transition={p_xfade}",
                            self.start.elapsed().as_secs_f64(),
                        ));
                    }
                }
                self.last_frame =
                    Some((now, [poll_ms, ops_ms, render_ms], tr.phase != Phase::Idle));
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

/// `%USERPROFILE%\Music\Musigraph Lyrics` when it exists (never created, never written).
fn default_lyrics_dir() -> Option<std::path::PathBuf> {
    let d = std::path::PathBuf::from(std::env::var_os("USERPROFILE")?)
        .join("Music")
        .join("Musigraph Lyrics");
    d.is_dir().then_some(d)
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
            let mut tracker: Option<MusicTracker> = None;
            match Loopback::start(move |block, rate, t_end| {
                if analyzer.as_ref().map(Analyzer::sample_rate) != Some(rate) {
                    analyzer = Analyzer::new(rate, AnalysisParams::default()).ok();
                    tracker = analyzer.as_ref().map(Analyzer::music_tracker);
                }
                if let (Some(a), Some(tr)) = (analyzer.as_mut(), tracker.as_mut()) {
                    a.push(block, t_end, |f| publish(&sh, tr, f));
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
            let analyzer = Analyzer::new(sr, AnalysisParams::default()).expect("valid rate");
            synth = Some(SynthSource {
                tracker: analyzer.music_tracker(),
                analyzer,
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
    let media = match (opts.media, &opts.source) {
        (MediaChoice::Demo, _) => MediaFeed::demo(),
        (MediaChoice::Auto, Source::Live) => MediaFeed::live().unwrap_or_else(|e| {
            eprintln!("now-playing unavailable: {e}");
            MediaFeed::Off
        }),
        _ => MediaFeed::Off,
    };
    let layout = opts.layout;
    let reduced_motion = opts.reduced_motion;
    let lyric_source = match (&opts.lyrics_dir, opts.media) {
        (Some(d), _) => LyricSource::Dir(d.clone()),
        (None, MediaChoice::Demo) => LyricSource::Demo,
        (None, MediaChoice::Auto) if matches!(opts.source, Source::Live) => {
            default_lyrics_dir().map_or(LyricSource::None, LyricSource::Dir)
        }
        _ => LyricSource::None,
    };
    let style = opts.style;
    let controls_always = opts.controls == ControlsMode::Always;
    if style.lifecycle() == mg_gfx::scene::Lifecycle::Experimental {
        eprintln!("experimental scene: {}", style.name());
    }
    let tune = MapTuning::default();
    let stale_after = Duration::from_secs_f32(tune.stale_after_s);
    let mut app = App {
        opts,
        motion_last: None,
        motion_reduced: reduced_motion,
        shared,
        capture,
        synth,
        mapper: {
            let mut m = SceneMapper::new(tune);
            m.set_reduced_motion(reduced_motion);
            m
        },
        stale_after,
        scene: None,
        overlay: None,
        media,
        pres: {
            let mut p = Presentation::new(layout);
            p.set_lyric_source(lyric_source);
            p.set_reduced_motion(reduced_motion);
            p
        },
        worker: Worker::start(),
        last_media_poll: None,
        last_np: None,
        switch: {
            let mut s = SceneSwitch::new(style);
            s.set_reduced(reduced_motion);
            s
        },
        last_auto_switch: Instant::now(),
        gpu: None,
        window: None,
        start: Instant::now(),
        last_tick: Instant::now(),
        last_seq: 0,
        last_change: Instant::now(),
        ctl: ControlsUi::new(controls_always),
        panel: ScenePanel::new(),
        pointer: None,
        keyboard_input: false,
        esc: EscState::default(),
        cursor_hidden: false,
        mods: ModifiersState::empty(),
        last_unsupported_note: None,
        title: String::new(),
        last_frame: None,
        intervals_ms: Vec::new(),
        hitches: Vec::new(),
        presented: 0,
        skipped: 0,
        skip_since: None,
        stale_ticks: 0,
        music_summary: MusicSummary::default(),
        timed_out: false,
        result: Ok(()),
        exit_reason: "event loop ended",
        focused: true,
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("event loop error: {e}");
        return 1;
    }
    if let Some(c) = app.capture.as_mut() {
        let st = c.status();
        c.stop();
        println!(
            "audio: opens={} discontinuities={} real_packets={} real_audio_ms={} synth_blocks={} synth_silence_ms={} last_error={}",
            st.opens,
            st.discontinuities,
            st.real_packets,
            st.real_audio_ms,
            st.synthesized_silence_blocks,
            st.synthesized_silence_ms,
            st.last_error.as_deref().unwrap_or("-")
        );
    }
    for h in &app.hitches {
        println!("{h}");
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
    println!("{}", app.music_summary.line());
    let reason = if app.result.is_err() {
        "error (see RESULT: FAILED below)"
    } else {
        app.exit_reason
    };
    println!("EXIT reason: {reason}");
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
    fn progress_keys_seek_and_only_they_do() {
        assert_eq!(progress_key_delta(KeyCode::ArrowLeft, false), Some(-5.0));
        assert_eq!(progress_key_delta(KeyCode::ArrowRight, false), Some(5.0));
        assert_eq!(progress_key_delta(KeyCode::ArrowRight, true), Some(15.0));
        assert_eq!(progress_key_delta(KeyCode::ArrowLeft, true), Some(-15.0));
        assert_eq!(
            progress_key_delta(KeyCode::Home, false),
            Some(f64::NEG_INFINITY)
        );
        assert_eq!(progress_key_delta(KeyCode::End, false), Some(f64::INFINITY));
        for k in [
            KeyCode::Space,
            KeyCode::Enter,
            KeyCode::Tab,
            KeyCode::Escape,
            KeyCode::KeyF,
        ] {
            assert_eq!(
                progress_key_delta(k, false),
                None,
                "{k:?} keeps its meaning"
            );
        }
    }

    #[test]
    fn the_cursor_hides_only_in_focused_fullscreen_with_the_bar_away() {
        assert!(hide_cursor(true, true, false));
        assert!(
            !hide_cursor(true, false, false),
            "alt-tabbed away: cursor stays"
        );
        assert!(!hide_cursor(true, true, true));
        assert!(!hide_cursor(false, true, false));
    }

    #[test]
    fn every_button_runs_the_action_of_its_key_and_controls_flag_parses() {
        let none = ModifiersState::empty();
        let key = |c| key_action(c, none, false).unwrap();
        assert_eq!(button_action(ButtonId::PlayPause), key(KeyCode::Space));
        assert_eq!(button_action(ButtonId::Previous), key(KeyCode::ArrowLeft));
        assert_eq!(button_action(ButtonId::Next), key(KeyCode::ArrowRight));
        assert_eq!(button_action(ButtonId::Scene), KeyAction::TogglePanel);
        assert_eq!(ButtonId::Scene.name(), "Scenes");
        assert_eq!(button_action(ButtonId::Lyrics), key(KeyCode::KeyL));
        assert_eq!(button_action(ButtonId::Artwork), key(KeyCode::KeyA));
        assert_eq!(button_action(ButtonId::Fullscreen), key(KeyCode::KeyF));
        assert_eq!(button_action(ButtonId::Exit), key(KeyCode::Escape));
        // the key map itself does not change with the bar (Tab is handled before it and is not a mapped key)
        assert_eq!(key_action(KeyCode::Tab, none, false), None);
        assert_eq!(key_action(KeyCode::Enter, none, false), None);
        assert_eq!(
            parse(&a(&[]), Source::Live).unwrap().controls,
            ControlsMode::Auto
        );
        for (v, m) in [
            ("auto", ControlsMode::Auto),
            ("always", ControlsMode::Always),
        ] {
            assert_eq!(
                parse(&a(&["--controls", v]), Source::Live)
                    .unwrap()
                    .controls,
                m
            );
        }
        for bad in [&["--controls"][..], &["--controls", "never"]] {
            assert!(parse(&a(bad), Source::Live).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn key_map_matches_the_approved_layout() {
        use KeyAction::*;
        let none = ModifiersState::empty();
        let k = |c| key_action(c, none, false);
        assert_eq!(
            k(KeyCode::Space),
            Some(Media(MediaCommand::TogglePlayPause))
        );
        assert_eq!(k(KeyCode::ArrowLeft), Some(Media(MediaCommand::Previous)));
        assert_eq!(k(KeyCode::ArrowRight), Some(Media(MediaCommand::Next)));
        assert_eq!(k(KeyCode::BracketLeft), Some(PrevScene));
        assert_eq!(k(KeyCode::BracketRight), Some(NextScene));
        assert_eq!(k(KeyCode::Digit1), Some(SceneAt(0)));
        assert_eq!(k(KeyCode::Digit9), Some(SceneAt(8)));
        assert_eq!(k(KeyCode::Digit0), None);
        assert_eq!(k(KeyCode::KeyL), Some(ToggleLyrics));
        assert_eq!(k(KeyCode::KeyA), Some(CycleArtwork));
        assert_eq!(k(KeyCode::KeyF), Some(ToggleFullscreen));
        assert_eq!(k(KeyCode::Escape), Some(Exit));
        assert_eq!(k(KeyCode::KeyQ), None);
        assert_eq!(k(KeyCode::ArrowUp), None);
    }

    #[test]
    fn esc_distinguishes_the_press_that_closed_the_panel_from_a_new_press() {
        // open: Esc closes the panel, nothing else
        assert_eq!(route_esc(true, true, false, false), EscRoute::Panel);
        // the closing press is held: its auto-repeat (during or after the fade) never exits
        assert_eq!(route_esc(false, true, true, true), EscRoute::Swallow);
        assert_eq!(route_esc(false, false, true, true), EscRoute::Swallow);
        // a repeat during the fade without the held flag is swallowed too
        assert_eq!(route_esc(false, true, true, false), EscRoute::Swallow);
        // a NEW press (not a repeat, key released) exits as before, also while the panel still fades
        assert_eq!(route_esc(false, false, false, false), EscRoute::App);
        assert_eq!(route_esc(false, true, false, false), EscRoute::App);
        assert_eq!(
            key_action(KeyCode::Escape, ModifiersState::empty(), false),
            Some(KeyAction::Exit)
        );
        // sequence: press (closes), repeats (swallowed), release, new press (exits)
        let mut held = false;
        let mut routes = Vec::new();
        for (open, visible, repeat, release) in [
            (true, true, false, false),
            (false, true, true, false),
            (false, false, true, false),
            (false, false, false, true),
            (false, false, false, false),
        ] {
            if release {
                held = false;
                continue;
            }
            let r = route_esc(open, visible, repeat, held);
            if r == EscRoute::Panel {
                held = true;
            }
            routes.push(r);
        }
        assert_eq!(
            routes,
            [
                EscRoute::Panel,
                EscRoute::Swallow,
                EscRoute::Swallow,
                EscRoute::App
            ]
        );
    }

    /// Drives the production `EscState` (the one the window handler uses) through real event sequences.
    #[test]
    fn esc_state_swallows_only_the_closing_presses_repeats_and_a_release_or_focus_loss_ends_it() {
        // press closes the panel; its auto-repeats during and after the fade are swallowed; after the release a
        // new press exits the app
        let mut e = EscState::default();
        assert_eq!(e.press(true, true, false), EscRoute::Panel);
        assert_eq!(e.press(false, true, true), EscRoute::Swallow);
        assert_eq!(e.press(false, false, true), EscRoute::Swallow);
        assert_eq!(e.press(false, false, true), EscRoute::Swallow);
        e.release();
        assert_eq!(
            e.press(false, false, false),
            EscRoute::App,
            "new press after release"
        );
        // a new press while the panel still fades (key already released) also exits
        let mut e = EscState::default();
        assert_eq!(e.press(true, true, false), EscRoute::Panel);
        e.release();
        assert_eq!(e.press(false, true, false), EscRoute::App);
        // the window loses focus while Esc is held: no release event will come, a later press must still exit
        let mut e = EscState::default();
        assert_eq!(e.press(true, true, false), EscRoute::Panel);
        e.focus_lost();
        assert_eq!(e.press(false, false, false), EscRoute::App);
        // no panel involved: Esc is never held and always exits
        let mut e = EscState::default();
        assert_eq!(e.press(false, false, false), EscRoute::App);
        assert_eq!(e.press(false, false, true), EscRoute::App);
    }

    #[test]
    fn the_auto_tour_never_runs_or_catches_up_while_or_after_the_panel_is_open() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        assert_eq!(tour_step(t0, at(9), 10, false), (t0, false));
        assert_eq!(tour_step(t0, at(10), 10, false), (at(10), true));
        // 100 s with the panel up (60 fps): never due, the clock follows `now`
        let mut last = t0;
        let mut due_count = 0;
        for f in 0..6000u64 {
            let (l, d) = tour_step(last, t0 + Duration::from_millis(f * 1000 / 60), 10, true);
            last = l;
            due_count += u32::from(d);
        }
        assert_eq!(due_count, 0);
        // the panel closes: no immediate switch and no burst; one switch per full period afterwards
        let closed = t0 + Duration::from_millis(100_000);
        let mut when = Vec::new();
        for f in 0..(25 * 60u64) {
            let now = closed + Duration::from_millis(f * 1000 / 60);
            let (l, d) = tour_step(last, now, 10, false);
            last = l;
            if d {
                when.push((now - closed).as_millis());
            }
        }
        assert_eq!(when.len(), 2, "{when:?}");
        assert!(
            when[0] >= 9_900 && when[1] - when[0] >= 9_900,
            "{when:?} ms after closing"
        );
    }

    #[test]
    fn the_open_panel_takes_esc_arrows_tab_and_enter_before_the_global_map() {
        let k = |c, shift, repeat| panel_key(c, shift, repeat);
        assert_eq!(k(KeyCode::Escape, false, false), Some(PanelKey::Escape));
        assert_eq!(k(KeyCode::Tab, false, false), Some(PanelKey::Tab));
        assert_eq!(k(KeyCode::Tab, true, false), Some(PanelKey::BackTab));
        assert_eq!(
            k(KeyCode::ArrowLeft, false, true),
            Some(PanelKey::Left),
            "arrows repeat"
        );
        assert_eq!(k(KeyCode::ArrowDown, false, false), Some(PanelKey::Down));
        assert_eq!(k(KeyCode::Enter, false, false), Some(PanelKey::Enter));
        assert_eq!(k(KeyCode::NumpadEnter, false, false), Some(PanelKey::Enter));
        assert_eq!(
            k(KeyCode::Enter, false, true),
            None,
            "Enter does not repeat"
        );
        // everything else stays global while the panel is open: play/pause, [ ], digits, L, A, F
        for c in [
            KeyCode::Space,
            KeyCode::BracketLeft,
            KeyCode::BracketRight,
            KeyCode::Digit1,
            KeyCode::KeyL,
            KeyCode::KeyA,
            KeyCode::KeyF,
        ] {
            assert_eq!(k(c, false, false), None, "{c:?}");
            assert!(
                key_action(c, ModifiersState::empty(), false).is_some(),
                "{c:?}"
            );
        }
    }

    #[test]
    fn key_map_ignores_repeat_and_system_modifiers_but_esc_always_exits() {
        for code in [
            KeyCode::Space,
            KeyCode::ArrowLeft,
            KeyCode::BracketRight,
            KeyCode::Digit2,
            KeyCode::KeyF,
        ] {
            assert_eq!(
                key_action(code, ModifiersState::empty(), true),
                None,
                "repeat"
            );
            for m in [
                ModifiersState::ALT,
                ModifiersState::CONTROL,
                ModifiersState::SUPER,
            ] {
                assert_eq!(key_action(code, m, false), None, "{code:?} {m:?}");
            }
            assert!(key_action(code, ModifiersState::SHIFT, false).is_some());
        }
        assert_eq!(
            key_action(KeyCode::Escape, ModifiersState::ALT, true),
            Some(KeyAction::Exit)
        );
    }

    #[test]
    fn fullscreen_toggle_ignores_ctrl_alt_and_win_combinations() {
        assert!(toggle_allowed(ModifiersState::empty()));
        assert!(toggle_allowed(ModifiersState::SHIFT));
        assert!(!toggle_allowed(ModifiersState::CONTROL));
        assert!(!toggle_allowed(ModifiersState::ALT));
        assert!(!toggle_allowed(ModifiersState::SUPER));
        assert!(!toggle_allowed(
            ModifiersState::CONTROL | ModifiersState::SHIFT
        ));
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

    #[test]
    fn parses_style_layout_media_and_auto_scene_options() {
        let o = parse(
            &a(&[
                "--style",
                "aurora",
                "--layout",
                "center",
                "--demo-media",
                "--auto-scenes",
                "30",
                "--lyrics-dir",
                "lyrics-folder",
            ]),
            Source::Live,
        )
        .unwrap();
        assert_eq!(o.style, SceneKind::Aurora);
        assert_eq!(parse(&a(&[]), Source::Live).unwrap().auto_scenes, None);
        assert_eq!(o.layout, Layout::Center);
        assert_eq!(o.media, MediaChoice::Demo);
        assert_eq!(o.auto_scenes, Some(30));
        assert_eq!(o.lyrics_dir, Some("lyrics-folder".into()));
        for bad in [
            &["--style", "plasma"][..],
            &["--style"],
            &["--layout", "left"],
            &["--auto-scenes", "1"],
            &["--auto-scenes", "x"],
            &["--lyrics-dir"],
        ] {
            assert!(parse(&a(bad), Source::Live).is_err(), "{bad:?}");
        }
        assert_eq!(SceneKind::Focus.next(), SceneKind::Orb);
        assert_eq!(SceneKind::Orb.next(), SceneKind::Spectrum);
        assert_eq!(SceneKind::Spectrum.prev(), SceneKind::Orb);
        assert_eq!(
            parse(&a(&["--style", "orb"]), Source::Live).unwrap().style,
            SceneKind::Orb
        );
        assert_eq!(SceneKind::parse("focus"), Some(SceneKind::Focus));
    }

    #[test]
    fn parses_the_fluid_style_and_tier() {
        let o = parse(&a(&["--style", "fluid", "--fluid-tier", "3"]), Source::Live).unwrap();
        assert_eq!((o.style, o.fluid_tier), (SceneKind::Fluid, 3));
        assert_eq!(parse(&a(&[]), Source::Live).unwrap().fluid_tier, 1);
        for t in ["0", "1", "2"] {
            let tier = parse(&a(&["--fluid-tier", t]), Source::Live)
                .unwrap()
                .fluid_tier;
            assert_eq!(tier.to_string(), t);
        }
        for bad in [
            &["--fluid-tier"][..],
            &["--fluid-tier", "4"],
            &["--fluid-tier", "-1"],
            &["--fluid-tier", "high"],
        ] {
            assert!(parse(&a(bad), Source::Live).is_err(), "{bad:?}");
        }
    }
}
