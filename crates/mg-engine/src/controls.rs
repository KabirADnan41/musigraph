//! Temporary on-screen control bar (bottom-centre pill): state, layout, hit testing, keyboard focus.
//!
//! Pure logic first (everything is unit-tested without a window or GPU), then `ControlsUi`, which turns that
//! state into the `ControlsDraw` the overlay pass renders. Owns: when the bar is visible (the ONE idle timer
//! that also drives cursor hiding), where every control is, which controls are enabled, what a click means.
//! Does NOT own: what an action does (scene_run's `run_action`, the same code the keys use) or how it is drawn
//! (mg-gfx `overlay.wgsl`). A control that cannot work is drawn disabled and ignores clicks: nothing is faked.
//!
//! Sizes are design pixels at 1080p scaled by `ui_scale` (surface height, never below 0.85x the OS DPI factor so
//! the 44 px hit targets stay usable on small high-DPI windows, never wider than the window).

use crate::presentation::{GpuOp, Job, Ready};
use mg_gfx::overlay::{
    ButtonDraw, CLOCK_SLOT, CONTROL_BUTTONS, ControlsDraw, ProgressDraw, TextDraw, icon,
};
use mg_media::{Capabilities, MediaCommand, NowPlaying, PlaybackState};

// ---- design constants (px at 1080p) -------------------------------------------------------------------------
const BAR_W: f32 = 560.0;
const BAR_H: f32 = 64.0;
const BOTTOM: f32 = 28.0;
const HIT: f32 = 44.0;
const HIT_PLAY: f32 = 48.0;
const GAP: f32 = 12.0;
const SEP_GAP: f32 = 32.0;
/// Progress line centre above the pill's top edge.
const PROG_GAP: f32 = 20.0;
const PROG_INSET: f32 = 56.0;
const PROG_HIT_H: f32 = 24.0;
const TIME_GAP: f32 = 10.0;
const TEXT_PX: f32 = 15.0;
/// Space needed above the pill top for the progress line, its knob and the time texts.
const ZONE_ABOVE: f32 = 128.0 - BOTTOM - BAR_H;

// ---- timings ------------------------------------------------------------------------------------------------
pub const SHOW_FADE_S: f32 = 0.18;
pub const HIDE_FADE_S: f32 = 0.40;
pub const IDLE_S: f32 = 2.5;
/// Pointer must travel this far (px) from where it last counted as activity to show the bar.
pub const MOVE_PX: f32 = 4.0;
const RISE_PX: f32 = 8.0;
/// The lyrics are fully lifted whenever the bar's alpha is at least this (they never overlap a visible bar).
const LIFT_FULL_ALPHA: f32 = 0.05;
/// Once the bar is (nearly) gone the lyrics settle back down with this time constant instead of snapping.
const LIFT_SETTLE_S: f32 = 0.09;
/// Lyrics rise at least this much of the surface height while the bar is visible.
pub const LYRIC_LIFT: f32 = 0.067;
const HOVER_S: f32 = 0.12;
const PRESS_S: f32 = 0.08;
const DISABLED_OPACITY: f32 = 0.35;
/// Above this alpha the bar counts as visible for clicks and keys.
const INTERACTIVE_ALPHA: f32 = 0.5;
/// Keyboard seek steps on the focused progress line (Shift: the long step).
pub const SEEK_STEP_S: f64 = 5.0;
pub const SEEK_STEP_LONG_S: f64 = 15.0;

// ---- contrast (WCAG 2.x relative luminance; linear-light compositing like the shader) ------------------------
/// Pill fill (also the progress backing and the time-text scrims): the palette tint, darkened.
const BG_BASE: f32 = 0.003;
const BG_TINT: f32 = 0.03;
/// Pill / backing opacity: the smallest that keeps resting icons and the time texts >= 4.5:1 even over a white
/// scene (derived in `contrast_tests`).
pub const PILL_ALPHA: f32 = 0.88;
/// Icons, times and the progress line never get darker than this per channel (palette text colours vary).
const FG_MIN: f32 = 0.85;

// ---- buttons ------------------------------------------------------------------------------------------------

/// Every control, in visual (and Tab) order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonId {
    Previous,
    PlayPause,
    Next,
    Scene,
    Lyrics,
    Artwork,
    Fullscreen,
    Exit,
}

impl ButtonId {
    pub const ALL: [ButtonId; CONTROL_BUTTONS] = [
        ButtonId::Previous,
        ButtonId::PlayPause,
        ButtonId::Next,
        ButtonId::Scene,
        ButtonId::Lyrics,
        ButtonId::Artwork,
        ButtonId::Fullscreen,
        ButtonId::Exit,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    /// Accessible name (tooltips and a future screen-reader bridge; the icon is the only visible label).
    #[allow(dead_code)] // read by the tooltip slice (the names are tested here)
    pub fn name(self) -> &'static str {
        match self {
            ButtonId::Previous => "Previous track",
            ButtonId::PlayPause => "Play or pause",
            ButtonId::Next => "Next track",
            ButtonId::Scene => "Next scene",
            ButtonId::Lyrics => "Show or hide lyrics",
            ButtonId::Artwork => "Change artwork layout",
            ButtonId::Fullscreen => "Toggle fullscreen",
            ButtonId::Exit => "Exit",
        }
    }
}

/// What the current media source lets the user do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Availability {
    pub caps: Capabilities,
    pub state: PlaybackState,
    /// A position and a positive duration are known (the progress line exists).
    pub has_timeline: bool,
}

impl Availability {
    pub fn new(caps: Capabilities, np: Option<&NowPlaying>) -> Self {
        Self {
            caps,
            state: np.map_or(PlaybackState::Other, |n| n.state),
            has_timeline: np.is_some_and(|n| {
                n.position_s.is_some() && n.duration_s.is_some_and(|d| d.is_finite() && d > 0.0)
            }),
        }
    }

    /// The progress line reacts to clicks only when the session can seek and a timeline exists.
    pub fn can_seek(&self) -> bool {
        self.caps.can_seek && self.has_timeline
    }
}

/// Media buttons follow the session's capabilities (play/pause also via the state-based fallback the command
/// path uses); the view buttons are always available.
pub fn enabled(id: ButtonId, a: &Availability) -> bool {
    match id {
        ButtonId::Previous => a.caps.can_previous,
        ButtonId::Next => a.caps.can_next,
        ButtonId::PlayPause => a
            .caps
            .resolve(MediaCommand::TogglePlayPause, a.state)
            .is_some(),
        _ => true,
    }
}

/// Active (toggled on) states of the view buttons.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Toggles {
    pub lyrics: bool,
    pub artwork: bool,
    pub fullscreen: bool,
}

impl Toggles {
    fn on(&self, id: ButtonId) -> bool {
        match id {
            ButtonId::Lyrics => self.lyrics,
            ButtonId::Artwork => self.artwork,
            ButtonId::Fullscreen => self.fullscreen,
            _ => false,
        }
    }
}

// ---- layout -------------------------------------------------------------------------------------------------

/// Pixels per design pixel for a surface and OS DPI factor (see the module docs).
pub fn ui_scale(surface: (u32, u32), dpi: f32) -> f32 {
    let (w, h) = (surface.0.max(1) as f32, surface.1.max(1) as f32);
    let dpi = if dpi.is_finite() {
        dpi.clamp(1.0, 4.0)
    } else {
        1.0
    };
    (h / 1080.0)
        .max(0.85 * dpi)
        .min(w / (BAR_W + 40.0))
        .max(0.1)
}

/// Top-left based rectangle, surface pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    /// Half-open: the left/top edge is inside, the right/bottom edge is not.
    pub fn contains(&self, p: (f32, f32)) -> bool {
        p.0 >= self.x && p.0 < self.x + self.w && p.1 >= self.y && p.1 < self.y + self.h
    }

    fn cx(&self) -> f32 {
        self.x + 0.5 * self.w
    }

    fn cy(&self) -> f32 {
        self.y + 0.5 * self.h
    }
}

/// Where everything is for one surface size (visibility-independent; the rise animation is added when drawing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub scale: f32,
    pub bar: Rect,
    pub buttons: [Rect; CONTROL_BUTTONS],
    /// Divider x between the transport group and the view group.
    pub sep_x: f32,
    /// Progress track centre line and hit rectangle.
    pub prog_y: f32,
    pub prog_x0: f32,
    pub prog_x1: f32,
    pub prog_hit: Rect,
    /// Hover area: pill plus the progress zone above it.
    pub area: Rect,
}

pub fn layout(surface: (u32, u32), dpi: f32) -> Geometry {
    let s = ui_scale(surface, dpi);
    let (w, h) = (surface.0 as f32, surface.1 as f32);
    let bar = Rect {
        x: 0.5 * (w - BAR_W * s),
        y: h - (BOTTOM + BAR_H) * s,
        w: BAR_W * s,
        h: BAR_H * s,
    };
    let sizes = ButtonId::ALL.map(|b| {
        if b == ButtonId::PlayPause {
            HIT_PLAY
        } else {
            HIT
        }
    });
    let group1: f32 = sizes[..3].iter().sum::<f32>() + 2.0 * GAP;
    let group2: f32 = sizes[3..].iter().sum::<f32>() + 4.0 * GAP;
    let total = group1 + SEP_GAP + group2;
    let mut x = bar.x + 0.5 * (BAR_W - total) * s;
    let cy = bar.cy();
    let mut buttons = [Rect::default(); CONTROL_BUTTONS];
    let mut sep_x = 0.0;
    for (i, sz) in sizes.iter().enumerate() {
        if i == 3 {
            sep_x = x + (0.5 * SEP_GAP - GAP) * s;
            x += (SEP_GAP - GAP) * s;
        }
        buttons[i] = Rect {
            x,
            y: cy - 0.5 * sz * s,
            w: sz * s,
            h: sz * s,
        };
        x += (sz + GAP) * s;
    }
    let prog_y = bar.y - PROG_GAP * s;
    let (prog_x0, prog_x1) = (bar.x + PROG_INSET * s, bar.x + (BAR_W - PROG_INSET) * s);
    Geometry {
        scale: s,
        bar,
        buttons,
        sep_x,
        prog_y,
        prog_x0,
        prog_x1,
        prog_hit: Rect {
            x: prog_x0,
            y: prog_y - 0.5 * PROG_HIT_H * s,
            w: prog_x1 - prog_x0,
            h: PROG_HIT_H * s,
        },
        area: Rect {
            x: bar.x - 8.0 * s,
            y: bar.y - ZONE_ABOVE * s,
            w: bar.w + 16.0 * s,
            h: bar.h + (ZONE_ABOVE + 8.0) * s,
        },
    }
}

/// Which button (if any) is under the point. Gaps, the divider and everything outside return `None`.
pub fn hit_test(g: &Geometry, p: (f32, f32)) -> Option<ButtonId> {
    ButtonId::ALL
        .into_iter()
        .find(|b| g.buttons[b.index()].contains(p))
}

/// Played fraction (0..1) for a point on the progress hit area, else `None`.
pub fn progress_hit(g: &Geometry, p: (f32, f32)) -> Option<f32> {
    g.prog_hit
        .contains(p)
        .then(|| ((p.0 - g.prog_x0) / (g.prog_x1 - g.prog_x0).max(1.0)).clamp(0.0, 1.0))
}

/// Seek target (seconds) for a clicked fraction: never the very end (that would roll over to the next track).
pub fn seek_seconds(frac: f32, duration: f64) -> f64 {
    if !(duration.is_finite() && duration > 0.0) || !frac.is_finite() {
        return 0.0;
    }
    (f64::from(frac.clamp(0.0, 1.0)) * duration).clamp(0.0, (duration - 0.25).max(0.0))
}

/// Fraction of the surface height the lyrics must rise while the bar is visible: at least `LYRIC_LIFT`, more
/// when the lowest lyric line at `anchor` would still touch the bar's zone (progress line and times included).
pub fn lyric_lift(anchor: f32, surface: (u32, u32), dpi: f32) -> f32 {
    if !anchor.is_finite() {
        return LYRIC_LIFT;
    }
    let h = surface.1.max(1) as f32;
    let zone_top = h - 128.0 * ui_scale(surface, dpi);
    let clearance = 8.0 * ui_scale(surface, dpi);
    let lowest = crate::lyric_view::lowest_ink_frac(anchor) * h;
    ((lowest - zone_top + clearance) / h).clamp(LYRIC_LIFT, 0.30)
}

/// Whole seconds shown for a time; unknown or negative values are 0.
fn whole_secs(secs: f64) -> u64 {
    if secs.is_finite() && secs > 0.0 {
        secs as u64
    } else {
        0
    }
}

/// Seek target for a key on the focused progress line: `delta` seconds from `position` (Home/End pass
/// `f64::NEG_INFINITY` / `f64::INFINITY`), clamped to 0..=duration-0.25 like a click (never rolls over).
pub fn key_seek_seconds(position: f64, delta: f64, duration: f64) -> Option<f64> {
    if !(duration.is_finite() && duration > 0.0) || delta.is_nan() {
        return None;
    }
    let p = if position.is_finite() { position } else { 0.0 };
    Some((p + delta).clamp(0.0, (duration - 0.25).max(0.0)))
}

/// "m:ss" (or "h:mm:ss"); unknown or negative values show 0:00.
pub fn fmt_time(secs: f64) -> String {
    let t = whole_secs(secs);
    if t >= 3600 {
        format!("{}:{:02}:{:02}", t / 3600, t % 3600 / 60, t % 60)
    } else {
        format!("{}:{:02}", t / 60, t % 60)
    }
}

/// Pill fill (linear RGB) for a palette tint: the tint, darkened (also the progress backing and time scrims).
fn pill_bg(tint: [f32; 3]) -> [f32; 3] {
    tint.map(|c| BG_BASE + BG_TINT * c.clamp(0.0, 1.0))
}

// ---- visibility state machine --------------------------------------------------------------------------------

/// When the bar is visible. One idle timer for bar and cursor. Alpha moves along its easing curve in *progress*
/// space (fade in: ease-out-cubic over 180 ms; fade out: ease-in-quad over 400 ms), so reversing mid-fade
/// continues from the current alpha instead of restarting, and the result does not depend on the frame rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Visibility {
    alpha: f32,
    wants: bool,
    idle: f32,
    hovering: bool,
    pressed: bool,
    always: bool,
    lift: f32,
}

fn smoothstep(p: f32) -> f32 {
    p * p * (3.0 - 2.0 * p)
}

impl Visibility {
    /// `always`: the bar never hides (the `--controls always` developer flag).
    pub fn new(always: bool) -> Self {
        Self {
            alpha: 0.0,
            wants: always,
            idle: IDLE_S,
            hovering: false,
            pressed: false,
            always,
            lift: 0.0,
        }
    }

    /// Any key press, click, or pointer movement beyond the threshold.
    pub fn activity(&mut self) {
        self.idle = 0.0;
        self.wants = true;
    }

    /// Pointer over the bar area: the bar never hides while this holds.
    pub fn set_hovering(&mut self, on: bool) {
        self.hovering = on;
    }

    /// A mouse button is held: the bar never hides while this holds.
    pub fn set_pressed(&mut self, on: bool) {
        self.pressed = on;
    }

    pub fn tick(&mut self, dt: f32) {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.25)
        } else {
            0.0
        };
        if self.always {
            self.wants = true;
        }
        // The spec's "4 s while the pointer is over the bar" is subsumed by "never hide while hovering":
        // the 2.5 s countdown starts when the pointer leaves or the button is released.
        if self.hovering || self.pressed || self.always {
            self.idle = 0.0;
        } else {
            self.idle += dt;
        }
        if self.idle >= IDLE_S {
            self.wants = false;
        }
        if self.wants {
            let t = 1.0 - (1.0 - self.alpha).max(0.0).cbrt();
            let t = (t + dt / SHOW_FADE_S).min(1.0);
            self.alpha = 1.0 - (1.0 - t).powi(3);
        } else {
            let u = (1.0 - self.alpha).max(0.0);
            let u = u.sqrt().min(1.0);
            let u = (u + dt / HIDE_FADE_S).min(1.0);
            self.alpha = 1.0 - u * u;
        }
        // The lift follows the bar's real alpha: never below what the bar needs (full from LIFT_FULL_ALPHA up);
        // only once the bar is nearly gone does it settle back down, smoothly.
        let need = smoothstep((self.alpha / LIFT_FULL_ALPHA).clamp(0.0, 1.0));
        self.lift = if need >= self.lift {
            need
        } else {
            need.max(self.lift * (-dt / LIFT_SETTLE_S).exp())
        };
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Bar wants to be (or is) shown; the cursor is hidden only when this is false.
    pub fn wants_visible(&self) -> bool {
        self.wants
    }

    pub fn fully_hidden(&self) -> bool {
        !self.wants && self.alpha <= 0.0
    }

    /// Clicks and keys act on the bar only while it is really visible.
    pub fn interactive(&self) -> bool {
        self.alpha >= INTERACTIVE_ALPHA
    }

    /// 0..1: how far the lyrics have risen (1 whenever the bar's alpha is >= `LIFT_FULL_ALPHA`).
    pub fn lift(&self) -> f32 {
        self.lift.clamp(0.0, 1.0)
    }

    /// 8 px rise (towards the bottom while hidden), in design px.
    fn rise(&self) -> f32 {
        RISE_PX * (1.0 - self.alpha)
    }
}

// ---- keyboard focus -----------------------------------------------------------------------------------------

/// Keyboard focus across the progress line (when seeking is possible) and the enabled buttons. The ring is
/// shown only when focus came from the keyboard.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Focus {
    cur: Option<ButtonId>,
    /// The progress line has focus (then `cur` is `None`).
    prog: bool,
    keyboard: bool,
}

impl Focus {
    /// Tab (`back` = Shift+Tab) over the buttons only (see `tab_with`).
    #[cfg(test)]
    pub fn tab(&mut self, back: bool, ok: impl Fn(ButtonId) -> bool) {
        self.tab_with(back, ok, false);
    }

    /// Tab (`back` = Shift+Tab): next/previous focusable, wrapping around: the enabled buttons left to right,
    /// then the progress line (only when `prog_ok`; last, so Tab from nothing still starts at the transport).
    pub fn tab_with(&mut self, back: bool, ok: impl Fn(ButtonId) -> bool, prog_ok: bool) {
        let nb = ButtonId::ALL.len();
        let n = nb + 1; // slot i < nb = button i, slot nb = progress
        let cur = if self.prog {
            Some(nb)
        } else {
            self.cur.map(ButtonId::index)
        };
        let start = match (cur, back) {
            (Some(c), false) => c + 1,
            (Some(c), true) => c + n - 1,
            (None, false) => 0,
            (None, true) => n - 1,
        };
        let slot = (0..n)
            .map(|k| if back { start + n - k } else { start + k } % n)
            .find(|&i| {
                if i == nb {
                    prog_ok
                } else {
                    ok(ButtonId::ALL[i])
                }
            });
        self.prog = slot == Some(nb);
        self.cur = slot.filter(|&i| i < nb).map(|i| ButtonId::ALL[i]);
        self.keyboard = slot.is_some();
    }

    /// The progress line has keyboard focus (draw its ring; arrows seek).
    pub fn progress_ring(&self) -> bool {
        self.prog && self.keyboard
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Drops focus when its button has become disabled.
    pub fn retain(&mut self, ok: impl Fn(ButtonId) -> bool) {
        if self.cur.is_some_and(|b| !ok(b)) {
            self.clear();
        }
    }

    /// Drops progress focus when seeking has become unavailable.
    fn retain_progress(&mut self, ok: bool) {
        if self.prog && !ok {
            self.clear();
        }
    }

    /// The button to draw the ring on.
    pub fn ring(&self) -> Option<ButtonId> {
        self.cur.filter(|_| self.keyboard)
    }
}

// ---- time texts ---------------------------------------------------------------------------------------------

/// Elapsed / duration strings rendered through the DirectWrite worker, at most one request per slot per change
/// (so once per second for the elapsed time), only while the bar is shown.
#[derive(Default)]
struct Clock {
    /// Whole seconds last requested per slot (the string is formatted only when this changes: no per-frame
    /// allocation).
    asked: [Option<u64>; 2],
    px: f32,
    serial: [u64; 2],
    dims: [Option<(u32, u32, u32)>; 2],
}

impl Clock {
    fn want(
        &mut self,
        texts: [Option<u64>; 2],
        px: f32,
        jobs: &mut Vec<Job>,
        ops: &mut Vec<GpuOp>,
    ) {
        let resized = (px / self.px.max(1.0) - 1.0).abs() > 0.1;
        for (i, text) in texts.into_iter().enumerate() {
            if text == self.asked[i] && !resized {
                continue;
            }
            self.serial[i] += 1;
            match text {
                Some(t) => jobs.push(Job::Clock {
                    serial: self.serial[i],
                    slot: i,
                    text: fmt_time(t as f64),
                    px,
                }),
                None => {
                    if self.dims[i].take().is_some() {
                        ops.push(GpuOp::ClearText(CLOCK_SLOT + i));
                    }
                }
            }
            self.asked[i] = text;
        }
        self.px = px;
    }

    fn on_ready(&mut self, r: Ready, ops: &mut Vec<GpuOp>) {
        let Ready::Clock {
            serial,
            slot,
            bitmap,
        } = r
        else {
            return;
        };
        if slot > 1 || serial != self.serial[slot] {
            return; // superseded
        }
        match bitmap {
            Some(b) => {
                self.dims[slot] = Some((b.width, b.height, b.pad));
                ops.push(GpuOp::SetText(CLOCK_SLOT + slot, b));
            }
            None => {
                if self.dims[slot].take().is_some() {
                    ops.push(GpuOp::ClearText(CLOCK_SLOT + slot));
                }
            }
        }
    }
}

// ---- the UI ----------------------------------------------------------------------------------------------------

/// What a pointer click resolved to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Click {
    Button(ButtonId),
    /// Seek to this played fraction (only produced when the session can seek).
    Seek(f32),
}

/// Per-frame inputs of the bar.
pub struct FrameInput {
    pub dt: f32,
    pub av: Availability,
    pub toggles: Toggles,
    pub position: Option<f64>,
    pub duration: Option<f64>,
    /// Linear RGB of the overlay text colour and the palette-derived tint.
    pub fg: [f32; 3],
    pub tint: [f32; 3],
}

pub struct ControlsUi {
    pub vis: Visibility,
    focus: Focus,
    geom: Geometry,
    surface: (u32, u32),
    dpi: f32,
    cursor: Option<(f32, f32)>,
    /// Where the pointer last counted as activity (movement is measured from here, so slow drifts count).
    anchor: Option<(f32, f32)>,
    pressed: Option<ButtonId>,
    hover: [f32; CONTROL_BUTTONS],
    press: [f32; CONTROL_BUTTONS],
    prog_hover: f32,
    clock: Clock,
    jobs: Vec<Job>,
    ops: Vec<GpuOp>,
    /// `advance` already ran for this frame (so `frame` does not tick a second time).
    advanced: bool,
}

fn step_to(v: &mut f32, target: f32, dt: f32, dur: f32) {
    let s = dt / dur;
    *v += (target - *v).clamp(-s, s);
}

impl ControlsUi {
    pub fn new(always: bool) -> Self {
        Self {
            vis: Visibility::new(always),
            focus: Focus::default(),
            geom: layout((1920, 1080), 1.0),
            surface: (1920, 1080),
            dpi: 1.0,
            cursor: None,
            anchor: None,
            pressed: None,
            hover: [0.0; CONTROL_BUTTONS],
            press: [0.0; CONTROL_BUTTONS],
            prog_hover: 0.0,
            clock: Clock::default(),
            jobs: Vec::new(),
            ops: Vec::new(),
            advanced: false,
        }
    }

    /// Advances the visibility by `dt` ahead of `frame`, so the lyric lift of this frame already follows this
    /// frame's bar alpha (the lyrics are laid out before the bar is drawn). Optional: `frame` ticks otherwise.
    pub fn advance(&mut self, dt: f32) {
        let hover_area = self.cursor.is_some_and(|p| self.geom.area.contains(p));
        self.vis.set_hovering(hover_area);
        self.vis.tick(dt);
        self.advanced = true;
    }

    /// The window lost focus: the same as the pointer leaving (no mouse button can stay held, so an alt-tab
    /// while pressing cannot pin the bar), and keyboard focus mode ends.
    pub fn focus_lost(&mut self) {
        self.pointer_left();
        self.vis.set_hovering(false);
        self.focus.clear();
        self.press = [0.0; CONTROL_BUTTONS];
    }

    /// The progress line has keyboard focus while the bar is visible and seeking works: Left/Right/Home/End
    /// seek instead of changing tracks.
    pub fn progress_focused(&self, av: &Availability) -> bool {
        self.vis.interactive() && self.focus.progress_ring() && av.can_seek()
    }

    /// Current surface and DPI factor; recomputes the layout when they changed.
    pub fn set_view(&mut self, surface: (u32, u32), dpi: f32) {
        if surface != self.surface || dpi != self.dpi {
            self.surface = surface;
            self.dpi = dpi;
            self.geom = layout(surface, dpi);
        }
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geom
    }

    pub fn take_jobs(&mut self) -> Vec<Job> {
        std::mem::take(&mut self.jobs)
    }

    pub fn take_ops(&mut self) -> Vec<GpuOp> {
        std::mem::take(&mut self.ops)
    }

    pub fn on_ready(&mut self, r: Ready) {
        self.clock.on_ready(r, &mut self.ops);
    }

    pub fn lyric_lift(&self, anchor: f32) -> f32 {
        lyric_lift(anchor, self.surface, self.dpi) * self.vis.lift()
    }

    /// Any key press (including Tab).
    pub fn key_activity(&mut self) {
        self.vis.activity();
    }

    pub fn pointer_moved(&mut self, pos: (f32, f32)) {
        self.cursor = Some(pos);
        let far = self.anchor.is_none_or(|a| {
            let (dx, dy) = (pos.0 - a.0, pos.1 - a.1);
            dx * dx + dy * dy > MOVE_PX * MOVE_PX
        });
        if far {
            self.anchor = Some(pos);
            self.vis.activity();
            // real mouse use ends keyboard focus mode (focus-visible): the ring hides, Space is global again
            self.focus.clear();
        }
    }

    pub fn pointer_left(&mut self) {
        self.cursor = None;
        self.anchor = None;
        self.pressed = None;
        self.vis.set_pressed(false);
    }

    /// Left button pressed or released. A button click completes on release over the same enabled button; the
    /// progress line seeks on press. Never acts while the bar is not really visible (that click only reveals it).
    pub fn mouse_left(&mut self, down: bool, av: &Availability) -> Option<Click> {
        if down {
            let live = self.vis.interactive();
            self.vis.activity();
            self.vis.set_pressed(true);
            let p = self.cursor?;
            if !live {
                return None;
            }
            if let Some(b) = hit_test(&self.geom, p) {
                self.focus.clear(); // pointer use ends keyboard focus mode
                if enabled(b, av) {
                    self.pressed = Some(b);
                }
                return None;
            }
            if av.can_seek() {
                return progress_hit(&self.geom, p).map(Click::Seek);
            }
            None
        } else {
            self.vis.set_pressed(false);
            self.vis.activity();
            let was = self.pressed.take()?;
            let over = self.cursor.and_then(|p| hit_test(&self.geom, p));
            (over == Some(was) && enabled(was, av) && self.vis.interactive())
                .then_some(Click::Button(was))
        }
    }

    /// Tab / Shift+Tab: shows the bar and moves the focus ring across the enabled buttons and (when seeking
    /// works) the progress line.
    pub fn tab(&mut self, back: bool, av: &Availability) {
        self.vis.activity();
        self.focus.tab_with(back, |b| enabled(b, av), av.can_seek());
    }

    /// Enter / Space: the focused button, when the bar is visible, focus came from the keyboard and the
    /// button is enabled. `None` lets Space fall through to the global play/pause shortcut.
    pub fn activate_focused(&mut self, av: &Availability) -> Option<ButtonId> {
        self.vis.activity();
        if !self.vis.interactive() {
            return None;
        }
        self.focus.ring().filter(|b| enabled(*b, av))
    }

    /// Advances all state by `dt` and returns what to draw (`None` while hidden: zero draw cost).
    pub fn frame(&mut self, i: &FrameInput) -> Option<ControlsDraw> {
        let g = self.geom;
        if !std::mem::take(&mut self.advanced) {
            self.advance(i.dt);
            self.advanced = false;
        }
        if self.vis.fully_hidden() {
            self.focus.clear();
            self.pressed = None;
            self.hover = [0.0; CONTROL_BUTTONS];
            self.press = [0.0; CONTROL_BUTTONS];
            self.prog_hover = 0.0;
        }
        let want_texts = !self.vis.fully_hidden() && i.av.has_timeline;
        let duration = i.duration.filter(|d| *d > 0.0);
        let texts = if want_texts {
            [
                // elapsed never shows more than the duration (positions can overshoot at the end of a track)
                i.position
                    .map(|p| whole_secs(duration.map_or(p, |d| p.min(d)))),
                duration.map(whole_secs),
            ]
        } else {
            [None, None]
        };
        self.clock
            .want(texts, TEXT_PX * g.scale, &mut self.jobs, &mut self.ops);
        let a = self.vis.alpha();
        if a <= 0.002 {
            return None;
        }
        self.focus.retain(|b| enabled(b, &i.av));
        self.focus.retain_progress(i.av.can_seek());
        let live = self.vis.interactive();
        let over = self.cursor.and_then(|p| hit_test(&g, p)).filter(|_| live);
        let prog_over =
            i.av.can_seek() && live && self.cursor.is_some_and(|p| progress_hit(&g, p).is_some());
        step_to(
            &mut self.prog_hover,
            f32::from(u8::from(prog_over)),
            i.dt,
            HOVER_S,
        );
        let dy = self.vis.rise() * g.scale;
        let ring = self.focus.ring();
        let buttons = std::array::from_fn(|k| {
            let id = ButtonId::ALL[k];
            let ok = enabled(id, &i.av);
            step_to(
                &mut self.hover[k],
                f32::from(u8::from(ok && over == Some(id))),
                i.dt,
                HOVER_S,
            );
            step_to(
                &mut self.press[k],
                f32::from(u8::from(ok && self.pressed == Some(id) && over == Some(id))),
                i.dt,
                PRESS_S,
            );
            let r = g.buttons[k];
            ButtonDraw {
                cx: r.cx(),
                cy: r.cy() + dy,
                half: 0.5 * r.w,
                icon: match id {
                    ButtonId::Previous => icon::PREV,
                    ButtonId::PlayPause if i.av.state == PlaybackState::Playing => icon::PAUSE,
                    ButtonId::PlayPause => icon::PLAY,
                    ButtonId::Next => icon::NEXT,
                    ButtonId::Scene => icon::SCENE,
                    ButtonId::Lyrics => icon::LYRICS,
                    ButtonId::Artwork => icon::LAYOUT,
                    ButtonId::Fullscreen => icon::FULLSCREEN,
                    ButtonId::Exit => icon::EXIT,
                },
                opacity: if ok { 1.0 } else { DISABLED_OPACITY },
                hover: self.hover[k],
                toggled: f32::from(u8::from(ok && i.toggles.on(id))),
                focus: ring == Some(id),
                press: self.press[k],
            }
        });
        let s = g.scale;
        let progress = (i.av.has_timeline && i.duration.is_some_and(|d| d > 0.0)).then(|| {
            let d = i.duration.unwrap_or(1.0);
            ProgressDraw {
                x0: g.prog_x0,
                x1: g.prog_x1,
                y: g.prog_y + dy,
                half_thick: (1.5 + 1.5 * self.prog_hover) * s,
                frac: (i.position.unwrap_or(0.0) / d).clamp(0.0, 1.0) as f32,
                hover: self.prog_hover,
                focus: self.focus.progress_ring(),
            }
        });
        let text_rect = |slot: usize, right_aligned: bool| {
            let (w, h, pad) = self.clock.dims[slot]?;
            progress?;
            let (w, h, pad) = (w as f32, h as f32, pad as f32);
            let x = if right_aligned {
                g.prog_x0 - TIME_GAP * s - w + pad
            } else {
                g.prog_x1 + TIME_GAP * s - pad
            };
            Some(TextDraw {
                x,
                y: g.prog_y + dy - 0.5 * h,
                w,
                h,
                alpha: 1.0,
            })
        };
        let tint = i.tint;
        let peak = tint.iter().copied().fold(0.0f32, f32::max).max(1e-3);
        Some(ControlsDraw {
            alpha: a,
            scale: s,
            bar: [g.bar.cx(), g.bar.cy() + dy, 0.5 * g.bar.w, 0.5 * g.bar.h],
            sep: [g.sep_x, g.bar.cy() + dy, 0.28 * g.bar.h],
            buttons,
            progress,
            elapsed: text_rect(0, true),
            duration: text_rect(1, false),
            time_alpha: 1.0,
            bg: pill_bg(tint),
            bg_alpha: PILL_ALPHA,
            fg: i.fg.map(|c| c.max(FG_MIN)),
            // accent: the palette tint lifted to a readable brightness
            accent: tint.map(|c| (0.30 + 0.70 * c / peak).min(1.0)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HD: (u32, u32) = (1920, 1080);

    fn av(caps: Capabilities, state: PlaybackState, timeline: bool) -> Availability {
        Availability {
            caps,
            state,
            has_timeline: timeline,
        }
    }

    fn all() -> Availability {
        av(Capabilities::all(), PlaybackState::Playing, true)
    }

    fn off() -> Availability {
        av(Capabilities::default(), PlaybackState::Other, false)
    }

    fn run(v: &mut Visibility, secs: f32, fps: f32) {
        let n = (secs * fps).round() as usize;
        for _ in 0..n {
            v.tick(1.0 / fps);
        }
    }

    // ---- visibility state machine --------------------------------------------------------------------------

    #[test]
    fn starts_hidden_shows_on_activity_in_180ms_and_hides_after_idle() {
        let mut v = Visibility::new(false);
        run(&mut v, 1.0, 60.0);
        assert!(v.fully_hidden() && v.alpha() == 0.0 && !v.wants_visible());
        v.activity();
        run(&mut v, 0.18, 1000.0);
        assert!(v.alpha() > 0.999, "180 ms ease-out-cubic: {}", v.alpha());
        // ease-OUT: at 90 ms it is already past 85 %
        let mut w = Visibility::new(false);
        w.activity();
        run(&mut w, 0.09, 1000.0);
        assert!(w.alpha() > 0.85, "{}", w.alpha());
        // idle: still wanted at 2.4 s, unwanted just after 2.5 s, fully gone 0.4 s later
        let mut v = Visibility::new(false);
        v.activity();
        run(&mut v, 2.4, 100.0);
        assert!(v.wants_visible() && v.alpha() > 0.99);
        run(&mut v, 0.12, 100.0);
        assert!(!v.wants_visible(), "hide countdown started");
        run(&mut v, 0.45, 100.0);
        assert!(v.fully_hidden(), "400 ms fade out: {}", v.alpha());
    }

    #[test]
    fn fade_out_is_ease_in_so_it_leaves_slowly_first() {
        let mut v = Visibility::new(false);
        v.activity();
        run(&mut v, 0.5, 1000.0);
        v.idle = IDLE_S; // idle timeout reached
        run(&mut v, 0.1, 1000.0);
        assert!(
            v.alpha() > 0.9,
            "after 25% of the fade still >90%: {}",
            v.alpha()
        );
        run(&mut v, 0.3, 1000.0);
        assert!(v.alpha() < 0.05, "{}", v.alpha());
    }

    #[test]
    fn never_hides_while_hovering_or_pressed_and_counts_from_release() {
        let mut v = Visibility::new(false);
        v.activity();
        v.set_hovering(true);
        run(&mut v, 10.0, 60.0);
        assert!(v.wants_visible() && v.alpha() > 0.999, "hover keeps it");
        v.set_hovering(false);
        run(&mut v, 2.4, 60.0);
        assert!(
            v.wants_visible(),
            "2.5 s counted from the moment the pointer left"
        );
        run(&mut v, 0.2, 60.0);
        assert!(!v.wants_visible());
        let mut v = Visibility::new(false);
        v.activity();
        v.set_pressed(true);
        run(&mut v, 10.0, 60.0);
        assert!(v.wants_visible(), "a held mouse button keeps it");
        v.set_pressed(false);
        run(&mut v, 2.4, 60.0);
        assert!(v.wants_visible());
    }

    #[test]
    fn activity_during_fade_out_reverses_smoothly_from_the_current_alpha() {
        let mut v = Visibility::new(false);
        v.activity();
        run(&mut v, 0.5, 1000.0);
        v.idle = IDLE_S;
        run(&mut v, 0.30, 1000.0);
        let mid = v.alpha();
        assert!(mid > 0.0 && mid < 0.5, "mid fade-out: {mid}");
        v.activity();
        v.tick(0.001);
        assert!(
            v.alpha() >= mid && v.alpha() - mid < 0.05,
            "continues upwards from {mid}, no restart: {}",
            v.alpha()
        );
        run(&mut v, 0.2, 1000.0);
        assert!(v.alpha() > 0.999);
        // and a fade-in interrupted by idle expiry reverses from where it is
        let mut v = Visibility::new(false);
        v.activity();
        run(&mut v, 0.05, 1000.0);
        let a = v.alpha();
        v.idle = IDLE_S;
        v.tick(0.001);
        assert!(v.alpha() <= a && a - v.alpha() < 0.05);
    }

    #[test]
    fn alpha_does_not_depend_on_the_frame_rate() {
        let at = |fps: f32| {
            let mut v = Visibility::new(false);
            v.activity();
            run(&mut v, 0.1, fps);
            let a_in = v.alpha();
            v.set_hovering(true);
            run(&mut v, 3.0, fps);
            v.set_hovering(false);
            v.idle = IDLE_S; // the countdown itself is frame-quantised by design; the fade is what must match
            run(&mut v, 0.2, fps);
            (a_in, v.alpha())
        };
        let base = at(1000.0);
        for fps in [30.0, 60.0, 144.0, 240.0] {
            let r = at(fps);
            assert!(
                (r.0 - base.0).abs() < 0.01,
                "fade-in at {fps} fps: {r:?} vs {base:?}"
            );
            assert!(
                (r.1 - base.1).abs() < 0.03,
                "fade-out at {fps} fps: {r:?} vs {base:?}"
            );
        }
        // hostile deltas are harmless
        let mut v = Visibility::new(false);
        v.activity();
        for dt in [f32::NAN, f32::INFINITY, -1.0, 1e9] {
            v.tick(dt);
            assert!((0.0..=1.0).contains(&v.alpha()) && v.alpha().is_finite());
        }
    }

    #[test]
    fn always_mode_never_hides_and_lyrics_lift_is_smooth_and_interruptible() {
        let mut v = Visibility::new(true);
        run(&mut v, 30.0, 60.0);
        assert!(v.alpha() > 0.999 && v.wants_visible());
        let mut v = Visibility::new(false);
        v.activity();
        // (requirement changed in the review fixes: the lift follows the bar's alpha, not its own 250 ms ramp)
        v.tick(0.001);
        let early = v.lift();
        assert!(
            early > 0.0 && early < 1.0 && v.alpha() < LIFT_FULL_ALPHA,
            "eases in with the first alpha: {early}"
        );
        run(&mut v, 0.2, 1000.0);
        assert!((v.lift() - 1.0).abs() < 1e-6);
        v.idle = IDLE_S;
        run(&mut v, 0.05, 1000.0);
        let down = v.lift();
        v.activity();
        v.tick(0.001);
        assert!(v.lift() >= down, "reverses from the current value");
    }

    // ---- layout, hit testing -------------------------------------------------------------------------------

    #[test]
    fn layout_is_a_centred_560x64_pill_28px_from_the_bottom_at_1080p() {
        let g = layout(HD, 1.0);
        assert_eq!(g.scale, 1.0);
        assert_eq!((g.bar.w, g.bar.h), (560.0, 64.0));
        assert_eq!(g.bar.x + g.bar.w / 2.0, 960.0, "centred");
        assert_eq!(g.bar.y + g.bar.h, 1080.0 - 28.0, "28 px from the bottom");
        let p = g.buttons[ButtonId::PlayPause.index()];
        assert_eq!((p.w, p.h), (48.0, 48.0), "play/pause is larger");
        for b in [ButtonId::Previous, ButtonId::Next, ButtonId::Exit] {
            assert_eq!(g.buttons[b.index()].w, 44.0);
        }
        // order left to right, inside the pill, vertically centred, no overlap, group gap larger than button gap
        let mut right = g.bar.x;
        for (i, r) in g.buttons.iter().enumerate() {
            assert!(
                r.x >= right + if i == 0 { 0.0 } else { 12.0 } - 0.001,
                "{i}"
            );
            assert!((r.cy() - g.bar.cy()).abs() < 0.001);
            right = r.x + r.w;
        }
        assert!(right <= g.bar.x + g.bar.w);
        let gap = |a: usize, b: usize| g.buttons[b].x - (g.buttons[a].x + g.buttons[a].w);
        assert!(
            gap(2, 3) > gap(0, 1) && gap(2, 3) > gap(3, 4),
            "divider gap"
        );
        assert!(g.sep_x > g.buttons[2].x + g.buttons[2].w && g.sep_x < g.buttons[3].x);
        // progress line above the pill, spanning inside its width
        assert!(g.prog_y < g.bar.y && g.prog_x0 > g.bar.x && g.prog_x1 < g.bar.x + g.bar.w);
    }

    #[test]
    fn layout_scales_with_surface_height_dpi_and_width() {
        let g1 = layout(HD, 1.0);
        for (res, k) in [((2560u32, 1440u32), 4.0 / 3.0), ((3840, 2160), 2.0)] {
            let g = layout(res, 1.0);
            assert!((g.scale - k).abs() < 1e-4);
            assert!((g.bar.w - 560.0 * k).abs() < 0.01 && (g.bar.h - 64.0 * k).abs() < 0.01);
            assert!(((res.1 as f32 - (g.bar.y + g.bar.h)) - 28.0 * k).abs() < 0.01);
            for i in 0..8 {
                assert!((g.buttons[i].w - g1.buttons[i].w * k).abs() < 0.01);
            }
        }
        // small high-DPI window: the scale never drops below 0.85 x the DPI factor (hit targets stay usable)
        assert!((ui_scale((1280, 720), 1.0) - 0.85).abs() < 1e-4);
        assert!((ui_scale((1280, 720), 2.0) - 1.7).abs() < 1e-4);
        assert!(layout((1280, 720), 2.0).buttons[0].w > 44.0 * 1.5);
        // never wider than a narrow window
        let g = layout((400, 300), 1.0);
        assert!(g.bar.x >= 0.0 && g.bar.x + g.bar.w <= 400.0);
        // hostile input
        assert!(ui_scale((0, 0), f32::NAN) > 0.0);
        assert!(layout((0, 0), 1.0).bar.w.is_finite());
    }

    #[test]
    fn hit_test_finds_every_button_and_nothing_in_gaps_edges_or_outside() {
        for res in [HD, (1280, 720), (3840, 2160), (800, 600)] {
            for dpi in [1.0, 1.5] {
                let g = layout(res, dpi);
                for id in ButtonId::ALL {
                    let r = g.buttons[id.index()];
                    for p in [
                        (r.cx(), r.cy()),
                        (r.x, r.y),
                        (r.x + r.w - 0.01, r.y + r.h - 0.01),
                    ] {
                        assert_eq!(hit_test(&g, p), Some(id), "{id:?} {res:?} {p:?}");
                    }
                    // just outside every edge
                    for p in [
                        (r.x - 0.01, r.cy()),
                        (r.x + r.w, r.cy()),
                        (r.cx(), r.y - 0.01),
                        (r.cx(), r.y + r.h),
                    ] {
                        assert_ne!(hit_test(&g, p), Some(id), "{id:?} {p:?}");
                    }
                }
                // the gap between neighbours and the divider
                for k in 0..7 {
                    let (a, b) = (g.buttons[k], g.buttons[k + 1]);
                    assert_eq!(
                        hit_test(&g, (0.5 * (a.x + a.w + b.x), a.cy())),
                        None,
                        "gap {k}"
                    );
                }
                assert_eq!(hit_test(&g, (g.sep_x, g.bar.cy())), None);
                // pill padding, above and below the pill, far away, off-surface
                assert_eq!(hit_test(&g, (g.bar.x + 2.0, g.bar.cy())), None);
                assert_eq!(hit_test(&g, (g.bar.cx(), g.bar.y - 5.0)), None);
                assert_eq!(hit_test(&g, (0.0, 0.0)), None);
                assert_eq!(hit_test(&g, (-50.0, -50.0)), None);
                assert_eq!(hit_test(&g, (f32::NAN, 1.0)), None);
            }
        }
    }

    #[test]
    fn hit_targets_meet_the_minimum_size_and_never_overlap() {
        for res in [(640u32, 360u32), HD, (3840, 2160)] {
            let g = layout(res, 1.0);
            for r in &g.buttons {
                assert!(r.w >= 24.0 && r.h >= 24.0, "WCAG 2.5.8 minimum at {res:?}");
            }
            for a in 0..8 {
                for b in a + 1..8 {
                    let (p, q) = (g.buttons[a], g.buttons[b]);
                    assert!(p.x + p.w <= q.x + 0.001, "{a}/{b} overlap");
                }
            }
            assert!(g.prog_hit.h >= 24.0 * g.scale - 0.01);
        }
    }

    #[test]
    fn progress_click_maps_to_a_clamped_fraction_and_seconds() {
        let g = layout(HD, 1.0);
        let mid = 0.5 * (g.prog_x0 + g.prog_x1);
        assert!((progress_hit(&g, (mid, g.prog_y)).unwrap() - 0.5).abs() < 1e-4);
        assert_eq!(progress_hit(&g, (g.prog_x0, g.prog_y)), Some(0.0));
        assert!(progress_hit(&g, (g.prog_x1 - 0.01, g.prog_y)).unwrap() > 0.999);
        assert_eq!(progress_hit(&g, (g.prog_x1 + 1.0, g.prog_y)), None);
        assert_eq!(progress_hit(&g, (mid, g.prog_y - 20.0)), None);
        assert_eq!(progress_hit(&g, (mid, g.bar.cy())), None, "not on the pill");
        assert_eq!(seek_seconds(0.5, 200.0), 100.0);
        assert_eq!(seek_seconds(-3.0, 200.0), 0.0);
        assert!(seek_seconds(7.0, 200.0) < 200.0, "never the very end");
        assert_eq!(seek_seconds(1.0, 200.0), 199.75);
        assert_eq!(seek_seconds(f32::NAN, 200.0), 0.0);
        assert_eq!(seek_seconds(0.5, f64::NAN), 0.0);
        assert_eq!(seek_seconds(0.5, 0.0), 0.0);
        assert_eq!(seek_seconds(1.0, 0.1), 0.0);
    }

    // ---- availability, focus -------------------------------------------------------------------------------

    #[test]
    fn enabled_state_follows_the_capabilities_for_every_combination() {
        for bits in 0u32..64 {
            let caps = Capabilities {
                can_play: bits & 1 != 0,
                can_pause: bits & 2 != 0,
                can_toggle: bits & 4 != 0,
                can_next: bits & 8 != 0,
                can_previous: bits & 16 != 0,
                can_seek: bits & 32 != 0,
            };
            for state in [
                PlaybackState::Playing,
                PlaybackState::Paused,
                PlaybackState::Stopped,
                PlaybackState::Other,
            ] {
                let a = av(caps, state, true);
                assert_eq!(enabled(ButtonId::Next, &a), caps.can_next);
                assert_eq!(enabled(ButtonId::Previous, &a), caps.can_previous);
                // exactly when the command path would send something
                assert_eq!(
                    enabled(ButtonId::PlayPause, &a),
                    caps.resolve(MediaCommand::TogglePlayPause, state).is_some()
                );
                assert_eq!(a.can_seek(), caps.can_seek);
                assert!(
                    !av(caps, state, false).can_seek(),
                    "no timeline, no seeking"
                );
                for b in [
                    ButtonId::Scene,
                    ButtonId::Lyrics,
                    ButtonId::Artwork,
                    ButtonId::Fullscreen,
                    ButtonId::Exit,
                ] {
                    assert!(enabled(b, &a), "view buttons never depend on the player");
                }
            }
        }
        // Off feed / no session: every media control disabled, view controls still work
        for b in [ButtonId::Previous, ButtonId::PlayPause, ButtonId::Next] {
            assert!(!enabled(b, &off()));
        }
        assert!(enabled(ButtonId::Exit, &off()));
        // pause-only player: toggle falls back to Pause while playing, Play is unavailable
        let pause_only = Capabilities {
            can_pause: true,
            ..Capabilities::default()
        };
        assert!(enabled(
            ButtonId::PlayPause,
            &av(pause_only, PlaybackState::Playing, true)
        ));
        assert!(!enabled(
            ButtonId::PlayPause,
            &av(pause_only, PlaybackState::Paused, true)
        ));
    }

    #[test]
    fn every_button_has_a_distinct_accessible_name() {
        let mut names: Vec<_> = ButtonId::ALL.iter().map(|b| b.name()).collect();
        assert!(names.iter().all(|n| n.len() > 3));
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 8);
        for (i, b) in ButtonId::ALL.iter().enumerate() {
            assert_eq!(b.index(), i);
        }
    }

    #[test]
    fn focus_moves_in_visual_order_skips_disabled_wraps_and_drops_when_disabled() {
        let a = all();
        let mut f = Focus::default();
        let ok = |b| enabled(b, &a);
        assert_eq!(f.ring(), None);
        let mut seen = vec![];
        for _ in 0..9 {
            f.tab(false, ok);
            seen.push(f.ring().unwrap());
        }
        assert_eq!(&seen[..8], &ButtonId::ALL, "visual order");
        assert_eq!(seen[8], ButtonId::Previous, "wraps");
        f.tab(true, ok);
        assert_eq!(
            f.ring(),
            Some(ButtonId::Exit),
            "Shift+Tab goes back, wrapping"
        );
        // only the view buttons are enabled when nothing plays
        let o = off();
        let mut f = Focus::default();
        f.tab(false, |b| enabled(b, &o));
        assert_eq!(
            f.ring(),
            Some(ButtonId::Scene),
            "disabled media buttons are skipped"
        );
        f.tab(true, |b| enabled(b, &o));
        assert_eq!(
            f.ring(),
            Some(ButtonId::Exit),
            "Shift+Tab from the first enabled wraps"
        );
        let mut f = Focus::default();
        f.tab(true, |b| enabled(b, &o));
        assert_eq!(
            f.ring(),
            Some(ButtonId::Exit),
            "Shift+Tab from nothing starts at the end"
        );
        // focus on a button that becomes disabled is dropped
        let mut f = Focus::default();
        f.tab(false, ok);
        f.retain(|b| b != ButtonId::Previous);
        assert_eq!(f.ring(), None);
        // nothing enabled: no focus, no panic
        let mut f = Focus::default();
        f.tab(false, |_| false);
        assert_eq!(f.ring(), None);
    }

    // ---- ControlsUi: input -> state --------------------------------------------------------------------------

    fn ui_visible() -> ControlsUi {
        let mut u = ControlsUi::new(false);
        u.set_view(HD, 1.0);
        u.key_activity();
        for _ in 0..30 {
            u.vis.tick(1.0 / 60.0);
        }
        assert!(u.vis.interactive());
        u
    }

    fn centre(u: &ControlsUi, b: ButtonId) -> (f32, f32) {
        let r = u.geometry().buttons[b.index()];
        (r.cx(), r.cy())
    }

    fn input(a: Availability) -> FrameInput {
        FrameInput {
            dt: 1.0 / 60.0,
            av: a,
            toggles: Toggles::default(),
            position: Some(30.0),
            duration: Some(120.0),
            fg: [0.9; 3],
            tint: [0.2, 0.4, 0.8],
        }
    }

    #[test]
    fn pointer_must_move_more_than_4px_to_show_the_bar_and_drift_accumulates() {
        let mut u = ControlsUi::new(false);
        u.set_view(HD, 1.0);
        u.pointer_moved((100.0, 100.0)); // first position counts
        run(&mut u.vis, 3.0, 60.0);
        assert!(!u.vis.wants_visible());
        u.pointer_moved((102.0, 102.0)); // 2.8 px: no
        u.pointer_moved((103.0, 100.0));
        assert!(
            !u.vis.wants_visible(),
            "jitter below 4 px does not show the bar"
        );
        u.pointer_moved((100.0, 105.0)); // 5 px from the anchor
        assert!(u.vis.wants_visible());
    }

    #[test]
    fn click_activates_only_enabled_visible_buttons_on_release_over_the_same_one() {
        let a = all();
        let mut u = ui_visible();
        u.pointer_moved(centre(&u, ButtonId::Next));
        assert_eq!(u.mouse_left(true, &a), None, "press alone does nothing");
        assert_eq!(u.mouse_left(false, &a), Some(Click::Button(ButtonId::Next)));
        // press on one button, release on another: nothing
        u.pointer_moved(centre(&u, ButtonId::Next));
        u.mouse_left(true, &a);
        u.pointer_moved(centre(&u, ButtonId::Scene));
        assert_eq!(u.mouse_left(false, &a), None);
        // disabled button: never a click
        let o = off();
        u.pointer_moved(centre(&u, ButtonId::PlayPause));
        u.mouse_left(true, &o);
        assert_eq!(u.mouse_left(false, &o), None);
        // view buttons always work
        u.pointer_moved(centre(&u, ButtonId::Exit));
        u.mouse_left(true, &o);
        assert_eq!(u.mouse_left(false, &o), Some(Click::Button(ButtonId::Exit)));
        // gap and outside: nothing
        let g = *u.geometry();
        u.pointer_moved((
            0.5 * (g.buttons[0].x + g.buttons[0].w + g.buttons[1].x),
            g.bar.cy(),
        ));
        u.mouse_left(true, &a);
        assert_eq!(u.mouse_left(false, &a), None);
    }

    #[test]
    fn a_click_while_the_bar_is_hidden_only_reveals_it() {
        let a = all();
        let mut u = ControlsUi::new(false);
        u.set_view(HD, 1.0);
        u.pointer_moved(centre(&u, ButtonId::Exit));
        run(&mut u.vis, 5.0, 60.0);
        assert!(u.vis.fully_hidden());
        assert_eq!(u.mouse_left(true, &a), None);
        assert_eq!(
            u.mouse_left(false, &a),
            None,
            "no accidental exit through an invisible bar"
        );
        assert!(u.vis.wants_visible(), "but it is revealed");
        // while it is still fading in below the threshold: also nothing
        u.vis.tick(0.01);
        assert!(!u.vis.interactive());
        u.mouse_left(true, &a);
        assert_eq!(u.mouse_left(false, &a), None);
    }

    #[test]
    fn progress_seeks_only_when_the_session_can_seek() {
        let g = *ui_visible().geometry();
        let p = (0.5 * (g.prog_x0 + g.prog_x1), g.prog_y);
        let mut u = ui_visible();
        u.pointer_moved(p);
        match u.mouse_left(true, &all()) {
            Some(Click::Seek(f)) => assert!((f - 0.5).abs() < 0.01),
            other => panic!("{other:?}"),
        }
        u.mouse_left(false, &all());
        let no_seek = av(
            Capabilities {
                can_seek: false,
                ..Capabilities::all()
            },
            PlaybackState::Playing,
            true,
        );
        assert_eq!(u.mouse_left(true, &no_seek), None, "read-only progress");
        u.mouse_left(false, &no_seek);
        let no_timeline = av(Capabilities::all(), PlaybackState::Playing, false);
        assert_eq!(u.mouse_left(true, &no_timeline), None);
    }

    #[test]
    fn keyboard_activation_needs_a_visible_bar_and_keyboard_focus() {
        let a = all();
        let mut u = ui_visible();
        assert_eq!(
            u.activate_focused(&a),
            None,
            "no focus: Space stays the global shortcut"
        );
        u.tab(false, &a);
        assert_eq!(u.activate_focused(&a), Some(ButtonId::Previous));
        u.tab(false, &a);
        assert_eq!(u.activate_focused(&a), Some(ButtonId::PlayPause));
        // a disabled focused button never activates
        let o = off();
        assert_eq!(u.activate_focused(&o), None);
        // pointer use ends keyboard focus mode
        u.pointer_moved(centre(&u, ButtonId::Scene));
        u.mouse_left(true, &a);
        u.mouse_left(false, &a);
        assert_eq!(u.activate_focused(&a), None);
        // a hidden bar: Tab shows it, but the first Space is still the global one until it is visible
        let mut h = ControlsUi::new(false);
        h.set_view(HD, 1.0);
        h.tab(false, &a);
        assert_eq!(h.activate_focused(&a), None, "bar not visible yet");
    }

    #[test]
    fn frame_draws_nothing_while_hidden_and_maps_states_when_visible() {
        let mut u = ControlsUi::new(false);
        u.set_view(HD, 1.0);
        assert!(u.frame(&input(all())).is_none(), "hidden: zero draw cost");
        u.key_activity();
        let mut d = None;
        for _ in 0..30 {
            d = u.frame(&input(all()));
        }
        let d = d.expect("visible");
        assert!(d.alpha > 0.99 && d.bar[1] > 0.0);
        assert!(d.progress.is_some_and(|p| (p.frac - 0.25).abs() < 1e-4));
        assert_eq!(
            d.buttons[1].icon,
            icon::PAUSE,
            "playing shows the pause icon"
        );
        assert!(d.buttons.iter().all(|b| b.opacity == 1.0 && !b.focus));
        // paused shows play; Off: media buttons dimmed, view buttons not, no timeline, no progress
        let mut p = input(av(Capabilities::all(), PlaybackState::Paused, true));
        p.toggles.lyrics = true;
        let d = u.frame(&p).unwrap();
        assert_eq!(d.buttons[1].icon, icon::PLAY);
        assert_eq!(d.buttons[ButtonId::Lyrics.index()].toggled, 1.0);
        assert_eq!(d.buttons[ButtonId::Artwork.index()].toggled, 0.0);
        let mut o = input(off());
        o.position = None;
        o.duration = None;
        let d = u.frame(&o).unwrap();
        for b in [ButtonId::Previous, ButtonId::PlayPause, ButtonId::Next] {
            let bd = d.buttons[b.index()];
            assert_eq!(bd.opacity, DISABLED_OPACITY);
            assert_eq!(bd.hover, 0.0, "a disabled control shows no hover");
        }
        assert_eq!(d.buttons[ButtonId::Exit.index()].opacity, 1.0);
        assert!(d.progress.is_none(), "no timeline: no progress line");
        // disabled buttons show no hover even with the pointer on them
        u.pointer_moved(centre(&u, ButtonId::Next));
        for _ in 0..30 {
            let d = u.frame(&o).unwrap();
            assert_eq!(d.buttons[ButtonId::Next.index()].hover, 0.0);
        }
        u.pointer_moved(centre(&u, ButtonId::Exit));
        let mut last = None;
        for _ in 0..30 {
            last = u.frame(&o);
        }
        assert!(last.unwrap().buttons[ButtonId::Exit.index()].hover > 0.99);
    }

    #[test]
    fn rise_is_8px_at_the_start_of_the_fade_and_zero_when_settled() {
        let mut u = ControlsUi::new(false);
        u.set_view(HD, 1.0);
        u.key_activity();
        let first = u.frame(&input(all())).unwrap();
        let base = layout(HD, 1.0).bar.cy();
        assert!(
            first.bar[1] - base > 4.0 && first.bar[1] - base <= 8.0,
            "{}",
            first.bar[1] - base
        );
        let mut d = first;
        for _ in 0..30 {
            d = u.frame(&input(all())).unwrap();
        }
        assert!((d.bar[1] - base).abs() < 0.01);
    }

    #[test]
    fn the_tab_ring_is_drawn_only_for_keyboard_focus() {
        let a = all();
        let mut u = ui_visible();
        u.tab(false, &a);
        let d = u.frame(&input(a)).unwrap();
        assert!(d.buttons[0].focus && d.buttons.iter().filter(|b| b.focus).count() == 1);
        u.pointer_moved(centre(&u, ButtonId::Next));
        u.mouse_left(true, &a);
        let d = u.frame(&input(a)).unwrap();
        assert!(
            d.buttons.iter().all(|b| !b.focus),
            "mouse use removes the ring"
        );
    }

    #[test]
    fn time_texts_are_requested_once_per_change_and_superseded_results_are_dropped() {
        let mut u = ui_visible();
        let mut i = input(all());
        u.frame(&i);
        let jobs = u.take_jobs();
        assert_eq!(jobs.len(), 2, "elapsed and duration");
        u.frame(&i);
        assert!(u.take_jobs().is_empty(), "unchanged text: no new job");
        i.position = Some(30.9);
        u.frame(&i);
        assert!(
            u.take_jobs().is_empty(),
            "same second: no new job (once per second at most)"
        );
        i.position = Some(31.0);
        u.frame(&i);
        let jobs = u.take_jobs();
        assert_eq!(jobs.len(), 1, "only the elapsed text changed");
        let Job::Clock {
            serial, slot, text, ..
        } = &jobs[0]
        else {
            panic!()
        };
        assert_eq!((*slot, text.as_str()), (0, "0:31"));
        let bmp = || mg_text::TextBitmap {
            width: 10,
            height: 6,
            pad: 1,
            coverage: vec![255; 60],
        };
        u.on_ready(Ready::Clock {
            serial: serial - 1,
            slot: 0,
            bitmap: Some(bmp()),
        });
        assert!(u.take_ops().is_empty(), "stale result ignored");
        u.on_ready(Ready::Clock {
            serial: *serial,
            slot: 0,
            bitmap: Some(bmp()),
        });
        assert!(matches!(u.take_ops()[..], [GpuOp::SetText(CLOCK_SLOT, _)]));
        // hidden bar requests nothing; losing the timeline clears the texts
        i.av.has_timeline = false;
        u.frame(&i);
        assert!(
            u.take_ops()
                .iter()
                .any(|o| matches!(o, GpuOp::ClearText(CLOCK_SLOT)))
        );
        assert_eq!(fmt_time(0.0), "0:00");
        assert_eq!(fmt_time(59.9), "0:59");
        assert_eq!(fmt_time(61.0), "1:01");
        assert_eq!(fmt_time(3725.0), "1:02:05");
        assert_eq!(fmt_time(f64::NAN), "0:00");
        assert_eq!(fmt_time(-4.0), "0:00");
    }

    #[test]
    fn lyric_lift_is_at_least_the_spec_and_eases_with_the_bar() {
        for s in [HD, (1280, 720), (720, 1280), (3840, 2160)] {
            for anchor in [0.70f32, 0.72, 0.80] {
                let l = lyric_lift(anchor, s, 1.0);
                assert!((LYRIC_LIFT..=0.30).contains(&l), "{s:?} {anchor}: {l}");
            }
        }
        assert!(
            (lyric_lift(0.70, HD, 1.0) - 0.067).abs() < 1e-6,
            "no extra lift needed at the default anchor"
        );
        let mut u = ui_visible();
        assert!((u.lyric_lift(0.72) - lyric_lift(0.72, HD, 1.0)).abs() < 1e-3);
        run(&mut u.vis, 5.0, 60.0);
        assert!(u.lyric_lift(0.72) < 1e-6, "no lift when hidden");
    }

    #[test]
    fn lyric_lift_is_full_while_the_bar_is_visible_and_settles_only_once_it_is_nearly_gone() {
        for fps in [30.0f32, 60.0, 144.0] {
            let dt = 1.0 / fps;
            let mut v = Visibility::new(false);
            v.activity();
            for _ in 0..(0.5 * fps) as usize {
                v.tick(dt);
                if v.alpha() >= LIFT_FULL_ALPHA {
                    assert!(
                        v.lift() > 0.999,
                        "show {fps}: alpha {} lift {}",
                        v.alpha(),
                        v.lift()
                    );
                }
            }
            // hide: held at 1 while the bar is still at least faintly there, then settles smoothly
            let mut prev = v.lift();
            let mut t = 0.0;
            while t < IDLE_S + HIDE_FADE_S + 1.0 {
                v.tick(dt);
                t += dt;
                if v.alpha() >= 0.1 {
                    assert!(
                        v.lift() > 0.999,
                        "hide {fps}: alpha {} lift {}",
                        v.alpha(),
                        v.lift()
                    );
                }
                assert!(
                    v.lift() <= prev + 1e-6,
                    "never bounces back up while hiding"
                );
                assert!(prev - v.lift() < 0.5, "no snap: {prev} -> {}", v.lift());
                prev = v.lift();
            }
            assert!(v.lift() < 1e-3 && v.fully_hidden());
        }
    }

    #[test]
    fn focus_loss_releases_a_held_button_so_the_bar_can_hide() {
        let a = all();
        let mut u = ui_visible();
        u.pointer_moved(centre(&u, ButtonId::Next));
        u.mouse_left(true, &a); // alt-tab now: the release never arrives
        u.focus_lost();
        let mut i = input(a);
        i.dt = 0.05;
        for _ in 0..((IDLE_S + HIDE_FADE_S + 0.5) / 0.05) as usize {
            u.frame(&i);
        }
        assert!(
            u.vis.fully_hidden(),
            "a press held across focus loss does not pin the bar"
        );
        assert_eq!(
            u.mouse_left(false, &a),
            None,
            "the late release clicks nothing"
        );
        // keyboard focus mode ends too
        let mut u = ui_visible();
        u.tab(false, &a);
        u.focus_lost();
        assert_eq!(u.activate_focused(&a), None);
    }

    #[test]
    fn the_progress_line_is_last_in_the_tab_order_only_when_seeking_works() {
        let a = all();
        let mut u = ui_visible();
        for b in ButtonId::ALL {
            u.tab(false, &a);
            assert_eq!(u.activate_focused(&a), Some(b));
            assert!(!u.progress_focused(&a));
        }
        u.tab(false, &a);
        assert!(u.progress_focused(&a), "after Exit");
        assert_eq!(
            u.activate_focused(&a),
            None,
            "Space on the progress line stays global"
        );
        let d = u.frame(&input(a)).unwrap();
        assert!(d.progress.unwrap().focus && d.buttons.iter().all(|b| !b.focus));
        u.tab(false, &a);
        assert_eq!(u.activate_focused(&a), Some(ButtonId::Previous), "wraps");
        u.tab(true, &a);
        assert!(u.progress_focused(&a), "Shift+Tab back onto it");
        // read-only session or no timeline: never focusable, and focus on it is dropped
        let ro = av(
            Capabilities {
                can_seek: false,
                ..Capabilities::all()
            },
            PlaybackState::Playing,
            true,
        );
        assert!(!u.progress_focused(&ro));
        u.frame(&input(ro));
        assert!(!u.progress_focused(&a), "dropped once seeking went away");
        let mut u = ui_visible();
        u.tab(true, &ro);
        assert_eq!(u.activate_focused(&ro), Some(ButtonId::Exit));
        let nt = av(Capabilities::all(), PlaybackState::Playing, false);
        let mut u = ui_visible();
        u.tab(true, &nt);
        assert_eq!(u.activate_focused(&nt), Some(ButtonId::Exit));
    }

    #[test]
    fn key_seeks_step_from_the_position_and_clamp_like_a_click() {
        assert_eq!(key_seek_seconds(30.0, -SEEK_STEP_S, 120.0), Some(25.0));
        assert_eq!(key_seek_seconds(30.0, SEEK_STEP_LONG_S, 120.0), Some(45.0));
        assert_eq!(key_seek_seconds(3.0, -SEEK_STEP_S, 120.0), Some(0.0));
        assert_eq!(
            key_seek_seconds(118.0, SEEK_STEP_S, 120.0),
            Some(119.75),
            "never the very end"
        );
        assert_eq!(
            key_seek_seconds(60.0, f64::NEG_INFINITY, 120.0),
            Some(0.0),
            "Home"
        );
        assert_eq!(
            key_seek_seconds(60.0, f64::INFINITY, 120.0),
            Some(119.75),
            "End"
        );
        assert_eq!(key_seek_seconds(f64::NAN, SEEK_STEP_S, 120.0), Some(5.0));
        assert_eq!(key_seek_seconds(10.0, 5.0, 0.0), None);
        assert_eq!(key_seek_seconds(10.0, 5.0, f64::NAN), None);
        assert_eq!(key_seek_seconds(10.0, f64::NAN, 100.0), None);
    }

    #[test]
    fn real_pointer_movement_ends_keyboard_focus_mode() {
        let a = all();
        let mut u = ui_visible();
        u.pointer_moved((10.0, 10.0));
        u.tab(false, &a);
        u.pointer_moved((11.0, 11.0)); // jitter below the threshold keeps it
        assert_eq!(u.activate_focused(&a), Some(ButtonId::Previous));
        u.pointer_moved((40.0, 40.0));
        assert_eq!(
            u.activate_focused(&a),
            None,
            "Space is the global shortcut again"
        );
        assert!(u.frame(&input(a)).unwrap().buttons.iter().all(|b| !b.focus));
    }

    #[test]
    fn elapsed_text_never_exceeds_the_duration() {
        let mut u = ui_visible();
        let mut i = input(all());
        i.position = Some(125.4);
        u.frame(&i);
        let texts: Vec<_> = u
            .take_jobs()
            .into_iter()
            .filter_map(|j| match j {
                Job::Clock { slot, text, .. } => Some((slot, text)),
                _ => None,
            })
            .collect();
        assert!(texts.contains(&(0, "2:00".to_string())), "{texts:?}");
        assert!(texts.contains(&(1, "2:00".to_string())), "{texts:?}");
    }

    // ---- contrast --------------------------------------------------------------------------------------------

    /// WCAG contrast ratio of two relative luminances.
    fn ratio(a: f32, b: f32) -> f32 {
        (a.max(b) + 0.05) / (a.min(b) + 0.05)
    }

    /// Relative luminance of linear RGB.
    fn lum(c: [f32; 3]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    /// `src` at `a` over `dst` (luminance is linear, so compositing luminances is exact).
    fn over(dst: f32, src: f32, a: f32) -> f32 {
        src * a + dst * (1.0 - a)
    }

    /// Worst-case backing luminance (pill fill at `alpha` over every scene luminance 0..=1 and every tint).
    fn backings(alpha: f32) -> impl Iterator<Item = f32> {
        let tints = [
            [0.0; 3],
            [1.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.4, 0.8],
        ];
        (0..=100).flat_map(move |k| {
            tints
                .into_iter()
                .map(move |t| over(k as f32 / 100.0, lum(pill_bg(t)), alpha))
        })
    }

    /// Darkest icon / text colour the bar can draw.
    fn fg_lum() -> f32 {
        lum([FG_MIN; 3])
    }

    fn icons_ok(alpha: f32) -> bool {
        backings(alpha).all(|b| ratio(over(b, fg_lum(), 1.0), b) >= 4.5)
    }

    #[test]
    fn pill_alpha_is_the_smallest_that_keeps_icons_and_times_at_4_5_to_1_over_any_scene() {
        let min = (0..=1000)
            .map(|k| k as f32 / 1000.0)
            .find(|a| icons_ok(*a))
            .expect("some opacity works");
        assert!(
            icons_ok(PILL_ALPHA),
            "icons and the time texts (same scrim) >= 4.5:1"
        );
        assert!(
            PILL_ALPHA <= min + 0.02,
            "restrained: {PILL_ALPHA} vs the minimum {min}"
        );
        // the times use the same fill and opacity as their scrim, and the shader really draws those scrims
        let src = include_str!("../../mg-gfx/src/overlay.wgsl");
        assert!(src.contains("o = over(o, c.bg.rgb, c.bg.w * scrim * ta);"));
        assert!(src.contains("o = over(o, c.bg.rgb, c.bg.w * under * al);"));
    }

    #[test]
    fn progress_line_keeps_3_to_1_over_its_backing_on_any_scene() {
        use mg_gfx::overlay::{PLAYED_ALPHA, TRACK_ALPHA};
        let src = include_str!("../../mg-gfx/src/overlay.wgsl");
        assert_eq!((TRACK_ALPHA, PLAYED_ALPHA), (0.12, 0.95));
        assert!(
            src.contains("trk * (0.12 + 0.83 * played)"),
            "shader mirrors the constants"
        );
        for b in backings(PILL_ALPHA) {
            let played = over(b, fg_lum(), PLAYED_ALPHA);
            let rest = over(b, fg_lum(), TRACK_ALPHA);
            assert!(ratio(played, b) >= 3.0, "played vs backing {b}");
            assert!(ratio(played, rest) >= 3.0, "played vs unplayed {b}");
        }
    }
}
