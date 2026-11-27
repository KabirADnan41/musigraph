//! Scene transition state machine (pure, no GPU, no audio). One scene is drawn at every moment: the old one
//! exits layer by layer (Exit), the screen holds a quiet palette "bridge" (Bridge), the new one enters (Enter);
//! content of the two scenes is never blended. `update` returns a `Frame`: which scene to draw and how veiled each
//! layer is. This file is the single source of truth for timings, curves, retarget rules and the safety limits.
//!
//! * Veils run 0 (fully visible) to 1 (fully hidden). Levels are a pure function of the time inside the phase
//!   (`layer_levels`), so reversing an exit replays the same frames backwards.
//! * Retargeting never queues: the latest request wins and presence stays continuous (see `request`).
//! * Rapid switching shortens the phases (`shorten`) and a luminance-swing guard (WCAG 2.3.1 spirit) holds the
//!   bridge instead of flashing: never more than 3 swings above `SWING_AMP` per second, per-frame change capped.

use mg_gfx::scene::SceneKind;

/// Phase lengths in seconds plus which effects are on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Profile {
    pub exit: f32,
    pub bridge: f32,
    pub enter: f32,
    /// Layers leave and enter one after another (off: all layers share one curve).
    pub staged: bool,
    /// Monolithic scenes close with an iris vignette.
    pub iris: bool,
    /// The bridge breathes +-2 % lightness at 0.25 Hz.
    pub breathing: bool,
}

/// Default for the selector, `[ ]` and the number keys.
pub const QUICK: Profile = Profile {
    exit: 0.45,
    bridge: 0.20,
    enter: 0.65,
    staged: true,
    iris: true,
    breathing: false,
};
/// Tour / automatic changes only (`--auto-scenes`).
pub const FULL: Profile = Profile {
    exit: 0.9,
    bridge: 0.4,
    enter: 1.1,
    staged: true,
    iris: true,
    breathing: true,
};
/// Reduced motion: a plain fade through the bridge, no stagger, iris or breathing.
pub const REDUCED: Profile = Profile {
    exit: 0.35,
    bridge: 0.15,
    enter: 0.40,
    staged: false,
    iris: false,
    breathing: false,
};

/// Enter starts when this fraction of the bridge has passed; its first frames are fully veiled, so they overlap
/// the bridge tail without a visible pause.
const BRIDGE_ENTER_AT: f32 = 0.6;
/// Layer order (atmosphere, geometry, particles): start offsets as a fraction of the phase; each layer takes `SPAN`.
const EXIT_OFF: [f32; 3] = [0.30, 0.15, 0.0];
const ENTER_OFF: [f32; 3] = [0.0, 0.15, 0.30];
const SPAN: f32 = 0.70;
/// Presence never changes faster than a full swing per this many seconds, per frame (shortening limit).
const SLEW_S: f32 = 0.15;
/// A luminance step bigger than this per frame is never produced.
const MAX_DL: f32 = 0.1;
/// A luminance swing (a reversal of at least this size) counts toward the flash guard.
const SWING_AMP: f32 = 0.1;
const MAX_SWINGS_PER_S: usize = 3;
/// Requests in this trailing window drive the adaptive shortening.
const WINDOW_S: f32 = 1.5;
const SHORTEN_FLOOR: f32 = 0.35;
/// Bridge luminance low-pass time constant and bounds (relative luminance).
const BRIDGE_TAU: f32 = 0.3;
const BRIDGE_MIN: f32 = 0.04;
const BRIDGE_MAX: f32 = 0.25;
/// Quiet this long after the last request releases a held bridge.
const HOLD_RELEASE_S: f32 = 0.4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Exit,
    Bridge,
    Enter,
}

/// What to draw this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// The one scene to draw (the outgoing one until Enter starts).
    pub scene: SceneKind,
    pub phase: Phase,
    /// Veil per layer (atmosphere, geometry, particles) and the global veil (last) for scenes without layers.
    pub veil: [f32; 4],
    /// Relative luminance of the bridge (0 while idle).
    pub bridge_luma: f32,
    /// Bridge breathing amplitude (0 or 0.02) and iris strength (0 or 1).
    pub breathing: f32,
    pub iris: f32,
}

/// Veil of one layer-set at time `x` (0..1) inside an exit (0 visible -> 1 hidden) or enter (1 -> 0) phase.
/// Exit: smoothstep on the layer's local time, then ^1.5; enter: (1-u)^3 (presence 1-(1-u)^3). Index 3 is the
/// global veil over the whole phase. Unstaged: every layer follows the global curve.
pub fn layer_levels(x: f32, exiting: bool, staged: bool) -> [f32; 4] {
    let mut v = [0.0; 4];
    for (i, out) in v.iter_mut().enumerate() {
        let (off, span) = match (staged && i < 3, exiting) {
            (true, true) => (EXIT_OFF[i], SPAN),
            (true, false) => (ENTER_OFF[i], SPAN),
            _ => (0.0, 1.0),
        };
        let l = ((x - off) / span).clamp(0.0, 1.0);
        *out = if exiting {
            (l * l * (3.0 - 2.0 * l)).powf(1.5)
        } else {
            (1.0 - l).powi(3)
        };
    }
    v
}

/// Bridge relative luminance: never black, never brighter than either scene, 0.04..0.25.
pub fn bridge_target(l_src: f32, l_dst: f32) -> f32 {
    (0.8 * l_src.min(l_dst))
        .min(BRIDGE_MAX)
        .clamp(BRIDGE_MIN, BRIDGE_MAX)
}

/// Relative-luminance estimate of a scene lit by `palette` (no GPU readback): the scenes share the palette.
pub fn scene_luma(palette: &[[f32; 3]; 4]) -> f32 {
    let y = |c: &[f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    0.3 * palette.iter().map(y).sum::<f32>() / 4.0
}

/// Phase shortening factor for `n` requests in the trailing window (1 = none).
pub fn shorten(n: usize) -> f32 {
    if n <= 1 {
        1.0
    } else {
        (1.0 / (1.0 + 0.6 * (n - 1) as f32)).max(SHORTEN_FLOOR)
    }
}

#[derive(Clone, Copy, Debug)]
enum St {
    Idle,
    /// `dir` 1 forward, -1 reversing back to `start`. `scale` shortens a fast exit. Veil = start + (1-start)*curve.
    Exit {
        to: SceneKind,
        x: f32,
        dir: f32,
        scale: f32,
        start: [f32; 4],
        staged: bool,
    },
    Bridge {
        to: SceneKind,
        t: f32,
    },
    /// Veil = start * curve.
    Enter {
        x: f32,
        start: [f32; 4],
        staged: bool,
    },
}

pub struct SceneSwitch {
    shown: SceneKind,
    st: St,
    prof: Profile,
    reduced: bool,
    clock: f32,
    /// Times of recent effective requests (bounded by the window).
    requests: Vec<f32>,
    /// Luminance-swing tracker over the emitted luminance proxy.
    ext: f32,
    dir: i8,
    swings: Vec<f32>,
    holding: bool,
    bridge_y: Option<f32>,
}

fn max_delta(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

impl SceneSwitch {
    pub fn new(start: SceneKind) -> Self {
        Self {
            shown: start,
            st: St::Idle,
            prof: QUICK,
            reduced: false,
            clock: 0.0,
            requests: Vec::new(),
            ext: 0.0,
            dir: 0,
            swings: Vec::new(),
            holding: false,
            bridge_y: None,
        }
    }

    /// Reduced motion: every transition uses `REDUCED`.
    pub fn set_reduced(&mut self, on: bool) {
        self.reduced = on;
    }

    fn profile(&self) -> Profile {
        if self.reduced { REDUCED } else { self.prof }
    }

    /// The scene that is (or will be) the destination: base for "next/previous".
    pub fn target(&self) -> SceneKind {
        match self.st {
            St::Idle => self.shown,
            St::Exit { to, dir, .. } if dir > 0.0 => to,
            St::Exit { .. } => self.shown,
            St::Bridge { to, .. } => to,
            St::Enter { .. } => self.shown,
        }
    }

    fn levels(&self) -> [f32; 4] {
        match self.st {
            St::Idle => [0.0; 4],
            St::Exit {
                x, start, staged, ..
            } => {
                let b = layer_levels(x, true, staged);
                std::array::from_fn(|i| start[i] + (1.0 - start[i]) * b[i])
            }
            St::Bridge { .. } => [1.0; 4],
            St::Enter { x, start, staged } => {
                let b = layer_levels(x, false, staged);
                std::array::from_fn(|i| start[i] * b[i])
            }
        }
    }

    /// Asks for `to` (QUICK pace). Returns true if the request changed anything.
    pub fn request(&mut self, to: SceneKind) -> bool {
        self.request_with(to, QUICK)
    }

    /// Asks for `to` at pace `prof` (the tour uses `FULL`). Latest request wins, nothing is queued:
    /// Idle -> Exit; Exit -> the scene being left reverses it (time runs backwards), another scene replaces the
    /// target; Bridge -> replaces the target; Enter -> the scene entering is ignored, another starts a fast exit
    /// from the presence reached (remaining time scaled by it). Asking for the shown scene when idle does nothing.
    pub fn request_with(&mut self, to: SceneKind, prof: Profile) -> bool {
        let changed = match self.st {
            St::Idle => {
                if to == self.shown {
                    return false;
                }
                self.st = St::Exit {
                    to,
                    x: 0.0,
                    dir: 1.0,
                    scale: 1.0,
                    start: [0.0; 4],
                    staged: self.profile().staged,
                };
                true
            }
            St::Exit {
                to: cur,
                x,
                dir,
                scale,
                start,
                staged,
            } => {
                let (nto, ndir) = if to == self.shown {
                    (cur, -1.0)
                } else {
                    (to, 1.0)
                };
                let same = nto == cur && ndir == dir;
                self.st = St::Exit {
                    to: nto,
                    x,
                    dir: ndir,
                    scale,
                    start,
                    staged,
                };
                !same
            }
            St::Bridge { to: cur, t } => {
                self.st = St::Bridge { to, t };
                to != cur
            }
            St::Enter { .. } => {
                if to == self.shown {
                    return false;
                }
                self.fast_exit(to);
                true
            }
        };
        if changed {
            self.prof = prof;
            self.requests.push(self.clock);
        }
        changed
    }

    /// Enter -> Exit from the veil reached: continuous per layer, no stagger, remaining time scaled by presence.
    fn fast_exit(&mut self, to: SceneKind) {
        let start = self.levels();
        self.st = St::Exit {
            to,
            x: 0.0,
            dir: 1.0,
            scale: (1.0 - start[3]).max(0.25),
            start,
            staged: false,
        };
    }

    /// Moves `x` by `dx` unless a veil would change faster than the per-frame limits; then by less.
    fn limited(x: f32, dx: f32, dt: f32, lim: f32, f: impl Fn(f32) -> [f32; 4]) -> f32 {
        let (v0, mut step) = (f(x), dx);
        let lim = lim.min(dt / SLEW_S);
        for _ in 0..4 {
            let d = max_delta(&v0, &f(x + step));
            if d <= lim || d <= 0.0 {
                break;
            }
            step *= 0.98 * lim / d;
        }
        x + step
    }

    /// Advances by `dt` seconds. `luma(scene)` is the scene's relative luminance (estimate; no GPU readback).
    pub fn update(&mut self, dt: f32, luma: impl Fn(SceneKind) -> f32) -> Frame {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.25)
        } else {
            0.0
        };
        self.clock += dt;
        let t0 = self.clock - WINDOW_S;
        self.requests.retain(|&t| t >= t0);
        let n = self.requests.len();
        let s = shorten(n);
        let prof = self.profile();

        // flash guard: three swings in the last second hold the bridge until requests stop
        let t1 = self.clock - 1.0;
        self.swings.retain(|&t| t >= t1);
        // the settle that follows the guard is itself a (last) swing, so it acts one swing early
        if self.swings.len() + 1 >= MAX_SWINGS_PER_S {
            self.holding = true;
        }
        if self.holding
            && self
                .requests
                .iter()
                .all(|&t| t < self.clock - HOLD_RELEASE_S)
        {
            self.holding = false;
            self.swings.clear();
        }
        if self.holding {
            match self.st {
                St::Exit {
                    to,
                    x,
                    scale,
                    start,
                    staged,
                    ..
                } => {
                    self.st = St::Exit {
                        to,
                        x,
                        dir: 1.0,
                        scale,
                        start,
                        staged,
                    };
                }
                St::Enter { .. } => self.fast_exit(self.shown),
                _ => {}
            }
        }

        let dest = match self.st {
            St::Exit { to, .. } | St::Bridge { to, .. } => to,
            _ => self.shown,
        };
        let (ys, yd) = (luma(self.shown), luma(dest));
        let by_target = bridge_target(ys, yd);
        let lum_lim = MAX_DL / (ys - by_target).abs().max(MAX_DL);

        match self.st {
            St::Idle => {}
            St::Exit {
                to,
                x,
                dir,
                scale,
                start,
                staged,
            } => {
                let dx = dir * dt / (prof.exit * s * scale).max(1e-3);
                let f = |x: f32| {
                    let b = layer_levels(x, true, staged);
                    std::array::from_fn(|i| start[i] + (1.0 - start[i]) * b[i])
                };
                let x = Self::limited(x, dx, dt, lum_lim, f).clamp(0.0, 1.0);
                self.st = if x >= 1.0 {
                    St::Bridge { to, t: 0.0 }
                } else if x <= 0.0 && dir < 0.0 {
                    if start == [0.0; 4] {
                        St::Idle
                    } else {
                        St::Enter {
                            x: 0.0,
                            start,
                            staged: false,
                        }
                    }
                } else {
                    St::Exit {
                        to,
                        x,
                        dir,
                        scale,
                        start,
                        staged,
                    }
                };
            }
            St::Bridge { to, t } => {
                let t = if self.holding { t } else { t + dt };
                self.st = if t >= prof.bridge * BRIDGE_ENTER_AT * s {
                    self.shown = to;
                    St::Enter {
                        x: 0.0,
                        start: [1.0; 4],
                        staged: prof.staged && n <= 3,
                    }
                } else {
                    St::Bridge { to, t }
                };
            }
            St::Enter { x, start, staged } => {
                let dx = dt / (prof.enter * s).max(1e-3);
                let f = |x: f32| {
                    let b = layer_levels(x, false, staged);
                    std::array::from_fn(|i| start[i] * b[i])
                };
                let x = Self::limited(x, dx, dt, lum_lim, f).clamp(0.0, 1.0);
                self.st = if x >= 1.0 {
                    St::Idle
                } else {
                    St::Enter { x, start, staged }
                };
            }
        }

        let veil = self.levels();
        let active = !matches!(self.st, St::Idle);
        // bridge luminance: low-passed so a retarget never jumps it; steady during a burst (n > 3)
        let bridge_luma = if active {
            let y = self.bridge_y.get_or_insert(by_target);
            if n <= 3 {
                *y += (by_target - *y) * (1.0 - (-dt / BRIDGE_TAU).exp());
            }
            *y
        } else {
            self.bridge_y = None;
            0.0
        };
        let y = ys * (1.0 - veil[3]) + bridge_luma * veil[3];
        self.track_swings(y);

        let phase = match self.st {
            St::Idle => Phase::Idle,
            St::Exit { .. } => Phase::Exit,
            St::Bridge { .. } => Phase::Bridge,
            St::Enter { .. } => Phase::Enter,
        };
        Frame {
            scene: self.shown,
            phase,
            veil,
            bridge_luma,
            breathing: if active && prof.breathing { 0.02 } else { 0.0 },
            iris: if active && prof.iris { 1.0 } else { 0.0 },
        }
    }

    fn track_swings(&mut self, y: f32) {
        if self.dir == 0 && self.swings.is_empty() && self.ext == 0.0 {
            self.ext = y;
        }
        let d = f32::from(self.dir);
        if self.dir == 0 {
            if (y - self.ext).abs() > SWING_AMP {
                self.dir = if y > self.ext { 1 } else { -1 };
                self.ext = y;
            }
        } else if (y - self.ext) * d > 0.0 {
            self.ext = y;
        } else if (self.ext - y) * d > SWING_AMP {
            self.swings.push(self.clock);
            self.dir = -self.dir;
            self.ext = y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use SceneKind::*;

    const DT: f32 = 1.0 / 60.0;
    const DIM: fn(SceneKind) -> f32 = |_| 0.2;

    fn step(s: &mut SceneSwitch) -> Frame {
        s.update(DT, DIM)
    }

    fn run(s: &mut SceneSwitch, secs: f32) -> Frame {
        let mut f = step(s);
        for _ in 1..(secs * 60.0).round() as usize {
            f = step(s);
        }
        f
    }

    /// Frames until Idle again (max 10 s), recording every frame.
    fn until_idle(s: &mut SceneSwitch) -> Vec<Frame> {
        let mut v = Vec::new();
        for _ in 0..600 {
            let f = step(s);
            v.push(f);
            if f.phase == Phase::Idle {
                break;
            }
        }
        v
    }

    /// Largest per-frame veil change between consecutive frames.
    fn max_step(frames: &[Frame]) -> f32 {
        frames
            .windows(2)
            .map(|w| max_delta(&w[0].veil, &w[1].veil))
            .fold(0.0, f32::max)
    }

    #[test]
    fn layer_levels_stage_exit_particles_first_and_enter_atmosphere_first() {
        let e = layer_levels(0.5, true, true);
        assert!(
            e[2] > e[1] && e[1] > e[0],
            "exit: particles, geometry, atmosphere {e:?}"
        );
        let n = layer_levels(0.5, false, true);
        assert!(
            n[0] < n[1] && n[1] < n[2],
            "enter: atmosphere, geometry, particles {n:?}"
        );
        assert_eq!(layer_levels(0.0, true, true), [0.0; 4]);
        assert_eq!(layer_levels(1.0, true, true), [1.0; 4]);
        assert_eq!(layer_levels(0.0, false, true), [1.0; 4]);
        assert_eq!(layer_levels(1.0, false, true), [0.0; 4]);
        let u = layer_levels(0.4, true, false);
        assert!(
            u.iter().all(|&v| v == u[0]),
            "unstaged layers share one curve"
        );
    }

    #[test]
    fn an_undisturbed_quick_switch_exits_bridges_enters_and_draws_one_scene() {
        let mut s = SceneSwitch::new(Spectrum);
        assert!(s.request(Orb));
        assert_eq!(s.target(), Orb);
        let frames = until_idle(&mut s);
        let secs = frames.len() as f32 * DT;
        let expect = QUICK.exit + QUICK.bridge * BRIDGE_ENTER_AT + QUICK.enter;
        assert!((secs - expect).abs() < 0.1, "{secs} vs {expect}");
        let first_enter = frames.iter().position(|f| f.phase == Phase::Enter).unwrap();
        assert!(
            frames[..first_enter].iter().all(|f| f.scene == Spectrum),
            "old scene until Enter"
        );
        assert!(
            frames[first_enter..].iter().all(|f| f.scene == Orb),
            "new scene from Enter"
        );
        let b = frames.iter().find(|f| f.phase == Phase::Bridge).unwrap();
        assert_eq!(b.veil, [1.0; 4]);
        assert!((0.04..=0.25).contains(&b.bridge_luma));
        let last = frames.last().unwrap();
        assert_eq!(
            (last.phase, last.scene, last.veil, last.bridge_luma),
            (Phase::Idle, Orb, [0.0; 4], 0.0)
        );
        assert_eq!((last.breathing, last.iris), (0.0, 0.0));
    }

    #[test]
    fn every_ordered_scene_pair_including_experimental_ones_transitions_smoothly() {
        for from in SceneKind::ALL {
            for to in SceneKind::ALL {
                let mut s = SceneSwitch::new(from);
                assert_eq!(s.request(to), from != to);
                let frames = if from == to {
                    vec![step(&mut s)]
                } else {
                    until_idle(&mut s)
                };
                let last = frames.last().unwrap();
                assert_eq!(
                    (last.phase, last.scene),
                    (Phase::Idle, to),
                    "{from:?}->{to:?}"
                );
                assert!(
                    max_step(&frames) <= DT / SLEW_S + 1e-4,
                    "{from:?}->{to:?} jump"
                );
                // exit veils only rise, enter veils only fall, the bridge sits at 1
                for w in frames.windows(2) {
                    match w[1].phase {
                        Phase::Exit => assert!(w[1].veil[3] >= w[0].veil[3]),
                        Phase::Enter => assert!(w[1].veil[3] <= w[0].veil[3] + 1e-6),
                        _ => {}
                    }
                }
            }
        }
    }

    #[test]
    fn reversing_an_exit_replays_the_same_frames_backwards_with_no_bridge() {
        let mut s = SceneSwitch::new(Spectrum);
        s.request(Aurora);
        let fwd: Vec<Frame> = (0..20).map(|_| step(&mut s)).collect();
        assert!(
            s.request(Spectrum),
            "asking for the scene being left reverses"
        );
        assert_eq!(s.target(), Spectrum);
        // (the request counts toward the adaptive shortening, so the backwards run is faster, but every frame
        // lies on the same curve: invert the global veil to x and compare all layers with layer_levels(x))
        let back: Vec<Frame> = (0..12).map(|_| step(&mut s)).collect();
        assert!(back[0].veil[3] < fwd[19].veil[3], "veil falls");
        for f in &back {
            let (mut lo, mut hi) = (0.0f32, 1.0f32);
            for _ in 0..40 {
                let m = 0.5 * (lo + hi);
                if layer_levels(m, true, true)[3] < f.veil[3] {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            let want = layer_levels(lo, true, true);
            for i in 0..4 {
                assert!((f.veil[i] - want[i]).abs() < 1e-3, "layer {i}");
            }
            assert_eq!(f.phase, Phase::Exit);
        }
        assert!(fwd.len() == 20);
        let rest = until_idle(&mut s);
        assert!(
            rest.iter()
                .all(|f| f.phase != Phase::Bridge && f.scene == Spectrum)
        );
        assert_eq!(rest.last().unwrap().phase, Phase::Idle);
        // and forward again: a fresh request resumes the exit with no jump
        s.request(Focus);
        assert!(step(&mut s).veil[3] < 0.05);
    }

    #[test]
    fn a_request_during_exit_or_bridge_replaces_the_target_without_a_jump() {
        let mut s = SceneSwitch::new(Spectrum);
        s.request(Aurora);
        let a = run(&mut s, 0.2);
        s.request(Focus);
        assert_eq!(s.target(), Focus);
        let b = step(&mut s);
        assert!(max_delta(&a.veil, &b.veil) <= DT / SLEW_S + 1e-4);
        assert_eq!(b.phase, Phase::Exit);
        let mut f = b;
        while f.phase != Phase::Bridge {
            f = step(&mut s);
        }
        s.request(Orb);
        assert_eq!(s.target(), Orb);
        let rest = until_idle(&mut s);
        let last = rest.last().unwrap();
        assert_eq!((last.scene, last.phase), (Orb, Phase::Idle));
    }

    #[test]
    fn a_request_during_enter_fast_exits_from_the_presence_reached() {
        for other in [Focus, Spectrum] {
            let mut s = SceneSwitch::new(Spectrum);
            s.request(Aurora);
            let mut f = step(&mut s);
            while f.phase != Phase::Enter {
                f = step(&mut s);
            }
            let before = run(&mut s, 0.3);
            assert_eq!(before.phase, Phase::Enter);
            assert!(!s.request(Aurora), "the scene entering is ignored");
            assert!(s.request(other));
            assert_eq!(s.target(), other);
            let after = step(&mut s);
            assert_eq!(after.phase, Phase::Exit, "never a hard cut");
            assert!(
                max_delta(&before.veil, &after.veil) <= DT / SLEW_S + 1e-4,
                "continuous per layer"
            );
            assert_eq!(
                after.scene, Aurora,
                "the entering scene keeps drawing while it leaves"
            );
            let mut frames = vec![after];
            frames.extend(until_idle(&mut s));
            assert!(max_step(&frames) <= DT / SLEW_S + 1e-4);
            let last = frames.last().unwrap();
            assert_eq!((last.scene, last.phase), (other, Phase::Idle));
            let exit_len = frames.iter().filter(|f| f.phase == Phase::Exit).count() as f32 * DT;
            assert!(
                exit_len < QUICK.exit,
                "remaining time is scaled: {exit_len}"
            );
        }
    }

    #[test]
    fn there_is_no_queue_the_latest_request_wins_and_the_machine_settles() {
        let kinds = SceneKind::ALL;
        let mut s = SceneSwitch::new(Spectrum);
        let mut rng = 12345u32;
        let mut last = Spectrum;
        for _ in 0..100 {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            last = kinds[(rng >> 16) as usize % kinds.len()];
            s.request(last);
            for _ in 0..(rng >> 28) {
                step(&mut s);
            }
            assert!(
                s.requests.len() <= 100
                    && s.requests.iter().all(|&t| s.clock - t <= WINDOW_S + 0.3)
            );
        }
        let end = until_idle(&mut s);
        let f = end.last().unwrap();
        assert_eq!(f.phase, Phase::Idle);
        assert_eq!(f.scene, s.target());
        // after the last request settles nothing is pending: the scene is the last one asked for unless it was
        // asked for while it was being left (that reverses to the shown scene), never an older request
        let f2 = run(&mut s, 5.0);
        assert_eq!((f2.phase, f2.scene), (Phase::Idle, f.scene));
        let _ = last;
        assert!(s.requests.is_empty(), "the window drained: no hidden state");
    }

    #[test]
    fn re_selecting_the_active_scene_when_idle_does_nothing() {
        let mut s = SceneSwitch::new(Focus);
        assert!(!s.request(Focus));
        let f = s.update(1.0, DIM);
        assert_eq!((f.phase, f.scene, f.veil), (Phase::Idle, Focus, [0.0; 4]));
        assert_eq!(s.target(), Focus);
    }

    #[test]
    fn hostile_time_steps_are_harmless() {
        let mut s = SceneSwitch::new(Spectrum);
        s.request(Aurora);
        for dt in [
            f32::NAN,
            -5.0,
            1e9,
            f32::INFINITY,
            0.0,
            f32::NEG_INFINITY,
            0.25,
        ] {
            let f = s.update(dt, DIM);
            assert!(
                f.veil
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                "{dt}: {f:?}"
            );
            assert!(f.bridge_luma.is_finite());
        }
        for _ in 0..200 {
            s.update(0.25, DIM);
        }
        assert_eq!(s.update(0.0, DIM).phase, Phase::Idle);
    }

    #[test]
    fn full_is_slower_than_quick_and_reduced_is_a_plain_fade_with_no_effects() {
        let total = |setup: &dyn Fn(&mut SceneSwitch)| {
            let mut s = SceneSwitch::new(Spectrum);
            setup(&mut s);
            until_idle(&mut s)
        };
        let quick = total(&|s| {
            s.request(Orb);
        });
        let full = total(&|s| {
            s.request_with(Orb, FULL);
        });
        let red = total(&|s| {
            s.set_reduced(true);
            s.request_with(Orb, FULL);
        });
        assert!(full.len() > quick.len() && quick.len() > red.len());
        assert!(
            (full.len() as f32 * DT - (FULL.exit + FULL.bridge * 0.6 + FULL.enter)).abs() < 0.1
        );
        assert!(
            full.iter().any(|f| f.breathing == 0.02) && quick.iter().all(|f| f.breathing == 0.0)
        );
        assert!(red.iter().all(|f| f.iris == 0.0 && f.breathing == 0.0));
        assert!(
            red.iter().all(|f| f.veil.iter().all(|&v| v == f.veil[0])),
            "no stagger when reduced"
        );
        assert!(
            quick.iter().any(|f| f.veil[0] != f.veil[2]),
            "quick staggers"
        );
    }

    #[test]
    fn rapid_switching_shortens_phases_down_to_a_floor() {
        assert_eq!(shorten(0), 1.0);
        assert_eq!(shorten(1), 1.0);
        assert!((shorten(2) - 0.625).abs() < 1e-6);
        assert_eq!(shorten(50), SHORTEN_FLOOR);
        let kinds = [Spectrum, Orb, Focus, Aurora];
        let mut s = SceneSwitch::new(Spectrum);
        let mut worst = 0.0f32;
        let mut frames = Vec::new();
        for i in 0..40 {
            s.request(kinds[(i + 1) % 4]);
            for _ in 0..6 {
                let f = step(&mut s);
                frames.push(f);
            }
        }
        for w in frames.windows(2) {
            worst = worst.max(max_delta(&w[0].veil, &w[1].veil));
        }
        assert!(
            worst <= DT / SLEW_S + 1e-4,
            "burst per-frame change {worst}"
        );
        let end = until_idle(&mut s);
        assert!(
            end.len() as f32 * DT < 2.0,
            "settles quickly after the burst"
        );
        // a burst of > 3 enters without the stagger
        let mut s = SceneSwitch::new(Spectrum);
        for k in [Aurora, Orb, Focus, Aurora, Orb] {
            s.request(k);
            step(&mut s);
        }
        let mut f = step(&mut s);
        for _ in 0..600 {
            if f.phase == Phase::Enter {
                break;
            }
            f = step(&mut s);
        }
        let g = step(&mut s);
        assert!(
            g.veil.iter().all(|&v| v == g.veil[0]),
            "n > 3: no enter stagger {g:?}"
        );
    }

    /// Swings of a luminance sequence with the guard's own definition, counted independently of the machine.
    fn count_swings(ys: &[f32]) -> Vec<usize> {
        let (mut ext, mut dir) = (ys[0], 0i8);
        let mut at = Vec::new();
        for (i, &y) in ys.iter().enumerate() {
            let d = f32::from(dir);
            if dir == 0 {
                if (y - ext).abs() > SWING_AMP {
                    dir = if y > ext { 1 } else { -1 };
                    ext = y;
                }
            } else if (y - ext) * d > 0.0 {
                ext = y;
            } else if (ext - y) * d > SWING_AMP {
                at.push(i);
                dir = -dir;
                ext = y;
            }
        }
        at
    }

    #[test]
    fn the_flash_guard_never_allows_more_than_three_luminance_swings_per_second() {
        let bright = |_: SceneKind| 0.9;
        let mut s = SceneSwitch::new(Spectrum);
        let mut ys = Vec::new();
        let mut frames = Vec::new();
        // hammer: reverse the exit every 0.1 s for 4 s
        for i in 0..240 {
            if i % 6 == 0 {
                s.request(if (i / 6) % 2 == 0 { Aurora } else { Spectrum });
            }
            let f = s.update(DT, bright);
            ys.push(0.9 * (1.0 - f.veil[3]) + f.bridge_luma * f.veil[3]);
            frames.push(f);
        }
        let swings = count_swings(&ys);
        for (k, &i) in swings.iter().enumerate() {
            let in_window = swings[k..].iter().take_while(|&&j| j < i + 60).count();
            assert!(
                in_window <= MAX_SWINGS_PER_S,
                "{in_window} swings within 1 s at frame {i}"
            );
        }
        assert!(
            !swings.is_empty(),
            "the hammer must actually swing luminance"
        );
        assert!(
            ys.windows(2).all(|w| (w[1] - w[0]).abs() <= MAX_DL + 1e-3),
            "per-frame luminance step"
        );
        assert!(
            frames.iter().any(|f| f.phase == Phase::Bridge),
            "the guard parks on the bridge"
        );
        // requests stop: it releases and finishes on the last scene asked for
        let end = (0..600).map(|_| s.update(DT, bright)).last().unwrap();
        assert_eq!((end.phase, end.scene), (Phase::Idle, s.target()));
    }

    #[test]
    fn the_bridge_is_never_black_never_brighter_than_either_scene_and_is_low_passed() {
        assert_eq!(bridge_target(0.0, 0.5), BRIDGE_MIN);
        assert_eq!(bridge_target(0.9, 0.9), BRIDGE_MAX);
        assert!((bridge_target(0.2, 0.9) - 0.16).abs() < 1e-6);
        assert!(bridge_target(0.1, 0.9) <= 0.1 || bridge_target(0.1, 0.9) == BRIDGE_MIN);
        // a retarget to a much brighter scene moves the bridge gradually (tau 0.3 s)
        let lum = |k: SceneKind| if k == Aurora { 0.05 } else { 0.9 };
        let mut s = SceneSwitch::new(Spectrum);
        s.request(Orb);
        let f = run(&mut s, 0.5);
        let y0 = f.bridge_luma;
        s.request(Aurora);
        let a = s.update(DT, lum);
        assert!(
            (a.bridge_luma - y0).abs() < 0.01,
            "no jump: {y0} -> {}",
            a.bridge_luma
        );
    }
}
