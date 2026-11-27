//! Music-to-motion mapper for Scene 1 (see docs\architecture\SPECTRUM-SCENE-MAPPING.md).
//!
//! Pure and deterministic: `AudioFrame` (real analysis) + elapsed time -> `SceneParams`. All timing is in
//! seconds (dt based), never frame counts, so music looks the same at any refresh rate. Every constant in
//! `MapTuning` is a PROVISIONAL engineering default, to be tuned on real music. Graphics quality never
//! enters this module: musical intensity and render cost are separate systems.
//!
//! Rule of timescales: fast -> geometry (bars, rings); medium -> brightness/local motion; slow -> colour,
//! density, speed. Colour listens only to slow signals, so a beat can never change colour.

use mg_dsp::AudioFrame;
use mg_dsp::features::MusicFeatures;
use mg_dsp::smooth::Smoother;
use mg_gfx::scene::{SLOTS, SceneMode, SceneParams};

#[derive(Clone, Debug)]
pub struct MapTuning {
    /// Max bar growth (fraction of full height per second).
    pub rise_per_s: f32,
    /// Bar fall speed (gravity-like).
    /// Bar fall speed range (fraction of full height per second): slow, liquid songs fall at the first value,
    /// dense, rhythmic songs at the second (from the long-term onset rate).
    pub fall_slow_per_s: f32,
    pub fall_fast_per_s: f32,
    pub peak_hold_s: f32,
    pub peak_fall_per_s: f32,
    /// Loudness gate: bars are scaled by smoothstep(lo, hi, rms_db) between `gate_floor` and 1.
    pub gate_lo_db: f32,
    pub gate_hi_db: f32,
    pub gate_floor: f32,
    /// Energy state follows `AudioFrame::energy` (seconds).
    pub energy_up_s: f32,
    pub energy_down_s: f32,
    pub ring_min_gap_s: f32,
    pub ring_amp: f32,
    /// A frame older than this (no new sequence number) is treated as absent audio.
    pub stale_after_s: f32,
}

impl Default for MapTuning {
    fn default() -> Self {
        Self {
            rise_per_s: 14.0,
            fall_slow_per_s: 0.6,
            fall_fast_per_s: 3.2,
            peak_hold_s: 0.25,
            peak_fall_per_s: 0.6,
            gate_lo_db: -55.0,
            gate_hi_db: -25.0,
            gate_floor: 0.25,
            energy_up_s: 3.0,
            energy_down_s: 6.0,
            ring_min_gap_s: 0.12,
            ring_amp: 0.15,
            stale_after_s: 0.25,
        }
    }
}

/// What the render loop hands the mapper each frame.
pub struct Input<'a> {
    pub frame: &'a AudioFrame,
    /// Onset events since the previous render frame, and the strongest strength among them (0..1).
    pub onsets: u32,
    pub onset_strength: f32,
    /// False when no new analysis frame arrived recently (capture stalled/failed) or none exists.
    pub fresh: bool,
    pub mode: SceneMode,
}

pub struct SceneMapper {
    tune: MapTuning,
    t: f32,
    scroll: [f32; 2],
    dust: f32,
    bars: [f32; SLOTS],
    peaks: [f32; SLOTS],
    hold: [f32; SLOTS],
    gate: Smoother,
    energy: Smoother,
    swell: Smoother,
    tilt: f32,
    glow: Smoother,
    haze: Smoother,
    song_tilt: f32,
    onset_rate: f32,
    rings: [[f32; 2]; 2],
    since_onset: f32,
    /// Aurora ribbon phases in cycles, wrapped at `PHASE_WRAP` (an integer, so the sines stay seamless).
    aurora_ph: [f32; 4],
    /// Orb: smoothed bands, flow and rotation phases (cycles, unwrapped: an hour is a few hundred), ripple impacts.
    orb_bass: Smoother,
    orb_mid: Smoother,
    orb_treble: Smoother,
    orb_breath: Smoother,
    orb_flow: f32,
    orb_rot: f32,
    orb_az: [f32; 2],
    /// Album Focus rotation phase (slow, grows without wrapping: one hour is only a few hundred units).
    focus_rot: f32,
    onset_count: u32,
    /// Latest music descriptors, rest values while audio is stale/absent. CPU-side only; never packed.
    music: MusicFeatures,
    fluid: FluidMap,
    /// slot -> (band position 0..1, group index)
    slot_map: [(f32, usize); SLOTS],
}

/// Fluid scene mapping (F1, Experimental; design: docs\design\OPUS-CONSULT-2026-11-25.md, "Addendum: Fluid
/// scene design"). Everything is smoothed here, dt based; the GPU sim only integrates. Emitter positions come
/// from phase accumulators that are never reset (tempo bends an orbit, it never jumps it). Each emitter has its
/// own orbit period, phase offset, spin direction and strength time constant: overlap and follow-through
/// instead of three parts moving in unison ("twinning").
struct FluidMap {
    /// Per emitter: x and y orbit phases (cycles, wrapped to 0..1; the sines make the wrap seamless).
    orbit: [[f32; 2]; 3],
    strength: [f32; 3],
    push: f32,
    speed: f32,
    freq: f32,
    phase: f32,
    tau: f32,
    build: f32,
    exposure: f32,
    gate: f32,
    vort: [[f32; 4]; 2],
    vhue: [f32; 2],
    since_vort: f32,
    /// Orbit reach 0..1 from the rhythm (onset density), slow.
    spread: f32,
    spawns: u32,
    /// Impulsiveness 0..1 (crest factor and onset density): fast attack, slow release. Impulsive music churns
    /// (fine curl, strong swirl, compact splats, short dye memory); sustained music drifts laminar and remembers.
    imp: f32,
    /// Sustained low-frequency lift (sh/s): a steady bass or drone pushes a broad plume up from the bass emitter.
    lift: f32,
    /// Treble spread 0..1: how far the treble splat widens into a thin veil.
    tspread: f32,
    /// Seconds an unconfirmed onset may still spawn a vortex, and its strength (0..1).
    onset_wait: f32,
    onset_str: f32,
    /// Seconds since the last real onset.
    onset_age: f32,
}

/// Lissajous periods (s) and phase offsets per emitter (bass, mid, treble), y/x frequency ratios (irrational).
const FL_PERIOD: [f32; 3] = [77.0, 48.0, 29.0];
const FL_OFFSET: [f32; 3] = [0.0, 0.37, 0.71];
const FL_RATIO: [f32; 3] = [1.618_034, std::f32::consts::SQRT_2, 1.732_051];
/// Orbit centre height (uv, y down) per emitter: bass lowest, treble highest.
const FL_HEIGHT: [f32; 3] = [0.34, 0.30, 0.27];
/// Strength follow time constants (s): bass leads, treble settles last.
const FL_STRENGTH_TAU: [f32; 3] = [0.25, 0.4, 0.6];
/// Spin per emitter (1/s at full strength); alternating sign.
const FL_SWIRL: [f32; 3] = [0.5, -0.7, 0.9];
/// Splat radius (sh) and gain per emitter: the bass paints dense ink, the treble a lighter, airier veil.
const FL_SIGMA: [f32; 3] = [0.085, 0.07, 0.08];
const FL_GAIN: [f32; 3] = [1.0, 0.9, 0.65];
/// Minimum time between two onset vortices and how long one lives (its envelope is ~0 by then).
pub const FLUID_VORTEX_GAP_S: f32 = 0.25;
const FL_VORTEX_LIFE_S: f32 = 2.0;
/// How long an onset may wait for its accent to confirm it (the features trail the onset by a frame or two).
const FL_ONSET_WAIT_S: f32 = 0.1;
/// Everything the sim injects stays in this uv box (y down): above artwork, lyrics and the control bar.
pub const FLUID_SAFE_X: (f32, f32) = (0.15, 0.85);
pub const FLUID_SAFE_Y: (f32, f32) = (0.12, 0.58);
/// Slowest flow ever (sh/s): silence still breathes, barely.
pub const FLUID_SPEED_FLOOR: f32 = 0.008;
/// Slowest flow while music plays (sh/s): a calm pad still drifts.
const FL_MUSIC_SPEED_MIN: f32 = 0.03;
/// Sustained low-end lift: peak upward speed (sh/s) and jet width (sh): a broad, slow, large-scale plume.
const FL_LIFT: f32 = 0.1;
const FL_LIFT_WIDTH: f32 = 0.3;

fn ema(y: f32, x: f32, dt: f32, tau: f32) -> f32 {
    y + (x - y) * (1.0 - (-dt / tau).exp())
}

impl FluidMap {
    fn new() -> Self {
        Self {
            orbit: [[0.0; 2]; 3],
            strength: [0.0; 3],
            push: 0.0,
            speed: FLUID_SPEED_FLOOR,
            freq: 3.0,
            phase: 0.0,
            tau: 3.0,
            build: 0.0,
            exposure: 0.85,
            gate: 0.0,
            vort: [[0.5, 0.3, 0.0, 100.0]; 2],
            vhue: [0.0; 2],
            since_vort: 10.0,
            spread: 0.0,
            spawns: 0,
            imp: 0.0,
            lift: 0.0,
            tspread: 0.0,
            onset_wait: 0.0,
            onset_str: 0.0,
            onset_age: 100.0,
        }
    }

    /// `silent`: no fresh audio or the analyzer's silence flag. `absolute`: the frame's absolute group levels
    /// (`AudioFrame::group_medium`, 0..1). `dt` is already sanitised by the caller.
    fn update(
        &mut self,
        m: &MusicFeatures,
        absolute: &[f32; mg_dsp::GROUPS],
        silent: bool,
        onsets: u32,
        onset_strength: f32,
        dt: f32,
    ) -> mg_gfx::fluid::FluidParams {
        let c = |v: f32| fin(v).clamp(0.0, 1.0);
        let band = |a: &[f32; mg_dsp::GROUPS], i: usize| c(a[2 * i]).max(c(a[2 * i + 1]));
        // AGC normalises every band to its own recent range, so it cannot tell a bass-heavy song from a bright
        // one; the absolute band level gates each emitter (presence) and AGC shapes it within that.
        // The analyzer's 60-250 Hz group also catches the skirt of a strong 250-500 Hz group (a 262 Hz chord
        // reads ~0.65 there): the bass presence discounts half of the group above, so a pad does not drive the
        // bass emitter. Sub-bass (< 60 Hz) always counts in full.
        let low_abs = c(absolute[0]).max(c(absolute[1]) - 0.5 * c(absolute[2]));
        let presence = |i: usize| (2.0 * if i == 0 { low_abs } else { band(absolute, i) }).min(1.0);
        let drive = |i: usize| 0.4 * band(&m.long, i) + 0.6 * band(&m.short, i);
        let level = if silent { 0.0 } else { c(m.level) };
        let ceiling = 0.5 + 0.5 * level; // quiet songs stay quiet even with full AGC
        // sustained sound (pads) flows slowly; rhythm (onsets per second) lets it move
        let rhythm = (fin(m.onset_density) / 3.0).clamp(0.0, 1.0);
        let calm = c(m.calm);
        // impulsiveness: a peaky signal (crest ~1 s), a dense rhythm, or a real onset just now (the averages
        // above need seconds to notice hats); reacts within ~0.3 s, settles over ~2 s
        self.onset_age = (self.onset_age + dt).min(100.0);
        if !silent && onsets > 0 {
            self.onset_age = 0.0;
        }
        let imp_t = if silent {
            0.0
        } else {
            // a steady chord's beating reaches ~8 dB, a 2 Hz kick ~14 dB; two onsets per second already count fully
            smoothstep(9.0, 15.0, fin(m.crest_db))
                .max((fin(m.onset_density) / 2.0).clamp(0.0, 1.0))
                .max(if self.onset_age < 0.15 { 0.7 } else { 0.0 })
        };
        self.imp = ema(
            self.imp,
            imp_t,
            dt,
            if imp_t > self.imp { 0.3 } else { 2.0 },
        );
        let sustain = 1.0 - self.imp;
        // build: gentle intensification, slew-limited to 0.1 per second so a riser never jumps
        let b_t = if silent { 0.0 } else { c(m.build) };
        self.build += (b_t - self.build).clamp(-0.1 * dt, 0.1 * dt);
        // injection gate: exactly 0 within 0.75 s of the analyzer's silence flag; back in 0.4 s
        self.gate = if silent {
            (self.gate - dt / 0.75).max(0.0)
        } else {
            (self.gate + dt / 0.4).min(1.0)
        };
        // slow, phase-continuous drivers
        let speed_t = if silent {
            FLUID_SPEED_FLOOR
        } else {
            ((0.05
                + 0.20 * presence(1) * band(&m.long, 1)
                + 0.12 * presence(0) * band(&m.long, 0)
                + 0.12 * presence(2) * band(&m.long, 2))
                * (1.0 - 0.6 * calm)
                * (0.25 + 0.75 * rhythm)
                * (1.0 + 0.4 * self.build))
                .max(FL_MUSIC_SPEED_MIN)
        };
        self.speed = ema(self.speed, speed_t.max(FLUID_SPEED_FLOOR), dt, 1.0);
        if !silent {
            // structure scale: bright sound -> fine curl, low sound -> large, slow structures (sqrt: a pad's
            // centroid of ~0.1 already reads as finer than a pure bass)
            let fq = 2.4 + 1.6 * c(m.centroid).sqrt() * (1.0 - 0.5 * presence(0));
            self.freq = ema(self.freq, fq, dt, 2.0);
            // a sustained high tone spreads; hats (impulsive) stay compact
            self.tspread = ema(self.tspread, presence(2) * sustain, dt, 1.0);
        }
        self.phase = (self.phase + dt * (0.03 + 0.12 * band(&m.long, 2))).rem_euclid(256.0);
        // dye memory: impulsive music forgets fast (separate puffs), sustained calm music keeps long trails
        let tau_t = if silent {
            3.0
        } else {
            3.0 + 6.0 * sustain * (0.3 + 0.7 * calm)
        };
        self.tau = ema(self.tau, tau_t, dt, if silent { 0.3 } else { 1.5 });
        // a steady low end lifts a broad plume (slow ease in and out); impacts are the push below
        let lift_t = if silent {
            0.0
        } else {
            FL_LIFT * presence(0) * sustain * ceiling
        };
        self.lift = ema(self.lift, lift_t, dt, 1.5);
        // exposure: one-pole (1.5 s) and slew-limited (0.3/s): never a flash
        let e_t = 0.85 + 0.3 * level;
        let step = ema(self.exposure, e_t, dt, 1.5) - self.exposure;
        self.exposure += step.clamp(-0.3 * dt, 0.3 * dt);
        // bass accent (short above its own long average: an impact, not a steady bass line) -> push (fast attack,
        // slow release: ease-out), heat and a wider bass splat
        let bs = if silent {
            0.0
        } else {
            band(&m.accent, 0) * presence(0)
        };
        let push_t = 0.25 * bs * ceiling;
        self.push = ema(
            self.push,
            push_t,
            dt,
            if push_t > self.push { 0.04 } else { 0.3 },
        );
        // energetic music travels the canvas, calm music stays near where it is (never a jump: rate, not phase)
        let orbit_rate = 0.4 + 8.0 * self.speed;
        if !silent {
            self.spread = ema(self.spread, rhythm, dt, 2.0);
        }
        let mut emit = [[0.0f32; 4]; 3];
        let mut swirl = [0.0f32; 4];
        for i in 0..3 {
            let target = if silent {
                0.0
            } else {
                FL_GAIN[i] * ceiling * presence(i) * (0.4 + 0.6 * drive(i))
            };
            self.strength[i] = ema(self.strength[i], target, dt, FL_STRENGTH_TAU[i]);
            let o = &mut self.orbit[i];
            o[0] = (o[0] + dt * orbit_rate / FL_PERIOD[i]).fract();
            o[1] = (o[1] + dt * orbit_rate * FL_RATIO[i] / FL_PERIOD[i]).fract();
            let tau2 = std::f32::consts::TAU;
            // calm music gathers near the middle, rhythmic music spreads; heavier bands sit lower
            let reach = 0.55 + 0.45 * self.spread;
            let x = 0.5 + 0.33 * reach * (tau2 * (o[0] + FL_OFFSET[i])).sin();
            // impulsive bass strikes from lower down (the jet throws it up); a sustained low end hangs higher
            let strike = if i == 0 { 0.07 * self.imp } else { 0.0 };
            let y =
                FL_HEIGHT[i] + strike + 0.14 * reach * (tau2 * (o[1] + 1.7 * FL_OFFSET[i])).sin();
            // sustained sound spreads a broad soft veil, impulsive sound compact dense puffs; kicks splash wider
            let broad = 1.0 + 0.6 * sustain;
            let sigma = FL_SIGMA[i] * broad + if i == 0 { 0.2 * self.push } else { 0.0 };
            // a bass impact (push) splashes extra dye: kicks paint, a steady bass line only flows
            // the treble veil thins as it spreads (up to 3x wider, half the peak): airy, never a slab
            let splash = match i {
                0 => 1.0 + 4.0 * self.push,
                2 => 1.0 / (1.0 + self.tspread),
                _ => 1.0,
            };
            let s =
                (self.strength[i] * splash * self.gate * (1.0 + 0.3 * self.build)).clamp(0.0, 1.3);
            // defensive: the orbit already lies inside the safe zone; a clamp keeps it there whatever the inputs
            let (x, y) = (
                x.clamp(FLUID_SAFE_X.0, FLUID_SAFE_X.1),
                y.clamp(FLUID_SAFE_Y.0, FLUID_SAFE_Y.1),
            );
            // the splat never reaches the lyrics line (0.62)
            emit[i] = [x, y, s, sigma.min(0.6 - y)];
            // impulsive music spins its emitters, sustained music barely stirs them (laminar drift)
            swirl[i] = FL_SWIRL[i] * (0.3 + 0.7 * self.strength[i]) * (0.4 + 0.6 * self.imp);
        }
        swirl[3] = (self.push + self.lift) * self.gate;
        // the jet is as narrow as the bass splat for impacts and broad (large-scale) for a sustained low end
        let jet_w = (self.push * 2.0 * emit[0][3] + self.lift * FL_LIFT_WIDTH)
            / (self.push + self.lift).max(1e-4);
        let heat = 1.5 * bs * ceiling * self.gate;

        // onset + accent -> one smooth vortex (alpha envelope in the shader), >= 250 ms apart, two slots
        for v in &mut self.vort {
            v[3] = (v[3] + dt).min(100.0);
            if v[3] > FL_VORTEX_LIFE_S {
                *v = [v[0], v[1], 0.0, 100.0];
            }
        }
        self.since_vort += dt;
        // the features trail the onset by a frame or two (the short envelope has not risen yet on the onset's
        // own tick), so an onset waits up to FL_ONSET_WAIT_S for its accent to confirm it
        self.onset_wait = (self.onset_wait - dt).max(0.0);
        if !silent && onsets > 0 {
            self.onset_wait = FL_ONSET_WAIT_S;
            self.onset_str = c(onset_strength);
        }
        let accent = (0..3).map(|i| band(&m.accent, i)).fold(0.0f32, f32::max);
        if !silent
            && self.onset_wait > 0.0
            && self.since_vort >= FLUID_VORTEX_GAP_S
            && accent > 0.04 + 0.12 * calm
        {
            self.onset_wait = 0.0;
            self.since_vort = 0.0;
            self.spawns = self.spawns.wrapping_add(1);
            let dom = (0..3)
                .max_by(|&a, &b| band(&m.accent, a).total_cmp(&band(&m.accent, b)))
                .unwrap_or(0);
            let ang = self.spawns as f32 * 2.399_963; // golden angle: never the same spot twice
            let x = (emit[dom][0] + 0.08 * ang.cos()).clamp(0.2, 0.8);
            let y = (emit[dom][1] + 0.08 * ang.sin()).clamp(0.15, 0.5);
            let sign = if self.spawns.is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            let amp = sign * 0.3 * (0.4 + 0.6 * self.onset_str) * ceiling * (1.0 - 0.5 * calm);
            let slot = usize::from(self.vort[1][3] > self.vort[0][3]); // the older one
            self.vort[slot] = [x, y, amp, 0.0];
            self.vhue[slot] = c(m.centroid);
        }
        mg_gfx::fluid::FluidParams {
            emit,
            swirl,
            vort: self.vort,
            flow: [self.speed, self.freq, self.phase, 0.012],
            look: [self.tau, heat, self.exposure, jet_w],
            misc: [
                self.vhue[0],
                self.vhue[1],
                0.2 + 0.8 * self.imp,
                self.tspread,
            ],
        }
    }
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn fin(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Frequency group of a band position (log axis 30 Hz..16 kHz), matching mg-dsp's group edges.
fn group_of(pos: f32) -> usize {
    let hz = 30.0 * (16_000.0f32 / 30.0).powf(pos);
    [60.0, 250.0, 500.0, 2000.0, 4000.0]
        .iter()
        .filter(|&&e| hz >= e)
        .count()
}

impl SceneMapper {
    pub fn new(tune: MapTuning) -> Self {
        let mut slot_map = [(0.0, 0); SLOTS];
        for (i, m) in slot_map.iter_mut().enumerate() {
            let c = (i as f32 - (SLOTS as f32 - 1.0) / 2.0).abs();
            let pos = (c / ((SLOTS as f32 - 1.0) / 2.0)).clamp(0.0, 1.0);
            *m = (pos, group_of(pos));
        }
        let e = (tune.energy_up_s * 1000.0, tune.energy_down_s * 1000.0);
        Self {
            gate: Smoother::new(150.0, 600.0),
            energy: Smoother::new(e.0, e.1),
            swell: Smoother::new(120.0, 500.0),
            tilt: 0.0,
            glow: Smoother::new(80.0, 400.0),
            haze: Smoother::new(600.0, 2000.0),
            tune,
            t: 0.0,
            scroll: [0.0, 0.0],
            dust: 0.0,
            bars: [0.0; SLOTS],
            peaks: [0.0; SLOTS],
            hold: [0.0; SLOTS],
            song_tilt: 0.0,
            onset_rate: 1.5,
            rings: [[100.0, 0.0]; 2],
            since_onset: 10.0,
            aurora_ph: [0.0; 4],
            orb_bass: Smoother::new(60.0, 700.0),
            orb_mid: Smoother::new(150.0, 900.0),
            orb_treble: Smoother::new(100.0, 500.0),
            orb_breath: Smoother::new(400.0, 1500.0),
            orb_flow: 0.0,
            orb_rot: 0.0,
            orb_az: [0.5; 2],
            focus_rot: 0.0,
            onset_count: 0,
            slot_map,
            music: MusicFeatures::default(),
            fluid: FluidMap::new(),
        }
    }

    /// Records this frame's music features; stale or absent audio gives the rest values (no fake
    /// activity). Scenes read it through `music()` to derive their own uniform fields.
    pub fn observe_music(&mut self, m: &MusicFeatures, fresh: bool) {
        self.music = if fresh { *m } else { MusicFeatures::default() };
    }

    pub fn music(&self) -> &MusicFeatures {
        &self.music
    }

    /// Advances the scene by `dt` seconds (clamped to 0..0.25) and returns the parameters to draw.
    pub fn update(&mut self, input: &Input, dt: f32) -> SceneParams {
        let dt = fin(dt).clamp(0.0, 0.25);
        let silent_frame = AudioFrame::default();
        // Stale or unavailable audio is shown as silence / error state, never as invented activity.
        let live = input.fresh;
        let f = if live { input.frame } else { &silent_frame };
        let silent = f.silent || !live;
        self.t += dt;

        // ---- fast: bars (frequency-specific, mirrored around the bass centre) -------------------
        let bands = f.band_count.clamp(1, mg_dsp::MAX_BANDS);
        let gate_t = if silent {
            0.0
        } else {
            self.tune.gate_floor
                + (1.0 - self.tune.gate_floor)
                    * smoothstep(self.tune.gate_lo_db, self.tune.gate_hi_db, fin(f.rms_db))
        };
        let gate = self.gate.step(gate_t, dt);
        // Song-adaptive motion: long-term onset rate (tau 20 s, only while music plays) sets the fall speed.
        if live && !silent && dt > 0.0 {
            let inst = input.onsets as f32 / dt;
            self.onset_rate += (inst - self.onset_rate) * (1.0 - (-dt / 20.0).exp());
        }
        let rate_n = (self.onset_rate / 4.0).clamp(0.0, 1.0);
        let fall = self.tune.fall_slow_per_s
            + (self.tune.fall_fast_per_s - self.tune.fall_slow_per_s) * rate_n;
        let mut target = [0.0f32; SLOTS];
        for (i, tg) in target.iter_mut().enumerate() {
            let fp = self.slot_map[i].0 * (bands as f32 - 1.0);
            let (i0, fr) = (fp.floor() as usize, fp.fract());
            let i1 = (i0 + 1).min(bands - 1);
            *tg = (fin(f.bands[i0]) * (1.0 - fr) + fin(f.bands[i1]) * fr).clamp(0.0, 1.0);
        }
        // 3-tap spatial smoothing (a second pass for the wider bass kernel near the centre).
        for pass in 0..2 {
            let src = target;
            for i in 0..SLOTS {
                if pass == 1 && self.slot_map[i].0 > 0.2 {
                    continue;
                }
                let (l, r) = (src[i.saturating_sub(1)], src[(i + 1).min(SLOTS - 1)]);
                target[i] = 0.25 * l + 0.5 * src[i] + 0.25 * r;
            }
        }
        for i in 0..SLOTS {
            let tg = target[i] * gate;
            let h = self.bars[i];
            self.bars[i] = if tg > h {
                tg.min(h + self.tune.rise_per_s * dt)
            } else {
                tg.max(h - fall * dt)
            }
            .clamp(0.0, 1.0);
            // peak caps: hold, then ease into a fall
            if self.bars[i] >= self.peaks[i] {
                self.peaks[i] = self.bars[i];
                self.hold[i] = 0.0;
            } else {
                self.hold[i] += dt;
                if self.hold[i] > self.tune.peak_hold_s {
                    let ease = ((self.hold[i] - self.tune.peak_hold_s) / 0.3 + 0.2).min(1.0);
                    self.peaks[i] =
                        (self.peaks[i] - self.tune.peak_fall_per_s * ease * dt).max(self.bars[i]);
                }
            }
        }
        let mut bright = [0.0f32; SLOTS];
        for (i, b) in bright.iter_mut().enumerate() {
            *b = fin(f.group_medium[self.slot_map[i].1]).clamp(0.0, 1.0);
        }

        // ---- medium: environment ---------------------------------------------------------------
        let gm = |g: usize| fin(f.group_medium[g]).clamp(0.0, 1.0);
        let gs = |g: usize| fin(f.group_slow[g]).clamp(0.0, 1.0);
        let swell = self.swell.step(0.6 * gm(0) + 0.4 * gm(1), dt);
        let glow_t = if silent {
            0.25
        } else {
            0.35 + 0.4 * fin(f.energy).clamp(0.0, 1.0) + 0.15 * 0.5 * (gm(4) + gm(5))
        };
        let glow = self.glow.step(glow_t.clamp(0.0, 1.0), dt);

        // ---- slow: adaptive intensity, atmosphere, colour --------------------------------------
        let e_t = if silent {
            0.1
        } else {
            fin(f.energy).clamp(0.0, 1.0)
        };
        let s = self.energy.step(e_t, dt).clamp(0.0, 1.0);
        let haze = self
            .haze
            .step(0.3 + 0.7 * (0.5 * s + 0.5 * swell.min(1.0)), dt)
            .clamp(0.0, 1.0);
        let (bass, treble) = (gs(0) + gs(1), gs(4) + gs(5));
        // signed one-pole (tau 4 s); starts neutral
        let tilt_t = ((bass - treble) / (bass + treble + 0.05)).clamp(-1.0, 1.0);
        self.tilt += (tilt_t - self.tilt) * (1.0 - (-dt / 4.0).exp());
        let tilt = self.tilt;
        let warp = 0.15 + 0.35 * s;
        let speed = 0.4 + 0.8 * s;
        self.scroll[0] += 0.02 * speed * dt;
        self.scroll[1] += 0.012 * speed * dt;
        // dust falls (0.015 + 0.04 s) screen heights per second = that many / 0.022 cells; one phase unit = 128 cells
        self.dust = (self.dust + (0.015 + 0.04 * s) / 0.022 * dt / 128.0).fract();
        // Aurora: ribbon phases advance faster with intensity; Focus rotates slowly; band levels bass -> treble.
        const PHASE_WRAP: f32 = 64.0;
        for (i, ph) in self.aurora_ph.iter_mut().enumerate() {
            *ph = (*ph + dt * (0.04 + 0.20 * s) * (1.0 + 0.15 * i as f32)) % PHASE_WRAP;
        }
        self.focus_rot += dt * (0.02 + 0.06 * s);
        let mut aurora_band = [0.0f32; 4];
        let mut count = [0u32; 4];
        for i in 0..SLOTS {
            let g = ((self.slot_map[i].0 * 4.0) as usize).min(3);
            aurora_band[g] += self.bars[i];
            count[g] += 1;
        }
        for (b, n) in aurora_band.iter_mut().zip(count) {
            *b = (*b / n.max(1) as f32).clamp(0.0, 1.0);
        }

        // Orb: bass/mid/treble smoothed here (bass attack 60 ms, release 700 ms); mid sets flow and rotation
        // speed, and calm music (low slow energy) caps both. Flow/rotation phases are integrated, never reset.
        let mix = |a: usize, b: usize| 0.5 * (gm(a) + gm(b));
        let quiet = |v: f32| if silent { 0.0 } else { v };
        let o_bass = self
            .orb_bass
            .step(quiet(0.6 * gm(0) + 0.4 * gm(1)), dt)
            .clamp(0.0, 1.0);
        let o_mid = self.orb_mid.step(quiet(mix(2, 3)), dt).clamp(0.0, 1.0);
        let o_treble = self.orb_treble.step(quiet(mix(4, 5)), dt).clamp(0.0, 1.0);
        let o_breath = self
            .orb_breath
            .step(quiet(0.6 * gm(0) + 0.4 * gm(1)), dt)
            .clamp(0.0, 1.0);
        self.orb_flow += dt * (0.012 + 0.10 * o_mid).min(0.03 + 0.14 * s);
        self.orb_rot += dt * (0.006 + 0.03 * o_mid).min(0.01 + 0.04 * s);
        let orb_band = [o_bass, o_mid, o_treble, o_breath];

        // Colour: two fixed palette endpoints, blended by the slow per-song spectral tilt (tau 20 s).
        // Bass-heavy songs sit near teal+amber, bright songs near periwinkle+rose. Music only moves lightness
        // (slow energy) and this slow blend; there is no hue from energy and nothing per beat.
        let song_t = if silent {
            self.song_tilt
        } else {
            (bass - treble) / (bass + treble + 0.05)
        };
        self.song_tilt += (song_t.clamp(-1.0, 1.0) - self.song_tilt) * (1.0 - (-dt / 20.0).exp());
        let b = (0.5 - 0.9 * self.song_tilt).clamp(0.0, 1.0); // 0 = bass-heavy endpoint, 1 = bright endpoint
        let k = 0.85 + 0.30 * s;
        let mix3 = |a: [f32; 3], c: [f32; 3]| [0, 1, 2].map(|i| (a[i] + (c[i] - a[i]) * b) * k);
        let body = mix3([0.114, 0.480, 0.546], [0.275, 0.392, 0.871]);
        let accent = mix3([0.888, 0.552, 0.266], [0.871, 0.254, 0.323]);
        let palette = [
            mix3([0.020, 0.120, 0.160], [0.030, 0.060, 0.180]),
            body,
            accent,
            [0, 1, 2].map(|i| 0.5 * (body[i] + accent[i])),
        ];

        // ---- fast: onset impulses (real onsets only; refractory; capped) -------------------------
        for r in &mut self.rings {
            r[0] = (r[0] + dt).min(100.0);
        }
        self.since_onset += dt;
        if live && !silent && input.onsets > 0 && self.since_onset >= self.tune.ring_min_gap_s {
            self.since_onset = 0.0;
            self.onset_count = self.onset_count.wrapping_add(1);
            self.rings[1] = self.rings[0];
            self.orb_az[1] = self.orb_az[0];
            self.orb_az[0] = (self.onset_count as f32 * 0.618_034).fract();
            let amp = self.tune.ring_amp * fin(input.onset_strength).clamp(0.0, 1.0);
            self.rings[0] = [0.0, amp.min(0.25)];
        }
        // Fluid: its own block (never packed into the shared uniform); uses last frame's music features.
        let fluid = self.fluid.update(
            &self.music,
            &f.group_medium,
            silent,
            if live { input.onsets } else { 0 },
            input.onset_strength,
            dt,
        );

        SceneParams {
            fluid,
            time: self.t,
            energy: s,
            glow,
            haze,
            warp,
            swell: swell.clamp(0.0, 1.0),
            tilt,
            scroll: self.scroll,
            dust: self.dust,
            aurora_phase: self.aurora_ph,
            aurora_band,
            orb_flow: [self.orb_flow, self.orb_rot, self.orb_az[0], self.orb_az[1]],
            orb_band,
            focus_rot: self.focus_rot,
            front_x0: (self.onset_count as f32 * 0.618_034).fract(),
            // the runner overwrites `focus` and `focus_active` from the artwork layout
            focus: [0.5, 0.4, 0.16, 0.16],
            focus_active: 0.0,
            rings: self.rings,
            palette,
            mode: if live { input.mode } else { SceneMode::NoAudio },
            bars: self.bars,
            peaks: self.peaks,
            bright,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same loud signal, then silence: only the (forced) long-term onset rate differs.
    fn bar_after_release(onset_rate: f32) -> f32 {
        let mut m = SceneMapper::new(MapTuning::default());
        let loud = AudioFrame {
            silent: false,
            bands: [1.0; mg_dsp::MAX_BANDS],
            band_count: mg_dsp::MAX_BANDS,
            rms_db: -10.0,
            ..AudioFrame::default()
        };
        let input = |frame, fresh| Input {
            frame,
            onsets: 0,
            onset_strength: 0.0,
            fresh,
            mode: SceneMode::Live,
        };
        for _ in 0..90 {
            m.update(&input(&loud, true), 1.0 / 60.0);
        }
        m.onset_rate = onset_rate; // set after the loud phase so only the release is affected
        let mut p = m.update(&input(&loud, false), 1.0 / 60.0);
        for _ in 0..24 {
            p = m.update(&input(&loud, false), 1.0 / 60.0);
        }
        p.bars.iter().cloned().fold(0.0, f32::max)
    }

    #[test]
    fn bars_fall_faster_for_rhythmic_songs_with_everything_else_equal() {
        let slow_song = bar_after_release(0.0);
        let fast_song = bar_after_release(4.0);
        assert!(slow_song > 0.4, "liquid songs keep their bars: {slow_song}");
        assert!(
            fast_song + 0.25 < slow_song,
            "rhythmic {fast_song} vs liquid {slow_song}"
        );
    }

    #[test]
    fn dust_phase_wraps_and_never_jumps() {
        let mut m = SceneMapper::new(MapTuning::default());
        let f = AudioFrame::default();
        let mut prev = 0.0f32;
        for i in 0..200_000 {
            let p = m.update(
                &Input {
                    frame: &f,
                    onsets: 0,
                    onset_strength: 0.0,
                    fresh: true,
                    mode: SceneMode::Live,
                },
                1.0 / 60.0,
            );
            assert!((0.0..1.0).contains(&p.dust), "phase {}", p.dust);
            if i > 0 && p.dust > prev {
                assert!(p.dust - prev < 0.01, "jump {prev} -> {}", p.dust);
            }
            prev = p.dust;
        }
    }

    fn live(frame: &AudioFrame) -> Input<'_> {
        Input {
            frame,
            onsets: 0,
            onset_strength: 0.0,
            fresh: true,
            mode: SceneMode::Live,
        }
    }

    #[test]
    fn aurora_bands_follow_frequency_and_phases_stay_wrapped_and_continuous() {
        let mut m = SceneMapper::new(MapTuning::default());
        // energy only in the lowest bands: ribbon 0 (bass) must be taller than ribbon 3 (treble)
        let mut bands = [0.0f32; mg_dsp::MAX_BANDS];
        for b in bands.iter_mut().take(10) {
            *b = 0.9;
        }
        let bass = AudioFrame {
            silent: false,
            bands,
            band_count: mg_dsp::MAX_BANDS,
            rms_db: -10.0,
            energy: 0.9,
            ..AudioFrame::default()
        };
        let mut p = m.update(&live(&bass), 1.0 / 60.0);
        for _ in 0..120 {
            p = m.update(&live(&bass), 1.0 / 60.0);
        }
        assert!(
            p.aurora_band[0] > p.aurora_band[3] + 0.1,
            "{:?}",
            p.aurora_band
        );
        assert!(p.aurora_band.iter().all(|b| (0.0..=1.0).contains(b)));
        // phases advance, never leave 0..64, and the only large step is the wrap itself
        let mut prev = p.aurora_phase;
        let mut wrapped = false;
        for _ in 0..(60 * 600) {
            let q = m.update(&live(&bass), 1.0 / 60.0);
            for i in 0..4 {
                assert!(
                    (0.0..64.0).contains(&q.aurora_phase[i]),
                    "{:?}",
                    q.aurora_phase
                );
                let step = q.aurora_phase[i] - prev[i];
                if step < 0.0 {
                    wrapped = true;
                    assert!((step + 64.0).abs() < 0.1, "wrap step {step}");
                } else {
                    assert!(step < 0.1, "phase jumped by {step}");
                }
            }
            prev = q.aurora_phase;
        }
        assert!(wrapped, "ten minutes of loud music must reach the wrap");
    }

    #[test]
    fn the_onset_front_origin_changes_with_every_onset() {
        let mut m = SceneMapper::new(MapTuning::default());
        let f = AudioFrame {
            silent: false,
            bands: [0.8; mg_dsp::MAX_BANDS],
            band_count: mg_dsp::MAX_BANDS,
            rms_db: -10.0,
            ..AudioFrame::default()
        };
        let mut origins = Vec::new();
        for k in 0..4 {
            for _ in 0..30 {
                m.update(&live(&f), 1.0 / 60.0);
            }
            let p = m.update(
                &Input {
                    onsets: 1,
                    onset_strength: 0.8,
                    ..live(&f)
                },
                1.0 / 60.0,
            );
            origins.push(p.front_x0);
            assert!((0.0..1.0).contains(&p.front_x0), "onset {k}");
        }
        origins.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        assert_eq!(origins.len(), 4, "every onset moves the front origin");
    }

    fn loud_frame(bands_lo: f32, bands_hi: f32) -> AudioFrame {
        let mut f = AudioFrame {
            silent: false,
            band_count: mg_dsp::MAX_BANDS,
            rms_db: -10.0,
            energy: 0.8,
            ..AudioFrame::default()
        };
        f.group_medium = [bands_lo, bands_lo, 0.1, 0.1, bands_hi, bands_hi];
        f.group_slow = f.group_medium;
        f
    }

    fn run_orb(f: &AudioFrame, dt: f32, secs: f32) -> SceneParams {
        let mut m = SceneMapper::new(MapTuning::default());
        let mut p = m.update(&live(f), dt);
        for _ in 0..(secs / dt) as usize {
            p = m.update(&live(f), dt);
        }
        p
    }

    #[test]
    fn orb_bands_follow_frequency_are_bounded_and_monotonic() {
        let bass = run_orb(&loud_frame(0.9, 0.0), 1.0 / 60.0, 3.0);
        let treble = run_orb(&loud_frame(0.0, 0.9), 1.0 / 60.0, 3.0);
        assert!(
            bass.orb_band[0] > treble.orb_band[0] + 0.3,
            "{:?}",
            bass.orb_band
        );
        assert!(
            treble.orb_band[2] > bass.orb_band[2] + 0.3,
            "{:?}",
            treble.orb_band
        );
        let mut prev = 0.0;
        for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 5.0] {
            let p = run_orb(&loud_frame(level, 0.0), 1.0 / 60.0, 3.0);
            assert!(
                p.orb_band.iter().all(|b| (0.0..=1.0).contains(b)),
                "{:?}",
                p.orb_band
            );
            assert!(
                p.orb_band[0] >= prev - 1e-6,
                "bass not monotonic at {level}"
            );
            prev = p.orb_band[0];
        }
    }

    #[test]
    fn orb_values_are_finite_for_nan_and_inf_input() {
        let mut f = loud_frame(f32::NAN, f32::INFINITY);
        f.energy = f32::NAN;
        f.rms_db = f32::NAN;
        f.group_slow = [f32::INFINITY; mg_dsp::GROUPS];
        let mut m = SceneMapper::new(MapTuning::default());
        for dt in [f32::NAN, f32::INFINITY, 0.0, 1.0 / 60.0] {
            let p = m.update(
                &Input {
                    onsets: 1,
                    onset_strength: f32::NAN,
                    ..live(&f)
                },
                dt,
            );
            for v in p
                .orb_band
                .iter()
                .chain(&p.orb_flow)
                .chain(p.rings.iter().flatten())
            {
                assert!(v.is_finite(), "{v}");
            }
            assert!(p.orb_band.iter().all(|b| (0.0..=1.0).contains(b)));
        }
    }

    #[test]
    fn orb_bands_are_frame_rate_independent() {
        let f = loud_frame(0.8, 0.5);
        let (a, b) = (run_orb(&f, 1.0 / 30.0, 2.0), run_orb(&f, 1.0 / 144.0, 2.0));
        for i in 0..4 {
            assert!(
                (a.orb_band[i] - b.orb_band[i]).abs() < 0.03,
                "{i}: {:?} {:?}",
                a.orb_band,
                b.orb_band
            );
        }
        for i in 0..2 {
            assert!(
                (a.orb_flow[i] - b.orb_flow[i]).abs() < 0.03,
                "{:?} {:?}",
                a.orb_flow,
                b.orb_flow
            );
        }
    }

    #[test]
    fn orb_bass_attacks_fast_and_releases_slowly() {
        let mut m = SceneMapper::new(MapTuning::default());
        let loud = loud_frame(1.0, 0.0);
        for _ in 0..6 {
            m.update(&live(&loud), 0.01); // 60 ms = one attack time constant
        }
        let up = m.update(&live(&loud), 0.0).orb_band[0];
        assert!(up > 0.4, "attack: {up}");
        for _ in 0..100 {
            m.update(&live(&loud), 0.01);
        }
        let quiet = loud_frame(0.0, 0.0);
        for _ in 0..70 {
            m.update(&live(&quiet), 0.01); // 700 ms = one release time constant
        }
        let down = m.update(&live(&quiet), 0.0).orb_band[0];
        assert!(down > 0.25 && down < 0.5, "release after 700 ms: {down}");
    }

    #[test]
    fn silence_has_no_ripples_and_a_slow_dim_orb() {
        let mut m = SceneMapper::new(MapTuning::default());
        let silent = AudioFrame::default();
        let mut p = m.update(&live(&silent), 1.0 / 60.0);
        let mut prev = p.orb_flow;
        for _ in 0..600 {
            p = m.update(
                &Input {
                    onsets: 1,
                    onset_strength: 1.0,
                    ..live(&silent)
                },
                1.0 / 60.0,
            );
            assert!(p.rings.iter().all(|r| r[1] == 0.0), "ripple in silence");
            assert!(p.orb_band.iter().all(|&b| b < 1e-3), "{:?}", p.orb_band);
            assert!(
                (p.orb_flow[0] - prev[0]) * 60.0 <= 0.0125,
                "flow speed in silence"
            );
            prev = p.orb_flow;
        }
    }

    #[test]
    fn calm_music_caps_motion_and_flow_never_goes_backwards() {
        let mut m = SceneMapper::new(MapTuning::default());
        let mut f = loud_frame(0.1, 0.1);
        f.group_medium = [0.1, 0.1, 1.0, 1.0, 0.1, 0.1]; // strong mids but low energy
        f.energy = 0.05;
        let mut prev = [0.0f32; 4];
        for i in 0..1200 {
            let p = m.update(&live(&f), 1.0 / 60.0);
            if i > 300 {
                assert!(
                    (p.orb_flow[0] - prev[0]) * 60.0 < 0.08,
                    "calm flow {}",
                    (p.orb_flow[0] - prev[0]) * 60.0
                );
                assert!(p.orb_flow[0] >= prev[0] && p.orb_flow[1] >= prev[1]);
            }
            prev = p.orb_flow;
        }
    }

    #[test]
    fn every_onset_starts_a_ripple_from_a_new_impact_point_and_at_most_two_live() {
        let mut m = SceneMapper::new(MapTuning::default());
        let f = loud_frame(0.8, 0.5);
        let mut origins = Vec::new();
        for k in 0..4 {
            for _ in 0..30 {
                m.update(&live(&f), 1.0 / 60.0);
            }
            let p = m.update(
                &Input {
                    onsets: 1,
                    onset_strength: 0.8,
                    ..live(&f)
                },
                1.0 / 60.0,
            );
            assert!(p.rings[0][1] > 0.0 && p.rings[0][0] < 0.05, "onset {k}");
            assert!((0.0..1.0).contains(&p.orb_flow[2]), "onset {k}");
            origins.push(p.orb_flow[2]);
        }
        origins.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        assert_eq!(origins.len(), 4, "every onset moves the impact point");
    }
}

/// Fluid mapper checks (CPU only): real analyzer + mapper on synthetic fixtures, or hand-made features.
#[cfg(test)]
pub(crate) mod fluid_tests {
    use super::*;
    use crate::sim::{Sim, Tick};
    use mg_dsp::synth;
    use mg_gfx::fluid::FluidParams;

    pub(crate) const SR: u32 = 48_000;

    fn n(secs: f32) -> usize {
        (secs * SR as f32) as usize
    }

    pub(crate) fn kick(secs: f32) -> Vec<f32> {
        synth::kick_train(120.0, secs, 0.8, SR).0
    }

    /// The first `secs` of a built-in fixture.
    pub(crate) fn fx(name: &str, secs: f32) -> Vec<f32> {
        let mut x = crate::sim::fixture(name, SR).expect("fixture");
        assert!(x.len() >= n(secs), "{name} is shorter than {secs} s");
        x.truncate(n(secs));
        x
    }

    /// Hi-hats: short noise bursts every 0.25 s at the demo fixture's hat level (broadband, treble-heavy).
    pub(crate) fn hats(secs: f32) -> Vec<f32> {
        fx("hats", secs)
    }

    /// Steady three-note pad with a 1.5 s fade-in (no attack transient).
    pub(crate) fn pad(secs: f32) -> Vec<f32> {
        fx("pad", secs)
    }

    pub(crate) fn run(x: &[f32], fps: f32) -> Vec<Tick> {
        Sim::new(SR, fps, MapTuning::default()).run(x)
    }

    /// Times at which a vortex slot restarted (age back near 0 with an amplitude).
    fn spawn_times(t: &[Tick]) -> Vec<f32> {
        t.windows(2)
            .filter(|w| {
                (0..2).any(|s| {
                    let (a, b) = (w[0].params.fluid.vort[s], w[1].params.fluid.vort[s]);
                    b[2] != 0.0 && b[3] < a[3]
                })
            })
            .map(|w| w[1].t)
            .collect()
    }

    fn in_safe_zone(f: &FluidParams) -> bool {
        let inside = |x: f32, y: f32| {
            (FLUID_SAFE_X.0..=FLUID_SAFE_X.1).contains(&x)
                && (FLUID_SAFE_Y.0..=FLUID_SAFE_Y.1).contains(&y)
        };
        f.emit
            .iter()
            .all(|e| inside(e[0], e[1]) && e[1] + e[3] < 0.62)
            && f.vort.iter().all(|v| inside(v[0], v[1]))
    }

    fn all_finite(f: &FluidParams) -> bool {
        f.emit
            .iter()
            .flatten()
            .chain(&f.swirl)
            .chain(f.vort.iter().flatten())
            .chain(&f.flow)
            .chain(&f.look)
            .chain(&f.misc)
            .all(|v| v.is_finite())
    }

    #[test]
    fn fluid_kick_drives_the_bass_emitter_within_two_seconds() {
        let t = run(&kick(2.0), 60.0);
        let best = t
            .iter()
            .map(|k| k.params.fluid.emit[0][2])
            .fold(0.0, f32::max);
        assert!(best > 0.6, "bass emitter strength {best}");
        assert!(t.iter().all(|k| in_safe_zone(&k.params.fluid)));
    }

    /// Means of the mapper outputs over the last 4 s of 8 s of a fixture, plus the vortex spawns in that window.
    struct Char {
        tau: f32,
        detail: f32,
        lift: f32,
        jet_w: f32,
        freq: f32,
        spread: f32,
        bass_emit: f32,
        spawns: usize,
    }

    fn character(name: &str) -> Char {
        // the looping 9 kHz fixture is 4 s long; the same tone for 8 s
        let x = match name {
            "treble" => synth::sine(9000.0, 0.4, SR, n(8.0)),
            _ => fx(name, 8.0),
        };
        let t = run(&x, 60.0);
        let tail = &t[t.len() - 240..];
        let avg = |f: &dyn Fn(&FluidParams) -> f32| {
            tail.iter().map(|k| f(&k.params.fluid)).sum::<f32>() / tail.len() as f32
        };
        Char {
            tau: avg(&|f| f.look[0]),
            detail: avg(&|f| f.misc[2]),
            lift: avg(&|f| f.swirl[3]),
            jet_w: avg(&|f| f.look[3]),
            freq: avg(&|f| f.flow[1]),
            spread: avg(&|f| f.misc[3]),
            bass_emit: avg(&|f| f.emit[0][2]),
            spawns: spawn_times(tail).len(),
        }
    }

    /// FL1: kick, pad, sub-bass drone, hats and a 9 kHz tone move differently because their spectral and temporal
    /// features differ (crest, onsets, low end, centroid), not because of noise or a louder global intensity.
    #[test]
    fn fluid_motion_character_follows_the_music_features() {
        let [kick, pad, drone, hats, treble] =
            ["kick", "pad", "drone", "hats", "treble"].map(character);
        for (n, c) in [
            ("kick", &kick),
            ("pad", &pad),
            ("drone", &drone),
            ("hats", &hats),
            ("treble", &treble),
        ] {
            eprintln!(
                "FLUID char {n:6} tau={:.2} detail={:.2} lift={:.3} jet_w={:.3} freq={:.2} spread={:.2} bass_emit={:.2} spawns={}",
                c.tau, c.detail, c.lift, c.jet_w, c.freq, c.spread, c.bass_emit, c.spawns
            );
        }
        // impulsive (kick, hats): vortices, churning fine curl, short dye memory
        assert!(kick.spawns >= 4 && hats.spawns >= 4, "onsets spin vortices");
        assert!(pad.spawns == 0 && drone.spawns == 0 && treble.spawns == 0);
        for (c, s) in [(&kick, "kick"), (&hats, "hats")] {
            assert!(c.detail > 0.6, "{s} churns: detail {}", c.detail);
            assert!(c.tau < 5.0, "{s} forgets fast: tau {}", c.tau);
        }
        // sustained (pad, drone, tone): laminar, long memory
        for (c, s) in [(&pad, "pad"), (&drone, "drone"), (&treble, "treble")] {
            assert!(c.detail < 0.35, "{s} is laminar: detail {}", c.detail);
            assert!(
                c.tau > kick.tau + 1.5,
                "{s} remembers: tau {} vs kick {}",
                c.tau,
                kick.tau
            );
        }
        // a steady low end lifts a broad, large-scale plume; kick impacts are narrow jets; a pad lifts little
        assert!(
            drone.lift > 0.06 && drone.jet_w > 0.25,
            "drone plume {} {}",
            drone.lift,
            drone.jet_w
        );
        assert!(
            kick.jet_w < drone.jet_w - 0.05,
            "kick jet {} vs drone {}",
            kick.jet_w,
            drone.jet_w
        );
        assert!(
            pad.lift < 0.5 * drone.lift,
            "pad lift {} vs drone {}",
            pad.lift,
            drone.lift
        );
        // the pad's 262 Hz spill does not drive the bass emitter like a real low end does
        assert!(
            pad.bass_emit < 0.5 * drone.bass_emit,
            "pad bass {} vs drone {}",
            pad.bass_emit,
            drone.bass_emit
        );
        // structure scale follows the centroid: drone largest, pad finer, highs finest
        assert!(drone.freq + 0.3 < pad.freq && pad.freq + 0.5 < treble.freq.min(hats.freq));
        // a sustained high tone spreads into a wide veil; hats stay compact
        assert!(
            treble.spread > 0.8 && hats.spread < 0.5 * treble.spread,
            "spread {} {}",
            treble.spread,
            hats.spread
        );
    }

    #[test]
    fn fluid_pad_gives_no_impulses_and_slow_flow() {
        let t = run(&pad(12.0), 60.0);
        assert_eq!(spawn_times(&t), Vec::<f32>::new(), "a pad has no onsets");
        let last = &t.last().unwrap().params.fluid;
        // sustained sound flows at most half as fast as a kick pattern of similar loudness
        let kick_speed = run(&kick(12.0), 60.0).last().unwrap().params.fluid.flow[0];
        assert!(
            last.flow[0] < 0.5 * kick_speed,
            "pad flow speed {} vs kick {kick_speed}",
            last.flow[0]
        );
        assert!(last.vort.iter().all(|v| v[2] == 0.0));
        assert!(
            last.emit.iter().any(|e| e[2] > 0.2),
            "a pad still paints: {:?}",
            last.emit
        );
    }

    #[test]
    fn fluid_silence_stops_injection_within_1_2_s_and_keeps_a_speed_floor() {
        let mut x = kick(4.0);
        let stop = x.len() as f32 / SR as f32;
        x.extend(vec![0.0; n(6.0)]);
        let t = run(&x, 60.0);
        assert!(
            t.iter()
                .filter(|k| k.t < stop)
                .any(|k| k.params.fluid.emit[0][2] > 0.3)
        );
        for k in t.iter().filter(|k| k.t >= stop + 1.2) {
            let f = &k.params.fluid;
            assert!(
                f.emit.iter().all(|e| e[2] == 0.0) && f.swirl[3] == 0.0 && f.look[1] == 0.0,
                "t={} still injecting {:?}",
                k.t,
                f.emit
            );
        }
        assert!(
            t.iter()
                .all(|k| k.params.fluid.flow[0] >= FLUID_SPEED_FLOOR - 1e-6)
        );
        let last = &t.last().unwrap().params.fluid;
        assert!(last.flow[0] < 0.012, "silent flow {}", last.flow[0]);
        assert!(last.look[0] < 3.2, "silent memory tau {}", last.look[0]);
        assert!(
            spawn_times(&t).iter().all(|&s| s < stop + 0.4),
            "no impulse in silence"
        );
    }

    fn live_frame() -> AudioFrame {
        AudioFrame {
            silent: false,
            band_count: mg_dsp::MAX_BANDS,
            rms_db: -12.0,
            group_medium: [0.5; mg_dsp::GROUPS],
            energy: 0.7,
            ..AudioFrame::default()
        }
    }

    /// Steady features for `secs` at `fps` (an onset every `onset_every` frames, 0 = none): every frame's params.
    fn steady(m: &MusicFeatures, secs: f32, fps: f32, onset_every: usize) -> Vec<FluidParams> {
        let mut map = SceneMapper::new(MapTuning::default());
        let f = live_frame();
        (0..(secs * fps) as usize)
            .map(|i| {
                map.observe_music(m, true);
                let onsets = u32::from(onset_every > 0 && i % onset_every == 0);
                map.update(
                    &Input {
                        frame: &f,
                        onsets,
                        onset_strength: 0.8,
                        fresh: true,
                        mode: SceneMode::Live,
                    },
                    1.0 / fps,
                )
                .fluid
            })
            .collect()
    }

    #[test]
    fn calm_music_raises_the_fluid_memory() {
        let base = MusicFeatures {
            level: 0.8,
            long: [0.6; mg_dsp::GROUPS],
            ..MusicFeatures::default()
        };
        let calm = steady(&MusicFeatures { calm: 1.0, ..base }, 8.0, 60.0, 0);
        let busy = steady(&MusicFeatures { calm: 0.0, ..base }, 8.0, 60.0, 0);
        let (tc, tb) = (calm.last().unwrap().look[0], busy.last().unwrap().look[0]);
        assert!(tc > tb + 3.0, "calm tau {tc} vs busy {tb}");
        assert!((3.5..=9.0).contains(&tc) && (3.4..=9.0).contains(&tb));
        let (sc, sb) = (calm.last().unwrap().flow[0], busy.last().unwrap().flow[0]);
        assert!(sc < sb, "calm flows slower: {sc} vs {sb}");
    }

    #[test]
    fn fluid_vortices_are_spaced_and_at_most_two_live() {
        let m = MusicFeatures {
            level: 0.9,
            calm: 0.0,
            long: [0.7; mg_dsp::GROUPS],
            short: [1.0; mg_dsp::GROUPS],
            accent: [0.6; mg_dsp::GROUPS],
            ..MusicFeatures::default()
        };
        let ps = steady(&m, 10.0, 144.0, 1); // an onset on every frame
        let mut spawns = Vec::new();
        let mut signs = Vec::new();
        for (i, w) in ps.windows(2).enumerate() {
            for s in 0..2 {
                if w[1].vort[s][2] != 0.0 && w[1].vort[s][3] < w[0].vort[s][3] {
                    spawns.push((i + 1) as f32 / 144.0);
                    signs.push(w[1].vort[s][2].signum());
                }
            }
            assert!(w[1].vort.iter().filter(|v| v[2] != 0.0).count() <= 2);
        }
        assert!(spawns.len() >= 30, "spawns {}", spawns.len());
        assert!(
            spawns
                .windows(2)
                .all(|s| s[1] - s[0] >= FLUID_VORTEX_GAP_S - 1e-3),
            "{spawns:?}"
        );
        // consecutive vortices spin in opposite directions (no twinning) and are capped
        assert!(signs.windows(2).all(|s| s[0] != s[1]));
        assert!(ps.iter().flat_map(|p| p.vort).all(|v| v[2].abs() <= 0.3));
    }

    #[test]
    fn fluid_emitters_and_vortices_stay_in_the_safe_zone() {
        let mut map = SceneMapper::new(MapTuning::default());
        let f = live_frame();
        for i in 0..(30 * 60 * 30) {
            let t = i as f32 / 30.0;
            let s = |k: f32| 0.5 + 0.5 * (t * k).sin();
            let m = MusicFeatures {
                level: s(0.1),
                calm: s(0.07),
                build: s(0.3),
                centroid: s(0.05),
                long: [s(0.2); mg_dsp::GROUPS],
                short: [s(3.0); mg_dsp::GROUPS],
                accent: [s(5.0); mg_dsp::GROUPS],
                ..MusicFeatures::default()
            };
            map.observe_music(&m, true);
            let p = map.update(
                &Input {
                    frame: &f,
                    onsets: u32::from(i % 7 == 0),
                    onset_strength: 1.0,
                    fresh: true,
                    mode: SceneMode::Live,
                },
                1.0 / 30.0,
            );
            assert!(in_safe_zone(&p.fluid), "t={t}: {:?}", p.fluid);
        }
    }

    #[test]
    fn fluid_outputs_are_finite_and_bounded_for_hostile_input() {
        let m = MusicFeatures {
            agc: [f32::NAN; mg_dsp::GROUPS],
            short: [f32::INFINITY; mg_dsp::GROUPS],
            long: [f32::NEG_INFINITY; mg_dsp::GROUPS],
            accent: [f32::NAN; mg_dsp::GROUPS],
            level: f32::INFINITY,
            centroid: f32::NAN,
            onset_event: true,
            onset_density: f32::NAN,
            crest_db: f32::INFINITY,
            calm: f32::NAN,
            build: 1e30,
        };
        let f = AudioFrame {
            rms_db: f32::NAN,
            energy: f32::INFINITY,
            ..live_frame()
        };
        let mut map = SceneMapper::new(MapTuning::default());
        for k in 0..2000 {
            map.observe_music(&m, k % 3 != 0);
            let dt = [f32::NAN, f32::INFINITY, -1.0, 0.0, 1e-9, 0.016, 5.0][k % 7];
            let p = map
                .update(
                    &Input {
                        frame: &f,
                        onsets: u32::MAX,
                        onset_strength: f32::NAN,
                        fresh: k % 5 != 0,
                        mode: SceneMode::Live,
                    },
                    dt,
                )
                .fluid;
            assert!(all_finite(&p), "{p:?}");
            assert!(in_safe_zone(&p), "{p:?}");
            assert!(p.emit.iter().all(|e| (0.0..=1.3).contains(&e[2])));
            assert!((FLUID_SPEED_FLOOR - 1e-6..=0.5).contains(&p.flow[0]));
            assert!((2.4..=4.0).contains(&p.flow[1]) && (0.0..256.0).contains(&p.flow[2]));
            assert!((3.0 - 1e-3..=9.0).contains(&p.look[0]) && (0.5..=1.5).contains(&p.look[2]));
        }
    }
}

/// Fluid on the GPU (headless; skipped without a hardware device): real analyzer + mapper on synthetic fixtures,
/// the simulation stepped exactly as the live loop steps it (fixed 1/60 s steps from the scene time).
#[cfg(test)]
mod fluid_gpu_tests {
    use super::fluid_tests::{SR, hats, kick, pad, run};
    use super::*;
    use crate::sim::Tick;
    use mg_gfx::context::{Headless, init_headless};
    use mg_gfx::scene::{SceneKind, SceneSet};

    const RES: (u32, u32) = (320, 180);
    /// uv height below which the artwork, lyrics and control bar live.
    const SPLIT: f32 = 0.62;

    fn device() -> Option<Headless> {
        init_headless()
            .map_err(|e| eprintln!("SKIPPED (no hardware GPU in this environment): {e}"))
            .ok()
    }

    fn scene(h: &Headless, tier: u8) -> SceneSet {
        let mut s = SceneSet::new_headless(&h.device);
        s.set_fluid_tier(tier);
        s
    }

    /// Dye energy (sum of the three dye channels) above and below uv y = 0.62.
    fn energy(s: &SceneSet, h: &Headless) -> (f64, f64) {
        let (w, hh, d) = s.fluid_dye(&h.device, &h.queue).expect("dye readback");
        let (mut up, mut lo) = (0.0, 0.0);
        for y in 0..hh {
            let upper = (y as f32 + 0.5) / (hh as f32) < SPLIT;
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let e = f64::from(d[i] + d[i + 1] + d[i + 2]);
                if upper {
                    up += e;
                } else {
                    lo += e;
                }
            }
        }
        (up, lo)
    }

    /// Steps the sim through every tick as the live loop does while Fluid is shown; `each` sees every tick.
    fn drive(
        s: &mut SceneSet,
        h: &Headless,
        ticks: &[Tick],
        mut each: impl FnMut(&SceneSet, &Tick),
    ) {
        for t in ticks {
            s.fluid_advance(&h.device, &h.queue, RES, &t.params);
            each(s, t);
        }
    }

    fn render(s: &SceneSet, h: &Headless, t: &Tick) -> Vec<u8> {
        s.render_offscreen_scenes(
            &h.device,
            &h.queue,
            RES,
            &t.params,
            SceneKind::Fluid,
            None,
            &|_, _| {},
        )
        .expect("render")
    }

    #[test]
    fn fluid_kicks_paint_more_than_hats_and_stay_above_the_lyrics_band_headless() {
        let Some(h) = device() else { return };
        for tier in [0u8, 1] {
            let mut out = Vec::new();
            for x in [kick(6.0), hats(6.0)] {
                let mut s = scene(&h, tier);
                let mut worst_low = 0.0f64;
                drive(&mut s, &h, &run(&x, 60.0), |s, t| {
                    if t.t > 1.0 && ((t.t * 60.0).round() as usize).is_multiple_of(30) {
                        let (u, l) = energy(s, &h);
                        if u + l > 1e-3 {
                            worst_low = worst_low.max(l / (u + l));
                        }
                    }
                });
                out.push((energy(&s, &h).0, worst_low));
            }
            eprintln!(
                "FLUID T{tier} upper energy kick={:.1} hats={:.1}, worst share below 0.62: kick={:.3} hats={:.3}",
                out[0].0, out[1].0, out[0].1, out[1].1
            );
            assert!(
                out[0].0 > 1.5 * out[1].0,
                "T{tier}: kick {:.1} vs hats {:.1}",
                out[0].0,
                out[1].0
            );
            assert!(
                out.iter().all(|o| o.1 < 0.25),
                "T{tier}: dye below the lyrics line {out:?}"
            );
        }
    }

    #[test]
    fn fluid_fades_after_the_music_stops_headless() {
        let Some(h) = device() else { return };
        let mut x = kick(6.0);
        let stop = x.len() as f32 / SR as f32;
        x.extend(vec![0.0; SR as usize * 9]);
        let mut s = scene(&h, 1);
        let mut at = [None; 3];
        drive(&mut s, &h, &run(&x, 60.0), |s, t| {
            for (k, dt) in [0.0f32, 2.0, 8.0].into_iter().enumerate() {
                if at[k].is_none() && t.t >= stop + dt {
                    let (u, l) = energy(s, &h);
                    at[k] = Some(u + l);
                }
            }
        });
        let [Some(e0), Some(e2), Some(e8)] = at else {
            panic!("probes missing {at:?}")
        };
        eprintln!(
            "FLUID fade: stop={e0:.1} +2s={:.3} +8s={:.3}",
            e2 / e0,
            e8 / e0
        );
        assert!(e0 > 1.0, "kicks painted something: {e0}");
        assert!(e2 >= 0.4 * e0, "+2 s keeps {:.3}", e2 / e0);
        assert!(e8 <= 0.12 * e0, "+8 s keeps {:.3}", e8 / e0);
    }

    /// Per-pixel luminance 0..1.
    fn lum(img: &[u8]) -> Vec<f32> {
        img.chunks_exact(4)
            .map(|p| (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2])) / 765.0)
            .collect()
    }

    /// 32x18 box-filtered RGB thumbnail (0..1) of a 320x180 frame.
    fn thumb(img: &[u8]) -> Vec<f32> {
        let mut t = vec![0.0f32; 32 * 18 * 3];
        for y in 0..180usize {
            for x in 0..320usize {
                for c in 0..3 {
                    t[((y / 10) * 32 + x / 10) * 3 + c] +=
                        f32::from(img[(y * 320 + x) * 4 + c]) / 255.0 / 100.0;
                }
            }
        }
        t
    }

    #[test]
    fn fluid_frames_never_flash_and_fixtures_look_different_headless() {
        let Some(h) = device() else { return };
        let sine = |f: f32, a: f32| mg_dsp::synth::sine(f, a, SR, SR as usize * 8);
        let demo = crate::sim::fixture("demo", SR).expect("demo");
        let fixtures: [(&str, Vec<f32>); 7] = [
            ("silence", vec![0.0; SR as usize * 8]),
            ("kick", kick(8.0)),
            ("hats", hats(8.0)),
            ("pad", pad(8.0)),
            ("bass", sine(70.0, 0.5)),
            ("treble", sine(9000.0, 0.4)),
            ("demo", demo[..SR as usize * 8].to_vec()),
        ];
        let mut finals: Vec<(&str, Vec<f32>, f32)> = Vec::new();
        for (name, x) in &fixtures {
            let mut s = scene(&h, 1);
            let (mut worst_mean, mut worst_p95) = (0.0f32, 0.0f32);
            let mut prev: Option<Vec<f32>> = None;
            let mut last = Vec::new();
            drive(&mut s, &h, &run(x, 60.0), |s, t| {
                let img = render(s, &h, t);
                let l = lum(&img);
                if let Some(p) = &prev {
                    let mut d: Vec<f32> = l.iter().zip(p).map(|(a, b)| (a - b).abs()).collect();
                    let mean_d =
                        (l.iter().sum::<f32>() - p.iter().sum::<f32>()).abs() / l.len() as f32;
                    worst_mean = worst_mean.max(mean_d);
                    let k = d.len() * 95 / 100;
                    d.select_nth_unstable_by(k, f32::total_cmp);
                    worst_p95 = worst_p95.max(d[k]);
                }
                prev = Some(l);
                last = img;
            });
            eprintln!(
                "FLUID {name:8} worst mean-lum delta/frame={worst_mean:.4} worst p95 pixel delta={worst_p95:.4}"
            );
            assert!(
                worst_mean <= 0.02,
                "{name}: mean luminance jumped {worst_mean}"
            );
            assert!(worst_p95 <= 0.06, "{name}: p95 pixel delta {worst_p95}");
            assert!(last.chunks_exact(4).all(|p| p[3] == 255));
            if let Ok(dir) = std::env::var("MG_SCENE_DUMP") {
                let _ = std::fs::write(
                    format!("{dir}/fluid-test-{name}.png"),
                    mg_media::synth::png_rgba(RES.0, RES.1, &last),
                );
            }
            // mean luminance above the lyrics line (rows with uv y < 0.62)
            let rows = (SPLIT * RES.1 as f32) as usize;
            let upper = lum(&last[..rows * RES.0 as usize * 4]).iter().sum::<f32>()
                / (rows * RES.0 as usize) as f32;
            finals.push((*name, thumb(&last), upper));
        }
        // FL2: a 9 kHz tone is genuinely visible above the lyrics line, not only by the thumbnail distance
        let up = |n: &str| finals.iter().find(|f| f.0 == n).expect("fixture").2;
        let (u_sil, u_tre) = (up("silence"), up("treble"));
        eprintln!("FLUID upper-region luminance silence={u_sil:.4} treble={u_tre:.4}");
        assert!(
            u_tre >= u_sil + 0.02,
            "treble barely visible: {u_tre:.4} vs silence {u_sil:.4}"
        );
        let mut alike = Vec::new();
        for a in 0..finals.len() {
            for b in a + 1..finals.len() {
                let (x, y) = (&finals[a].1, &finals[b].1);
                let d = x.iter().zip(y).map(|(p, q)| (p - q).abs()).sum::<f32>() / x.len() as f32;
                eprintln!("FLUID thumb {} vs {}: {d:.4}", finals[a].0, finals[b].0);
                if d <= 0.03 {
                    alike.push(format!("{} and {} ({d:.4})", finals[a].0, finals[b].0));
                }
            }
        }
        assert!(alike.is_empty(), "fixtures look alike: {alike:?}");
    }

    #[test]
    fn fluid_is_deterministic_and_frame_rate_independent_headless() {
        let Some(h) = device() else { return };
        let demo = crate::sim::fixture("demo", SR).expect("demo");
        let ticks = run(&demo[..SR as usize * 4], 60.0);
        let dye = |ticks: &[Tick]| {
            let mut s = scene(&h, 1);
            drive(&mut s, &h, ticks, |_, _| {});
            s.fluid_dye(&h.device, &h.queue).expect("dye").2
        };
        let (a, b) = (dye(&ticks), dye(&ticks));
        assert!(
            a == b,
            "two runs of the same input must give the same field"
        );
        // the same 10 s of kicks shown at 30, 60 and 144 Hz end with the same amount of dye (within 5%)
        let x = kick(10.0);
        let mut e = Vec::new();
        for fps in [30.0f32, 60.0, 144.0] {
            let mut s = scene(&h, 1);
            drive(&mut s, &h, &run(&x, fps), |_, _| {});
            let (u, l) = energy(&s, &h);
            e.push(u + l);
        }
        eprintln!(
            "FLUID energy at 10 s: 30 Hz {:.1}, 60 Hz {:.1}, 144 Hz {:.1}",
            e[0], e[1], e[2]
        );
        for (i, fps) in [(0, 30), (2, 144)] {
            assert!(
                (e[i] - e[1]).abs() <= 0.05 * e[1],
                "{fps} Hz {:.1} vs 60 Hz {:.1}",
                e[i],
                e[1]
            );
        }
    }

    /// Dye field shape after 8 s of a fixture from a clear field: (energy, fine structure = sum |grad| / energy in
    /// 1/texel, horizontal spread = energy-weighted std of x in widths, mean height in uv y).
    fn shape(h: &Headless, x: &[f32]) -> (f64, f64, f64, f64) {
        let mut s = scene(h, 1);
        drive(&mut s, h, &run(x, 60.0), |_, _| {});
        let (w, hh, d) = s.fluid_dye(&h.device, &h.queue).expect("dye");
        let e = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            f64::from(d[i] + d[i + 1] + d[i + 2])
        };
        let (mut sum, mut grad, mut mx, mut mx2, mut my) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for y in 0..hh - 1 {
            for x in 0..w - 1 {
                let v = e(x, y);
                let (fx, fy) = (f64::from(x) / f64::from(w), f64::from(y) / f64::from(hh));
                sum += v;
                grad += (e(x + 1, y) - v).abs() + (e(x, y + 1) - v).abs();
                mx += v * fx;
                mx2 += v * fx * fx;
                my += v * fy;
            }
        }
        let n = sum.max(1e-9);
        (
            sum,
            grad / n,
            (mx2 / n - (mx / n).powi(2)).max(0.0).sqrt(),
            my / n,
        )
    }

    /// FL1 / FL2 on the GPU: the fixtures leave differently SHAPED fields (not just more or less dye): a kick
    /// churns finer structure than a sustained drone, the drone's plume rises higher than the kick's strikes, and
    /// a 9 kHz tone spreads a wide veil where silence leaves nothing.
    #[test]
    fn fluid_fixtures_leave_differently_shaped_fields_headless() {
        let Some(h) = device() else { return };
        let sine = |f: f32, a: f32| mg_dsp::synth::sine(f, a, SR, SR as usize * 8);
        let [silence, kick, pad, drone, treble] = [
            vec![0.0; SR as usize * 8],
            kick(8.0),
            pad(8.0),
            super::fluid_tests::fx("drone", 8.0),
            sine(9000.0, 0.4),
        ]
        .map(|x| shape(&h, &x));
        for (n, s) in [
            ("silence", silence),
            ("kick", kick),
            ("pad", pad),
            ("drone", drone),
            ("treble", treble),
        ] {
            eprintln!(
                "FLUID shape {n:7} energy={:.1} fine={:.4} x_spread={:.3} mean_y={:.3}",
                s.0, s.1, s.2, s.3
            );
        }
        assert_eq!(silence.0, 0.0, "silence paints nothing");
        assert!(treble.0 > 500.0, "a 9 kHz tone paints: {}", treble.0);
        assert!(
            treble.2 > 1.15 * drone.2.max(pad.2),
            "the treble veil is the widest: {} vs drone {} / pad {}",
            treble.2,
            drone.2,
            pad.2
        );
        assert!(
            kick.1 > 1.15 * drone.1,
            "a kick churns finer structure than a drone: {} vs {}",
            kick.1,
            drone.1
        );
        assert!(
            drone.3 + 0.02 < kick.3,
            "the drone plume rides higher than the kick strikes: {} vs {}",
            drone.3,
            kick.3
        );
    }

    /// Ten simulated minutes of extreme, then hostile, input on T0 (no rendering): the field stays finite and bounded.
    #[test]
    fn fluid_ten_minute_extreme_soak_stays_finite_headless() {
        let Some(h) = device() else { return };
        let mut s = scene(&h, 0);
        let mut map = SceneMapper::new(MapTuning::default());
        let loud = MusicFeatures {
            agc: [1.0; mg_dsp::GROUPS],
            short: [1.0; mg_dsp::GROUPS],
            long: [1.0; mg_dsp::GROUPS],
            accent: [1.0; mg_dsp::GROUPS],
            level: 1.0,
            centroid: 1.0,
            onset_event: true,
            onset_density: 10.0,
            crest_db: 40.0,
            calm: 0.0,
            build: 1.0,
        };
        let hostile = MusicFeatures {
            short: [f32::NAN; mg_dsp::GROUPS],
            level: f32::INFINITY,
            centroid: f32::NAN,
            ..loud
        };
        let frame = AudioFrame {
            silent: false,
            band_count: mg_dsp::MAX_BANDS,
            rms_db: -3.0,
            energy: 1.0,
            group_medium: [1.0; mg_dsp::GROUPS],
            ..AudioFrame::default()
        };
        // 20 frames per second, three fixed steps each: 12 000 frames = 600 simulated seconds
        for i in 0..12_000u32 {
            let m = if i < 9_000 { &loud } else { &hostile };
            map.observe_music(m, true);
            let p = map.update(
                &Input {
                    frame: &frame,
                    onsets: u32::from(i % 3 == 0),
                    onset_strength: 1.0,
                    fresh: true,
                    mode: SceneMode::Live,
                },
                0.05,
            );
            s.fluid_step(&h.device, &h.queue, RES, &p, 3);
            if i % 3_000 == 2_999 {
                let (_, _, d) = s.fluid_dye(&h.device, &h.queue).expect("dye");
                let max = d.iter().cloned().fold(0.0f32, f32::max);
                eprintln!("FLUID soak {} s: max texel {max:.3}", (i + 1) / 20);
                assert!(
                    d.iter().all(|v| v.is_finite() && (0.0..=3.0).contains(v)),
                    "at frame {i}"
                );
            }
        }
    }
}
