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
use mg_dsp::smooth::Smoother;
use mg_gfx::scene::{SLOTS, SceneMode, SceneParams};

#[derive(Clone, Debug)]
pub struct MapTuning {
    /// Max bar growth (fraction of full height per second).
    pub rise_per_s: f32,
    /// Bar fall speed (gravity-like).
    pub fall_per_s: f32,
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
    /// Palette hue slew limit (hue units/s; 15 degrees/s = 0.0417).
    pub hue_rate: f32,
    /// A frame older than this (no new sequence number) is treated as absent audio.
    pub stale_after_s: f32,
}

impl Default for MapTuning {
    fn default() -> Self {
        Self {
            rise_per_s: 14.0,
            fall_per_s: 1.5,
            peak_hold_s: 0.25,
            peak_fall_per_s: 0.6,
            gate_lo_db: -55.0,
            gate_hi_db: -25.0,
            gate_floor: 0.25,
            energy_up_s: 3.0,
            energy_down_s: 6.0,
            ring_min_gap_s: 0.12,
            ring_amp: 0.15,
            hue_rate: 0.0417,
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
    bars: [f32; SLOTS],
    peaks: [f32; SLOTS],
    hold: [f32; SLOTS],
    gate: Smoother,
    energy: Smoother,
    swell: Smoother,
    tilt: f32,
    glow: Smoother,
    haze: Smoother,
    hue: [f32; 3],
    rings: [[f32; 2]; 2],
    since_onset: f32,
    /// slot -> (band position 0..1, group index)
    slot_map: [(f32, usize); SLOTS],
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

/// HSV (hue in turns, may exceed 1) to linear RGB.
fn hsv_linear(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(1.0) * 6.0;
    let (i, f) = (h.floor() as i32, h.fract());
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    let c = match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    };
    c.map(|x| x.powf(2.2))
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
            bars: [0.0; SLOTS],
            peaks: [0.0; SLOTS],
            hold: [0.0; SLOTS],
            hue: [0.62, 0.52, 0.50],
            rings: [[100.0, 0.0]; 2],
            since_onset: 10.0,
            slot_map,
        }
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
                tg.max(h - self.tune.fall_per_s * dt)
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

        let bright_t = (-tilt).max(0.0);
        let hue_t = [
            (0.62 + 0.26 * s) * (1.0 - 0.7 * bright_t) + 0.52 * 0.7 * bright_t, // base: indigo -> purple
            0.50 + 0.36 * s, // bars: teal -> violet
            0.52 + 0.65 * s, // accent: azure -> warm amber
        ];
        let max_step = self.tune.hue_rate * dt;
        for (h, t) in self.hue.iter_mut().zip(hue_t) {
            *h += (t - *h).clamp(-max_step, max_step);
        }
        let palette = [
            hsv_linear(
                self.hue[0],
                (0.65 + 0.15 * s) * (1.0 - 0.35 * bright_t),
                0.45 + 0.30 * s,
            ),
            hsv_linear(self.hue[1], 0.60 + 0.20 * s, 0.70 + 0.10 * s),
            hsv_linear(
                self.hue[2],
                (0.35 + 0.40 * s) * (1.0 - 0.4 * bright_t),
                0.95,
            ),
        ];

        // ---- fast: onset impulses (real onsets only; refractory; capped) -------------------------
        for r in &mut self.rings {
            r[0] = (r[0] + dt).min(100.0);
        }
        self.since_onset += dt;
        if live && !silent && input.onsets > 0 && self.since_onset >= self.tune.ring_min_gap_s {
            self.since_onset = 0.0;
            self.rings[1] = self.rings[0];
            let amp = self.tune.ring_amp * fin(input.onset_strength).clamp(0.0, 1.0);
            self.rings[0] = [0.0, amp.min(0.25)];
        }

        SceneParams {
            time: self.t,
            energy: s,
            glow,
            haze,
            warp,
            swell: swell.clamp(0.0, 1.0),
            tilt,
            scroll: self.scroll,
            rings: self.rings,
            palette,
            mode: if live { input.mode } else { SceneMode::NoAudio },
            bars: self.bars,
            peaks: self.peaks,
            bright,
        }
    }
}
