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
    /// Album Focus rotation phase (slow, grows without wrapping: one hour is only a few hundred units).
    focus_rot: f32,
    onset_count: u32,
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
            focus_rot: 0.0,
            onset_count: 0,
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
            dust: self.dust,
            aurora_phase: self.aurora_ph,
            aurora_band,
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
}
