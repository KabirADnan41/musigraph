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
        }
    }

    /// Records this frame's music features; stale or absent audio gives the rest values (no fake
    /// activity). Scenes read it through `music()` to derive their own uniform fields.
    pub fn observe_music(&mut self, m: &MusicFeatures, fresh: bool) {
        self.music = if fresh { *m } else { MusicFeatures::default() };
    }

    #[cfg_attr(not(test), allow(dead_code))] // first scene consumer lands in a later slice
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
