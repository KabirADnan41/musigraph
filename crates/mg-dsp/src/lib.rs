//! Centralised, real-time audio analysis (P2). Pure code: no OS access, no `unsafe`.
//!
//! One `Analyzer` turns captured samples into timestamped `AudioFrame`s that every scene consumes
//! (never one FFT per scene). Features: log-spaced band levels, six named frequency groups at three
//! response speeds (fast/medium/slow), RMS/peak, silence, spectral flux + onset detection, and an
//! adaptive musical-energy estimate. All timing constants are independently designed defaults to be
//! tuned on native hardware (not derived from any reference application). See ADR-0008.
#![forbid(unsafe_code)]

pub mod bands;
pub mod features;
pub mod smooth;
pub mod synth;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use smooth::Smoother;
use std::sync::Arc;

pub const MAX_BANDS: usize = 64;
pub const FFT_SIZE: usize = 2048;
pub const GROUPS: usize = 6;
/// sub-bass, bass, low-mid, mid, upper-mid, treble.
pub const GROUP_NAMES: [&str; GROUPS] =
    ["sub_bass", "bass", "low_mid", "mid", "upper_mid", "treble"];
const GROUP_EDGES_HZ: [f32; GROUPS + 1] = [20.0, 60.0, 250.0, 500.0, 2000.0, 4000.0, 16000.0];
const FLUX_HISTORY: usize = 64;

/// Tunable analysis parameters (defaults are design starting points).
#[derive(Clone, Debug)]
pub struct AnalysisParams {
    pub band_count: usize,
    pub f_min: f32,
    pub f_max: f32,
    /// Samples between analysis frames (512 = ~10.7 ms at 48 kHz).
    pub hop: usize,
    /// RMS below this (dBFS) counts as silence once held for `silence_hold_s`.
    pub silence_db: f32,
    pub silence_hold_s: f32,
    /// Dynamic range mapped to 0..1 below the rolling reference level.
    pub range_db: f32,
    /// The rolling reference never drops below this (limits gain on quiet material).
    pub ref_min_db: f32,
    /// How fast the reference falls when the music gets quieter.
    pub ref_decay_db_per_s: f32,
    pub fast: (f32, f32),
    pub medium: (f32, f32),
    pub slow: (f32, f32),
    /// Onset threshold = mean + k * std of recent spectral flux (plus `onset_min_flux`).
    pub onset_k: f32,
    pub onset_min_flux: f32,
    pub onset_min_gap_s: f32,
    /// Spectral tilt compensation (dB per octave about 1 kHz): music spectra fall roughly 4-6 dB/oct, so
    /// without it every song looks like the same bass hump. 0 disables. Clamped to +-18 dB per band.
    pub tilt_db_per_oct: f32,
}

impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            band_count: 64,
            f_min: 30.0,
            f_max: 16000.0,
            hop: 512,
            silence_db: -70.0,
            silence_hold_s: 0.3,
            range_db: 45.0,
            ref_min_db: -30.0,
            ref_decay_db_per_s: 2.0,
            fast: (5.0, 120.0),
            medium: (40.0, 300.0),
            slow: (400.0, 1500.0),
            onset_k: 4.0,
            onset_min_flux: 0.015,
            onset_min_gap_s: 0.09,
            tilt_db_per_oct: 4.5,
        }
    }
}

/// One timestamped analysis result. Plain data, cheap to copy; values are 0..1 unless noted.
#[derive(Clone, Copy, Debug)]
pub struct AudioFrame {
    pub seq: u64,
    /// Capture time (ns) of the newest sample in the analysis window, same clock as the caller's.
    pub t_capture_ns: u64,
    pub sample_rate: u32,
    pub band_count: usize,
    /// Per-band level, fast response, adaptively normalised.
    pub bands: [f32; MAX_BANDS],
    pub group_fast: [f32; GROUPS],
    pub group_medium: [f32; GROUPS],
    pub group_slow: [f32; GROUPS],
    /// RMS level in dBFS, floored at -120 (digital silence).
    pub rms_db: f32,
    pub peak: f32,
    pub silent: bool,
    /// Spectral flux of the normalised bands (rise only).
    pub flux: f32,
    /// True only on the frame in which an onset was detected.
    pub onset_event: bool,
    /// Decaying pulse that jumps on each onset (fast release).
    pub onset: f32,
    /// Slow adaptive musical energy: loudness, onset density and bass activity.
    pub energy: f32,
}

impl Default for AudioFrame {
    fn default() -> Self {
        Self {
            seq: 0,
            t_capture_ns: 0,
            sample_rate: 48_000,
            band_count: MAX_BANDS,
            bands: [0.0; MAX_BANDS],
            group_fast: [0.0; GROUPS],
            group_medium: [0.0; GROUPS],
            group_slow: [0.0; GROUPS],
            rms_db: -120.0,
            peak: 0.0,
            silent: true,
            flux: 0.0,
            onset_event: false,
            onset: 0.0,
            energy: 0.0,
        }
    }
}

pub struct Analyzer {
    sr: u32,
    p: AnalysisParams,
    window: Vec<f32>,
    win_scale: f32,
    ring: Vec<f32>,
    pos: usize,
    filled: usize,
    since_hop: usize,
    fft: Arc<dyn Fft<f32>>,
    buf: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    mag: Vec<f32>,
    band_bins: Vec<(usize, usize)>,
    group_bins: [(usize, usize); GROUPS],
    band_tilt_db: Vec<f32>,
    group_tilt_db: [f32; GROUPS],
    band_s: Vec<Smoother>,
    gf: [Smoother; GROUPS],
    gm: [Smoother; GROUPS],
    gs: [Smoother; GROUPS],
    cur: [f32; MAX_BANDS],
    prev: [f32; MAX_BANDS],
    ref_db: f32,
    silent_s: f32,
    flux_hist: [f32; FLUX_HISTORY],
    flux_n: usize,
    flux_i: usize,
    since_onset_s: f32,
    onset_s: Smoother,
    density_s: Smoother,
    energy_s: Smoother,
    seq: u64,
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

impl Analyzer {
    /// Creates an analyzer for a validated sample rate (8 kHz to 384 kHz).
    pub fn new(sample_rate: u32, mut p: AnalysisParams) -> Result<Self, String> {
        if !(8_000..=384_000).contains(&sample_rate) {
            return Err(format!("unsupported sample rate {sample_rate}"));
        }
        let d = AnalysisParams::default();
        let ok = |v: f32, lo: f32| if v.is_finite() && v >= lo { v } else { lo };
        // Untrusted/config values must never poison the state: fall back to safe values.
        p.range_db = ok(p.range_db, 6.0);
        p.ref_decay_db_per_s = ok(p.ref_decay_db_per_s, 0.0);
        p.silence_hold_s = ok(p.silence_hold_s, 0.0);
        p.onset_k = ok(p.onset_k, 0.0);
        p.onset_min_flux = ok(p.onset_min_flux, 0.0);
        p.onset_min_gap_s = ok(p.onset_min_gap_s, 0.0);
        p.ref_min_db = if p.ref_min_db.is_finite() {
            p.ref_min_db.min(0.0)
        } else {
            d.ref_min_db
        };
        p.silence_db = if p.silence_db.is_finite() {
            p.silence_db
        } else {
            d.silence_db
        };
        if !(p.f_min.is_finite() && p.f_max.is_finite() && p.f_min >= 1.0 && p.f_max > p.f_min) {
            (p.f_min, p.f_max) = (d.f_min, d.f_max);
        }
        p.tilt_db_per_oct = if p.tilt_db_per_oct.is_finite() {
            p.tilt_db_per_oct.clamp(-12.0, 12.0)
        } else {
            d.tilt_db_per_oct
        };
        p.band_count = p.band_count.clamp(1, MAX_BANDS);
        p.hop = p.hop.clamp(64, FFT_SIZE);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / FFT_SIZE as f32).cos())
            .collect();
        let win_scale = 2.0 / window.iter().sum::<f32>();
        let fft = FftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        let edges = bands::log_edges_hz(p.band_count, p.f_min, p.f_max);
        let band_bins = (0..p.band_count)
            .map(|b| bands::bin_range(edges[b], edges[b + 1], sample_rate, FFT_SIZE))
            .collect();
        let mut group_bins = [(1, 1); GROUPS];
        for (g, gb) in group_bins.iter_mut().enumerate() {
            *gb = bands::bin_range(
                GROUP_EDGES_HZ[g],
                GROUP_EDGES_HZ[g + 1],
                sample_rate,
                FFT_SIZE,
            );
        }
        let tilt = |f: f32| (p.tilt_db_per_oct * (f / 1000.0).log2()).clamp(-18.0, 18.0);
        let band_tilt_db: Vec<f32> = (0..p.band_count)
            .map(|b| tilt((edges[b] * edges[b + 1]).sqrt()))
            .collect();
        let mut group_tilt_db = [0.0; GROUPS];
        for (g, t) in group_tilt_db.iter_mut().enumerate() {
            *t = tilt((GROUP_EDGES_HZ[g] * GROUP_EDGES_HZ[g + 1]).sqrt());
        }
        let sm = |(a, r): (f32, f32)| Smoother::new(a, r);
        Ok(Self {
            sr: sample_rate,
            window,
            win_scale,
            ring: vec![0.0; FFT_SIZE],
            pos: 0,
            filled: 0,
            since_hop: 0,
            fft,
            buf: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            scratch,
            mag: vec![0.0; FFT_SIZE / 2 + 1],
            band_bins,
            group_bins,
            band_tilt_db,
            group_tilt_db,
            band_s: vec![sm(p.fast); MAX_BANDS],
            gf: [sm(p.fast); GROUPS],
            gm: [sm(p.medium); GROUPS],
            gs: [sm(p.slow); GROUPS],
            cur: [0.0; MAX_BANDS],
            prev: [0.0; MAX_BANDS],
            ref_db: p.ref_min_db,
            silent_s: 0.0,
            flux_hist: [0.0; FLUX_HISTORY],
            flux_n: 0,
            flux_i: 0,
            since_onset_s: 10.0,
            onset_s: Smoother::new(1.0, 140.0),
            density_s: Smoother::new(300.0, 2000.0),
            energy_s: Smoother::new(600.0, 2500.0),
            seq: 0,
            p,
        })
    }

    /// A `features::MusicTracker` matched to this analyzer's (clamped) parameters.
    pub fn music_tracker(&self) -> features::MusicTracker {
        features::MusicTracker::new(self.sr, &self.p)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sr
    }

    /// Feeds samples (mono, nominal -1..1; NaN/inf become 0 and values are clamped). `t_end_ns` is the
    /// capture time of the LAST sample in `samples`. Calls `on_frame` for every completed analysis frame.
    pub fn push(&mut self, samples: &[f32], t_end_ns: u64, mut on_frame: impl FnMut(&AudioFrame)) {
        let n = samples.len();
        for (i, &s) in samples.iter().enumerate() {
            let s = if s.is_finite() {
                s.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            self.ring[self.pos] = s;
            self.pos = (self.pos + 1) % FFT_SIZE;
            self.filled = (self.filled + 1).min(FFT_SIZE);
            self.since_hop += 1;
            if self.since_hop >= self.p.hop && self.filled == FFT_SIZE {
                self.since_hop = 0;
                let behind_ns =
                    ((n - 1 - i) as u64).saturating_mul(1_000_000_000) / u64::from(self.sr);
                let frame = self.process(t_end_ns.saturating_sub(behind_ns));
                on_frame(&frame);
            }
        }
    }

    fn norm(&self, level_db: f32) -> f32 {
        ((level_db - (self.ref_db - self.p.range_db)) / self.p.range_db).clamp(0.0, 1.0)
    }

    fn process(&mut self, t_ns: u64) -> AudioFrame {
        let dt = self.p.hop as f32 / self.sr as f32;
        for i in 0..FFT_SIZE {
            self.buf[i] = Complex::new(self.ring[(self.pos + i) % FFT_SIZE] * self.window[i], 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        for (m, c) in self.mag.iter_mut().zip(self.buf.iter()) {
            *m = c.norm() * self.win_scale;
        }
        let rms = (self.ring.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        let rms_db = db(rms).max(-120.0);
        let peak = (0..self.p.hop)
            .map(|k| self.ring[(self.pos + FFT_SIZE - 1 - k) % FFT_SIZE].abs())
            .fold(0.0, f32::max);
        if rms_db < self.p.silence_db {
            self.silent_s += dt;
        } else {
            self.silent_s = 0.0;
        }
        let silent = self.silent_s >= self.p.silence_hold_s;

        // Raw per-band / per-group levels in dBFS.
        let mut band_db = [-120.0f32; MAX_BANDS];
        let mut peak_db = -120.0f32;
        for (b, &(lo, hi)) in self.band_bins.iter().enumerate() {
            let m = self.mag[lo..=hi].iter().copied().fold(0.0, f32::max);
            band_db[b] = db(m) + self.band_tilt_db[b];
            peak_db = peak_db.max(band_db[b]);
        }
        let mut group_db = [-120.0f32; GROUPS];
        for (g, &(lo, hi)) in self.group_bins.iter().enumerate() {
            group_db[g] =
                db(self.mag[lo..=hi].iter().copied().fold(0.0, f32::max)) + self.group_tilt_db[g];
        }
        // Rolling reference: rises at once, falls slowly, never below ref_min_db (limits gain).
        if peak_db > self.ref_db {
            self.ref_db = peak_db.min(0.0);
        } else {
            self.ref_db = (self.ref_db - self.p.ref_decay_db_per_s * dt).max(self.p.ref_min_db);
        }

        self.prev = self.cur;
        for b in 0..self.p.band_count {
            self.cur[b] = if silent { 0.0 } else { self.norm(band_db[b]) };
        }
        // The very first frame has no predecessor: its "rise from nothing" must not enter the statistics.
        let flux = if self.seq == 0 {
            0.0
        } else {
            (0..self.p.band_count)
                .map(|b| (self.cur[b] - self.prev[b]).max(0.0))
                .sum::<f32>()
                / self.p.band_count as f32
        };

        // Onset: flux above an adaptive threshold, with a refractory gap; never during silence.
        self.since_onset_s += dt;
        let n = self.flux_n.max(1) as f32;
        let mean = self.flux_hist[..self.flux_n].iter().sum::<f32>() / n;
        let var = self.flux_hist[..self.flux_n]
            .iter()
            .map(|f| (f - mean) * (f - mean))
            .sum::<f32>()
            / n;
        let thr = mean + self.p.onset_k * var.sqrt() + self.p.onset_min_flux;
        let event = !silent
            && self.flux_n >= 8
            && flux > thr
            && self.since_onset_s >= self.p.onset_min_gap_s;
        self.flux_hist[self.flux_i] = flux;
        self.flux_i = (self.flux_i + 1) % FLUX_HISTORY;
        self.flux_n = (self.flux_n + 1).min(FLUX_HISTORY);
        let strength = if event {
            (0.4 + (flux - thr) * 6.0).clamp(0.4, 1.0)
        } else {
            0.0
        };
        if event {
            self.since_onset_s = 0.0;
        }
        let onset = self.onset_s.step(strength, dt);

        let mut bands = [0.0; MAX_BANDS];
        for (b, v) in bands.iter_mut().enumerate().take(self.p.band_count) {
            *v = self.band_s[b].step(self.cur[b], dt);
        }
        let (mut gf, mut gm, mut gs) = ([0.0; GROUPS], [0.0; GROUPS], [0.0; GROUPS]);
        for g in 0..GROUPS {
            let x = if silent { 0.0 } else { self.norm(group_db[g]) };
            gf[g] = self.gf[g].step(x, dt);
            gm[g] = self.gm[g].step(x, dt);
            gs[g] = self.gs[g].step(x, dt);
        }

        // Adaptive energy: absolute loudness + onset density + bass activity (not volume alone).
        let loud = if silent {
            0.0
        } else {
            ((rms_db + 55.0) / 50.0).clamp(0.0, 1.0)
        };
        let density = self.density_s.step(if event { 1.0 / dt } else { 0.0 }, dt);
        let density_n = (density / 4.0).clamp(0.0, 1.0);
        let energy = self.energy_s.step(
            0.55 * loud + 0.25 * density_n + 0.20 * 0.5 * (gm[0] + gm[1]),
            dt,
        );

        self.seq += 1;
        AudioFrame {
            seq: self.seq,
            t_capture_ns: t_ns,
            sample_rate: self.sr,
            band_count: self.p.band_count,
            bands,
            group_fast: gf,
            group_medium: gm,
            group_slow: gs,
            rms_db,
            peak,
            silent,
            flux,
            onset_event: event,
            onset,
            energy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    const SR: u32 = 48_000;

    /// Runs `samples` through a fresh analyzer in 10 ms chunks and returns every frame.
    fn run(samples: &[f32]) -> Vec<AudioFrame> {
        let mut a = Analyzer::new(SR, AnalysisParams::default()).unwrap();
        let mut frames = Vec::new();
        let mut t = 0u64;
        for chunk in samples.chunks(480) {
            t += chunk.len() as u64 * 1_000_000_000 / u64::from(SR);
            a.push(chunk, t, |f| frames.push(*f));
        }
        frames
    }

    fn last(frames: &[AudioFrame]) -> AudioFrame {
        *frames.last().expect("frames")
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
            .0
    }

    #[test]
    fn rejects_unsupported_sample_rates() {
        assert!(Analyzer::new(4_000, AnalysisParams::default()).is_err());
        assert!(Analyzer::new(500_000, AnalysisParams::default()).is_err());
        assert!(Analyzer::new(44_100, AnalysisParams::default()).is_ok());
    }

    #[test]
    fn tones_land_in_the_expected_groups() {
        let cases = [
            (40.0, 0usize),
            (120.0, 1),
            (350.0, 2),
            (1000.0, 3),
            (3000.0, 4),
            (8000.0, 5),
        ];
        for (f, group) in cases {
            let fr = last(&run(&synth::sine(f, 0.5, SR, SR as usize)));
            assert_eq!(
                argmax(&fr.group_fast),
                group,
                "{f} Hz -> {:?}",
                fr.group_fast
            );
        }
    }

    #[test]
    fn different_frequencies_give_different_band_peaks() {
        let a = last(&run(&synth::sine(200.0, 0.5, SR, SR as usize)));
        let b = last(&run(&synth::sine(5000.0, 0.5, SR, SR as usize)));
        assert!(
            argmax(&b.bands) > argmax(&a.bands) + 20,
            "{} vs {}",
            argmax(&a.bands),
            argmax(&b.bands)
        );
    }

    #[test]
    fn sweep_moves_the_peak_monotonically() {
        let frames = run(&synth::sweep(60.0, 12000.0, 0.5, SR, 6 * SR as usize));
        let peaks: Vec<usize> = frames
            .iter()
            .step_by(20)
            .map(|f| argmax(&f.bands))
            .collect();
        let mut worst_drop = 0;
        for w in peaks.windows(2) {
            worst_drop = worst_drop.max(w[0].saturating_sub(w[1]));
        }
        assert!(worst_drop <= 3, "peak moved backwards by {worst_drop}");
        assert!(peaks.last().unwrap() > &(peaks[2] + 30));
    }

    #[test]
    fn silence_produces_no_activity_and_no_onsets() {
        let frames = run(&vec![0.0; 3 * SR as usize]);
        let f = last(&frames);
        assert!(f.silent);
        assert!(f.bands.iter().all(|&b| b == 0.0));
        assert!(f.energy < 0.01 && f.onset == 0.0);
        assert!(
            frames.iter().all(|f| !f.onset_event),
            "no fake beats in silence"
        );
    }

    #[test]
    fn tone_then_silence_decays_and_music_returns_cleanly() {
        let mut s = synth::sine(440.0, 0.5, SR, 2 * SR as usize);
        s.extend(vec![0.0; 6 * SR as usize]);
        s.extend(synth::sine(440.0, 0.5, SR, SR as usize));
        let frames = run(&s);
        let per_s = SR as usize / 512;
        let during = frames[2 * per_s - 5];
        assert!(during.group_fast.iter().cloned().fold(0.0, f32::max) > 0.8);
        let after = frames[8 * per_s - 5];
        assert!(
            after.silent && after.bands.iter().all(|&b| b < 0.02),
            "bands settle to zero"
        );
        assert!(after.energy < 0.1, "energy relaxes: {}", after.energy);
        let back = last(&frames);
        assert!(
            !back.silent && back.group_fast[2] > 0.8,
            "music resumes: {}",
            back.group_fast[2]
        );
    }

    #[test]
    fn kick_pattern_produces_onsets_that_match_the_kicks() {
        let (x, kicks) = synth::kick_train(120.0, 8.0, 0.8, SR);
        let frames = run(&x);
        let events: Vec<f32> = frames
            .iter()
            .filter(|f| f.onset_event)
            .map(|f| f.t_capture_ns as f32 / 1e9)
            .collect();
        let matched = kicks
            .iter()
            .filter(|&&k| events.iter().any(|&e| e >= k && e <= k + 0.1))
            .count();
        let false_events = events
            .iter()
            .filter(|&&e| !kicks.iter().any(|&k| e >= k && e <= k + 0.1))
            .count();
        assert!(
            matched * 100 >= kicks.len() * 85,
            "matched {matched}/{}: {events:?}",
            kicks.len()
        );
        assert!(false_events <= 2, "false events {false_events}");
    }

    #[test]
    fn steady_tone_makes_no_onsets_after_the_start() {
        let frames = run(&synth::sine(220.0, 0.5, SR, 6 * SR as usize));
        let late = frames.iter().skip(60).filter(|f| f.onset_event).count();
        assert_eq!(late, 0);
    }

    #[test]
    fn loud_has_more_energy_than_quiet_and_sudden_increase_is_seen_by_fast_before_slow() {
        let loud = last(&run(&synth::sine(100.0, 0.8, SR, 4 * SR as usize)));
        let quiet = last(&run(&synth::sine(100.0, 0.02, SR, 4 * SR as usize)));
        assert!(
            loud.energy > quiet.energy + 0.1,
            "{} vs {}",
            loud.energy,
            quiet.energy
        );
        assert!(
            quiet.group_fast[1] > 0.3,
            "quiet music stays visible: {}",
            quiet.group_fast[1]
        );
        let mut s = vec![0.0; SR as usize / 2];
        s.extend(synth::sine(100.0, 0.8, SR, SR as usize / 4));
        let frames = run(&s);
        let f = frames.last().unwrap();
        assert!(f.group_fast[1] > f.group_slow[1], "fast reacts first");
    }

    #[test]
    fn gradual_buildup_raises_energy() {
        let n = 8 * SR as usize;
        let s: Vec<f32> = (0..n)
            .map(|i| 0.02 + 0.7 * i as f32 / n as f32)
            .zip(synth::noise(7, 1.0, n))
            .map(|(a, w)| a * w * 0.7)
            .collect();
        let frames = run(&s);
        let early = frames[frames.len() / 8].energy;
        let late = last(&frames).energy;
        assert!(late > early + 0.15, "{early} -> {late}");
    }

    #[test]
    fn equal_loudness_but_different_spectra_look_different() {
        let sine = synth::sine(200.0, 0.3, SR, 3 * SR as usize);
        let mut noise = synth::noise(3, 1.0, 3 * SR as usize);
        let k = synth::rms(&sine) / synth::rms(&noise);
        noise.iter_mut().for_each(|v| *v *= k);
        let (a, b) = (last(&run(&sine)), last(&run(&noise)));
        assert!((a.rms_db - b.rms_db).abs() < 1.0, "equal loudness");
        let spread = |f: &AudioFrame| f.bands.iter().filter(|&&v| v > 0.5).count();
        assert!(
            spread(&b) > spread(&a) + 10,
            "noise {} vs sine {}",
            spread(&b),
            spread(&a)
        );
    }

    #[test]
    fn deterministic_and_safe_on_hostile_input() {
        let s = synth::noise(11, 0.5, SR as usize);
        let (a, b) = (run(&s), run(&s));
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.bands == y.bands
            && x.energy == y.energy
            && x.t_capture_ns == y.t_capture_ns));
        let mut hostile = vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 10.0, -10.0];
        hostile.extend(synth::sine(1000.0, 3.0, SR, SR as usize));
        for f in run(&hostile) {
            assert!(
                f.bands
                    .iter()
                    .chain(&f.group_fast)
                    .chain(&f.group_slow)
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            );
            assert!(f.energy.is_finite() && f.flux.is_finite() && f.rms_db.is_finite());
        }
    }

    #[test]
    fn timestamps_are_monotonic_and_match_the_sample_clock() {
        let frames = run(&synth::sine(300.0, 0.5, SR, SR as usize));
        assert!(
            frames
                .windows(2)
                .all(|w| w[1].t_capture_ns > w[0].t_capture_ns && w[1].seq == w[0].seq + 1)
        );
        let dt = (frames[1].t_capture_ns - frames[0].t_capture_ns) as f32 / 1e9;
        assert!((dt - 512.0 / SR as f32).abs() < 0.0005, "dt {dt}");
    }

    fn run_chunked(samples: &[f32], chunk: usize) -> Vec<AudioFrame> {
        let mut a = Analyzer::new(SR, AnalysisParams::default()).unwrap();
        let mut frames = Vec::new();
        let mut done = 0u64;
        for c in samples.chunks(chunk) {
            done += c.len() as u64;
            a.push(c, done * 1_000_000_000 / u64::from(SR), |f| frames.push(*f));
        }
        frames
    }

    #[test]
    fn results_do_not_depend_on_buffer_size() {
        let mut s = synth::sine(250.0, 0.4, SR, SR as usize);
        s.extend(synth::kick_train(120.0, 2.0, 0.7, SR).0);
        let reference = run_chunked(&s, 480);
        for chunk in [1usize, 137, 512, 1024, 4096, 100_000] {
            let f = run_chunked(&s, chunk);
            assert_eq!(f.len(), reference.len(), "chunk {chunk}");
            assert!(
                f.iter().zip(&reference).all(|(a, b)| a.bands == b.bands
                    && a.onset_event == b.onset_event
                    && a.energy == b.energy),
                "chunk {chunk} changed the analysis"
            );
        }
    }

    #[test]
    fn discontinuities_stay_finite_and_recover() {
        // Hard splice between two tones, a dropout of zeros, and a full-scale click.
        let mut s = synth::sine(300.0, 0.5, SR, SR as usize / 2);
        s.extend(synth::sine(3000.0, 0.5, SR, SR as usize / 2));
        s.extend(vec![0.0; 2400]);
        s.push(1.0);
        s.push(-1.0);
        s.extend(synth::sine(300.0, 0.5, SR, SR as usize));
        let frames = run(&s);
        assert!(frames.iter().all(|f| f.bands.iter().all(|v| v.is_finite())
            && f.energy.is_finite()
            && (0.0..=1.0).contains(&f.energy)));
        assert!(
            last(&frames).group_fast[2] > 0.6,
            "recovers: {:?}",
            last(&frames).group_fast
        );
    }

    #[test]
    fn long_run_is_numerically_stable() {
        // 10 minutes of alternating loud/quiet music-like material, fed in 10 ms chunks.
        let mut a = Analyzer::new(SR, AnalysisParams::default()).unwrap();
        let mut worst = 0.0f32;
        let mut frames = 0u64;
        for sec in 0..600u32 {
            let amp = if sec % 20 < 10 { 0.6 } else { 0.03 };
            let x = synth::sine(110.0 + (sec % 7) as f32 * 300.0, amp, SR, SR as usize);
            for c in x.chunks(480) {
                a.push(c, 0, |f| {
                    frames += 1;
                    worst = worst
                        .max(f.energy)
                        .max(f.bands.iter().cloned().fold(0.0, f32::max));
                    assert!(f.energy.is_finite() && f.rms_db.is_finite() && f.flux.is_finite());
                });
            }
        }
        assert!(
            frames > 50_000 && worst <= 1.0,
            "frames {frames}, worst {worst}"
        );
    }

    #[test]
    fn quiet_passage_after_loud_stays_visible_but_calmer() {
        let mut s = synth::sine(100.0, 0.8, SR, 4 * SR as usize);
        s.extend(synth::sine(100.0, 0.03, SR, 4 * SR as usize));
        let frames = run(&s);
        let per_s = SR as usize / 512;
        let loud = frames[4 * per_s - 3];
        let quiet_early = frames[5 * per_s];
        let quiet_late = last(&frames);
        assert!(
            quiet_early.group_fast[1] > 0.3,
            "still visible: {}",
            quiet_early.group_fast[1]
        );
        assert!(
            quiet_late.energy < loud.energy - 0.1,
            "calmer: {} vs {}",
            quiet_late.energy,
            loud.energy
        );
    }

    #[test]
    fn energy_orders_recording_levels_without_saturating_the_quiet_one() {
        let e = |amp: f32| last(&run(&synth::noise(5, amp, 5 * SR as usize))).energy;
        let (hi, mid, lo) = (e(0.7), e(0.15), e(0.02));
        assert!(hi > mid + 0.05 && mid > lo + 0.05, "{hi} {mid} {lo}");
        assert!(lo < 0.5 * hi, "quiet recording stays calm: {lo} vs {hi}");
    }

    #[test]
    fn rhythmic_material_has_more_onset_activity_than_a_sustained_tone() {
        let count = |x: &[f32]| run(x).iter().filter(|f| f.onset_event).count();
        let (kicks, _) = synth::kick_train(120.0, 6.0, 0.7, SR);
        let tone = synth::sine(55.0, 0.7, SR, 6 * SR as usize);
        assert!(
            count(&kicks) >= 8 && count(&tone) <= 1,
            "{} vs {}",
            count(&kicks),
            count(&tone)
        );
    }

    #[test]
    fn stationary_noise_makes_almost_no_onsets() {
        // Flux of stationary noise is random; the adaptive threshold must keep false beats rare.
        for amp in [0.05f32, 0.3] {
            let frames = run(&synth::noise(9, amp, 20 * SR as usize));
            let n = frames.iter().skip(60).filter(|f| f.onset_event).count();
            assert!(n <= 3, "amp {amp}: {n} false onsets in ~19 s");
        }
    }

    #[test]
    fn hostile_parameters_cannot_poison_the_analyzer() {
        let p = AnalysisParams {
            range_db: f32::NAN,
            ref_decay_db_per_s: f32::INFINITY,
            f_min: -5.0,
            f_max: f32::NAN,
            onset_k: f32::NAN,
            tilt_db_per_oct: f32::NAN,
            silence_db: f32::NAN,
            ref_min_db: f32::NAN,
            band_count: 10_000,
            hop: 0,
            ..AnalysisParams::default()
        };
        let mut a = Analyzer::new(SR, p).unwrap();
        let mut ok = true;
        a.push(&synth::noise(3, 0.5, SR as usize), 0, |f| {
            ok &= f.bands.iter().all(|v| v.is_finite()) && f.energy.is_finite();
        });
        assert!(ok);
    }

    // NOTE (design): band/group levels are shape-normalised against a rolling reference, so a steady
    // tone looks full at both 0.8 and 0.1 amplitude. Absolute loudness is carried by `rms_db` and
    // `energy`; scenes must combine both (ADR-0008), never rely on band height for loudness.

    /// A music-like spectrum: components log-spaced 50 Hz-14 kHz with amplitude falling 4.5 dB per octave
    /// (typical of mastered music). Returns (mean bass level, mean treble level) of the normalised bands.
    fn bass_and_treble_levels(tilt: f32) -> (f32, f32) {
        let n = 3 * SR as usize;
        let mut x = vec![0.0f32; n];
        for k in 0..40 {
            let f = 50.0 * (14_000.0f32 / 50.0).powf(k as f32 / 39.0);
            let amp = 0.02 * (f / 1000.0).powf(-0.75); // -4.5 dB per octave
            for (v, s) in x.iter_mut().zip(synth::sine(f * 1.013, amp, SR, n)) {
                *v += s;
            }
        }
        let p = AnalysisParams {
            tilt_db_per_oct: tilt,
            ..AnalysisParams::default()
        };
        let mut a = Analyzer::new(SR, p).unwrap();
        let mut out = (0.0, 0.0);
        for c in x.chunks(480) {
            a.push(c, 0, |f| {
                let m = |r: std::ops::Range<usize>| {
                    f.bands[r.clone()].iter().sum::<f32>() / r.len() as f32
                };
                out = (m(0..16), m(40..64));
            });
        }
        out
    }

    #[test]
    fn tilt_compensation_flattens_a_music_like_spectrum() {
        let (fb, ft) = bass_and_treble_levels(0.0);
        let (tb, tt) = bass_and_treble_levels(4.5);
        eprintln!("flat: bass {fb:.2} treble {ft:.2}; tilted: bass {tb:.2} treble {tt:.2}");
        assert!(fb - ft > 0.3, "fixture must be bass-heavy without tilt");
        assert!(
            (tb - tt).abs() < (fb - ft) - 0.2,
            "tilt must reduce the bass-treble gap"
        );
        assert!(tt > ft + 0.2, "treble becomes visible");
    }

    #[test]
    fn works_at_other_sample_rates() {
        for sr in [22_050u32, 44_100, 96_000] {
            let mut a = Analyzer::new(sr, AnalysisParams::default()).unwrap();
            let mut last_frame = None;
            let x = synth::sine(1000.0, 0.5, sr, sr as usize);
            for c in x.chunks(441) {
                a.push(c, 0, |f| last_frame = Some(*f));
            }
            let f = last_frame.unwrap();
            assert_eq!(argmax(&f.group_fast), 3, "sr {sr}");
        }
    }
}
