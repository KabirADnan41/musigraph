//! Higher-level, ADDITIVE music features derived from the existing `AudioFrame` stream (ADR-0008).
//!
//! Purpose: `AudioFrame` levels are normalised against one global rolling reference with fast
//! envelopes, so very different songs saturate to similar 0..1 dynamics. `MusicTracker` adds slow,
//! per-song adaptation and musical descriptors WITHOUT changing any `AudioFrame` value.
//!
//! Input: `AudioFrame`s from one `Analyzer` (use `Analyzer::music_tracker`). Output: one `MusicFeatures`
//! per frame. All time constants are in seconds, so results do not depend on the hop size (the onset
//! path works on ~10 ms bins, so hops above 10 ms stretch its 96-bin history proportionally).
//! Invariants: every output is finite and bounded; no allocation after construction (fixed arrays);
//! silence decays everything to the rest values of `MusicFeatures::default()`; values come only from
//! audio (no random, no timers, no genre rules). Trust boundary: N/A (inputs are already sanitised).
use crate::smooth::Smoother;
use crate::{AnalysisParams, AudioFrame, GROUPS, MAX_BANDS};

const BIN_S: f32 = 0.01;
const HIST: usize = 96; // ~1 s of 10 ms flux bins
const AGC_RANGE_FLOOR: f32 = 0.25;
/// A group whose recent high (`group_fast` is dB-normalised on the global reference) is below `LO`
/// is treated as empty; fully present from `HI` (steady pink noise: quietest group peaks near 0.2).
const PRESENT_LO: f32 = 0.01;
const PRESENT_HI: f32 = 0.05;
/// Silence this long (s) ends the song: the next sound re-seeds every adaptive state.
const RESTART_S: f32 = 0.5;
/// After a (re)start, adaptive states track the input for this long (s) instead of integrating the
/// analyzer's own attack transient as music.
const SEED_S: f32 = 0.5;
/// Everything that must fall to rest quickly in silence.
const TAU_REST: f32 = 1.5;

/// Per-frame music descriptors. Defaults are the rest (silence) values.
#[derive(Clone, Copy, Debug)]
pub struct MusicFeatures {
    /// Per-group level normalised to this song's own recent low/high (0..1; 0.5 when steady).
    /// Dead band: a group whose recent absolute high is at/below the noise floor (`PRESENT_LO`) reads
    /// 0, not 0.5; it fades in to full AGC by `PRESENT_HI`, so empty bands never look half-active.
    pub agc: [f32; GROUPS],
    /// Per-group short envelope of `agc` (attack 25 ms, release 200 ms).
    pub short: [f32; GROUPS],
    /// Per-group long envelope of `agc` (tau 8 s).
    pub long: [f32; GROUPS],
    /// Per-group `max(short - long, 0)`: how far a band is above its own recent average.
    pub accent: [f32; GROUPS],
    /// Absolute loudness 0..1 (45 s EMA of RMS dB, -50..-10 dBFS): quiet songs stay low.
    pub level: f32,
    /// Spectral centroid 0..1 on a log axis over ~200 Hz..8 kHz (tau 1.5 s).
    pub centroid: f32,
    /// True only on the bin in which a flux onset was detected.
    pub onset_event: bool,
    /// Onsets per second, averaged over ~4 s.
    pub onset_density: f32,
    /// Peak / RMS over ~1 s in dB, 0..40.
    pub crest_db: f32,
    /// 1 = calm (low crest and sparse onsets), smoothly to 0.
    pub calm: f32,
    /// 0..1: level rising over ~8 s, plus a rising centroid.
    pub build: f32,
}

impl Default for MusicFeatures {
    fn default() -> Self {
        Self {
            agc: [0.0; GROUPS],
            short: [0.0; GROUPS],
            long: [0.0; GROUPS],
            accent: [0.0; GROUPS],
            level: 0.0,
            centroid: 0.0,
            onset_event: false,
            onset_density: 0.0,
            crest_db: 0.0,
            calm: 1.0,
            build: 0.0,
        }
    }
}

fn fin(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_finite() { x.clamp(lo, hi) } else { lo }
}

/// One-pole step with time constant `tau_s`, independent of `dt`.
fn ema(y: f32, x: f32, dt: f32, tau_s: f32) -> f32 {
    y + (x - y) * (1.0 - (-dt / tau_s).exp())
}

/// Running low/high envelope (asymmetric EMA approximating p10/p90): each side moves away from the
/// signal with tau/6 and back toward it with tau.
#[derive(Clone, Copy, Default)]
struct Range {
    lo: f32,
    hi: f32,
}

impl Range {
    fn update(&mut self, x: f32, dt: f32, tau: f32) {
        self.hi = ema(self.hi, x, dt, if x > self.hi { tau / 6.0 } else { tau });
        self.lo = ema(self.lo, x, dt, if x < self.lo { tau / 6.0 } else { tau });
    }

    /// `(x-lo)/(hi-lo)` with the span floored so steady material sits at 0.5, not at an extreme.
    fn norm(&self, x: f32) -> f32 {
        let span = (self.hi - self.lo).max(AGC_RANGE_FLOOR);
        (0.5 + (x - 0.5 * (self.hi + self.lo)) / span).clamp(0.0, 1.0)
    }
}

pub struct MusicTracker {
    dt: f32,
    centre_x: [f32; MAX_BANDS], // log-frequency position of each band, 0..1 over 200 Hz..8 kHz
    bands: usize,
    range: [Range; GROUPS],
    short: [f32; GROUPS],
    long: [f32; GROUPS],
    age_s: f32,
    silent_s: f32,
    started: bool,
    level_pow: f32,
    lvl_short: f32,
    lvl_long: f32,
    centroid: f32,
    cen_short: f32,
    cen_long: f32,
    peak_hold: f32,
    mean_sq: f32,
    crest: f32,
    calm: f32,
    // onset path (10 ms bins)
    acc_flux: f32,
    acc_dt: f32,
    hist: [f32; HIST],
    hist_n: usize,
    hist_i: usize,
    since_onset_s: f32,
    density: Smoother,
    last: MusicFeatures,
}

impl MusicTracker {
    /// `p` must be the analyzer's (clamped) params; use `Analyzer::music_tracker`.
    pub fn new(sample_rate: u32, p: &AnalysisParams) -> Self {
        // Hardened for hostile params: no panic, finite dt (hop >= 1, sample rate >= 1 kHz).
        let band_count = p.band_count.clamp(1, MAX_BANDS);
        let dt = p.hop.max(1) as f32 / sample_rate.max(1000) as f32;
        let edges = crate::bands::log_edges_hz(band_count, p.f_min, p.f_max);
        let (l0, l1) = (200f32.ln(), 8000f32.ln());
        let mut centre_x = [0.0; MAX_BANDS];
        for (b, c) in centre_x.iter_mut().enumerate().take(band_count) {
            *c = (((edges[b] * edges[b + 1]).sqrt().ln() - l0) / (l1 - l0)).clamp(0.0, 1.0);
        }
        Self {
            dt,
            centre_x,
            bands: band_count,
            range: [Range::default(); GROUPS],
            short: [0.0; GROUPS],
            long: [0.0; GROUPS],
            age_s: 0.0,
            silent_s: 0.0,
            started: false,
            level_pow: 0.0,
            lvl_short: -80.0,
            lvl_long: -80.0,
            centroid: 0.0,
            cen_short: 0.0,
            cen_long: 0.0,
            peak_hold: 0.0,
            mean_sq: 0.0,
            crest: 0.0,
            calm: 1.0,
            acc_flux: 0.0,
            acc_dt: 0.0,
            hist: [0.0; HIST],
            hist_n: 0,
            hist_i: 0,
            since_onset_s: 10.0,
            density: Smoother::new(4000.0, 4000.0),
            last: MusicFeatures::default(),
        }
    }

    /// Advances by one analysis frame and returns the features (all finite and bounded).
    pub fn step(&mut self, f: &AudioFrame) -> MusicFeatures {
        let dt = self.dt;
        let silent = f.silent;
        if silent {
            self.silent_s += dt;
            if self.silent_s >= RESTART_S {
                self.started = false; // the next sound re-learns quickly
            }
        } else {
            self.silent_s = 0.0;
        }
        self.age_s += dt;
        let rms_db = fin(f.rms_db, -120.0, 0.0);
        let first = !silent && !self.started;

        // Spectral centroid target (weights = squared band levels, so leakage skirts matter little).
        let (mut sw, mut swx) = (0.0f32, 0.0f32);
        for b in 0..self.bands {
            let w = fin(f.bands[b], 0.0, 1.0).powi(2);
            sw += w;
            swx += w * self.centre_x[b];
        }
        let target = if silent || sw < 1e-6 { 0.0 } else { swx / sw };

        if first {
            self.started = true;
            self.age_s = 0.0;
        }
        let seed = !silent && self.age_s < SEED_S;
        if seed {
            // (Re)start: slow states track the input until the analyzer settles, so steady material
            // shows no fake accent, build or onset burst; stale pre-silence history is dropped.
            self.level_pow = 10f32.powf(rms_db / 10.0);
            self.lvl_short = rms_db;
            self.lvl_long = rms_db;
            self.cen_short = target;
            self.cen_long = target;
            self.mean_sq = 10f32.powf(rms_db / 10.0);
            self.hist_n = 0;
            self.hist_i = 0;
        }
        let warm = self.age_s < 8.0;
        let tau_agc = if warm { 4.0 } else { 30.0 };

        // Per-group AGC, short/long envelopes, accent.
        let mut out = MusicFeatures::default();
        for g in 0..GROUPS {
            let x = fin(f.group_fast[g], 0.0, 1.0);
            if seed {
                self.range[g] = Range { lo: x, hi: x };
            }
            let n = if silent {
                0.0
            } else {
                self.range[g].update(x, dt, tau_agc);
                let present =
                    ((self.range[g].hi - PRESENT_LO) / (PRESENT_HI - PRESENT_LO)).clamp(0.0, 1.0);
                self.range[g].norm(x) * present * present * (3.0 - 2.0 * present)
            };
            if seed {
                self.short[g] = n;
                self.long[g] = n;
            }
            out.agc[g] = n;
            let tau = if n > self.short[g] { 0.025 } else { 0.2 };
            self.short[g] = ema(self.short[g], n, dt, tau);
            out.short[g] = self.short[g];
            self.long[g] = ema(self.long[g], n, dt, if silent { TAU_REST } else { 8.0 });
            out.long[g] = self.long[g];
            out.accent[g] = (out.short[g] - out.long[g]).max(0.0);
        }

        // Absolute level: EMA in the power domain (sparse kicks must not read as silence), then dB.
        // Its slope is (2 s EMA - 8 s EMA) / 6 s for a linear ramp.
        let tau_lvl = if silent {
            0.5
        } else if warm {
            4.0
        } else {
            45.0
        };
        let power_in = if silent {
            0.0
        } else {
            10f32.powf(rms_db / 10.0)
        };
        self.level_pow = ema(self.level_pow, power_in, dt, tau_lvl);
        let level_db = 10.0 * self.level_pow.max(1e-12).log10();
        out.level = fin((level_db + 50.0) / 40.0, 0.0, 1.0);
        let l = rms_db.max(-80.0);
        self.lvl_short = ema(self.lvl_short, l, dt, 2.0);
        self.lvl_long = ema(self.lvl_long, l, dt, 8.0);

        self.centroid = ema(
            self.centroid,
            target,
            dt,
            if silent { TAU_REST } else { 1.5 },
        );
        out.centroid = fin(self.centroid, 0.0, 1.0);
        self.cen_short = ema(self.cen_short, self.centroid, dt, 2.0);
        self.cen_long = ema(self.cen_long, self.centroid, dt, 8.0);

        // Crest factor: decaying peak hold (tau 1 s) over 1 s mean-square RMS.
        self.peak_hold = fin(f.peak, 0.0, 1.0).max(self.peak_hold * (-dt).exp());
        let power = if silent {
            0.0
        } else {
            10f32.powf(rms_db / 10.0)
        };
        self.mean_sq = ema(self.mean_sq, power, dt, 1.0);
        let crest_t = if silent || self.mean_sq < 1e-9 {
            0.0
        } else {
            (20.0 * self.peak_hold.max(1e-9).log10() - 10.0 * self.mean_sq.log10()).clamp(0.0, 40.0)
        };
        self.crest = ema(self.crest, crest_t, dt, if silent { TAU_REST } else { 0.5 });
        out.crest_db = fin(self.crest, 0.0, 40.0);

        // Onsets: half-wave flux summed into ~10 ms bins, threshold median + 1.5 MAD over ~1 s.
        self.since_onset_s += dt;
        self.acc_flux += fin(f.flux, 0.0, 10.0);
        self.acc_dt += dt;
        let mut event = false;
        if self.acc_dt >= BIN_S {
            let (bin_flux, bin_dt) = (self.acc_flux, self.acc_dt);
            self.acc_flux = 0.0;
            self.acc_dt = 0.0;
            let thr = if self.hist_n >= 16 {
                let mut s = [0.0f32; HIST];
                s[..self.hist_n].copy_from_slice(&self.hist[..self.hist_n]);
                let med = median(&mut s[..self.hist_n]);
                for v in &mut s[..self.hist_n] {
                    *v = (*v - med).abs();
                }
                // 2x median: steady noise fluctuates ~1 sigma above its mean, so MAD alone fires ~1.5/s.
                2.0 * med + 1.5 * median(&mut s[..self.hist_n]) + 0.02
            } else {
                f32::MAX
            };
            event = !silent && bin_flux > thr && self.since_onset_s >= 0.09;
            if event {
                self.since_onset_s = 0.0;
            }
            self.hist[self.hist_i] = bin_flux;
            self.hist_i = (self.hist_i + 1) % HIST;
            self.hist_n = (self.hist_n + 1).min(HIST);
            self.density
                .step(if event { 1.0 / bin_dt } else { 0.0 }, bin_dt);
        }
        out.onset_event = event;
        out.onset_density = fin(self.density.value(), 0.0, 200.0);

        // TODO(tune-on-real-audio): the crest gate (9 -> 6 dB) was tuned on synthetic fixtures and MAY be
        // too strict for real music (pads/ballads are often 10-15 dB crest). Thresholds unchanged.
        // Calm: smooth gates on crest (6 -> 9 dB) and density (0.8 -> 1.6 /s), then a 1.5 s smoother.
        let c_crest = ((9.0 - out.crest_db) / 3.0).clamp(0.0, 1.0);
        let c_dens = ((1.6 - out.onset_density) / 0.8).clamp(0.0, 1.0);
        self.calm = ema(self.calm, c_crest * c_dens, dt, 1.5);
        out.calm = fin(self.calm, 0.0, 1.0);

        // Build: 1 dB/s level rise = full; 0.05 centroid/s rise = full (70/30 mix).
        let slope_db = (self.lvl_short - self.lvl_long) / 6.0;
        let slope_c = (self.cen_short - self.cen_long) / 6.0;
        out.build = if silent {
            0.0
        } else {
            fin(
                0.7 * slope_db.clamp(0.0, 1.0) + 0.3 * (slope_c / 0.05).clamp(0.0, 1.0),
                0.0,
                1.0,
            )
        };
        self.last = out;
        out
    }

    /// Features from the last `step` (rest values before the first).
    pub fn last(&self) -> &MusicFeatures {
        &self.last
    }
}

fn median(s: &mut [f32]) -> f32 {
    s.sort_unstable_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Analyzer, synth};

    const SR: u32 = 48_000;
    const N30: usize = 30 * SR as usize;

    /// Scales to the target RMS (dBFS).
    fn at_db(mut x: Vec<f32>, target_db: f32) -> Vec<f32> {
        let g = 10f32.powf(target_db / 20.0) / synth::rms(&x).max(1e-9);
        x.iter_mut().for_each(|v| *v *= g);
        x
    }

    fn pink(n: usize) -> Vec<f32> {
        let (mut b0, mut b1, mut b2) = (0.0f32, 0.0f32, 0.0f32); // Kellet-style 3-pole filter
        synth::noise(7, 1.0, n)
            .into_iter()
            .map(|w| {
                b0 = 0.99765 * b0 + w * 0.099;
                b1 = 0.963 * b1 + w * 0.2965;
                b2 = 0.57 * b2 + w * 1.0526;
                b0 + b1 + b2 + w * 0.1848
            })
            .collect()
    }

    fn pad(n: usize) -> Vec<f32> {
        // Phase offsets keep the chord's crest factor low, as a real sustained pad's is.
        (0..n)
            .map(|i| {
                let t = i as f32 / SR as f32;
                [(220.0, 0.0, 0.3), (329.63, 2.1, 0.1), (277.18, 4.2, 0.05)]
                    .iter()
                    .map(|&(f, ph, a)| a * (std::f32::consts::TAU * f * t + ph).sin())
                    .sum()
            })
            .collect()
    }

    fn kick(n: usize) -> Vec<f32> {
        let (mut x, _) = synth::kick_train(120.0, n as f32 / SR as f32, 0.8, SR);
        for (a, b) in x.iter_mut().zip(synth::noise(3, 0.004, n)) {
            *a += b;
        }
        x
    }

    /// 8 hats/s: 8 ms bursts of differentiated (high-passed) noise.
    fn hats(n: usize) -> Vec<f32> {
        let w = synth::noise(5, 1.0, n);
        let period = SR as usize / 8;
        (0..n)
            .map(|i| {
                let t = (i % period) as f32 / SR as f32;
                let d = if i > 0 { w[i] - w[i - 1] } else { 0.0 };
                d * (-t * 400.0).exp()
            })
            .collect()
    }

    fn run(x: &[f32], hop: usize) -> Vec<MusicFeatures> {
        let p = AnalysisParams {
            hop,
            ..AnalysisParams::default()
        };
        let mut a = Analyzer::new(SR, p).unwrap();
        let mut t = a.music_tracker();
        let mut out = Vec::new();
        let mut ns = 0u64;
        for c in x.chunks(480) {
            ns += c.len() as u64 * 1_000_000_000 / u64::from(SR);
            a.push(c, ns, |f| out.push(t.step(f)));
        }
        out
    }

    fn mean<F: Fn(&MusicFeatures) -> f32>(m: &[MusicFeatures], f: F) -> f32 {
        m.iter().map(&f).sum::<f32>() / m.len() as f32
    }

    /// Normalised vector of means over the post-warm-up half.
    fn vector(m: &[MusicFeatures]) -> [f32; 6] {
        let m = &m[m.len() * 3 / 10..];
        [
            mean(m, |f| f.centroid),
            mean(m, |f| f.crest_db) / 20.0,
            mean(m, |f| f.onset_density) / 4.0,
            mean(m, |f| f.accent.iter().sum::<f32>() / GROUPS as f32) * 4.0,
            mean(m, |f| f.calm),
            mean(m, |f| f.build) * 4.0,
        ]
    }

    fn dist(a: &[f32; 6], b: &[f32; 6]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    fn all_finite(m: &[MusicFeatures]) -> bool {
        m.iter().all(|f| {
            let v = [
                f.level,
                f.centroid,
                f.onset_density,
                f.crest_db,
                f.calm,
                f.build,
            ];
            v.iter().all(|x| x.is_finite() && *x >= 0.0)
                && f.agc
                    .iter()
                    .chain(&f.short)
                    .chain(&f.long)
                    .chain(&f.accent)
                    .all(|x| (0.0..=1.0).contains(x))
                && f.level <= 1.0
                && f.centroid <= 1.0
                && f.calm <= 1.0
                && f.build <= 1.0
        })
    }

    #[test]
    fn five_songs_are_clearly_separated() {
        let sweep = synth::sweep(100.0, 8000.0, 0.5, SR, N30);
        let songs = [
            ("pink", pink(N30)),
            ("pad", pad(N30)),
            ("kick", kick(N30)),
            ("hats", hats(N30)),
            ("sweep", sweep),
        ];
        let v: Vec<[f32; 6]> = songs
            .iter()
            .map(|(_, s)| vector(&run(&at_db(s.clone(), -18.0), 512)))
            .collect();
        for i in 0..5 {
            for j in i + 1..5 {
                let d = dist(&v[i], &v[j]);
                eprintln!("{} vs {}: {d:.2}", songs[i].0, songs[j].0);
                assert!(
                    d > 0.25,
                    "{} vs {}: {d} ({:?} / {:?})",
                    songs[i].0,
                    songs[j].0,
                    v[i],
                    v[j]
                );
            }
        }
    }

    #[test]
    fn pad_is_calm_without_onsets_and_kick_rate_matches_tempo() {
        let m = run(&at_db(pad(N30), -18.0), 512);
        let late = &m[m.len() / 2..];
        assert!(mean(late, |f| f.onset_density) < 0.1);
        assert!(late.iter().all(|f| !f.onset_event));
        assert!(
            late.iter().all(|f| f.calm > 0.95),
            "calm crest {} dens {}",
            mean(late, |f| f.crest_db),
            mean(late, |f| f.onset_density)
        );
        // Kick: count flux-path onset events over 24 s of steady state (existing detector not used).
        let m = run(&at_db(kick(N30), -18.0), 512);
        let dt = 512.0 / SR as f32;
        let (a, b) = ((5.0 / dt) as usize, (29.0 / dt) as usize);
        let per_min = m[a..b].iter().filter(|f| f.onset_event).count() as f32 / 24.0 * 60.0;
        assert!((118.0..=122.0).contains(&per_min), "{per_min} onsets/min");
        let d = mean(&m[m.len() / 2..], |f| f.onset_density);
        assert!((1.8..=2.2).contains(&d), "density {d}");
    }

    #[test]
    fn sweep_centroid_rises_monotonically() {
        let m = run(
            &at_db(synth::sweep(100.0, 8000.0, 0.5, SR, N30), -18.0),
            512,
        );
        let c: Vec<f32> = m.iter().step_by(40).map(|f| f.centroid).collect();
        let rank = |v: &[f32]| -> Vec<f32> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
            let mut r = vec![0.0; v.len()];
            idx.iter().enumerate().for_each(|(k, &i)| r[i] = k as f32);
            r
        };
        let (rc, ri) = (rank(&c), (0..c.len()).map(|i| i as f32).collect::<Vec<_>>());
        let n = c.len() as f32;
        let d2: f32 = rc.iter().zip(&ri).map(|(a, b)| (a - b) * (a - b)).sum();
        let rho = 1.0 - 6.0 * d2 / (n * (n * n - 1.0));
        assert!(rho > 0.9, "spearman {rho}");
    }

    #[test]
    fn silence_decays_to_rest_and_stays_finite() {
        let mut x = at_db(kick(10 * SR as usize), -18.0);
        x.extend(vec![0.0; 25 * SR as usize]);
        let m = run(&x, 512);
        assert!(all_finite(&m));
        let f = m.last().unwrap();
        let rest = MusicFeatures::default();
        assert!(
            f.agc
                .iter()
                .chain(&f.short)
                .chain(&f.accent)
                .all(|&v| v < 0.01)
        );
        assert!(f.long.iter().all(|&v| v < 0.02), "{:?}", f.long);
        assert!(f.level < 0.02 && f.centroid < 0.02 && f.crest_db < 0.5 && f.build == 0.0);
        assert!(f.onset_density < 0.02 && !f.onset_event);
        assert!(f.calm > 0.98 && rest.calm == 1.0);
    }

    #[test]
    fn hop_size_does_not_change_the_means() {
        let x = at_db(kick(N30), -18.0);
        let (a, b) = (vector(&run(&x, 512)), vector(&run(&x, 1024)));
        for i in 0..6 {
            assert!((a[i] - b[i]).abs() < 0.2, "component {i}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn hostile_input_never_produces_non_finite_output() {
        let mut x = vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 1e30, -1e30];
        x.extend(at_db(pink(SR as usize), -18.0));
        x.extend(vec![1e30; 5000]);
        x.extend(vec![f32::NAN; 5000]);
        x.extend(at_db(kick(2 * SR as usize), -12.0));
        assert!(all_finite(&run(&x, 512)));
        let f = AudioFrame {
            rms_db: f32::NAN,
            flux: f32::INFINITY,
            peak: f32::NAN,
            group_fast: [f32::NAN; GROUPS],
            silent: false,
            ..AudioFrame::default()
        };
        let mut t = MusicTracker::new(SR, &AnalysisParams::default());
        assert!(all_finite(&[t.step(&f), t.step(&f)]));
    }

    #[test]
    fn agc_is_level_invariant_but_absolute_level_is_not() {
        let song = kick(N30);
        let quiet = run(&at_db(song.clone(), -36.0), 512);
        let loud = run(&at_db(song, -14.0), 512);
        let (q, l) = (&quiet[quiet.len() / 2..], &loud[loud.len() / 2..]);
        let agc = |m: &[MusicFeatures]| mean(m, |f| f.agc.iter().sum::<f32>() / GROUPS as f32);
        assert!((agc(q) - agc(l)).abs() < 0.15, "{} vs {}", agc(q), agc(l));
        let (lq, ll) = (mean(q, |f| f.level), mean(l, |f| f.level));
        assert!(ll - lq > 0.4, "level {lq} vs {ll}");
    }

    const DT: f32 = 512.0 / SR as f32;

    fn at(s: f32) -> usize {
        (s / DT) as usize
    }

    /// `lead` s of pink, `gap` s of digital silence, then `tail` s of the same pink, all at -18 dBFS.
    /// Returns the features and the frame index at which the sound restarts.
    fn pink_gap_pink(lead: f32, gap: f32, tail: f32) -> (Vec<MusicFeatures>, usize) {
        let sec = |s: f32| (s * SR as f32) as usize;
        let mut x = at_db(pink(sec(lead + tail)), -18.0);
        let t1 = x.split_off(sec(lead));
        x.extend(vec![0.0; sec(gap)]);
        x.extend(t1);
        (run(&x, 512), at(lead + gap))
    }

    fn mean_accent(m: &[MusicFeatures]) -> f32 {
        mean(m, |f| f.accent.iter().sum::<f32>() / GROUPS as f32)
    }

    #[test]
    fn steady_noise_has_no_fake_accent_or_build_at_start_or_restart() {
        let (m, t0) = pink_gap_pink(6.0, 5.0, 8.0);
        let a_cold = mean_accent(&m[at(0.5)..at(5.0)]);
        let a_warm = mean_accent(&m[t0 + at(0.5)..t0 + at(5.0)]);
        eprintln!("accent cold {a_cold} restart {a_warm}");
        assert!(a_cold < 0.12 && a_warm < 0.12, "{a_cold} {a_warm}");
        let (b_cold, b_warm) = (m[at(5.0)].build, m[t0 + at(4.0)].build);
        eprintln!("build cold {b_cold} restart {b_warm}");
        assert!(b_cold < 0.1 && b_warm < 0.1, "{b_cold} {b_warm}");
    }

    #[test]
    fn steady_noise_after_silence_gives_at_most_one_false_onset() {
        let (m, t0) = pink_gap_pink(6.0, 5.0, 8.0);
        let n = m[t0..t0 + at(3.0)].iter().filter(|f| f.onset_event).count();
        assert!(n <= 1, "{n} false onsets in 3 s");
    }

    #[test]
    fn empty_group_reads_zero_and_steady_noise_groups_stay_mid() {
        let m = run(&at_db(synth::sine(80.0, 1.0, SR, N30), -18.0), 512);
        let late = &m[m.len() / 2..];
        let (bass, treble) = (mean(late, |f| f.agc[1]), mean(late, |f| f.agc[5]));
        let ts = mean(late, |f| f.short[5]);
        eprintln!("sine bass agc {bass} treble agc {treble} short {ts}");
        assert!(
            treble < 0.1 && ts < 0.1 && bass > 0.3,
            "{bass} {treble} {ts}"
        );
        let m = run(&at_db(pink(N30), -18.0), 512);
        let late = &m[m.len() / 2..];
        for g in 0..GROUPS {
            let a = mean(late, |f| f.agc[g]);
            eprintln!("pink group {g} agc {a}");
            assert!((0.35..0.65).contains(&a), "group {g} agc {a}");
        }
    }

    #[test]
    fn one_second_gap_rewarms_like_a_long_gap() {
        let (m, t0) = pink_gap_pink(20.0, 1.0, 8.0);
        let (before, after) = (m[at(19.0)].level, m[t0 + at(3.0)].level);
        eprintln!("level before {before} after {after}");
        assert!((before - after).abs() < 0.1, "{before} vs {after}");
    }

    #[test]
    fn hostile_constructor_params_do_not_panic_or_go_nan() {
        for (sr, band_count, hop) in [
            (0u32, 10_000usize, 0usize),
            (1, 0, 1),
            (u32::MAX, 3, usize::MAX),
        ] {
            let p = AnalysisParams {
                band_count,
                hop,
                ..AnalysisParams::default()
            };
            let mut t = MusicTracker::new(sr, &p);
            let f = AudioFrame {
                silent: false,
                rms_db: -20.0,
                ..AudioFrame::default()
            };
            assert!(all_finite(&[t.step(&f), t.step(&f)]));
        }
    }
}
