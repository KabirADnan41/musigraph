//! Deterministic offline pipeline: samples -> real `Analyzer` -> `SceneMapper` -> `SceneParams`, stepped at a
//! chosen render rate. SYNTHETIC input only (tests, `--scene-selftest`, labelled previews); the live path
//! in `scene_run` uses the same analyzer and mapper with captured system audio instead.

use crate::scene_map::{Input, MapTuning, SceneMapper};
use mg_dsp::features::{MusicFeatures, MusicTracker};
use mg_dsp::{AnalysisParams, Analyzer, AudioFrame};
use mg_gfx::scene::{SceneMode, SceneParams};

/// Names of the built-in SYNTHETIC fixtures (tests and labelled previews only).
pub const FIXTURES: [&str; 12] = [
    "silence", "bass", "mid", "treble", "sweep", "kick", "buildup", "demo", "pad", "drone", "hats",
    "noise",
];

/// Generates a looping synthetic fixture (a few seconds up to 16 s) at `sr`. Never used in live mode.
pub fn fixture(name: &str, sr: u32) -> Option<Vec<f32>> {
    use mg_dsp::synth;
    let n = |s: f32| (s * sr as f32) as usize;
    Some(match name {
        "silence" => vec![0.0; n(4.0)],
        "bass" => synth::sine(70.0, 0.5, sr, n(4.0)),
        "mid" => synth::sine(1000.0, 0.5, sr, n(4.0)),
        "treble" => synth::sine(9000.0, 0.4, sr, n(4.0)),
        "sweep" => synth::sweep(40.0, 14_000.0, 0.5, sr, n(8.0)),
        "kick" => synth::kick_train(120.0, 8.0, 0.8, sr).0,
        "buildup" => synth::noise(5, 1.0, n(16.0))
            .iter()
            .enumerate()
            .map(|(i, w)| w * (0.02 + 0.5 * i as f32 / n(16.0) as f32))
            .collect(),
        "demo" => {
            // Kick + gated bass + hats + a swelling pad: a rough "song" shape for previews.
            let len = n(16.0);
            let mut out = synth::kick_train(110.0, 16.0, 0.6, sr).0;
            out.resize(len, 0.0);
            let bass = synth::sine(82.0, 0.22, sr, len);
            let hat = synth::noise(9, 0.12, len);
            let pad: Vec<f32> = [262.0f32, 330.0, 392.0]
                .iter()
                .map(|&f| synth::sine(f, 1.0, sr, len))
                .fold(vec![0.0; len], |a, b| {
                    a.iter().zip(&b).map(|(x, y)| x + y).collect()
                });
            for i in 0..len {
                let t = i as f32 / sr as f32;
                let gate = if (t * 2.0).fract() < 0.5 { 1.0 } else { 0.0 };
                let hat_gate = if (t / 0.2727).fract() < 0.1 { 1.0 } else { 0.0 };
                let swell = 0.02 + 0.06 * t / 16.0;
                out[i] += bass[i] * gate + hat[i] * hat_gate + pad[i] * swell;
                out[i] = out[i].clamp(-1.0, 1.0);
            }
            out
        }
        // sustained C-major pad (no attack transient: 1.5 s fade-in)
        "pad" => {
            let notes: Vec<Vec<f32>> = [262.0f32, 330.0, 392.0]
                .iter()
                .map(|&f| synth::sine(f, 0.15, sr, n(12.0)))
                .collect();
            (0..n(12.0))
                .map(|i| {
                    let fade = (i as f32 / n(1.5) as f32).min(1.0);
                    fade * fade * (notes[0][i] + notes[1][i] + notes[2][i])
                })
                .collect()
        }
        // sub-bass drone (below the 60 Hz group edge)
        "drone" => synth::sine(45.0, 0.5, sr, n(8.0)),
        // hi-hats only: 30 ms noise bursts every 0.25 s
        "hats" => synth::noise(9, 0.12, n(8.0))
            .iter()
            .enumerate()
            .map(|(i, w)| {
                if (i as f32 / sr as f32 / 0.25).fract() < 0.12 {
                    *w
                } else {
                    0.0
                }
            })
            .collect(),
        // steady broadband (white) noise
        "noise" => synth::noise(5, 0.3, n(8.0)),
        _ => return None,
    })
}

#[cfg_attr(not(test), allow(dead_code))] // `t` and `frame` are read by the unit tests only
pub struct Tick {
    pub t: f32,
    pub frame: AudioFrame,
    pub onsets: u32,
    pub params: SceneParams,
    pub music: MusicFeatures,
}

pub struct Sim {
    analyzer: Analyzer,
    mapper: SceneMapper,
    sr: u32,
    fps: f32,
    carry: f64,
    last: AudioFrame,
    tracker: MusicTracker,
    music: MusicFeatures,
    samples_done: u64,
    ticks: u64,
}

impl Sim {
    pub fn new(sr: u32, fps: f32, tune: MapTuning) -> Self {
        let analyzer = Analyzer::new(sr, AnalysisParams::default()).expect("valid rate");
        Self {
            tracker: analyzer.music_tracker(),
            music: MusicFeatures::default(),
            analyzer,
            mapper: SceneMapper::new(tune),
            sr,
            fps,
            carry: 0.0,
            last: AudioFrame::default(),
            samples_done: 0,
            ticks: 0,
        }
    }

    /// Feeds `samples` (mono, `sr`) and returns one `Tick` per render frame covered.
    pub fn run(&mut self, samples: &[f32]) -> Vec<Tick> {
        let per_tick = f64::from(self.sr) / f64::from(self.fps);
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < samples.len() {
            self.carry += per_tick;
            let n = (self.carry.floor() as usize).min(samples.len() - pos);
            self.carry -= n as f64;
            let chunk = &samples[pos..pos + n];
            pos += n;
            self.samples_done += n as u64;
            let t_ns = self.samples_done * 1_000_000_000 / u64::from(self.sr);
            let (mut onsets, mut strength) = (0u32, 0.0f32);
            let (last, tracker, music) = (&mut self.last, &mut self.tracker, &mut self.music);
            self.analyzer.push(chunk, t_ns, |f| {
                if f.onset_event {
                    onsets += 1;
                    strength = strength.max(f.onset);
                }
                *last = *f;
                *music = tracker.step(f);
            });
            let dt = 1.0 / self.fps;
            let params = self.mapper.update(
                &Input {
                    frame: &self.last,
                    onsets,
                    onset_strength: strength,
                    fresh: true,
                    mode: SceneMode::Synthetic,
                },
                dt,
            );
            self.mapper.observe_music(&self.music, true);
            self.ticks += 1;
            out.push(Tick {
                t: self.ticks as f32 * dt,
                frame: self.last,
                onsets,
                params,
                music: *self.mapper.music(),
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mg_dsp::synth;
    use mg_gfx::scene::SLOTS;

    const SR: u32 = 48_000;

    fn run(x: &[f32], fps: f32) -> Vec<Tick> {
        Sim::new(SR, fps, MapTuning::default()).run(x)
    }

    fn sec(n: f32) -> usize {
        (n * SR as f32) as usize
    }

    /// Mean bar height over a slot range.
    fn mean(p: &SceneParams, r: std::ops::Range<usize>) -> f32 {
        let s: f32 = p.bars[r.clone()].iter().sum();
        s / r.len() as f32
    }
    fn centre(p: &SceneParams) -> f32 {
        mean(p, 40..56)
    }
    fn edges(p: &SceneParams) -> f32 {
        (mean(p, 0..16) + mean(p, 80..96)) / 2.0
    }
    fn all_finite(p: &SceneParams) -> bool {
        p.pack((100, 100))
            .chunks_exact(4)
            .all(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]).is_finite())
    }

    #[test]
    fn silence_is_flat_with_no_rings_and_no_motion() {
        let t = run(&vec![0.0; sec(30.0)], 60.0);
        for k in &t[240..] {
            assert!(k.params.bars.iter().all(|&b| b == 0.0), "t={}", k.t);
            assert!(k.params.peaks.iter().all(|&b| b < 1e-3));
            assert!(
                k.params.rings.iter().all(|r| r[1] == 0.0),
                "no fake impulses"
            );
        }
        assert!(t.iter().all(|k| all_finite(&k.params) && k.onsets == 0));
        assert!(t.last().unwrap().params.glow < 0.3 && t.last().unwrap().params.energy < 0.15);
    }

    #[test]
    fn bass_mid_and_treble_tones_light_different_places() {
        let at = |f: f32| {
            let t = run(&synth::sine(f, 0.5, SR, sec(2.0)), 60.0);
            t.last().unwrap().params.clone()
        };
        let (bass, mid, treb) = (at(80.0), at(1000.0), at(9000.0));
        assert!(
            centre(&bass) > 0.5 && edges(&bass) < 0.15,
            "bass {} {}",
            centre(&bass),
            edges(&bass)
        );
        assert!(
            edges(&treb) > centre(&treb) + 0.04 && edges(&treb) > 0.04,
            "treble {} {}",
            edges(&treb),
            centre(&treb)
        );
        // the mid tone peaks between the two
        let peak_slot = |p: &SceneParams| {
            (48..96)
                .fold(
                    (48, 0.0),
                    |m, i| if p.bars[i] > m.1 { (i, p.bars[i]) } else { m },
                )
                .0
        };
        assert!(peak_slot(&bass) < peak_slot(&mid) && peak_slot(&mid) < peak_slot(&treb));
        // mirrored layout
        for i in 0..SLOTS / 2 {
            assert!((bass.bars[i] - bass.bars[SLOTS - 1 - i]).abs() < 1e-5);
        }
    }

    #[test]
    fn sweep_moves_the_lit_slot_outward() {
        let t = run(&synth::sweep(50.0, 12_000.0, 0.5, SR, sec(6.0)), 60.0);
        let peak = |p: &SceneParams| {
            (48..96)
                .fold(
                    (48, 0.0),
                    |m, i| if p.bars[i] > m.1 { (i, p.bars[i]) } else { m },
                )
                .0
        };
        let a = peak(&t[60].params);
        let b = peak(&t[180].params);
        let c = peak(&t[340].params);
        assert!(a + 8 < b && b + 8 < c, "{a} {b} {c}");
    }

    #[test]
    fn louder_music_gives_taller_bars_and_more_intensity() {
        let make = |amp| {
            let t = run(&synth::noise(4, amp, sec(8.0)), 60.0);
            t.last().unwrap().params.clone()
        };
        let (quiet, loud) = (make(0.02), make(0.6));
        assert!(
            mean(&loud, 0..96) > mean(&quiet, 0..96) + 0.1,
            "{} vs {}",
            mean(&loud, 0..96),
            mean(&quiet, 0..96)
        );
        assert!(loud.energy > quiet.energy + 0.1 && loud.glow > quiet.glow + 0.05);
        assert!(mean(&quiet, 0..96) > 0.02, "quiet music is still visible");
    }

    #[test]
    fn kick_pattern_spawns_rings_on_real_onsets_only() {
        let (x, kicks) = synth::kick_train(120.0, 10.0, 0.8, SR);
        let t = run(&x, 60.0);
        // ring spawn = ring age reset to ~0 with an amplitude
        let spawns: Vec<f32> = t
            .windows(2)
            .filter(|w| {
                w[1].params.rings[0][0] < 0.001
                    && w[1].params.rings[0][1] > 0.0
                    && w[0].params.rings[0][0] > 0.01
            })
            .map(|w| w[1].t)
            .collect();
        let matched = kicks
            .iter()
            .filter(|&&k| spawns.iter().any(|&s| s >= k && s <= k + 0.15))
            .count();
        let extra = spawns
            .iter()
            .filter(|&&s| !kicks.iter().any(|&k| s >= k && s <= k + 0.15))
            .count();
        assert!(
            matched * 100 >= kicks.len() * 85,
            "matched {matched}/{} spawns {spawns:?}",
            kicks.len()
        );
        assert!(extra <= 2, "false rings {extra}");
        assert!(
            spawns.windows(2).all(|w| w[1] - w[0] >= 0.11),
            "refractory respected"
        );
        assert!(
            t.iter()
                .all(|k| k.params.rings.iter().all(|r| r[1] <= 0.25))
        );
    }

    #[test]
    fn sudden_transient_after_silence_is_seen_within_100ms() {
        let mut x = vec![0.0; sec(2.0)];
        let burst = synth::noise(8, 0.9, sec(0.05));
        let at = x.len();
        x.extend(burst);
        x.extend(vec![0.0; sec(1.0)]);
        let t = run(&x, 60.0);
        let t_event = at as f32 / SR as f32;
        let first = t
            .iter()
            .find(|k| k.params.rings[0][1] > 0.0)
            .map(|k| k.t)
            .expect("a ring appeared");
        assert!(
            first >= t_event && first - t_event < 0.1,
            "latency {}",
            first - t_event
        );
        assert!(
            t.iter()
                .filter(|k| k.t < t_event)
                .all(|k| k.params.rings[0][1] == 0.0)
        );
    }

    #[test]
    fn buildup_raises_intensity_gradually() {
        let n = sec(12.0);
        let x: Vec<f32> = synth::noise(6, 1.0, n)
            .iter()
            .enumerate()
            .map(|(i, w)| w * (0.03 + 0.6 * i as f32 / n as f32))
            .collect();
        let t = run(&x, 60.0);
        let e = |s: f32| t[(s * 60.0) as usize].params.energy;
        let h = |s: f32| t[(s * 60.0) as usize].params.haze;
        assert!(
            e(2.0) + 0.1 < e(11.5) && h(2.0) + 0.1 < h(11.5),
            "{} {} {} {}",
            e(2.0),
            e(11.5),
            h(2.0),
            h(11.5)
        );
        // gradual, no jumps: the slow state moves by less than 0.1 per second
        assert!(
            t.windows(60)
                .all(|w| (w[59].params.energy - w[0].params.energy).abs() < 0.12)
        );
    }

    #[test]
    fn stop_decays_smoothly_and_resume_recovers() {
        let mut x = synth::sine(200.0, 0.5, SR, sec(3.0));
        x.extend(vec![0.0; sec(10.0)]);
        x.extend(synth::sine(200.0, 0.5, SR, sec(2.0)));
        let t = run(&x, 60.0);
        let at = |s: f32| &t[(s * 60.0) as usize].params;
        let tallest = |p: &SceneParams| p.bars.iter().cloned().fold(0.0, f32::max);
        assert!(tallest(at(2.9)) > 0.4);
        assert!(
            at(4.8).bars.iter().all(|&b| b < 0.05),
            "bars gone within ~1.8 s of stopping"
        );
        assert!(
            at(13.0).energy < 0.3 && at(13.0).glow < 0.3,
            "settled: {} {}",
            at(13.0).energy,
            at(13.0).glow
        );
        let bad = t
            .iter()
            .skip(240)
            .take(480)
            // a ring may still be fading from the hard cut at t=3 s (a real click), but none may SPAWN later
            .find(|k| {
                k.params
                    .rings
                    .iter()
                    .any(|r| r[1] > 0.0 && k.t - r[0] > 3.5)
            });
        assert!(
            bad.is_none(),
            "ring during silence at t={:?} {:?} silent={}",
            bad.map(|k| k.t),
            bad.map(|k| k.params.rings),
            bad.is_some_and(|k| k.frame.silent)
        );
        assert!(tallest(&t.last().unwrap().params) > 0.4, "resumes");
        // bars never jump faster than the configured rise limit
        let limit = MapTuning::default().rise_per_s / 60.0 + 1e-4;
        assert!(
            t.windows(2)
                .all(|w| (0..SLOTS).all(|i| w[1].params.bars[i] - w[0].params.bars[i] <= limit))
        );
    }

    #[test]
    fn same_music_has_the_same_timing_at_any_frame_rate() {
        let (x, _) = synth::kick_train(100.0, 6.0, 0.7, SR);
        let a = run(&x, 30.0);
        let b = run(&x, 144.0);
        let av = |t: &[Tick], s: f32| t[(s * t.len() as f32 / 6.0) as usize].params.energy;
        for s in [2.0f32, 4.0, 5.9] {
            assert!(
                (av(&a, s) - av(&b, s)).abs() < 0.06,
                "energy at {s}: {} vs {}",
                av(&a, s),
                av(&b, s)
            );
        }
        let m = |t: &[Tick]| t.iter().map(|k| centre(&k.params)).sum::<f32>() / t.len() as f32;
        assert!((m(&a) - m(&b)).abs() < 0.08, "{} vs {}", m(&a), m(&b));
    }

    #[test]
    fn stale_audio_decays_to_the_error_state_without_inventing_activity() {
        let mut m = SceneMapper::new(MapTuning::default());
        let mut sim = Sim::new(SR, 60.0, MapTuning::default());
        let live = sim
            .run(&synth::sine(300.0, 0.5, SR, sec(2.0)))
            .last()
            .unwrap()
            .frame;
        for _ in 0..120 {
            m.update(
                &Input {
                    frame: &live,
                    onsets: 0,
                    onset_strength: 0.0,
                    fresh: true,
                    mode: SceneMode::Live,
                },
                1.0 / 60.0,
            );
        }
        // capture stalls: no fresh frames, but a stale onset request must be ignored
        let mut last = SceneParams::default();
        for _ in 0..300 {
            last = m.update(
                &Input {
                    frame: &live,
                    onsets: 3,
                    onset_strength: 1.0,
                    fresh: false,
                    mode: SceneMode::Live,
                },
                1.0 / 60.0,
            );
        }
        assert_eq!(last.mode, SceneMode::NoAudio);
        assert!(
            last.bars.iter().all(|&b| b == 0.0)
                && last.rings.iter().all(|r| r[1] == 0.0 || r[0] > 1.0)
        );
    }

    #[test]
    fn hostile_frames_and_dt_never_produce_invalid_parameters() {
        let mut m = SceneMapper::new(MapTuning::default());
        let f = AudioFrame {
            silent: false,
            bands: [f32::NAN; mg_dsp::MAX_BANDS],
            group_fast: [f32::INFINITY; mg_dsp::GROUPS],
            group_medium: [f32::NAN; mg_dsp::GROUPS],
            group_slow: [-5.0; mg_dsp::GROUPS],
            rms_db: f32::NAN,
            energy: f32::INFINITY,
            band_count: 9999,
            ..AudioFrame::default()
        };
        for dt in [0.0, 1e-9, 0.016, 5.0, f32::NAN, -1.0] {
            let p = m.update(
                &Input {
                    frame: &f,
                    onsets: u32::MAX,
                    onset_strength: f32::NAN,
                    fresh: true,
                    mode: SceneMode::Live,
                },
                dt,
            );
            assert!(all_finite(&p), "dt {dt}");
            assert!(
                p.bars
                    .iter()
                    .chain(&p.peaks)
                    .all(|v| (0.0..=1.0).contains(v))
            );
            assert!(p.palette.iter().flatten().all(|v| (0.0..=1.5).contains(v)));
        }
    }

    #[test]
    fn different_songs_settle_on_different_palettes_and_motion() {
        let n = sec(40.0);
        let bassy: Vec<f32> = synth::sine(70.0, 0.4, SR, n)
            .iter()
            .zip(synth::noise(2, 0.02, n))
            .map(|(a, w)| a + w)
            .collect();
        let bright: Vec<f32> = synth::sine(6000.0, 0.3, SR, n)
            .iter()
            .zip(synth::noise(3, 0.1, n))
            .map(|(a, w)| a + w)
            .collect();
        let (pb, pr) = (
            run(&bassy, 60.0).pop().unwrap().params,
            run(&bright, 60.0).pop().unwrap().params,
        );
        let dist = |a: &[f32; 3], b: &[f32; 3]| (0..3).map(|i| (a[i] - b[i]).abs()).sum::<f32>();
        let d = dist(&pb.palette[1], &pr.palette[1]) + dist(&pb.palette[2], &pr.palette[2]);
        assert!(d > 0.5, "palettes too similar: {d}");
    }

    #[test]
    fn colour_listens_only_to_slow_signals_so_beats_cannot_flash_it() {
        let (x, _) = synth::kick_train(150.0, 8.0, 0.8, SR);
        let t = run(&x, 60.0);
        // palette moves by at most the configured hue slew per second -> no per-beat colour change
        let lum = |p: &SceneParams| p.palette.iter().flatten().sum::<f32>();
        assert!(
            t.windows(10)
                .all(|w| (lum(&w[9].params) - lum(&w[0].params)).abs() < 0.35)
        );
    }

    /// FNV-1a over the packed GPU bytes of every tick (golden: music plumbing must not change GPU output).
    fn pack_hash(name: &str) -> u64 {
        let x = fixture(name, SR).unwrap();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for t in run(&x[..sec(3.0).min(x.len())], 60.0) {
            for (i, b) in t.params.pack((1280, 720)).into_iter().enumerate() {
                // Album Focus moved to its own uniform block (FocusParams): ex2 and ex3.x / .z / .w of the shared block
                // are retired and no scene reads them. Every byte that is still live is hashed.
                let b = if (1312..1332).contains(&i) || (1336..1344).contains(&i) {
                    0
                } else {
                    b
                };
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        h
    }

    #[test]
    fn packed_uniform_bytes_match_golden_hashes() {
        // Captured from the code BEFORE the music plumbing existed (3 s of each fixture at 60 fps, 1280x720), with the
        // retired Focus bytes masked: the same masked hash was recomputed on the unchanged parent commit (a8cb2c0), so
        // every byte that is still read by a scene is identical to what it was.
        assert_eq!(pack_hash("kick"), 0x62ad_de97_0db8_b8d4);
        assert_eq!(pack_hash("demo"), 0x5fd9_bb52_3c3a_dc1e);
    }

    /// Mean of a per-tick music value over the last second of a 6 s run.
    fn music_tail(name: &str, f: impl Fn(&MusicFeatures) -> f32) -> f32 {
        let x = fixture(name, SR).unwrap();
        let t = run(&x[..sec(6.0).min(x.len())], 60.0);
        assert!(t.iter().all(|k| f(&k.music).is_finite()));
        t[t.len() - 60..].iter().map(|k| f(&k.music)).sum::<f32>() / 60.0
    }

    #[test]
    fn music_features_differ_per_fixture_and_are_finite() {
        let level = |m: &MusicFeatures| m.level;
        let dens = |m: &MusicFeatures| m.onset_density;
        let crest = |m: &MusicFeatures| m.crest_db;
        let calm = |m: &MusicFeatures| m.calm;
        assert!(music_tail("silence", calm) > 0.9);
        assert_eq!(music_tail("silence", level), 0.0);
        assert!(
            music_tail("kick", dens) > music_tail("mid", dens) + 0.5,
            "kick is rhythmic, a pad is not"
        );
        assert!(
            music_tail("kick", crest) > music_tail("mid", crest) + 3.0,
            "kick is peaky, a sine is not"
        );
        assert!(
            music_tail("mid", calm) > music_tail("kick", calm) + 0.3,
            "steady tone is calmer than a kick train"
        );
        assert!(music_tail("kick", level) > music_tail("silence", level));
    }

    #[test]
    fn stale_audio_gives_rest_music_values() {
        let mut m = SceneMapper::new(MapTuning::default());
        let loud = MusicFeatures {
            level: 0.9,
            calm: 0.0,
            onset_density: 5.0,
            ..MusicFeatures::default()
        };
        m.observe_music(&loud, true);
        assert_eq!(m.music().level, 0.9);
        m.observe_music(&loud, false);
        let rest = MusicFeatures::default();
        assert_eq!(
            (m.music().level, m.music().calm, m.music().onset_density),
            (rest.level, rest.calm, rest.onset_density)
        );
    }
}
