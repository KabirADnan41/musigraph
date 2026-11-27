//! Deterministic SYNTHETIC audio for tests, benchmarks and labelled previews only.
//! Production scenes must never be driven by this module (ADR-0008).

use std::f32::consts::TAU;

/// Sine tone of `n` samples.
pub fn sine(freq: f32, amp: f32, sample_rate: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (TAU * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// Logarithmic sweep from `f0` to `f1` over `n` samples.
pub fn sweep(f0: f32, f1: f32, amp: f32, sample_rate: u32, n: usize) -> Vec<f32> {
    let k = (f1 / f0).ln();
    let t_total = n as f32 / sample_rate as f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let phase = TAU * f0 * t_total / k * ((k * t / t_total).exp() - 1.0);
            amp * phase.sin()
        })
        .collect()
}

/// Kick drum pattern: a 55 Hz sine burst with a fast attack and ~120 ms decay at each beat.
/// Returns the samples and the beat start times in seconds.
pub fn kick_train(bpm: f32, seconds: f32, amp: f32, sample_rate: u32) -> (Vec<f32>, Vec<f32>) {
    let n = (seconds * sample_rate as f32) as usize;
    let period = 60.0 / bpm;
    let mut out = vec![0.0; n];
    let mut times = Vec::new();
    let mut t0 = 0.25; // leave a short silent lead-in
    while t0 < seconds - 0.2 {
        times.push(t0);
        let start = (t0 * sample_rate as f32) as usize;
        for i in 0..(0.3 * sample_rate as f32) as usize {
            if start + i >= n {
                break;
            }
            let t = i as f32 / sample_rate as f32;
            let env = (1.0 - (-t * 800.0).exp()) * (-t * 25.0).exp();
            out[start + i] += amp * env * (TAU * (55.0 + 40.0 * (-t * 30.0).exp()) * t).sin();
        }
        t0 += period;
    }
    (out, times)
}

/// White-ish noise from a seeded xorshift generator (deterministic).
pub fn noise(seed: u32, amp: f32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            amp * ((s as f32 / u32::MAX as f32) * 2.0 - 1.0)
        })
        .collect()
}

/// Root-mean-square of a buffer.
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}
