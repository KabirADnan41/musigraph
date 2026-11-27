//! Frequency band layout: logarithmically spaced bands mapped onto FFT bins.

/// Logarithmically spaced band edges (count + 1 values) from `f_min` to `f_max` Hz.
pub fn log_edges_hz(count: usize, f_min: f32, f_max: f32) -> Vec<f32> {
    let count = count.max(1);
    let (lo, hi) = (f_min.max(1.0).ln(), f_max.max(f_min + 1.0).ln());
    (0..=count)
        .map(|i| (lo + (hi - lo) * i as f32 / count as f32).exp())
        .collect()
}

/// Inclusive FFT bin range `[first, last]` covering `[f_lo, f_hi)` Hz; always at least one bin
/// and never beyond Nyquist. A band narrower than one bin uses the nearest bin.
pub fn bin_range(f_lo: f32, f_hi: f32, sample_rate: u32, fft_size: usize) -> (usize, usize) {
    let bin_hz = sample_rate as f32 / fft_size as f32;
    let max_bin = fft_size / 2;
    let first = ((f_lo / bin_hz).round() as usize).clamp(1, max_bin);
    let last = (((f_hi / bin_hz).ceil() as usize).saturating_sub(1)).clamp(first, max_bin);
    (first, last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_log_spaced_and_monotonic() {
        let e = log_edges_hz(64, 30.0, 16000.0);
        assert_eq!(e.len(), 65);
        assert!((e[0] - 30.0).abs() < 0.01 && (e[64] - 16000.0).abs() < 1.0);
        let r0 = e[1] / e[0];
        for w in e.windows(2) {
            assert!(w[1] > w[0]);
            assert!((w[1] / w[0] - r0).abs() < 1e-3, "constant ratio");
        }
    }

    #[test]
    fn bin_ranges_are_valid_for_all_supported_rates() {
        for sr in [8_000u32, 22_050, 44_100, 48_000, 96_000, 192_000, 384_000] {
            let e = log_edges_hz(64, 30.0, 16000.0);
            let mut prev_first = 0;
            for b in 0..64 {
                let (f, l) = bin_range(e[b], e[b + 1], sr, 2048);
                assert!(f >= 1 && f <= l && l <= 1024, "sr {sr} band {b}: {f}..{l}");
                assert!(f >= prev_first, "bands never move backwards");
                prev_first = f;
            }
        }
    }

    #[test]
    fn one_kilohertz_lands_in_the_expected_bin() {
        let (f, l) = bin_range(900.0, 1100.0, 48_000, 2048);
        let bin_1k = (1000.0f32 / (48_000.0 / 2048.0)).round() as usize;
        assert!(f <= bin_1k && bin_1k <= l);
    }
}
