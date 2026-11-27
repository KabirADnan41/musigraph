//! Privacy-safe running summary of `MusicFeatures` for tuning the calm gate on real music.
//! Counts, running sums and one fixed histogram only: no per-frame storage, no audio, no titles, no disk.

use mg_dsp::features::MusicFeatures;

const CREST_BINS: usize = 160; // 0..40 dB in 0.25 dB steps (MusicFeatures::crest_db range)
const CREST_MAX: f32 = 40.0;

#[derive(Default)]
pub struct MusicSummary {
    frames: u64,
    calm: f64,
    calm_gt_half: u64,
    onset_density: f64,
    build: f64,
    level: f64,
    crest: Option<Box<[u32; CREST_BINS]>>,
}

impl MusicSummary {
    pub fn add(&mut self, m: &MusicFeatures) {
        let fin = |x: f32| if x.is_finite() { x } else { 0.0 };
        self.frames += 1;
        self.calm += f64::from(fin(m.calm));
        self.calm_gt_half += u64::from(fin(m.calm) > 0.5);
        self.onset_density += f64::from(fin(m.onset_density));
        self.build += f64::from(fin(m.build));
        self.level += f64::from(fin(m.level));
        let bin = (fin(m.crest_db).clamp(0.0, CREST_MAX) / CREST_MAX * CREST_BINS as f32) as usize;
        self.crest.get_or_insert_with(|| Box::new([0; CREST_BINS]))[bin.min(CREST_BINS - 1)] += 1;
    }

    /// Bin-centre estimate of the crest percentile `p` in 0..1 (0 when empty).
    fn crest_pct(&self, p: f64) -> f32 {
        let Some(h) = self.crest.as_ref() else {
            return 0.0;
        };
        let target = ((self.frames as f64 - 1.0) * p).round() as u64;
        let mut acc = 0u64;
        for (i, &c) in h.iter().enumerate() {
            acc += u64::from(c);
            if acc > target {
                return (i as f32 + 0.5) * CREST_MAX / CREST_BINS as f32;
            }
        }
        CREST_MAX
    }

    pub fn line(&self) -> String {
        let n = self.frames.max(1) as f64;
        format!(
            "MUSIC summary: frames={} calm_mean={:.3} calm_frac_gt_0.5={:.3} crest_db_p50={:.1} crest_db_p90={:.1} onset_density_mean={:.2} build_mean={:.3} level_mean={:.3}",
            self.frames,
            self.calm / n,
            self.calm_gt_half as f64 / n,
            self.crest_pct(0.5),
            self.crest_pct(0.9),
            self.onset_density / n,
            self.build / n,
            self.level / n
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(calm: f32, crest: f32) -> MusicFeatures {
        MusicFeatures {
            calm,
            crest_db: crest,
            onset_density: 2.0,
            build: 0.5,
            level: 0.25,
            ..MusicFeatures::default()
        }
    }

    #[test]
    fn empty_summary_is_zero_and_finite() {
        let l = MusicSummary::default().line();
        assert!(
            l.starts_with("MUSIC summary: frames=0 calm_mean=0.000 "),
            "{l}"
        );
    }

    #[test]
    fn means_fractions_and_percentiles() {
        let mut s = MusicSummary::default();
        for i in 0..100 {
            // 30 calm frames at crest 6 dB, 70 non-calm at crest 20 dB
            s.add(&m(
                if i < 30 { 1.0 } else { 0.0 },
                if i < 30 { 6.0 } else { 20.0 },
            ));
        }
        let l = s.line();
        assert!(
            l.contains("frames=100 calm_mean=0.300 calm_frac_gt_0.5=0.300"),
            "{l}"
        );
        assert!(
            l.contains("onset_density_mean=2.00 build_mean=0.500 level_mean=0.250"),
            "{l}"
        );
        assert!(
            (s.crest_pct(0.5) - 20.0).abs() < 0.25,
            "{}",
            s.crest_pct(0.5)
        );
        assert!((s.crest_pct(0.9) - 20.0).abs() < 0.25);
        assert!((s.crest_pct(0.1) - 6.0).abs() < 0.25);
    }

    #[test]
    fn non_finite_and_out_of_range_inputs_do_not_panic() {
        let mut s = MusicSummary::default();
        s.add(&m(f32::NAN, f32::INFINITY));
        s.add(&m(1.0, -5.0));
        s.add(&m(1.0, 1e9));
        assert!(s.crest_pct(1.0) <= CREST_MAX);
        assert!(!s.line().contains("NaN"));
    }
}
