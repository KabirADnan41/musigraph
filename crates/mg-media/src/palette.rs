//! Album-art palette extraction (pure, deterministic, no OS access).
//!
//! Input: a small RGBA8 image (the caller decodes and downsamples to about 64x64 off the render loop).
//! Output: a role palette in LINEAR sRGB (what the shaders consume):
//! bg, main, secondary, accent, highlight, text.
//!
//! Method (all in OKLab): k-means (k = 6, deterministic initialisation, 10 iterations), score clusters by
//! population^0.5 x (chroma + 0.02) so a vivid minority beats a dull majority, then assign roles with fixed
//! target lightness/chroma so the result is attractive and readable whatever the artwork looks like.
//! Monochrome art stays monochrome; dark/bright art still gets a dark background and light text.
//! The input is untrusted pixel data: sizes are bounded and nothing here can panic on any content.

/// Maximum pixels examined (the caller should pass about 64x64 = 4096).
pub const MAX_PIXELS: usize = 128 * 128;

pub type Rgb = [f32; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Palette {
    pub bg: Rgb,
    pub main: Rgb,
    pub secondary: Rgb,
    pub accent: Rgb,
    pub highlight: Rgb,
    pub text: Rgb,
}

impl Palette {
    /// Neutral palette used when there is no (usable) artwork: dark slate, warm sand, teal, coral.
    pub fn fallback() -> Self {
        let c = |r: u8, g: u8, b: u8| [r, g, b].map(|v| srgb_to_linear(f32::from(v) / 255.0));
        Self {
            bg: c(0x0B, 0x0E, 0x13),
            main: c(0xF2, 0xC4, 0x8D),
            secondary: c(0x5F, 0xB8, 0xC2),
            accent: c(0xE8, 0x73, 0x5A),
            highlight: c(0xFF, 0xF1, 0xDC),
            text: c(0xF6, 0xF1, 0xEA),
        }
    }

    /// Mixes two palettes in OKLab (never in hue angle, which would sweep through a rainbow). `t` in 0..1.
    pub fn mix(&self, other: &Palette, t: f32) -> Palette {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let m = |a: Rgb, b: Rgb| {
            let (x, y) = (to_oklab(a), to_oklab(b));
            from_oklab([
                x[0] + (y[0] - x[0]) * t,
                x[1] + (y[1] - x[1]) * t,
                x[2] + (y[2] - x[2]) * t,
            ])
        };
        Palette {
            bg: m(self.bg, other.bg),
            main: m(self.main, other.main),
            secondary: m(self.secondary, other.secondary),
            accent: m(self.accent, other.accent),
            highlight: m(self.highlight, other.highlight),
            text: m(self.text, other.text),
        }
    }
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

type Lab = [f32; 3];

/// Linear sRGB -> OKLab (Bjorn Ottosson).
pub fn to_oklab(c: Rgb) -> Lab {
    let l = (0.412_221_47 * c[0] + 0.536_332_55 * c[1] + 0.051_445_995 * c[2])
        .max(0.0)
        .cbrt();
    let m = (0.211_903_5 * c[0] + 0.680_699_5 * c[1] + 0.107_396_96 * c[2])
        .max(0.0)
        .cbrt();
    let s = (0.088_302_46 * c[0] + 0.281_718_85 * c[1] + 0.629_978_7 * c[2])
        .max(0.0)
        .cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

/// OKLab -> linear sRGB, unclamped.
fn from_oklab_raw(c: Lab) -> Rgb {
    let l = (c[0] + 0.396_337_78 * c[1] + 0.215_803_76 * c[2]).powi(3);
    let m = (c[0] - 0.105_561_35 * c[1] - 0.063_854_17 * c[2]).powi(3);
    let s = (c[0] - 0.089_484_18 * c[1] - 1.291_485_5 * c[2]).powi(3);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_4 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

fn in_gamut(c: Rgb) -> bool {
    c.iter().all(|v| (-0.001..=1.001).contains(v))
}

fn from_oklab(c: Lab) -> Rgb {
    from_oklab_raw(c).map(|v| v.clamp(0.0, 1.0))
}

/// Colour with the given lightness, hue and at most `chroma`, reduced until it fits the sRGB gamut.
fn lch(l: f32, chroma: f32, hue: f32) -> Rgb {
    let mut c = chroma.max(0.0);
    for _ in 0..24 {
        let rgb = from_oklab_raw([l, c * hue.cos(), c * hue.sin()]);
        if in_gamut(rgb) {
            return rgb.map(|v| v.clamp(0.0, 1.0));
        }
        c *= 0.85;
    }
    from_oklab([l, 0.0, 0.0])
}

#[derive(Clone, Copy)]
struct Cluster {
    n: usize,
    l: f32,
    a: f32,
    b: f32,
}

impl Cluster {
    fn chroma(&self) -> f32 {
        self.a.hypot(self.b)
    }
    fn hue(&self) -> f32 {
        self.b.atan2(self.a)
    }
}

fn hue_gap(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(std::f32::consts::TAU);
    d.min(std::f32::consts::TAU - d)
}

/// Extracts a palette from RGBA8 pixels (`width * height * 4` bytes). Never panics: malformed or empty
/// input, or fully transparent art, yields `Palette::fallback()`.
pub fn extract(rgba: &[u8], width: usize, height: usize) -> Palette {
    let n = width.saturating_mul(height).min(MAX_PIXELS);
    if n == 0 || rgba.len() < n.saturating_mul(4) {
        return Palette::fallback();
    }
    let mut pts: Vec<Lab> = rgba[..n * 4]
        .chunks_exact(4)
        .filter(|p| p[3] >= 128)
        .map(|p| to_oklab([0, 1, 2].map(|i| srgb_to_linear(f32::from(p[i]) / 255.0))))
        .collect();
    if pts.len() < 16 {
        return Palette::fallback();
    }
    // deterministic initialisation: evenly spaced quantiles of lightness
    pts.sort_by(|x, y| x[0].total_cmp(&y[0]));
    const K: usize = 6;
    let mut centres: Vec<Lab> = (0..K)
        .map(|i| pts[(2 * i + 1) * pts.len() / (2 * K)])
        .collect();
    let mut assign = vec![0usize; pts.len()];
    let mut clusters = [Cluster {
        n: 0,
        l: 0.0,
        a: 0.0,
        b: 0.0,
    }; K];
    for _ in 0..10 {
        let mut sums = [[0.0f32; 3]; K];
        let mut counts = [0usize; K];
        for (p, asg) in pts.iter().zip(assign.iter_mut()) {
            let d = |c: &Lab| (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
            let best = (0..K)
                .min_by(|&i, &j| d(&centres[i]).total_cmp(&d(&centres[j])))
                .unwrap_or(0);
            *asg = best;
            counts[best] += 1;
            for k in 0..3 {
                sums[best][k] += p[k];
            }
        }
        for i in 0..K {
            if counts[i] > 0 {
                let f = counts[i] as f32;
                centres[i] = [sums[i][0] / f, sums[i][1] / f, sums[i][2] / f];
            }
            clusters[i] = Cluster {
                n: counts[i],
                l: centres[i][0],
                a: centres[i][1],
                b: centres[i][2],
            };
        }
    }
    let total = pts.len() as f32;
    let mut cs: Vec<Cluster> = clusters
        .iter()
        .copied()
        .filter(|c| c.n as f32 >= 0.03 * total)
        .collect();
    if cs.is_empty() {
        return Palette::fallback();
    }
    let score = |c: &Cluster| (c.n as f32 / total).sqrt() * (c.chroma() + 0.02);
    let max_chroma = cs.iter().map(Cluster::chroma).fold(0.0, f32::max);
    let mono = max_chroma < 0.03;

    // background: the most populous dark cluster (else the darkest), forced dark and nearly neutral
    let bg_c = cs
        .iter()
        .filter(|c| c.l < 0.5)
        .max_by_key(|c| c.n)
        .copied()
        .unwrap_or_else(|| {
            *cs.iter()
                .min_by(|a, b| a.l.total_cmp(&b.l))
                .expect("non-empty")
        });
    let bg = lch(
        bg_c.l.clamp(0.10, 0.18),
        bg_c.chroma().min(0.05),
        bg_c.hue(),
    );

    // main: best-scoring cluster in a usable lightness band (else best overall)
    cs.sort_by(|a, b| score(b).total_cmp(&score(a)));
    let main_c = cs
        .iter()
        .find(|c| (0.3..=0.9).contains(&c.l))
        .copied()
        .unwrap_or(cs[0]);
    let main_h = main_c.hue();
    let main_chroma = if mono {
        0.015
    } else {
        main_c.chroma().clamp(0.10, 0.22)
    };
    let main = lch(0.72, main_chroma, main_h);

    let rest: Vec<Cluster> = cs
        .iter()
        .copied()
        .filter(|c| hue_gap(c.hue(), main_h) > 35f32.to_radians() && c.chroma() > 0.02)
        .collect();
    let rot = |deg: f32| main_h + deg.to_radians();
    let sec_c = rest.first().copied();
    let secondary = match (mono, sec_c) {
        (true, _) | (_, None) => lch(0.62, main_chroma, rot(25.0)),
        (_, Some(c)) => lch(0.62, c.chroma().clamp(0.10, 0.22), c.hue()),
    };
    let acc_c = rest
        .iter()
        .filter(|c| sec_c.is_none_or(|s| hue_gap(c.hue(), s.hue()) > 1e-3))
        .max_by(|a, b| a.chroma().total_cmp(&b.chroma()))
        .copied();
    let accent = match (mono, acc_c) {
        (true, _) | (_, None) => lch(0.80, main_chroma, rot(-40.0)),
        (_, Some(c)) => lch(0.80, c.chroma().clamp(0.10, 0.22), c.hue()),
    };
    Palette {
        bg,
        main,
        secondary,
        accent,
        highlight: lch(0.93, main_chroma * 0.35, main_h),
        text: lch(0.96, if mono { 0.0 } else { 0.02 }, main_h),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(px: &[[u8; 4]], w: usize, h: usize) -> Vec<u8> {
        (0..w * h).flat_map(|i| px[i % px.len()]).collect()
    }
    fn lum(c: Rgb) -> f32 {
        to_oklab(c)[0]
    }
    fn finite_in_range(p: &Palette) -> bool {
        [p.bg, p.main, p.secondary, p.accent, p.highlight, p.text]
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    }

    #[test]
    fn red_art_gives_a_red_palette_and_blue_art_a_blue_one() {
        let red = extract(
            &solid(
                &[[200, 30, 30, 255], [90, 10, 10, 255], [220, 60, 50, 255]],
                64,
                64,
            ),
            64,
            64,
        );
        let blue = extract(
            &solid(
                &[[30, 60, 200, 255], [10, 20, 90, 255], [60, 100, 230, 255]],
                64,
                64,
            ),
            64,
            64,
        );
        assert!(red.main[0] > red.main[2] + 0.1, "red main {:?}", red.main);
        assert!(
            blue.main[2] > blue.main[0] + 0.1,
            "blue main {:?}",
            blue.main
        );
        let d: f32 = (0..3).map(|i| (red.main[i] - blue.main[i]).abs()).sum();
        assert!(d > 0.5, "palettes too similar {d}");
    }

    #[test]
    fn roles_are_readable_and_in_gamut_for_extreme_art() {
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("black", solid(&[[0, 0, 0, 255]], 64, 64)),
            ("white", solid(&[[255, 255, 255, 255]], 64, 64)),
            (
                "gray",
                solid(&[[128, 128, 128, 255], [90, 90, 90, 255]], 64, 64),
            ),
            (
                "neon",
                solid(
                    &[[255, 0, 200, 255], [0, 255, 40, 255], [0, 60, 255, 255]],
                    64,
                    64,
                ),
            ),
            (
                "noise",
                (0..64 * 64 * 4)
                    .map(|i| {
                        ((i * 2654435761u64 as usize) >> 7) as u8 | if i % 4 == 3 { 255 } else { 0 }
                    })
                    .collect(),
            ),
        ];
        for (name, px) in cases {
            let p = extract(&px, 64, 64);
            assert!(finite_in_range(&p), "{name}: {p:?}");
            assert!(
                lum(p.bg) < 0.25,
                "{name}: background must be dark, L={}",
                lum(p.bg)
            );
            assert!(
                lum(p.text) - lum(p.bg) >= 0.6,
                "{name}: text contrast {} vs {}",
                lum(p.text),
                lum(p.bg)
            );
            assert!(
                lum(p.main) > 0.5,
                "{name}: main must be light enough, L={}",
                lum(p.main)
            );
        }
    }

    #[test]
    fn monochrome_art_stays_monochrome() {
        let p = extract(
            &solid(
                &[
                    [200, 200, 200, 255],
                    [40, 40, 40, 255],
                    [120, 120, 120, 255],
                ],
                64,
                64,
            ),
            64,
            64,
        );
        for c in [p.main, p.secondary, p.accent, p.text] {
            let chroma = to_oklab(c)[1].hypot(to_oklab(c)[2]);
            assert!(chroma < 0.04, "chroma {chroma} in {c:?}");
        }
    }

    #[test]
    fn a_vivid_minority_beats_a_dull_majority() {
        // 90% grey-brown, 10% saturated teal: main should be teal-ish (g,b > r)
        let mut px = Vec::new();
        for i in 0..64 * 64 {
            px.extend(if i % 10 == 0 {
                [0u8, 170, 180, 255]
            } else {
                [110, 100, 95, 255]
            });
        }
        let p = extract(&px, 64, 64);
        assert!(p.main[2] > p.main[0] + 0.05, "main {:?}", p.main);
    }

    #[test]
    fn missing_malformed_or_transparent_art_falls_back() {
        let f = Palette::fallback();
        assert_eq!(extract(&[], 0, 0), f);
        assert_eq!(extract(&[1, 2, 3], 4, 4), f); // buffer shorter than claimed
        assert_eq!(extract(&solid(&[[255, 0, 0, 0]], 64, 64), 64, 64), f); // fully transparent
        assert_eq!(extract(&[0u8; 16], usize::MAX, usize::MAX), f);
        assert!(finite_in_range(&f));
    }

    #[test]
    fn extraction_is_deterministic() {
        let px = solid(
            &[[200, 30, 30, 255], [30, 160, 90, 255], [20, 20, 40, 255]],
            64,
            64,
        );
        assert_eq!(extract(&px, 64, 64), extract(&px, 64, 64));
    }

    #[test]
    fn mixing_moves_smoothly_between_palettes_without_leaving_gamut() {
        let a = Palette::fallback();
        let b = extract(
            &solid(&[[30, 60, 200, 255], [10, 20, 90, 255]], 64, 64),
            64,
            64,
        );
        let close = |x: Rgb, y: Rgb| (0..3).all(|i| (x[i] - y[i]).abs() < 1e-3);
        assert!(
            close(a.mix(&b, 0.0).main, a.main),
            "t=0 is the first palette"
        );
        let mut prev = a.main;
        for i in 1..=20 {
            let m = a.mix(&b, i as f32 / 20.0);
            assert!(finite_in_range(&m));
            let step: f32 = (0..3).map(|k| (m.main[k] - prev[k]).abs()).sum();
            assert!(step < 0.2, "step {step}");
            prev = m.main;
        }
        let end: f32 = (0..3).map(|k| (prev[k] - b.main[k]).abs()).sum();
        assert!(end < 0.02);
        assert!(
            close(a.mix(&b, f32::NAN).main, a.main),
            "NaN t is treated as 0"
        );
    }
}
