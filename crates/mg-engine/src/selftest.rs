//! `--scene-selftest`: deterministic end-to-end validation of the music-to-visual pipeline with SYNTHETIC
//! fixtures: samples -> real Analyzer -> SceneMapper -> SceneParams -> (headless GPU render, if a hardware
//! device exists) -> measured pixels. Prints measurements and exits non-zero if an expectation fails.
//! It never opens audio devices or windows.

use crate::scene_map::MapTuning;
use crate::sim::{FIXTURES, Sim, Tick, fixture};
use mg_gfx::context::init_headless;
use mg_gfx::scene::{SLOTS, SpectrumScene};

const SR: u32 = 48_000;
const FPS: f32 = 60.0;
const RES: (u32, u32) = (160, 90);

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

/// Mean RGB luminance (0..1) of a pixel rectangle.
fn region_lum(img: &[u8], x0: u32, x1: u32, y0: u32, y1: u32) -> f32 {
    let mut s = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = ((y * RES.0 + x) * 4) as usize;
            s += f64::from(img[i]) + f64::from(img[i + 1]) + f64::from(img[i + 2]);
        }
    }
    (s / f64::from((x1 - x0) * (y1 - y0)) / 3.0 / 255.0) as f32
}

struct Measured {
    mean_lum: Vec<f32>,
    centre_lum: f32,
    edge_lum: f32,
}

fn render_all(ticks: &[Tick]) -> Option<Measured> {
    let h = init_headless().ok()?;
    let scene = SpectrumScene::new_headless(&h.device);
    let mut lum = Vec::with_capacity(ticks.len());
    let (mut centre, mut edge) = (0.0, 0.0);
    for (i, t) in ticks.iter().enumerate() {
        let img = scene
            .render_offscreen(&h.device, &h.queue, RES, &t.params)
            .ok()?;
        lum.push(region_lum(&img, 0, RES.0, 0, RES.1));
        if i + 1 == ticks.len() {
            // band above the horizon where bars live: centre = bass, edges = treble
            centre = region_lum(&img, 64, 96, 35, 53);
            edge = (region_lum(&img, 8, 40, 35, 53) + region_lum(&img, 120, 152, 35, 53)) / 2.0;
        }
    }
    Some(Measured {
        mean_lum: lum,
        centre_lum: centre,
        edge_lum: edge,
    })
}

/// Largest absolute change of mean frame luminance within any 100 ms window.
fn max_delta_100ms(lum: &[f32]) -> f32 {
    let w = (FPS * 0.1).round() as usize;
    lum.windows(w + 1)
        .map(|s| (s[w] - s[0]).abs())
        .fold(0.0, f32::max)
}

pub fn run() -> i32 {
    let mut failures = Vec::new();
    let mut gpu_seen = false;
    for name in FIXTURES {
        let Some(x) = fixture(name, SR) else { continue };
        let take = x.len().min(SR as usize * 8);
        let ticks = Sim::new(SR, FPS, MapTuning::default()).run(&x[..take]);
        let last = &ticks.last().expect("ticks").params;
        let spawns = ticks
            .windows(2)
            .filter(|w| {
                w[1].params.rings[0][1] > 0.0
                    && w[1].params.rings[0][0] < 0.001
                    && w[0].params.rings[0][0] > 0.01
            })
            .count();
        let onsets: u32 = ticks.iter().map(|t| t.onsets).sum();
        let bar_centre = mean(&last.bars[40..56]);
        let bar_edge = (mean(&last.bars[0..16]) + mean(&last.bars[80..96])) / 2.0;
        let tallest = last.bars.iter().cloned().fold(0.0, f32::max);
        let finite = ticks.iter().all(|t| {
            t.params
                .pack(RES)
                .chunks_exact(4)
                .all(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]).is_finite())
        });
        if let Ok(dir) = std::env::var("MG_SCENE_DUMP") {
            // Developer aid: raw RGBA of the last frame at 960x540 (never inside the repository).
            if let Ok(h) = init_headless() {
                let scene = SpectrumScene::new_headless(&h.device);
                if let Ok(img) = scene.render_offscreen(
                    &h.device,
                    &h.queue,
                    (960, 540),
                    &ticks[ticks.len() - 1].params,
                ) {
                    let _ = std::fs::write(format!("{dir}/{name}.rgba"), img);
                }
            }
        }
        let measured = render_all(&ticks);
        gpu_seen |= measured.is_some();
        let (ld, cl, el, ml) = match &measured {
            Some(m) => (
                max_delta_100ms(&m.mean_lum),
                m.centre_lum,
                m.edge_lum,
                mean(&m.mean_lum),
            ),
            None => (f32::NAN, f32::NAN, f32::NAN, f32::NAN),
        };
        println!(
            "FIXTURE {name:8} ticks={} onsets={onsets} ring_spawns={spawns} bars(centre={bar_centre:.2} edge={bar_edge:.2} max={tallest:.2}) energy={:.2} haze={:.2} glow={:.2} | rendered: mean_lum={ml:.3} centre_lum={cl:.3} edge_lum={el:.3} max_lum_delta_100ms={ld:.3}",
            ticks.len(),
            last.energy,
            last.haze,
            last.glow
        );
        let mut expect = |ok: bool, what: &str| {
            if !ok {
                failures.push(format!("{name}: {what}"));
            }
        };
        expect(finite, "non-finite shader parameter");
        expect(
            ld.is_nan() || ld < 0.10,
            "luminance changed by >= 0.10 within 100 ms",
        );
        match name {
            "silence" => {
                expect(
                    tallest == 0.0 && spawns == 0 && onsets == 0,
                    "silence must be flat, no impulses",
                );
            }
            "bass" => {
                expect(
                    bar_centre > 0.4 && bar_edge < 0.1,
                    "bass must light the centre",
                );
                expect(
                    cl.is_nan() || cl > el + 0.01,
                    "rendered centre must be brighter than the edges",
                );
            }
            "treble" => {
                expect(bar_edge > bar_centre, "treble must light the outer slots");
                expect(
                    el.is_nan() || el > cl,
                    "rendered edges must be brighter than the centre",
                );
            }
            "kick" => {
                let kicks = mg_dsp::synth::kick_train(120.0, 8.0, 0.8, SR).1.len();
                expect(
                    spawns * 100 >= kicks * 85 && spawns <= kicks + 2,
                    &format!("ring spawns {spawns} vs {kicks} kicks"),
                );
            }
            "buildup" => {
                let early = ticks[120].params.energy;
                expect(last.energy > early + 0.1, "buildup must raise intensity");
            }
            _ => {}
        }
        let _ = SLOTS;
    }
    if !gpu_seen {
        println!(
            "NOTE: no hardware GPU device available; rendered-pixel checks were skipped (parameter checks still ran)."
        );
    }
    if failures.is_empty() {
        println!("SELFTEST RESULT: ok");
        0
    } else {
        for f in &failures {
            eprintln!("SELFTEST FAILURE: {f}");
        }
        eprintln!("SELFTEST RESULT: FAILED");
        1
    }
}
