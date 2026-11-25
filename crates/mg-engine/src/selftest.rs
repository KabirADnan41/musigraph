//! `--scene-selftest`: deterministic end-to-end validation of the music-to-visual pipeline with SYNTHETIC
//! fixtures: samples -> real Analyzer -> SceneMapper -> SceneParams -> (headless GPU render, if a hardware
//! device exists) -> measured pixels. Prints measurements and exits non-zero if an expectation fails.
//! It never opens audio devices or windows.

use crate::scene_map::MapTuning;
use crate::sim::{FIXTURES, Sim, Tick, fixture};
use mg_gfx::context::{BackendChoice, Headless, init_headless_order};
use mg_gfx::scene::{SLOTS, SceneKind, SceneSet};

const SR: u32 = 48_000;
const FPS: f32 = 60.0;
const RES: (u32, u32) = (160, 90);

/// Hardware device for the measurements. `MG_SELFTEST_BACKEND=vulkan|dx12|gl` restricts it to one backend
/// (to verify the shader on every backend); default is the automatic order.
fn headless() -> Result<Headless, String> {
    match std::env::var("MG_SELFTEST_BACKEND")
        .ok()
        .as_deref()
        .and_then(BackendChoice::parse)
    {
        Some(b) => init_headless_order(&[b]),
        None => init_headless_order(&mg_gfx::context::AUTO_ORDER),
    }
}

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
    let h = headless().ok()?;
    let scene = SceneSet::new_headless(&h.device);
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

/// Composes the real presentation pipeline (demo tracks -> decode -> palette -> text bitmaps -> overlay) over
/// a rendered scene, headless. Checks that artwork and text appear, that different albums give different
/// colour themes, and (with `MG_SCENE_DUMP=<dir>`) writes PNGs for visual inspection. Returns whether a GPU
/// was available.
fn presentation_previews(failures: &mut Vec<String>) -> bool {
    use crate::media_feed::demo_tracks;
    use crate::presentation::{
        GpuOp, Layout, Presentation, apply_album, process_art, process_text,
    };
    use mg_gfx::overlay::Overlay;
    let Ok(h) = headless() else { return false };
    let Some(text) = mg_text::TextRenderer::new() else {
        failures.push("presentation: DirectWrite unavailable".into());
        return true;
    };
    let res = (960u32, 540u32);
    let x = fixture("demo", SR).expect("demo fixture");
    let ticks = Sim::new(SR, FPS, MapTuning::default()).run(&x[..x.len().min(SR as usize * 12)]);
    let base_params = ticks.last().expect("ticks").params.clone();
    let scene = SceneSet::new_headless(&h.device);
    let dump = std::env::var("MG_SCENE_DUMP").ok();
    let mut themes: Vec<(String, [f32; 3])> = Vec::new();
    for (kind, layout) in SceneKind::ALL
        .into_iter()
        .flat_map(|k| [Layout::Corner, Layout::Center].map(move |l| (k, l)))
    {
        for (i, n) in demo_tracks().iter().enumerate() {
            if kind != SceneKind::Spectrum && i > 2 {
                continue; // the other scenes are previewed on three albums to keep the self-test quick
            }
            let mut ov = Overlay::new_headless(&h.device, &h.queue);
            let mut pres = Presentation::new(layout);
            pres.on_now_playing(Some(n), res);
            for job in pres.take_jobs() {
                let r = match job {
                    crate::presentation::Job::Art { track_id, bytes } => {
                        process_art(track_id, &bytes)
                    }
                    crate::presentation::Job::Text(t) => process_text(&text, &t),
                };
                pres.on_ready(r);
            }
            let mut last = None;
            for _ in 0..300 {
                last = Some(pres.update(1.0 / 60.0, res, base_params.energy, base_params.swell));
                for op in pres.take_ops() {
                    match op {
                        GpuOp::SetArt(a) => {
                            ov.set_art(&h.device, &h.queue, (a.width, a.height), &a.rgba);
                        }
                        GpuOp::ClearArt => ov.clear_art(&h.device),
                        GpuOp::SetText(k, b) => {
                            ov.set_text(&h.device, &h.queue, k, (b.width, b.height), &b.coverage);
                        }
                        GpuOp::ClearText(k) => ov.clear_text(&h.device, k),
                    }
                }
            }
            let op = last.expect("frames");
            let mut p = base_params.clone();
            let (album, w) = pres.album();
            apply_album(&mut p.palette, &album, w, 0.85 + 0.30 * p.energy);
            if let Some(a) = op.art {
                let (w, hh) = (res.0 as f32, res.1 as f32);
                p.focus = [a.cx / w, a.cy / hh, a.hw / hh, a.hh / hh];
                p.focus_active = a.alpha;
            }
            let bare =
                scene.render_offscreen_scenes(&h.device, &h.queue, res, &p, kind, None, &|_, _| {});
            let img = scene.render_offscreen_scenes(
                &h.device,
                &h.queue,
                res,
                &p,
                kind,
                None,
                &|enc, view| ov.draw(&h.queue, enc, view, &op),
            );
            let (Ok(bare), Ok(img)) = (bare, img) else {
                failures.push(format!("presentation {layout:?}/{i}: render failed"));
                continue;
            };
            let changed = img.iter().zip(&bare).filter(|(a, b)| a != b).count();
            if n.art.is_some() && op.art.is_none() {
                failures.push(format!("presentation {layout:?}/{i}: artwork not shown"));
            }
            if op.title.is_none() || changed < 300 {
                failures.push(format!(
                    "presentation {layout:?}/{i}: title/text not drawn (changed bytes {changed})"
                ));
            }
            // mean colour of the bar band: the album theme as it actually renders
            let (mut s, mut cnt) = ([0.0f32; 3], 0.0f32);
            for y in 200..300u32 {
                for xx in 300..660u32 {
                    let k = ((y * res.0 + xx) * 4) as usize;
                    for c in 0..3 {
                        s[c] += f32::from(bare[k + c]);
                    }
                    cnt += 1.0;
                }
            }
            let m = s.map(|v| v / cnt / 255.0);
            if kind == SceneKind::Spectrum && layout == Layout::Corner {
                themes.push((format!("track {i}"), m));
            }
            let lum: f32 =
                bare.iter().map(|&v| f32::from(v)).sum::<f32>() / bare.len() as f32 / 255.0;
            if !(0.002..0.6).contains(&lum) {
                failures.push(format!(
                    "{} {layout:?}/{i}: mean luminance {lum:.3} out of range",
                    kind.name()
                ));
            }
            println!("PRESENT {} {layout:?} track {i}:", kind.name());
            println!(
                "  detail: art={} title={} artist={} changed_bytes={changed} mean_rgb=({:.3},{:.3},{:.3})",
                op.art.is_some(),
                op.title.is_some(),
                op.artist.is_some(),
                m[0],
                m[1],
                m[2]
            );
            if let Some(dir) = &dump {
                let png = mg_media::synth::png_rgba(res.0, res.1, &img);
                let _ = std::fs::write(
                    format!("{dir}/present-{}-{layout:?}-{i}.png", kind.name()).to_lowercase(),
                    png,
                );
            }
        }
    }
    for a in 0..4 {
        for b in a + 1..4 {
            let d: f32 = (0..3)
                .map(|c| (themes[a].1[c] - themes[b].1[c]).abs())
                .sum();
            if d < 0.02 {
                failures.push(format!(
                    "{} and {} render with nearly the same colour theme ({d:.3})",
                    themes[a].0, themes[b].0
                ));
            }
        }
    }
    true
}

pub fn run(allow_no_gpu: bool) -> i32 {
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
            if let Ok(h) = headless() {
                let scene = SceneSet::new_headless(&h.device);
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
            ml.is_nan() || (ml.is_finite() && ml > 0.001),
            "rendered image is black or NaN",
        );
        expect(
            ld.is_nan() || ld < 0.10,
            "luminance changed by >= 0.10 within 100 ms",
        );
        match name {
            "silence" => {
                expect(
                    cl.is_nan() || (cl < 0.08 && el < 0.08 && ml < 0.05),
                    "silence must render as a calm, dark, flat scene",
                );
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
    gpu_seen |= presentation_previews(&mut failures);
    if !gpu_seen {
        println!(
            "NOTE: no hardware GPU device available; rendered-pixel checks were NOT run (parameter checks ran)."
        );
        if !allow_no_gpu {
            eprintln!(
                "SELFTEST RESULT: FAILED (no hardware GPU; pass --allow-no-gpu to run parameter checks only)"
            );
            return 3;
        }
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
