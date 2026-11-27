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

/// Orb: largest mean-luminance change within any 100 ms over the same tick series (flash/flicker guard).
/// `None` only without a GPU; a render error on a working GPU gives `Some(NaN)`, which fails the check.
fn orb_flicker(ticks: &[Tick]) -> Option<f32> {
    let h = headless().ok()?;
    let scene = SceneSet::new_headless(&h.device);
    let mut lum = Vec::with_capacity(ticks.len());
    for t in ticks {
        let Ok(img) = scene.render_offscreen_scenes(
            &h.device,
            &h.queue,
            RES,
            &t.params,
            SceneKind::Orb,
            None,
            &|_, _| {},
        ) else {
            return Some(f32::NAN);
        };
        lum.push(region_lum(&img, 0, RES.0, 0, RES.1));
    }
    Some(max_delta_100ms(&lum))
}

/// Largest change (max - min) of mean frame luminance within any 100 ms window: a flash that rises and falls
/// back inside the window counts, not only the difference between its endpoints.
fn max_delta_100ms(lum: &[f32]) -> f32 {
    let w = (FPS * 0.1).round() as usize;
    lum.windows(w + 1)
        .map(|s| {
            let (lo, hi) = s
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            hi - lo
        })
        .fold(0.0, f32::max)
}

/// Composes the real presentation pipeline (demo tracks -> decode -> palette -> text bitmaps -> overlay) over
/// a rendered scene, headless. Checks that artwork and text appear, that different albums give different
/// colour themes, and (with `MG_SCENE_DUMP=<dir>`) writes PNGs for visual inspection. Returns whether a GPU
/// was available.
fn presentation_previews(failures: &mut Vec<String>) -> bool {
    use crate::media_feed::demo_tracks;
    use crate::presentation::{GpuOp, Layout, Presentation, apply_album};
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
            pres.set_lyric_source(crate::lyric_view::LyricSource::Demo);
            pres.set_album_scene(kind == SceneKind::Focus, res);
            pres.set_bar(
                res.1 as f32 - 128.0 * crate::controls::ui_scale(res, 1.0),
                0.0,
            );
            pres.set_lyric_anchor(match kind {
                SceneKind::Spectrum => 0.70,
                SceneKind::Aurora => 0.72,
                SceneKind::Orb | SceneKind::Fluid => 0.72,
                SceneKind::Focus if layout == Layout::Center => 0.80,
                SceneKind::Focus => 0.72,
            });
            pres.on_now_playing(Some(n), res);
            for job in pres.take_jobs() {
                pres.on_ready(crate::presentation::process_job(Some(&text), job));
            }
            let mut last = None;
            let lyric_from: Option<f64> = std::env::var("MG_LYRIC_FROM")
                .ok()
                .and_then(|v| v.parse().ok());
            for frame in 0..300 {
                // MG_LYRIC_FROM: play the lyrics from that position (preview of a line change) instead of holding 10.5 s
                pres.set_position(Some(
                    lyric_from.map_or(10.5, |f| f + f64::from(frame) / 60.0),
                ));
                for job in pres.take_jobs() {
                    pres.on_ready(crate::presentation::process_job(Some(&text), job));
                }
                last = Some(pres.update(1.0 / 60.0, res, base_params.energy, base_params.swell));
                for op in pres.take_ops() {
                    match op {
                        GpuOp::SetArt(a) => {
                            ov.set_art(&h.device, &h.queue, (a.width, a.height), &a.rgba);
                        }
                        GpuOp::ShiftArt => ov.shift_art(&h.device),
                        GpuOp::ClearPrevArt => ov.clear_prev_art(&h.device),
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
            if i <= 2 && op.lyrics.iter().all(Option::is_none) {
                failures.push(format!(
                    "{} {layout:?}/{i}: synchronised lyrics not drawn",
                    kind.name()
                ));
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

/// Album Focus composition previews (the Focus scene with the Center layout) across window shapes, with and
/// without lyrics and cover, through the real pipeline (decode, palette, DirectWrite text, overlay). Checks that
/// the cover, title block and lyrics appear where expected; prints the text-job worker time; with
/// `MG_SCENE_DUMP=<dir>` writes `album-<w>x<h>-<lyrics>-<art>.png`. `MG_LYRIC_FROM=<s>` plays the lyrics from that
/// position (to catch a slide) instead of holding 10.5 s.
fn album_previews(failures: &mut Vec<String>) -> bool {
    use crate::lyric_view::LyricSource;
    use crate::media_feed::demo_tracks;
    use crate::presentation::{GpuOp, Job, Layout, Presentation, apply_album, process_job};
    use mg_gfx::overlay::Overlay;
    let Ok(h) = headless() else { return false };
    let Some(text) = mg_text::TextRenderer::new() else {
        failures.push("album: DirectWrite unavailable".into());
        return true;
    };
    println!(
        "ALBUM fonts: {}",
        if text.has_variable_cuts() {
            "Segoe UI Variable Display/Text"
        } else {
            "Segoe UI (fallback)"
        }
    );
    let x = fixture("demo", SR).expect("demo fixture");
    let ticks = Sim::new(SR, FPS, MapTuning::default()).run(&x[..x.len().min(SR as usize * 12)]);
    let base_params = ticks.last().expect("ticks").params.clone();
    let scene = SceneSet::new_headless(&h.device);
    let dump = std::env::var("MG_SCENE_DUMP").ok();
    let lyric_from: Option<f64> = std::env::var("MG_LYRIC_FROM")
        .ok()
        .and_then(|v| v.parse().ok());
    let tracks = demo_tracks();
    for res in [(960u32, 540u32), (1260, 540), (720, 540), (540, 960)] {
        for (lyrics, art) in [(true, true), (false, true), (true, false), (false, false)] {
            let mut n = tracks[0].clone();
            n.album = "Harbour Lights".into();
            if !art {
                n.art = None;
            }
            let mut ov = Overlay::new_headless(&h.device, &h.queue);
            let mut pres = Presentation::new(Layout::Center);
            pres.set_album_scene(true, res);
            pres.set_bar(
                res.1 as f32 - 128.0 * crate::controls::ui_scale(res, 1.0),
                0.0,
            );
            if lyrics {
                pres.set_lyric_source(LyricSource::Demo);
            }
            pres.on_now_playing(Some(&n), res);
            let (mut text_ms, mut last) = (0.0f64, None);
            for frame in 0..300 {
                pres.set_position(Some(
                    lyric_from.map_or(10.5, |f| f + f64::from(frame) / 60.0),
                ));
                for job in pres.take_jobs() {
                    let is_text = matches!(job, Job::Text(_));
                    let t0 = std::time::Instant::now();
                    let r = process_job(Some(&text), job);
                    if is_text {
                        text_ms += t0.elapsed().as_secs_f64() * 1000.0;
                    }
                    pres.on_ready(r);
                }
                last = Some(pres.update(1.0 / 60.0, res, base_params.energy, base_params.swell));
                for op in pres.take_ops() {
                    match op {
                        GpuOp::ShiftArt => ov.shift_art(&h.device),
                        GpuOp::SetArt(a) => {
                            ov.set_art(&h.device, &h.queue, (a.width, a.height), &a.rgba);
                        }
                        GpuOp::ClearArt => ov.clear_art(&h.device),
                        GpuOp::ClearPrevArt => ov.clear_prev_art(&h.device),
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
            let tag = format!(
                "album-{}x{}-{}-{}",
                res.0,
                res.1,
                if lyrics { "lyrics" } else { "nolyrics" },
                if art { "art" } else { "noart" }
            );
            let Ok(img) = scene.render_offscreen_scenes(
                &h.device,
                &h.queue,
                res,
                &p,
                SceneKind::Focus,
                None,
                &|enc, view| ov.draw(&h.queue, enc, view, &op),
            ) else {
                failures.push(format!("{tag}: render failed"));
                continue;
            };
            let drawn_lyrics = op.lyrics.iter().flatten().filter(|d| d.alpha > 0.5).count();
            if art != op.art.is_some_and(|a| a.alpha > 0.99) {
                failures.push(format!("{tag}: cover shown = {}", op.art.is_some()));
            }
            if op.title.is_none_or(|t| t.alpha < 0.99) || op.artist.is_none() || op.meta.is_none() {
                failures.push(format!("{tag}: title block incomplete"));
            }
            if lyrics != (drawn_lyrics > 0) {
                failures.push(format!("{tag}: {drawn_lyrics} lyric lines drawn"));
            }
            println!(
                "ALBUM {tag}: art={} title={:?} lyric_lines={drawn_lyrics} clip={:?} text_job_ms={text_ms:.1}",
                op.art
                    .map(|a| (a.cx as i32, a.cy as i32, (2.0 * a.hh) as i32))
                    .is_some(),
                op.title
                    .map(|t| (t.x as i32, t.y as i32, t.w as i32, t.h as i32)),
                op.lyric_clip.map(|c| (c[0] as i32, c[1] as i32))
            );
            if let Some(dir) = &dump {
                let png = mg_media::synth::png_rgba(res.0, res.1, &img);
                let _ = std::fs::write(format!("{dir}/{tag}.png"), png);
            }
        }
    }
    true
}

/// Control bar: enabled (demo caps, hover, lyrics toggled on), disabled (no media source) and keyboard-focus
/// states rendered headless over a real scene with DirectWrite time texts. Checks that the bar is drawn, that the
/// states differ where they should and that nothing is black/NaN; with `MG_SCENE_DUMP=<dir>` writes PNGs.
fn controls_previews(failures: &mut Vec<String>) -> bool {
    use crate::controls::{Availability, ButtonId, ControlsUi, FrameInput, Toggles};
    use crate::presentation::{GpuOp, process_job};
    use mg_gfx::overlay::{Overlay, OverlayParams};
    use mg_media::Capabilities;
    let Ok(h) = headless() else { return false };
    let Some(text) = mg_text::TextRenderer::new() else {
        failures.push("controls: DirectWrite unavailable".into());
        return true;
    };
    let res = (1280u32, 720u32);
    let x = fixture("demo", SR).expect("demo fixture");
    let ticks = Sim::new(SR, FPS, MapTuning::default()).run(&x[..x.len().min(SR as usize * 12)]);
    let base_params = ticks.last().expect("ticks").params.clone();
    let scene = SceneSet::new_headless(&h.device);
    let dump = std::env::var("MG_SCENE_DUMP").ok();
    let demo = Availability {
        caps: Capabilities::all(),
        state: mg_media::PlaybackState::Playing,
        has_timeline: true,
    };
    let off = Availability {
        caps: Capabilities::default(),
        state: mg_media::PlaybackState::Other,
        has_timeline: false,
    };
    // (name, availability, hover target, tab presses)
    let cases = [
        ("enabled", demo, Some(ButtonId::Next), 0),
        ("disabled", off, None, 0),
        ("focus", demo, None, 2),
    ];
    let bare = scene.render_offscreen_scenes(
        &h.device,
        &h.queue,
        res,
        &base_params,
        SceneKind::Spectrum,
        None,
        &|_, _| {},
    );
    let Ok(bare) = bare else {
        failures.push("controls: bare render failed".into());
        return true;
    };
    let mut images: Vec<(&str, Vec<u8>)> = Vec::new();
    for (name, av, hover, tabs) in cases {
        let mut ui = ControlsUi::new(true);
        ui.set_view(res, 1.0);
        ui.key_activity();
        if let Some(b) = hover {
            let r = ui.geometry().buttons[b.index()];
            ui.pointer_moved((r.x + 0.5 * r.w, r.y + 0.5 * r.h));
        }
        for _ in 0..tabs {
            ui.tab(false, &av);
        }
        let mut ov = Overlay::new_headless(&h.device, &h.queue);
        let input = |dt| FrameInput {
            dt,
            av,
            toggles: Toggles {
                lyrics: true,
                artwork: true,
                fullscreen: false,
            },
            position: av.has_timeline.then_some(83.0),
            duration: av.has_timeline.then_some(212.0),
            fg: [0.9, 0.9, 0.92],
            tint: [0.18, 0.38, 0.85],
        };
        let mut draw = None;
        for _ in 0..45 {
            draw = ui.frame(&input(1.0 / 60.0));
            // time texts go through the same job path as the live loop
            for job in ui.take_jobs() {
                ui.on_ready(process_job(Some(&text), job));
            }
            for op in ui.take_ops() {
                if let GpuOp::SetText(k, b) = op {
                    ov.set_text(&h.device, &h.queue, k, (b.width, b.height), &b.coverage);
                }
            }
        }
        let Some(draw) = draw else {
            failures.push(format!("controls {name}: nothing to draw"));
            continue;
        };
        let p = OverlayParams {
            res,
            shadow: 0.5,
            accent: [0.18, 0.38, 0.85],
            controls: Some(draw.clone()),
            ..Default::default()
        };
        let Ok(img) = scene.render_offscreen_scenes(
            &h.device,
            &h.queue,
            res,
            &base_params,
            SceneKind::Spectrum,
            None,
            &|enc, view| ov.draw(&h.queue, enc, view, &p),
        ) else {
            failures.push(format!("controls {name}: render failed"));
            continue;
        };
        let changed = img.iter().zip(&bare).filter(|(a, b)| a != b).count();
        let bright = img
            .chunks_exact(4)
            .any(|px| px[0].max(px[1]).max(px[2]) > 180);
        let opaque = img.chunks_exact(4).all(|px| px[3] == 255);
        if changed < 4000 || !bright || !opaque {
            failures.push(format!(
                "controls {name}: changed={changed} bright_icons={bright} opaque={opaque}"
            ));
        }
        if av.has_timeline && (draw.elapsed.is_none() || draw.duration.is_none()) {
            failures.push(format!("controls {name}: time texts missing"));
        }
        println!(
            "CONTROLS {name}: changed_bytes={changed} progress={} times={}",
            draw.progress.is_some(),
            draw.elapsed.is_some() && draw.duration.is_some()
        );
        if let Some(dir) = &dump {
            let png = mg_media::synth::png_rgba(res.0, res.1, &img);
            let _ = std::fs::write(format!("{dir}/controls-{name}.png"), png);
        }
        images.push((name, img));
    }
    if images.len() == 3 {
        let d = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();
        if d(&images[0].1, &images[1].1) < 500 || d(&images[0].1, &images[2].1) < 200 {
            failures.push("controls: enabled/disabled/focus states do not differ".into());
        }
    }
    true
}

/// Orb: real analyzer + mapper on the synthetic fixtures, rendered headless (Orb scene only). Returns whether a
/// GPU was available. Thresholds are on mean absolute byte difference (0..255) at 160x90 and sit well below the
/// measured differences but far above render noise (identical params give 0).
fn orb_checks(failures: &mut Vec<String>) -> bool {
    let Ok(h) = headless() else { return false };
    let scene = SceneSet::new_headless(&h.device);
    let render = |p: &mg_gfx::scene::SceneParams| {
        scene.render_offscreen_scenes(
            &h.device,
            &h.queue,
            RES,
            p,
            SceneKind::Orb,
            None,
            &|_, _| {},
        )
    };
    let mad = |a: &[u8], b: &[u8]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| u64::from(x.abs_diff(*y)))
            .sum::<u64>() as f64
            / a.len() as f64
    };
    let mut frames = Vec::new();
    for name in ["silence", "bass", "treble", "kick"] {
        let x = fixture(name, SR).expect("fixture");
        let ticks = Sim::new(SR, FPS, MapTuning::default()).run(&x[..x.len().min(SR as usize * 8)]);
        // kick: a frame 0.25 s into a ripple, so the travelling wave is on screen
        let pick = ticks
            .iter()
            .rev()
            .find(|t| t.params.rings[0][1] > 0.0 && (0.2..0.3).contains(&t.params.rings[0][0]))
            .unwrap_or_else(|| ticks.last().expect("ticks"));
        let p = pick.params.clone();
        let Ok(img) = render(&p) else {
            failures.push(format!("orb/{name}: render failed"));
            continue;
        };
        let orb = region_lum(&img, 80, 100, 30, 50);
        let corners = (region_lum(&img, 0, 20, 0, 15) + region_lum(&img, 140, 160, 75, 90)) / 2.0;
        println!(
            "ORB {name:8} bands={:.2?} ripple_age={:.2} orb_lum={orb:.3} corner_lum={corners:.3}",
            p.orb_band, p.rings[0][0]
        );
        if !(orb.is_finite() && orb > 0.02 && orb > corners + 0.02) {
            failures.push(format!(
                "orb/{name}: orb {orb:.3} must be brighter than corners {corners:.3}"
            ));
        }
        if let Ok(dir) = std::env::var("MG_SCENE_DUMP") {
            let big = scene.render_offscreen_scenes(
                &h.device,
                &h.queue,
                (960, 540),
                &p,
                SceneKind::Orb,
                None,
                &|_, _| {},
            );
            if let Ok(big) = big {
                let _ = std::fs::write(
                    format!("{dir}/orb-{name}.png"),
                    mg_media::synth::png_rgba(960, 540, &big),
                );
            }
        }
        if name == "kick" {
            let mut calm = p.clone();
            calm.rings = [[100.0, 0.0]; 2];
            if let Ok(b) = render(&calm) {
                let d = mad(&img, &b);
                println!("ORB kick ripple vs same frame without ripple: mad={d:.3}");
                if !(0.05..8.0).contains(&d) {
                    failures.push(format!(
                        "orb/kick: ripple difference {d:.3} outside 0.05..8 (no flash, but visible)"
                    ));
                }
            }
        }
        frames.push((name, img));
    }
    if let [(_, s), (_, b), (_, t), _] = &frames[..] {
        for (what, d, min) in [
            ("silence vs bass", mad(s, b), 1.0),
            ("silence vs treble", mad(s, t), 0.4),
            ("bass vs treble", mad(b, t), 0.5),
        ] {
            println!("ORB {what}: mad={d:.3} (min {min})");
            if d < min {
                failures.push(format!(
                    "orb: {what} frames too similar (mad {d:.3} < {min})"
                ));
            }
        }
        let (ls, lb) = (
            region_lum(s, 60, 100, 20, 60),
            region_lum(b, 60, 100, 20, 60),
        );
        if lb < ls * 1.1 {
            failures.push(format!(
                "orb: silence {ls:.3} must be dimmer than bass {lb:.3}"
            ));
        }
    }
    true
}

/// Fluid (F1, Experimental): the real analyzer + mapper on synthetic fixtures drive the dye simulation exactly as
/// the live loop does (fixed 1/60 s steps from the scene time), rendered headless at `RES` after every tick.
/// Checks: finite field, a flicker bound like Orb's, and silence / bass / treble frames that differ. With
/// `MG_SCENE_DUMP=<dir>` writes 960x540 PNGs; with `MG_FLUID_PERF=1` prints GPU ms per tier (sim step, 1080p draw).
fn fluid_checks(failures: &mut Vec<String>) -> bool {
    let Ok(h) = headless() else { return false };
    let mut scene = SceneSet::new_headless(&h.device);
    let mad = |a: &[u8], b: &[u8]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| u64::from(x.abs_diff(*y)))
            .sum::<u64>() as f64
            / a.len() as f64
    };
    let dump = std::env::var("MG_SCENE_DUMP").ok();
    let mut frames = Vec::new();
    for name in ["silence", "bass", "treble", "kick", "demo", "pad"] {
        let x = fixture(name, SR).expect("fixture");
        let ticks = Sim::new(SR, FPS, MapTuning::default()).run(&x[..x.len().min(SR as usize * 8)]);
        scene.fluid_release(); // every fixture starts from a clear field, like a scene entry
        let mut lum = Vec::with_capacity(ticks.len());
        let mut last = Vec::new();
        for t in &ticks {
            scene.fluid_advance(&h.device, &h.queue, RES, &t.params);
            let Ok(img) = scene.render_offscreen_scenes(
                &h.device,
                &h.queue,
                RES,
                &t.params,
                SceneKind::Fluid,
                None,
                &|_, _| {},
            ) else {
                failures.push(format!("fluid/{name}: render failed"));
                return true;
            };
            lum.push(region_lum(&img, 0, RES.0, 0, RES.1));
            last = img;
        }
        let finite = scene
            .fluid_dye(&h.device, &h.queue)
            .is_ok_and(|(_, _, d)| d.iter().all(|v| v.is_finite()));
        let flicker = max_delta_100ms(&lum);
        let upper = region_lum(&last, 0, RES.0, 0, 50);
        let lower = region_lum(&last, 0, RES.0, 62, RES.1);
        println!(
            "FLUID {name:8} finite={finite} max_lum_delta_100ms={flicker:.3} mean_lum={:.3} upper_lum={upper:.3} lower_lum={lower:.3}",
            mean(&lum)
        );
        if !finite {
            failures.push(format!("fluid/{name}: non-finite dye"));
        }
        if !(flicker.is_finite() && flicker < 0.10) {
            failures.push(format!(
                "fluid/{name}: luminance changed by {flicker:.3} within 100 ms"
            ));
        }
        if let Some(dir) = &dump {
            let p = &ticks.last().expect("ticks").params;
            if let Ok(big) = scene.render_offscreen_scenes(
                &h.device,
                &h.queue,
                (960, 540),
                p,
                SceneKind::Fluid,
                None,
                &|_, _| {},
            ) {
                let _ = std::fs::write(
                    format!("{dir}/fluid-{name}.png"),
                    mg_media::synth::png_rgba(960, 540, &big),
                );
            }
        }
        frames.push((name, last));
    }
    if let [(_, s), (_, b), (_, t), (_, k), _, (_, p)] = &frames[..] {
        for (what, d, min) in [
            ("silence vs bass", mad(s, b), 1.0),
            ("silence vs treble", mad(s, t), 0.5),
            ("bass vs treble", mad(b, t), 0.5),
            // FL1: impulsive kick, sustained pad and a steady bass must not look alike
            ("kick vs bass", mad(k, b), 0.5),
            ("kick vs pad", mad(k, p), 0.5),
            ("pad vs bass", mad(p, b), 0.5),
        ] {
            println!("FLUID {what}: mad={d:.3} (min {min})");
            if d < min {
                failures.push(format!(
                    "fluid: {what} frames too similar (mad {d:.3} < {min})"
                ));
            }
        }
    }
    if std::env::var("MG_FLUID_PERF").is_ok_and(|v| v == "1") {
        let x = fixture("demo", SR).expect("demo");
        let ticks = Sim::new(SR, FPS, MapTuning::default()).run(&x[..SR as usize * 8]);
        let p = &ticks.last().expect("ticks").params;
        for tier in 0..=3u8 {
            scene.set_fluid_tier(tier);
            scene.fluid_release();
            for t in &ticks {
                scene.fluid_advance(&h.device, &h.queue, (1920, 1080), &t.params);
            }
            match scene.fluid_gpu_ms(&h.device, &h.queue, (1920, 1080), p, 120) {
                Ok((sim, draw)) => println!(
                    "FLUID PERF tier {tier} grid {:?}: sim_step_ms={sim:.3} draw_1080p_ms={draw:.3} (headless, submit-to-completion, {})",
                    scene.fluid_grid(),
                    h.info.name
                ),
                Err(e) => failures.push(format!("fluid perf T{tier}: {e}")),
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
        let orb_ld = orb_flicker(&ticks);
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
        expect(
            orb_ld.is_none_or(|d| d.is_finite() && d < 0.10),
            "orb: render failed or luminance changed by >= 0.10 within 100 ms",
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
    gpu_seen |= album_previews(&mut failures);
    gpu_seen |= controls_previews(&mut failures);
    gpu_seen |= orb_checks(&mut failures);
    gpu_seen |= fluid_checks(&mut failures);
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

#[cfg(test)]
mod tests {
    use super::max_delta_100ms;

    #[test]
    fn flicker_counts_a_flash_inside_the_window_not_only_its_endpoints() {
        // a 0.2 flash that rises and falls back between two samples 100 ms apart
        let mut lum = vec![0.1f32; 20];
        lum[9] = 0.3;
        assert!((max_delta_100ms(&lum) - 0.2).abs() < 1e-6);
        assert_eq!(max_delta_100ms(&[0.1; 20]), 0.0);
        // a steady ramp of 0.01 per frame: 6 frames = 0.06 per 100 ms
        let ramp: Vec<f32> = (0..30).map(|i| i as f32 * 0.01).collect();
        assert!((max_delta_100ms(&ramp) - 0.06).abs() < 1e-5);
    }
}
