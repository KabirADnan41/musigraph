//! Scene-transition checks of `--scene-selftest` (headless, real `SceneSwitch` + shaders on the selected backend).
//! Per ordered scene pair and pace (QUICK, FULL tour, REDUCED): never a blank frame, bounded per-frame luminance
//! change, at most three luminance swings per second, first frame = source idle, last frame = destination idle,
//! one scene per frame (lifecycle), bridge frames flat (no scene content). Plus: the overlay is not faded by a
//! transition, 200 random transitions leave GPU memory flat, and `MG_SCENE_DUMP=<dir>` writes filmstrip PNGs.
//! Luminance is the mean RELATIVE luminance of the raw (linear) headless target, the quantity WCAG 2.3.1 uses.

use crate::scene_switch::{FULL, Frame, Phase, SceneSwitch, scene_luma};
use crate::selftest::{headless, srgb_view};
use mg_gfx::context::{Headless, allocated_bytes};
use mg_gfx::overlay::{Overlay, OverlayParams};
use mg_gfx::scene::{Fx, SceneKind, SceneMode, SceneParams, SceneSet};

const RES: (u32, u32) = (320, 180);
const DT: f32 = 1.0 / 60.0;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Pace {
    Quick,
    Full,
    Reduced,
}

fn y_of(img: &[u8]) -> f32 {
    let s: f64 = img
        .chunks_exact(4)
        .map(|p| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]))
        .sum();
    (s / (img.len() / 4) as f64 / 255.0) as f32
}

fn y_std(img: &[u8]) -> f32 {
    let m = f64::from(y_of(img));
    let v: f64 = img
        .chunks_exact(4)
        .map(|p| {
            let y =
                (0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]))
                    / 255.0;
            (y - m) * (y - m)
        })
        .sum();
    (v / (img.len() / 4) as f64).sqrt() as f32
}

fn mad(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(x.abs_diff(*y)))
        .sum::<f64>()
        / a.len() as f64
}

/// Swing indices of a luminance series: reversals of at least `amp` (same definition as the switch's guard).
fn swings(ys: &[f32], amp: f32) -> Vec<usize> {
    let (mut ext, mut dir, mut at) = (ys[0], 0i8, Vec::new());
    for (i, &y) in ys.iter().enumerate() {
        let d = f32::from(dir);
        if dir == 0 {
            if (y - ext).abs() > amp {
                dir = if y > ext { 1 } else { -1 };
                ext = y;
            }
        } else if (y - ext) * d > 0.0 {
            ext = y;
        } else if (ext - y) * d > amp {
            at.push(i);
            dir = -dir;
            ext = y;
        }
    }
    at
}

struct Ctx<'a> {
    h: &'a Headless,
    scene: SceneSet,
    p: SceneParams,
}

impl Ctx<'_> {
    fn render(
        &mut self,
        k: SceneKind,
        fx: Fx,
        ov: Option<(&Overlay, &OverlayParams)>,
    ) -> Option<Vec<u8>> {
        self.p.fx = fx;
        self.scene
            .prepare(&self.h.device, &self.h.queue, RES, &self.p, k);
        let queue = self.h.queue.clone();
        self.scene
            .render_offscreen_scenes(
                &self.h.device,
                &self.h.queue,
                RES,
                &self.p,
                k,
                &|enc, view| {
                    if let Some((ov, op)) = ov {
                        ov.draw(&queue, enc, view, op);
                    }
                },
            )
            .ok()
    }

    /// The lifecycle rule after drawing `k`: only that scene holds Fluid / Focus resources.
    fn lifecycle_ok(&self, k: SceneKind) -> bool {
        self.scene.fluid_grid().is_some() == (k == SceneKind::Fluid)
            && self.scene.focus_room_alive() == (k == SceneKind::Focus)
    }
}

fn switch_for(from: SceneKind, to: SceneKind, pace: Pace) -> SceneSwitch {
    let mut s = SceneSwitch::new(from);
    s.set_reduced(pace == Pace::Reduced);
    if pace == Pace::Quick {
        s.request(to);
    } else {
        s.request_with(to, FULL);
    }
    s
}

struct Run {
    frames: Vec<Frame>,
    imgs: Vec<Vec<u8>>,
    lifecycle_ok: bool,
}

fn run_pair(c: &mut Ctx, from: SceneKind, to: SceneKind, pace: Pace) -> Option<Run> {
    let mut s = switch_for(from, to, pace);
    let pal = c.p.palette;
    let (mut frames, mut imgs, mut lifecycle_ok) = (Vec::new(), Vec::new(), true);
    for _ in 0..600 {
        let f = s.update(DT, |k| scene_luma(k, &pal));
        imgs.push(c.render(f.scene, f.fx(), None)?);
        lifecycle_ok &= c.lifecycle_ok(f.scene);
        frames.push(f);
        if f.phase == Phase::Idle {
            break;
        }
    }
    Some(Run {
        frames,
        imgs,
        lifecycle_ok,
    })
}

fn dump_strip(dir: &str, name: &str, run: &Run) {
    let n = 12usize.min(run.imgs.len());
    let pick: Vec<&Vec<u8>> = (0..n)
        .map(|i| &run.imgs[i * (run.imgs.len() - 1) / (n - 1).max(1)])
        .collect();
    let (w, h) = (RES.0 as usize, RES.1 as usize);
    let mut out = vec![0u8; w * n * h * 4];
    for (t, img) in pick.iter().enumerate() {
        let img = srgb_view(img);
        for y in 0..h {
            let dst = (y * w * n + t * w) * 4;
            out[dst..dst + w * 4].copy_from_slice(&img[y * w * 4..(y + 1) * w * 4]);
        }
    }
    let png = mg_media::synth::png_rgba((w * n) as u32, h as u32, &out);
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(format!("{dir}/transition-{name}.png"), png);
}

/// Returns whether a GPU was available.
pub fn transition_checks(failures: &mut Vec<String>) -> bool {
    let Ok(h) = headless() else { return false };
    let x = crate::sim::fixture("demo", 48_000).expect("demo fixture");
    let ticks = crate::sim::Sim::new(48_000, 60.0, crate::scene_map::MapTuning::default())
        .run(&x[..x.len().min(48_000 * 12)]);
    let mut p = ticks.last().expect("ticks").params.clone();
    p.mode = SceneMode::Live;
    let mut c = Ctx {
        h: &h,
        scene: SceneSet::new_headless(&h.device),
        p,
    };
    let dump = std::env::var("MG_SCENE_DUMP").ok();
    // idle frame and luminance of every scene (the reference for first/last frames and the bridge bounds)
    let mut idle = Vec::new();
    for k in SceneKind::ALL {
        let Some(img) = c.render(k, Default::default(), None) else {
            failures.push(format!("transition: idle render of {k:?} failed"));
            return true;
        };
        println!(
            "TRANSITION idle {k:?}: measured Y={:.4} (palette estimate {:.4})",
            y_of(&img),
            scene_luma(k, &c.p.palette)
        );
        idle.push(img);
    }
    let mut worst = (0.0f32, 0.0f32);
    let mut count = 0;
    for pace in [Pace::Quick, Pace::Full, Pace::Reduced] {
        for from in SceneKind::ALL {
            for to in SceneKind::ALL {
                if from == to {
                    continue;
                }
                let name = format!("{from:?}->{to:?}/{pace:?}");
                let Some(r) = run_pair(&mut c, from, to, pace) else {
                    failures.push(format!("transition {name}: render failed"));
                    continue;
                };
                count += 1;
                let ys: Vec<f32> = r.imgs.iter().map(|i| y_of(i)).collect();
                let (ysrc, ydst) = (y_of(&idle[from.index()]), y_of(&idle[to.index()]));
                let min_y = ys.iter().copied().fold(f32::MAX, f32::min);
                let max_dl = ys
                    .windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .fold(0.0, f32::max);
                let sw = swings(&ys, 0.1);
                let per_s = sw
                    .iter()
                    .map(|&i| sw.iter().filter(|&&j| j >= i && j < i + 60).count())
                    .max()
                    .unwrap_or(0);
                let floor = 0.5 * ysrc.min(ydst).min(0.04);
                let first = mad(&r.imgs[0], &idle[from.index()]);
                let last_f = r.frames.last().expect("frames");
                let last = r.imgs.last().expect("imgs");
                let ref_last = c
                    .render(last_f.scene, Default::default(), None)
                    .unwrap_or_default();
                let flat = r
                    .frames
                    .iter()
                    .zip(&r.imgs)
                    .filter(|(f, _)| f.phase == Phase::Bridge)
                    .map(|(_, i)| y_std(i))
                    .fold(0.0, f32::max);
                let bridge_y = r
                    .frames
                    .iter()
                    .zip(&ys)
                    .filter(|(f, _)| f.phase == Phase::Bridge)
                    .map(|(_, &y)| y)
                    .fold(0.0, f32::max);
                worst = (worst.0.max(max_dl), worst.1.max(per_s as f32));
                let secs = r.frames.len() as f32 * DT;
                let mut bad = Vec::new();
                if r.frames.last().map(|f| f.phase) != Some(Phase::Idle) || last_f.scene != to {
                    bad.push("did not end idle on the destination".to_string());
                }
                if min_y.is_nan() || min_y < floor {
                    bad.push(format!("blank frame: min Y {min_y:.4} < {floor:.4}"));
                }
                if max_dl > 0.105 {
                    bad.push(format!("per-frame luminance step {max_dl:.3} > 0.1"));
                }
                if per_s > 3 {
                    bad.push(format!("{per_s} luminance swings within 1 s"));
                }
                if first > 1.5 {
                    bad.push(format!(
                        "first frame differs from the source idle frame (mad {first:.2})"
                    ));
                }
                if mad(last, &ref_last) != 0.0 || last_f.fx() != Default::default() {
                    bad.push("last frame is not the destination idle frame".into());
                }
                if flat > 0.06 {
                    bad.push(format!(
                        "bridge frame not flat (std {flat:.3}): scene content leaks"
                    ));
                }
                if bridge_y > (1.25 * ysrc.min(ydst)).max(0.06) {
                    bad.push(format!(
                        "bridge Y {bridge_y:.3} brighter than the scenes ({ysrc:.3}, {ydst:.3})"
                    ));
                }
                if !r.lifecycle_ok {
                    bad.push("a heavy scene's resources were held while it was not drawn".into());
                }
                if pace == Pace::Reduced
                    && r.frames.iter().any(|f| {
                        f.iris != 0.0
                            || f.breathing != 0.0
                            || f.veil.iter().any(|&v| v != f.veil[0])
                    })
                {
                    bad.push("reduced motion used stagger, iris or breathing".into());
                }
                for b in &bad {
                    failures.push(format!("transition {name}: {b}"));
                }
                if let (Some(dir), true) = (
                    &dump,
                    matches!(
                        (from, to, pace),
                        (SceneKind::Spectrum, SceneKind::Orb, Pace::Quick)
                            | (SceneKind::Focus, SceneKind::Fluid, Pace::Quick)
                            | (SceneKind::Aurora, SceneKind::Spectrum, Pace::Full)
                            | (SceneKind::Orb, SceneKind::Focus, Pace::Reduced)
                    ),
                ) {
                    dump_strip(dir, &name.replace(['>', '/'], "-").to_lowercase(), &r);
                }
                if matches!(
                    (from, to),
                    (SceneKind::Spectrum, SceneKind::Orb) | (SceneKind::Focus, SceneKind::Fluid)
                ) {
                    println!(
                        "TRANSITION {name}: {secs:.2}s min_Y={min_y:.3} max_dY={max_dl:.3} swings/s={per_s} bridge_Y={bridge_y:.3} bridge_std={flat:.3} first_mad={first:.2} {}",
                        if bad.is_empty() { "ok" } else { "FAILED" }
                    );
                }
            }
        }
    }
    println!(
        "TRANSITION {count} pair runs: worst per-frame dY {:.3}, worst swings/s {}",
        worst.0, worst.1
    );
    overlay_check(&mut c, failures);
    churn_check(&mut c, failures);
    timing_report(&mut c);
    true
}

/// The overlay is drawn after the scene and never fades: strong overlay pixels (icons, text) over every frame of a
/// transition match those over the idle frame.
fn overlay_check(c: &mut Ctx, failures: &mut Vec<String>) {
    use crate::controls::{Availability, ControlsUi, FrameInput, Toggles};
    use crate::presentation::{GpuOp, process_job};
    use mg_media::Capabilities;
    let Some(text) = mg_text::TextRenderer::new() else {
        failures.push("transition overlay: DirectWrite unavailable".into());
        return;
    };
    let av = Availability {
        caps: Capabilities::all(),
        state: mg_media::PlaybackState::Playing,
        has_timeline: true,
    };
    let mut ui = ControlsUi::new(true);
    ui.set_view(RES, 1.0);
    ui.key_activity();
    let mut ov = Overlay::new_headless(&c.h.device, &c.h.queue);
    let input = |dt| FrameInput {
        dt,
        av,
        toggles: Toggles {
            lyrics: true,
            artwork: true,
            fullscreen: false,
        },
        position: Some(83.0),
        duration: Some(212.0),
        fg: [0.9, 0.9, 0.92],
        tint: [0.18, 0.38, 0.85],
    };
    let mut draw = None;
    for _ in 0..45 {
        draw = ui.frame(&input(DT));
        for job in ui.take_jobs() {
            ui.on_ready(process_job(Some(&text), job));
        }
        for op in ui.take_ops() {
            if let GpuOp::SetText(k, b) = op {
                ov.set_text(&c.h.device, &c.h.queue, k, (b.width, b.height), &b.coverage);
            }
        }
    }
    let Some(draw) = draw else {
        failures.push("transition overlay: nothing to draw".into());
        return;
    };
    let op = OverlayParams {
        res: RES,
        shadow: 0.5,
        accent: [0.18, 0.38, 0.85],
        controls: Some(draw),
        ..Default::default()
    };
    // core icon and text pixels: at least 150 levels above the same frame without the overlay (the scenes are dark,
    // the overlay is near white, so antialiased edges are excluded and the count is independent of the backdrop)
    let strong = |with: &[u8], bare: &[u8]| {
        with.chunks_exact(4)
            .zip(bare.chunks_exact(4))
            .filter(|(a, b)| a.iter().zip(*b).any(|(x, y)| x.abs_diff(*y) > 150))
            .count()
    };
    let mut s = switch_for(SceneKind::Spectrum, SceneKind::Orb, Pace::Quick);
    let pal = c.p.palette;
    let mut counts = Vec::new();
    let mut first = true;
    for _ in 0..600 {
        let f = s.update(DT, |k| scene_luma(k, &pal));
        let (Some(with), Some(bare)) = (
            c.render(f.scene, f.fx(), Some((&ov, &op))),
            c.render(f.scene, f.fx(), None),
        ) else {
            failures.push("transition overlay: render failed".into());
            return;
        };
        let n = strong(&with, &bare);
        if first {
            println!("TRANSITION overlay: {n} strong overlay pixels on the first frame");
            first = false;
        }
        counts.push(n);
        if f.phase == Phase::Idle {
            break;
        }
    }
    let (lo, hi) = (
        counts.iter().min().copied().unwrap_or(0),
        counts.iter().max().copied().unwrap_or(0),
    );
    println!(
        "TRANSITION overlay: strong pixels min {lo} max {hi} over {} frames",
        counts.len()
    );
    // bright icons over the dim bridge/scene differ from it by more than 100 levels in every frame
    if lo < 200 || (hi - lo) * 4 > hi {
        failures.push(format!(
            "transition overlay: overlay pixels changed during the transition ({lo}..{hi})"
        ));
    }
}

/// 200 random transitions (requests at random moments, retargets included): the drawn-scene lifecycle holds every
/// frame and GPU memory does not grow.
/// Brings the switch to rest on a scene that holds no heavy resources and returns the allocator bytes there, so two
/// samples are comparable (Fluid / Focus hold ~1.6 MB at this size while they are drawn).
fn settled_bytes(c: &mut Ctx, s: &mut SceneSwitch) -> Option<u64> {
    let pal = c.p.palette;
    s.request(SceneKind::Spectrum);
    for _ in 0..600 {
        let f = s.update(DT, |k| scene_luma(k, &pal));
        c.render(f.scene, f.fx(), None);
        if f.phase == Phase::Idle {
            break;
        }
    }
    mg_gfx::context::wait_idle(&c.h.device);
    allocated_bytes(&c.h.device)
}

fn churn_check(c: &mut Ctx, failures: &mut Vec<String>) {
    let mut s = SceneSwitch::new(SceneKind::Spectrum);
    let pal = c.p.palette;
    let (mut rng, mut frames, mut bad, mut warm) = (99u32, 0u32, 0u32, None);
    for i in 0..240 {
        if i == 40 {
            // the first 40 requests warm the allocator pools; growth is judged after that
            warm = settled_bytes(c, &mut s);
        }
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        s.request(SceneKind::ALL[(rng >> 16) as usize % SceneKind::ALL.len()]);
        for _ in 0..(3 + (rng >> 27)) {
            let f = s.update(DT, |k| scene_luma(k, &pal));
            if c.render(f.scene, f.fx(), None).is_none() || !c.lifecycle_ok(f.scene) {
                bad += 1;
            }
            frames += 1;
        }
    }
    let now = settled_bytes(c, &mut s);
    println!(
        "TRANSITION churn: 240 requests ({frames} frames, 200 after warm-up), bad frames {bad}, GPU allocator bytes at rest {warm:?} -> {now:?}"
    );
    if bad > 0 {
        failures.push(format!(
            "transition churn: {bad} frames failed to render or broke the lifecycle"
        ));
    }
    if let (Some(w), Some(n)) = (warm, now) {
        if n > w + w / 10 + (1 << 18) {
            failures.push(format!("transition churn: GPU memory grew {w} -> {n}"));
        }
    }
}

/// Informational, never a failure. NOT GPU time (the renderer has no timestamp queries): CPU cost of the pure state
/// machine, and wall-clock submit-to-readback of one offscreen frame per phase at 1280x720 (readback included).
fn timing_report(c: &mut Ctx) {
    let pal = c.p.palette;
    let mut s = SceneSwitch::new(SceneKind::Spectrum);
    s.request(SceneKind::Orb);
    let t = std::time::Instant::now();
    let n = 200_000;
    for i in 0..n {
        if i % 90 == 0 {
            s.request(SceneKind::ALL[i / 90 % 5]);
        }
        std::hint::black_box(s.update(DT, |k| scene_luma(k, &pal)));
    }
    println!(
        "TRANSITION cpu: SceneSwitch::update {:.0} ns/call (not GPU time)",
        t.elapsed().as_nanos() as f64 / n as f64
    );
    let big = (1280u32, 720u32);
    let mut s = switch_for(SceneKind::Spectrum, SceneKind::Orb, Pace::Quick);
    let mut by_phase: [Vec<f64>; 4] = Default::default();
    let idle_ms = {
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let _ = c.scene.render_offscreen_scenes(
                &c.h.device,
                &c.h.queue,
                big,
                &c.p,
                SceneKind::Spectrum,
                &|_, _| {},
            );
        }
        t.elapsed().as_secs_f64() * 100.0
    };
    for _ in 0..600 {
        let f = s.update(DT, |k| scene_luma(k, &pal));
        c.p.fx = f.fx();
        c.scene.prepare(&c.h.device, &c.h.queue, big, &c.p, f.scene);
        let t = std::time::Instant::now();
        let _ = c.scene.render_offscreen_scenes(
            &c.h.device,
            &c.h.queue,
            big,
            &c.p,
            f.scene,
            &|_, _| {},
        );
        by_phase[f.phase as usize].push(t.elapsed().as_secs_f64() * 1000.0);
        if f.phase == Phase::Idle {
            break;
        }
    }
    let line = |n: &str, v: &Vec<f64>| {
        let mut v = v.clone();
        v.sort_by(f64::total_cmp);
        println!(
            "TRANSITION wall-clock {n}: mean {:.2} ms max {:.2} ms ({} frames)",
            v.iter().sum::<f64>() / v.len().max(1) as f64,
            v.last().copied().unwrap_or(0.0),
            v.len()
        );
    };
    println!(
        "TRANSITION wall-clock idle Spectrum 1280x720: mean {idle_ms:.2} ms (readback included)"
    );
    line("exit", &by_phase[Phase::Exit as usize]);
    line("bridge", &by_phase[Phase::Bridge as usize]);
    line("enter", &by_phase[Phase::Enter as usize]);
}
