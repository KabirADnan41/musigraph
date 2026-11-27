//! Scene-panel thumbnails: one small deterministic render per scene, made once at runtime (nothing is baked
//! into the repository). Input is the SYNTHETIC "demo" fixture through the real analyzer and mapper, a fixed
//! tick, a fixed palette: no audio device, no RNG, no clock. The render uses a THROWAWAY `SceneSet`, so the
//! live Fluid simulation and Focus room are never touched; only the pixels leave this module.

use crate::scene_map::MapTuning;
use crate::sim::{Sim, Tick, fixture};
use mg_gfx::context::{Device, Queue};
use mg_gfx::overlay::{THUMB_CELLS, THUMB_SIZE};
use mg_gfx::scene::{SceneKind, SceneMode, SceneSet};

const SR: u32 = 48_000;
const FPS: f32 = 60.0;
/// Which fixture tick is photographed (5 s in: the field has developed, a beat has landed).
const TICK: usize = 300;

/// sRGB RGBA8 pixels (`THUMB_SIZE`) per scene index; None where a render failed (the panel shows a placeholder).
pub type Thumbs = [Option<Vec<u8>>; THUMB_CELLS];

fn ticks() -> Vec<Tick> {
    let x = fixture("demo", SR).expect("demo fixture");
    let n = (SR as usize * 6).min(x.len());
    Sim::new(SR, FPS, MapTuning::default()).run(&x[..n])
}

/// An original, procedural album cover (generated here, no image asset, no real artwork): a dusk gradient, soft
/// ripples and a low sun. Deterministic arithmetic only.
pub fn procedural_cover() -> mg_media::ArtImage {
    const N: u32 = 96;
    let rgba = (0..N * N)
        .flat_map(|i| {
            let (x, y) = (
                (i % N) as f32 / (N - 1) as f32,
                (i / N) as f32 / (N - 1) as f32,
            );
            let (dx, dy) = (x - 0.62, y - 0.68);
            let r = (dx * dx + dy * dy).sqrt();
            let sun = (1.0 - (r / 0.17).powi(2)).clamp(0.0, 1.0);
            let ripple = 0.5 + 0.5 * (r * 38.0).sin();
            let glow = (1.0 - r * 1.6).clamp(0.0, 1.0);
            let base = [0.10 + 0.38 * y, 0.07 + 0.10 * x, 0.30 + 0.30 * (1.0 - y)];
            let tint = [0.95, 0.52, 0.28];
            let c: [f32; 3] = std::array::from_fn(|k| {
                let v = base[k] * (0.82 + 0.18 * ripple) + tint[k] * (0.30 * glow + 0.70 * sun);
                v.clamp(0.0, 1.0)
            });
            [
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
                255,
            ]
        })
        .collect();
    mg_media::ArtImage {
        width: N,
        height: N,
        rgba,
    }
}

/// The Focus picture the way Album Focus shows it: the cover's room light and palette in the throwaway scene, the
/// cover plate over it. The live Focus scene is not involved.
fn render_focus(
    scene: &mut SceneSet,
    device: &Device,
    queue: &Queue,
    p: &mut mg_gfx::scene::SceneParams,
) -> Option<Vec<u8>> {
    use crate::presentation::{apply_album, art_extra};
    use mg_gfx::overlay::{ArtDraw, Overlay, OverlayParams};
    let cover = procedural_cover();
    scene.set_focus_art(queue, art_extra(&cover, 0).map(|e| *e.light));
    let (w, h) = (THUMB_SIZE.0 as f32, THUMB_SIZE.1 as f32);
    let aspect = w / h;
    // the Album layout's cover rectangle (centre x, centre y, half size in screen heights), as the self-test uses
    let rect = [0.31 * aspect, 0.46, 0.28, 0.28];
    p.album.rect = rect;
    p.album.art_w = [1.0, 0.0];
    let (px, tw, th) = cover.thumbnail(64);
    if tw > 0 && th > 0 {
        let pal = mg_media::palette::extract(&px, tw, th);
        apply_album(&mut p.palette, &pal, 1.0, 0.85 + 0.30 * p.energy);
    }
    scene.focus_acquire(device);
    let mut ov = Overlay::new(
        device,
        queue,
        mg_gfx::context::TextureFormat::Rgba8UnormSrgb,
    );
    if !ov.set_art(device, queue, (cover.width, cover.height), &cover.rgba) {
        return None;
    }
    let op = OverlayParams {
        res: THUMB_SIZE,
        art: Some(ArtDraw {
            cx: rect[0] * h,
            cy: rect[1] * h,
            hw: rect[2] * h,
            hh: rect[2] * h,
            alpha: 1.0,
            radius: 3.0,
            shadow: 1.0,
            lod: 0.0,
        }),
        ..Default::default()
    };
    let img = scene
        .render_offscreen_scenes(
            device,
            queue,
            THUMB_SIZE,
            p,
            SceneKind::Focus,
            None,
            &|enc, view| ov.draw(queue, enc, view, &op),
        )
        .ok();
    scene.set_focus_art(queue, None);
    img
}

/// Renders every scene with a throwaway scene set (dropped on return). Blocking GPU work with a readback per
/// scene, meant to run once before the first visible frame; wrap it in `context::with_error_scope`.
pub fn generate(device: &Device, queue: &Queue) -> Thumbs {
    let ticks = ticks();
    let mut scene = SceneSet::new(device, wgpu_srgb(), 1.0);
    let mut out: Thumbs = Default::default();
    let Some(last) = ticks.get(TICK.min(ticks.len().saturating_sub(1))) else {
        return out;
    };
    for kind in SceneKind::ALL {
        if kind == SceneKind::Fluid {
            // the dye field needs its history: replay the fixed ticks up to the photographed one
            for t in ticks.iter().take(TICK + 1) {
                scene.fluid_advance(device, queue, THUMB_SIZE, &t.params);
            }
        }
        let mut p = last.params.clone();
        p.mode = SceneMode::Live; // the Synthetic marker is for the live picture, not a preview
        out[kind.index()] = if kind == SceneKind::Focus {
            render_focus(&mut scene, device, queue, &mut p)
        } else {
            scene
                .render_offscreen_scenes(device, queue, THUMB_SIZE, &p, kind, None, &|_, _| {})
                .ok()
        };
        if kind == SceneKind::Fluid {
            scene.fluid_release();
        }
    }
    out
}

/// `generate` inside error scopes: a validation or out-of-memory error yields no thumbnails (placeholders)
/// instead of reaching the app's fatal uncaptured-error handler.
pub fn generate_guarded(device: &Device, queue: &Queue) -> Thumbs {
    guarded(device, || generate(device, queue))
}

fn guarded(device: &Device, work: impl FnOnce() -> Thumbs) -> Thumbs {
    let r = mg_gfx::context::with_error_scope(device, work);
    mg_gfx::context::wait_idle(device); // the throwaway scene set is dropped: let the GPU free it now
    match r {
        Ok(t) => t,
        Err(e) => {
            eprintln!("scene thumbnails unavailable: {e}");
            Thumbs::default()
        }
    }
}

fn wgpu_srgb() -> mg_gfx::context::TextureFormat {
    mg_gfx::context::TextureFormat::Rgba8UnormSrgb
}

#[cfg(test)]
mod tests {
    use super::*;
    use mg_gfx::context::init_headless;

    fn mean_luma(img: &[u8]) -> f32 {
        img.chunks_exact(4)
            .map(|p| f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2]))
            .sum::<f32>()
            / (img.len() / 4) as f32
            / 3.0
    }

    #[test]
    fn a_gpu_error_during_generation_is_caught_and_yields_placeholders_not_an_app_exit() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        // inject a real wgpu validation error inside the generation work: a zero-sized texture
        let t = guarded(&h.device, || {
            let _ = mg_gfx::context::invalid_texture(&h.device);
            generate(&h.device, &h.queue)
        });
        assert!(
            t.iter().all(Option::is_none),
            "the scoped error discards the thumbnails"
        );
        // and the device is still usable afterwards (nothing escaped to the fatal handler)
        let ok = guarded(&h.device, || generate(&h.device, &h.queue));
        assert!(ok.iter().all(Option::is_some));
    }

    #[test]
    fn generating_thumbnails_leaves_no_gpu_memory_behind_and_a_failure_degrades_to_placeholders() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        let before = mg_gfx::context::allocated_bytes(&h.device);
        let t = generate_guarded(&h.device, &h.queue);
        assert!(t.iter().all(Option::is_some));
        drop(t);
        mg_gfx::context::wait_idle(&h.device);
        if let (Some(b), Some(a)) = (before, mg_gfx::context::allocated_bytes(&h.device)) {
            eprintln!("GPU allocations: {b} before, {a} after generation (throwaway set released)");
            assert!(a <= b + 4 * 1024 * 1024, "{b} -> {a}");
        }
        // the cover is procedural, deterministic and not flat
        let c = procedural_cover();
        assert_eq!(c.rgba.len(), (c.width * c.height * 4) as usize);
        assert_eq!(c.rgba, procedural_cover().rgba);
        assert!(
            c.rgba.chunks_exact(4).any(|p| p[0] > 200) && c.rgba.chunks_exact(4).any(|p| p[0] < 80)
        );
    }

    #[test]
    fn thumbnails_are_non_blank_distinct_and_deterministic() {
        let Ok(h) = init_headless() else {
            eprintln!("SKIPPED (no hardware GPU)");
            return;
        };
        let t0 = std::time::Instant::now();
        let a = generate(&h.device, &h.queue);
        eprintln!(
            "thumbnail generation: {:.0} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        let b = generate(&h.device, &h.queue);
        if let Ok(dir) = std::env::var("MG_SCENE_DUMP") {
            for k in SceneKind::ALL {
                let png = mg_media::synth::png_rgba(
                    THUMB_SIZE.0,
                    THUMB_SIZE.1,
                    a[k.index()].as_ref().unwrap(),
                );
                let _ = std::fs::write(format!("{dir}/thumb-{}.png", k.name()), png);
            }
        }
        let len = (THUMB_SIZE.0 * THUMB_SIZE.1 * 4) as usize;
        for k in SceneKind::ALL {
            let img = a[k.index()]
                .as_ref()
                .unwrap_or_else(|| panic!("{k:?} missing"));
            assert_eq!(img.len(), len);
            assert!(mean_luma(img) > 6.0, "{k:?} is blank: {}", mean_luma(img));
            assert_eq!(a[k.index()], b[k.index()], "{k:?} is not deterministic");
        }
        for (i, x) in SceneKind::ALL.iter().enumerate() {
            for y in &SceneKind::ALL[i + 1..] {
                let (p, q) = (
                    a[x.index()].as_ref().unwrap(),
                    a[y.index()].as_ref().unwrap(),
                );
                let mad = p
                    .iter()
                    .zip(q)
                    .map(|(u, v)| u64::from(u.abs_diff(*v)))
                    .sum::<u64>() as f64
                    / p.len() as f64;
                assert!(mad > 2.0, "{x:?} and {y:?} look the same (mad {mad:.2})");
            }
        }
    }
}
