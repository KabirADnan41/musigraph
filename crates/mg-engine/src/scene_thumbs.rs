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

/// Renders every scene. Blocking GPU work: call it from a worker thread, never from the frame loop.
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
        out[kind.index()] = scene
            .render_offscreen_scenes(device, queue, THUMB_SIZE, &p, kind, None, &|_, _| {})
            .ok();
        if kind == SceneKind::Fluid {
            scene.fluid_release();
        }
    }
    out
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
