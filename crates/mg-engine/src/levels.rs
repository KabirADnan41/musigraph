//! `--print-levels`: live-audio smoke mode. Captures the default output (loopback) into the central
//! analyzer and prints DERIVED numbers only (RMS/peak/group levels/onsets). Samples are never stored,
//! logged or printed (ADR-0003). Not synthetic: every number comes from what the system is playing.

use mg_audio::capture::Loopback;
use mg_dsp::{AnalysisParams, Analyzer, AudioFrame, GROUPS};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Shared {
    last: Option<AudioFrame>,
    frames: u64,
    onsets: u64,
    peak: f32,
    max_rms_db: f32,
}

/// Runs for `seconds`, printing one line per 250 ms. Returns the process exit code.
pub fn run(seconds: u32) -> i32 {
    let shared = Arc::new(Mutex::new(Shared {
        max_rms_db: -120.0,
        ..Shared::default()
    }));
    let sh = shared.clone();
    let mut analyzer: Option<Analyzer> = None;
    let cap = Loopback::start(move |block, rate, t_end_ns| {
        if analyzer.as_ref().map(Analyzer::sample_rate) != Some(rate) {
            analyzer = Analyzer::new(rate, AnalysisParams::default()).ok();
        }
        let Some(a) = analyzer.as_mut() else { return };
        a.push(block, t_end_ns, |f| {
            if let Ok(mut s) = sh.lock() {
                s.frames += 1;
                s.onsets += u64::from(f.onset_event);
                s.peak = s.peak.max(f.peak);
                s.max_rms_db = s.max_rms_db.max(f.rms_db);
                s.last = Some(*f);
            }
        });
    });
    let mut cap = match cap {
        Ok(c) => c,
        Err(e) => {
            eprintln!("RESULT: FAILED: {e}");
            return 1;
        }
    };
    println!(
        "LIVE audio (WASAPI loopback of the default output). Derived values only; no samples are stored."
    );
    let start = Instant::now();
    let mut next = start;
    while start.elapsed() < Duration::from_secs(u64::from(seconds)) {
        next += Duration::from_millis(250);
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
        let st = cap.status();
        let (line, _) = match shared.lock() {
            Ok(mut s) => {
                let peak = std::mem::take(&mut s.peak);
                match s.last {
                    Some(f) => {
                        let g = |a: &[f32; GROUPS]| {
                            a.iter()
                                .map(|v| format!("{v:.2}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                        };
                        (
                            format!(
                                "t={:5.2}s rate={} rms_db={:7.1} peak={:.3} silent={} groups_fast=[{}] energy={:.2} onsets={} opens={} synth_silence={} err={}",
                                start.elapsed().as_secs_f32(),
                                f.sample_rate,
                                f.rms_db,
                                peak,
                                f.silent as u8,
                                g(&f.group_fast),
                                f.energy,
                                s.onsets,
                                st.opens,
                                st.synthesized_silence_blocks,
                                st.last_error.as_deref().unwrap_or("-"),
                            ),
                            true,
                        )
                    }
                    None => (
                        format!(
                            "t={:5.2}s waiting for audio opens={} err={}",
                            start.elapsed().as_secs_f32(),
                            st.opens,
                            st.last_error.as_deref().unwrap_or("-")
                        ),
                        false,
                    ),
                }
            }
            Err(_) => ("state poisoned".into(), false),
        };
        println!("{line}");
    }
    let st = cap.status();
    cap.stop();
    let (frames, onsets, max_rms) = shared
        .lock()
        .map(|s| (s.frames, s.onsets, s.max_rms_db))
        .unwrap_or((0, 0, -120.0));
    println!(
        "SUMMARY frames={frames} onsets={onsets} max_rms_db={max_rms:.1} opens={} discontinuities={} synth_silence_blocks={} last_error={}",
        st.opens,
        st.discontinuities,
        st.synthesized_silence_blocks,
        st.last_error.as_deref().unwrap_or("-")
    );
    if frames > 0 {
        0
    } else {
        eprintln!("RESULT: FAILED: no analysis frames were produced");
        1
    }
}
