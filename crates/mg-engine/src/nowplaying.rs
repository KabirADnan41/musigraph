//! `--print-nowplaying [--seconds N]`: smoke mode that prints what the Windows media sessions report
//! (title, artist, album, app, state, position, artwork size). Local console output only; nothing is
//! stored or sent anywhere. Metadata and system audio are separate sources and may not match.

use mg_media::NowPlayingWatcher;
use std::time::{Duration, Instant};

pub fn run(seconds: u32) -> i32 {
    let mut w = match NowPlayingWatcher::start() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("RESULT: FAILED: {e}");
            return 1;
        }
    };
    let start = Instant::now();
    let mut seen = 0u32;
    while start.elapsed() < Duration::from_secs(u64::from(seconds)) {
        std::thread::sleep(Duration::from_millis(1000));
        match w.now_playing() {
            Some(n) => {
                seen += 1;
                println!(
                    "t={:3.0}s state={:?} app={:?} title={:?} artist={:?} album={:?} pos={} dur={} art_bytes={} track_id={:016x}",
                    start.elapsed().as_secs_f32(),
                    n.state,
                    n.app_id,
                    n.title,
                    n.artist,
                    n.album,
                    n.position_now(Instant::now())
                        .map_or("-".into(), |p| format!("{p:.1}s")),
                    n.duration_s.map_or("-".into(), |d| format!("{d:.1}s")),
                    n.art.as_ref().map_or(0, Vec::len),
                    n.track_id
                );
            }
            None => println!(
                "t={:3.0}s no usable media session (polls={} error={})",
                start.elapsed().as_secs_f32(),
                w.polls(),
                w.last_error().unwrap_or_else(|| "-".into())
            ),
        }
    }
    w.stop();
    println!("SUMMARY snapshots_with_metadata={seen}");
    0
}
