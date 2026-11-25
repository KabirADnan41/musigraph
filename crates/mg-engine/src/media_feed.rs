//! Where now-playing snapshots come from for the scene runner.
//!
//! * `Live`: the Windows media-session watcher (SMTC). Describes ONE application's session; it does not
//!   prove that the captured system audio is that session (WASAPI loopback hears the whole mix).
//! * `Demo`: original, clearly synthetic tracks with generated artwork and invented titles, for previews and
//!   tests only (no copyrighted art or real listening data). Selected with `--demo-media`.
//! * `Off`: no metadata, nothing drawn.
//!
//! Polling is throttled: the render loop asks a few times a second, never per frame.

use mg_media::{NowPlaying, NowPlayingWatcher, PlaybackState};
use std::time::{Duration, Instant};

const POLL_EVERY: Duration = Duration::from_millis(250);
/// How long each demo track plays before the next one starts.
pub const DEMO_TRACK_SECS: u64 = 22;

pub enum MediaFeed {
    Off,
    Live {
        watcher: NowPlayingWatcher,
        last_poll: Option<Instant>,
        cache: Option<NowPlaying>,
    },
    Demo {
        tracks: Vec<NowPlaying>,
        start: Instant,
    },
}

impl MediaFeed {
    pub fn live() -> Result<Self, String> {
        Ok(MediaFeed::Live {
            watcher: NowPlayingWatcher::start()?,
            last_poll: None,
            cache: None,
        })
    }

    pub fn demo() -> Self {
        MediaFeed::Demo {
            tracks: demo_tracks(),
            start: Instant::now(),
        }
    }

    /// Latest snapshot, refreshed at most every 250 ms (cheap clone otherwise).
    pub fn poll(&mut self, now: Instant) -> Option<NowPlaying> {
        match self {
            MediaFeed::Off => None,
            MediaFeed::Live {
                watcher,
                last_poll,
                cache,
            } => {
                if last_poll.is_none_or(|t| now.duration_since(t) >= POLL_EVERY) {
                    *last_poll = Some(now);
                    *cache = watcher.now_playing();
                }
                cache.clone()
            }
            MediaFeed::Demo { tracks, start } => {
                let elapsed = now.duration_since(*start).as_secs_f64();
                let i = (elapsed as u64 / DEMO_TRACK_SECS) as usize % tracks.len().max(1);
                let mut t = tracks.get(i).cloned()?;
                // a live timeline, as a real player would report it
                t.position_s = Some(elapsed % DEMO_TRACK_SECS as f64);
                t.duration_s = Some(DEMO_TRACK_SECS as f64);
                t.sampled_at = now;
                Some(t)
            }
        }
    }
}

fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

/// Procedural cover: vertical gradient `c0 -> c1`, a soft disc of `c2` and a darker band. Deterministic.
fn cover(
    w: u32,
    h: u32,
    c0: [f32; 3],
    c1: [f32; 3],
    c2: [f32; 3],
    disc: (f32, f32, f32),
) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            let mut c = lerp(c0, c1, v);
            let d = ((u - disc.0).powi(2) + (v - disc.1).powi(2)).sqrt();
            let k = (1.0 - ((d - disc.2) / 0.06).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            c = lerp(c, c2, k);
            let band = ((v - 0.82).abs() < 0.05) as u8 as f32 * 0.35;
            c = lerp(c, [0.0; 3], band);
            out.extend(c.map(|f| (f.clamp(0.0, 1.0) * 255.0) as u8));
            out.push(255);
        }
    }
    out
}

fn track(id: u64, title: &str, artist: &str, art: Option<Vec<u8>>) -> NowPlaying {
    NowPlaying {
        title: title.into(),
        artist: artist.into(),
        album: String::new(),
        app_id: "demo".into(),
        state: PlaybackState::Playing,
        position_s: None,
        duration_s: None,
        sampled_at: Instant::now(),
        art: art.map(|b| std::sync::Arc::from(b.into_boxed_slice())),
        track_id: id,
    }
}

fn png(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    mg_media::synth::png_rgba(w, h, rgba)
}

/// Original demo tracks: distinct palettes, a non-square cover, Bangla text, a very long title with several
/// artists, a monochrome cover and a track without artwork.
pub fn demo_tracks() -> Vec<NowPlaying> {
    let c = |r: u8, g: u8, b: u8| [r, g, b].map(|v| f32::from(v) / 255.0);
    vec![
        track(
            1,
            "Midnight Harbour",
            "Aster Vale",
            Some(png(
                160,
                160,
                &cover(
                    160,
                    160,
                    c(20, 40, 120),
                    c(240, 140, 40),
                    c(255, 235, 150),
                    (0.5, 0.38, 0.2),
                ),
            )),
        ),
        track(
            2,
            "The Lantern Parade",
            "Iris Moon & The Paper Boats",
            Some(png(
                200,
                120,
                &cover(
                    200,
                    120,
                    c(200, 30, 120),
                    c(40, 10, 60),
                    c(255, 200, 80),
                    (0.3, 0.5, 0.25),
                ),
            )),
        ),
        track(
            3,
            "নীল আকাশের গান",
            "সুরেলা",
            Some(png(
                160,
                160,
                &cover(
                    160,
                    160,
                    c(10, 110, 90),
                    c(170, 220, 80),
                    c(240, 250, 230),
                    (0.65, 0.4, 0.18),
                ),
            )),
        ),
        track(
            4,
            "A Very Long Title For Testing How Narrow Windows Trim Text Without Breaking The Composition",
            "Artist One, Artist Two, Artist Three & Friends",
            Some(png(
                160,
                160,
                &cover(
                    160,
                    160,
                    c(30, 30, 30),
                    c(200, 200, 200),
                    c(120, 120, 120),
                    (0.5, 0.5, 0.22),
                ),
            )),
        ),
        track(5, "No Artwork Example", "Anonymous", None),
    ]
}

/// Original, invented demo lyrics (never real songs) for the demo tracks, keyed by title; `None` for others.
pub fn demo_lyrics(title: &str) -> Option<mg_media::lrc::Lyrics> {
    let lrc = match title {
        "Midnight Harbour" => {
            "[00:02.00]Lanterns drift across the water\n[00:05.00]Every wave remembers your name\n[00:08.00]\n[00:10.00]Hold the light a little longer\n[00:14.00]The tide is turning slow\n[00:18.00]We sail before the morning\n"
        }
        "The Lantern Parade" => {
            "[00:02.00]Paper boats and golden sparks\n[00:06.00]Dancing through the evening air\n[00:10.00]Every window hums along\n[00:14.00]Nobody wants to go home\n[00:18.00]Not while the music is here\n"
        }
        "নীল আকাশের গান" => {
            "[00:02.00]নীল আকাশে ভাসে মেঘের দল\n[00:06.00]সুরের ঢেউয়ে দুলছে মন\n[00:10.00]আলো জ্বলে নদীর বুকে\n[00:14.00]ভোরের হাওয়ায় নতুন গান\n"
        }
        t if t.starts_with("A Very Long Title") => {
            "[00:02.00]These words are only a test of the layout\n[00:06.00]A single lyric line that keeps going and going until it is far wider than any window should ever need to show\n[00:12.00]Short again\n"
        }
        "No Artwork Example" => {
            "[00:03.00]No cover, only words\n[00:08.00]And the light still follows the sound\n"
        }
        _ => return None,
    };
    mg_media::lrc::parse(lrc.as_bytes()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presentation::scene_roles;
    use mg_media::palette::extract;

    #[test]
    fn demo_tracks_are_original_distinct_and_cover_the_edge_cases() {
        let t = demo_tracks();
        assert!(t.len() >= 5);
        let ids: std::collections::HashSet<u64> = t.iter().map(|n| n.track_id).collect();
        assert_eq!(ids.len(), t.len(), "unique track ids");
        assert!(t.iter().any(|n| n.art.is_none()), "a track without artwork");
        assert!(
            t.iter().any(|n| n.title.chars().count() > 60),
            "a very long title"
        );
        assert!(t.iter().any(|n| !n.title.is_ascii()), "non-Latin text");
        assert!(t.iter().all(|n| n.is_usable()));
    }

    #[test]
    fn different_artwork_gives_clearly_different_scene_colours() {
        let roles: Vec<[[f32; 3]; 4]> = demo_tracks()
            .iter()
            .filter_map(|n| n.art.as_ref())
            .map(|b| {
                let img = mg_media::decode_art(b).expect("demo art decodes");
                let (px, w, h) = img.thumbnail(64);
                scene_roles(&extract(&px, w, h))
            })
            .collect();
        assert!(roles.len() >= 4);
        let dist = |a: &[[f32; 3]; 4], b: &[[f32; 3]; 4]| -> f32 {
            (0..4)
                .map(|i| (0..3).map(|c| (a[i][c] - b[i][c]).abs()).sum::<f32>())
                .sum()
        };
        for i in 0..roles.len() {
            for j in i + 1..roles.len() {
                assert!(
                    dist(&roles[i], &roles[j]) > 0.5,
                    "albums {i} and {j} look alike"
                );
            }
        }
    }

    #[test]
    fn demo_feed_cycles_tracks() {
        let mut f = MediaFeed::demo();
        let t0 = Instant::now();
        let a = f.poll(t0).unwrap().track_id;
        let b = f
            .poll(t0 + Duration::from_secs(DEMO_TRACK_SECS + 1))
            .unwrap()
            .track_id;
        assert_ne!(a, b);
        assert!(MediaFeed::Off.poll(t0).is_none());
    }
}
