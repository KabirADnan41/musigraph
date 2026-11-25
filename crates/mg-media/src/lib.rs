//! Now-playing metadata from Windows media sessions (System Media Transport Controls).
//!
//! Audio capture (mg-audio) and media-session metadata are SEPARATE sources: WASAPI loopback hears the
//! whole system mix, and a media session describes one application. This crate never claims the two
//! match; `NowPlaying::app_id` tells the scene which application the metadata came from.
//!
//! Everything a media session reports (text, artwork bytes, timeline) is UNTRUSTED: strings are
//! sanitised and length-limited, artwork is size-limited, nothing is executed or persisted. No network.

#![deny(unsafe_code)]

#[cfg(windows)]
mod session;
#[cfg(windows)]
pub use session::NowPlayingWatcher;

pub mod palette;

use std::time::{Duration, Instant};

/// Maximum characters kept per text field.
pub const MAX_TEXT_CHARS: usize = 200;
/// Maximum artwork payload accepted from a session (compressed image bytes).
pub const MAX_ART_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaybackState {
    Playing,
    Paused,
    Stopped,
    /// Closed, opened, changing: nothing reliable to show.
    Other,
}

/// Snapshot of the selected media session. Plain data; cheap to clone except `art`.
#[derive(Clone, Debug, PartialEq)]
pub struct NowPlaying {
    pub title: String,
    pub artist: String,
    pub album: String,
    /// Application user-model id of the session (for example a player or browser), sanitised.
    pub app_id: String,
    pub state: PlaybackState,
    /// Timeline position (seconds) at `sampled_at`, if the session reports one.
    pub position_s: Option<f64>,
    pub duration_s: Option<f64>,
    pub sampled_at: Instant,
    /// Raw, still-compressed artwork bytes (decoded elsewhere, off the render loop); None if absent.
    pub art: Option<std::sync::Arc<[u8]>>,
    /// Changes whenever title/artist/album/app change (a new track), so consumers can reset per-track state.
    pub track_id: u64,
}

impl NowPlaying {
    /// Playback position now: interpolated from the last sample while playing (sessions update their
    /// timeline only occasionally), frozen while paused, clamped to the duration.
    pub fn position_now(&self, now: Instant) -> Option<f64> {
        let p = self.position_s?;
        let adv = if self.state == PlaybackState::Playing {
            now.saturating_duration_since(self.sampled_at).as_secs_f64()
        } else {
            0.0
        };
        let v = p + adv;
        Some(self.duration_s.map_or(v, |d| v.min(d)).max(0.0))
    }

    /// True if the metadata is worth showing (a titled, non-closed session).
    pub fn is_usable(&self) -> bool {
        !self.title.is_empty() && self.state != PlaybackState::Other
    }
}

/// Invisible formatting characters that can reorder or spoof displayed text (bidi controls, word joiner,
/// BOM, invisible operators, tag characters). ZWJ/ZWNJ (U+200D/200C) are kept: Bangla, Persian and other
/// scripts need them.
fn is_spoofing_format_char(c: char) -> bool {
    matches!(c, '\u{061C}' | '\u{180E}' | '\u{200B}' | '\u{200E}' | '\u{200F}'
        | '\u{202A}'..='\u{202E}' | '\u{2060}'..='\u{2064}' | '\u{2066}'..='\u{206F}'
        | '\u{FEFF}' | '\u{E0000}'..='\u{E007F}')
}

/// Sanitises untrusted text: control and spoofing format characters become spaces, whitespace is collapsed,
/// length-limited (characters). Not normalised.
pub fn sanitize_text(s: &str) -> String {
    let mut out = String::new();
    let (mut chars, mut last_space) = (0usize, true);
    for c in s.chars() {
        let c = if c.is_control() || is_spoofing_format_char(c) {
            ' '
        } else {
            c
        };
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
                chars += 1;
            }
            last_space = true;
        } else {
            out.push(c);
            chars += 1;
            last_space = false;
        }
        if chars >= MAX_TEXT_CHARS {
            break;
        }
    }
    out.trim().to_string()
}

/// 100 ns ticks between 1601-01-01 and 1970-01-01.
pub const EPOCH_DIFF_TICKS: i64 = 116_444_736_000_000_000;

fn ticks_to_s(t: i64) -> f64 {
    t as f64 / 10_000_000.0
}

/// Converts a session timeline (all values in 100 ns ticks, UNTRUSTED) into `(position_s, duration_s)`.
/// Saturating arithmetic only. While playing, the position is advanced by the time since the session last
/// updated its timeline (`last_updated` is a FILETIME; a value <= 0 means unknown and adds nothing).
/// The result is clamped to 0..duration. `None` when the session reports no usable duration.
pub fn timeline_seconds(
    start: i64,
    end: i64,
    pos: i64,
    last_updated: i64,
    now_unix_ticks: i64,
    playing: bool,
) -> Option<(f64, f64)> {
    if end <= start {
        return None;
    }
    let dur = end.saturating_sub(start);
    let mut pos_s = ticks_to_s(pos.saturating_sub(start).clamp(0, dur));
    let dur_s = ticks_to_s(dur);
    if playing && last_updated > 0 {
        let now = now_unix_ticks.saturating_add(EPOCH_DIFF_TICKS);
        pos_s += ticks_to_s(now.saturating_sub(last_updated).max(0)).min(3600.0);
    }
    Some((pos_s.min(dur_s), dur_s))
}

/// A candidate media session as seen by the selection logic.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub index: usize,
    pub state: PlaybackState,
    pub is_system_current: bool,
}

/// Picks which session to show. A playing session wins; among playing sessions the one Windows
/// considers current wins, otherwise the first. If nothing plays, the system-current session is kept
/// (paused metadata stays visible), otherwise none. `None` means "show no song information".
pub fn select_session(c: &[Candidate]) -> Option<usize> {
    let playing: Vec<&Candidate> = c
        .iter()
        .filter(|x| x.state == PlaybackState::Playing)
        .collect();
    if let Some(p) = playing
        .iter()
        .find(|x| x.is_system_current)
        .or(playing.first())
    {
        return Some(p.index);
    }
    c.iter()
        .find(|x| {
            x.is_system_current && matches!(x.state, PlaybackState::Paused | PlaybackState::Stopped)
        })
        .map(|x| x.index)
}

/// Metadata older than this without any successful refresh is treated as stale.
pub const STALE_AFTER: Duration = Duration::from_secs(15);

#[cfg(test)]
mod tests {
    use super::*;

    fn np(state: PlaybackState, pos: Option<f64>, dur: Option<f64>, at: Instant) -> NowPlaying {
        NowPlaying {
            title: "t".into(),
            artist: String::new(),
            album: String::new(),
            app_id: String::new(),
            state,
            position_s: pos,
            duration_s: dur,
            sampled_at: at,
            art: None,
            track_id: 1,
        }
    }

    #[test]
    fn position_interpolates_only_while_playing_and_clamps() {
        let t0 = Instant::now();
        let later = t0 + Duration::from_secs(5);
        let playing = np(PlaybackState::Playing, Some(10.0), Some(100.0), t0);
        assert!((playing.position_now(later).unwrap() - 15.0).abs() < 1e-6);
        let paused = np(PlaybackState::Paused, Some(10.0), Some(100.0), t0);
        assert_eq!(paused.position_now(later), Some(10.0));
        let near_end = np(PlaybackState::Playing, Some(98.0), Some(100.0), t0);
        assert_eq!(near_end.position_now(later), Some(100.0));
        assert_eq!(
            np(PlaybackState::Playing, None, None, t0).position_now(later),
            None
        );
        // a clock going backwards never yields a negative or shrinking position
        assert_eq!(
            playing.position_now(t0 - Duration::from_secs(1)),
            Some(10.0)
        );
    }

    #[test]
    fn text_is_sanitised_and_bounded() {
        assert_eq!(
            sanitize_text("  Hello\u{0}\n\tWorld \u{202e}x "),
            "Hello World x"
        );
        assert_eq!(sanitize_text(""), "");
        let long = "a".repeat(5000);
        assert_eq!(sanitize_text(&long).chars().count(), MAX_TEXT_CHARS);
        // multi-script text survives (Bangla, CJK, RTL)
        assert_eq!(sanitize_text("আমার সোনার বাংলা"), "আমার সোনার বাংলা");
        assert_eq!(sanitize_text("歌曲 مرحبا"), "歌曲 مرحبا");
        // bidi controls, word joiner, BOM and tag characters are neutralised; ZWJ/ZWNJ survive
        assert_eq!(
            sanitize_text("a\u{202A}b\u{2066}c\u{200F}d\u{2060}e\u{FEFF}f\u{E0041}g"),
            "a b c d e f g"
        );
        assert_eq!(
            sanitize_text("ক\u{200D}ষ می\u{200C}خواهم"),
            "ক\u{200D}ষ می\u{200C}خواهم"
        );
        // markup/command-like text is just text
        assert_eq!(
            sanitize_text("<script>rm -rf /</script>"),
            "<script>rm -rf /</script>"
        );
    }

    #[test]
    fn timeline_is_saturating_clamped_and_age_aware() {
        let s = 10_000_000i64; // one second in ticks
        let now = 1_700_000_000i64 * s; // unix ticks
        let upd = now + EPOCH_DIFF_TICKS - 5 * s; // updated 5 s ago (FILETIME)
        let (p, d) = timeline_seconds(0, 200 * s, 100 * s, upd, now, true).unwrap();
        assert!(
            (p - 105.0).abs() < 1e-6 && (d - 200.0).abs() < 1e-6,
            "{p} {d}"
        );
        assert_eq!(
            timeline_seconds(0, 200 * s, 100 * s, upd, now, false)
                .unwrap()
                .0,
            100.0
        );
        // unknown update time adds nothing; near the end clamps to the duration
        assert_eq!(
            timeline_seconds(0, 200 * s, 100 * s, 0, now, true)
                .unwrap()
                .0,
            100.0
        );
        assert_eq!(
            timeline_seconds(0, 200 * s, 199 * s, upd, now, true)
                .unwrap()
                .0,
            200.0
        );
        // hostile values never panic or produce NaN
        for v in [i64::MIN, i64::MAX, -1, 0] {
            if let Some((p, d)) = timeline_seconds(v, i64::MAX, v, v, v, true) {
                assert!(p.is_finite() && d.is_finite() && p >= 0.0 && p <= d);
            }
            let _ = timeline_seconds(i64::MIN, v, i64::MAX, i64::MIN, i64::MAX, true);
        }
        assert_eq!(timeline_seconds(5, 5, 5, upd, now, true), None);
    }

    #[test]
    fn session_selection_prefers_playing_then_system_current() {
        use PlaybackState::*;
        let c = |i, s, cur| Candidate {
            index: i,
            state: s,
            is_system_current: cur,
        };
        assert_eq!(select_session(&[]), None);
        assert_eq!(
            select_session(&[c(0, Paused, true), c(1, Playing, false)]),
            Some(1)
        );
        assert_eq!(
            select_session(&[c(0, Playing, false), c(1, Playing, true)]),
            Some(1)
        );
        assert_eq!(
            select_session(&[c(0, Playing, false), c(1, Playing, false)]),
            Some(0)
        );
        assert_eq!(
            select_session(&[c(0, Paused, false), c(1, Paused, true)]),
            Some(1)
        );
        // nothing playing and nothing current: show no song information rather than guess
        assert_eq!(
            select_session(&[c(0, Paused, false), c(1, Stopped, false)]),
            None
        );
        assert_eq!(select_session(&[c(0, Other, true)]), None);
    }
}
