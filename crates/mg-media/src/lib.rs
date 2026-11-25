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
    pub art: Option<Vec<u8>>,
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

/// Sanitises untrusted text: control characters become spaces, whitespace is collapsed, length-limited.
pub fn sanitize_text(s: &str) -> String {
    let mut out = String::new();
    let mut last_space = true;
    for c in s.chars() {
        let c = if c.is_control() || c == '\u{202e}' || c == '\u{200b}' {
            ' '
        } else {
            c
        };
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
            }
            last_space = true;
        } else {
            out.push(c);
            last_space = false;
        }
        if out.chars().count() >= MAX_TEXT_CHARS {
            break;
        }
    }
    out.trim().to_string()
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
        // markup/command-like text is just text
        assert_eq!(
            sanitize_text("<script>rm -rf /</script>"),
            "<script>rm -rf /</script>"
        );
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
