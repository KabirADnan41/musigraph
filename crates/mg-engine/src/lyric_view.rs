//! On-screen synchronised lyrics (pure state machine; the text bitmaps come from the worker thread).
//!
//! Lyrics exist only when the user supplied a `.lrc` file for the playing track (or in the explicitly
//! labelled demo mode): nothing is fetched, guessed or invented. Up to three lines are visible: the line
//! being sung, the next one (dim) and the one just finished (fading out). Line index `i` always uses slot
//! `i % 3`, so when the song advances the textures do not move: the old current line becomes "previous" and
//! fades, the next line becomes current and brightens, and only an invisible slot is ever re-uploaded.
//! Motion is a slow slide and fade; nothing pulses with the beat (the only music link is a slow glow boost).

use crate::presentation::{Eased, GpuOp, Job, Ready};
use mg_gfx::overlay::TextDraw;
use mg_media::lrc::Lyrics;
use mg_text::TextBitmap;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

/// Where lyrics may come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LyricSource {
    None,
    /// A folder of user-supplied `.lrc` files.
    Dir(PathBuf),
    /// The built-in original demo lyrics (with `--demo-media`).
    Demo,
}

const FADE_IN_S: f32 = 0.35;
const FADE_OUT_S: f32 = 0.30;
/// Alpha of the upcoming line.
const NEXT_ALPHA: f32 = 0.35;
/// Distance between the previous, current and next line, in line heights.
const BAND: f32 = 1.35;
/// First text-texture slot used by lyrics (0 and 1 are title and artist).
pub const FIRST_TEXT_SLOT: usize = 2;

#[derive(Default)]
struct Slot {
    line: Option<usize>,
    alpha: f32,
    dims: Option<(u32, u32, u32)>,
    /// Rendered for an older window height: swap in the new bitmap at once instead of waiting for a fade.
    size_stale: bool,
    /// Last role was "next": when it leaves it fades out in its own band instead of sweeping across the current one.
    below: bool,
    y: Eased,
    x: Eased,
}

pub struct LyricView {
    source: LyricSource,
    track: Option<u64>,
    lyrics: Option<Arc<Lyrics>>,
    position: Option<f64>,
    serial: u64,
    cache: HashMap<usize, Option<TextBitmap>>,
    requested: HashSet<usize>,
    slots: [Slot; 3],
    /// Vertical position of the current line as a fraction of the window height (set per scene).
    anchor: f32,
    /// Window size the cached line bitmaps were rendered for.
    for_size: (u32, u32),
}

/// Pixel size of lyric text for a window height.
pub fn lyric_px(h: f32) -> f32 {
    0.036 * h
}

/// Lowest ink the lyric stack can reach (as a fraction of the window height) when the current line sits at
/// `anchor`: a line that slides in starts one band below the "next" position (two bands under the current line)
/// and its bitmap is about 1.6 text sizes tall. Used to keep the lyrics clear of the control bar.
pub fn lowest_ink_frac(anchor: f32) -> f32 {
    anchor + (2.0 * BAND * 1.4 + 1.6) * 0.036 + 0.002
}

fn lyric_max_width(w: f32) -> u32 {
    (0.80 * w).max(64.0) as u32
}

impl LyricView {
    pub fn new(source: LyricSource) -> Self {
        Self {
            source,
            track: None,
            lyrics: None,
            position: None,
            serial: 0,
            cache: HashMap::new(),
            requested: HashSet::new(),
            slots: Default::default(),
            anchor: 0.72,
            for_size: (0, 0),
        }
    }

    pub fn set_anchor(&mut self, frac: f32) {
        if frac.is_finite() {
            self.anchor = frac.clamp(0.3, 0.92);
        }
    }

    /// Latest playback position in seconds (`None` when the session reports no timeline: no lyrics shown).
    pub fn set_position(&mut self, pos: Option<f64>) {
        self.position = pos.filter(|p| p.is_finite());
    }

    #[cfg(test)]
    pub fn has_lyrics(&self) -> bool {
        self.lyrics.is_some()
    }

    /// A new track (or none): drop everything of the old one and, if a source is configured, ask the worker
    /// for this track's lyrics. Slots keep their textures until they have faded out.
    pub fn on_track(&mut self, track: Option<u64>, title: &str, artist: &str, jobs: &mut Vec<Job>) {
        self.track = track;
        self.lyrics = None;
        self.cache.clear();
        self.requested.clear();
        self.serial += 1;
        for s in &mut self.slots {
            s.line = s.line.filter(|_| s.dims.is_some()).map(|_| usize::MAX); // stale content, never "current"
        }
        if let (Some(t), false) = (track, self.source == LyricSource::None) {
            jobs.push(Job::Lyrics {
                track_id: t,
                source: self.source.clone(),
                title: title.to_string(),
                artist: artist.to_string(),
            });
        }
    }

    pub fn on_ready(&mut self, r: Ready) {
        match r {
            Ready::Lyrics { track_id, lyrics } if Some(track_id) == self.track => {
                self.lyrics = lyrics;
            }
            Ready::Line {
                track_id,
                serial,
                idx,
                bitmap,
            } if Some(track_id) == self.track && serial == self.serial => {
                self.cache.insert(idx, bitmap);
            }
            _ => {}
        }
    }

    /// Advances fades and slides, requests missing line bitmaps, emits texture ops for invisible slots, and
    /// returns the three lyric draws (slot order = `line % 3`).
    pub fn update(
        &mut self,
        dt: f32,
        surface: (u32, u32),
        ops: &mut Vec<GpuOp>,
        jobs: &mut Vec<Job>,
    ) -> [Option<TextDraw>; 3] {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.25)
        } else {
            0.0
        };
        let (w, h) = (surface.0 as f32, surface.1 as f32);
        if surface.0 == 0 || surface.1 == 0 {
            return [None; 3];
        }
        // a resized window needs the lines at a new pixel size
        let moved = |now: u32, then: u32| (now as f32 / then.max(1) as f32 - 1.0).abs() > 0.15;
        if self.for_size != (0, 0)
            && (moved(surface.0, self.for_size.0) || moved(surface.1, self.for_size.1))
        {
            self.cache.clear();
            self.requested.clear();
            self.serial += 1;
            for s in &mut self.slots {
                s.size_stale = true;
            }
        }
        let cursor = match (&self.lyrics, self.position) {
            (Some(l), Some(p)) => Some(l.at(p)),
            _ => None,
        };
        let (cur, next) = cursor.map_or((None, None), |c| (c.current, c.next));
        // keep memory bounded when a long file is played or scrubbed through: only lines near the cursor stay
        if self.cache.len() > 12 {
            let c = cur.or(next).unwrap_or(0);
            self.cache.retain(|&k, _| k.abs_diff(c) <= 3);
            self.requested.retain(|&k| k.abs_diff(c) <= 3);
        }
        // request and place the lines that matter right now
        if let (Some(lyr), Some(track)) = (self.lyrics.clone(), self.track) {
            for idx in [cur, next].into_iter().flatten() {
                if self.cache.contains_key(&idx) || self.requested.contains(&idx) {
                    continue;
                }
                let text = &lyr.lines[idx].text;
                if text.is_empty() {
                    self.cache.insert(idx, None); // instrumental gap: nothing to draw
                } else {
                    self.requested.insert(idx);
                    self.for_size = surface;
                    jobs.push(Job::Line {
                        track_id: track,
                        serial: self.serial,
                        idx,
                        text: text.clone(),
                        px: lyric_px(h),
                        max_width: lyric_max_width(w),
                    });
                }
            }
            for idx in [cur, next].into_iter().flatten() {
                let s = idx % 3;
                let slot = &mut self.slots[s];
                let same = slot.line == Some(idx);
                let may_swap = slot.alpha < 0.05 || (same && slot.size_stale);
                if (!same || slot.size_stale) && may_swap {
                    if let Some(entry) = self.cache.get(&idx) {
                        match entry {
                            Some(b) => {
                                slot.dims = Some((b.width, b.height, b.pad));
                                ops.push(GpuOp::SetText(FIRST_TEXT_SLOT + s, b.clone()));
                            }
                            None => {
                                if slot.dims.take().is_some() {
                                    ops.push(GpuOp::ClearText(FIRST_TEXT_SLOT + s));
                                }
                            }
                        }
                        slot.line = Some(idx);
                        slot.size_stale = false;
                    }
                }
            }
        }
        // fades and positions by role
        let lh = lyric_px(h) * 1.4;
        let base_y = self.anchor * h;
        // lines that left (seek, track change) fade out first: a hidden line waits instead of crossfading on top
        let on_stage = |s: &Slot| {
            s.line.is_some() && (s.line == cur || (s.line == next && (s.below || s.alpha < 0.01)))
        };
        let leaving = self.slots.iter().any(|s| s.alpha > 0.08 && !on_stage(s));
        let mut out = [None; 3];
        for (s, slot) in self.slots.iter_mut().enumerate() {
            let (role_alpha, dy) = match slot.line {
                Some(l) if Some(l) == cur => (1.0, 0.0),
                Some(_) if on_stage(slot) => (NEXT_ALPHA, BAND * lh), // (a line that was current and is now next first fades out above)
                _ => (0.0, if slot.below { BAND } else { -BAND } * lh),
            };
            // a hidden next line starts one band lower and slides in: it never pops up inside the band the rising
            // current line is still crossing
            if role_alpha > 0.0 {
                slot.below = dy > 0.0;
            }
            let role_alpha = if leaving && slot.alpha < 0.08 {
                0.0
            } else {
                role_alpha
            };
            let entering = slot.alpha < 0.01 && role_alpha > 0.0;
            let spawn = if entering && dy > 0.0 { BAND * lh } else { 0.0 };
            let rate = if role_alpha > slot.alpha {
                1.0 / FADE_IN_S
            } else {
                1.0 / FADE_OUT_S
            };
            let step = rate * dt;
            slot.alpha += (role_alpha - slot.alpha).clamp(-step, step);
            let Some((tw, th, pad)) = slot.dims else {
                continue;
            };
            let snap = slot.alpha < 0.01;
            let ink = tw.saturating_sub(2 * pad) as f32;
            let x = slot.x.to(0.5 * (w - ink) - pad as f32, dt, snap);
            let y = slot
                .y
                .to(base_y + dy + spawn - pad as f32, dt, snap || entering);
            if slot.alpha > 0.002 {
                out[s] = Some(TextDraw {
                    x,
                    y,
                    w: tw as f32,
                    h: th as f32,
                    alpha: slot.alpha,
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SURF: (u32, u32) = (1920, 1080);

    fn lyrics() -> Arc<Lyrics> {
        Arc::new(
            mg_media::lrc::parse(
                b"[00:01.00]one\n[00:05.00]two\n[00:09.00]\n[00:12.00]four\n[00:16.00]five\n",
            )
            .unwrap(),
        )
    }

    fn bmp() -> TextBitmap {
        TextBitmap {
            width: 120,
            height: 40,
            pad: 8,
            coverage: vec![200; 4800],
        }
    }

    struct Rig {
        v: LyricView,
        ops: Vec<GpuOp>,
        jobs: Vec<Job>,
    }

    impl Rig {
        fn new() -> Self {
            let mut v = LyricView::new(LyricSource::Demo);
            let mut jobs = Vec::new();
            v.on_track(Some(7), "T", "A", &mut jobs);
            v.on_ready(Ready::Lyrics {
                track_id: 7,
                lyrics: Some(lyrics()),
            });
            Self {
                v,
                ops: Vec::new(),
                jobs,
            }
        }

        /// Runs `secs` of frames at `pos` (position advances with time when `play`), answering line jobs.
        fn run(&mut self, mut pos: f64, secs: f32, play: bool) -> [Option<TextDraw>; 3] {
            let mut out = [None; 3];
            for _ in 0..(secs * 60.0) as usize {
                self.v.set_position(Some(pos));
                out = self
                    .v
                    .update(1.0 / 60.0, SURF, &mut self.ops, &mut self.jobs);
                for j in std::mem::take(&mut self.jobs) {
                    if let Job::Line {
                        track_id,
                        serial,
                        idx,
                        ..
                    } = j
                    {
                        self.v.on_ready(Ready::Line {
                            track_id,
                            serial,
                            idx,
                            bitmap: Some(bmp()),
                        });
                    }
                }
                if play {
                    pos += 1.0 / 60.0;
                }
            }
            out
        }
    }

    #[test]
    fn a_configured_source_requests_lyrics_once_per_track_and_none_requests_nothing() {
        let mut jobs = Vec::new();
        let mut v = LyricView::new(LyricSource::Dir("x".into()));
        v.on_track(Some(1), "Title", "Artist", &mut jobs);
        assert!(matches!(jobs.as_slice(), [Job::Lyrics { track_id: 1, .. }]));
        let mut none = LyricView::new(LyricSource::None);
        let mut jobs = Vec::new();
        none.on_track(Some(1), "Title", "Artist", &mut jobs);
        assert!(jobs.is_empty(), "no source: nothing is looked up");
    }

    #[test]
    fn the_current_line_is_bright_the_next_is_dim_and_a_finished_line_fades_away() {
        let mut r = Rig::new();
        let o = r.run(6.0, 1.5, false); // two is being sung, line three is an instrumental gap
        let cur = o[1].expect("current line (index 1 -> slot 1)");
        assert!(cur.alpha > 0.99, "current {}", cur.alpha);
        assert!(o[2].is_none(), "index 2 is a gap and draws nothing");
        // advance to line four: "two" becomes the previous line and fades out, "four" becomes current
        let o = r.run(12.5, 1.5, false);
        assert!(
            o[0].is_some_and(|d| d.alpha > 0.99),
            "index 3 -> slot 0 is current"
        );
        // "two" (slot 1) has faded out and its slot now holds the NEXT line ("five", index 4) at the dim alpha
        assert!(
            o[1].is_some_and(|d| (d.alpha - NEXT_ALPHA).abs() < 0.02),
            "the previous line is gone; its slot shows the next line dimly"
        );
    }

    #[test]
    fn the_next_line_is_shown_dim_and_below_the_current_one() {
        let mut r = Rig::new();
        let o = r.run(13.0, 1.5, false); // "four" current (idx 3), "five" next (idx 4)
        let (cur, next) = (o[0].unwrap(), o[1].unwrap());
        assert!(
            cur.alpha > 0.99 && (next.alpha - NEXT_ALPHA).abs() < 0.02,
            "{cur:?} {next:?}"
        );
        assert!(next.y > cur.y, "next line sits below the current one");
    }

    #[test]
    fn pause_freezes_the_display_and_a_seek_replaces_the_lines_without_stale_content() {
        let mut r = Rig::new();
        let a = r.run(6.0, 1.5, false);
        let b = r.run(6.0, 1.0, false); // paused: identical
        assert_eq!(a[1].map(|d| d.alpha), b[1].map(|d| d.alpha));
        // seek far ahead: the old line fades before the new one is uploaded into the same slot
        r.ops.clear();
        let o = r.run(13.0, 2.0, false);
        assert!(o[0].is_some_and(|d| d.alpha > 0.99));
        assert!(
            r.ops
                .iter()
                .any(|op| matches!(op, GpuOp::SetText(s, _) if *s >= FIRST_TEXT_SLOT)),
            "new lines uploaded"
        );
    }

    #[test]
    fn late_results_for_a_previous_track_or_old_size_are_ignored_and_new_track_clears_lyrics() {
        let mut r = Rig::new();
        r.v.on_ready(Ready::Line {
            track_id: 99,
            serial: r.v.serial,
            idx: 0,
            bitmap: Some(bmp()),
        });
        assert!(r.v.cache.is_empty(), "other track ignored");
        r.v.on_ready(Ready::Line {
            track_id: 7,
            serial: r.v.serial + 5,
            idx: 0,
            bitmap: Some(bmp()),
        });
        assert!(r.v.cache.is_empty(), "wrong serial ignored");
        r.run(6.0, 1.0, false);
        let mut jobs = Vec::new();
        r.v.on_track(Some(8), "Other", "A", &mut jobs);
        assert!(
            !r.v.has_lyrics(),
            "the previous track's lyrics must not stay"
        );
        let o = r.run(6.0, 1.0, false);
        assert!(
            o.iter().all(|d| d.is_none_or(|d| d.alpha < 0.05)),
            "nothing of the old track stays visible"
        );
    }

    #[test]
    fn no_position_means_no_lyrics_and_hostile_values_are_harmless() {
        let mut r = Rig::new();
        let mut ops = Vec::new();
        let mut jobs = Vec::new();
        r.v.set_position(None);
        let o = r.v.update(1.0 / 60.0, SURF, &mut ops, &mut jobs);
        assert!(o.iter().all(Option::is_none) && jobs.is_empty());
        r.v.set_position(Some(f64::NAN));
        let _ = r.v.update(f32::NAN, (0, 0), &mut ops, &mut jobs);
        r.v.set_anchor(f32::NAN);
        r.v.set_anchor(5.0);
        assert!(r.v.anchor <= 0.92);
    }

    #[test]
    fn lyrics_stay_on_screen_for_any_window_shape() {
        for &(w, h) in &[(1920u32, 1080u32), (3440, 1440), (720, 1280), (320, 180)] {
            let mut v = LyricView::new(LyricSource::Demo);
            let mut jobs = Vec::new();
            v.on_track(Some(1), "T", "A", &mut jobs);
            v.on_ready(Ready::Lyrics {
                track_id: 1,
                lyrics: Some(lyrics()),
            });
            let mut ops = Vec::new();
            let mut out = [None; 3];
            for _ in 0..120 {
                v.set_position(Some(13.0));
                out = v.update(1.0 / 60.0, (w, h), &mut ops, &mut jobs);
                for j in std::mem::take(&mut jobs) {
                    if let Job::Line {
                        track_id,
                        serial,
                        idx,
                        ..
                    } = j
                    {
                        let tw = lyric_max_width(w as f32).min(w);
                        let bmp = TextBitmap {
                            width: tw,
                            height: (lyric_px(h as f32) * 1.6) as u32 + 2,
                            pad: 2,
                            coverage: vec![
                                1;
                                (tw * ((lyric_px(h as f32) * 1.6) as u32 + 2)) as usize
                            ],
                        };
                        v.on_ready(Ready::Line {
                            track_id,
                            serial,
                            idx,
                            bitmap: Some(bmp),
                        });
                    }
                }
            }
            for d in out.into_iter().flatten() {
                assert!(
                    d.x + 2.0 >= 0.0 && d.x + d.w <= w as f32 + 2.0,
                    "{w}x{h} x {d:?}"
                );
                assert!(d.y >= 0.0 && d.y + d.h <= h as f32 + 2.0, "{w}x{h} y {d:?}");
            }
        }
    }

    #[test]
    fn a_width_only_change_re_renders_the_lines_at_the_new_size() {
        let mut r = Rig::new();
        r.run(13.0, 1.0, false);
        let serial = r.v.serial;
        // same height, much narrower window (e.g. a snap): lines must be requested again
        r.v.set_position(Some(13.0));
        r.v.update(1.0 / 60.0, (900, 1080), &mut r.ops, &mut r.jobs);
        assert!(
            r.v.serial > serial,
            "width change must invalidate the cached lines"
        );
        assert!(
            r.jobs
                .iter()
                .any(|j| matches!(j, Job::Line { max_width, .. } if *max_width <= 720))
        );
    }

    #[test]
    fn scrubbing_a_long_file_keeps_the_cache_bounded() {
        let long: String = (0..400)
            .map(|i| format!("[{:02}:{:02}.00]line {i}\n", i / 60, i % 60))
            .collect();
        let mut v = LyricView::new(LyricSource::Demo);
        let mut jobs = Vec::new();
        v.on_track(Some(1), "T", "A", &mut jobs);
        v.on_ready(Ready::Lyrics {
            track_id: 1,
            lyrics: Some(Arc::new(mg_media::lrc::parse(long.as_bytes()).unwrap())),
        });
        let mut ops = Vec::new();
        for step in 0..400 {
            v.set_position(Some(step as f64 + 0.5));
            v.update(1.0 / 60.0, SURF, &mut ops, &mut jobs);
            for j in std::mem::take(&mut jobs) {
                if let Job::Line {
                    track_id,
                    serial,
                    idx,
                    ..
                } = j
                {
                    v.on_ready(Ready::Line {
                        track_id,
                        serial,
                        idx,
                        bitmap: Some(bmp()),
                    });
                }
            }
            assert!(v.cache.len() <= 20, "cache grew to {}", v.cache.len());
        }
    }

    #[test]
    fn content_of_the_previous_track_is_never_shown_as_current_after_a_track_change() {
        let mut r = Rig::new();
        r.run(6.0, 1.5, false);
        let mut jobs = Vec::new();
        r.v.on_track(Some(8), "Other", "A", &mut jobs);
        r.v.on_ready(Ready::Lyrics {
            track_id: 8,
            lyrics: Some(lyrics()),
        });
        let o = r.run(6.0, 0.05, false);
        // right after the change the old line may still be fading, but never at full brightness again
        assert!(o.iter().flatten().all(|d| d.alpha < 1.0));
    }
}

#[cfg(test)]
mod overlap_tests {
    use super::*;

    fn lyr() -> Arc<Lyrics> {
        Arc::new(crate::media_feed::demo_lyrics("The Lantern Parade").unwrap())
    }

    /// Bitmap the size of a real rendered line (ink height ~ 1.6 px of text size, 2 px pad).
    fn sized(w: u32, h: u32) -> TextBitmap {
        let (bw, bh) = (
            lyric_max_width(w as f32).min(w) / 2,
            (lyric_px(h as f32) * 1.6) as u32 + 4,
        );
        TextBitmap {
            width: bw,
            height: bh,
            pad: 2,
            coverage: vec![1; (bw * bh) as usize],
        }
    }

    fn frame(v: &mut LyricView, pos: f64, s: (u32, u32)) -> [Option<TextDraw>; 3] {
        let (mut ops, mut jobs) = (Vec::new(), Vec::new());
        v.set_position(Some(pos));
        let out = v.update(1.0 / 60.0, s, &mut ops, &mut jobs);
        for j in jobs {
            if let Job::Line {
                track_id,
                serial,
                idx,
                ..
            } = j
            {
                v.on_ready(Ready::Line {
                    track_id,
                    serial,
                    idx,
                    bitmap: Some(sized(s.0, s.1)),
                });
            }
        }
        out
    }

    /// Worst vertical overlap (pixels of ink band) of two draws that are both visible.
    fn worst_overlap(o: &[Option<TextDraw>; 3], pad: f32) -> f32 {
        let vis: Vec<_> = o.iter().flatten().filter(|d| d.alpha > 0.08).collect();
        let mut worst = 0.0f32;
        for (i, a) in vis.iter().enumerate() {
            for b in &vis[i + 1..] {
                let (a0, a1) = (a.y + pad, a.y + a.h - pad);
                let (b0, b1) = (b.y + pad, b.y + b.h - pad);
                worst = worst.max(a1.min(b1) - a0.max(b0));
            }
        }
        worst
    }

    #[test]
    fn visible_lyric_lines_never_overlap_around_line_changes() {
        for s in [(1920u32, 1080u32), (1280, 720), (720, 1280)] {
            for anchor in [0.70f32, 0.72, 0.80] {
                // boundaries of the demo lyrics (2, 6, 10, 14, 18 s)
                for boundary in [2.0f64, 6.0, 10.0, 14.0, 18.0] {
                    let mut v = LyricView::new(LyricSource::Demo);
                    v.set_anchor(anchor);
                    v.on_track(Some(1), "T", "A", &mut Vec::new());
                    v.on_ready(Ready::Lyrics {
                        track_id: 1,
                        lyrics: Some(lyr()),
                    });
                    let mut t = boundary - 2.0;
                    while t < boundary - 0.3 {
                        frame(&mut v, t, s);
                        t += 1.0 / 60.0;
                    }
                    while t < boundary + 1.2 {
                        let o = frame(&mut v, t, s);
                        let ov = worst_overlap(&o, 2.0);
                        assert!(
                            ov <= 2.0,
                            "{s:?} anchor {anchor} boundary {boundary} t {t:.2}: overlap {ov:.1}px {o:?}"
                        );
                        t += 1.0 / 60.0;
                    }
                }
            }
        }
    }

    /// Lowest visible ink (px) of the three lyric slots.
    fn lowest(o: &[Option<TextDraw>; 3], pad: f32) -> f32 {
        o.iter()
            .flatten()
            .filter(|d| d.alpha > 0.08)
            .map(|d| d.y + d.h - pad)
            .fold(0.0, f32::max)
    }

    #[test]
    fn lifted_lyrics_never_reach_the_control_bar_zone_at_any_size_anchor_or_dpi() {
        use crate::controls::{lyric_lift, ui_scale};
        for s in [
            (1920u32, 1080u32),
            (1280, 720),
            (720, 1280),
            (3840, 2160),
            (800, 600),
        ] {
            for dpi in [1.0f32, 1.5, 2.0] {
                for anchor in [0.70f32, 0.72, 0.80] {
                    let zone_top = s.1 as f32 - 128.0 * ui_scale(s, dpi);
                    let mut worst_lifted = 0.0f32;
                    let mut worst_plain = 0.0f32;
                    for lift in [0.0, lyric_lift(anchor, s, dpi)] {
                        let mut v = LyricView::new(LyricSource::Demo);
                        v.set_anchor(anchor - lift);
                        v.on_track(Some(1), "T", "A", &mut Vec::new());
                        v.on_ready(Ready::Lyrics {
                            track_id: 1,
                            lyrics: Some(lyr()),
                        });
                        let mut worst = 0.0f32;
                        let mut t = 0.0f64;
                        while t < 20.0 {
                            let o = frame(&mut v, t, s);
                            worst = worst.max(lowest(&o, 2.0));
                            t += 1.0 / 30.0;
                        }
                        if lift == 0.0 {
                            worst_plain = worst;
                        } else {
                            worst_lifted = worst;
                        }
                    }
                    assert!(
                        worst_lifted > 0.0 && worst_lifted <= zone_top,
                        "{s:?} dpi {dpi} anchor {anchor}: lyrics end at {worst_lifted:.1}, bar zone starts at {zone_top:.1}"
                    );
                    // the check is not vacuous: where the unlifted lines reach the zone, the lift is what clears it
                    if worst_plain > zone_top {
                        assert!(worst_lifted < worst_plain);
                    }
                }
            }
        }
    }

    #[test]
    fn seeks_and_pauses_never_stack_visible_lines() {
        let s = (1920, 1080);
        let mut v = LyricView::new(LyricSource::Demo);
        v.on_track(Some(1), "T", "A", &mut Vec::new());
        v.on_ready(Ready::Lyrics {
            track_id: 1,
            lyrics: Some(lyr()),
        });
        // (position, seconds to hold): play, seek forward, pause, seek back, seek to the same neighbourhood
        for (pos, secs) in [
            (3.0f64, 1.5f32),
            (11.0, 1.0),
            (11.0, 1.0),
            (3.0, 0.5),
            (19.0, 0.5),
            (6.1, 0.2),
            (14.2, 1.0),
        ] {
            for _ in 0..(secs * 60.0) as usize {
                let o = frame(&mut v, pos, s);
                let ov = worst_overlap(&o, 2.0);
                assert!(ov <= 2.0, "seek to {pos}: overlap {ov:.1}px {o:?}");
            }
        }
    }

    #[test]
    fn a_track_change_never_stacks_old_and_new_lines() {
        let s = (1920, 1080);
        let mut v = LyricView::new(LyricSource::Demo);
        v.on_track(Some(1), "T", "A", &mut Vec::new());
        v.on_ready(Ready::Lyrics {
            track_id: 1,
            lyrics: Some(lyr()),
        });
        for _ in 0..90 {
            frame(&mut v, 11.0, s);
        }
        v.on_track(Some(2), "T", "A", &mut Vec::new());
        v.on_ready(Ready::Lyrics {
            track_id: 2,
            lyrics: Some(lyr()),
        });
        for _ in 0..120 {
            let o = frame(&mut v, 3.0, s);
            let ov = worst_overlap(&o, 2.0);
            assert!(ov <= 2.0, "track change: overlap {ov:.1}px {o:?}");
        }
    }
}
