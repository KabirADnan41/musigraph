//! On-screen synchronised lyrics (pure state machine; the text bitmaps come from the worker thread).
//!
//! Lyrics exist only when the user supplied a `.lrc` file for the playing track (or in the explicitly
//! labelled demo mode): nothing is fetched, guessed or invented. Line index `i` always uses slot `i % 3`, so the
//! textures never move between slots and only an invisible slot is ever re-uploaded.
//!
//! Two presentations:
//! * Classic (every scene except Album Focus): the line being sung, the next one (dim) and the one just finished
//!   (fading out), centred at a per-scene anchor; a slow slide and fade with its own fade state.
//! * Column (Album Focus, `set_column`): a start-aligned column where the current line sits at the anchor and the
//!   next line waits below at context scale. The whole state is a pure function of the playback position: the
//!   slide into a line ENDS at that line's timestamp and lasts min(600 ms, 45 % of the time since the previous
//!   line started), so slides never overlap and the current line is never late; a seek or track change resolves
//!   directly to the position (no queued or replayed motion). Lines are drawn inside a feathered mask.
//!
//! Nothing pulses with the beat (the only music link is a slow glow boost).

use crate::presentation::{
    EASE_IN_OUT, Eased, GpuOp, Job, LYRIC_CONTEXT_SCALE, LYRIC_CURRENT, Ready,
};
use mg_gfx::overlay::TextDraw;
use mg_media::lrc::Lyrics;
use mg_text::{LineStyle, TextBitmap};
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
/// Lyric text size as a fraction of the window height, line height and rendered bitmap height in text sizes.
const TEXT_FRAC: f32 = 0.036;
const LINE_H: f32 = 1.4;
const BITMAP_H: f32 = 1.6;
/// First text-texture slot used by lyrics (0 and 1 are title and artist).
pub const FIRST_TEXT_SLOT: usize = 2;

/// Column: longest slide, its share of the previous line's duration, the reduced-motion crossfade (s).
pub const SLIDE_MAX_S: f64 = 0.6;
pub const SLIDE_GAP_FRAC: f64 = 0.45;
pub const RM_FADE_S: f64 = 0.2;
/// A position jump larger than this (beyond the frame time) is a seek.
const SEEK_JUMP_S: f64 = 0.35;
/// Gap between column lines and the mask feather (fractions of the window height).
const COL_GAP: f32 = 0.016;
const COL_FEATHER: f32 = 0.03;
/// A line whose bitmap lands while it is on screen fades in this fast instead of popping.
const SHOWN_S: f32 = 0.15;

/// Album Focus lyric column (surface pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Column {
    pub x: f32,
    pub w: f32,
    /// Vertical centre of the current line.
    pub anchor: f32,
    /// Visible band: ink outside `top..bottom` is masked with a feathered edge.
    pub top: f32,
    pub bottom: f32,
    /// Centre lines in the column (portrait) instead of aligning them to their start edge.
    pub centered: bool,
}

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
    /// Column mode: 0..1 fade-in after the slot received a new line.
    shown: f32,
}

pub struct LyricView {
    source: LyricSource,
    track: Option<u64>,
    lyrics: Option<Arc<Lyrics>>,
    /// The lookup for the current track has answered (or there is nothing to look up).
    resolved: bool,
    position: Option<f64>,
    serial: u64,
    cache: HashMap<usize, Option<TextBitmap>>,
    requested: HashSet<usize>,
    slots: [Slot; 3],
    /// Vertical position of the current line as a fraction of the window height (set per scene).
    anchor: f32,
    /// Extra rise (fraction of the window height) while the control bar shows. Applied after the position
    /// easing, so the lines follow the bar's alpha exactly instead of lagging behind it.
    lift: f32,
    /// Window size the cached line bitmaps were rendered for.
    for_size: (u32, u32),
    column: Option<Column>,
    reduced: bool,
    /// Column mode: previous position (seek detection) and the line a seek resolved to (its slide is skipped).
    last_pos: Option<f64>,
    settled: Option<i64>,
    emph: [f32; 3],
    clip: Option<[f32; 3]>,
}

/// Pixel size of lyric text for a window height.
pub fn lyric_px(h: f32) -> f32 {
    TEXT_FRAC * h
}

/// Lowest ink the lyric stack can reach (as a fraction of the window height) when the current line sits at
/// `anchor`: a line that slides in starts one band below the "next" position (two bands under the current line)
/// and its bitmap is about 1.6 text sizes tall. Used to keep the lyrics clear of the control bar.
pub fn lowest_ink_frac(anchor: f32) -> f32 {
    anchor + (2.0 * BAND * LINE_H + BITMAP_H) * TEXT_FRAC + 0.002
}

fn lyric_max_width(w: f32) -> u32 {
    (0.80 * w).max(64.0) as u32
}

/// Duration (s) of the column slide into line `n`: at most `SLIDE_MAX_S` (`RM_FADE_S` with reduced motion) and
/// at most `SLIDE_GAP_FRAC` of the time since line `n - 1` started, so it always fits inside that line.
pub fn slide_secs(l: &Lyrics, n: usize, reduced: bool) -> f64 {
    let max = if reduced { RM_FADE_S } else { SLIDE_MAX_S };
    let gap = match n.checked_sub(1) {
        Some(p) if n < l.lines.len() => l.lines[n].start_s - l.lines[p].start_s,
        _ => f64::INFINITY,
    };
    max.min(SLIDE_GAP_FRAC * gap).max(1e-3)
}

/// Column state at `pos`: `(a, b, s)` = sliding from line `a` to line `b = a + 1` with raw progress `s` (0..1),
/// or resting on `a` (`a == b`, `s = 1`). Line -1 means "before the first line". A pure function of the position.
pub fn column_phase(l: &Lyrics, pos: f64, reduced: bool) -> (i64, i64, f32) {
    let c = l.at(pos);
    let a = c.current.map_or(-1, |v| v as i64);
    if let Some(n) = c.next {
        let d = slide_secs(l, n, reduced);
        let start = l.lines[n].start_s - d;
        if pos >= start {
            return (a, n as i64, ((pos - start) / d).clamp(0.0, 1.0) as f32);
        }
    }
    (a, a, 1.0)
}

/// One line's placement in a resting column state: top (px), scale, alpha, emphasis.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rest {
    top: f32,
    scale: f32,
    alpha: f32,
    emph: f32,
}

impl LyricView {
    pub fn new(source: LyricSource) -> Self {
        Self {
            resolved: source == LyricSource::None,
            source,
            track: None,
            lyrics: None,
            position: None,
            serial: 0,
            cache: HashMap::new(),
            requested: HashSet::new(),
            slots: Default::default(),
            anchor: 0.72,
            lift: 0.0,
            for_size: (0, 0),
            column: None,
            reduced: false,
            last_pos: None,
            settled: None,
            emph: [1.0; 3],
            clip: None,
        }
    }

    pub fn set_anchor(&mut self, frac: f32) {
        if frac.is_finite() {
            self.anchor = frac.clamp(0.3, 0.92);
        }
    }

    /// Rise above the anchor for the control bar (see `controls::ControlsUi::lyric_lift`).
    pub fn set_lift(&mut self, frac: f32) {
        self.lift = if frac.is_finite() {
            frac.clamp(0.0, 0.3)
        } else {
            0.0
        };
    }

    pub fn set_reduced_motion(&mut self, on: bool) {
        self.reduced = on;
    }

    /// Switches between the classic view (`None`) and the Album Focus column. Changing the kind drops the
    /// rendered lines (they differ in style); a moved or resized column keeps them.
    pub fn set_column(&mut self, c: Option<Column>) {
        let c = c.filter(|c| {
            [c.x, c.w, c.anchor, c.top, c.bottom]
                .iter()
                .all(|v| v.is_finite())
        });
        if c.is_some() != self.column.is_some() {
            self.cache.clear();
            self.requested.clear();
            self.serial += 1;
            self.slots = Default::default();
            self.last_pos = None;
            self.settled = None;
        }
        self.column = c;
    }

    /// The Album Focus column in use (`None` = classic view).
    #[cfg(test)]
    pub fn column(&self) -> Option<Column> {
        self.column
    }

    /// Per slot: 0 = context colour, 1 = current-line colour (classic: always 1).
    pub fn emphasis(&self) -> [f32; 3] {
        self.emph
    }

    /// Column mask (top, bottom, feather) in pixels; `None` in the classic view.
    pub fn clip(&self) -> Option<[f32; 3]> {
        self.clip
    }

    /// Latest playback position in seconds (`None` when the session reports no timeline: no lyrics shown).
    pub fn set_position(&mut self, pos: Option<f64>) {
        self.position = pos.filter(|p| p.is_finite());
    }

    pub fn has_lyrics(&self) -> bool {
        self.lyrics.is_some()
    }

    /// The lyric lookup for the current track has answered (always true without a source).
    pub fn resolved(&self) -> bool {
        self.resolved
    }

    /// A new track (or none): drop everything of the old one and, if a source is configured, ask the worker
    /// for this track's lyrics. Slots keep their textures until they have faded out.
    pub fn on_track(&mut self, track: Option<u64>, title: &str, artist: &str, jobs: &mut Vec<Job>) {
        self.track = track;
        self.lyrics = None;
        self.cache.clear();
        self.requested.clear();
        self.serial += 1;
        self.last_pos = None;
        self.settled = None;
        self.resolved = self.source == LyricSource::None || track.is_none();
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
                self.resolved = true;
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

    /// Asks the worker for line `idx` unless it is cached or on its way (an empty line is cached as nothing).
    fn request(
        &mut self,
        lyr: &Lyrics,
        track: u64,
        idx: usize,
        surface: (u32, u32),
        jobs: &mut Vec<Job>,
    ) {
        if idx >= lyr.lines.len() || self.cache.contains_key(&idx) || self.requested.contains(&idx)
        {
            return;
        }
        let text = &lyr.lines[idx].text;
        if text.is_empty() {
            self.cache.insert(idx, None); // instrumental gap: nothing to draw
            return;
        }
        let (w, h) = (surface.0 as f32, surface.1 as f32);
        let (style, step) = match self.column {
            Some(c) => (
                LineStyle {
                    align: if c.centered {
                        mg_text::Align::Center
                    } else {
                        mg_text::Align::Start
                    },
                    ..LYRIC_CURRENT.style(text, h, c.w)
                },
                LYRIC_CURRENT.step,
            ),
            None => (
                LineStyle {
                    px: lyric_px(h),
                    weight: 500,
                    max_width: lyric_max_width(w),
                    ..Default::default()
                },
                1.0,
            ),
        };
        self.requested.insert(idx);
        self.for_size = surface;
        jobs.push(Job::Line {
            track_id: track,
            serial: self.serial,
            idx,
            text: text.clone(),
            style,
            step,
        });
    }

    /// A resized window needs the lines at a new pixel size.
    fn check_resize(&mut self, surface: (u32, u32)) {
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
        if surface.0 == 0 || surface.1 == 0 {
            return [None; 3];
        }
        self.check_resize(surface);
        if let Some(col) = self.column {
            return self.update_column(col, dt, surface, ops, jobs);
        }
        self.emph = [1.0; 3];
        self.clip = None;
        let (w, h) = (surface.0 as f32, surface.1 as f32);
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
                self.request(&lyr, track, idx, surface, jobs);
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
        let lh = lyric_px(h) * LINE_H;
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
                    y: y - self.lift * h,
                    w: tw as f32,
                    h: th as f32,
                    alpha: slot.alpha,
                });
            }
        }
        out
    }

    /// Ink height (px, scale 1) of line `i`: the rendered bitmap, an estimate while it is on its way, a short
    /// gap for an instrumental line and nothing outside the file.
    fn line_h(&self, i: i64, est: f32) -> f32 {
        let Some(l) = self.lyrics.as_ref() else {
            return 0.0;
        };
        if i < 0 || i as usize >= l.lines.len() {
            return 0.0;
        }
        match self.cache.get(&(i as usize)) {
            Some(Some(b)) => b.height.saturating_sub(2 * b.pad) as f32,
            Some(None) => 0.5 * est,
            None => est,
        }
    }

    /// Where line `i` rests while line `c` is current: the current line centred on the anchor at full size,
    /// the next one below at context scale, the one after it (hidden) further down, the previous one (hidden)
    /// above. Every state is a stack with `gap` between ink boxes, so interpolating two states never overlaps.
    fn rest(&self, c: i64, i: i64, col: &Column, est: f32, gap: f32) -> Option<Rest> {
        let k = LYRIC_CONTEXT_SCALE;
        let hc = self.line_h(c, est);
        let cur_top = col.anchor - 0.5 * hc;
        let next_top = cur_top + hc + gap;
        Some(match i - c {
            0 => Rest {
                top: cur_top,
                scale: 1.0,
                alpha: 1.0,
                emph: 1.0,
            },
            1 => Rest {
                top: next_top,
                scale: k,
                alpha: 1.0,
                emph: 0.0,
            },
            2 => Rest {
                top: next_top + self.line_h(c + 1, est) * k + gap,
                scale: k,
                alpha: 0.0,
                emph: 0.0,
            },
            -1 => Rest {
                top: cur_top - gap - self.line_h(i, est),
                scale: 1.0,
                alpha: 0.0,
                emph: 1.0,
            },
            _ => return None,
        })
    }

    /// Album Focus column: see the module docs. Everything on screen follows from the position.
    fn update_column(
        &mut self,
        col: Column,
        dt: f32,
        surface: (u32, u32),
        ops: &mut Vec<GpuOp>,
        jobs: &mut Vec<Job>,
    ) -> [Option<TextDraw>; 3] {
        let h = surface.1 as f32;
        self.clip = Some([col.top, col.bottom, COL_FEATHER * h]);
        self.emph = [0.0; 3];
        let (Some(lyr), Some(pos), Some(track)) = (self.lyrics.clone(), self.position, self.track)
        else {
            self.last_pos = None;
            return [None; 3];
        };
        // a jump of the timeline (seek, first frame) resolves at once: a slide in progress there is skipped
        let jumped = self
            .last_pos
            .is_none_or(|p| (pos - p).abs() > SEEK_JUMP_S + f64::from(dt));
        self.last_pos = Some(pos);
        let (a, b, s) = column_phase(&lyr, pos, self.reduced);
        if jumped && a != b {
            self.settled = Some(b);
        }
        if self.settled.is_some_and(|t| t != b) {
            self.settled = None;
        }
        let (a, s) = if self.settled == Some(b) {
            (b, 1.0)
        } else {
            (a, s)
        };
        // bounded memory and the lines that matter: the window a..=a+2 plus one prefetched line
        let first = a.max(0) as usize;
        if self.cache.len() > 12 {
            self.cache.retain(|&k, _| k.abs_diff(first) <= 4);
            self.requested.retain(|&k| k.abs_diff(first) <= 4);
        }
        for idx in first..first + 4 {
            self.request(&lyr, track, idx, surface, jobs);
        }
        let last = (a + 2).min(lyr.lines.len() as i64 - 1);
        for i in a.max(0)..=last {
            let idx = i as usize;
            let slot = &mut self.slots[idx % 3];
            if slot.line == Some(idx) && !slot.size_stale {
                continue;
            }
            if let Some(entry) = self.cache.get(&idx) {
                match entry {
                    Some(b) => {
                        slot.dims = Some((b.width, b.height, b.pad));
                        ops.push(GpuOp::SetText(FIRST_TEXT_SLOT + idx % 3, b.clone()));
                    }
                    None => {
                        if slot.dims.take().is_some() {
                            ops.push(GpuOp::ClearText(FIRST_TEXT_SLOT + idx % 3));
                        }
                    }
                }
                if slot.line != Some(idx) {
                    slot.shown = 0.0;
                }
                slot.line = Some(idx);
                slot.size_stale = false;
            }
        }
        for slot in &mut self.slots {
            slot.shown = (slot.shown + dt / SHOWN_S).min(1.0);
        }
        // reduced motion: no movement; the column switches state at the midpoint of a 200 ms crossfade
        let (sp, fade) = if self.reduced {
            (if s >= 0.5 { 1.0 } else { 0.0 }, (2.0 * s - 1.0).abs())
        } else {
            (EASE_IN_OUT.at(s), 1.0)
        };
        let est = LYRIC_CURRENT.size * h * 1.15;
        let gap = COL_GAP * h;
        let mut out = [None; 3];
        for i in a.max(0)..=last {
            let (Some(r0), Some(r1)) = (
                self.rest(a, i, &col, est, gap),
                self.rest(b, i, &col, est, gap),
            ) else {
                continue;
            };
            let lerp = |x: f32, y: f32| x + (y - x) * sp;
            let (top, scale, emph) = (
                lerp(r0.top, r1.top),
                lerp(r0.scale, r1.scale),
                lerp(r0.emph, r1.emph),
            );
            let idx = i as usize;
            let slot = &self.slots[idx % 3];
            let Some((tw, th, pad)) = slot.dims.filter(|_| slot.line == Some(idx)) else {
                continue;
            };
            let alpha = lerp(r0.alpha, r1.alpha) * fade * slot.shown;
            if alpha <= 0.002 {
                continue;
            }
            let ink = tw.saturating_sub(2 * pad) as f32 * scale;
            let rtl = mg_text::is_rtl(&lyr.lines[idx].text);
            let x_ink = if col.centered {
                col.x + 0.5 * (col.w - ink)
            } else if rtl {
                col.x + col.w - ink
            } else {
                col.x
            };
            let p = pad as f32 * scale;
            out[idx % 3] = Some(TextDraw {
                x: x_ink - p,
                y: top - p,
                w: tw as f32 * scale,
                h: th as f32 * scale,
                alpha,
            });
            self.emph[idx % 3] = emph;
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
                .any(|j| matches!(j, Job::Line { style, .. } if style.max_width <= 720))
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

    /// The whole chain the engine runs each frame (bar visibility -> lift -> real LyricView layout), sampled every
    /// 20 ms through the 180 ms show and the 400 ms hide: whenever the bar is visible (alpha >= 0.05) no lyric
    /// ink reaches its zone, at several sizes, DPI factors, anchors and lyric positions (incl. line changes).
    #[test]
    fn lyrics_clear_the_bar_at_every_alpha_through_show_and_hide() {
        use crate::controls::{HIDE_FADE_S, IDLE_S, Visibility, lyric_lift, ui_scale};
        const DT: f32 = 0.02;
        for s in [(1920u32, 1080u32), (1280, 720), (720, 1280), (800, 600)] {
            for dpi in [1.0f32, 1.5, 2.0] {
                for anchor in [0.70f32, 0.72, 0.80] {
                    let zone_top = s.1 as f32 - 128.0 * ui_scale(s, dpi);
                    let need = lyric_lift(anchor, s, dpi);
                    // positions just before demo line changes (2, 6, 10 s) and mid-line
                    for start in [1.9f64, 5.95, 8.0, 9.9] {
                        let mut v = LyricView::new(LyricSource::Demo);
                        v.set_anchor(anchor);
                        v.on_track(Some(1), "T", "A", &mut Vec::new());
                        v.on_ready(Ready::Lyrics {
                            track_id: 1,
                            lyrics: Some(lyr()),
                        });
                        let mut t = start - 1.5;
                        while t < start {
                            frame(&mut v, t, s);
                            t += 1.0 / 60.0;
                        }
                        let mut bar = Visibility::new(false);
                        let step = |bar: &mut Visibility, v: &mut LyricView, t: &mut f64| {
                            bar.tick(DT);
                            v.set_lift(need * bar.lift());
                            // (the lyric animation advances 1/60 s per sample, the song clock 20 ms: harmless here)
                            let o = frame(v, *t, s);
                            *t += f64::from(DT);
                            (bar.alpha(), lowest(&o, 2.0))
                        };
                        let mut samples = Vec::new();
                        bar.activity();
                        for _ in 0..=((0.18 / DT) as usize + 1) {
                            samples.push(step(&mut bar, &mut v, &mut t));
                        }
                        // idle until the fade-out starts, then sample the whole 400 ms hide
                        while bar.wants_visible() || bar.alpha() > 0.0 {
                            samples.push(step(&mut bar, &mut v, &mut t));
                            assert!(samples.len() < ((IDLE_S + HIDE_FADE_S + 1.0) / DT) as usize);
                        }
                        let visible: Vec<_> = samples.iter().filter(|(a, _)| *a >= 0.05).collect();
                        assert!(visible.len() > 10, "sampled the transitions");
                        for (a, low) in visible {
                            assert!(
                                *low <= zone_top,
                                "{s:?} dpi {dpi} anchor {anchor} start {start}: alpha {a:.3} ink {low:.1} > zone {zone_top:.1}"
                            );
                        }
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

#[cfg(test)]
mod column_tests {
    use super::*;

    const S: (u32, u32) = (1920, 1080);
    const COL: Column = Column {
        x: 960.0,
        w: 700.0,
        anchor: 453.6,
        top: 194.4,
        bottom: 590.0,
        centered: false,
    };

    /// Lines at 1, 5, 6.2 (a rapid line), 6.5, 12 s; an instrumental gap at 9 s.
    fn sample() -> Arc<Lyrics> {
        Arc::new(
            mg_media::lrc::parse(
                b"[00:01.00]one\n[00:05.00]two\n[00:06.20]three\n[00:06.50]four\n[00:09.00]\n[00:12.00]six\n",
            )
            .unwrap(),
        )
    }

    fn view(l: Arc<Lyrics>, reduced: bool) -> LyricView {
        let mut v = LyricView::new(LyricSource::Demo);
        v.set_reduced_motion(reduced);
        v.set_column(Some(COL));
        v.on_track(Some(1), "T", "A", &mut Vec::new());
        v.on_ready(Ready::Lyrics {
            track_id: 1,
            lyrics: Some(l),
        });
        v
    }

    /// One frame at `pos`; line jobs answered at once with a one-line bitmap of the requested size.
    fn frame(v: &mut LyricView, pos: f64) -> [Option<TextDraw>; 3] {
        let (mut ops, mut jobs) = (Vec::new(), Vec::new());
        v.set_position(Some(pos));
        let out = v.update(1.0 / 60.0, S, &mut ops, &mut jobs);
        for j in jobs {
            if let Job::Line {
                track_id,
                serial,
                idx,
                style,
                ..
            } = j
            {
                let (w, h) = ((style.px * 8.0) as u32, (style.px * 1.15) as u32 + 4);
                v.on_ready(Ready::Line {
                    track_id,
                    serial,
                    idx,
                    bitmap: Some(TextBitmap {
                        width: w,
                        height: h,
                        pad: 2,
                        coverage: vec![1; (w * h) as usize],
                    }),
                });
            }
        }
        out
    }

    fn settle(v: &mut LyricView, pos: f64) -> [Option<TextDraw>; 3] {
        let mut o = [None; 3];
        for _ in 0..30 {
            o = frame(v, pos);
        }
        o
    }

    fn mid(d: &TextDraw) -> f32 {
        d.y + 0.5 * d.h
    }

    #[test]
    fn the_slide_adapts_to_the_gap_and_ends_on_the_line_timestamp() {
        let l = sample();
        for n in 1..l.lines.len() {
            let gap = l.lines[n].start_s - l.lines[n - 1].start_s;
            let d = slide_secs(&l, n, false);
            assert!(
                d <= SLIDE_MAX_S + 1e-9 && d <= SLIDE_GAP_FRAC * gap + 1e-9,
                "line {n}: {d} for gap {gap}"
            );
            assert!(d < gap, "a slide never exceeds the gap");
            assert!(slide_secs(&l, n, true) <= RM_FADE_S + 1e-9);
            // the slide into n runs over [t_n - d, t_n]: at t_n it is complete and n is current
            let t = l.lines[n].start_s;
            assert_eq!(column_phase(&l, t, false), (n as i64, n as i64, 1.0));
            let (a, b, s) = column_phase(&l, t - 0.5 * d, false);
            assert_eq!((a, b), (n as i64 - 1, n as i64));
            assert!((s - 0.5).abs() < 1e-3);
            assert_eq!(
                column_phase(&l, t - d - 0.01, false).2,
                1.0,
                "rest before the slide"
            );
        }
        // the rapid line (0.3 s gap) slides in 135 ms, a regular one takes the full 600 ms
        assert!((slide_secs(&l, 3, false) - 0.135).abs() < 1e-6);
        assert!((slide_secs(&l, 1, false) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn playing_through_line_changes_keeps_lines_ordered_apart_and_in_the_column() {
        let l = sample();
        let mut v = view(l.clone(), false);
        let mut t = 0.0f64;
        settle(&mut v, t);
        while t < 13.0 {
            let o = frame(&mut v, t);
            let vis: Vec<_> = o.iter().flatten().filter(|d| d.alpha > 0.02).collect();
            for (i, a) in vis.iter().enumerate() {
                assert!(a.x + 2.0 >= COL.x - 0.5 && a.x + a.w - 2.0 <= COL.x + COL.w + 0.5);
                for b in &vis[i + 1..] {
                    let ov = (a.y + a.h - 2.0).min(b.y + b.h - 2.0) - (a.y + 2.0).max(b.y + 2.0);
                    assert!(ov <= 0.5, "t {t:.2}: lines overlap by {ov:.1}px {o:?}");
                }
            }
            // the line being sung is the emphasised one, in place, from its timestamp until the next line's slide
            // starts (never late: the slide into a line ends at its timestamp)
            let (a, b, _) = column_phase(&l, t, false);
            if let Some(c) = l
                .at(t)
                .current
                .filter(|&c| a == b && !l.lines[c].text.is_empty())
            {
                let d = o[c % 3].expect("current line drawn");
                assert!(
                    v.emphasis()[c % 3] > 0.99,
                    "t {t:.2}: line {c} not current yet"
                );
                assert!(
                    (mid(&d) - COL.anchor).abs() < 1.0,
                    "t {t:.2}: current line at {}",
                    mid(&d)
                );
            }
            t += 1.0 / 60.0;
        }
        assert_eq!(v.clip(), Some([COL.top, COL.bottom, COL_FEATHER * 1080.0]));
    }

    #[test]
    fn a_seek_resolves_straight_to_the_position_without_replaying_a_slide() {
        let l = sample();
        let mut v = view(l.clone(), false);
        settle(&mut v, 2.0);
        // seek into the slide towards line 1 (window 4.4..5.0 s): its end state shows on the first frame
        let o = frame(&mut v, 4.7);
        let o = if o[1].is_some() {
            o
        } else {
            frame(&mut v, 4.7)
        }; // the bitmap may need one frame
        let d = o[1].expect("line 1 drawn");
        assert!(v.emphasis()[1] > 0.99 && (mid(&d) - COL.anchor).abs() < 1.0);
        // and it stays there (no queued motion) until the next slide starts
        let mut t = 4.72;
        while t < 5.5 {
            let o = frame(&mut v, t);
            let d = o[1].expect("line 1");
            assert!((mid(&d) - COL.anchor).abs() < 1.0, "t {t:.2} moved");
            t += 1.0 / 60.0;
        }
        // a backwards seek: the first frame already shows the earlier state, nothing of the later one
        let o = frame(&mut v, 1.5);
        let d = o[0].expect("line 0 current");
        assert!(v.emphasis()[0] > 0.99 && (mid(&d) - COL.anchor).abs() < 1.0);
        assert!(o[2].is_none(), "no line from the old position");
    }

    #[test]
    fn reduced_motion_crossfades_lines_without_moving_them() {
        let mut v = view(sample(), true);
        settle(&mut v, 3.0);
        let mut seen: [Option<f32>; 3] = [None; 3];
        let mut t = 4.5f64;
        while t < 5.5 {
            let o = frame(&mut v, t);
            for (k, d) in o.iter().enumerate() {
                if let Some(d) = d.filter(|d| d.alpha > 0.05) {
                    // a line only changes place while invisible (the crossfade dips through zero)
                    if let Some(y) = seen[k] {
                        assert!((d.y - y).abs() < 0.5, "t {t:.2}: slot {k} moved");
                    }
                    seen[k] = Some(d.y);
                } else {
                    seen[k] = None;
                }
            }
            t += 1.0 / 60.0;
        }
    }

    #[test]
    fn switching_view_kind_drops_the_old_lines_and_classic_is_unchanged() {
        let mut v = view(sample(), false);
        settle(&mut v, 2.0);
        assert!(v.clip().is_some());
        v.set_column(None);
        let (mut ops, mut jobs) = (Vec::new(), Vec::new());
        v.set_position(Some(2.0));
        let o = v.update(1.0 / 60.0, S, &mut ops, &mut jobs);
        assert!(
            o.iter().all(Option::is_none),
            "column bitmaps are not reused by the classic view"
        );
        assert!(v.clip().is_none() && v.emphasis() == [1.0; 3]);
        assert!(
            !jobs.is_empty()
                && jobs
                    .iter()
                    .all(|j| matches!(j, Job::Line { style, .. } if style.max_lines == 1))
        );
    }
}
