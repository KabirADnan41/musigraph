//! Track presentation: now-playing metadata -> artwork, title/artist/meta text, album palette and layout.
//!
//! Split in two so the rules are testable without a GPU or a media session:
//! * `Presentation` (pure, no I/O): reacts to track changes, accepts ready results only for the CURRENT
//!   track (a late result for a previous track is dropped), runs the track-change choreography (old text out,
//!   artwork crossfade, staggered text entrance; every tween retargets from its current value), places the
//!   composition for the window shape, tints text from the album palette and produces `OverlayParams`.
//! * `Worker` (one background thread): decodes artwork (bounded), extracts the palette and renders text
//!   bitmaps, so nothing here ever blocks the render loop. Latest request wins; superseded ones are skipped.
//!
//! Album Focus (the Focus scene with the Center layout) is the art-anchored composition of
//! `docs/design/ALBUM-FOCUS-SPEC-2026-11-27.md`: geometry in `album_geom`, type roles in `Role`, the lyric column
//! in `lyric_view::Column`. Other scenes keep the compact Corner layout (or the legacy Center one).
//!
//! Privacy: titles, artists and artwork are shown on screen only. They are never logged, written to disk
//! or sent anywhere.

use crate::lyric_view::{Column, LyricSource, LyricView};
use mg_gfx::overlay::{ArtDraw, META_SLOT, OverlayParams, TextColors, TextDraw};
use mg_gfx::scene::{ART_GRID, ArtLight, FocusParams};
use mg_media::lrc::Lyrics;
use mg_media::palette::{Palette, to_oklab};
use mg_media::{ArtImage, NowPlaying};
use mg_text::{Align, Cut, LineStyle, TextBitmap};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// ---- tunable motion (seconds; distances as fractions of the window height) ---------------------------

/// How long the album palette takes to move to a new track's colours (OKLab).
pub const PALETTE_SECS: f32 = 3.0;
/// Old text leaves: strong ease-out, rising a little.
pub const TEXT_OUT_S: f32 = 0.22;
pub const TEXT_OUT_DY: f32 = -0.008;
/// New text enters: title, artist and meta start this long after the track change (60 ms stagger), each
/// taking `TEXT_IN_S` while settling from `TEXT_IN_DY` below.
pub const TEXT_ENTER_AT: [f32; 3] = [0.30, 0.36, 0.42];
pub const TEXT_STAGGER_S: f32 = 0.06;
pub const TEXT_IN_S: f32 = 0.36;
pub const TEXT_IN_DY: f32 = 0.010;
/// The new cover fades in over the (held) previous one.
pub const ART_XFADE_S: f32 = 0.70;
/// A stale cover leaves this fast while no new cover is there, so it is gone before the new text enters.
pub const ART_OUT_S: f32 = 0.28;
/// Reduced motion: no movement or blur; covers fade over 300 ms, text over 200 ms.
pub const RM_ART_S: f32 = 0.30;
pub const RM_TEXT_S: f32 = 0.20;
/// A track change waits at most this long for the lyric lookup before choosing the no-lyrics composition
/// (shorter than the title's entrance delay, so the wait is never visible).
pub const LYRICS_WAIT_S: f32 = 0.28;
/// A resized window gets new text bitmaps once its size has been stable this long.
const RESIZE_SETTLE_S: f32 = 0.15;
const LAYOUT_EASE_S: f32 = 0.22;

/// Blur bridge: sampling LOD of the artwork at crossfade progress `t` (0..1), `2·sin(πt)` (spec): sharp at both ends,
/// two mip levels soft in the middle, where the two unrelated pictures overlap most (emil-design-eng: blur masks an
/// imperfect crossfade). Needs the mip chain `art_extra` makes; the caller passes 0 in reduced motion.
pub fn art_blur_lod(t: f32) -> f32 {
    if t.is_finite() {
        2.0 * (std::f32::consts::PI * t.clamp(0.0, 1.0)).sin()
    } else {
        0.0
    }
}

// ---- easing -----------------------------------------------------------------------------------------

/// CSS-style cubic Bézier timing curve through (0,0), (x1,y1), (x2,y2), (1,1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curve(pub f32, pub f32, pub f32, pub f32);

/// Strong ease-out for entrances and exits (animate skill: `cubic-bezier(0.23, 1, 0.32, 1)`).
pub const EASE_OUT: Curve = Curve(0.23, 1.0, 0.32, 1.0);
/// Strong ease-in-out for on-screen movement (the lyric column slide).
pub const EASE_IN_OUT: Curve = Curve(0.77, 0.0, 0.175, 1.0);

impl Curve {
    /// y for progress x (0..1). x(t) is monotonic for x1, x2 in 0..1, so a fixed bisection is exact enough
    /// (2^-24) and cannot fail to converge.
    pub fn at(self, x: f32) -> f32 {
        if x.is_nan() || x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let bez = |a: f32, b: f32, t: f32| {
            let u = 1.0 - t;
            3.0 * u * u * t * a + 3.0 * u * t * t * b + t * t * t
        };
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if bez(self.0, self.2, mid) < x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        bez(self.1, self.3, 0.5 * (lo + hi))
    }
}

/// A time-based tween. `retarget` starts a new move from wherever the value is NOW, so an interrupted
/// animation never jumps (a transition, not a keyframe).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tween {
    from: f32,
    to: f32,
    t: f32,
    delay: f32,
    dur: f32,
    curve: Curve,
}

impl Tween {
    pub(crate) fn at(v: f32) -> Self {
        Self {
            from: v,
            to: v,
            t: 0.0,
            delay: 0.0,
            dur: 0.0,
            curve: EASE_OUT,
        }
    }

    pub(crate) fn value(&self) -> f32 {
        let k = if self.dur > 0.0 {
            (self.t - self.delay) / self.dur
        } else if self.t >= self.delay {
            1.0
        } else {
            0.0
        };
        self.from + (self.to - self.from) * self.curve.at(k)
    }

    pub(crate) fn target(&self) -> f32 {
        self.to
    }

    pub(crate) fn retarget(&mut self, to: f32, dur: f32, delay: f32, curve: Curve) {
        *self = Self {
            from: self.value(),
            to,
            t: 0.0,
            delay: delay.max(0.0),
            dur: dur.max(0.0),
            curve,
        };
    }

    pub(crate) fn step(&mut self, dt: f32) {
        self.t = (self.t + dt).min(self.delay + self.dur + 1.0);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Compact artwork and text in the lower-left corner; the visualisation stays dominant.
    Corner,
    /// Large artwork; in the Focus scene this is the Album Focus composition.
    Center,
    /// Nothing drawn over the scene (the palette still follows the album).
    Off,
}

impl Layout {
    pub fn next(self) -> Self {
        match self {
            Layout::Corner => Layout::Center,
            Layout::Center => Layout::Off,
            Layout::Off => Layout::Corner,
        }
    }
}

/// What is actually drawn: the layout, specialised by scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    Corner,
    Center,
    Album,
    Off,
}

// ---- type roles -------------------------------------------------------------------------------------

/// One typographic role: size (fraction of the window height), weight, optical cut, tracking (em), line
/// height, how many lines it may wrap to and the size step tried before a word ellipsis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Role {
    pub size: f32,
    pub weight: u16,
    pub display: bool,
    pub tracking: f32,
    pub line_height: f32,
    pub lines: u8,
    pub step: f32,
}

pub const TITLE_LYRICS: Role = Role {
    size: 0.036,
    weight: 600,
    display: true,
    tracking: -0.015,
    line_height: 1.10,
    lines: 2,
    step: 0.85,
};
pub const TITLE_SOLO: Role = Role {
    size: 0.058,
    ..TITLE_LYRICS
};
pub const ARTIST_LYRICS: Role = Role {
    size: 0.022,
    weight: 400,
    display: false,
    tracking: 0.0,
    line_height: 0.0,
    lines: 1,
    step: 1.0,
};
pub const ARTIST_SOLO: Role = Role {
    size: 0.028,
    ..ARTIST_LYRICS
};
pub const META: Role = Role {
    size: 0.017,
    ..ARTIST_LYRICS
};
pub const LYRIC_CURRENT: Role = Role {
    size: 0.044,
    weight: 600,
    display: true,
    tracking: -0.012,
    line_height: 1.15,
    lines: 2,
    step: 0.88,
};
/// Context lyric lines are the current-line bitmap drawn at this scale (0.028 H / 0.044 H), so a line grows
/// into place instead of being swapped for a second rendering.
pub const LYRIC_CONTEXT_SCALE: f32 = 0.028 / 0.044;
const CORNER_TITLE: Role = Role {
    size: 0.030,
    line_height: 0.0,
    lines: 1,
    step: 1.0,
    ..TITLE_LYRICS
};
const CORNER_ARTIST: Role = Role {
    size: 0.019,
    ..ARTIST_LYRICS
};
const CENTER_TITLE: Role = Role {
    size: 0.046,
    ..CORNER_TITLE
};
const CENTER_ARTIST: Role = Role {
    size: 0.030,
    ..ARTIST_LYRICS
};
/// Height of one unwrapped DirectWrite line box (Segoe UI ascent + descent) in multiples of the font size, and
/// the share of it below the baseline.
const LINE_BOX: f32 = 1.33;
const SINGLE_DESCENT: f32 = 0.19;

/// Smallest text drawn (better-typography: UI text rarely below 12 px; keeps the meta line legible in small windows).
pub const MIN_PX: f32 = 12.0;

/// Bangla is drawn 8 % larger with looser lines, CJK with line height 1.3 (mg-text drops tracking outside
/// Latin/Greek/Cyrillic by itself). Returns (size factor, minimum line height).
pub fn script_tune(s: &str) -> (f32, f32) {
    let (mut bangla, mut cjk) = (false, false);
    for c in s.chars() {
        match c as u32 {
            0x0980..=0x09FF => bangla = true,
            0x3040..=0x30FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xAC00..=0xD7AF
            | 0xF900..=0xFAFF => cjk = true,
            _ => {}
        }
    }
    if bangla {
        (1.08, 1.35)
    } else if cjk {
        (1.0, 1.3)
    } else {
        (1.0, 0.0)
    }
}

impl Role {
    /// The DirectWrite style for `text` at window height `h`. The Display cut is used only at 24 px and up
    /// (better-typography: optical cuts by size).
    pub fn style(&self, text: &str, h: f32, max_width: f32) -> LineStyle {
        let (k, lh) = script_tune(text);
        let px = (self.size * h * k).max(MIN_PX);
        LineStyle {
            px,
            weight: self.weight,
            max_width: max_width.max(64.0) as u32,
            max_lines: self.lines,
            tracking_em: self.tracking,
            line_height: if self.line_height > 0.0 || lh > 0.0 {
                self.line_height.max(lh)
            } else {
                0.0
            },
            cut: if self.display && px >= 24.0 {
                Cut::Display
            } else {
                Cut::Text
            },
            ..LineStyle::default()
        }
    }
}

// ---- worker protocol --------------------------------------------------------------------------------

/// One text item of the title block: what to draw, how, and the size step tried before an ellipsis.
pub struct TextItem {
    pub text: String,
    pub style: LineStyle,
    pub step: f32,
}

pub struct TextJob {
    pub track_id: u64,
    pub serial: u64,
    /// Title, artist, meta.
    pub items: [Option<TextItem>; 3],
}

/// A rendered text block and what the layout needs to know about it.
#[derive(Clone, Debug, PartialEq)]
pub struct Fitted {
    pub bitmap: TextBitmap,
    /// Font size actually used (after a size step) and the number of lines.
    pub px: f32,
    pub lines: u32,
    /// Rendered with uniform line spacing (wrapping allowed) and its line height.
    pub wrap: bool,
    pub line_height: f32,
    pub rtl: bool,
}

pub enum Job {
    Art {
        track_id: u64,
        bytes: Arc<[u8]>,
        /// Build the blur-bridge mip chain too (off in reduced motion: the bridge is not drawn).
        mips: bool,
    },
    Text(TextJob),
    /// Find and parse the user-supplied lyrics file for a track (or build the demo lyrics).
    Lyrics {
        track_id: u64,
        source: LyricSource,
        title: String,
        artist: String,
    },
    /// Render one lyric line (`step` < 1: one size step before a word ellipsis).
    Line {
        track_id: u64,
        serial: u64,
        idx: usize,
        text: String,
        style: LineStyle,
        step: f32,
    },
    /// Render one control-bar time string (slot 0 elapsed, 1 duration).
    Clock {
        serial: u64,
        slot: usize,
        text: String,
        px: f32,
    },
}

pub enum Ready {
    Art {
        track_id: u64,
        image: Option<ArtImage>,
        palette: Option<Palette>,
        /// Mips and room-light colours of `image`, made on the worker (never on the render thread).
        extra: Option<ArtExtra>,
    },
    Text {
        track_id: u64,
        serial: u64,
        title: Option<Fitted>,
        artist: Option<Fitted>,
        meta: Option<Fitted>,
    },
    Lyrics {
        track_id: u64,
        lyrics: Option<Arc<Lyrics>>,
    },
    Line {
        track_id: u64,
        serial: u64,
        idx: usize,
        bitmap: Option<TextBitmap>,
    },
    Clock {
        serial: u64,
        slot: usize,
        bitmap: Option<TextBitmap>,
    },
}

/// What the GPU side must do (applied by the render loop; uploads happen once per track change).
pub enum GpuOp {
    /// The current cover becomes the previous one (crossfade source).
    ShiftArt,
    /// Replaces the current cover (with its mips for the overlay and its room-light colours for the scene).
    SetArt(ArtImage, Option<ArtExtra>),
    ClearArt,
    ClearPrevArt,
    SetText(usize, TextBitmap),
    ClearText(usize),
}

// ---- palette ----------------------------------------------------------------------------------------

fn peak(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2])
}

/// Scales a colour so its largest component equals `to` (keeps hue and chroma ratios; near-black stays
/// neutral rather than exploding into noise).
fn normalise(c: [f32; 3], to: f32) -> [f32; 3] {
    let m = peak(c);
    if m < 1e-3 {
        return [to * 0.6; 3];
    }
    c.map(|v| (v * to / m).clamp(0.0, 1.0))
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

/// Maps the album palette roles onto the scene's four colour slots (linear RGB):
/// haze/base <- bg tinted by secondary, bar body <- main, accent/tips <- accent lifted toward highlight,
/// secondary <- secondary. Brightness is normalised per slot so every album lands at a comparable overall
/// level and only hue and saturation change from song to song.
pub fn scene_roles(p: &Palette) -> [[f32; 3]; 4] {
    [
        normalise(lerp3(p.bg, p.secondary, 0.35), 0.15),
        normalise(p.main, 0.62),
        normalise(lerp3(p.accent, p.highlight, 0.3), 0.92),
        normalise(p.secondary, 0.5),
    ]
}

/// Blends the scene's own (no-artwork) palette toward the album roles by `w` (0..1).
pub fn apply_album(scene: &mut [[f32; 3]; 4], album: &Palette, w: f32, k: f32) {
    let roles = scene_roles(album);
    let w = w.clamp(0.0, 1.0);
    for (s, r) in scene.iter_mut().zip(roles) {
        *s = lerp3(*s, r.map(|v| v * k), w);
    }
}

/// OKLab -> linear sRGB (Björn Ottosson's matrices; mg-media keeps its own copy private).
fn oklab_to_linear(c: [f32; 3]) -> [f32; 3] {
    let l = (c[0] + 0.396_337_78 * c[1] + 0.215_803_76 * c[2]).powi(3);
    let m = (c[0] - 0.105_561_35 * c[1] - 0.063_854_17 * c[2]).powi(3);
    let s = (c[0] - 0.089_484_18 * c[1] - 1.291_485_5 * c[2]).powi(3);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_4 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
    .map(|v| {
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    })
}

/// OKLab lightness per text role (spec typography table): readable on the dark environment at any palette.
pub const L_TITLE: f32 = 0.93;
pub const L_ARTIST: f32 = 0.80;
pub const L_META: f32 = 0.62;
pub const L_LYRIC: f32 = 0.95;
pub const L_CONTEXT: f32 = 0.66;
/// Text tint chroma (better-colors: a tinted neutral holds one hue across its steps).
const TEXT_CHROMA: f32 = 0.025;

/// Text colours for a palette: fixed OKLab lightness per role, a light tint at the most colourful of the
/// highlight/accent/main hues (never plain grey), and the ambient shadow at the background's hue (L 0.05).
pub fn role_colors(p: &Palette) -> TextColors {
    let lab = |c: [f32; 3]| {
        let v = to_oklab(c);
        if v.iter().all(|x| x.is_finite()) {
            v
        } else {
            [0.0; 3]
        }
    };
    let chroma = |v: [f32; 3]| v[1].hypot(v[2]);
    let src = [p.highlight, p.accent, p.main]
        .map(lab)
        .into_iter()
        .fold(
            [0.0, 0.0, 0.0],
            |a, b| if chroma(b) > chroma(a) { b } else { a },
        );
    let hue = if chroma(src) > 1e-4 {
        src[2].atan2(src[1])
    } else {
        1.2 // warm neutral when the art has no colour at all
    };
    let tint = |l: f32| oklab_to_linear([l, TEXT_CHROMA * hue.cos(), TEXT_CHROMA * hue.sin()]);
    let bg = lab(p.bg);
    let bc = chroma(bg).min(0.02);
    let bh = bg[2].atan2(bg[1]);
    TextColors {
        title: tint(L_TITLE),
        artist: tint(L_ARTIST),
        meta: tint(L_META),
        lyric: tint(L_LYRIC),
        context: tint(L_CONTEXT),
        shade: oklab_to_linear([0.05, bc * bh.cos(), bc * bh.sin()]),
    }
}

struct PaletteState {
    from: Palette,
    to: Palette,
    cur: Palette,
    t: f32,
    w: f32,
    w_target: f32,
}

impl PaletteState {
    fn new() -> Self {
        let f = Palette::fallback();
        Self {
            from: f,
            to: f,
            cur: f,
            t: 1.0,
            w: 0.0,
            w_target: 0.0,
        }
    }

    /// New album colours (or none): the move starts from what is on screen right now, so a change in the
    /// middle of a transition stays continuous.
    fn set_target(&mut self, p: Option<Palette>) {
        match p {
            Some(p) => {
                self.from = self.cur;
                self.to = p;
                self.t = 0.0;
                self.w_target = 1.0;
            }
            None => self.w_target = 0.0,
        }
    }

    fn update(&mut self, dt: f32) {
        self.t = (self.t + dt / PALETTE_SECS).min(1.0);
        let s = self.t * self.t * (3.0 - 2.0 * self.t);
        self.cur = self.from.mix(&self.to, s);
        let step = dt / PALETTE_SECS;
        self.w += (self.w_target - self.w).clamp(-step, step);
    }
}

// ---- Album Focus geometry -------------------------------------------------------------------------

/// Outer margin, gap above the control-bar zone, gap between groups, gap inside a group (fractions of H).
pub const MARGIN: f32 = 0.04;
pub const ZONE_GAP: f32 = 0.02;
pub const GROUP_GAP: f32 = 0.05;
pub const IN_GAP: f32 = 0.012;
/// Top band kept empty above the lyric column and the smallest useful lyric column.
const LYRIC_TOP: f32 = 0.18;
const LYRIC_MIN_H: f32 = 0.16;
/// Landscape needs at least this aspect; below 1.65 the 16:10 / 4:3 rules apply (16:10 = 1.6 is listed with 4:3
/// in the spec); below 1.15 the portrait stack.
const WIDE_ASPECT: f32 = 1.65;
const PORTRAIT_ASPECT: f32 = 1.15;

/// Where Album Focus puts things for one window (pixels). The cover slot exists even without a cover: the
/// text-led composition keeps the same grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlbumGeom {
    /// Cover slot centre x, centre y and side.
    pub art: [f32; 3],
    pub col_x: f32,
    pub col_w: f32,
    /// Lyric column (x, width): the text column in landscape, 0.84 W centred in portrait.
    pub lyric: [f32; 2],
    /// Portrait: one centred column under the cover.
    pub portrait: bool,
}

/// Height (px) the title block can reach: a two-line title, artist and meta.
pub fn block_max_h(h: f32, lyrics: bool) -> f32 {
    let (t, a) = if lyrics {
        (TITLE_LYRICS, ARTIST_LYRICS)
    } else {
        (TITLE_SOLO, ARTIST_SOLO)
    };
    h * (2.0 * t.size * t.line_height
        + IN_GAP
        + a.size * LINE_BOX
        + 0.5 * IN_GAP
        + META.size * LINE_BOX)
}

/// Album Focus geometry (spec "Composition"): landscape stage min(W, 1.9H) with the cover at (0.31 Sw, 0.46 H),
/// 0.56 H; 16:10 and 4:3 with min(0.52 H, 0.40 W) at 0.29 W; portrait min(0.62 W, 0.44 H) from 0.10 H. The cover
/// shrinks (never clips) to keep everything above `zone_top` (the control-bar zone) and inside the margin.
/// The column width does not depend on whether a cover is present, so a late cover never re-wraps the text.
pub fn album_geom(w: f32, h: f32, zone_top: f32, lyrics: bool) -> AlbumGeom {
    let aspect = w / h.max(1.0);
    let bottom = (h * (1.0 - MARGIN)).min(zone_top - ZONE_GAP * h);
    if aspect >= PORTRAIT_ASPECT {
        let sw = w.min(1.9 * h);
        let x0 = 0.5 * (w - sw);
        let (mut s, cx) = if aspect >= WIDE_ASPECT {
            (0.56 * h, x0 + 0.31 * sw)
        } else {
            ((0.52 * h).min(0.40 * w), 0.29 * w)
        };
        let mut cy = 0.46 * h;
        // with lyrics the column above the title block needs room: the cover's bottom (= the block's last
        // baseline) may not rise above this
        let min_bottom = if lyrics {
            (LYRIC_TOP + LYRIC_MIN_H + GROUP_GAP) * h + block_max_h(h, true)
        } else {
            0.0
        };
        if cy + 0.5 * s > bottom {
            s = s
                .min(bottom - MARGIN * h - 0.04 * h)
                .min(2.0 * (bottom - cy).max(0.0))
                .max(0.2 * h);
            cy = (bottom - 0.5 * s).min(cy);
        }
        if cy + 0.5 * s < min_bottom && min_bottom <= bottom {
            cy = min_bottom - 0.5 * s;
        }
        // the cover's left edge never leaves the margin
        let cx = cx.max(MARGIN * h + 0.5 * s);
        let col_x = cx + 0.5 * s + 0.06 * h;
        let col_w = (0.38 * sw)
            .min(x0 + sw - col_x - 0.10 * sw)
            .min(w - MARGIN * h - col_x);
        if aspect >= WIDE_ASPECT || col_w >= 0.30 * w {
            return AlbumGeom {
                art: [cx, cy, s],
                col_x,
                col_w,
                lyric: [col_x, col_w],
                portrait: false,
            };
        }
    }
    // portrait (or a column under 0.30 W): cover centred from 0.10 H, text stacked below
    let top = 0.10 * h;
    let need = 0.045 * h
        + block_max_h(h, lyrics)
        + if lyrics {
            (GROUP_GAP + LYRIC_MIN_H) * h
        } else {
            0.0
        };
    let s = (0.62 * w)
        .min(0.44 * h)
        .min(bottom - top - need)
        .max(0.12 * h);
    // without lyrics the cover + title group is centred vertically (no dead bottom third)
    let top = if lyrics {
        top
    } else {
        (0.5 * (bottom - s - need)).max(top)
    };
    // with lyrics the title block is narrower than the lyrics (0.84 W) so the cover stays the largest element
    let col_w = if lyrics { 0.64 * w } else { 0.84 * w };
    AlbumGeom {
        art: [0.5 * w, top + 0.5 * s, s],
        col_x: 0.5 * (w - col_w),
        col_w,
        lyric: [0.08 * w, 0.84 * w],
        portrait: true,
    }
}

/// Top of the title block (px). Landscape with lyrics: its last baseline sits on the cover's bottom edge;
/// landscape without lyrics: centred on the cover; portrait: 0.045 H under the cover (text-led: centred in the
/// cover slot, or near its top when lyrics follow).
pub fn album_block_top(
    g: &AlbumGeom,
    h: f32,
    has_art: bool,
    lyrics: bool,
    block_h: f32,
    last_desc: f32,
) -> f32 {
    let [_, cy, s] = g.art;
    if g.portrait {
        return match (has_art, lyrics) {
            (true, _) => cy + 0.5 * s + 0.045 * h,
            // text-led: the title takes the cover's place (centred in its slot), lyrics follow below
            (false, false) => cy - 0.5 * block_h,
            (false, true) => cy - 0.5 * s + 0.08 * h,
        };
    }
    if lyrics {
        cy + 0.5 * s - (block_h - last_desc)
    } else {
        cy - 0.5 * block_h
    }
}

/// The lyric column for a placed title block (`block` = top, bottom in px).
pub fn album_lyric_column(g: &AlbumGeom, h: f32, zone_top: f32, block: (f32, f32)) -> Column {
    let bottom_limit = (h * (1.0 - MARGIN)).min(zone_top - ZONE_GAP * h);
    if g.portrait {
        let top = block.1 + GROUP_GAP * h;
        let bottom = bottom_limit.max(top + 0.05 * h);
        Column {
            x: g.lyric[0],
            w: g.lyric[1],
            anchor: top + 0.40 * (bottom - top),
            top,
            bottom,
            centered: true,
        }
    } else {
        let top = LYRIC_TOP * h;
        let bottom = (block.0 - GROUP_GAP * h).max(top + 0.05 * h);
        Column {
            x: g.lyric[0],
            w: g.lyric[1],
            anchor: (0.42 * h).min(bottom - 0.12 * h).max(top + 0.06 * h),
            top,
            bottom,
            centered: false,
        }
    }
}

// ---- the state machine ------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Eased(f32, bool);

impl Eased {
    pub(crate) fn to(&mut self, target: f32, dt: f32, snap: bool) -> f32 {
        if snap || !self.1 || !self.0.is_finite() {
            self.0 = target;
            self.1 = true;
        } else {
            self.0 += (target - self.0) * (1.0 - (-dt / LAYOUT_EASE_S).exp());
        }
        self.0
    }
}

/// An uploaded text bitmap and its metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Dims {
    w: u32,
    h: u32,
    pad: u32,
    /// Height of the line boxes (bitmap without padding) and the part of it below the last baseline.
    box_h: f32,
    desc: f32,
    rtl: bool,
}

impl Dims {
    fn of(f: &Fitted) -> Self {
        let b = &f.bitmap;
        let box_h = b.height.saturating_sub(2 * b.pad) as f32;
        let desc = if f.wrap {
            0.2 * f.line_height.max(0.8) * f.px
        } else {
            SINGLE_DESCENT * box_h
        };
        Self {
            w: b.width,
            h: b.height,
            pad: b.pad,
            box_h,
            desc,
            rtl: f.rtl,
        }
    }
    fn ink_w(&self) -> f32 {
        self.w.saturating_sub(2 * self.pad) as f32
    }
}

/// One element of the title block (title, artist, meta).
#[derive(Clone, Copy, Debug)]
struct TextEl {
    dims: Option<Dims>,
    alpha: Tween,
    /// Vertical offset, fraction of the window height.
    dy: Tween,
    x: Eased,
    y: Eased,
}

impl Default for TextEl {
    fn default() -> Self {
        Self {
            dims: None,
            alpha: Tween::at(0.0),
            dy: Tween::at(0.0),
            x: Eased::default(),
            y: Eased::default(),
        }
    }
}

/// One artwork texture (0 = current, 1 = previous) as the CPU sees it.
#[derive(Clone, Copy, Debug)]
struct ArtLayer {
    aspect: f32,
    /// Linear fade 0..1 (drawn through a smoothstep).
    alpha: f32,
    /// False once the track it belongs to has ended.
    want: bool,
    rect: [Eased; 4],
}

const TEXT_SLOT: [usize; 3] = [0, 1, META_SLOT];

pub struct Presentation {
    layout: Layout,
    album_scene: bool,
    reduced_motion: bool,
    track_id: Option<u64>,
    /// Display strings (title, artist, meta); the raw ones only go to the lyric lookup.
    shown: [String; 3],
    art_requested: bool,
    /// The current track's cover is being decoded (the layout keeps the cover slot).
    art_expected: bool,
    serial: u64,
    /// The current track still needs its first text request (it may wait briefly for the lyric lookup).
    need_text: bool,
    /// Album Focus with the lyric column for this track (decided at the track's first text request).
    lyric_variant: bool,
    text_for: (u32, u32),
    seen_surface: (u32, u32),
    settled_for: f32,
    /// Time since the last track change (drives the text entrance schedule).
    since_change: f32,
    jobs: Vec<Job>,
    ops: Vec<GpuOp>,
    art: [Option<ArtLayer>; 2],
    art_pending: Option<Option<(ArtImage, Option<ArtExtra>)>>,
    els: [TextEl; 3],
    text_pending: Option<[Option<Fitted>; 3]>,
    palette: PaletteState,
    lyrics: LyricView,
    lyric_draws: [Option<TextDraw>; 3],
    /// User switch: lyrics are still looked up and tracked, only their drawing is suppressed.
    lyrics_on: bool,
    /// Control-bar zone top (px from the top; 0 = unknown) and how far the bar is shown (0..1).
    zone_top: f32,
    bar_vis: f32,
    /// Cover slot (centre x, y, side) of the last composition, and the room-light inputs derived from the last
    /// frame (see `merge_focus`).
    slot: [f32; 3],
    focus: FocusParams,
}

impl Presentation {
    pub fn new(layout: Layout) -> Self {
        Self {
            layout,
            album_scene: false,
            reduced_motion: false,
            track_id: None,
            shown: Default::default(),
            art_requested: false,
            art_expected: false,
            serial: 0,
            need_text: false,
            lyric_variant: false,
            text_for: (0, 0),
            seen_surface: (0, 0),
            settled_for: 0.0,
            since_change: 0.0,
            jobs: Vec::new(),
            ops: Vec::new(),
            art: [None, None],
            art_pending: None,
            els: [TextEl::default(); 3],
            text_pending: None,
            palette: PaletteState::new(),
            lyrics: LyricView::new(LyricSource::None),
            lyric_draws: [None; 3],
            lyrics_on: true,
            zone_top: 0.0,
            bar_vis: 0.0,
            slot: [0.0; 3],
            focus: FocusParams::default(),
        }
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    fn style(&self) -> Style {
        match self.layout {
            Layout::Off => Style::Off,
            Layout::Corner => Style::Corner,
            Layout::Center if self.album_scene => Style::Album,
            Layout::Center => Style::Center,
        }
    }

    pub fn set_layout(&mut self, l: Layout, surface: (u32, u32)) {
        if l != self.layout {
            self.layout = l;
            self.recompose(surface);
        }
    }

    /// The scene in front is Album Focus' scene (the Center layout becomes the Album Focus composition).
    pub fn set_album_scene(&mut self, on: bool, surface: (u32, u32)) {
        if on != self.album_scene {
            let before = self.style();
            self.album_scene = on;
            if self.style() != before {
                self.recompose(surface);
            }
        }
    }

    /// Reduced motion: no movement or blur, shorter fades.
    pub fn set_reduced_motion(&mut self, on: bool) {
        self.reduced_motion = on;
        self.lyrics.set_reduced_motion(on);
    }

    /// Control-bar zone top in pixels (the bar's area including its progress line) and the bar's visibility
    /// (0..1, following its fade). Album Focus keeps clear of the zone; the Corner block rises over it.
    pub fn set_bar(&mut self, zone_top: f32, visible: f32) {
        if zone_top.is_finite() {
            self.zone_top = zone_top.max(0.0);
        }
        self.bar_vis = if visible.is_finite() {
            visible.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    pub fn take_jobs(&mut self) -> Vec<Job> {
        std::mem::take(&mut self.jobs)
    }

    pub fn lyrics_on(&self) -> bool {
        self.lyrics_on
    }

    /// Shows or hides the lyrics (returns the new state). In Album Focus this is a user decision, so the
    /// composition follows at once (keyboard-initiated: no animation).
    pub fn toggle_lyrics(&mut self) -> bool {
        self.lyrics_on = !self.lyrics_on;
        if self.style() == Style::Album && self.track_id.is_some() {
            self.lyric_variant = self.lyrics_on && self.lyrics.has_lyrics();
            let s = self.text_for;
            self.request_text(s);
        }
        self.lyrics_on
    }

    pub fn set_lyric_source(&mut self, s: LyricSource) {
        self.lyrics = LyricView::new(s);
        self.lyrics.set_reduced_motion(self.reduced_motion);
    }

    /// Playback position in seconds for the lyrics (None when unknown or nothing plays).
    pub fn set_position(&mut self, pos: Option<f64>) {
        self.lyrics.set_position(pos);
    }

    /// Vertical anchor of the current lyric line (fraction of the window height), set per scene.
    pub fn set_lyric_anchor(&mut self, frac: f32) {
        self.lyrics.set_anchor(frac);
    }

    /// Extra rise of the lyrics above the anchor while the control bar shows (follows the bar's alpha, unlagged).
    pub fn set_lyric_lift(&mut self, frac: f32) {
        self.lyrics.set_lift(frac);
    }

    pub fn take_ops(&mut self) -> Vec<GpuOp> {
        std::mem::take(&mut self.ops)
    }

    /// Current album palette and how strongly it applies (0 = scene's own colours).
    pub fn album(&self) -> (Palette, f32) {
        (self.palette.cur, self.palette.w)
    }

    /// Same-track re-render for another style or variant: new bitmaps swap in at once.
    fn recompose(&mut self, surface: (u32, u32)) {
        if self.style() == Style::Album {
            self.lyric_variant = self.lyrics_on && self.lyrics.has_lyrics();
        }
        if !self.need_text {
            self.request_text(surface);
        }
    }

    /// Text column width for the current style (independent of the cover, see `album_geom`).
    fn column_width(&self, w: f32, h: f32) -> f32 {
        match self.style() {
            Style::Album => album_geom(w, h, self.zone(h), self.lyric_variant).col_w,
            Style::Center => 0.80 * w,
            _ => (0.44 * w).min(1.6 * h),
        }
    }

    fn zone(&self, h: f32) -> f32 {
        if self.zone_top > 0.0 {
            self.zone_top
        } else {
            h
        }
    }

    fn roles(&self) -> [Role; 3] {
        match self.style() {
            Style::Album if self.lyric_variant => [TITLE_LYRICS, ARTIST_LYRICS, META],
            Style::Album => [TITLE_SOLO, ARTIST_SOLO, META],
            Style::Center => [CENTER_TITLE, CENTER_ARTIST, META],
            _ => [CORNER_TITLE, CORNER_ARTIST, META],
        }
    }

    fn request_text(&mut self, surface: (u32, u32)) {
        if self.track_id.is_none() || self.shown[0].is_empty() || surface.0 == 0 || surface.1 == 0 {
            return;
        }
        let (w, h) = (surface.0 as f32, surface.1 as f32);
        let col = self.column_width(w, h);
        let roles = self.roles();
        // the meta line belongs to Album Focus only; the compact layouts keep title and artist
        let with_meta = self.style() == Style::Album;
        // centred blocks (portrait, legacy Center) centre the lines of a wrapped title inside their bitmap too
        let centred = match self.style() {
            Style::Album => album_geom(w, h, self.zone(h), self.lyric_variant).portrait,
            Style::Center => true,
            _ => false,
        };
        let items = std::array::from_fn(|i| {
            let s = &self.shown[i];
            (!s.trim().is_empty() && (i < 2 || with_meta)).then(|| TextItem {
                text: s.clone(),
                style: LineStyle {
                    align: if centred { Align::Center } else { Align::Start },
                    ..roles[i].style(s, h, col)
                },
                step: roles[i].step,
            })
        });
        self.serial += 1;
        self.text_for = surface;
        self.jobs.push(Job::Text(TextJob {
            track_id: self.track_id.unwrap_or(0),
            serial: self.serial,
            items,
        }));
    }

    /// The first text request of a track; in Album Focus it waits (briefly) for the lyric lookup, because
    /// the composition with and without lyrics differs and is decided once per track.
    fn maybe_first_text(&mut self, surface: (u32, u32)) {
        let waiting = self.style() == Style::Album
            && !self.lyrics.resolved()
            && self.since_change < LYRICS_WAIT_S;
        if self.need_text && surface.0 != 0 && surface.1 != 0 && !waiting {
            self.lyric_variant = self.lyrics_on && self.lyrics.has_lyrics();
            self.need_text = false;
            self.request_text(surface);
        }
    }

    /// Feeds the latest now-playing snapshot (or `None` when nothing plays). Cheap; call a few times a second.
    pub fn on_now_playing(&mut self, np: Option<&NowPlaying>, surface: (u32, u32)) {
        let id = np.map(|n| n.track_id);
        if id != self.track_id {
            // New track (or none): the old text leaves, the old cover starts fading, and late results for
            // the old track are ignored (track id + serial).
            self.track_id = id;
            self.art_pending = None;
            self.text_pending = None;
            self.art_requested = false;
            self.art_expected = false;
            self.serial += 1;
            self.text_for = (0, 0);
            self.since_change = 0.0;
            self.palette.set_target(None);
            for l in self.art.iter_mut().flatten() {
                l.want = false;
            }
            let (out_s, curve) = (self.text_time(TEXT_OUT_S), EASE_OUT);
            let out_dy = if self.reduced_motion {
                0.0
            } else {
                TEXT_OUT_DY
            };
            for e in &mut self.els {
                let dy = e.dy.value();
                e.alpha.retarget(0.0, out_s, 0.0, curve);
                e.dy.retarget(
                    if self.reduced_motion { dy } else { out_dy },
                    out_s,
                    0.0,
                    curve,
                );
            }
            let (t, a) = np.map_or((String::new(), String::new()), |n| {
                (n.title.clone(), n.artist.clone())
            });
            self.lyrics.on_track(id, &t, &a, &mut self.jobs);
            self.shown = Default::default();
            self.need_text = false;
            if let Some(n) = np {
                let d = mg_media::display::display_meta(&n.title, &n.artist);
                let meta = [n.album.trim(), d.version.trim()]
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(" · ");
                self.shown = [d.title, d.artist, meta];
                self.need_text = true;
                self.maybe_first_text(surface);
                if let Some(bytes) = &n.art {
                    self.art_requested = true;
                    self.art_expected = true;
                    self.jobs.push(Job::Art {
                        track_id: n.track_id,
                        bytes: bytes.clone(),
                        mips: !self.reduced_motion,
                    });
                }
            }
        } else if let Some(n) = np {
            // Same track: artwork may arrive late (the session loads it after the title).
            if !self.art_requested {
                if let Some(bytes) = &n.art {
                    self.art_requested = true;
                    self.art_expected = true;
                    self.jobs.push(Job::Art {
                        track_id: n.track_id,
                        bytes: bytes.clone(),
                        mips: !self.reduced_motion,
                    });
                }
            }
        }
    }

    /// Accepts a worker result. Results for any other track (or a superseded text request) are dropped.
    pub fn on_ready(&mut self, r: Ready) {
        match r {
            r @ (Ready::Lyrics { .. } | Ready::Line { .. }) => self.lyrics.on_ready(r),
            Ready::Clock { .. } => {} // the control bar owns these (controls.rs)
            Ready::Art {
                track_id,
                image,
                palette,
                extra,
            } => {
                if Some(track_id) != self.track_id {
                    return;
                }
                self.palette.set_target(palette);
                self.art_pending = Some(image.map(|i| (i, extra)));
            }
            Ready::Text {
                track_id,
                serial,
                title,
                artist,
                meta,
            } => {
                if Some(track_id) != self.track_id || serial != self.serial {
                    return;
                }
                self.text_pending = Some([title, artist, meta]);
            }
        }
    }

    fn text_time(&self, s: f32) -> f32 {
        if self.reduced_motion { RM_TEXT_S } else { s }
    }

    /// New cover ready, crossfade bookkeeping and fades. The cover never scales or moves with the music.
    fn step_art(&mut self, dt: f32) {
        if let Some(p) = self.art_pending.take() {
            self.art_expected = false;
            if let Some((img, extra)) = p {
                let aspect = img.width as f32 / img.height.max(1) as f32;
                // Of the two textures keep the one that contributes more to the picture as the crossfade
                // source; the other (an obsolete, mostly invisible transition) is dropped.
                let keep_cur = match (self.art[0], self.art[1]) {
                    (Some(c), Some(p)) => c.alpha >= p.alpha * (1.0 - c.alpha),
                    (Some(_), None) => true,
                    (None, _) => false,
                };
                if keep_cur {
                    self.ops.push(GpuOp::ShiftArt);
                    self.art[1] = self.art[0].take();
                }
                self.ops.push(GpuOp::SetArt(img, extra));
                self.art[0] = Some(ArtLayer {
                    aspect,
                    alpha: 0.0,
                    want: true,
                    rect: [Eased::default(); 4],
                });
            }
        }
        let rm = self.reduced_motion;
        let rate_in = dt / if rm { RM_ART_S } else { ART_XFADE_S };
        let rate_out = dt / if rm { RM_ART_S } else { ART_OUT_S };
        let held = self.art[0].is_some_and(|c| c.want);
        if let Some(c) = &mut self.art[0] {
            c.alpha = (c.alpha + if c.want { rate_in } else { -rate_out }).clamp(0.0, 1.0);
        }
        if let Some(p) = &mut self.art[1] {
            if !held {
                p.alpha = (p.alpha - rate_out).max(0.0);
            }
        }
        let done = self.art[0].is_some_and(|c| c.want && c.alpha >= 1.0);
        if self.art[1].is_some_and(|p| done || p.alpha <= 0.0) {
            self.art[1] = None;
            self.ops.push(GpuOp::ClearPrevArt);
        }
        if self.art[0].is_some_and(|c| !c.want && c.alpha <= 0.0) {
            self.art[0] = None;
            self.ops.push(GpuOp::ClearArt);
        }
    }

    /// New text bitmaps: a re-render of what is shown swaps in at once; after a track change they wait until
    /// the old text has left, then enter on the stagger schedule.
    fn step_text(&mut self, dt: f32) {
        for e in &mut self.els {
            e.alpha.step(dt);
            e.dy.step(dt);
        }
        let shown = self.els.iter().any(|e| e.alpha.target() > 0.0);
        let gone = self.els.iter().all(|e| e.alpha.value() <= 0.001);
        if self.text_pending.is_none() || !(shown || gone) {
            return;
        }
        let items = self.text_pending.take().unwrap_or_default();
        let rm = self.reduced_motion;
        let quick = self.text_time(TEXT_IN_S);
        for (i, f) in items.into_iter().enumerate() {
            let e = &mut self.els[i];
            match f {
                Some(f) => {
                    e.dims = Some(Dims::of(&f));
                    self.ops.push(GpuOp::SetText(TEXT_SLOT[i], f.bitmap));
                    if !shown {
                        let delay =
                            (TEXT_ENTER_AT[i] - self.since_change).max(TEXT_STAGGER_S * i as f32);
                        let dur = if rm { RM_TEXT_S } else { TEXT_IN_S };
                        e.alpha = Tween::at(0.0);
                        e.alpha.retarget(1.0, dur, delay, EASE_OUT);
                        e.dy = Tween::at(if rm { 0.0 } else { TEXT_IN_DY });
                        e.dy.retarget(0.0, dur, delay, EASE_OUT);
                    } else if e.alpha.target() <= 0.0 {
                        // a re-render that adds an element (e.g. a meta line) fades it in quickly
                        e.alpha.retarget(1.0, quick, 0.0, EASE_OUT);
                    }
                    e.x = Eased::default(); // new metrics: place at once
                    e.y = Eased::default();
                }
                None => {
                    if e.dims.take().is_some() {
                        self.ops.push(GpuOp::ClearText(TEXT_SLOT[i]));
                    }
                    e.alpha = Tween::at(0.0);
                }
            }
        }
    }

    /// Advances fades, tweens and palette; emits GPU ops for content that is ready; returns the overlay to
    /// draw. `energy` (0..1) only drives the slow lyric glow; nothing else reacts to the music here.
    pub fn update(
        &mut self,
        dt: f32,
        surface: (u32, u32),
        energy: f32,
        _swell: f32,
    ) -> OverlayParams {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.25)
        } else {
            0.0
        };
        self.palette.update(dt);
        self.since_change += dt;
        if surface != self.seen_surface {
            self.seen_surface = surface;
            self.settled_for = 0.0;
        } else {
            self.settled_for += dt;
        }
        self.maybe_first_text(surface);
        // a resized window gets text at the new pixel size once the size has settled
        if !self.need_text
            && self.text_for != (0, 0)
            && surface.0 != 0
            && surface.1 != 0
            && surface != self.text_for
            && self.settled_for >= RESIZE_SETTLE_S
        {
            self.request_text(surface);
        }
        self.step_art(dt);
        self.step_text(dt);

        let style = self.style();
        let (w, h) = (surface.0 as f32, surface.1 as f32);
        let mut out = OverlayParams {
            res: surface,
            energy,
            lyric_glow: energy.clamp(0.0, 1.0),
            lyric_emph: [1.0; 3],
            shadow: if style == Style::Album { 0.25 } else { 0.55 },
            colors: role_colors(&self.palette.cur),
            accent: lerp3(
                Palette::fallback().secondary,
                lerp3(self.palette.cur.main, self.palette.cur.accent, 0.5),
                self.palette.w,
            ),
            ..Default::default()
        };
        if surface.0 == 0 || surface.1 == 0 {
            self.lyric_draws = [None; 3];
            return out;
        }
        let has_art = self.art_expected || self.art[0].is_some_and(|c| c.want);
        let column = if style == Style::Off {
            None
        } else {
            self.place(style, w, h, has_art, dt, &mut out)
        };
        self.lyrics.set_column(column);
        self.lyric_draws = self
            .lyrics
            .update(dt, surface, &mut self.ops, &mut self.jobs);
        let hide = !self.lyrics_on || (style == Style::Album && !self.lyric_variant);
        if !hide {
            out.lyrics = self.lyric_draws;
            out.lyric_emph = self.lyrics.emphasis();
            out.lyric_clip = self.lyrics.clip();
        }
        self.focus = focus_feed(&out, self.slot);
        out
    }

    /// Copies the room-light inputs of the last `update` (light rectangle, cover weights, text pocket) into `f`;
    /// the music fields, the bar and the cover colours are not touched.
    pub fn merge_focus(&self, f: &mut FocusParams) {
        let g = &self.focus;
        f.rect = g.rect;
        f.art_w = g.art_w;
        f.ink = g.ink;
        f.ink_cap = g.ink_cap;
        f.ink_on = g.ink_on;
    }

    /// Places cover and title block for `style`, writes them into `out` and returns the lyric column
    /// (Album Focus with lyrics only).
    fn place(
        &mut self,
        style: Style,
        w: f32,
        h: f32,
        has_art: bool,
        dt: f32,
        out: &mut OverlayParams,
    ) -> Option<Column> {
        let zone = self.zone(h);
        // cover slot (centre, side), text column and alignment
        let (slot, col_x, col_w, centered, geom) = match style {
            Style::Album => {
                let g = album_geom(w, h, zone, self.lyric_variant);
                let (col_x, centered) = if g.portrait {
                    (g.col_x, true)
                } else if has_art {
                    (g.col_x, false)
                } else {
                    // text-led: the column starts on the cover's left edge (one shared grid edge)
                    (g.art[0] - 0.5 * g.art[2], false)
                };
                (g.art, col_x, g.col_w, centered, Some(g))
            }
            Style::Center => {
                let side = 0.36 * h;
                ([0.5 * w, 0.37 * h, side], 0.1 * w, 0.8 * w, true, None)
            }
            _ => {
                let (m, side) = (0.055 * h, 0.20 * h);
                // rise over the control-bar zone while the bar shows (follows the bar's own 180 ms fade)
                let need = ((h - m) - (zone - ZONE_GAP * h)).max(0.0) * self.bar_vis;
                let raise = need;
                let cy = h - m - 0.5 * side - raise;
                let left = if has_art { m + side + 0.03 * h } else { m };
                (
                    [m + 0.5 * side, cy, side],
                    left,
                    (0.44 * w).min(1.6 * h),
                    false,
                    None,
                )
            }
        };
        // ---- covers (each layer keeps its own rect; a leaving cover stays where it was) ----
        let [cx, cy, side] = slot;
        self.slot = slot;
        let crossfading = self.art[1].is_some();
        // blur bridge: both pictures soften together while they overlap (never in reduced motion)
        let bridge = if crossfading && !self.reduced_motion {
            art_blur_lod(self.art[0].map_or(1.0, |c| c.alpha))
        } else {
            0.0
        };
        for (k, layer) in self.art.iter_mut().enumerate() {
            let Some(l) = layer else { continue };
            let a = l.aspect.clamp(0.2, 5.0);
            let (sw, sh) = if a >= 1.0 { (1.0, 1.0 / a) } else { (a, 1.0) };
            let target = [cx, cy, 0.5 * side * sw, 0.5 * side * sh];
            let snap = l.alpha < 0.01;
            let r: [f32; 4] = std::array::from_fn(|i| {
                if l.want {
                    l.rect[i].to(target[i], dt, snap)
                } else {
                    l.rect[i].0 // a leaving cover stays where it is
                }
            });
            let t = l.alpha;
            let draw = ArtDraw {
                cx: r[0],
                cy: r[1],
                hw: r[2],
                hh: r[3],
                alpha: t * t * (3.0 - 2.0 * t),
                radius: 0.02 * 2.0 * r[2].max(r[3]),
                shadow: 1.0,
                lod: bridge,
            };
            if k == 0 {
                out.art = Some(draw);
            } else {
                out.art_prev = Some(draw);
            }
        }
        // ---- title block: stacked by real line boxes, start-aligned (RTL lines to the block's start edge) ----
        let present: Vec<usize> = (0..3).filter(|&i| self.els[i].dims.is_some()).collect();
        let lyrics = style == Style::Album && self.lyric_variant;
        let (mut block_h, mut last_desc, mut block_w) = (0.0f32, 0.0f32, 0.0f32);
        for (n, &i) in present.iter().enumerate() {
            let Some(d) = self.els[i].dims else { continue };
            if n > 0 {
                block_h += if i == 2 { 0.5 * IN_GAP * h } else { IN_GAP * h };
            }
            block_h += d.box_h;
            last_desc = d.desc;
            block_w = block_w.max(d.ink_w());
        }
        if present.is_empty() {
            block_h = block_max_h(h, lyrics) * 0.6;
        }
        let top = match (style, geom) {
            (Style::Album, Some(g)) => album_block_top(&g, h, has_art, lyrics, block_h, last_desc),
            (Style::Center, _) => {
                if has_art {
                    cy + 0.5 * side + 0.045 * h
                } else {
                    0.40 * h
                }
            }
            _ => {
                if has_art {
                    cy - 0.5 * block_h
                } else {
                    cy + 0.5 * side - block_h
                }
            }
        };
        let mut y = top;
        for (n, &i) in present.iter().enumerate() {
            let e = &mut self.els[i];
            let Some(d) = e.dims else { continue };
            if n > 0 {
                y += if i == 2 { 0.5 * IN_GAP * h } else { IN_GAP * h };
            }
            let ink = d.ink_w();
            let x_ink = if centered {
                col_x + 0.5 * (col_w - ink)
            } else if d.rtl {
                col_x + block_w - ink
            } else {
                col_x
            };
            let leaving = e.alpha.target() <= 0.0;
            let snap = e.alpha.value() < 0.01;
            // a leaving element keeps its place (only its own exit offset moves it)
            let x = if leaving {
                e.x.0
            } else {
                e.x.to(x_ink - d.pad as f32, dt, snap)
            };
            let yy = if leaving {
                e.y.0
            } else {
                e.y.to(y - d.pad as f32, dt, snap)
            };
            let alpha = e.alpha.value().clamp(0.0, 1.0);
            if alpha > 0.002 {
                let draw = TextDraw {
                    x,
                    y: yy + e.dy.value() * h,
                    w: d.w as f32,
                    h: d.h as f32,
                    alpha,
                };
                match i {
                    0 => out.title = Some(draw),
                    1 => out.artist = Some(draw),
                    _ => out.meta = Some(draw),
                }
            }
            y += d.box_h;
        }
        match (style, geom) {
            (Style::Album, Some(g)) if lyrics => {
                Some(album_lyric_column(&g, h, zone, (top, top + block_h)))
            }
            _ => None,
        }
    }
}

// ---- room light inputs (Album Focus environment) -------------------------------------------------------

/// Required contrast of each role over the environment (spec "Contrast": title and current lyric 7:1, artist
/// 4.5:1, context and meta 3:1) and the headroom kept for dither and shadow.
const RATIO_TITLE: f32 = 7.0;
const RATIO_ARTIST: f32 = 4.5;
const RATIO_CONTEXT: f32 = 3.0;
const POCKET_HEADROOM: f32 = 0.8;
/// The pocket never caps above this relative luminance (spec: Y <= 0.06), nor below `POCKET_MIN_Y` (never a scrim).
const POCKET_MAX_Y: f32 = 0.06;
const POCKET_MIN_Y: f32 = 0.01;

fn rel_lum(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Largest background luminance that still gives `ratio`:1 for text of relative luminance `text_y` (WCAG form
/// (Yt + 0.05) / (Yb + 0.05)), with headroom, clamped to the pocket range.
pub fn pocket_cap(text_y: f32, ratio: f32) -> f32 {
    (((text_y + 0.05) / ratio - 0.05) * POCKET_HEADROOM).clamp(POCKET_MIN_Y, POCKET_MAX_Y)
}

/// The room-light inputs the presentation owns, in screen heights (top-left origin): the light source rectangle
/// (the drawn cover, or the empty cover slot as a soft point source), the weights of the current and previous
/// cover colours, and up to four text ink rectangles with the luminance cap each one needs and their own alpha.
/// Music fields, bar visibility and the cover colours themselves are added by the runner / the scene.
pub fn focus_feed(out: &OverlayParams, slot: [f32; 3]) -> FocusParams {
    let h = out.res.1.max(1) as f32;
    let mut f = FocusParams {
        rect: match out.art.or(out.art_prev) {
            Some(a) => [a.cx / h, a.cy / h, a.hw / h, a.hh / h],
            None => [
                slot[0] / h,
                slot[1] / h,
                0.5 * slot[2] / h,
                0.5 * slot[2] / h,
            ],
        },
        ..FocusParams::default()
    };
    let a0 = out.art.map_or(0.0, |a| a.alpha.clamp(0.0, 1.0));
    let a1 = out.art_prev.map_or(0.0, |a| a.alpha.clamp(0.0, 1.0));
    f.art_w = [a0, a1 * (1.0 - a0)];
    let c = &out.colors;
    let rect = |d: &TextDraw| [d.x / h, d.y / h, (d.x + d.w) / h, (d.y + d.h) / h];
    let mut n = 0;
    for (d, col, ratio) in [
        (out.title, c.title, RATIO_TITLE),
        (out.artist, c.artist, RATIO_ARTIST),
        (out.meta, c.meta, RATIO_CONTEXT),
    ] {
        if let Some(d) = d.filter(|d| d.alpha > 0.002) {
            f.ink[n] = rect(&d);
            f.ink_cap[n] = pocket_cap(rel_lum(col), ratio);
            f.ink_on[n] = d.alpha.clamp(0.0, 1.0);
            n += 1;
        }
    }
    // the lyric column: x from the drawn lines, y from the column mask when there is one (the slide moves lines
    // inside it); the cap serves the strictest role in it (current 7:1, context 3:1)
    let lines: Vec<&TextDraw> = out
        .lyrics
        .iter()
        .flatten()
        .filter(|d| d.alpha > 0.002)
        .collect();
    if let Some(first) = lines.first() {
        let mut r = rect(first);
        for d in &lines[1..] {
            let q = rect(d);
            r = [
                r[0].min(q[0]),
                r[1].min(q[1]),
                r[2].max(q[2]),
                r[3].max(q[3]),
            ];
        }
        if let Some(clip) = out.lyric_clip {
            r[1] = clip[0] / h;
            r[3] = clip[1] / h;
        }
        f.ink[n] = r;
        f.ink_cap[n] = pocket_cap(rel_lum(c.lyric), RATIO_TITLE)
            .min(pocket_cap(rel_lum(c.context), RATIO_CONTEXT));
        f.ink_on[n] = lines
            .iter()
            .fold(0.0f32, |m, d| m.max(d.alpha))
            .clamp(0.0, 1.0);
    }
    f
}

// ---- worker thread ----------------------------------------------------------------------------------

pub struct Worker {
    tx: Option<Sender<Job>>,
    rx: Receiver<Ready>,
    thread: Option<JoinHandle<()>>,
    done: Arc<std::sync::atomic::AtomicBool>,
}

/// Extra levels of the blur-bridge mip chain: the bridge samples up to lod 2.
pub const ART_MIP_LEVELS: usize = 2;

/// What the worker derives from a decoded cover beyond its pixels (never computed on the render thread).
#[derive(Clone, Debug, PartialEq)]
pub struct ArtExtra {
    /// Box-filtered mip levels 1.. (RGBA8, sRGB-encoded, averaged in linear light); empty when not needed.
    pub mips: Vec<Vec<u8>>,
    /// The room-light colours of the cover.
    pub light: Box<ArtLight>,
}

fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let c = i as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        })
    })
}

fn linear_to_srgb8(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let c = if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0 + 0.5) as u8
}

/// Mip levels 1..=`levels` of `img` (RGBA8 sRGB) and the room-light colours (8x8 box grid + mean, linear RGB),
/// from one linearised copy of the image. Memory: one f32 copy of the cover (<= 4 MB at the 512 px limit) that is
/// dropped before returning. `None` for a malformed image.
pub fn art_extra(img: &ArtImage, levels: usize) -> Option<ArtExtra> {
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || img.rgba.len() != w * h * 4 {
        return None;
    }
    let lut = srgb_to_linear_lut();
    let mut cur: Vec<f32> = Vec::with_capacity(w * h * 4);
    for p in img.rgba.chunks_exact(4) {
        cur.extend_from_slice(&[
            lut[usize::from(p[0])],
            lut[usize::from(p[1])],
            lut[usize::from(p[2])],
            f32::from(p[3]) / 255.0,
        ]);
    }
    // 8x8 grid: each cell is the box average of its pixel range (at least one pixel for tiny covers)
    let n = ART_GRID;
    let mut grid = [[0.0f32; 3]; ART_GRID * ART_GRID];
    for (i, cell) in grid.iter_mut().enumerate() {
        let (cx, cy) = (i % n, i / n);
        let (x0, y0) = (cx * w / n, cy * h / n);
        let (x1, y1) = (
            ((cx + 1) * w / n).max(x0 + 1),
            ((cy + 1) * h / n).max(y0 + 1),
        );
        let mut acc = [0.0f32; 3];
        for y in y0..y1.min(h) {
            for x in x0..x1.min(w) {
                let o = (y * w + x) * 4;
                for c in 0..3 {
                    acc[c] += cur[o + c];
                }
            }
        }
        let cnt = ((y1.min(h) - y0) * (x1.min(w) - x0)).max(1) as f32;
        *cell = acc.map(|v| v / cnt);
    }
    let mut mean = [0.0f32; 3];
    for c in &grid {
        for k in 0..3 {
            mean[k] += c[k] / grid.len() as f32;
        }
    }
    let light = ArtLight {
        mean_y: 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2],
        grid,
        mean,
    };
    // mips: each level boxes the previous one in linear light
    let mut mips = Vec::new();
    let (mut sw, mut sh) = (w, h);
    for _ in 0..levels {
        if sw <= 1 && sh <= 1 {
            break;
        }
        let (nw, nh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut next = Vec::with_capacity(nw * nh * 4);
        let mut bytes = Vec::with_capacity(nw * nh * 4);
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let o = ((2 * y + dy).min(sh - 1) * sw + (2 * x + dx).min(sw - 1)) * 4;
                    for c in 0..4 {
                        acc[c] += 0.25 * cur[o + c];
                    }
                }
                next.extend_from_slice(&acc);
                bytes.extend_from_slice(&[
                    linear_to_srgb8(acc[0]),
                    linear_to_srgb8(acc[1]),
                    linear_to_srgb8(acc[2]),
                    (acc[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                ]);
            }
        }
        mips.push(bytes);
        cur = next;
        (sw, sh) = (nw, nh);
    }
    Some(ArtExtra {
        mips,
        light: Box::new(light),
    })
}

pub fn process_art(track_id: u64, bytes: &[u8], mips: bool) -> Ready {
    let image = mg_media::decode_art(bytes);
    let palette = image.as_ref().and_then(|img| {
        let (px, w, h) = img.thumbnail(64);
        (w > 0 && h > 0).then(|| mg_media::palette::extract(&px, w, h))
    });
    let extra = image
        .as_ref()
        .and_then(|img| art_extra(img, if mips { ART_MIP_LEVELS } else { 0 }));
    Ready::Art {
        track_id,
        image,
        palette,
        extra,
    }
}

/// Number of lines in a block rendered with uniform line spacing (`px` x `lh` per line).
fn lines_in(b: &TextBitmap, px: f32, lh: f32) -> u32 {
    let box_h = b.height.saturating_sub(2 * b.pad) as f32;
    ((box_h / (px * lh).max(1.0)).round() as u32).max(1)
}

/// Renders `text` in `style`. A wrapping style that needs more lines than allowed is tried once more at
/// `step` x the size before the word ellipsis applies, and a wrapped block is balanced: the narrowest width
/// that keeps its line count (better-typography: balance short headings, never leave a lone last word).
/// At most 6 layouts per call; runs on the worker thread only.
pub fn render_fit(
    r: &mg_text::TextRenderer,
    text: &str,
    style: &LineStyle,
    step: f32,
) -> Option<Fitted> {
    let wrap = style.max_lines > 1;
    let lh = if style.line_height.is_finite() && style.line_height >= 0.8 {
        style.line_height.min(2.0)
    } else {
        1.2
    };
    let fitted = |bitmap: TextBitmap, s: &LineStyle| Fitted {
        lines: if wrap { lines_in(&bitmap, s.px, lh) } else { 1 },
        px: s.px,
        wrap,
        line_height: lh,
        rtl: mg_text::is_rtl(text),
        bitmap,
    };
    if !wrap {
        return r.render_line(text, style).map(|b| fitted(b, style));
    }
    let mut s = *style;
    // one line more than allowed shows whether the text fits (an overflowing text fills the extra line)
    let probe = |s: &LineStyle, max_width: u32| LineStyle {
        max_lines: (s.max_lines + 1).min(4),
        max_width,
        ..*s
    };
    let mut b = r.render_line(text, &probe(&s, s.max_width))?;
    if lines_in(&b, s.px, lh) > u32::from(s.max_lines) {
        if step.is_finite() && step > 0.0 && step < 1.0 {
            s.px *= step;
        }
        b = r.render_line(text, &s)?;
    }
    let n = lines_in(&b, s.px, lh);
    if n >= 2 && n <= u32::from(s.max_lines) {
        let (mut lo, mut hi) = (s.max_width / 2, s.max_width);
        for _ in 0..4 {
            let mid = (lo + hi) / 2;
            match r.render_line(text, &probe(&s, mid)) {
                Some(c) if lines_in(&c, s.px, lh) == n => {
                    hi = mid;
                    b = c;
                }
                _ => lo = mid,
            }
        }
    }
    Some(fitted(b, &s))
}

/// Renders the title-block items of a job (title, artist, meta), each in its own role style.
pub fn process_text(r: &mg_text::TextRenderer, t: &TextJob) -> Ready {
    let [title, artist, meta] = std::array::from_fn(|i| {
        t.items[i]
            .as_ref()
            .and_then(|it| render_fit(r, &it.text, &it.style, it.step))
    });
    Ready::Text {
        track_id: t.track_id,
        serial: t.serial,
        title,
        artist,
        meta,
    }
}

/// Looks up and parses the lyrics for a track: the user's `.lrc` folder, or the built-in demo lyrics.
pub fn process_lyrics(track_id: u64, source: &LyricSource, title: &str, artist: &str) -> Ready {
    let lyrics = match source {
        LyricSource::None => None,
        LyricSource::Demo => crate::media_feed::demo_lyrics(title),
        LyricSource::Dir(dir) => {
            mg_media::lrc::find(dir, title, artist).and_then(|p| mg_media::lrc::load(&p))
        }
    };
    Ready::Lyrics {
        track_id,
        lyrics: lyrics.map(Arc::new),
    }
}

/// Renders one lyric line in the style its view asked for.
pub fn process_line(
    r: Option<&mg_text::TextRenderer>,
    track_id: u64,
    serial: u64,
    idx: usize,
    text: &str,
    style: &LineStyle,
    step: f32,
) -> Ready {
    Ready::Line {
        track_id,
        serial,
        idx,
        bitmap: r
            .and_then(|r| render_fit(r, text, style, step))
            .map(|f| f.bitmap),
    }
}

/// Runs any job synchronously (used by the worker and by the headless self-test).
pub fn process_job(text: Option<&mg_text::TextRenderer>, job: Job) -> Ready {
    match job {
        Job::Art {
            track_id,
            bytes,
            mips,
        } => process_art(track_id, &bytes, mips),
        Job::Text(t) => match text {
            Some(r) => process_text(r, &t),
            None => Ready::Text {
                track_id: t.track_id,
                serial: t.serial,
                title: None,
                artist: None,
                meta: None,
            },
        },
        Job::Lyrics {
            track_id,
            source,
            title,
            artist,
        } => process_lyrics(track_id, &source, &title, &artist),
        Job::Line {
            track_id,
            serial,
            idx,
            text: line,
            style,
            step,
        } => process_line(text, track_id, serial, idx, &line, &style, step),
        Job::Clock {
            serial,
            slot,
            text: s,
            px,
        } => Ready::Clock {
            serial,
            slot,
            bitmap: text.and_then(|r| {
                r.render_line(
                    &s,
                    &mg_text::LineStyle {
                        px,
                        weight: 500,
                        max_width: (px * 10.0) as u32,
                        ..Default::default()
                    },
                )
            }),
        },
    }
}

fn run_worker(rx: Receiver<Job>, tx: Sender<Ready>) {
    let text = mg_text::TextRenderer::new(); // owns its COM guard (dropped last)
    while let Ok(first) = rx.recv() {
        // latest request of each kind wins; lyric lines are kept per line index
        let (mut art, mut txt, mut lyr) = (None, None, None);
        let mut lines: Vec<Job> = Vec::new();
        for j in std::iter::once(first).chain(std::iter::from_fn(|| rx.try_recv().ok())) {
            match j {
                Job::Art { .. } => art = Some(j),
                Job::Text(_) => txt = Some(j),
                Job::Lyrics { .. } => lyr = Some(j),
                Job::Line { .. } | Job::Clock { .. } => lines.push(j),
            }
        }
        // cheap, visible things first; the (bounded but slower) artwork decode last
        for j in txt.into_iter().chain(lyr).chain(lines).chain(art) {
            if tx.send(process_job(text.as_ref(), j)).is_err() {
                return;
            }
        }
    }
}

impl Worker {
    pub fn start() -> Self {
        let (jtx, jrx) = channel();
        let (rtx, rrx) = channel();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d2 = done.clone();
        let thread = std::thread::Builder::new()
            .name("mg-presentation".into())
            .spawn(move || {
                let _ =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_worker(jrx, rtx)));
                d2.store(true, std::sync::atomic::Ordering::Release);
            })
            .ok();
        Self {
            tx: Some(jtx),
            rx: rrx,
            thread,
            done,
        }
    }

    pub fn submit(&self, j: Job) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(j);
        }
    }

    pub fn try_recv(&self) -> Option<Ready> {
        self.rx.try_recv().ok()
    }

    /// Stops the thread; waits a bounded time (a stuck imaging call must never hang shutdown).
    pub fn stop(&mut self) {
        self.tx = None; // ends the worker's recv loop
        let Some(t) = self.thread.take() else { return };
        let deadline = Instant::now() + Duration::from_secs(3);
        while !self.done.load(std::sync::atomic::Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if self.done.load(std::sync::atomic::Ordering::Acquire) {
            let _ = t.join();
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---- tests ------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mg_media::PlaybackState;

    const SURF: (u32, u32) = (1920, 1080);
    const DT: f32 = 1.0 / 60.0;

    fn np(id: u64, title: &str, art: Option<Arc<[u8]>>) -> NowPlaying {
        NowPlaying {
            title: title.into(),
            artist: "Artist".into(),
            album: String::new(),
            app_id: String::new(),
            state: PlaybackState::Playing,
            position_s: None,
            duration_s: None,
            sampled_at: Instant::now(),
            caps: Default::default(),
            art,
            track_id: id,
        }
    }

    fn img(c: [u8; 4]) -> ArtImage {
        ArtImage {
            width: 4,
            height: 2,
            rgba: (0..8).flat_map(|_| c).collect(),
        }
    }

    /// A coverage block `w x h` with a 2 px pad (the width doubles as a marker in some tests).
    fn bmp_sized(w: u32, h: u32) -> TextBitmap {
        TextBitmap {
            width: w,
            height: h,
            pad: 2,
            coverage: vec![200; (w * h) as usize],
        }
    }

    fn bmp() -> TextBitmap {
        bmp_sized(40, 20)
    }

    fn fit(b: TextBitmap) -> Fitted {
        Fitted {
            bitmap: b,
            px: 16.0,
            lines: 1,
            wrap: false,
            line_height: 1.2,
            rtl: false,
        }
    }

    fn text(title: Option<TextBitmap>, artist: Option<TextBitmap>) -> Ready {
        Ready::Text {
            track_id: 0,
            serial: 0,
            title: title.map(fit),
            artist: artist.map(fit),
            meta: None,
        }
    }

    /// The text result for the newest request of `p` (track id and serial filled in).
    fn answer(p: &Presentation, r: Ready) -> Ready {
        match r {
            Ready::Text {
                title,
                artist,
                meta,
                ..
            } => Ready::Text {
                track_id: p.track_id.unwrap_or(0),
                serial: p.serial,
                title,
                artist,
                meta,
            },
            r => r,
        }
    }

    fn run(p: &mut Presentation, secs: f32) -> OverlayParams {
        let mut o = None;
        for _ in 0..(secs * 60.0).round() as usize {
            o = Some(p.update(DT, SURF, 0.5, 0.2));
        }
        o.expect("at least one frame")
    }

    fn art_job_count(jobs: &[Job]) -> usize {
        jobs.iter().filter(|j| matches!(j, Job::Art { .. })).count()
    }

    /// Fake worker: realistic bitmap sizes from the requested styles (worst case: wrapping roles fill two lines
    /// at full column width), real demo lyrics, a square synthetic cover. Returns how many jobs were answered.
    fn pump(p: &mut Presentation) -> usize {
        let jobs = p.take_jobs();
        let n = jobs.len();
        let fake = |s: &LineStyle| {
            let lh = if s.line_height >= 0.8 {
                s.line_height
            } else {
                1.2
            };
            let (w, h) = if s.max_lines > 1 {
                (s.max_width, (2.0 * s.px * lh).ceil() as u32)
            } else {
                (
                    s.max_width.min((s.px * 14.0) as u32),
                    (s.px * LINE_BOX).ceil() as u32,
                )
            };
            Fitted {
                bitmap: bmp_sized(w + 4, h + 4),
                px: s.px,
                lines: if s.max_lines > 1 { 2 } else { 1 },
                wrap: s.max_lines > 1,
                line_height: lh,
                rtl: false,
            }
        };
        for j in jobs {
            let r = match j {
                Job::Text(t) => Ready::Text {
                    track_id: t.track_id,
                    serial: t.serial,
                    title: t.items[0].as_ref().map(|i| fake(&i.style)),
                    artist: t.items[1].as_ref().map(|i| fake(&i.style)),
                    meta: t.items[2].as_ref().map(|i| fake(&i.style)),
                },
                Job::Art { track_id, .. } => Ready::Art {
                    track_id,
                    image: Some(ArtImage {
                        width: 8,
                        height: 8,
                        rgba: vec![128; 256],
                    }),
                    palette: Some(Palette::fallback()),
                    extra: None,
                },
                Job::Lyrics {
                    track_id,
                    source,
                    title,
                    artist,
                } => process_lyrics(track_id, &source, &title, &artist),
                Job::Line {
                    track_id,
                    serial,
                    idx,
                    style,
                    ..
                } => Ready::Line {
                    track_id,
                    serial,
                    idx,
                    bitmap: Some(fake(&style).bitmap),
                },
                Job::Clock { .. } => continue,
            };
            p.on_ready(r);
        }
        n
    }

    // ---- easing and tweens ----

    #[test]
    fn curves_hit_their_endpoints_are_monotonic_and_tweens_retarget_without_a_jump() {
        for c in [EASE_OUT, EASE_IN_OUT] {
            assert_eq!(c.at(0.0), 0.0);
            assert_eq!(c.at(1.0), 1.0);
            assert_eq!(c.at(f32::NAN), 0.0);
            let mut prev = 0.0;
            for i in 1..=100 {
                let v = c.at(i as f32 / 100.0);
                assert!(v >= prev - 1e-5, "{c:?} not monotonic at {i}");
                prev = v;
            }
        }
        // the strong ease-out is front-loaded: past 70 % of the way at 30 % of the time
        assert!(EASE_OUT.at(0.3) > 0.7);
        let mut t = Tween::at(0.0);
        t.retarget(1.0, 0.36, 0.0, EASE_OUT);
        for _ in 0..6 {
            t.step(DT);
        }
        let before = t.value();
        assert!(before > 0.0 && before < 1.0);
        t.retarget(0.0, 0.22, 0.0, EASE_OUT); // interrupted: leaves from where it is
        assert!((t.value() - before).abs() < 1e-6, "retarget jumped");
        let mut last = before;
        for _ in 0..30 {
            t.step(DT);
            let v = t.value();
            assert!(v <= last + 1e-6 && (v - last).abs() < 0.35, "{last} -> {v}");
            last = v;
        }
        assert_eq!(t.value(), 0.0);
        // a delay holds the start value
        let mut d = Tween::at(0.0);
        d.retarget(1.0, 0.36, 0.30, EASE_OUT);
        d.step(0.29);
        assert_eq!(d.value(), 0.0);
    }

    // ---- existing contracts, adapted ----

    #[test]
    fn lyrics_toggle_hides_drawing_without_losing_state() {
        let mut p = Presentation::new(Layout::Corner);
        p.set_lyric_source(LyricSource::Demo);
        p.on_now_playing(Some(&np(1, "Midnight Harbour", None)), SURF);
        p.set_position(Some(11.0));
        for _ in 0..90 {
            pump(&mut p);
            p.update(DT, SURF, 0.0, 0.0);
        }
        let o = p.update(DT, SURF, 0.0, 0.0);
        assert!(o.lyrics.iter().any(Option::is_some), "lyrics drawn");
        assert!(!p.toggle_lyrics());
        let o = p.update(DT, SURF, 0.0, 0.0);
        assert!(o.lyrics.iter().all(Option::is_none));
        assert!(
            p.lyric_draws.iter().any(Option::is_some),
            "tracking state is kept"
        );
        assert!(p.toggle_lyrics());
        assert!(
            p.update(DT, SURF, 0.0, 0.0)
                .lyrics
                .iter()
                .any(Option::is_some)
        );
    }

    #[test]
    fn a_track_change_requests_text_and_art_once() {
        let mut p = Presentation::new(Layout::Corner);
        let art: Arc<[u8]> = Arc::from(vec![1u8, 2, 3]);
        let n = np(1, "T", Some(art));
        p.on_now_playing(Some(&n), SURF);
        let jobs = p.take_jobs();
        assert_eq!(jobs.len(), 2);
        assert_eq!(art_job_count(&jobs), 1);
        p.on_now_playing(Some(&n), SURF); // same track polled again: nothing new
        assert!(p.take_jobs().is_empty());
    }

    #[test]
    fn late_artwork_for_the_same_track_is_requested_when_it_appears() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "T", None)), SURF);
        assert_eq!(art_job_count(&p.take_jobs()), 0);
        p.on_now_playing(Some(&np(1, "T", Some(Arc::from(vec![9u8])))), SURF);
        assert_eq!(art_job_count(&p.take_jobs()), 1);
    }

    #[test]
    fn results_for_a_previous_track_never_reach_the_screen() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "Old", Some(Arc::from(vec![1u8])))), SURF);
        p.take_jobs();
        p.on_now_playing(Some(&np(2, "New", None)), SURF);
        p.take_jobs();
        // the slow decode for track 1 finishes after track 2 started
        p.on_ready(Ready::Art {
            track_id: 1,
            image: Some(img([255, 0, 0, 255])),
            palette: Some(Palette::fallback()),
            extra: None,
        });
        p.on_ready(Ready::Text {
            track_id: 1,
            serial: 1,
            title: Some(fit(bmp())),
            artist: None,
            meta: None,
        });
        run(&mut p, 2.0);
        assert!(p.take_ops().is_empty(), "stale content was uploaded");
        assert_eq!(p.album().1, 0.0, "stale palette applied");
    }

    #[test]
    fn a_resized_text_request_supersedes_the_older_result() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "T", None)), SURF);
        let Job::Text(first) = p.take_jobs().remove(0) else {
            panic!()
        };
        p.set_layout(Layout::Center, SURF);
        p.on_ready(Ready::Text {
            track_id: 1,
            serial: first.serial,
            title: Some(fit(bmp())),
            artist: None,
            meta: None,
        });
        run(&mut p, 1.0);
        assert!(
            p.take_ops().is_empty(),
            "the superseded text must be dropped"
        );
    }

    #[test]
    fn a_new_cover_crossfades_over_the_held_previous_one_which_is_then_released() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "A", Some(Arc::from(vec![1u8])))), SURF);
        p.take_jobs();
        p.on_ready(Ready::Art {
            track_id: 1,
            image: Some(img([200, 20, 20, 255])),
            palette: Some(Palette::fallback()),
            extra: None,
        });
        let o = run(&mut p, 1.5);
        assert!(matches!(p.take_ops().as_slice(), [GpuOp::SetArt(..)]));
        assert!(o.art.expect("shown").alpha > 0.99);
        // track 2 arrives with its artwork decoded 50 ms later
        p.on_now_playing(Some(&np(2, "B", Some(Arc::from(vec![2u8])))), SURF);
        p.take_jobs();
        run(&mut p, 0.05);
        p.on_ready(Ready::Art {
            track_id: 2,
            image: Some(img([20, 20, 200, 255])),
            palette: Some(Palette::fallback()),
            extra: None,
        });
        let o = p.update(DT, SURF, 0.5, 0.2);
        assert!(
            matches!(
                p.take_ops().as_slice(),
                [GpuOp::ShiftArt, GpuOp::SetArt(..)]
            ),
            "old cover becomes the crossfade source, the new one is uploaded at once"
        );
        let held = o.art_prev.expect("previous cover held").alpha;
        assert!(held > 0.6, "held, not blanked: {held}");
        assert!(o.art.expect("new cover").alpha < 0.05, "fades in from 0");
        // halfway the old one is still held at the same level: no dip to the background
        let o = run(&mut p, 0.35);
        assert!((o.art_prev.expect("still held").alpha - held).abs() < 1e-4);
        let a = o.art.expect("new").alpha;
        assert!(a > 0.2 && a < 0.9, "crossfading {a}");
        let o = run(&mut p, 0.5);
        assert!(o.art.expect("new").alpha > 0.99);
        assert!(o.art_prev.is_none(), "previous cover released");
        assert!(
            p.take_ops()
                .iter()
                .any(|op| matches!(op, GpuOp::ClearPrevArt))
        );
    }

    #[test]
    fn missing_artwork_shows_text_only_and_stopping_clears_everything() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "Only text", None)), SURF);
        p.take_jobs();
        p.on_ready(answer(&p, text(Some(bmp()), Some(bmp()))));
        let o = run(&mut p, 2.0);
        assert!(o.art.is_none() && o.title.is_some() && o.artist.is_some());
        p.on_now_playing(None, SURF);
        let o = run(&mut p, 2.0);
        assert!(o.is_empty(), "everything faded out");
    }

    #[test]
    fn corrupt_artwork_falls_back_to_no_artwork_and_scene_colours() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "T", Some(Arc::from(vec![0u8; 4])))), SURF);
        p.take_jobs();
        p.on_ready(Ready::Art {
            track_id: 1,
            image: None,
            palette: None,
            extra: None,
        });
        let o = run(&mut p, 2.0);
        assert!(o.art.is_none());
        assert_eq!(p.album().1, 0.0);
    }

    #[test]
    fn palette_moves_smoothly_in_bounded_steps_and_returns_to_scene_colours() {
        let mut s = PaletteState::new();
        let (a, b) = (Palette::fallback(), {
            let mut b = Palette::fallback();
            b.main = [0.05, 0.1, 0.9];
            b.accent = [0.9, 0.05, 0.1];
            b
        });
        s.set_target(Some(a));
        for _ in 0..200 {
            s.update(DT);
        }
        s.set_target(Some(b));
        let mut prev = scene_roles(&s.cur);
        let mut max_step = 0.0f32;
        for _ in 0..200 {
            s.update(DT);
            let now = scene_roles(&s.cur);
            for i in 0..4 {
                for c in 0..3 {
                    max_step = max_step.max((now[i][c] - prev[i][c]).abs());
                }
            }
            prev = now;
        }
        assert!(max_step < 0.03, "palette jumped by {max_step} in one frame");
        assert!(
            (s.cur.main[2] - b.main[2]).abs() < 0.01,
            "reached the new colours"
        );
        s.set_target(None);
        for _ in 0..200 {
            s.update(DT);
        }
        assert_eq!(s.w, 0.0);
    }

    #[test]
    fn scene_roles_are_finite_bounded_and_comparably_bright_for_any_palette() {
        let extremes = [
            Palette::fallback(),
            Palette {
                bg: [0.0; 3],
                main: [0.0; 3],
                secondary: [0.0; 3],
                accent: [0.0; 3],
                highlight: [0.0; 3],
                text: [0.0; 3],
            },
            Palette {
                bg: [1.0; 3],
                main: [1.0; 3],
                secondary: [1.0; 3],
                accent: [1.0; 3],
                highlight: [1.0; 3],
                text: [1.0; 3],
            },
            Palette {
                bg: [f32::NAN, 0.0, 0.0],
                main: [0.0, 1e-9, 0.0],
                secondary: [1.0, 0.0, 0.0],
                accent: [0.0, 1.0, 0.0],
                highlight: [0.0, 0.0, 1.0],
                text: [0.5; 3],
            },
        ];
        for p in extremes {
            for r in scene_roles(&p) {
                assert!(
                    r.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                    "{p:?} -> {r:?}"
                );
                assert!(peak(r) > 0.05, "role vanished: {r:?}");
            }
            // text roles: finite, tinted (never plain grey), lightness order title > artist > meta, lyric > context
            let c = role_colors(&p);
            let l = |v: [f32; 3]| to_oklab(v)[0];
            for v in [c.title, c.artist, c.meta, c.lyric, c.context, c.shade] {
                assert!(
                    v.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
                    "{c:?}"
                );
            }
            assert!((l(c.title) - L_TITLE).abs() < 0.03, "{}", l(c.title));
            assert!(l(c.title) > l(c.artist) && l(c.artist) > l(c.meta));
            assert!(l(c.lyric) > l(c.context) && l(c.shade) < 0.1);
            let lab = to_oklab(c.title);
            assert!(lab[1].hypot(lab[2]) > 0.01, "title is tinted: {lab:?}");
        }
    }

    #[test]
    fn layouts_stay_on_screen_for_any_aspect_and_preserve_artwork_aspect() {
        for (layout, album) in [
            (Layout::Corner, false),
            (Layout::Center, false),
            (Layout::Center, true),
        ] {
            for &(w, h) in &[
                (1920u32, 1080u32),
                (3440, 1440),
                (1024, 768),
                (720, 1280),
                (320, 180),
            ] {
                let mut p = Presentation::new(layout);
                p.set_album_scene(album, (w, h));
                p.on_now_playing(Some(&np(1, "T", Some(Arc::from(vec![1u8])))), (w, h));
                p.jobs.retain(|j| !matches!(j, Job::Art { .. }));
                p.on_ready(Ready::Art {
                    track_id: 1,
                    image: Some(img([10, 10, 10, 255])), // a 2:1 image
                    palette: None,
                    extra: None,
                });
                for _ in 0..90 {
                    pump(&mut p);
                    p.update(DT, (w, h), 0.5, 0.2);
                }
                let o = p.update(DT, (w, h), 0.5, 0.2);
                let a = o.art.expect("art");
                assert!((a.hw / a.hh - 2.0).abs() < 1e-3, "aspect {}", a.hw / a.hh);
                let tag = format!("{layout:?}/album={album} {w}x{h}");
                assert!(a.cx - a.hw >= 0.0 && a.cx + a.hw <= w as f32, "{tag} art x");
                assert!(a.cy - a.hh >= 0.0 && a.cy + a.hh <= h as f32, "{tag} art y");
                let t = o.title.expect("title");
                assert!(
                    t.x + 2.0 >= 0.0 && t.x + t.w <= w as f32 + 2.0,
                    "{tag} title x {} {}",
                    t.x,
                    t.w
                );
                let ar = o.artist.expect("artist");
                assert!(ar.y + ar.h <= h as f32 + 2.0, "{tag} artist y");
            }
        }
    }

    #[test]
    fn layout_cycles_and_off_draws_nothing_but_keeps_the_palette() {
        assert_eq!(Layout::Corner.next().next().next(), Layout::Corner);
        let mut p = Presentation::new(Layout::Off);
        p.on_now_playing(Some(&np(1, "T", None)), SURF);
        p.take_jobs();
        p.on_ready(Ready::Art {
            track_id: 1,
            image: Some(img([10, 200, 10, 255])),
            palette: Some(Palette::fallback()),
            extra: None,
        });
        let o = run(&mut p, 3.5);
        assert!(o.art.is_none() && o.title.is_none());
        assert!(p.album().1 > 0.99);
    }

    #[test]
    fn a_same_track_text_rerender_is_applied_without_waiting_for_a_fade_out() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "T", None)), SURF);
        p.take_jobs();
        p.on_ready(answer(&p, text(Some(bmp()), None)));
        run(&mut p, 2.0);
        assert!(matches!(p.take_ops().as_slice(), [GpuOp::SetText(0, _)]));
        // layout change (keyboard: no animation): new pixel size requested, swapped in at once
        p.set_layout(Layout::Center, SURF);
        let Job::Text(j) = p.take_jobs().remove(0) else {
            panic!()
        };
        p.on_ready(Ready::Text {
            track_id: 1,
            serial: j.serial,
            title: Some(fit(bmp())),
            artist: None,
            meta: None,
        });
        let o = p.update(DT, SURF, 0.5, 0.2);
        assert!(
            matches!(p.take_ops().as_slice(), [GpuOp::SetText(0, _)]),
            "re-render not applied"
        );
        assert!(
            o.title.expect("still visible").alpha > 0.9,
            "no fade-out for a re-render"
        );
    }

    #[test]
    fn a_track_without_artwork_clears_the_previous_cover() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "A", Some(Arc::from(vec![1u8])))), SURF);
        p.take_jobs();
        p.on_ready(Ready::Art {
            track_id: 1,
            image: Some(img([200, 20, 20, 255])),
            palette: Some(Palette::fallback()),
            extra: None,
        });
        run(&mut p, 1.5);
        p.take_ops();
        p.on_now_playing(Some(&np(2, "B", None)), SURF);
        p.take_jobs();
        // the stale cover is gone before the new title starts entering (300 ms)
        let o = run(&mut p, 0.3);
        assert!(
            p.take_ops().iter().any(|op| matches!(op, GpuOp::ClearArt)),
            "cover texture kept"
        );
        assert!(o.art.is_none(), "cover (and its layout gap) must be gone");
        assert!(p.art.iter().all(Option::is_none));
    }

    #[test]
    fn a_track_change_while_minimised_still_gets_text_after_a_previous_track_was_shown() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "A", None)), SURF);
        p.take_jobs();
        p.on_now_playing(Some(&np(2, "B", None)), (0, 0));
        assert!(p.take_jobs().is_empty());
        p.update(DT, SURF, 0.0, 0.0);
        assert_eq!(p.take_jobs().len(), 1, "second track must get its text");
    }

    #[test]
    fn text_is_requested_once_the_surface_has_a_size() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "T", None)), (0, 0)); // minimised
        assert!(p.take_jobs().is_empty());
        p.update(DT, SURF, 0.0, 0.0);
        assert_eq!(
            p.take_jobs().len(),
            1,
            "text requested after the window got a size"
        );
    }

    // ---- track-change choreography ----

    /// Track `id` fully shown (text answered with marker width `mark`), returns the presentation.
    fn shown_track(p: &mut Presentation, id: u64, mark: u32) {
        p.on_now_playing(Some(&np(id, "T", None)), SURF);
        p.take_jobs();
        p.on_ready(Ready::Text {
            track_id: id,
            serial: p.serial,
            title: Some(fit(bmp_sized(mark, 20))),
            artist: Some(fit(bmp())),
            meta: Some(fit(bmp())),
        });
    }

    #[test]
    fn old_text_leaves_in_220_ms_and_new_text_enters_on_the_stagger_schedule() {
        let mut p = Presentation::new(Layout::Corner);
        shown_track(&mut p, 1, 101);
        let o = run(&mut p, 1.5);
        let (y0, h) = (o.title.expect("shown").y, SURF.1 as f32);
        assert!(o.title.expect("t").alpha > 0.999);
        // track 2: its text is ready at once
        p.on_now_playing(Some(&np(2, "U", None)), SURF);
        p.take_jobs();
        p.on_ready(Ready::Text {
            track_id: 2,
            serial: p.serial,
            title: Some(fit(bmp_sized(102, 20))),
            artist: Some(fit(bmp())),
            meta: Some(fit(bmp())),
        });
        let mut t = 0.0f32;
        let mut prev_alpha = 1.0f32;
        let (mut title_in, mut artist_in, mut meta_in) = (None, None, None);
        while t < 1.2 {
            let o = p.update(DT, SURF, 0.0, 0.0);
            t += DT;
            let new = p.els[0].dims.is_some_and(|d| d.w == 102);
            match (new, o.title) {
                (false, Some(d)) => {
                    // the old title only fades and rises
                    assert!(d.alpha <= prev_alpha + 1e-6 && d.y <= y0 + 1e-3);
                    prev_alpha = d.alpha;
                    assert!(t <= TEXT_OUT_S + 2.0 * DT, "old title still visible at {t}");
                }
                (true, Some(d)) if title_in.is_none() => {
                    title_in = Some(t);
                    assert!(
                        d.y > y0 && d.y - y0 <= TEXT_IN_DY * h + 0.5,
                        "enters from below"
                    );
                }
                _ => {}
            }
            if new && artist_in.is_none() && o.artist.is_some() {
                artist_in = Some(t);
            }
            if new && meta_in.is_none() && o.meta.is_some() {
                meta_in = Some(t);
            }
        }
        let (ti, ai, mi) = (title_in.unwrap(), artist_in.unwrap(), meta_in.unwrap());
        assert!((ti - TEXT_ENTER_AT[0]).abs() < 3.0 * DT, "title at {ti}");
        assert!((ai - TEXT_ENTER_AT[1]).abs() < 3.0 * DT, "artist at {ai}");
        assert!((mi - TEXT_ENTER_AT[2]).abs() < 3.0 * DT, "meta at {mi}");
        let o = run(&mut p, 0.5);
        assert!(o.title.expect("in").alpha > 0.999 && (o.title.unwrap().y - y0).abs() < 0.5);
    }

    #[test]
    fn rapid_skips_cancel_obsolete_transitions_and_never_show_stale_text() {
        let mut p = Presentation::new(Layout::Corner);
        shown_track(&mut p, 1, 101);
        run(&mut p, 1.5);
        p.take_ops();
        // skip to 2, 80 ms later to 3, then 2's late text arrives
        p.on_now_playing(Some(&np(2, "U", None)), SURF);
        let serial2 = p.serial;
        run(&mut p, 0.08);
        p.on_now_playing(Some(&np(3, "V", None)), SURF);
        p.on_ready(Ready::Text {
            track_id: 2,
            serial: serial2,
            title: Some(fit(bmp_sized(102, 20))),
            artist: None,
            meta: None,
        });
        p.on_ready(answer(&p, text(Some(bmp_sized(103, 20)), Some(bmp()))));
        let mut last_old_alpha = 1.0f32;
        for _ in 0..90 {
            let o = p.update(DT, SURF, 0.0, 0.0);
            let shown = p.els[0].dims.map(|d| d.w);
            if let (Some(101), Some(d)) = (shown, o.title) {
                assert!(d.alpha <= last_old_alpha + 1e-6, "old text came back");
                last_old_alpha = d.alpha;
            }
        }
        let uploads: Vec<u32> = p
            .take_ops()
            .iter()
            .filter_map(|op| match op {
                GpuOp::SetText(0, b) => Some(b.width),
                _ => None,
            })
            .collect();
        assert_eq!(
            uploads,
            [103],
            "only the newest track's title is ever uploaded"
        );
        assert!(p.update(DT, SURF, 0.0, 0.0).title.expect("t3").alpha > 0.99);
    }

    #[test]
    fn rapid_cover_skips_keep_at_most_two_textures_and_end_on_the_newest_cover() {
        let mut p = Presentation::new(Layout::Corner);
        let cover = |p: &mut Presentation, id: u64, c: [u8; 4]| {
            p.on_now_playing(Some(&np(id, "T", Some(Arc::from(vec![c[0]])))), SURF);
            p.take_jobs();
            p.on_ready(Ready::Art {
                track_id: id,
                image: Some(img(c)),
                palette: None,
                extra: None,
            });
        };
        cover(&mut p, 1, [1, 0, 0, 255]);
        run(&mut p, 1.0);
        let mut textures = 0i32; // mirrors the overlay: current + previous
        let mut ops = Vec::new();
        for (id, c) in [
            (2u64, [2u8, 0, 0, 255]),
            (3, [3, 0, 0, 255]),
            (4, [4, 0, 0, 255]),
        ] {
            cover(&mut p, id, c);
            run(&mut p, 0.1);
            ops.extend(p.take_ops());
        }
        run(&mut p, 1.5);
        ops.extend(p.take_ops());
        let (mut cur, mut prev) = (Some(1u8), None::<u8>);
        for op in &ops {
            match op {
                GpuOp::ShiftArt => {
                    prev = cur.take();
                }
                GpuOp::SetArt(i, _) => cur = Some(i.rgba[0]),
                GpuOp::ClearArt => cur = None,
                GpuOp::ClearPrevArt => prev = None,
                _ => {}
            }
            textures = i32::from(cur.is_some()) + i32::from(prev.is_some());
            assert!(textures <= 2);
        }
        assert_eq!((cur, prev, textures), (Some(4), None, 1));
        let o = p.update(DT, SURF, 0.0, 0.0);
        assert!(o.art.expect("newest").alpha > 0.99 && o.art_prev.is_none());
    }

    #[test]
    fn late_artwork_fades_in_without_a_blank_frame_and_the_text_stays() {
        let mut p = Presentation::new(Layout::Center);
        p.set_album_scene(true, SURF);
        p.on_now_playing(Some(&np(1, "T", None)), SURF);
        for _ in 0..60 {
            pump(&mut p);
            p.update(DT, SURF, 0.0, 0.0);
        }
        let o = p.update(DT, SURF, 0.0, 0.0);
        assert!(o.art.is_none() && o.title.expect("text-led").alpha > 0.99);
        // the session delivers the cover two seconds into the track
        p.on_now_playing(Some(&np(1, "T", Some(Arc::from(vec![7u8])))), SURF);
        pump(&mut p);
        let mut last = 0.0f32;
        for _ in 0..60 {
            let o = p.update(DT, SURF, 0.0, 0.0);
            assert!(o.title.expect("never blank").alpha > 0.99);
            let a = o.art.expect("cover drawn from its first frame").alpha;
            assert!(a >= last && a - last < 0.1, "fades in: {last} -> {a}");
            last = a;
        }
        assert!(last > 0.99);
    }

    #[test]
    fn reduced_motion_moves_nothing_and_uses_the_short_fades() {
        let mut p = Presentation::new(Layout::Corner);
        p.set_reduced_motion(true);
        shown_track(&mut p, 1, 101);
        let o = run(&mut p, 1.0);
        let y0 = o.title.expect("t").y;
        p.on_now_playing(Some(&np(2, "U", None)), SURF);
        p.take_jobs();
        p.on_ready(Ready::Text {
            track_id: 2,
            serial: p.serial,
            title: Some(fit(bmp_sized(102, 20))),
            artist: Some(fit(bmp())),
            meta: Some(fit(bmp())),
        });
        for _ in 0..60 {
            if let Some(d) = p.update(DT, SURF, 0.0, 0.0).title {
                assert!((d.y - y0).abs() < 1e-3, "moved to {} from {y0}", d.y);
            }
        }
        // out 200 ms, in 200 ms after the 300 ms entrance delay
        assert!(run(&mut p, 0.1).title.expect("in").alpha > 0.99);
    }

    // ---- Album Focus composition ----

    #[derive(Clone, Copy, Debug)]
    struct R {
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    }

    impl R {
        fn of_text(t: &TextDraw) -> Self {
            Self {
                x0: t.x + 2.0,
                y0: t.y + 2.0,
                x1: t.x + t.w - 2.0,
                y1: t.y + t.h - 2.0,
            }
        }
        fn union(self, o: R) -> R {
            R {
                x0: self.x0.min(o.x0),
                y0: self.y0.min(o.y0),
                x1: self.x1.max(o.x1),
                y1: self.y1.max(o.y1),
            }
        }
        /// Separation (px) between two rects: positive when disjoint (the larger of the axis gaps).
        fn gap(self, o: R) -> f32 {
            (o.x0 - self.x1)
                .max(self.x0 - o.x1)
                .max(o.y0 - self.y1)
                .max(self.y0 - o.y1)
        }
        fn area(self) -> f32 {
            (self.x1 - self.x0) * (self.y1 - self.y0)
        }
    }

    /// The overlap matrix: aspect x resolution x bar x lyrics x cover. Cover, title block, lyric column and
    /// the control-bar zone are pairwise disjoint by >= 0.02 H and inside a 0.04 H margin; the cover is the
    /// largest element and (with lyrics) the current lyric outranks the title.
    #[test]
    fn album_focus_composition_never_overlaps_and_keeps_its_hierarchy() {
        const { assert!(LYRIC_CURRENT.size > TITLE_LYRICS.size) };
        const { assert!(TITLE_SOLO.size > TITLE_LYRICS.size && ARTIST_SOLO.size > ARTIST_LYRICS.size) };
        let mut cases = 0;
        for (aw, ah) in [(16u32, 9u32), (21, 9), (16, 10), (4, 3), (1, 1), (9, 16)] {
            for base in [720u32, 1080, 1440, 2160] {
                let (w, h) = if aw >= ah {
                    (base * aw / ah, base)
                } else {
                    (base, base * ah / aw)
                };
                let s = (w, h);
                let (wf, hf) = (w as f32, h as f32);
                let zone_top = hf - 128.0 * crate::controls::ui_scale(s, 1.0);
                for bar in [false, true] {
                    for lyrics in [false, true] {
                        for art in [false, true] {
                            cases += 1;
                            let tag = format!("{w}x{h} bar={bar} lyrics={lyrics} art={art}");
                            let mut p = Presentation::new(Layout::Center);
                            p.set_album_scene(true, s);
                            p.set_bar(zone_top, if bar { 1.0 } else { 0.0 });
                            if lyrics {
                                p.set_lyric_source(LyricSource::Demo);
                            }
                            let mut n =
                                np(1, "Midnight Harbour", art.then(|| Arc::from(vec![1u8])));
                            n.album = "Harbour Lights".into();
                            p.on_now_playing(Some(&n), s);
                            p.set_position(Some(11.0));
                            let mut o = OverlayParams::default();
                            for _ in 0..120 {
                                pump(&mut p);
                                o = p.update(DT, s, 0.0, 0.0);
                            }
                            let block = [o.title, o.artist, o.meta]
                                .iter()
                                .flatten()
                                .map(R::of_text)
                                .reduce(R::union)
                                .unwrap_or_else(|| panic!("{tag}: no title block"));
                            assert!(o.meta.is_some(), "{tag}: album/meta line drawn");
                            let mut rects = vec![("block", block)];
                            if art {
                                let a = o.art.unwrap_or_else(|| panic!("{tag}: no cover"));
                                let r = R {
                                    x0: a.cx - a.hw,
                                    y0: a.cy - a.hh,
                                    x1: a.cx + a.hw,
                                    y1: a.cy + a.hh,
                                };
                                assert!(r.area() > block.area(), "{tag}: cover must dominate");
                                assert!(a.hh >= 0.12 * hf, "{tag}: cover collapsed to {}", a.hh);
                                rects.push(("art", r));
                            } else {
                                assert!(o.art.is_none(), "{tag}");
                            }
                            if lyrics {
                                let c = p
                                    .lyrics
                                    .column()
                                    .unwrap_or_else(|| panic!("{tag}: no column"));
                                let [t, b, _] =
                                    o.lyric_clip.unwrap_or_else(|| panic!("{tag}: no mask"));
                                assert!(b - t >= 0.05 * hf, "{tag}: lyric column {t}..{b}");
                                rects.push((
                                    "lyrics",
                                    R {
                                        x0: c.x,
                                        y0: t,
                                        x1: c.x + c.w,
                                        y1: b,
                                    },
                                ));
                                let drawn: Vec<_> = o.lyrics.iter().flatten().collect();
                                assert!(!drawn.is_empty(), "{tag}: lyrics drawn");
                                // every visible line starts inside the column (a context line may peek past the
                                // feathered bottom edge); the current line lies entirely inside
                                for (k, d) in o.lyrics.iter().enumerate() {
                                    let Some(d) = d else { continue };
                                    let (top, bot) = (d.y + 2.0, d.y + d.h - 2.0);
                                    assert!(
                                        d.alpha < 0.05 || (top >= t - 1.0 && top < b),
                                        "{tag}: lyric line outside its column {d:?} {t}..{b}"
                                    );
                                    assert!(
                                        o.lyric_emph[k] < 0.99 || bot <= b + 1.0,
                                        "{tag}: current lyric cut off {d:?} {t}..{b}"
                                    );
                                    assert!(
                                        d.x + 2.0 >= c.x - 1.0
                                            && d.x + d.w - 2.0 <= c.x + c.w + 1.0,
                                        "{tag}: lyric line wider than the column"
                                    );
                                }
                            } else {
                                assert!(o.lyrics.iter().all(Option::is_none), "{tag}");
                            }
                            let m = MARGIN * hf - 0.5;
                            for (name, r) in &rects {
                                assert!(
                                    r.x0 >= m && r.y0 >= m && r.x1 <= wf - m && r.y1 <= hf - m,
                                    "{tag}: {name} outside the margin {r:?}"
                                );
                                if bar {
                                    assert!(
                                        r.y1 <= zone_top - ZONE_GAP * hf + 0.5,
                                        "{tag}: {name} in the bar zone {r:?} (zone {zone_top})"
                                    );
                                }
                            }
                            for i in 0..rects.len() {
                                for j in i + 1..rects.len() {
                                    let g = rects[i].1.gap(rects[j].1);
                                    assert!(
                                        g >= ZONE_GAP * hf - 0.5,
                                        "{tag}: {} and {} {g:.1}px apart {:?} {:?}",
                                        rects[i].0,
                                        rects[j].0,
                                        rects[i].1,
                                        rects[j].1
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 6 * 4 * 2 * 2 * 2);
    }

    #[test]
    fn landscape_album_focus_follows_the_spec_grid() {
        // 16:9 with lyrics: cover 0.56 H centred at (0.31 Sw, 0.46 H); column 0.06 H right of it; title block's
        // last baseline on the cover's bottom edge; current lyric at 0.42 H
        let g = album_geom(1920.0, 1080.0, 1080.0 - 128.0, true);
        assert!(!g.portrait);
        assert!((g.art[2] - 0.56 * 1080.0).abs() < 0.5);
        assert!((g.art[0] - 0.31 * 1920.0).abs() < 0.5 && (g.art[1] - 0.46 * 1080.0).abs() < 0.5);
        assert!((g.col_x - (g.art[0] + 0.5 * g.art[2] + 0.06 * 1080.0)).abs() < 0.5);
        let top = album_block_top(&g, 1080.0, true, true, 150.0, 10.0);
        assert!((top + 150.0 - 10.0 - (g.art[1] + 0.5 * g.art[2])).abs() < 0.5);
        let c = album_lyric_column(&g, 1080.0, 952.0, (top, top + 150.0));
        assert!((c.anchor - 0.42 * 1080.0).abs() < 0.5 && c.top >= 0.18 * 1080.0 - 0.5);
        // without lyrics the block is centred on the cover
        let top = album_block_top(&g, 1080.0, true, false, 200.0, 10.0);
        assert!((top + 100.0 - g.art[1]).abs() < 0.5);
        // 4:3 uses the narrower rule, 1:1 and 9:16 stack in portrait
        let g43 = album_geom(1440.0, 1080.0, 952.0, false);
        assert!(!g43.portrait && (g43.art[0] - 0.29 * 1440.0).abs() < 0.5);
        assert!(album_geom(1080.0, 1080.0, 952.0, false).portrait);
        assert!(album_geom(1080.0, 1920.0, 1700.0, true).portrait);
    }

    #[test]
    fn the_composition_is_decided_per_track_and_waits_briefly_for_the_lyric_lookup() {
        let mut p = Presentation::new(Layout::Center);
        p.set_album_scene(true, SURF);
        p.set_lyric_source(LyricSource::Demo);
        p.on_now_playing(Some(&np(1, "Midnight Harbour", None)), SURF);
        let jobs = p.take_jobs();
        assert!(
            jobs.iter().all(|j| !matches!(j, Job::Text(_))),
            "no text before the lyric lookup answered"
        );
        assert!(matches!(jobs.as_slice(), [Job::Lyrics { .. }]));
        for j in jobs {
            if let Job::Lyrics {
                track_id,
                source,
                title,
                artist,
            } = j
            {
                p.on_ready(process_lyrics(track_id, &source, &title, &artist));
            }
        }
        p.update(DT, SURF, 0.0, 0.0);
        let Some(Job::Text(t)) = p
            .take_jobs()
            .into_iter()
            .find(|j| matches!(j, Job::Text(_)))
        else {
            panic!("text requested once lyrics are known")
        };
        let px = |i: usize| t.items[i].as_ref().unwrap().style.px;
        assert!(
            (px(0) - TITLE_LYRICS.size * 1080.0).abs() < 0.5,
            "lyrics composition"
        );
        // a track without lyrics: after the lookup answers "none" the solo sizes are used
        p.on_now_playing(Some(&np(2, "No Lyrics Here", None)), SURF);
        for j in p.take_jobs() {
            if let Job::Lyrics {
                track_id,
                source,
                title,
                artist,
            } = j
            {
                p.on_ready(process_lyrics(track_id, &source, &title, &artist));
            }
        }
        p.update(DT, SURF, 0.0, 0.0);
        let Some(Job::Text(t)) = p
            .take_jobs()
            .into_iter()
            .find(|j| matches!(j, Job::Text(_)))
        else {
            panic!()
        };
        assert!((t.items[0].as_ref().unwrap().style.px - TITLE_SOLO.size * 1080.0).abs() < 0.5);
        // lyrics arriving after the decision do not re-lay the track out
        let serial = p.serial;
        p.on_ready(Ready::Lyrics {
            track_id: 2,
            lyrics: Some(Arc::new(mg_media::lrc::parse(b"[00:01.00]late\n").unwrap())),
        });
        run(&mut p, 0.5);
        assert_eq!(p.serial, serial, "no mid-track re-layout");
        assert!(!p.lyric_variant);
    }

    #[test]
    fn roles_choose_cut_and_script_adjustments() {
        let s = TITLE_SOLO.style("Hello", 1080.0, 600.0);
        assert_eq!(s.cut, Cut::Display);
        assert_eq!((s.max_lines, s.weight), (2, 600));
        assert!((s.tracking_em + 0.015).abs() < 1e-6);
        assert_eq!(META.style("x", 400.0, 600.0).px, MIN_PX, "size floor");
        assert_eq!(META.style("x", 1080.0, 600.0).cut, Cut::Text);
        // a Display role below 24 px falls back to the Text cut
        assert_eq!(CORNER_TITLE.style("x", 600.0, 400.0).cut, Cut::Text);
        let b = TITLE_LYRICS.style("নীল আকাশের গান", 1080.0, 600.0);
        assert!((b.px - 0.036 * 1080.0 * 1.08).abs() < 0.01 && (b.line_height - 1.35).abs() < 1e-6);
        let c = LYRIC_CURRENT.style("歌曲", 1080.0, 600.0);
        assert!((c.line_height - 1.3).abs() < 1e-6);
        assert_eq!(script_tune("plain"), (1.0, 0.0));
    }

    /// Text-fit corpus through the real DirectWrite renderer: at most two title lines, ink inside the column,
    /// a size step and ellipsis only when the text does not fit.
    #[cfg(windows)]
    #[test]
    fn text_fit_corpus_wraps_only_when_needed_and_stays_in_the_column() {
        let Some(r) = mg_text::TextRenderer::new() else {
            eprintln!("SKIPPED (no DirectWrite)");
            return;
        };
        let h = 1080.0;
        let col = album_geom(1920.0, h, 952.0, true).col_w;
        let corpus = [
            "Short",
            "An Extraordinarily Long Song Title That Keeps Going On and On", // 60+ characters
            "حبيبي يا نور العين يا ساكن خيالي",
            "নীল আকাশের গান আর মেঘের ভেলা",
            "夜に駆ける 青い月の光の中で踊る",
            "เพลงรักในสายฝนที่ไม่มีวันจบสิ้น",
            "Fire 🔥 Emoji 🎶 Party ✨ Tonight",
            "Supercalifragilisticexpialidocious-Antidisestablishmentarianism",
        ];
        for role in [TITLE_LYRICS, TITLE_SOLO, LYRIC_CURRENT] {
            for t in corpus {
                let st = role.style(t, h, col);
                let f = render_fit(&r, t, &st, role.step).expect(t);
                assert!(f.lines <= 2, "{t}: {} lines", f.lines);
                let ink = f.bitmap.width - 2 * f.bitmap.pad;
                assert!(
                    ink as f32 <= col + 1.0,
                    "{t}: {ink} px in a {col} px column"
                );
                assert!(f.bitmap.ink() > 50, "{t}: nothing drawn");
                assert_eq!(f.rtl, mg_text::is_rtl(t));
            }
            // short text: untouched size, one line
            let f = render_fit(&r, "Short", &role.style("Short", h, col), role.step).unwrap();
            assert_eq!((f.lines, f.px), (1, role.style("Short", h, col).px));
        }
        // a four-artist credit stays on one line (trimmed with an ellipsis if needed)
        let artists = "Aster Vale, Juno Reyes, The Lantern Collective & Mira Okonkwo-Lindqvist";
        let f = render_fit(&r, artists, &ARTIST_LYRICS.style(artists, h, col), 1.0).unwrap();
        assert_eq!(f.lines, 1);
        assert!((f.bitmap.width - 2 * f.bitmap.pad) as f32 <= col + 1.0);
        // a title that cannot fit in two lines steps down once, then ellipsises (still two lines)
        let long = "word ".repeat(40);
        let st = TITLE_SOLO.style(&long, h, col);
        let f = render_fit(&r, &long, &st, TITLE_SOLO.step).unwrap();
        assert!((f.px - st.px * TITLE_SOLO.step).abs() < 0.01 && f.lines == 2);
        // balanced: a two-line title is not wider than needed (the second line is not a lone word)
        let two = "The Night We Walked Home Through The Rain";
        let st = TITLE_SOLO.style(two, h, 700.0);
        let f = render_fit(&r, two, &st, 1.0).unwrap();
        if f.lines == 2 {
            assert!(((f.bitmap.width - 2 * f.bitmap.pad) as f32) < 700.0 * 0.95);
        }
    }

    // ---- Album Focus part 2: art colours, mips, blur bridge, text pocket ---------------------------------------

    fn solid(w: u32, h: u32, c: [u8; 4]) -> ArtImage {
        ArtImage {
            width: w,
            height: h,
            rgba: (0..w * h).flat_map(|_| c).collect(),
        }
    }

    #[test]
    fn mip_chain_has_the_right_sizes_averages_in_linear_light_and_bounds_memory() {
        // odd and non-square sizes follow max(1, w >> level)
        for (w, h) in [(512u32, 512u32), (500, 300), (7, 5), (3, 2), (1, 1), (2, 9)] {
            let e = art_extra(&solid(w, h, [10, 200, 90, 255]), ART_MIP_LEVELS).expect("extra");
            let possible = (w.max(h).ilog2() as usize).min(ART_MIP_LEVELS);
            assert_eq!(e.mips.len(), possible, "{w}x{h}");
            for (i, m) in e.mips.iter().enumerate() {
                let (mw, mh) = ((w >> (i + 1)).max(1), (h >> (i + 1)).max(1));
                assert_eq!(m.len(), (mw * mh * 4) as usize, "{w}x{h} level {}", i + 1);
                // a flat cover stays that colour (sRGB round trip within one step)
                assert!(
                    m.chunks_exact(4).all(|p| p[0].abs_diff(10) <= 1
                        && p[1].abs_diff(200) <= 1
                        && p[2].abs_diff(90) <= 1
                        && p[3] == 255),
                    "{w}x{h} level {} drifted",
                    i + 1
                );
            }
        }
        // a black/white checker averages to mid grey IN LINEAR LIGHT (0.5 linear = sRGB 188), not to sRGB 128
        let mut c = solid(4, 4, [0, 0, 0, 255]);
        for y in 0..4usize {
            for x in 0..4usize {
                if (x + y) % 2 == 0 {
                    c.rgba[(y * 4 + x) * 4..][..3].copy_from_slice(&[255, 255, 255]);
                }
            }
        }
        let e = art_extra(&c, 1).expect("extra");
        assert!(e.mips[0].chunks_exact(4).all(|p| p[0].abs_diff(188) <= 1));
        // zero levels = no mips (reduced motion), the colours are still made
        assert!(art_extra(&c, 0).expect("extra").mips.is_empty());
        // malformed input is refused, never a panic
        assert!(
            art_extra(
                &ArtImage {
                    width: 4,
                    height: 4,
                    rgba: vec![0; 10]
                },
                2
            )
            .is_none()
        );
        assert!(art_extra(&solid(0, 0, [0; 4]), 2).is_none());
        // the extra levels cost a third of the base at most
        let big = art_extra(&solid(512, 512, [1, 2, 3, 255]), ART_MIP_LEVELS).expect("extra");
        assert!(big.mips.iter().map(Vec::len).sum::<usize>() <= 512 * 512 * 4 / 3);
    }

    #[test]
    fn room_light_colours_follow_the_cover_layout_and_luminance() {
        // left half red, right half blue, top row brighter: the 8x8 grid is row-major, row 0 = top
        let (w, h) = (16u32, 16u32);
        let mut c = solid(w, h, [0, 0, 255, 255]);
        for y in 0..h as usize {
            for x in 0..(w / 2) as usize {
                c.rgba[(y * w as usize + x) * 4..][..3].copy_from_slice(&[255, 0, 0]);
            }
        }
        let l = art_extra(&c, 0).expect("extra").light;
        let at = |cx: usize, cy: usize| l.grid[cy * ART_GRID + cx];
        assert!(at(0, 0)[0] > 0.99 && at(0, 0)[2] < 0.01, "top-left is red");
        assert!(
            at(7, 7)[2] > 0.99 && at(7, 7)[0] < 0.01,
            "bottom-right is blue"
        );
        assert!(at(0, 7)[0] > 0.99 && at(7, 0)[2] > 0.99);
        // the mean is linear light: half red + half blue
        assert!((l.mean[0] - 0.5).abs() < 1e-3 && (l.mean[2] - 0.5).abs() < 1e-3);
        assert!((l.mean_y - (0.2126 * 0.5 + 0.0722 * 0.5)).abs() < 1e-3);
        // white and black
        let white = art_extra(&solid(8, 8, [255; 4]), 0).unwrap().light;
        let black = art_extra(&solid(8, 8, [0, 0, 0, 255]), 0).unwrap().light;
        assert!((white.mean_y - 1.0).abs() < 1e-3 && black.mean_y == 0.0);
        // tiny covers (smaller than the grid) still fill every cell
        let one = art_extra(&solid(1, 1, [255, 255, 0, 255]), 0)
            .unwrap()
            .light;
        assert!(one.grid.iter().all(|c| c[0] > 0.99 && c[2] < 0.01));
    }

    #[test]
    fn process_art_makes_the_extra_on_the_calling_thread_only_when_decodable() {
        // undecodable bytes: no image, no palette, no extra, and nothing panics
        let Ready::Art {
            image,
            palette,
            extra,
            ..
        } = process_art(5, &[1, 2, 3], true)
        else {
            panic!("art result expected")
        };
        assert!(image.is_none() && palette.is_none() && extra.is_none());
        // a real PNG decodes to an image whose extra matches `art_extra` of that image (mips only when asked)
        let png = mg_media::synth::png_rgba(8, 8, &vec![200u8; 8 * 8 * 4]);
        for (mips, n) in [(true, ART_MIP_LEVELS), (false, 0)] {
            let Ready::Art { image, extra, .. } = process_art(5, &png, mips) else {
                panic!("art result expected")
            };
            let (Some(image), Some(extra)) = (image, extra) else {
                return; // no image decoder in this environment: nothing more to verify
            };
            assert_eq!(extra.mips.len(), n);
            assert_eq!(Some(extra), art_extra(&image, n));
        }
    }

    #[test]
    fn stale_artwork_colours_and_mips_never_reach_the_gpu_ops() {
        let extra = || art_extra(&solid(8, 8, [255, 0, 0, 255]), 2);
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "Old", Some(Arc::from(vec![1u8])))), SURF);
        p.take_jobs();
        p.on_now_playing(Some(&np(2, "New", Some(Arc::from(vec![2u8])))), SURF);
        p.take_jobs();
        // the slow worker answers for track 1 after track 2 started
        p.on_ready(Ready::Art {
            track_id: 1,
            image: Some(solid(8, 8, [255, 0, 0, 255])),
            palette: Some(Palette::fallback()),
            extra: extra(),
        });
        run(&mut p, 1.0);
        assert!(p.take_ops().is_empty(), "stale cover or colours uploaded");
        // the current track's cover brings its own extra along, in the same op
        p.on_ready(Ready::Art {
            track_id: 2,
            image: Some(solid(8, 8, [0, 255, 0, 255])),
            palette: Some(Palette::fallback()),
            extra: art_extra(&solid(8, 8, [0, 255, 0, 255]), 2),
        });
        run(&mut p, 0.1);
        let ops = p.take_ops();
        let [GpuOp::SetArt(img, Some(e))] = ops.as_slice() else {
            panic!("expected one SetArt with its extra")
        };
        assert_eq!(img.rgba[1], 255);
        assert!(
            e.light.mean[1] > 0.99 && e.light.mean[0] < 0.01,
            "colours belong to the new cover"
        );
    }

    #[test]
    fn the_blur_bridge_is_two_mip_levels_at_the_middle_and_sharp_at_both_ends() {
        assert_eq!(art_blur_lod(0.0), 0.0);
        assert!((art_blur_lod(0.5) - 2.0).abs() < 1e-5);
        assert!(art_blur_lod(1.0).abs() < 1e-5);
        assert!(art_blur_lod(f32::NAN) == 0.0 && art_blur_lod(7.0).abs() < 1e-5);
        let mut last = 0.0;
        for i in 0..=50 {
            let l = art_blur_lod(i as f32 / 100.0);
            assert!(l >= last - 1e-6, "monotonic up to the middle");
            last = l;
        }
        // in the presentation: only while a crossfade runs, on both pictures, never in reduced motion
        let crossfade = |rm: bool| {
            let mut p = Presentation::new(Layout::Corner);
            p.set_reduced_motion(rm);
            p.on_now_playing(Some(&np(1, "A", Some(Arc::from(vec![1u8])))), SURF);
            p.take_jobs();
            p.on_ready(Ready::Art {
                track_id: 1,
                image: Some(img([200, 20, 20, 255])),
                palette: None,
                extra: None,
            });
            let steady = run(&mut p, 1.5);
            assert_eq!(
                steady.art.expect("cover").lod,
                0.0,
                "a steady cover is sharp"
            );
            p.on_now_playing(Some(&np(2, "B", Some(Arc::from(vec![2u8])))), SURF);
            p.take_jobs();
            run(&mut p, 0.05);
            p.on_ready(Ready::Art {
                track_id: 2,
                image: Some(img([20, 20, 200, 255])),
                palette: None,
                extra: None,
            });
            p.update(DT, SURF, 0.5, 0.2);
            let mut peak: f32 = 0.0;
            let mut both = true;
            for _ in 0..60 {
                let o = p.update(DT, SURF, 0.5, 0.2);
                if let (Some(a), Some(b)) = (o.art, o.art_prev) {
                    peak = peak.max(a.lod);
                    both &= a.lod == b.lod;
                }
            }
            let end = run(&mut p, 1.0);
            assert_eq!(
                end.art.expect("new cover").lod,
                0.0,
                "sharp again at the end"
            );
            (peak, both)
        };
        let (peak, both) = crossfade(false);
        assert!(peak > 1.5 && both, "bridge peak {peak}");
        assert_eq!(crossfade(true).0, 0.0, "reduced motion has no blur");
    }

    #[test]
    fn the_text_pocket_serves_each_role_its_contrast_and_follows_the_text() {
        // cap math: the WCAG ratio of text over a background at the cap is at least the requirement
        for (yt, ratio) in [(0.857f32, 7.0f32), (0.512, 4.5), (0.238, 3.0), (0.287, 3.0)] {
            let cap = pocket_cap(yt, ratio);
            assert!((0.01..=0.06).contains(&cap));
            assert!(
                (yt + 0.05) / (cap + 0.05) >= ratio,
                "{yt} {ratio} cap {cap}"
            );
        }
        // geometry: with a cover, text and lyrics the feed has one rectangle per element, in screen heights
        let mut p = Presentation::new(Layout::Center);
        p.set_album_scene(true, SURF);
        p.set_lyric_source(LyricSource::Demo);
        p.set_bar(SURF.1 as f32 - 128.0, 0.0);
        p.on_now_playing(Some(&np(1, "Title", Some(Arc::from(vec![1u8])))), SURF);
        p.set_position(Some(10.5));
        for _ in 0..240 {
            pump(&mut p);
            p.update(DT, SURF, 0.5, 0.2);
            p.take_ops();
        }
        let o = p.update(DT, SURF, 0.5, 0.2);
        let mut f = FocusParams::default();
        p.merge_focus(&mut f);
        let h = SURF.1 as f32;
        let a = o.art.expect("cover");
        assert!((f.rect[0] - a.cx / h).abs() < 1e-5 && (f.rect[3] - a.hh / h).abs() < 1e-5);
        assert!(f.art_w[0] > 0.99 && f.art_w[0] + f.art_w[1] <= 1.0 + 1e-6);
        let active = (0..4).filter(|&i| f.ink_on[i] > 0.0).count();
        let drawn = [o.title, o.artist, o.meta]
            .iter()
            .flatten()
            .filter(|d| d.alpha > 0.002)
            .count()
            + usize::from(o.lyrics.iter().flatten().any(|d| d.alpha > 0.002));
        assert!(
            drawn >= 2 && active == drawn,
            "pockets {active} vs drawn elements {drawn}"
        );
        for i in 0..4 {
            if f.ink_on[i] > 0.0 {
                let r = f.ink[i];
                assert!(r[2] > r[0] && r[3] > r[1] && r[0] >= 0.0 && r[3] <= 1.0 + 0.05);
                assert!((0.01..=0.06).contains(&f.ink_cap[i]));
            }
        }
        // the first rectangle is the title's own draw
        let t = o.title.expect("title");
        assert!(
            (f.ink[0][0] - t.x / h).abs() < 1e-5 && (f.ink[0][3] - (t.y + t.h) / h).abs() < 1e-5
        );
        // no cover: the light source is the (empty) cover slot, with no cover weight
        let mut q = Presentation::new(Layout::Center);
        q.set_album_scene(true, SURF);
        q.on_now_playing(Some(&np(1, "Title", None)), SURF);
        pump(&mut q);
        q.update(DT, SURF, 0.5, 0.2);
        let mut g = FocusParams::default();
        q.merge_focus(&mut g);
        assert_eq!(g.art_w, [0.0, 0.0]);
        assert!(
            g.rect[2] > 0.1 && g.rect[3] > 0.1,
            "the slot, not a point: {:?}",
            g.rect
        );
        // a text that is not shown has no pocket
        let mut r = Presentation::new(Layout::Center);
        r.set_album_scene(true, SURF);
        r.update(DT, SURF, 0.5, 0.2);
        let mut z = FocusParams::default();
        r.merge_focus(&mut z);
        assert_eq!(z.ink_on, [0.0; 4]);
    }
}
