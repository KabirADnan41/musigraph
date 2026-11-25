//! Track presentation: now-playing metadata -> artwork, title/artist text, album palette and layout.
//!
//! Split in two so the rules are testable without a GPU or a media session:
//! * `Presentation` (pure, no I/O): reacts to track changes, accepts ready results only for the CURRENT
//!   track (a late result for a previous track is dropped), fades old content out before new content is
//!   uploaded, eases layout rectangles and the album palette, and produces `OverlayParams`.
//! * `Worker` (one background thread): decodes artwork (bounded), extracts the palette and renders text
//!   bitmaps, so nothing here ever blocks the render loop. Latest request wins; superseded ones are skipped.
//!
//! Privacy: titles, artists and artwork are shown on screen only. They are never logged, written to disk
//! or sent anywhere.

use crate::lyric_view::{LyricSource, LyricView};
use mg_gfx::overlay::{ArtDraw, OverlayParams, TextDraw};
use mg_media::lrc::Lyrics;
use mg_media::palette::Palette;
use mg_media::{ArtImage, NowPlaying};
use mg_text::TextBitmap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long the album palette takes to move to a new track's colours.
pub const PALETTE_SECS: f32 = 2.5;
const FADE_OUT_S: f32 = 0.35;
const FADE_IN_S: f32 = 0.8;
const LAYOUT_EASE_S: f32 = 0.22;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Compact artwork and text in the lower-left corner; the visualisation stays dominant.
    Corner,
    /// Large artwork in the middle with the title below; the atmosphere surrounds it.
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

// ---- worker protocol --------------------------------------------------------------------------------

pub struct TextJob {
    pub track_id: u64,
    pub serial: u64,
    pub title: String,
    pub artist: String,
    pub title_px: f32,
    pub artist_px: f32,
    pub max_width: u32,
}

pub enum Job {
    Art {
        track_id: u64,
        bytes: Arc<[u8]>,
    },
    Text(TextJob),
    /// Find and parse the user-supplied lyrics file for a track (or build the demo lyrics).
    Lyrics {
        track_id: u64,
        source: LyricSource,
        title: String,
        artist: String,
    },
    /// Render one lyric line.
    Line {
        track_id: u64,
        serial: u64,
        idx: usize,
        text: String,
        px: f32,
        max_width: u32,
    },
}

pub enum Ready {
    Art {
        track_id: u64,
        image: Option<ArtImage>,
        palette: Option<Palette>,
    },
    Text {
        track_id: u64,
        serial: u64,
        title: Option<TextBitmap>,
        artist: Option<TextBitmap>,
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
}

/// What the GPU side must do (applied by the render loop; uploads happen once per track change).
pub enum GpuOp {
    SetArt(ArtImage),
    ClearArt,
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

// ---- the state machine ------------------------------------------------------------------------------

#[derive(Default)]
struct Fade {
    alpha: f32,
    want: bool,
}

impl Fade {
    fn step(&mut self, dt: f32) {
        let rate = if self.want {
            1.0 / FADE_IN_S
        } else {
            1.0 / FADE_OUT_S
        };
        let target = if self.want { 1.0 } else { 0.0 };
        let d = rate * dt;
        self.alpha += (target - self.alpha).clamp(-d, d);
    }
}

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

pub struct Presentation {
    layout: Layout,
    track_id: Option<u64>,
    title: String,
    artist: String,
    art_requested: bool,
    serial: u64,
    /// Serial of the request made for the current track change; later serials are same-track re-renders.
    track_serial: u64,
    text_for_height: u32,
    since_text_request: f32,
    jobs: Vec<Job>,
    ops: Vec<GpuOp>,
    art_fade: Fade,
    art_aspect: f32,
    art_pending: Option<Option<ArtImage>>,
    has_art_tex: bool,
    text_fade: Fade,
    text_pending: Option<(u64, Option<TextBitmap>, Option<TextBitmap>)>,
    /// (width, height, pad) of the uploaded title and artist bitmaps.
    text_dims: [Option<(u32, u32, u32)>; 2],
    palette: PaletteState,
    e_art: [Eased; 4],
    e_title: [Eased; 2],
    e_artist: [Eased; 2],
    lyrics: LyricView,
    lyric_draws: [Option<TextDraw>; 3],
    /// User switch: lyrics are still looked up and tracked, only their drawing is suppressed.
    lyrics_on: bool,
}

/// Pixel sizes derived from the surface height and layout (responsive: nothing is tied to one monitor).
fn text_sizes(layout: Layout, h: f32) -> (f32, f32) {
    match layout {
        Layout::Center => (0.046 * h, 0.030 * h),
        _ => (0.032 * h, 0.024 * h),
    }
}

fn text_max_width(layout: Layout, w: f32, h: f32) -> u32 {
    let m = match layout {
        Layout::Center => 0.80 * w,
        _ => (0.44 * w).min(1.6 * h),
    };
    m.max(64.0) as u32
}

impl Presentation {
    pub fn new(layout: Layout) -> Self {
        Self {
            layout,
            track_id: None,
            title: String::new(),
            artist: String::new(),
            art_requested: false,
            serial: 0,
            track_serial: 0,
            text_for_height: 0,
            since_text_request: 0.0,
            jobs: Vec::new(),
            ops: Vec::new(),
            art_fade: Fade::default(),
            art_aspect: 1.0,
            art_pending: None,
            has_art_tex: false,
            text_fade: Fade::default(),
            text_pending: None,
            text_dims: [None, None],
            palette: PaletteState::new(),
            lyrics: LyricView::new(LyricSource::None),
            lyrics_on: true,
            lyric_draws: [None; 3],
            e_art: [Eased::default(); 4],
            e_title: [Eased::default(); 2],
            e_artist: [Eased::default(); 2],
        }
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn set_layout(&mut self, l: Layout, surface: (u32, u32)) {
        if l != self.layout {
            self.layout = l;
            self.request_text(surface); // sizes differ per layout
        }
    }

    pub fn take_jobs(&mut self) -> Vec<Job> {
        std::mem::take(&mut self.jobs)
    }

    /// Where synchronised lyrics may come from (default: nowhere).
    /// Shows or hides the lyrics (returns the new state).
    pub fn toggle_lyrics(&mut self) -> bool {
        self.lyrics_on = !self.lyrics_on;
        self.lyrics_on
    }

    pub fn set_lyric_source(&mut self, s: LyricSource) {
        self.lyrics = LyricView::new(s);
    }

    /// Playback position in seconds for the lyrics (None when unknown or nothing plays).
    pub fn set_position(&mut self, pos: Option<f64>) {
        self.lyrics.set_position(pos);
    }

    /// Vertical anchor of the current lyric line (fraction of the window height), set per scene.
    pub fn set_lyric_anchor(&mut self, frac: f32) {
        self.lyrics.set_anchor(frac);
    }

    pub fn take_ops(&mut self) -> Vec<GpuOp> {
        std::mem::take(&mut self.ops)
    }

    /// Current album palette and how strongly it applies (0 = scene's own colours).
    pub fn album(&self) -> (Palette, f32) {
        (self.palette.cur, self.palette.w)
    }

    fn request_text(&mut self, surface: (u32, u32)) {
        if self.track_id.is_none() || self.title.is_empty() || surface.1 == 0 {
            return;
        }
        let (h, w) = (surface.1 as f32, surface.0 as f32);
        let (tp, ap) = text_sizes(self.layout, h);
        self.serial += 1;
        self.text_for_height = surface.1;
        self.since_text_request = 0.0;
        self.jobs.push(Job::Text(TextJob {
            track_id: self.track_id.unwrap_or(0),
            serial: self.serial,
            title: self.title.clone(),
            artist: self.artist.clone(),
            title_px: tp,
            artist_px: ap,
            max_width: text_max_width(self.layout, w, h),
        }));
    }

    /// Feeds the latest now-playing snapshot (or `None` when nothing plays). Cheap; call a few times a second.
    pub fn on_now_playing(&mut self, np: Option<&NowPlaying>, surface: (u32, u32)) {
        let id = np.map(|n| n.track_id);
        if id != self.track_id {
            // New track (or none): everything of the old track fades out; late results for it are ignored.
            self.track_id = id;
            self.art_pending = None;
            self.text_pending = None;
            self.art_fade.want = false;
            self.text_fade.want = false;
            self.art_requested = false;
            self.serial += 1; // invalidates text results still in flight
            self.track_serial = self.serial;
            self.text_for_height = 0; // the new track has no text yet (also when the window is minimised)
            self.palette.set_target(None);
            self.title.clear();
            self.artist.clear();
            let (t, a) = np.map_or((String::new(), String::new()), |n| {
                (n.title.clone(), n.artist.clone())
            });
            self.lyrics.on_track(id, &t, &a, &mut self.jobs);
            if let Some(n) = np {
                self.title = n.title.clone();
                self.artist = n.artist.clone();
                self.request_text(surface);
                self.track_serial = self.serial;
                if let (Some(bytes), false) = (&n.art, self.art_requested) {
                    self.art_requested = true;
                    self.jobs.push(Job::Art {
                        track_id: n.track_id,
                        bytes: bytes.clone(),
                    });
                }
            }
            if np.is_none_or(|n| n.art.is_none()) {
                self.art_pending = Some(None); // the previous cover must not stay (texture, layout gap)
            }
        } else if let Some(n) = np {
            // Same track: artwork may arrive late (the session loads it after the title).
            if !self.art_requested {
                if let Some(bytes) = &n.art {
                    self.art_requested = true;
                    self.jobs.push(Job::Art {
                        track_id: n.track_id,
                        bytes: bytes.clone(),
                    });
                }
            }
        }
    }

    /// Accepts a worker result. Results for any other track (or a superseded text request) are dropped.
    pub fn on_ready(&mut self, r: Ready) {
        match r {
            r @ (Ready::Lyrics { .. } | Ready::Line { .. }) => self.lyrics.on_ready(r),
            Ready::Art {
                track_id,
                image,
                palette,
            } => {
                if Some(track_id) != self.track_id {
                    return;
                }
                self.palette.set_target(palette);
                self.art_pending = Some(image);
            }
            Ready::Text {
                track_id,
                serial,
                title,
                artist,
            } => {
                if Some(track_id) != self.track_id || serial != self.serial {
                    return;
                }
                self.text_pending = Some((serial, title, artist));
            }
        }
    }

    /// Advances fades, easing and palette; emits GPU ops for content that is ready and whose predecessor
    /// has faded out; returns the overlay to draw. `energy` and `swell` (0..1) drive the artwork halo.
    pub fn update(
        &mut self,
        dt: f32,
        surface: (u32, u32),
        energy: f32,
        swell: f32,
    ) -> OverlayParams {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.25)
        } else {
            0.0
        };
        self.palette.update(dt);
        self.art_fade.step(dt);
        self.text_fade.step(dt);
        self.since_text_request += dt;
        self.lyric_draws = self
            .lyrics
            .update(dt, surface, &mut self.ops, &mut self.jobs);

        if self.art_pending.is_some() && self.art_fade.alpha <= 0.02 {
            match self.art_pending.take().flatten() {
                Some(img) => {
                    self.art_aspect = img.width as f32 / img.height.max(1) as f32;
                    self.ops.push(GpuOp::SetArt(img));
                    self.has_art_tex = true;
                    self.art_fade.want = true;
                }
                None => {
                    if self.has_art_tex {
                        self.ops.push(GpuOp::ClearArt);
                    }
                    self.has_art_tex = false;
                    self.art_fade.want = false;
                }
            }
        }
        // A same-track re-render (resize, layout change) swaps in at once; a new track waits for the fade-out.
        let text_ready = self.text_pending.as_ref().is_some_and(|(s, _, _)| {
            (*s != self.track_serial && self.text_fade.want) || self.text_fade.alpha <= 0.02
        });
        if text_ready {
            let (_, t, a) = self.text_pending.take().unwrap_or((0, None, None));
            for (i, b) in [t, a].into_iter().enumerate() {
                match b {
                    Some(b) => {
                        self.text_dims[i] = Some((b.width, b.height, b.pad));
                        self.ops.push(GpuOp::SetText(i, b));
                    }
                    None => {
                        if self.text_dims[i].take().is_some() {
                            self.ops.push(GpuOp::ClearText(i));
                        }
                    }
                }
            }
            self.text_fade.want = self.text_dims.iter().any(Option::is_some);
        }
        // A resized window or a layout change needs text at a new pixel size (debounced).
        if self.track_id.is_some()
            && !self.title.is_empty()
            && surface.1 != 0
            && (self.text_for_height == 0
                || (self.since_text_request > 0.6
                    && (surface.1 as f32 / self.text_for_height as f32 - 1.0).abs() > 0.15))
        {
            self.request_text(surface);
        }

        if self.layout == Layout::Off || surface.0 == 0 || surface.1 == 0 {
            return self.params(surface, energy, None, None, None);
        }
        let (w, h) = (surface.0 as f32, surface.1 as f32);
        let t = self.targets(w, h);
        let art_snap = self.art_fade.alpha < 0.01;
        let text_snap = self.text_fade.alpha < 0.01;
        let art = self.has_art_tex.then(|| {
            let e = &mut self.e_art;
            let cx = e[0].to(t.art[0], dt, art_snap);
            let cy = e[1].to(t.art[1], dt, art_snap);
            let hw = e[2].to(t.art[2], dt, art_snap);
            let hh = e[3].to(t.art[3], dt, art_snap);
            let react = 0.20 + 0.55 * energy.clamp(0.0, 1.0) + 0.15 * swell.clamp(0.0, 1.0);
            ArtDraw {
                cx,
                cy,
                hw,
                hh,
                alpha: self.art_fade.alpha,
                radius: 0.012 * h,
                halo: react,
                spread: (0.035 + 0.05 * swell.clamp(0.0, 1.0))
                    * h
                    * if self.layout == Layout::Center {
                        1.8
                    } else {
                        1.0
                    },
            }
        });
        let line = |i: usize, e: &mut [Eased; 2], target: Option<[f32; 2]>| {
            let (tw, th, pad) = self.text_dims[i]?;
            let [tx, ty] = target?;
            let x = e[0].to(tx - pad as f32, dt, text_snap);
            let y = e[1].to(ty - pad as f32, dt, text_snap);
            Some(TextDraw {
                x,
                y,
                w: tw as f32,
                h: th as f32,
                alpha: self.text_fade.alpha,
            })
        };
        let title = line(0, &mut self.e_title, t.title_xy(self.text_dims[0], w));
        let artist = line(1, &mut self.e_artist, t.artist_xy(self.text_dims[1], w));
        self.params(surface, energy, art, title, artist)
    }

    fn params(
        &self,
        surface: (u32, u32),
        energy: f32,
        art: Option<ArtDraw>,
        title: Option<TextDraw>,
        artist: Option<TextDraw>,
    ) -> OverlayParams {
        let (p, w) = (self.palette.cur, self.palette.w);
        let scene_text = Palette::fallback().text;
        OverlayParams {
            res: surface,
            time: 0.0,
            energy,
            art,
            title,
            artist,
            lyrics: if self.lyrics_on {
                self.lyric_draws
            } else {
                [None, None, None]
            },
            lyric_glow: energy.clamp(0.0, 1.0),
            shadow: 0.55,
            text_color: lerp3(scene_text, p.text, w),
            halo_color: lerp3(
                Palette::fallback().secondary,
                lerp3(p.main, p.accent, 0.5),
                w,
            ),
        }
    }

    fn targets(&self, w: f32, h: f32) -> Targets {
        let has_art = self.has_art_tex;
        let aspect = self.art_aspect.clamp(0.2, 5.0);
        // Fit the image inside a square slot (aspect preserved, never stretched).
        let (sw, sh) = if aspect >= 1.0 {
            (1.0, 1.0 / aspect)
        } else {
            (aspect, 1.0)
        };
        match self.layout {
            Layout::Center => {
                let side = 0.36 * h;
                let (cx, cy) = (0.5 * w, if has_art { 0.37 * h } else { 0.42 * h });
                let top = cy
                    + if has_art {
                        0.5 * side + 0.045 * h
                    } else {
                        -0.02 * h
                    };
                Targets {
                    art: [cx, cy, 0.5 * side * sw, 0.5 * side * sh],
                    center: true,
                    left: 0.0,
                    title_y: top,
                    artist_y: top + 0.062 * h,
                }
            }
            _ => {
                let m = 0.045 * h;
                let side = 0.15 * h;
                let left = if has_art { m + side + 0.022 * h } else { m };
                let cy = h - m - 0.5 * side;
                Targets {
                    art: [m + 0.5 * side, cy, 0.5 * side * sw, 0.5 * side * sh],
                    center: false,
                    left,
                    title_y: cy - 0.055 * h,
                    artist_y: cy + 0.002 * h,
                }
            }
        }
    }
}

struct Targets {
    art: [f32; 4],
    center: bool,
    left: f32,
    title_y: f32,
    artist_y: f32,
}

impl Targets {
    /// Top-left of the text INK (the bitmap pad is subtracted by the caller).
    fn xy(&self, dims: Option<(u32, u32, u32)>, w: f32, y: f32) -> Option<[f32; 2]> {
        let (tw, _, pad) = dims?;
        let ink = tw.saturating_sub(2 * pad) as f32;
        let x = if self.center {
            0.5 * (w - ink)
        } else {
            self.left
        };
        Some([x, y])
    }
    fn title_xy(&self, d: Option<(u32, u32, u32)>, w: f32) -> Option<[f32; 2]> {
        self.xy(d, w, self.title_y)
    }
    fn artist_xy(&self, d: Option<(u32, u32, u32)>, w: f32) -> Option<[f32; 2]> {
        self.xy(d, w, self.artist_y)
    }
}

// ---- worker thread ----------------------------------------------------------------------------------

pub struct Worker {
    tx: Option<Sender<Job>>,
    rx: Receiver<Ready>,
    thread: Option<JoinHandle<()>>,
    done: Arc<std::sync::atomic::AtomicBool>,
}

pub fn process_art(track_id: u64, bytes: &[u8]) -> Ready {
    let image = mg_media::decode_art(bytes);
    let palette = image.as_ref().and_then(|img| {
        let (px, w, h) = img.thumbnail(64);
        (w > 0 && h > 0).then(|| mg_media::palette::extract(&px, w, h))
    });
    Ready::Art {
        track_id,
        image,
        palette,
    }
}

/// Renders the title and artist lines of a job (title semibold, artist regular).
pub fn process_text(r: &mg_text::TextRenderer, t: &TextJob) -> Ready {
    let line = |s: &str, px: f32, weight: u16| {
        r.render_line(
            s,
            &mg_text::LineStyle {
                px,
                weight,
                max_width: t.max_width,
            },
        )
    };
    Ready::Text {
        track_id: t.track_id,
        serial: t.serial,
        title: line(&t.title, t.title_px, 600),
        artist: line(&t.artist, t.artist_px, 400),
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

/// Renders one lyric line (regular weight).
pub fn process_line(
    r: Option<&mg_text::TextRenderer>,
    track_id: u64,
    serial: u64,
    idx: usize,
    text: &str,
    px: f32,
    max_width: u32,
) -> Ready {
    let bitmap = r.and_then(|r| {
        r.render_line(
            text,
            &mg_text::LineStyle {
                px,
                weight: 500,
                max_width,
            },
        )
    });
    Ready::Line {
        track_id,
        serial,
        idx,
        bitmap,
    }
}

/// Runs any job synchronously (used by the worker and by the headless self-test).
pub fn process_job(text: Option<&mg_text::TextRenderer>, job: Job) -> Ready {
    match job {
        Job::Art { track_id, bytes } => process_art(track_id, &bytes),
        Job::Text(t) => match text {
            Some(r) => process_text(r, &t),
            None => Ready::Text {
                track_id: t.track_id,
                serial: t.serial,
                title: None,
                artist: None,
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
            px,
            max_width,
        } => process_line(text, track_id, serial, idx, &line, px, max_width),
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
                Job::Line { .. } => lines.push(j),
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

    #[test]
    fn lyrics_toggle_hides_drawing_without_losing_state() {
        let mut p = Presentation::new(Layout::Corner);
        assert!(p.lyrics_on);
        let d = TextDraw {
            x: 1.0,
            y: 2.0,
            w: 3.0,
            h: 4.0,
            alpha: 1.0,
        };
        p.lyric_draws = [Some(d), None, None];
        assert!(p.params(SURF, 0.0, None, None, None).lyrics[0].is_some());
        assert!(!p.toggle_lyrics());
        assert!(
            p.params(SURF, 0.0, None, None, None)
                .lyrics
                .iter()
                .all(Option::is_none)
        );
        assert_eq!(
            p.lyric_draws[0].map(|d| d.w),
            Some(3.0),
            "tracking state is kept"
        );
        assert!(p.toggle_lyrics());
        assert!(p.params(SURF, 0.0, None, None, None).lyrics[0].is_some());
    }

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

    fn bmp() -> TextBitmap {
        TextBitmap {
            width: 40,
            height: 20,
            pad: 4,
            coverage: vec![200; 800],
        }
    }

    fn run(p: &mut Presentation, secs: f32) -> OverlayParams {
        let mut o = None;
        for _ in 0..(secs * 60.0) as usize {
            o = Some(p.update(1.0 / 60.0, SURF, 0.5, 0.2));
        }
        o.expect("at least one frame")
    }

    fn art_job_count(jobs: &[Job]) -> usize {
        jobs.iter().filter(|j| matches!(j, Job::Art { .. })).count()
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
        });
        p.on_ready(Ready::Text {
            track_id: 1,
            serial: 1,
            title: Some(bmp()),
            artist: None,
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
            title: Some(bmp()),
            artist: None,
        });
        run(&mut p, 1.0);
        assert!(
            p.take_ops().is_empty(),
            "the superseded text must be dropped"
        );
    }

    #[test]
    fn old_content_fades_out_before_new_content_is_uploaded() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "A", Some(Arc::from(vec![1u8])))), SURF);
        p.take_jobs();
        p.on_ready(Ready::Art {
            track_id: 1,
            image: Some(img([200, 20, 20, 255])),
            palette: Some(Palette::fallback()),
        });
        let o = run(&mut p, 1.5);
        assert!(matches!(p.take_ops().as_slice(), [GpuOp::SetArt(_)]));
        assert!(o.art.expect("shown").alpha > 0.99);
        // track 2 arrives with its artwork already decoded
        p.on_now_playing(Some(&np(2, "B", Some(Arc::from(vec![2u8])))), SURF);
        p.take_jobs();
        p.on_ready(Ready::Art {
            track_id: 2,
            image: Some(img([20, 20, 200, 255])),
            palette: Some(Palette::fallback()),
        });
        let o = p.update(1.0 / 60.0, SURF, 0.5, 0.2);
        assert!(
            p.take_ops().is_empty(),
            "uploaded while the old artwork is still visible"
        );
        assert!(
            o.art.expect("old art still drawn").alpha < 1.0,
            "fading out"
        );
        let o = run(&mut p, 2.0);
        assert!(matches!(p.take_ops().as_slice(), [GpuOp::SetArt(_)]));
        assert!(o.art.expect("new art").alpha > 0.99);
    }

    #[test]
    fn missing_artwork_shows_text_only_and_stopping_clears_everything() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "Only text", None)), SURF);
        p.take_jobs();
        p.on_ready(Ready::Text {
            track_id: 1,
            serial: p.serial,
            title: Some(bmp()),
            artist: Some(bmp()),
        });
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
            s.update(1.0 / 60.0);
        }
        s.set_target(Some(b));
        let mut prev = scene_roles(&s.cur);
        let mut max_step = 0.0f32;
        for _ in 0..200 {
            s.update(1.0 / 60.0);
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
            s.update(1.0 / 60.0);
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
        }
    }

    #[test]
    fn layouts_stay_on_screen_for_any_aspect_and_preserve_artwork_aspect() {
        for layout in [Layout::Corner, Layout::Center] {
            for &(w, h) in &[
                (1920u32, 1080u32),
                (3440, 1440),
                (1024, 768),
                (720, 1280),
                (320, 180),
            ] {
                let mut p = Presentation::new(layout);
                p.on_now_playing(Some(&np(1, "T", None)), (w, h));
                p.take_jobs();
                p.art_aspect = 2.0; // a 2:1 image
                p.has_art_tex = true;
                p.art_fade = Fade {
                    alpha: 1.0,
                    want: true,
                };
                let tw = text_max_width(layout, w as f32, h as f32).min(1000) + 20; // a real bitmap never exceeds the layout width
                let (tp, ap) = text_sizes(layout, h as f32);
                p.text_dims = [
                    Some((
                        tw,
                        (tp * 1.4) as u32 + 2 * (tp * 0.3) as u32,
                        (tp * 0.3) as u32,
                    )),
                    Some((
                        tw.min(200),
                        (ap * 1.4) as u32 + 2 * (ap * 0.3) as u32,
                        (ap * 0.3) as u32,
                    )),
                ];
                p.text_fade = Fade {
                    alpha: 1.0,
                    want: true,
                };
                let o = p.update(0.0, (w, h), 0.5, 0.2);
                let a = o.art.expect("art");
                assert!((a.hw / a.hh - 2.0).abs() < 1e-3, "aspect {}", a.hw / a.hh);
                assert!(
                    a.cx - a.hw >= 0.0 && a.cx + a.hw <= w as f32,
                    "{layout:?} {w}x{h} art x"
                );
                assert!(
                    a.cy - a.hh >= 0.0 && a.cy + a.hh <= h as f32,
                    "{layout:?} {w}x{h} art y"
                );
                let t = o.title.expect("title");
                assert!(
                    t.x + 10.0 >= 0.0 && t.x + t.w <= w as f32 + 1.0,
                    "{layout:?} {w}x{h} title x {} {}",
                    t.x,
                    t.w
                );
                let ar = o.artist.expect("artist");
                assert!(ar.y + ar.h <= h as f32 + 1.0, "{layout:?} {w}x{h} artist y");
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
        });
        let o = run(&mut p, 3.0);
        assert!(o.art.is_none() && o.title.is_none());
        assert!(p.album().1 > 0.99);
    }

    #[test]
    fn a_same_track_text_rerender_is_applied_without_waiting_for_a_fade_out() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "T", None)), SURF);
        p.take_jobs();
        p.on_ready(Ready::Text {
            track_id: 1,
            serial: p.serial,
            title: Some(bmp()),
            artist: None,
        });
        run(&mut p, 2.0);
        assert!(matches!(p.take_ops().as_slice(), [GpuOp::SetText(0, _)]));
        // layout change: new pixel size requested, its result must replace the old bitmap at once
        p.set_layout(Layout::Center, SURF);
        let Job::Text(j) = p.take_jobs().remove(0) else {
            panic!()
        };
        p.on_ready(Ready::Text {
            track_id: 1,
            serial: j.serial,
            title: Some(bmp()),
            artist: None,
        });
        let o = p.update(1.0 / 60.0, SURF, 0.5, 0.2);
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
        });
        run(&mut p, 1.5);
        p.take_ops();
        p.on_now_playing(Some(&np(2, "B", None)), SURF);
        p.take_jobs();
        let o = run(&mut p, 1.5);
        assert!(
            p.take_ops().iter().any(|op| matches!(op, GpuOp::ClearArt)),
            "cover texture kept"
        );
        assert!(o.art.is_none(), "cover (and its layout gap) must be gone");
        assert!(!p.has_art_tex);
    }

    #[test]
    fn a_track_change_while_minimised_still_gets_text_after_a_previous_track_was_shown() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "A", None)), SURF);
        p.take_jobs();
        p.on_now_playing(Some(&np(2, "B", None)), (0, 0));
        assert!(p.take_jobs().is_empty());
        p.update(1.0 / 60.0, SURF, 0.0, 0.0);
        assert_eq!(p.take_jobs().len(), 1, "second track must get its text");
    }

    #[test]
    fn text_is_requested_once_the_surface_has_a_size() {
        let mut p = Presentation::new(Layout::Corner);
        p.on_now_playing(Some(&np(1, "T", None)), (0, 0)); // minimised
        assert!(p.take_jobs().is_empty());
        p.update(1.0 / 60.0, SURF, 0.0, 0.0);
        assert_eq!(
            p.take_jobs().len(),
            1,
            "text requested after the window got a size"
        );
    }
}
