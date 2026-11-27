//! Scene panel model (pure: no GPU, no windowing). Purpose: the state of the scene picker that the Scene
//! button opens, so every rule is testable without a window. Owns: open/close and its motion, tile layout as a
//! function of the surface, hit testing, keyboard focus and the key model. Does NOT own: which scenes exist
//! (`SceneKind::panel_main` / `panel_experimental`), which scene is on screen (`SceneSwitch::target`, passed in
//! through `set_current`), the drawing (`PanelDraw`, built from these accessors in `scene_run`) or the idle timer
//! (`Visibility::set_hold`). Input in, `PanelEvent` out: the caller turns `Select` into `SceneSwitch::request`.
//! Invariant: nothing here allocates per frame (layout is rebuilt only when the surface changes).

use crate::controls::{self, Rect};
use crate::presentation::{GpuOp, Job, Ready};
use crate::scene_thumbs::Thumbs;
use mg_gfx::overlay::{
    LABEL_CAPTION, LABEL_CELLS, LABEL_CURRENT, LABEL_EXPERIMENTAL_HEADING, LABEL_EXPERIMENTAL_PILL,
    LABEL_SIZE, PANEL_TILES, PanelDraw, PanelTileDraw,
};
use mg_gfx::scene::SceneKind;
use mg_text::{Align, Cut, LineStyle, TextBitmap};

/// Tiles the panel can hold (every scene).
pub const MAX_TILES: usize = SceneKind::ALL.len();

// Design pixels at 1080p; multiplied by `controls::ui_scale`.
pub const TILE_W: f32 = 240.0;
pub const GAP: f32 = 20.0;
pub const LABEL_GAP: f32 = 12.0;
pub const LABEL_H: f32 = 20.0;
pub const HEADING_H: f32 = 14.0;
pub const HEADING_GAP: f32 = 12.0;
pub const GROUP_GAP: f32 = 36.0;
pub const MARGIN: f32 = 24.0;
/// The "Representative previews" caption: height and distance below the tiles.
pub const CAPTION_H: f32 = 14.0;
pub const CAPTION_GAP: f32 = 18.0;
/// Tile shrink factors tried (largest first) when the content does not fit the surface.
const FACTORS: [f32; 5] = [1.0, 0.85, 0.7, 0.55, 0.45];

pub const OPEN_S: f32 = 0.22;
pub const CLOSE_S: f32 = 0.15;
pub const STAGGER_S: f32 = 0.03;
pub const REDUCED_S: f32 = 0.12;
pub const HOVER_S: f32 = 0.14;
/// A click shows its selection this long before the panel closes.
pub const CLICK_CLOSE_S: f32 = 0.12;
/// Tiles rise this far (design px) while fading in.
pub const RISE_PX: f32 = 12.0;

/// Keys the panel consumes while it is open (the caller maps physical keys to these).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelKey {
    Up,
    Down,
    Left,
    Right,
    Tab,
    BackTab,
    Home,
    End,
    Enter,
    Escape,
}

/// What an input did, for the caller to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelEvent {
    None,
    /// Switch to this scene (never produced for the scene already current).
    Select(SceneKind),
    /// The panel is closing (Esc, a click outside, or re-selecting the current scene).
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tile {
    pub kind: SceneKind,
    /// The 16:9 thumbnail area, surface pixels.
    pub rect: Rect,
    /// Thumbnail plus label: what a pointer hits.
    pub hit: Rect,
    /// Top-left of the label line box (height `LABEL_H * scale`, as wide as the tile; labels are centred in it).
    pub label: Rect,
    pub experimental: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanelLayout {
    pub scale: f32,
    pub count: usize,
    pub tiles: [Tile; MAX_TILES],
    /// Where the "EXPERIMENTAL" heading's line box starts (left edge) and its box, when there are such tiles.
    pub heading: Option<Rect>,
    /// Caption line box under the tiles (as wide as the block).
    pub caption: Rect,
    /// Columns of the grid (the tile count in one-row mode).
    pub cols: usize,
    /// Tile width factor chosen to fit (1.0 = full size).
    pub factor: f32,
    /// Bounding box of everything drawn.
    pub bounds: Rect,
}

/// Panel order: the main group (selector scenes) then the Experimental group; returns the order, the size of the
/// main group and the total.
pub fn order() -> ([SceneKind; MAX_TILES], usize, usize) {
    let mut o = [SceneKind::Spectrum; MAX_TILES];
    let mut n = 0;
    for k in SceneKind::panel_main() {
        o[n] = k;
        n += 1;
    }
    let main = n;
    for k in SceneKind::panel_experimental() {
        o[n] = k;
        n += 1;
    }
    (o, main, n)
}

/// Design-px placement of one candidate: tile rectangles, the heading box and the block size.
struct Placed {
    tiles: [Rect; MAX_TILES],
    heading: Option<Rect>,
    size: (f32, f32),
}

impl Placed {
    /// Appends the caption line below the block.
    fn with_caption(mut self) -> (Self, Rect) {
        let cap = Rect {
            x: 0.0,
            y: self.size.1 + CAPTION_GAP,
            w: self.size.0,
            h: CAPTION_H,
        };
        self.size.1 += CAPTION_GAP + CAPTION_H;
        (self, cap)
    }
}

fn place(n: usize, main: usize, cols: usize, factor: f32, one_row: bool) -> Placed {
    let tw = TILE_W * factor;
    let th = tw * 9.0 / 16.0;
    let row_h = th + LABEL_GAP + LABEL_H;
    let exp = n - main;
    let mut tiles = [Rect::default(); MAX_TILES];
    let at = |x: f32, y: f32| Rect { x, y, w: tw, h: th };
    if one_row {
        let y0 = if exp > 0 {
            HEADING_H + HEADING_GAP
        } else {
            0.0
        };
        let mut x = 0.0;
        let mut heading = None;
        for (i, t) in tiles.iter_mut().enumerate().take(n) {
            if i == main && exp > 0 {
                x += GROUP_GAP - GAP;
                heading = Some(Rect {
                    x,
                    y: 0.0,
                    w: tw,
                    h: HEADING_H,
                });
            }
            *t = at(x, y0);
            x += tw + GAP;
        }
        return Placed {
            tiles,
            heading,
            size: ((x - GAP).max(0.0), y0 + row_h),
        };
    }
    let rows = |c: usize| c.div_ceil(cols);
    for (i, t) in tiles.iter_mut().enumerate().take(main) {
        *t = at(
            (i % cols) as f32 * (tw + GAP),
            (i / cols) as f32 * (row_h + GAP),
        );
    }
    let mut bottom = rows(main) as f32 * row_h + rows(main).saturating_sub(1) as f32 * GAP;
    let mut heading = None;
    if exp > 0 {
        let hy = bottom + GROUP_GAP;
        heading = Some(Rect {
            x: 0.0,
            y: hy,
            w: tw,
            h: HEADING_H,
        });
        let y0 = hy + HEADING_H + HEADING_GAP;
        for j in 0..exp {
            tiles[main + j] = at(
                (j % cols) as f32 * (tw + GAP),
                y0 + (j / cols) as f32 * (row_h + GAP),
            );
        }
        bottom = y0 + rows(exp) as f32 * row_h + (rows(exp) - 1) as f32 * GAP;
    }
    let used = cols.min(main.max(exp)).max(1);
    Placed {
        tiles,
        heading,
        size: (used as f32 * (tw + GAP) - GAP, bottom),
    }
}

/// Lays the tiles out for `surface` (a pure function of the surface size and DPI factor): one row on ultrawide
/// (aspect >= 2), three columns on landscape, two on portrait; fewer columns and then smaller tiles until the
/// block fits between the margins and above the control bar. Text keeps its size; only thumbnails shrink.
pub fn layout(surface: (u32, u32), dpi: f32) -> PanelLayout {
    let (o, main, n) = order();
    let g = controls::layout(surface, dpi);
    let s = g.scale;
    let (w, h) = (surface.0.max(1) as f32, surface.1.max(1) as f32);
    let aspect = w / h;
    let one_row = aspect >= 2.0;
    let want = if one_row {
        n.max(1)
    } else if aspect >= 1.0 {
        3
    } else {
        2
    };
    let avail_w = (w - 2.0 * MARGIN * s).max(1.0);
    let top = MARGIN * s;
    let bottom = (g.area.y - MARGIN * s).max(top + 1.0);
    let avail_h = bottom - top;
    let mut chosen = None;
    'fit: for f in FACTORS {
        for cols in (1..=want.min(n.max(1))).rev() {
            let (p, cap) = place(n, main, cols, f, one_row).with_caption();
            let fits = p.size.0 * s <= avail_w && p.size.1 * s <= avail_h;
            chosen = Some((p, cap, f, cols));
            if fits {
                break 'fit;
            }
            if one_row {
                break; // one row has no fewer-columns step
            }
        }
    }
    let (p, cap, factor, cols) = chosen.expect("a candidate");
    let ox = ((w - p.size.0 * s) * 0.5).max(0.0);
    let oy = top + ((avail_h - p.size.1 * s) * 0.5).max(0.0);
    let mv = |r: Rect| Rect {
        x: ox + r.x * s,
        y: oy + r.y * s,
        w: r.w * s,
        h: r.h * s,
    };
    let blank = Tile {
        kind: SceneKind::Spectrum,
        rect: Rect::default(),
        hit: Rect::default(),
        label: Rect::default(),
        experimental: false,
    };
    let mut tiles = [blank; MAX_TILES];
    for (i, t) in tiles.iter_mut().enumerate().take(n) {
        let r = p.tiles[i];
        *t = Tile {
            kind: o[i],
            rect: mv(r),
            hit: mv(Rect {
                h: r.h + LABEL_GAP + LABEL_H,
                ..r
            }),
            label: mv(Rect {
                y: r.y + r.h + LABEL_GAP,
                h: LABEL_H,
                ..r
            }),
            experimental: i >= main,
        };
    }
    PanelLayout {
        scale: s,
        count: n,
        tiles,
        heading: p.heading.map(mv),
        caption: mv(cap),
        cols,
        factor,
        bounds: mv(Rect {
            x: 0.0,
            y: 0.0,
            w: p.size.0,
            h: p.size.1,
        }),
    }
}

// ---- motion curves ------------------------------------------------------------------------------------------------

fn bezier_at(a: f32, b: f32, t: f32) -> f32 {
    3.0 * (1.0 - t).powi(2) * t * a + 3.0 * (1.0 - t) * t * t * b + t * t * t
}

/// CSS cubic-bezier(0.2, 0.8, 0.2, 1) evaluated at x = `p` (the open curve).
fn bezier_out(p: f32) -> f32 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if bezier_at(0.2, 0.2, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    bezier_at(0.8, 1.0, 0.5 * (lo + hi))
}

/// Inverse of `bezier_out` (the progress that shows value `v`).
fn bezier_out_inv(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    if v >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if bezier_out(mid) < v {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Progress that shows value `v` on the curve of the given direction.
fn inv_curve(open: bool, v: f32) -> f32 {
    if open { bezier_out_inv(v) } else { v.sqrt() }
}

fn step_to(v: &mut f32, target: f32, dt: f32, dur: f32) {
    let s = dt / dur;
    *v += (target - *v).clamp(-s, s);
}

// ---- the panel ------------------------------------------------------------------------------------------------------

pub struct ScenePanel {
    surface: (u32, u32),
    dpi: f32,
    lay: PanelLayout,
    /// Target state (false while closing).
    open: bool,
    reduced: bool,
    /// Progress 0..1 of the scrim and of each tile; the shown value depends on the direction (`shown`).
    scrim: f32,
    tile_p: [f32; MAX_TILES],
    since_open: f32,
    /// A click's countdown to closing.
    close_in: Option<f32>,
    focus: usize,
    /// Focus came from the keyboard: the ring is drawn (pointer use hides it).
    keyboard: bool,
    hover_tile: Option<usize>,
    hover: [f32; MAX_TILES],
    current: SceneKind,
    jobs: Vec<Job>,
    ops: Vec<GpuOp>,
    /// Label scale bucket in the atlas, and the one already requested from the worker.
    labels_have: Option<u32>,
    labels_asked: Option<u32>,
    /// (width, height, pad) of each rendered label bitmap, 0 = not there.
    label_dims: [(f32, f32, f32); LABEL_CELLS],
    has_thumb: [bool; MAX_TILES],
}

impl ScenePanel {
    pub fn new() -> Self {
        let surface = (1920, 1080);
        Self {
            surface,
            dpi: 1.0,
            lay: layout(surface, 1.0),
            open: false,
            reduced: false,
            scrim: 0.0,
            tile_p: [0.0; MAX_TILES],
            since_open: 0.0,
            close_in: None,
            focus: 0,
            keyboard: false,
            hover_tile: None,
            hover: [0.0; MAX_TILES],
            current: SceneKind::Spectrum,
            jobs: Vec::new(),
            ops: Vec::new(),
            labels_have: None,
            labels_asked: None,
            label_dims: [(0.0, 0.0, 0.0); LABEL_CELLS],
            has_thumb: [false; MAX_TILES],
        }
    }

    /// Current surface and DPI factor; recomputes the layout only when they changed.
    pub fn set_view(&mut self, surface: (u32, u32), dpi: f32) {
        if surface != self.surface || dpi != self.dpi {
            self.surface = surface;
            self.dpi = dpi;
            self.lay = layout(surface, dpi);
            if self.open || self.visible() {
                self.want_labels();
            }
        }
    }

    #[cfg(test)]
    pub fn layout(&self) -> &PanelLayout {
        &self.lay
    }

    /// Reduced motion: a plain 120 ms opacity fade, no rise, hover snaps.
    pub fn set_reduced(&mut self, on: bool) {
        self.reduced = on;
    }

    /// The scene on screen (or fading in): marks the "Current" tile and makes re-selecting it a no-op.
    pub fn set_current(&mut self, kind: SceneKind) {
        self.current = kind;
    }

    #[cfg(test)]
    pub fn current(&self) -> SceneKind {
        self.current
    }

    /// The panel wants to be open (the idle timer is held while this is true).
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Anything of the panel is still on screen (opening, open or closing).
    pub fn visible(&self) -> bool {
        self.scrim > 0.0 || self.tile_p.iter().any(|p| *p > 0.0)
    }

    fn index_of(&self, kind: SceneKind) -> Option<usize> {
        self.lay.tiles[..self.lay.count]
            .iter()
            .position(|t| t.kind == kind)
    }

    fn shown(&self, p: f32) -> f32 {
        if self.reduced {
            p
        } else if self.open {
            bezier_out(p)
        } else {
            p * p
        }
    }

    /// Flips the direction without a visual jump: progress is re-expressed on the new curve.
    fn retarget(&mut self, open: bool) {
        if open == self.open {
            return;
        }
        if self.reduced {
            self.open = open;
            return;
        }
        let (scrim, tiles) = (self.shown(self.scrim), self.tile_p.map(|p| self.shown(p)));
        self.open = open;
        self.scrim = inv_curve(open, scrim);
        self.tile_p = tiles.map(|v| inv_curve(open, v));
    }

    /// Opens the panel on the current scene. `keyboard`: focus came from the keyboard (the ring shows).
    pub fn open(&mut self, current: SceneKind, keyboard: bool) {
        self.current = current;
        self.focus = self.index_of(current).unwrap_or(0);
        self.keyboard = keyboard;
        self.close_in = None;
        if !self.open {
            self.since_open = 0.0;
        }
        self.want_labels();
        self.retarget(true);
    }

    pub fn close(&mut self) {
        self.close_in = None;
        self.retarget(false);
    }

    pub fn toggle(&mut self, current: SceneKind, keyboard: bool) {
        if self.open {
            self.close();
        } else {
            self.open(current, keyboard);
        }
    }

    /// `[` / `]` while open: the scene changed under the panel, so selection and focus follow it.
    pub fn follow(&mut self, kind: SceneKind) {
        self.current = kind;
        if let Some(i) = self.index_of(kind) {
            self.focus = i;
        }
    }

    // ---- input ---------------------------------------------------------------------------------------------------

    /// Selects tile `i`: re-selecting the current scene only closes; otherwise `Select` (the panel's `current`
    /// moves at once, so a rapid second choice is compared against the latest one: last wins). `now`: close at
    /// once (keyboard) or after `CLICK_CLOSE_S` (pointer, so the choice is seen).
    fn choose(&mut self, i: usize, now: bool) -> PanelEvent {
        let kind = self.lay.tiles[i].kind;
        self.focus = i;
        if kind == self.current {
            self.close();
            return PanelEvent::Close;
        }
        self.current = kind;
        if now {
            self.close();
        } else {
            self.close_in = Some(CLICK_CLOSE_S);
        }
        PanelEvent::Select(kind)
    }

    pub fn key(&mut self, key: PanelKey) -> PanelEvent {
        if !self.open || self.lay.count == 0 {
            return PanelEvent::None;
        }
        self.keyboard = true;
        let n = self.lay.count;
        match key {
            PanelKey::Escape => {
                self.close();
                return PanelEvent::Close;
            }
            PanelKey::Enter => return self.choose(self.focus, true),
            PanelKey::Tab | PanelKey::Right => self.focus = (self.focus + 1) % n,
            PanelKey::BackTab | PanelKey::Left => self.focus = (self.focus + n - 1) % n,
            PanelKey::Home => self.focus = 0,
            PanelKey::End => self.focus = n - 1,
            PanelKey::Up => self.focus = self.vertical(false),
            PanelKey::Down => self.focus = self.vertical(true),
        }
        PanelEvent::None
    }

    /// Nearest tile in the next row down / up (smallest horizontal offset); wraps to the opposite end row.
    fn vertical(&self, down: bool) -> usize {
        let t = &self.lay.tiles[..self.lay.count];
        let cur = t[self.focus].rect;
        let (cx, cy) = (cur.x + 0.5 * cur.w, cur.y + 0.5 * cur.h);
        let eps = 0.5 * cur.h;
        let sign = if down { 1.0 } else { -1.0 };
        let pick = |keep: &dyn Fn(f32) -> bool, rank: &dyn Fn(f32) -> f32| {
            t.iter()
                .enumerate()
                .filter(|(_, k)| keep(k.rect.y + 0.5 * k.rect.h))
                .min_by(|(_, a), (_, b)| {
                    let key = |k: &Tile| {
                        (
                            rank(k.rect.y + 0.5 * k.rect.h),
                            (k.rect.x + 0.5 * k.rect.w - cx).abs(),
                        )
                    };
                    key(a)
                        .partial_cmp(&key(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
        };
        pick(&|y| sign * (y - cy) > eps, &|y| sign * (y - cy))
            .or_else(|| pick(&|_| true, &|y| sign * y)) // wrap to the far row
            .unwrap_or(self.focus)
    }

    /// Tile under the pointer (thumbnail or its label).
    pub fn hit_test(&self, p: (f32, f32)) -> Option<usize> {
        self.lay.tiles[..self.lay.count]
            .iter()
            .position(|t| t.hit.contains(p))
    }

    pub fn pointer_moved(&mut self, p: (f32, f32)) {
        if !self.open {
            return;
        }
        self.hover_tile = self.hit_test(p);
        if let Some(i) = self.hover_tile {
            self.keyboard = false; // pointer use ends keyboard focus mode
            self.focus = i;
        }
    }

    pub fn pointer_left(&mut self) {
        self.hover_tile = None;
    }

    /// A left click at `p` that the control bar did not take: a tile selects, empty space closes. Consumed
    /// while the panel is open.
    pub fn click(&mut self, p: (f32, f32)) -> PanelEvent {
        if !self.open {
            return PanelEvent::None;
        }
        self.keyboard = false;
        match self.hit_test(p) {
            Some(i) => self.choose(i, false),
            None => {
                self.close();
                PanelEvent::Close
            }
        }
    }

    // ---- time ----------------------------------------------------------------------------------------------------

    pub fn advance(&mut self, dt: f32) {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.25)
        } else {
            0.0
        };
        if let Some(c) = self.close_in {
            if c <= dt {
                self.close();
            } else {
                self.close_in = Some(c - dt);
            }
        }
        for i in 0..MAX_TILES {
            let want = f32::from(u8::from(self.open && self.hover_tile == Some(i)));
            if self.reduced {
                self.hover[i] = want;
            } else {
                step_to(&mut self.hover[i], want, dt, HOVER_S);
            }
        }
        let (open_s, close_s, stagger) = if self.reduced {
            (REDUCED_S, REDUCED_S, 0.0)
        } else {
            (OPEN_S, CLOSE_S, STAGGER_S)
        };
        if self.open {
            self.since_open += dt;
            self.scrim = (self.scrim + dt / open_s).min(1.0);
            for (i, p) in self.tile_p.iter_mut().enumerate() {
                if i < self.lay.count && (*p > 0.0 || self.since_open >= i as f32 * stagger) {
                    *p = (*p + dt / open_s).min(1.0);
                }
            }
        } else {
            self.scrim = (self.scrim - dt / close_s).max(0.0);
            for p in &mut self.tile_p {
                *p = (*p - dt / close_s).max(0.0);
            }
            if !self.visible() {
                self.hover = [0.0; MAX_TILES];
                self.hover_tile = None;
                self.keyboard = false;
            }
        }
    }

    // ---- what to draw --------------------------------------------------------------------------------------------

    /// Scrim opacity multiplier 0..1.
    pub fn scrim_alpha(&self) -> f32 {
        self.shown(self.scrim)
    }

    /// Opacity of tile `i` 0..1.
    pub fn tile_alpha(&self, i: usize) -> f32 {
        self.shown(self.tile_p[i.min(MAX_TILES - 1)])
    }

    /// How far tile `i` still has to rise, in surface pixels (0 in reduced motion and once settled).
    pub fn tile_rise_px(&self, i: usize) -> f32 {
        if self.reduced {
            0.0
        } else {
            RISE_PX * self.lay.scale * (1.0 - self.tile_alpha(i))
        }
    }

    /// Hover amount 0..1 of tile `i` (0.14 s ease; snaps in reduced motion).
    #[cfg(test)]
    pub fn tile_hover(&self, i: usize) -> f32 {
        self.hover[i.min(MAX_TILES - 1)]
    }

    /// The keyboard focus ring's tile, if the ring is shown.
    pub fn ring(&self) -> Option<usize> {
        (self.open && self.keyboard).then_some(self.focus)
    }

    #[cfg(test)]
    pub fn focus(&self) -> usize {
        self.focus
    }
}

// ---- runtime: labels, thumbnails and the draw ----------------------------------------------------------------------

/// Largest UI scale labels are rasterised for (the label cells hold 256 x 48 px; above it labels stay at this size).
const LABEL_SCALE_MAX: f32 = 2.2;

fn label_key(scale: f32) -> u32 {
    (scale.clamp(0.1, LABEL_SCALE_MAX) * 20.0).round() as u32
}

/// Rasterises the panel texts at UI scale `scale` (worker thread): one bitmap per atlas label cell. The scene
/// labels are 15 px medium, "Current" 10 px, the pill 9 px and the heading 11 px caps with 0.08 em tracking.
/// A text that does not fit its cell, or no text renderer, leaves the cell empty (that label is simply absent).
pub fn render_labels(
    text: Option<&mg_text::TextRenderer>,
    scale: f32,
) -> [Option<TextBitmap>; LABEL_CELLS] {
    let mut out: [Option<TextBitmap>; LABEL_CELLS] = Default::default();
    let Some(r) = text else { return out };
    let s = label_key(scale) as f32 / 20.0;
    let line = |t: &str, px: f32, weight: u16, tracking_em: f32| {
        let px = px * s;
        let pad = (px * 0.3).ceil() as u32;
        r.render_line(
            t,
            &LineStyle {
                px,
                weight,
                max_width: LABEL_SIZE.0.saturating_sub(2 * pad).max(8),
                tracking_em,
                cut: Cut::Text,
                align: Align::Start,
                ..Default::default()
            },
        )
        .filter(|b| b.width <= LABEL_SIZE.0 && b.height <= LABEL_SIZE.1)
    };
    for k in SceneKind::ALL {
        out[k.index()] = line(k.label(), 15.0, 500, 0.0);
    }
    out[LABEL_CURRENT] = line("Current", 10.0, 600, 0.02);
    out[LABEL_EXPERIMENTAL_PILL] = line("EXPERIMENTAL", 9.0, 600, 0.08);
    out[LABEL_EXPERIMENTAL_HEADING] = line("EXPERIMENTAL", 11.0, 600, 0.08);
    out[LABEL_CAPTION] = line("Representative previews", 11.0, 500, 0.02);
    out
}

impl ScenePanel {
    /// Asks the worker for the labels at the current UI scale (once per scale bucket; the atlas cells are
    /// rewritten in place, nothing is created).
    fn want_labels(&mut self) {
        let key = label_key(self.lay.scale);
        if self.labels_have != Some(key) {
            // labels of another scale would be too big or too small for these tiles: show none until the
            // right ones arrive (a few frames), rather than mis-sized text
            self.label_dims = [(0.0, 0.0, 0.0); LABEL_CELLS];
            if self.labels_asked != Some(key) {
                self.labels_asked = Some(key);
                self.jobs.push(Job::PanelLabels {
                    scale: self.lay.scale,
                });
            }
        }
    }

    pub fn take_jobs(&mut self) -> Vec<Job> {
        std::mem::take(&mut self.jobs)
    }

    pub fn take_ops(&mut self) -> Vec<GpuOp> {
        std::mem::take(&mut self.ops)
    }

    /// A worker result for the panel (labels); other results are not ours.
    pub fn on_ready(&mut self, r: Ready) {
        let Ready::PanelLabels { scale, cells } = r else {
            return;
        };
        if label_key(scale) != label_key(self.lay.scale) {
            return; // rendered for a scale the window no longer has: a newer job is on its way
        }
        self.labels_have = Some(label_key(scale));
        for (d, c) in self.label_dims.iter_mut().zip(cells.iter()) {
            *d = c.as_ref().map_or((0.0, 0.0, 0.0), |b| {
                (b.width as f32, b.height as f32, b.pad as f32)
            });
        }
        self.ops.push(GpuOp::SetPanelLabels(cells));
    }

    /// The runtime thumbnails arrived (once, shortly after start): upload them and show them.
    pub fn on_thumbs(&mut self, t: Thumbs) {
        self.has_thumb = std::array::from_fn(|i| t[i].is_some());
        self.ops.push(GpuOp::SetThumbs(Box::new(t)));
    }

    /// The draw for this frame, or None while nothing of the panel is on screen (zero cost).
    pub fn frame(&self, accent: [f32; 3]) -> Option<PanelDraw> {
        let alpha = self.scrim_alpha();
        if !self.visible() || alpha <= 0.002 {
            return None;
        }
        let s = self.lay.scale;
        let dims = |cell: usize| self.label_dims[cell];
        let (cur, exp) = (dims(LABEL_CURRENT), dims(LABEL_EXPERIMENTAL_PILL));
        // a pill hugs the ink of its text: half size = half ink + padding (6 / 2.5 design px)
        let pill_half = |d: (f32, f32, f32)| {
            (
                0.5 * (d.0 - 2.0 * d.2).max(0.0) + 6.0 * s,
                0.5 * (d.1 - 2.0 * d.2).max(0.0) + 2.5 * s,
            )
        };
        let mut tiles = [PanelTileDraw::default(); PANEL_TILES];
        for (i, out) in tiles.iter_mut().enumerate().take(self.lay.count) {
            let t = &self.lay.tiles[i];
            let (hover, rise) = (self.hover[i], self.tile_rise_px(i));
            let k = 1.0 + 0.03 * hover;
            let (cx, cy) = (t.rect.x + 0.5 * t.rect.w, t.rect.y + 0.5 * t.rect.h + rise);
            let (hw, hh) = (0.5 * t.rect.w * k, 0.5 * t.rect.h * k);
            let selected = t.kind == self.current;
            let (lw, lh, _) = dims(t.kind.index());
            let label = if lw > 0.0 {
                [
                    (t.label.x + 0.5 * (t.label.w - lw)).round(),
                    (t.label.y + 0.5 * (t.label.h - lh) + rise).round(),
                    lw,
                    lh,
                ]
            } else {
                [0.0; 4]
            };
            let (left, right, top) = (cx - hw, cx + hw, cy - hh);
            let mut cur_pill = [0.0; 4];
            let mut exp_pill = [0.0; 4];
            if selected && cur.0 > 0.0 {
                let (pw, ph) = pill_half(cur);
                cur_pill = [left + 8.0 * s + pw, top + 8.0 * s + ph, pw, ph];
            }
            if t.experimental && exp.0 > 0.0 {
                let (pw, ph) = pill_half(exp);
                let p = [right - 8.0 * s - pw, top + 8.0 * s + ph, pw, ph];
                // a narrow tile shows only one pill: the selection ("Current") wins
                if cur_pill[2] == 0.0 || cur_pill[0] + cur_pill[2] + 4.0 * s <= p[0] - p[2] {
                    exp_pill = p;
                }
            }
            *out = PanelTileDraw {
                rect: [cx, cy, hw, hh],
                alpha: self.tile_alpha(i),
                hover,
                selected,
                focus: self.ring() == Some(i),
                experimental: t.experimental,
                has_thumb: self.has_thumb[t.kind.index()],
                cell: t.kind.index(),
                label,
                cur_pill,
                exp_pill,
            };
        }
        let first_exp = self.lay.tiles[..self.lay.count]
            .iter()
            .position(|t| t.experimental)
            .unwrap_or(0);
        let hd = dims(LABEL_EXPERIMENTAL_HEADING);
        let heading = match self.lay.heading {
            Some(h) if hd.0 > 0.0 => [
                (h.x - hd.2).round(),
                (h.y + 0.5 * (h.h - hd.1)).round(),
                hd.0,
                hd.1,
            ],
            _ => [0.0; 4],
        };
        let cp = dims(LABEL_CAPTION);
        let caption = if cp.0 > 0.0 {
            let c = self.lay.caption;
            [
                (c.x + 0.5 * (c.w - cp.0)).round(),
                (c.y + 0.5 * (c.h - cp.1)).round(),
                cp.0,
                cp.1,
            ]
        } else {
            [0.0; 4]
        };
        Some(PanelDraw {
            alpha,
            scale: s,
            accent,
            count: self.lay.count,
            tiles,
            heading,
            heading_alpha: self.tile_alpha(first_exp),
            caption,
            cur_text: [cur.0, cur.1],
            exp_text: [exp.0, exp.1],
        })
    }
}
impl Default for ScenePanel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_switch::SceneSwitch;
    use SceneKind::*;
    use mg_text::TextBitmap;

    const UW: (u32, u32) = (2560, 1080); // 21:9
    const HD: (u32, u32) = (1920, 1080); // 16:9
    const XGA: (u32, u32) = (1024, 768); // 4:3
    const PORTRAIT: (u32, u32) = (1080, 1920); // 9:16
    const UHD: (u32, u32) = (3840, 2160);

    fn open_panel(surface: (u32, u32), current: SceneKind) -> ScenePanel {
        let mut p = ScenePanel::new();
        p.set_view(surface, 1.0);
        p.open(current, true);
        p
    }

    fn settle(p: &mut ScenePanel) {
        for _ in 0..60 {
            p.advance(1.0 / 60.0);
        }
    }

    // ---- order and the experimental rule ---------------------------------------------------------------------------

    #[test]
    fn the_order_is_main_then_experimental_and_covers_every_scene_once() {
        let (o, main, n) = order();
        assert_eq!((main, n), (3, 5));
        assert_eq!(&o[..n], [Spectrum, Focus, Orb, Aurora, Fluid]);
        let l = layout(HD, 1.0);
        assert!(l.tiles[..3].iter().all(|t| !t.experimental));
        assert!(l.tiles[3..5].iter().all(|t| t.experimental));
    }

    #[test]
    fn experimental_tiles_are_chosen_by_enter_and_click_but_never_by_scene_keys() {
        let mut p = open_panel(HD, Spectrum);
        for _ in 0..3 {
            p.key(PanelKey::Tab);
        }
        assert_eq!(p.layout().tiles[p.focus()].kind, Aurora);
        assert_eq!(p.key(PanelKey::Enter), PanelEvent::Select(Aurora));
        let mut p = open_panel(HD, Spectrum);
        let t = p.layout().tiles[4];
        let c = (t.rect.x + 5.0, t.rect.y + 5.0);
        assert_eq!(p.click(c), PanelEvent::Select(Fluid));
        // the scene keys never reach them (SceneKind owns that rule)
        assert!(SceneKind::selector().all(|k| k != Aurora && k != Fluid));
    }

    // ---- layout ------------------------------------------------------------------------------------------------------

    fn assert_fits(surface: (u32, u32), l: &PanelLayout) {
        let (w, h) = (surface.0 as f32, surface.1 as f32);
        let bar_top = controls::layout(surface, 1.0).area.y;
        for (i, t) in l.tiles[..l.count].iter().enumerate() {
            for r in [t.rect, t.hit, t.label] {
                assert!(
                    r.x >= 0.0 && r.y >= 0.0 && r.x + r.w <= w && r.y + r.h <= h,
                    "{surface:?} tile {i} off screen: {r:?}"
                );
            }
            assert!(
                t.hit.y + t.hit.h <= bar_top,
                "{surface:?} tile {i} over the bar zone"
            );
            assert!(
                (t.rect.w / t.rect.h - 16.0 / 9.0).abs() < 1e-3,
                "tiles are 16:9"
            );
            for (j, u) in l.tiles[..l.count].iter().enumerate().skip(i + 1) {
                let apart = t.hit.x + t.hit.w <= u.hit.x + 0.01
                    || u.hit.x + u.hit.w <= t.hit.x + 0.01
                    || t.hit.y + t.hit.h <= u.hit.y + 0.01
                    || u.hit.y + u.hit.h <= t.hit.y + 0.01;
                assert!(apart, "{surface:?} tiles {i} and {j} overlap");
            }
        }
        let c = l.caption;
        assert!(
            c.x >= 0.0 && c.y >= 0.0 && c.x + c.w <= w && c.y + c.h <= bar_top,
            "{surface:?} caption"
        );
        if let Some(hd) = l.heading {
            assert!(hd.x >= 0.0 && hd.x + hd.w <= w && hd.y >= 0.0 && hd.y + hd.h <= h);
        }
        // text stays readable: tiles never shrink below 45 % of the design size, labels keep their size
        assert!(l.factor >= 0.45);
    }

    #[test]
    fn layout_fits_without_overlap_for_ultrawide_landscape_4_3_and_portrait() {
        for surface in [
            UW,
            HD,
            XGA,
            PORTRAIT,
            UHD,
            (1280, 720),
            (800, 600),
            (720, 1280),
        ] {
            assert_fits(surface, &layout(surface, 1.0));
        }
    }

    #[test]
    fn columns_follow_the_aspect_ratio() {
        let uw = layout(UW, 1.0);
        let ys: Vec<f32> = uw.tiles[..5].iter().map(|t| t.rect.y).collect();
        assert!(
            ys.iter().all(|y| (y - ys[0]).abs() < 0.01),
            "ultrawide: one row"
        );
        assert_eq!(uw.factor, 1.0);
        let hd = layout(HD, 1.0);
        assert_eq!(hd.cols, 3);
        assert!(
            hd.tiles[3].rect.y > hd.tiles[0].rect.y,
            "experimental below the main row"
        );
        assert_eq!(layout(XGA, 1.0).cols, 3);
        let p = layout(PORTRAIT, 1.0);
        assert!((1..=2).contains(&p.cols), "portrait columns {}", p.cols);
        assert_eq!(p.cols, 2);
        assert!(p.heading.is_some());
    }

    #[test]
    fn layout_is_a_pure_function_and_hostile_surfaces_stay_finite() {
        assert_eq!(layout(HD, 1.0), layout(HD, 1.0));
        for surface in [(0, 0), (1, 1), (320, 200), (100_000, 10)] {
            let l = layout(surface, f32::NAN);
            assert!(
                l.tiles[..l.count].iter().all(|t| {
                    [t.rect.x, t.rect.y, t.rect.w, t.rect.h]
                        .iter()
                        .all(|v| v.is_finite())
                }),
                "{surface:?}"
            );
        }
        // the layout scales with the UI scale (4K tiles are twice the 1080p size)
        let (a, b) = (layout(HD, 1.0), layout(UHD, 1.0));
        assert!((b.tiles[0].rect.w / a.tiles[0].rect.w - 2.0).abs() < 0.01);
        assert!((a.tiles[0].rect.w - 240.0).abs() < 0.01);
    }

    #[test]
    fn hit_test_covers_thumbnail_and_label_but_not_the_gaps() {
        let p = open_panel(HD, Spectrum);
        let t = p.layout().tiles[1];
        assert_eq!(p.hit_test((t.rect.x + 1.0, t.rect.y + 1.0)), Some(1));
        assert_eq!(p.hit_test((t.label.x + 5.0, t.label.y + 5.0)), Some(1));
        assert_eq!(
            p.hit_test((t.rect.x - 5.0, t.rect.y + 5.0)),
            None,
            "the gap"
        );
        assert_eq!(p.hit_test((5.0, 5.0)), None);
    }

    // ---- keys and focus ----------------------------------------------------------------------------------------------

    #[test]
    fn tab_and_arrows_move_focus_and_wrap_and_nothing_selects_until_enter() {
        let mut p = open_panel(HD, Spectrum);
        assert_eq!(p.focus(), 0);
        assert_eq!(p.key(PanelKey::BackTab), PanelEvent::None);
        assert_eq!(p.focus(), 4, "Shift+Tab wraps backwards");
        assert_eq!(p.key(PanelKey::Tab), PanelEvent::None);
        assert_eq!(p.focus(), 0, "Tab wraps forwards");
        p.key(PanelKey::End);
        assert_eq!(p.focus(), 4);
        p.key(PanelKey::Home);
        assert_eq!(p.focus(), 0);
        p.key(PanelKey::Right);
        assert_eq!(p.focus(), 1);
        p.key(PanelKey::Left);
        p.key(PanelKey::Left);
        assert_eq!(p.focus(), 4);
        assert_eq!(p.ring(), Some(4), "keyboard focus shows the ring");
    }

    #[test]
    fn vertical_arrows_cross_the_group_rows_by_nearest_column_and_wrap() {
        let mut p = open_panel(HD, Spectrum);
        p.key(PanelKey::Down); // Spectrum (col 0) -> Aurora (col 0, experimental row)
        assert_eq!(p.layout().tiles[p.focus()].kind, Aurora);
        p.key(PanelKey::Down); // no row below: wraps to the top row, same column
        assert_eq!(p.layout().tiles[p.focus()].kind, Spectrum);
        p.key(PanelKey::Up); // wraps to the bottom row
        assert_eq!(p.layout().tiles[p.focus()].kind, Aurora);
        p.key(PanelKey::Right); // Fluid (col 1)
        p.key(PanelKey::Up);
        assert_eq!(p.layout().tiles[p.focus()].kind, Focus, "same column");
        // ultrawide: one row, so vertical keys stay put
        let mut u = open_panel(UW, Spectrum);
        u.key(PanelKey::Right);
        u.key(PanelKey::Down);
        assert_eq!(u.focus(), 1);
        u.key(PanelKey::Up);
        assert_eq!(u.focus(), 1);
    }

    #[test]
    fn enter_selects_and_closes_and_reselecting_the_current_scene_only_closes() {
        let mut p = open_panel(HD, Focus);
        assert_eq!(p.focus(), 1, "opens focused on the current scene");
        assert_eq!(
            p.key(PanelKey::Enter),
            PanelEvent::Close,
            "current scene: no-op close"
        );
        assert!(!p.is_open());
        let mut p = open_panel(HD, Focus);
        p.key(PanelKey::Right);
        assert_eq!(p.key(PanelKey::Enter), PanelEvent::Select(Orb));
        assert!(!p.is_open(), "Enter closes at once");
        assert_eq!(p.current(), Orb);
    }

    #[test]
    fn escape_closes_only_the_panel_and_a_closed_panel_ignores_keys() {
        let mut p = open_panel(HD, Spectrum);
        assert_eq!(p.key(PanelKey::Escape), PanelEvent::Close);
        assert!(!p.is_open());
        for k in [
            PanelKey::Escape,
            PanelKey::Enter,
            PanelKey::Tab,
            PanelKey::Down,
        ] {
            assert_eq!(
                p.key(k),
                PanelEvent::None,
                "{k:?} while closed falls through to the app"
            );
        }
    }

    #[test]
    fn the_scene_keys_move_the_selection_and_keep_the_panel_open() {
        let mut p = open_panel(HD, Spectrum);
        p.follow(Focus);
        assert!(p.is_open());
        assert_eq!(
            (p.current(), p.layout().tiles[p.focus()].kind),
            (Focus, Focus)
        );
        p.follow(Orb);
        assert_eq!(p.layout().tiles[p.focus()].kind, Orb);
    }

    // ---- pointer -----------------------------------------------------------------------------------------------------

    #[test]
    fn hover_moves_focus_without_a_ring_and_clicks_select_then_close_after_120_ms() {
        let mut p = open_panel(HD, Spectrum);
        settle(&mut p);
        let t = p.layout().tiles[2];
        p.pointer_moved((t.rect.x + 4.0, t.rect.y + 4.0));
        assert_eq!(p.focus(), 2);
        assert_eq!(p.ring(), None, "pointer mode: no ring");
        assert_eq!(
            p.click((t.rect.x + 4.0, t.rect.y + 4.0)),
            PanelEvent::Select(Orb)
        );
        assert!(p.is_open(), "the choice stays visible for a moment");
        for _ in 0..6 {
            p.advance(0.016); // 96 ms
        }
        assert!(p.is_open());
        for _ in 0..3 {
            p.advance(0.016);
        }
        assert!(!p.is_open(), "closed after ~120 ms");
    }

    #[test]
    fn a_click_on_empty_space_closes_and_never_selects() {
        let mut p = open_panel(HD, Spectrum);
        assert_eq!(p.click((3.0, 3.0)), PanelEvent::Close);
        assert!(!p.is_open());
        assert_eq!(
            p.click((3.0, 3.0)),
            PanelEvent::None,
            "closed: not consumed"
        );
    }

    #[test]
    fn rapid_selection_last_wins_through_the_scene_switch() {
        let mut p = open_panel(HD, Spectrum);
        let mut sw = SceneSwitch::new(Spectrum);
        let centre = |p: &ScenePanel, i: usize| {
            let r = p.layout().tiles[i].rect;
            (r.x + 0.5 * r.w, r.y + 0.5 * r.h)
        };
        let mut events = 0;
        for i in [1, 2, 3, 4, 2, 1, 4, 3] {
            let c = centre(&p, i);
            if let PanelEvent::Select(k) = p.click(c) {
                sw.request(k);
                events += 1;
            }
            p.advance(0.01);
        }
        assert_eq!(events, 8, "every distinct choice is a selection");
        assert_eq!(sw.target(), Aurora, "the last choice wins");
        // choosing the scene that is already the destination again does not restart anything
        let c = centre(&p, 3);
        assert_eq!(p.click(c), PanelEvent::Close);
    }

    // ---- motion ------------------------------------------------------------------------------------------------------

    #[test]
    fn opening_staggers_the_tiles_and_finishes_within_220_ms_plus_the_stagger() {
        let mut p = open_panel(HD, Spectrum);
        assert_eq!(p.scrim_alpha(), 0.0);
        p.advance(0.1);
        assert!(p.scrim_alpha() > 0.5, "the open curve is front loaded");
        assert!(p.tile_alpha(0) > p.tile_alpha(4), "tiles start 30 ms apart");
        assert!(p.tile_rise_px(4) > p.tile_rise_px(0));
        for _ in 0..16 {
            p.advance(0.016);
        }
        assert_eq!(p.scrim_alpha(), 1.0);
        assert!((0..5).all(|i| p.tile_alpha(i) == 1.0 && p.tile_rise_px(i) == 0.0));
    }

    #[test]
    fn closing_takes_150_ms_without_stagger_and_leaves_nothing_visible() {
        let mut p = open_panel(HD, Spectrum);
        settle(&mut p);
        p.close();
        p.advance(0.075);
        let a: Vec<f32> = (0..5).map(|i| p.tile_alpha(i)).collect();
        assert!(
            a.iter().all(|v| (v - a[0]).abs() < 1e-5),
            "no stagger on close: {a:?}"
        );
        assert!(p.visible());
        p.advance(0.1);
        assert!(!p.visible() && p.scrim_alpha() == 0.0);
    }

    #[test]
    fn reversing_mid_fade_does_not_jump() {
        let mut p = open_panel(HD, Spectrum);
        p.advance(0.06);
        let (s, t) = (p.scrim_alpha(), p.tile_alpha(0));
        p.close();
        assert!((p.scrim_alpha() - s).abs() < 1e-3 && (p.tile_alpha(0) - t).abs() < 1e-3);
        p.advance(0.03);
        let (s2, t2) = (p.scrim_alpha(), p.tile_alpha(0));
        p.open(Spectrum, false);
        assert!((p.scrim_alpha() - s2).abs() < 1e-3 && (p.tile_alpha(0) - t2).abs() < 1e-3);
    }

    #[test]
    fn reduced_motion_is_a_plain_120_ms_fade_with_no_rise_and_snapping_hover() {
        let mut p = ScenePanel::new();
        p.set_reduced(true);
        p.open(Spectrum, false);
        p.advance(0.06);
        assert!((p.scrim_alpha() - 0.5).abs() < 0.02, "linear");
        assert!((0..5).all(|i| (p.tile_alpha(i) - 0.5).abs() < 0.02 && p.tile_rise_px(i) == 0.0));
        p.advance(0.07);
        assert_eq!(p.scrim_alpha(), 1.0);
        let t = p.layout().tiles[1];
        p.pointer_moved((t.rect.x + 2.0, t.rect.y + 2.0));
        p.advance(0.001);
        assert_eq!(p.tile_hover(1), 1.0, "hover snaps");
        p.close();
        p.advance(0.13);
        assert!(!p.visible());
    }

    #[test]
    fn hover_eases_in_over_140_ms() {
        let mut p = open_panel(HD, Spectrum);
        settle(&mut p);
        let t = p.layout().tiles[0];
        p.pointer_moved((t.rect.x + 2.0, t.rect.y + 2.0));
        p.advance(0.07);
        assert!((p.tile_hover(0) - 0.5).abs() < 0.05);
        p.advance(0.1);
        assert_eq!(p.tile_hover(0), 1.0);
        p.pointer_left();
        p.advance(0.2);
        assert_eq!(p.tile_hover(0), 0.0);
    }

    #[test]
    fn hostile_time_steps_cannot_break_the_motion() {
        let mut p = open_panel(HD, Spectrum);
        for dt in [f32::NAN, f32::INFINITY, -1.0, 0.0, 100.0] {
            p.advance(dt);
            assert!(p.scrim_alpha().is_finite() && (0.0..=1.0).contains(&p.scrim_alpha()));
        }
    }

    // ---- runtime: labels, thumbnails, the draw and resource growth -------------------------------------------------

    fn fake_labels() -> Ready {
        let cells: [Option<TextBitmap>; LABEL_CELLS] = std::array::from_fn(|i| {
            Some(TextBitmap {
                width: 60 + i as u32,
                height: 24,
                pad: 4,
                coverage: vec![255; (60 + i) * 24],
            })
        });
        Ready::PanelLabels {
            scale: 1.0,
            cells: Box::new(cells),
        }
    }

    fn all_thumbs() -> Thumbs {
        std::array::from_fn(|_| Some(vec![1; 4]))
    }

    #[test]
    fn labels_are_requested_once_per_scale_and_uploaded_without_creating_anything_new() {
        let mut p = ScenePanel::new();
        p.set_view(HD, 1.0);
        assert!(
            p.take_jobs().is_empty(),
            "nothing is rendered until the panel is first opened"
        );
        p.open(Spectrum, false);
        let jobs = p.take_jobs();
        assert!(matches!(jobs[..], [Job::PanelLabels { .. }]));
        p.close();
        p.open(Spectrum, false);
        assert!(p.take_jobs().is_empty(), "already asked for this scale");
        p.on_ready(fake_labels());
        assert!(matches!(p.take_ops()[..], [GpuOp::SetPanelLabels(_)]));
        // a resize to another scale re-rasterises once (the same atlas cells are rewritten)
        p.set_view(UHD, 1.0);
        assert!(matches!(p.take_jobs()[..], [Job::PanelLabels { .. }]));
        p.set_view(UHD, 1.0);
        assert!(p.take_jobs().is_empty());
    }

    #[test]
    fn opening_and_closing_many_times_creates_no_work_after_the_first_open() {
        let mut p = ScenePanel::new();
        p.set_view(HD, 1.0);
        p.on_thumbs(all_thumbs());
        p.open(Spectrum, false);
        p.on_ready(fake_labels());
        let first = p.take_ops().len() + p.take_jobs().len();
        assert_eq!(first, 3, "thumbnails, labels, the one label job");
        for i in 0..1000 {
            p.toggle(Spectrum, i % 2 == 0);
            for _ in 0..30 {
                p.advance(1.0 / 60.0);
                let _ = p.frame([0.5; 3]);
            }
            assert!(
                p.take_jobs().is_empty() && p.take_ops().is_empty(),
                "cycle {i}"
            );
        }
    }

    #[test]
    fn the_draw_marks_the_current_scene_and_the_experimental_tiles_and_stays_inside_the_surface() {
        let mut p = ScenePanel::new();
        p.set_view(HD, 1.0);
        p.on_thumbs(all_thumbs());
        p.open(Focus, true);
        p.on_ready(fake_labels());
        assert!(p.frame([0.5; 3]).is_some_and(|d| d.alpha == 0.0) || p.frame([0.5; 3]).is_none());
        for _ in 0..40 {
            p.advance(1.0 / 60.0);
        }
        let d = p.frame([0.5; 3]).expect("open");
        assert_eq!((d.count, d.alpha), (5, 1.0));
        let l = *p.layout();
        for (i, t) in d.tiles[..5].iter().enumerate() {
            assert_eq!(t.cell, l.tiles[i].kind.index());
            assert_eq!(
                t.selected,
                l.tiles[i].kind == Focus,
                "only the current scene is marked"
            );
            assert_eq!(t.experimental, matches!(l.tiles[i].kind, Aurora | Fluid));
            assert!(t.has_thumb && t.alpha == 1.0);
            assert_eq!(t.focus, i == p.focus(), "keyboard focus draws one ring");
            let [cx, cy, hw, hh] = t.rect;
            assert!(cx - hw >= 0.0 && cx + hw <= 1920.0 && cy - hh >= 0.0 && cy + hh <= 1080.0);
            assert!(t.label[2] > 0.0, "labels present once rendered");
            assert_eq!(
                t.cur_pill[2] > 0.0,
                t.selected,
                "the Current pill is not colour alone"
            );
            assert_eq!(t.exp_pill[2] > 0.0, t.experimental);
        }
        assert!(d.heading[2] > 0.0 && d.heading_alpha == 1.0);
        assert!(
            d.caption[2] > 0.0,
            "the Representative previews caption is drawn as text"
        );
    }

    #[test]
    fn a_missing_preview_and_missing_labels_degrade_cleanly() {
        let mut p = ScenePanel::new();
        p.set_view(HD, 1.0);
        let mut t = all_thumbs();
        t[Orb.index()] = None;
        p.on_thumbs(t);
        p.open(Spectrum, false);
        for _ in 0..40 {
            p.advance(1.0 / 60.0);
        }
        let d = p
            .frame([0.5; 3])
            .expect("drawn even before the labels exist");
        let orb = d.tiles[..5].iter().find(|t| t.cell == Orb.index()).unwrap();
        assert!(!orb.has_thumb && orb.label[2] == 0.0 && orb.alpha == 1.0);
        assert!(d.tiles[..5].iter().filter(|t| t.has_thumb).count() == 4);
        assert!(p.frame([0.5; 3]).is_some_and(|d| d.heading[2] == 0.0));
    }

    #[test]
    fn hover_scales_the_tile_and_the_rise_shifts_it_and_a_closed_panel_draws_nothing() {
        let mut p = ScenePanel::new();
        p.set_view(HD, 1.0);
        assert!(p.frame([0.5; 3]).is_none(), "closed: zero cost");
        p.open(Spectrum, false);
        p.advance(0.05);
        let d = p.frame([0.5; 3]).unwrap();
        let base = p.layout().tiles[0].rect;
        assert!(d.tiles[0].rect[1] > base.y + 0.5 * base.h, "still rising");
        for _ in 0..40 {
            p.advance(1.0 / 60.0);
        }
        let t = p.layout().tiles[0];
        p.pointer_moved((t.rect.x + 3.0, t.rect.y + 3.0));
        for _ in 0..20 {
            p.advance(1.0 / 60.0);
        }
        let d = p.frame([0.5; 3]).unwrap();
        assert!(
            (d.tiles[0].rect[2] / (0.5 * t.rect.w) - 1.03).abs() < 1e-3,
            "hover scale 1.03"
        );
        p.close();
        for _ in 0..20 {
            p.advance(1.0 / 60.0);
        }
        assert!(p.frame([0.5; 3]).is_none());
    }

    fn labels_at(scale: f32) -> Ready {
        let Ready::PanelLabels { cells, .. } = fake_labels() else {
            unreachable!()
        };
        Ready::PanelLabels { scale, cells }
    }

    #[test]
    fn a_scale_change_never_shows_mis_sized_labels_and_stale_results_are_dropped() {
        let mut p = ScenePanel::new();
        p.set_view(HD, 1.0);
        p.open(Spectrum, false);
        p.take_jobs();
        p.on_ready(labels_at(1.0));
        for _ in 0..40 {
            p.advance(1.0 / 60.0);
        }
        assert!(p.frame([0.5; 3]).unwrap().tiles[0].label[2] > 0.0);
        // open panel, DPI / window change: the labels are hidden (not wrong-sized) and new ones requested
        p.set_view(UHD, 1.0);
        assert!(matches!(p.take_jobs()[..], [Job::PanelLabels { .. }]));
        let d = p.frame([0.5; 3]).unwrap();
        assert!(
            d.tiles[..5].iter().all(|t| t.label[2] == 0.0),
            "no stale labels"
        );
        assert_eq!(d.heading[2], 0.0);
        // the tiles stay usable meanwhile: hit test and selection use the layout, not the labels
        let t = p.layout().tiles[2];
        assert_eq!(
            p.click((t.label.x + 5.0, t.label.y + 3.0)),
            PanelEvent::Select(Orb)
        );
        // a result for the old scale arriving late is dropped; the right one restores the labels
        p.take_ops();
        p.on_ready(labels_at(1.0));
        assert!(
            p.take_ops()
                .iter()
                .all(|o| !matches!(o, GpuOp::SetPanelLabels(_)))
        );
        assert!(p.frame([0.5; 3]).unwrap().tiles[0].label[2] == 0.0);
        p.on_ready(labels_at(2.0));
        assert!(p.frame([0.5; 3]).unwrap().tiles[0].label[2] > 0.0);
        // closed panel: the change is picked up when it opens
        p.close();
        for _ in 0..20 {
            p.advance(1.0 / 60.0);
        }
        p.set_view(HD, 1.0);
        assert!(p.take_jobs().is_empty(), "closed: nothing asked yet");
        p.open(Spectrum, false);
        assert!(matches!(p.take_jobs()[..], [Job::PanelLabels { .. }]));
    }

    #[test]
    fn the_selector_works_without_any_thumbnail_or_label() {
        // injected failure: generation produced nothing (e.g. an error-scoped GPU failure) and no text renderer
        let mut p = ScenePanel::new();
        p.set_view(HD, 1.0);
        p.on_thumbs(Thumbs::default());
        p.open(Spectrum, true);
        for _ in 0..40 {
            p.advance(1.0 / 60.0);
        }
        let d = p.frame([0.5; 3]).expect("drawn");
        assert!(
            d.tiles[..5]
                .iter()
                .all(|t| !t.has_thumb && t.label[2] == 0.0 && t.alpha == 1.0)
        );
        // navigate, select, escape: all still work
        p.key(PanelKey::Tab);
        p.key(PanelKey::Down);
        assert_eq!(p.key(PanelKey::Enter), PanelEvent::Select(Fluid));
        p.open(Fluid, true);
        assert_eq!(p.key(PanelKey::Escape), PanelEvent::Close);
        let t = p.layout().tiles[0];
        p.open(Fluid, false);
        assert_eq!(
            p.click((t.rect.x + 3.0, t.rect.y + 3.0)),
            PanelEvent::Select(Spectrum)
        );
    }
}
