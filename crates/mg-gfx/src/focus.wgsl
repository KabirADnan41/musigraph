// Scene 3 "Album Focus": a room lit by the album art. The artwork itself (and the shadows under it) is drawn by the
// overlay pass; this shader draws only the room around it. It reads the shared prelude (common.wgsl) for the
// resolution, palette and mode, and its OWN block at @group(1) (`FU`, packed by scene::FocusParams; `fa`, the cover
// colours, uploaded only when a cover changes). Nothing here reads ex0..ex5, so no other scene's fields can leak in.
//
// Two passes. `fs_room` renders everything smooth into a small texture (480x270, RGBA16F; ~1/16 of the pixels of a
// 1080p frame): L0 an OKLab vertical gradient, L1 the light spilling from the cover rectangle (nearest point q of the
// rectangle, falloff (1+(d/r)^2)^-1.1, edge colours from the 8x8 grid, a broad lobe from the mean) and L2 domain-warped
// fbm haze that only exists where L1 lights it (OKLab-capped). The haze and the spill have no detail finer than
// ~0.15 H, so this is the spec's half-resolution haze tier, taken one step further after measuring the GT 730.
// `fs` then adds, at full resolution, L3 ~147 dust motes lit by that room light and glinting with the music, the
// vignette and three caps that only ever lower the result: the art stays the brightest thing near itself, a feathered
// contrast pocket under the text ink, and the floor dim while the control bar shows.
// Nothing is time-driven: motion phases are integrated on the CPU from the music (silence has a tiny floor).

struct FU {
    rect: vec4<f32>,                  // cover centre x, y, half width, half height (screen heights, top-left origin)
    a: vec4<f32>,                     // spill I, radius k, weight of the current cover, weight of the previous cover
    b: vec4<f32>,                     // haze flow phase, dust twinkle clock (cycles), warp W, twinkle depth 0..1
    c: vec4<f32>,                     // haze hue offset (rad), control-bar visibility, -, 1 / dust cell (cells of ~0.11 H at 16:9: ~147 motes at any aspect)
    ink: array<vec4<f32>, 4>,         // text ink rectangles x0, y0, x1, y1
    ink_cap: vec4<f32>,               // max relative luminance of the environment under each rectangle
    ink_on: vec4<f32>,                // 0..1 strength per rectangle (follows the text's own alpha)
};
@group(1) @binding(0) var<uniform> fu: FU;
// two cover slots (current, previous) x (64 grid cells row-major + 1 mean/meanY), linear RGB
@group(1) @binding(1) var<uniform> fa: array<vec4<f32>, 130>;
// the room light of `fs_room` (a 1x1 stand-in while that pass itself draws)
@group(1) @binding(2) var room_tex: texture_2d<f32>;
@group(1) @binding(3) var room_smp: sampler;

const ROOM_GAIN: f32 = 0.5;       // overall light of the spill (tuned on previews: the art stays the anchor)
const HAZE_L: f32 = 0.22;         // OKLab lightness cap of the additive haze
const POCKET_FEATHER: f32 = 0.06; // screen heights
const ROOM_W: f32 = 480.0;        // size of the room texture (scene.rs ROOM_SIZE)
const ROOM_H: f32 = 270.0;

fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

fn to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let l_ = pow(max(l, 0.0), 0.3333333);
    let m_ = pow(max(m, 0.0), 0.3333333);
    let s_ = pow(max(s, 0.0), 0.3333333);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_);
}

fn from_oklab(c: vec3<f32>) -> vec3<f32> {
    let l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    let m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    let s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return max(vec3<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s), vec3<f32>(0.0));
}

fn art_cell(slot: u32, ix: i32, iy: i32) -> vec3<f32> {
    return fa[slot * 65u + u32(clamp(iy, 0, 7) * 8 + clamp(ix, 0, 7))].rgb;
}

// Bilinear over the 8x8 edge-colour grid (cell centres at (i + 0.5) / 8), uv 0..1 across the cover.
fn art_edge(slot: u32, uv: vec2<f32>) -> vec3<f32> {
    let g = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * 8.0 - vec2<f32>(0.5);
    let i = floor(g);
    let f = g - i;
    let ix = i32(i.x);
    let iy = i32(i.y);
    let top = mix(art_cell(slot, ix, iy), art_cell(slot, ix + 1, iy), f.x);
    let bot = mix(art_cell(slot, ix, iy + 1), art_cell(slot, ix + 1, iy + 1), f.x);
    return mix(top, bot, f.y);
}

fn falloff(d: f32, r: f32) -> f32 { return pow(1.0 + sq(d / r), -1.1); }

// The light source rectangle: the cover, shrinking to a point at the empty slot as the cover goes away.
fn art_weight() -> f32 {
    let wc = clamp(fu.a.z, 0.0, 1.0);
    return wc + clamp(fu.a.w, 0.0, 1.0 - wc);
}

fn source_half(art_w: f32) -> vec2<f32> {
    return max(fu.rect.zw, vec2<f32>(0.02)) * art_w;
}

// L1: light spilling from the cover rectangle (a soft point source at the empty slot without a cover).
fn spill_at(p: vec2<f32>) -> vec3<f32> {
    let wc = clamp(fu.a.z, 0.0, 1.0);
    let wp = clamp(fu.a.w, 0.0, 1.0 - wc);
    let rest = max(1.0 - wc - wp, 0.0);
    let art_w = wc + wp;
    let he = source_half(art_w);
    let q = clamp(p, fu.rect.xy - he, fu.rect.xy + he);
    let d = length(p - q);
    let uv = (q - fu.rect.xy) / max(2.0 * he, vec2<f32>(0.0001)) + vec2<f32>(0.5);
    let pal_edge = mix(u.pal3.rgb, u.pal1.rgb, 0.25);
    let edge = wc * art_edge(0u, uv) + wp * art_edge(1u, uv) + rest * pal_edge;
    let avg = wc * fa[64].rgb + wp * fa[129].rgb + rest * u.pal1.rgb;
    let k = max(fu.a.y, 0.5);
    let core_r = 0.16 * k * mix(2.4, 1.0, art_w);
    let gain = ROOM_GAIN * fu.a.x * mix(0.45, 1.0, art_w);
    return gain * (0.55 * edge * falloff(d, core_r) + 0.35 * avg * falloff(d, 0.55 * k));
}

// Everything smooth: L0 gradient + L1 spill + L2 haze, at one point of the room (screen heights).
fn room_at(p: vec2<f32>) -> vec3<f32> {
    // L0: OKLab vertical gradient (hue from the palette), never black
    let pal0 = to_oklab(u.pal0.rgb);
    let chroma0 = clamp(length(pal0.yz) * 0.7, 0.012, 0.05);
    let hue0 = pal0.yz / max(length(pal0.yz), 0.0001);
    let l0 = mix(0.075, 0.15, smoothstep(0.0, 1.0, p.y));
    var col = from_oklab(vec3<f32>(l0, hue0 * chroma0));
    let spill = spill_at(p);
    let sl = lum(spill);
    // L2: domain-warped fbm haze, only where the spill lights it (flow phase from the CPU, never time)
    let fq = p * 1.4;
    let ph = fu.b.x;
    let w1 = fbm(fq + ph * vec2<f32>(1.0, 0.6));
    let w2 = fbm(fq + 3.5 * vec2<f32>(w1, 0.7 - w1) + ph * 1.3 * vec2<f32>(-0.6, 0.9));
    let ha = fbm(fq + fu.b.z * vec2<f32>(w2, 0.7 * w2 + 0.15));
    let shape = smoothstep(0.30, 0.72, ha);
    let slab = to_oklab(spill);
    let sdir = slab.yz / max(length(slab.yz), 0.0001);
    let cs = cos(fu.c.x);
    let sn = sin(fu.c.x);
    let rot = vec2<f32>(sdir.x * cs - sdir.y * sn, sdir.x * sn + sdir.y * cs);
    let haze = from_oklab(vec3<f32>(HAZE_L * shape, rot * 0.06 * shape)) * (0.25 + 0.75 * clamp(sl, 0.0, 1.0));
    col += spill * (0.82 + 0.36 * shape) + haze;
    return col;
}

@fragment
fn fs_room(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let aspect = u.res.x / u.res.y;
    let uv = frag.xy / vec2<f32>(ROOM_W, ROOM_H);
    let p = vec2<f32>(uv.x * aspect, uv.y);
    var col = room_at(p);
    // a luminance knee: a white cover lights the room, it does not flood it (and the text pocket stays a cap, not a scrim)
    col /= 1.0 + lum(col) / 0.2;
    // vignette
    let vc = (uv - vec2<f32>(0.5)) * vec2<f32>(1.0, 1.1);
    col *= 1.0 - 0.34 * smoothstep(0.25, 0.85, length(vc) * 1.25);
    // control bar: the floor dims at the bar's own alpha
    col *= 1.0 - 0.2 * clamp(fu.c.y, 0.0, 1.0) * smoothstep(0.78, 0.82, p.y);
    // alpha: how strongly the art-brightest cap applies here (within ~0.15 H of the cover, scaled by its weight)
    let art_w = art_weight();
    let he = source_half(art_w);
    let d = length(p - clamp(p, fu.rect.xy - he, fu.rect.xy + he));
    return vec4<f32>(col, (1.0 - smoothstep(0.12, 0.20, d)) * art_w);
}

// L3: one mote per cell (a few cells are empty), drifting with the haze flow. A mote never reaches past its own cell
// (offset 0.15..0.85, radius <= 0.10 cells), so only the pixel's cell is evaluated. `clk` is the twinkle clock,
// `depth` the music's twinkle depth; every mote has its own phase, rate and gain, so they never glint in unison.
fn dust(p: vec2<f32>, drift: vec2<f32>, inv_cell: f32, clk: f32, depth: f32) -> f32 {
    let g = (p + drift) * inv_cell;
    let id = floor(g);
    // two hashes, the rest derived (a cell's traits only need to look unrelated)
    let h1 = hash21(id + 3.1);
    let h2 = hash21(id * 1.3 + 7.7);
    let h3 = fract(h1 * 37.7 + h2 * 11.3);
    let h4 = fract(h2 * 29.1 + h1 * 5.7);
    let h5 = fract(h1 * 13.9 + h2 * 23.3);
    let pos = id + vec2<f32>(0.15) + 0.7 * vec2<f32>(h2, h3);
    let r = min((0.0032 + 0.0048 * h4 * h4) * inv_cell, 0.10);
    let dv = g - pos;
    let d2 = dot(dv, dv);
    if (d2 > 6.25 * r * r || h1 < 0.08) {
        return 0.0;
    }
    let glint = 1.0 + 0.4 * depth * h5 * (0.5 + 0.5 * sin(6.2831853 * (clk * (0.6 + 0.8 * h3) + h1)));
    return exp(-1.7 * d2 / (r * r)) * (0.35 + 0.65 * h4) * glint;
}

// One text ink rectangle: inside it (and over a 0.06 H feather around it) the environment is capped to `cap`.
fn pocket(col: vec3<f32>, p: vec2<f32>, r: vec4<f32>, on: f32, cap: f32) -> vec3<f32> {
    if (on > 0.001 && all(p > r.xy - vec2<f32>(POCKET_FEATHER)) && all(p < r.zw + vec2<f32>(POCKET_FEATHER))) {
        let sd = sd_round_box(p - 0.5 * (r.xy + r.zw), 0.5 * (r.zw - r.xy), 0.012);
        let m = (1.0 - smoothstep(0.0, POCKET_FEATHER, sd)) * on;
        return col * mix(1.0, min(1.0, cap / max(lum(col), 0.0001)), m);
    }
    return col;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let aspect = u.res.x / u.res.y;
    let p = frag.xy / u.res.y;                  // screen heights, origin top-left
    let uvs = frag.xy / u.res;

    let room = textureSampleLevel(room_tex, room_smp, uvs, 0.0);
    var col = room.rgb;
    let sl = lum(col);

    // L3: dust, lit by the room light (nothing glints in the dark)
    if (sl > 0.012) {
        let motes = dust(p, fu.b.x * vec2<f32>(0.5, -0.35), fu.c.w, fu.b.y, fu.b.w);
        col += motes * smoothstep(0.012, 0.08, sl) * (col * 2.2 + vec3<f32>(0.03) * sl);
    }

    col = col / (1.0 + 0.25 * col);

    // ---- caps: they only ever lower the result ---------------------------------------------------------------------
    // the art stays the brightest thing near itself: Y <= 0.6 x its mean Y (floor 0.04) within ~0.15 H (the room
    // pass stored the weight of this cap in alpha)
    if (room.a > 0.001) {
        let wc = clamp(fu.a.z, 0.0, 1.0);
        let wp = clamp(fu.a.w, 0.0, 1.0 - wc);
        let ycap = max(0.6 * (wc * fa[64].w + wp * fa[129].w) / max(wc + wp, 0.0001), 0.04);
        col *= mix(1.0, min(1.0, ycap / max(lum(col), 0.0001)), room.a);
    }
    // contrast pocket: under the text ink the environment is capped (a cap, not a scrim)
    col = pocket(col, p, fu.ink[0], fu.ink_on.x, fu.ink_cap.x);
    col = pocket(col, p, fu.ink[1], fu.ink_on.y, fu.ink_cap.y);
    col = pocket(col, p, fu.ink[2], fu.ink_on.z, fu.ink_cap.z);
    col = pocket(col, p, fu.ink[3], fu.ink_on.w, fu.ink_cap.w);

    col = mode_marker(col, frag, aspect, 0.92);
    col += (hash21(frag.xy) - 0.5) / 255.0;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
