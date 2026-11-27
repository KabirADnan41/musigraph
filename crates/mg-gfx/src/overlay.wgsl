// Overlay pass drawn AFTER the scene, at native surface resolution: album artwork on a plate (rounded corners,
// 1 px inner stroke, contact + ambient shadow; never scaled, pulsed or distorted by the music), an optional
// previous artwork for the track-change crossfade, and the song text (coverage bitmaps tinted per role from the
// album palette, light shadow). Output is premultiplied alpha.
// All placement is computed on the CPU in pixels; this shader only draws what it is given.

struct O {
    res: vec4<f32>,       // surface w, h, time, energy
    art: vec4<f32>,       // current artwork: centre x, centre y, half w, half h (px); half w 0 = absent
    art_p: vec4<f32>,     // alpha, corner radius (px), shadow strength (0..1), sampling lod (blur bridge)
    prev: vec4<f32>,      // previous artwork (crossfade source), same layout
    prev_p: vec4<f32>,
    shade: vec4<f32>,     // ambient shadow rgb (linear, album bg at L 0.05), unused
    title: vec4<f32>,     // left, top, width, height (px); width 0 = absent
    artist: vec4<f32>,
    meta_r: vec4<f32>,
    text_a: vec4<f32>,    // title alpha, artist alpha, meta alpha, text shadow strength
    col_title: vec4<f32>, // linear rgb per role
    col_artist: vec4<f32>,
    col_meta: vec4<f32>,
    col_lyr: vec4<f32>,   // current lyric rgb, slow glow boost (0..1, never per beat)
    col_ctx: vec4<f32>,   // context lyric rgb
    lyr0: vec4<f32>,      // lyric slot rects: left, top, width, height (px); width 0 = absent
    lyr1: vec4<f32>,
    lyr2: vec4<f32>,
    lyr_p: vec4<f32>,     // slot alphas 0..2, unused
    lyr_e: vec4<f32>,     // slot emphasis 0..2 (0 = context colour, 1 = current colour), unused
    lyr_clip: vec4<f32>,  // lyric column mask: top, bottom, feather (px), on (> 0.5)
};
@group(0) @binding(0) var<uniform> o: O;
@group(0) @binding(1) var art_tex: texture_2d<f32>;
@group(0) @binding(12) var prev_tex: texture_2d<f32>;
@group(0) @binding(2) var title_tex: texture_2d<f32>;
@group(0) @binding(3) var artist_tex: texture_2d<f32>;
@group(0) @binding(11) var meta_tex: texture_2d<f32>;
@group(0) @binding(5) var lyr_tex0: texture_2d<f32>;
@group(0) @binding(6) var lyr_tex1: texture_2d<f32>;
@group(0) @binding(7) var lyr_tex2: texture_2d<f32>;
@group(0) @binding(4) var smp: sampler;

// ---- control bar (second draw of the same pass, `fs_controls`, scissored to the bar area) -------------------
// All geometry is final surface pixels computed on the CPU (rise offset included).
struct C {
    res: vec4<f32>,       // surface w, h, master alpha, ui scale (px per design px)
    bar: vec4<f32>,       // pill centre x, centre y, half w, half h
    bg: vec4<f32>,        // pill rgb (linear), pill alpha
    fg: vec4<f32>,        // icon / text rgb
    accent: vec4<f32>,    // toggle / focus accent rgb
    prog: vec4<f32>,      // track x0, x1, centre y, half thickness (px); x1 <= x0 = no timeline
    prog_p: vec4<f32>,    // played fraction, hover 0..1 (knob), time text alpha, keyboard focus (1 = ring)
    sep: vec4<f32>,       // divider x, centre y, half height, unused
    t_el: vec4<f32>,      // elapsed text rect (left, top, w, h); w 0 = absent
    t_du: vec4<f32>,      // duration text rect
    // per button, 3 vec4: (cx, cy, half hit size, icon id) (opacity, hover, toggled, focus) (press, 0, 0, 0)
    b: array<vec4<f32>, 24>,
};
@group(0) @binding(8) var tex_el: texture_2d<f32>;
@group(0) @binding(9) var tex_du: texture_2d<f32>;
@group(0) @binding(10) var<uniform> c: C;

struct VOut { @builtin(position) pos: vec4<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var v: VOut;
    v.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    return v;
}

fn sd_round_box(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Text coverage at pixel p inside quad `rect` (0 outside). A quad drawn smaller than its bitmap (a context lyric
// line shown at reduced scale) is supersampled with 4 taps, so the downscaled glyphs do not shimmer.
fn text_cov(tex: texture_2d<f32>, rect: vec4<f32>, p: vec2<f32>) -> f32 {
    if (rect.z <= 0.0) { return 0.0; }
    let uv = (p - rect.xy) / rect.zw;
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { return 0.0; }
    let tw = f32(textureDimensions(tex).x);
    if (rect.z >= tw * 0.95) { return textureSampleLevel(tex, smp, uv, 0.0).r; }
    let q = 0.25 / rect.zw;   // a quarter of an output pixel in uv
    return 0.25 * (textureSampleLevel(tex, smp, uv + vec2<f32>(-q.x, -q.y), 0.0).r
        + textureSampleLevel(tex, smp, uv + vec2<f32>(q.x, -q.y), 0.0).r
        + textureSampleLevel(tex, smp, uv + vec2<f32>(-q.x, q.y), 0.0).r
        + textureSampleLevel(tex, smp, uv + vec2<f32>(q.x, q.y), 0.0).r);
}

// Soft shadow of a rounded box: 1 inside, falling to 0 over `blur` px outside.
fn box_shadow(p: vec2<f32>, c: vec2<f32>, hs: vec2<f32>, r: f32, blur: f32) -> f32 {
    let d = sd_round_box(p - c, hs, r);
    return 1.0 - smoothstep(-0.35 * blur, blur, d);
}

// Contact + ambient shadow of an artwork plate (sizes scale with the plate side S: y +0.0107S / blur 0.0214S,
// alpha 0.55, and y +0.054S / blur 0.16S, alpha 0.45; at S = 0.56H these are the spec's 0.006/0.012/0.03/0.09H).
fn plate_shadow(dst: vec4<f32>, p: vec2<f32>, rect: vec4<f32>, pp: vec4<f32>) -> vec4<f32> {
    if (pp.x <= 0.001 || rect.z <= 0.0) { return dst; }
    let side = 2.0 * max(rect.z, rect.w);
    let k = pp.x * clamp(pp.z, 0.0, 1.0);
    let amb = box_shadow(p, rect.xy + vec2<f32>(0.0, 0.054 * side), rect.zw, pp.y, 0.16 * side);
    var o2 = over(dst, o.shade.rgb, 0.45 * k * amb);
    let con = box_shadow(p, rect.xy + vec2<f32>(0.0, 0.0107 * side), rect.zw, pp.y, 0.0214 * side);
    return over(o2, vec3<f32>(0.0), 0.55 * k * con);
}

// Artwork pixels with an anti-aliased rounded edge and a 1 px inner stroke (white 8 %). The image is sampled
// 1:1 into its rect: no scaling, tinting or edge shading of the picture itself.
fn plate_art(dst: vec4<f32>, p: vec2<f32>, tex: texture_2d<f32>, rect: vec4<f32>, pp: vec4<f32>) -> vec4<f32> {
    if (pp.x <= 0.001 || rect.z <= 0.0) { return dst; }
    let d = sd_round_box(p - rect.xy, rect.zw, pp.y);
    let cover = (1.0 - smoothstep(-0.75, 0.75, d)) * pp.x;
    if (cover <= 0.0) { return dst; }
    let uv = clamp((p - rect.xy) / (rect.zw * 2.0) + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let t = textureSampleLevel(tex, smp, uv, pp.w).rgb;
    let ring = 1.0 - smoothstep(0.0, 1.0, abs(d + 0.5) - 0.5);
    return over(dst, mix(t, vec3<f32>(1.0), 0.08 * ring), cover);
}

// Light text shadow (strength from the CPU; at most 0.25 in Album Focus).
fn text_shadow(tex: texture_2d<f32>, rect: vec4<f32>, p: vec2<f32>, off: vec2<f32>) -> f32 {
    return max(text_cov(tex, rect, p - off), 0.5 * text_cov(tex, rect, p - 2.0 * off));
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag.xy;
    var c = vec4<f32>(0.0);   // premultiplied

    // ---- artwork: both shadows first, then the previous cover, then the current one over it ----------------
    c = plate_shadow(c, p, o.prev, o.prev_p);
    c = plate_shadow(c, p, o.art, o.art_p);
    c = plate_art(c, p, prev_tex, o.prev, o.prev_p);
    c = plate_art(c, p, art_tex, o.art, o.art_p);

    // ---- title block: shadows underneath, then coverage tinted per role ---------------------------------------
    let sh = o.text_a.w;
    let off = vec2<f32>(0.0, max(1.0, o.res.y * 0.0014));
    let shadow = max(max(text_shadow(title_tex, o.title, p, off) * o.text_a.x,
        text_shadow(artist_tex, o.artist, p, off) * o.text_a.y),
        text_shadow(meta_tex, o.meta_r, p, off) * o.text_a.z) * sh;
    c = over(c, vec3<f32>(0.0), clamp(shadow, 0.0, 0.85));
    c = over(c, o.col_title.rgb, clamp(text_cov(title_tex, o.title, p) * o.text_a.x, 0.0, 1.0));
    c = over(c, o.col_artist.rgb, clamp(text_cov(artist_tex, o.artist, p) * o.text_a.y, 0.0, 1.0));
    c = over(c, o.col_meta.rgb, clamp(text_cov(meta_tex, o.meta_r, p) * o.text_a.z, 0.0, 1.0));

    // ---- lyrics: up to three lines, colour from the slot's emphasis, optional feathered column mask ----------
    var m = 1.0;
    if (o.lyr_clip.w > 0.5) {
        let f = max(o.lyr_clip.z, 1.0);
        m = smoothstep(o.lyr_clip.x, o.lyr_clip.x + f, p.y) * (1.0 - smoothstep(o.lyr_clip.y - f, o.lyr_clip.y, p.y));
    }
    if (m > 0.0) {
        let boost = 1.0 + 0.10 * clamp(o.col_lyr.w, 0.0, 1.0);
        let a0 = o.lyr_p.x * m;
        let a1 = o.lyr_p.y * m;
        let a2 = o.lyr_p.z * m;
        let lshadow = max(max(text_shadow(lyr_tex0, o.lyr0, p, off) * a0, text_shadow(lyr_tex1, o.lyr1, p, off) * a1),
            text_shadow(lyr_tex2, o.lyr2, p, off) * a2) * sh;
        c = over(c, vec3<f32>(0.0), clamp(lshadow, 0.0, 0.85));
        let lc = min(o.col_lyr.rgb * boost, vec3<f32>(1.0));
        c = over(c, mix(o.col_ctx.rgb, lc, clamp(o.lyr_e.x, 0.0, 1.0)), clamp(text_cov(lyr_tex0, o.lyr0, p) * a0, 0.0, 1.0));
        c = over(c, mix(o.col_ctx.rgb, lc, clamp(o.lyr_e.y, 0.0, 1.0)), clamp(text_cov(lyr_tex1, o.lyr1, p) * a1, 0.0, 1.0));
        c = over(c, mix(o.col_ctx.rgb, lc, clamp(o.lyr_e.z, 0.0, 1.0)), clamp(text_cov(lyr_tex2, o.lyr2, p) * a2, 0.0, 1.0));
    }
    return c;
}

// ---- control bar ----------------------------------------------------------------------------------------------

fn cov(d: f32, fw: f32) -> f32 { return 1.0 - smoothstep(-fw, fw, d); }

fn sd_box(p: vec2<f32>, hb: vec2<f32>) -> f32 {
    let q = abs(p) - hb;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

fn sd_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

fn sd_tri(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    let e0 = p1 - p0; let e1 = p2 - p1; let e2 = p0 - p2;
    let v0 = p - p0; let v1 = p - p1; let v2 = p - p2;
    let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let d = min(min(vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                    vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                    vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
    return -sqrt(d.x) * sign(d.y);
}

// Skip-back glyph in unit space (x right, y down, extent about +-0.7).
fn skip_back(u: vec2<f32>) -> f32 {
    let bar = sd_box(u - vec2<f32>(-0.58, 0.0), vec2<f32>(0.08, 0.56)) - 0.05;
    let tri = sd_tri(u, vec2<f32>(0.62, -0.5), vec2<f32>(0.62, 0.5), vec2<f32>(-0.38, 0.0)) - 0.07;
    return min(bar, tri);
}

// Coverage (0..1) of icon `id` at unit-space point `u`; `fw` = one pixel in unit space.
fn icon_cov(id: u32, u: vec2<f32>, fw: f32) -> f32 {
    var d = 1.0e3;
    if (id == 0u) {                 // previous
        d = skip_back(u);
    } else if (id == 3u) {          // next
        d = skip_back(vec2<f32>(-u.x, u.y));
    } else if (id == 1u) {          // play
        d = sd_tri(u - vec2<f32>(0.08, 0.0), vec2<f32>(-0.5, -0.66), vec2<f32>(-0.5, 0.66), vec2<f32>(0.62, 0.0)) - 0.08;
    } else if (id == 2u) {          // pause
        d = sd_box(vec2<f32>(abs(u.x) - 0.34, u.y), vec2<f32>(0.17, 0.6)) - 0.07;
    } else if (id == 4u) {          // scenes: a 2 x 2 grid of rounded squares
        d = sd_box(abs(u) - vec2<f32>(0.36), vec2<f32>(0.26)) - 0.08;
    } else if (id == 5u) {          // lyrics: text lines
        let l0 = sd_seg(u, vec2<f32>(-0.7, -0.5), vec2<f32>(0.7, -0.5));
        let l1 = sd_seg(u, vec2<f32>(-0.7, 0.0), vec2<f32>(0.2, 0.0));
        let l2 = sd_seg(u, vec2<f32>(-0.7, 0.5), vec2<f32>(0.46, 0.5));
        d = min(l0, min(l1, l2)) - 0.1;
    } else if (id == 6u) {          // artwork layout: picture and text
        let pic = sd_box(u - vec2<f32>(-0.38, 0.0), vec2<f32>(0.32, 0.42)) - 0.1;
        let t0 = sd_seg(u, vec2<f32>(0.2, -0.28), vec2<f32>(0.7, -0.28)) - 0.09;
        let t1 = sd_seg(u, vec2<f32>(0.2, 0.12), vec2<f32>(0.55, 0.12)) - 0.09;
        d = min(pic, min(t0, t1));
    } else if (id == 7u) {          // fullscreen: corner brackets
        let q = abs(u);
        d = min(sd_seg(q, vec2<f32>(0.72, 0.72), vec2<f32>(0.72, 0.32)),
                sd_seg(q, vec2<f32>(0.72, 0.72), vec2<f32>(0.32, 0.72))) - 0.09;
    } else if (id == 8u) {          // exit: cross
        d = sd_seg(abs(u), vec2<f32>(0.0, 0.0), vec2<f32>(0.6, 0.6)) - 0.09;
    }
    return cov(d, fw);
}

// Rounded scrim behind a text quad (0 when the quad is absent), soft 2 px edge.
fn text_scrim(rect: vec4<f32>, p: vec2<f32>, s: f32) -> f32 {
    if (rect.z <= 0.0) { return 0.0; }
    let half = 0.5 * rect.zw + vec2<f32>(3.0 * s, 2.0 * s);
    let d = sd_round_box(p - rect.xy - 0.5 * rect.zw, half, min(half.x, half.y));
    return 1.0 - smoothstep(-2.0 * s, 0.0, d);
}

// premultiplied "over": `src` is a straight colour with alpha `sa`
fn over(dst: vec4<f32>, src: vec3<f32>, sa: f32) -> vec4<f32> {
    return vec4<f32>(src * sa + dst.rgb * (1.0 - sa), sa + dst.a * (1.0 - sa));
}

@fragment
fn fs_controls(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag.xy;
    let al = c.res.z;
    let s = c.res.w;
    var o = vec4<f32>(0.0);

    // ---- pill: soft shadow (not under the pill itself), translucent fill, 1 px inner highlight -------------
    let d = sd_round_box(p - c.bar.xy, c.bar.zw, c.bar.w);
    let ds = sd_round_box(p - c.bar.xy - vec2<f32>(0.0, 6.0 * s), c.bar.zw, c.bar.w);
    let inside = cov(d, 0.75);
    let shadow = (1.0 - smoothstep(-4.0 * s, 20.0 * s, ds)) * 0.42 * al * (1.0 - inside);
    o = over(o, vec3<f32>(0.0), shadow);
    o = over(o, c.bg.rgb, c.bg.w * al * inside);
    let top = clamp(-(p.y - c.bar.y) / max(c.bar.w, 1.0), -1.0, 1.0);
    let hw = 0.5 * max(s, 1.0);
    let ring = 1.0 - smoothstep(0.0, 1.0, abs(d + hw) - hw);
    o = over(o, vec3<f32>(1.0), 0.12 * (0.6 + 0.4 * top) * ring * al * inside);

    // ---- divider between the transport group and the view group ---------------------------------------------
    let dv = (1.0 - smoothstep(0.5, 1.5, abs(p.x - c.sep.x))) * step(abs(p.y - c.sep.y), c.sep.z);
    o = over(o, vec3<f32>(1.0), 0.16 * dv * al);

    // ---- buttons --------------------------------------------------------------------------------------------
    for (var i = 0u; i < 8u; i = i + 1u) {
        let v0 = c.b[i * 3u];
        let v1 = c.b[i * 3u + 1u];
        let v2 = c.b[i * 3u + 2u];
        let q = p - v0.xy;
        let half = v0.z;
        if (half <= 0.0 || abs(q.x) > half * 1.8 || abs(q.y) > half * 1.8) { continue; }
        // disabled: op = DISABLED_OPACITY (0.35), below 4.5:1 on purpose (inactive controls are WCAG-exempt)
        let op = v1.x;
        // hover / toggled plate: the pill fill, opaque, so the icon (fg or accent) keeps >= 4.5:1 on any scene
        let plate = cov(length(q) - half * 0.98, 0.75);
        o = over(o, c.bg.rgb, max(v1.y, v1.z) * plate * op * al);
        // hover cue: a 1.5 px fg ring just inside the plate edge, >= 3:1 against the pill (checked in controls.rs)
        let hring = cov(abs(length(q) - half * 0.98 + 0.75 * s) - 0.75 * s, 0.75);
        o = over(o, c.fg.rgb, 0.7 * v1.y * hring * op * al);
        // icon: 22 px in a 44 px hit target, shrinks a little while pressed
        let isz = half * 0.5 * (1.0 - 0.06 * v2.x);
        let ic = icon_cov(u32(v0.w + 0.5), q / isz, 1.0 / isz);
        o = over(o, mix(c.fg.rgb, c.accent.rgb, v1.z), ic * op * al);
        // toggled: accent dot under the icon (a second cue besides the colour)
        let dot = cov(length(q - vec2<f32>(0.0, half * 0.8)) - 2.2 * s, 0.75);
        o = over(o, c.accent.rgb, dot * v1.z * op * al);
        // keyboard focus ring: 2 px, 4 px outside the hit target, full strength (never faded by state)
        if (v1.w > 0.5) {
            let rw = max(1.0, s);
            let rr = half + 4.0 * s + rw;
            let fr = cov(abs(length(q) - rr) - rw, 0.75);
            o = over(o, vec3<f32>(1.0), fr * al);
        }
    }

    // ---- progress line and time texts -----------------------------------------------------------------------
    if (c.prog.y > c.prog.x) {
        let mid = 0.5 * (c.prog.x + c.prog.y);
        let hl = 0.5 * (c.prog.y - c.prog.x);
        let ht = c.prog.w;
        let pp = p - vec2<f32>(mid, c.prog.z);
        // backing: the pill fill at the pill opacity, a thin capsule just around the track, so the line keeps
        // its contrast over a bright scene (TRACK_ALPHA / PLAYED_ALPHA in overlay.rs; checked in controls.rs)
        // while focused the backing also covers the focus ring (4 px gap, 2 px ring), so the ring never sits on the
        // raw scene: the ring ends 7s + 2rw from the track and the pad is 7s + 3rw, a margin of rw >= 1 px at any
        // scale (rw has a 1 px floor, so a margin made of multiples of s alone would vanish below scale 1)
        let rw = max(1.0, s);
        let pad = 3.0 * s + step(0.5, c.prog_p.w) * (4.0 * s + 3.0 * rw);
        let under = cov(sd_round_box(pp, vec2<f32>(hl, ht) + vec2<f32>(pad), ht + pad), 0.75);
        o = over(o, c.bg.rgb, c.bg.w * under * al);
        let trk = cov(sd_round_box(pp, vec2<f32>(hl, ht), ht), 0.75);
        let xp = c.prog.x + clamp(c.prog_p.x, 0.0, 1.0) * (c.prog.y - c.prog.x);
        let played = 1.0 - smoothstep(-0.5, 0.5, p.x - xp);
        o = over(o, c.fg.rgb, trk * (0.12 + 0.83 * played) * al);
        let knob = cov(length(p - vec2<f32>(xp, c.prog.z)) - 6.0 * s * c.prog_p.y, 0.75);
        o = over(o, c.fg.rgb, knob * c.prog_p.y * al);
        // keyboard focus ring: 2 px, 4 px outside the resting backing capsule (same style as the buttons)
        if (c.prog_p.w > 0.5) {
            let fr = cov(abs(sd_round_box(pp, vec2<f32>(hl, ht) + vec2<f32>(7.0 * s + rw), ht + 7.0 * s + rw)) - rw, 0.75);
            o = over(o, vec3<f32>(1.0), fr * al);
        }
        // times: shadow first, then the text, same technique as the title
        let toff = vec2<f32>(0.0, max(1.0, 1.2 * s));
        let ta = c.prog_p.z * al;
        // soft local scrims (pill fill and opacity) so the times read on any scene
        let scrim = max(text_scrim(c.t_el, p, s), text_scrim(c.t_du, p, s));
        o = over(o, c.bg.rgb, c.bg.w * scrim * ta);
        let sh = max(text_cov(tex_el, c.t_el, p - toff), text_cov(tex_du, c.t_du, p - toff)) * 0.7 * ta;
        o = over(o, vec3<f32>(0.0), sh);
        let te = max(text_cov(tex_el, c.t_el, p), text_cov(tex_du, c.t_du, p));
        o = over(o, c.fg.rgb, te * ta);
    }
    return o;
}

// ---- scene panel (third draw of the same pass, `fs_panel`, whole surface) --------------------------------------
// The panel dims the picture a little (black 0.45, 0.65 towards the bottom, no blur: the visualisation keeps
// running underneath) and floats scene tiles on it. Thumbnails and labels come from ONE atlas (binding 13): the
// thumbnails above (3 x 2 cells of 256 x 144, cell = scene index), the label cells below (3 x 3 cells of 256 x 64,
// white with the glyph coverage in alpha, drawn 1:1). Layout constants mirror overlay.rs (THUMB_SIZE, LABEL_SIZE,
// LABEL_ORIGIN_Y, ATLAS_SIZE; a test checks them). All geometry is final surface pixels computed on the CPU.
struct P {
    res: vec4<f32>,       // surface w, h, master alpha, ui scale
    accent: vec4<f32>,    // accent rgb, heading alpha
    pill_t: vec4<f32>,    // "Current" text w, h; "EXPERIMENTAL" pill text w, h
    head: vec4<f32>,      // heading text rect (left, top, w, h); w 0 = none
    n: vec4<f32>,         // tile count
    // per tile, 6 vec4: rect (cx, cy, half w, half h); (alpha, hover, selected, focus); (experimental, has thumb,
    // atlas cell, 0); label rect; "Current" pill (cx, cy, half w, half h); "EXPERIMENTAL" pill
    t: array<vec4<f32>, 30>,
};
@group(0) @binding(13) var atlas_tex: texture_2d<f32>;
@group(0) @binding(14) var<uniform> pn: P;

const ATLAS_SIZE = vec2<f32>(768.0, 480.0);
const THUMB_SIZE = vec2<f32>(256.0, 144.0);
const LABEL_ORIGIN_Y = 288.0;

// Coverage of atlas label `cell` drawn 1:1 into `rect` (integer left/top; 0 outside or when absent).
fn label_cov(cell: u32, rect: vec4<f32>, p: vec2<f32>) -> f32 {
    if (rect.z <= 0.0) { return 0.0; }
    let q = floor(p - rect.xy);
    if (q.x < 0.0 || q.y < 0.0 || q.x >= rect.z || q.y >= rect.w) { return 0.0; }
    let org = vec2<f32>(f32(cell % 3u) * 256.0, LABEL_ORIGIN_Y + f32(cell / 3u) * 64.0);
    return textureLoad(atlas_tex, vec2<i32>(org + q), 0).a;
}

// A label bitmap of size `sz` centred on `c` (pixel aligned).
fn centred(c: vec2<f32>, sz: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(floor(c - 0.5 * sz + vec2<f32>(0.5)), sz);
}

fn panel_tile(dst: vec4<f32>, p: vec2<f32>, i: u32) -> vec4<f32> {
    let b = i * 6u;
    let r = pn.t[b];
    let v1 = pn.t[b + 1u];
    let v2 = pn.t[b + 2u];
    let lab = pn.t[b + 3u];
    let cp = pn.t[b + 4u];
    let ep = pn.t[b + 5u];
    let al = v1.x;
    if (al <= 0.001 || r.z <= 0.0) { return dst; }
    let s = pn.res.w;
    let rad = 10.0 * s;
    let q = p - r.xy;
    // cheap reject: only the tile plus its shadow and focus ring (40 design px), or its label, can change this pixel
    let near_tile = all(abs(q) <= r.zw + vec2<f32>(40.0 * s));
    let near_label = lab.z > 0.0 && p.x >= lab.x - 2.0 && p.x < lab.x + lab.z + 2.0 && p.y >= lab.y - 2.0 && p.y < lab.y + lab.w + 2.0;
    if (!near_tile && !near_label) { return dst; }
    let d = sd_round_box(q, r.zw, rad);
    let inside = cov(d, 0.75);
    var o2 = dst;
    // soft shadow under the floating tile (not under the tile itself)
    o2 = over(o2, vec3<f32>(0.0), 0.40 * al * box_shadow(p, r.xy + vec2<f32>(0.0, 6.0 * s), r.zw, rad, 18.0 * s) * (1.0 - inside));
    // thumbnail (or a clean placeholder), experimental tiles desaturated to 0.70, brightness 0.80 -> 1.0
    let uv01 = clamp(q / (2.0 * r.zw) + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    var tex = mix(vec3<f32>(0.045, 0.05, 0.06), vec3<f32>(0.11, 0.12, 0.14), uv01.y);
    if (v2.y > 0.5) {
        let cell = u32(v2.z + 0.5);
        let org = vec2<f32>(f32(cell % 3u) * THUMB_SIZE.x, f32(cell / 3u) * THUMB_SIZE.y);
        let t = clamp(uv01 * THUMB_SIZE, vec2<f32>(0.5), THUMB_SIZE - vec2<f32>(0.5));
        tex = textureSampleLevel(atlas_tex, smp, (org + t) / ATLAS_SIZE, 0.0).rgb;
    }
    let luma = dot(tex, vec3<f32>(0.2126, 0.7152, 0.0722));
    tex = mix(vec3<f32>(luma), tex, mix(1.0, 0.70, v2.x));
    tex = tex * mix(0.80, 1.0, max(v1.y, v1.z));
    o2 = over(o2, tex, inside * al);
    // 1 px stroke (white 0.12 at rest, 0.35 on hover)
    let ring = 1.0 - smoothstep(0.0, 1.0, abs(d + 0.5) - 0.5);
    o2 = over(o2, vec3<f32>(1.0), mix(0.12, 0.35, v1.y) * ring * al);
    // current scene: 2 px accent outline just inside the edge (and the "Current" pill: not colour alone)
    o2 = over(o2, pn.accent.rgb, v1.z * cov(abs(d + s) - s, 0.75) * al);
    if (v1.z > 0.5 && cp.z > 0.0) {
        o2 = over(o2, pn.accent.rgb, cov(sd_round_box(p - cp.xy, cp.zw, cp.w), 0.75) * al);
        o2 = over(o2, vec3<f32>(0.03, 0.035, 0.04), label_cov(5u, centred(cp.xy, pn.pill_t.xy), p) * al);
    }
    // experimental: a small outlined pill, neutral colours (no warning colours)
    if (v2.x > 0.5 && ep.z > 0.0) {
        let pd = sd_round_box(p - ep.xy, ep.zw, ep.w);
        o2 = over(o2, vec3<f32>(0.0), 0.40 * cov(pd, 0.75) * al);
        o2 = over(o2, vec3<f32>(1.0), 0.30 * (1.0 - smoothstep(0.0, 1.0, abs(pd + 0.5) - 0.5)) * al);
        o2 = over(o2, vec3<f32>(1.0), 0.60 * label_cov(6u, centred(ep.xy, pn.pill_t.zw), p) * al);
    }
    // label below the tile: a light shadow, then white 0.92
    let cell = u32(v2.z + 0.5);
    o2 = over(o2, vec3<f32>(0.0), 0.5 * label_cov(cell, lab + vec4<f32>(0.0, 1.0, 0.0, 0.0), p) * al);
    o2 = over(o2, vec3<f32>(1.0), 0.92 * label_cov(cell, lab, p) * al);
    // keyboard focus ring: white 2 px, 3 px outside the tile, between two black keylines, so it reads on any scene
    if (v1.w > 0.5) {
        let rw = max(1.0, s);
        let g = 3.0 * s;
        o2 = over(o2, vec3<f32>(0.0), 0.90 * cov(abs(d - (g - 0.5 * rw)) - 0.5 * rw, 0.75) * al);
        o2 = over(o2, vec3<f32>(0.0), 0.90 * cov(abs(d - (g + 2.5 * rw)) - 0.5 * rw, 0.75) * al);
        o2 = over(o2, vec3<f32>(1.0), cov(abs(d - (g + rw)) - rw, 0.75) * al);
    }
    return o2;
}

@fragment
fn fs_panel(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag.xy;
    let a = pn.res.z;
    // scrim: black 0.45, rising to 0.65 near the bottom
    let g = 0.45 + 0.20 * smoothstep(0.55, 1.0, p.y / pn.res.y);
    var col = over(vec4<f32>(0.0), vec3<f32>(0.0), g * a);
    col = over(col, vec3<f32>(1.0), 0.55 * pn.accent.w * label_cov(7u, pn.head, p));
    let n = u32(pn.n.x + 0.5);
    for (var i = 0u; i < 5u; i = i + 1u) {
        if (i >= n) { break; }
        col = panel_tile(col, p, i);
    }
    return col;
}
