// Overlay pass drawn AFTER the scene, at native surface resolution: album artwork (soft rounded edge,
// music-reactive halo, no border or panel) and song title/artist text (coverage bitmaps tinted with the
// palette, soft shadow for readability on any background). Output is premultiplied alpha.
// All placement is computed on the CPU in pixels; this shader only draws what it is given.

struct O {
    res: vec4<f32>,       // surface w, h, time, energy
    art_rect: vec4<f32>,  // centre x, centre y, half w, half h (pixels)
    art_p: vec4<f32>,     // alpha, corner radius (px), halo strength, halo spread (px)
    title: vec4<f32>,     // left, top, width, height (px); width 0 = absent
    artist: vec4<f32>,
    text_p: vec4<f32>,    // title alpha, artist alpha, shadow strength, unused
    col_text: vec4<f32>,  // linear rgb
    col_halo: vec4<f32>,  // linear rgb
    lyr0: vec4<f32>,      // lyric slot rects: left, top, width, height (px); width 0 = absent
    lyr1: vec4<f32>,
    lyr2: vec4<f32>,
    lyr_p: vec4<f32>,     // slot alphas 0..2, slow glow boost (0..1, never per beat)
};
@group(0) @binding(0) var<uniform> o: O;
@group(0) @binding(1) var art_tex: texture_2d<f32>;
@group(0) @binding(2) var title_tex: texture_2d<f32>;
@group(0) @binding(3) var artist_tex: texture_2d<f32>;
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
    prog_p: vec4<f32>,    // played fraction, hover 0..1 (knob), time text alpha, unused
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

// Text coverage at pixel p inside quad `rect` (0 outside). `shift` displaces the lookup (for the shadow).
fn text_cov(tex: texture_2d<f32>, rect: vec4<f32>, p: vec2<f32>) -> f32 {
    if (rect.z <= 0.0) { return 0.0; }
    let uv = (p - rect.xy) / rect.zw;
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { return 0.0; }
    return textureSampleLevel(tex, smp, uv, 0.0).r;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag.xy;
    var rgb = vec3<f32>(0.0);   // premultiplied
    var a = 0.0;

    // ---- artwork halo + artwork -------------------------------------------------------------------
    let art_a = o.art_p.x;
    if (art_a > 0.001 && o.art_rect.z > 0.0) {
        let c = o.art_rect.xy;
        let hs = o.art_rect.zw;
        let d = sd_round_box(p - c, hs, o.art_p.y);
        // halo: a soft glow that leaves the artwork like light, reacting to the music (strength from the CPU)
        let spread = max(o.art_p.w, 1.0);
        let halo = exp(-max(d, 0.0) / spread) * o.art_p.z * art_a;
        let halo_a = clamp(halo * 0.55, 0.0, 0.85);
        rgb = o.col_halo.rgb * halo_a;
        a = halo_a;
        // artwork pixels with an anti-aliased rounded edge
        let cover = (1.0 - smoothstep(-1.0, 1.0, d)) * art_a;
        if (cover > 0.0) {
            let uv = (p - c) / (hs * 2.0) + vec2<f32>(0.5);
            let t = textureSampleLevel(art_tex, smp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
            // soft inner edge shading keeps the picture from looking like a pasted rectangle
            let edge = 1.0 - 0.18 * smoothstep(-14.0, 0.0, d);
            rgb = t.rgb * edge * cover + rgb * (1.0 - cover);
            a = cover + a * (1.0 - cover);
        }
    }

    // ---- text: shadow first (underneath), then coverage tinted with the palette text colour ---------
    let sh = o.text_p.z;
    let off = vec2<f32>(0.0, max(1.0, o.res.y * 0.0014));
    var cov_t = text_cov(title_tex, o.title, p);
    var cov_a = text_cov(artist_tex, o.artist, p);
    let sh_t = max(text_cov(title_tex, o.title, p - off), 0.5 * text_cov(title_tex, o.title, p - 2.0 * off)) * sh * o.text_p.x;
    let sh_a = max(text_cov(artist_tex, o.artist, p - off), 0.5 * text_cov(artist_tex, o.artist, p - 2.0 * off)) * sh * o.text_p.y;
    let shadow = clamp(max(sh_t, sh_a), 0.0, 0.85);
    // shadow darkens what is already there (black, premultiplied)
    rgb = rgb * (1.0 - shadow);
    a = shadow + a * (1.0 - shadow);
    let ta = clamp(cov_t * o.text_p.x, 0.0, 1.0);
    let aa = clamp(cov_a * o.text_p.y, 0.0, 1.0);
    let tcol = o.col_text.rgb;
    rgb = tcol * ta + rgb * (1.0 - ta);
    a = ta + a * (1.0 - ta);
    rgb = tcol * 0.86 * aa + rgb * (1.0 - aa);   // artist line a touch dimmer than the title
    a = aa + a * (1.0 - aa);

    // ---- lyrics: up to three lines (previous fading out, current, next), same soft shadow as the title ----
    let boost = 1.0 + 0.10 * clamp(o.lyr_p.w, 0.0, 1.0);
    let lcol = min(tcol * boost, vec3<f32>(1.0));
    let sh0 = max(text_cov(lyr_tex0, o.lyr0, p - off), 0.5 * text_cov(lyr_tex0, o.lyr0, p - 2.0 * off)) * o.lyr_p.x;
    let sh1 = max(text_cov(lyr_tex1, o.lyr1, p - off), 0.5 * text_cov(lyr_tex1, o.lyr1, p - 2.0 * off)) * o.lyr_p.y;
    let sh2 = max(text_cov(lyr_tex2, o.lyr2, p - off), 0.5 * text_cov(lyr_tex2, o.lyr2, p - 2.0 * off)) * o.lyr_p.z;
    let lshadow = clamp(max(sh0, max(sh1, sh2)) * sh, 0.0, 0.85);
    rgb = rgb * (1.0 - lshadow);
    a = lshadow + a * (1.0 - lshadow);
    let l0 = clamp(text_cov(lyr_tex0, o.lyr0, p) * o.lyr_p.x, 0.0, 1.0);
    let l1 = clamp(text_cov(lyr_tex1, o.lyr1, p) * o.lyr_p.y, 0.0, 1.0);
    let l2 = clamp(text_cov(lyr_tex2, o.lyr2, p) * o.lyr_p.z, 0.0, 1.0);
    rgb = lcol * l0 + rgb * (1.0 - l0);
    a = l0 + a * (1.0 - l0);
    rgb = lcol * l1 + rgb * (1.0 - l1);
    a = l1 + a * (1.0 - l1);
    rgb = lcol * l2 + rgb * (1.0 - l2);
    a = l2 + a * (1.0 - l2);

    return vec4<f32>(rgb, a);
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
    } else if (id == 4u) {          // next scene: stacked layers
        let back = abs(sd_box(u - vec2<f32>(-0.2, -0.2), vec2<f32>(0.5, 0.5)) - 0.08) - 0.06;
        let front = sd_box(u - vec2<f32>(0.22, 0.22), vec2<f32>(0.5, 0.5)) - 0.08;
        d = min(max(back, -(front - 0.16)), front);
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
        let op = v1.x;
        // hover / toggled plate
        let plate = cov(length(q) - half * 0.98, 0.75);
        o = over(o, vec3<f32>(1.0), (0.13 * v1.y + 0.07 * v1.z) * plate * op * al);
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
        let under = cov(sd_round_box(pp, vec2<f32>(hl, ht) + vec2<f32>(1.5 * s), ht + 1.5 * s), 0.75);
        o = over(o, vec3<f32>(0.0), 0.30 * under * al);
        let trk = cov(sd_round_box(pp, vec2<f32>(hl, ht), ht), 0.75);
        let xp = c.prog.x + clamp(c.prog_p.x, 0.0, 1.0) * (c.prog.y - c.prog.x);
        let played = 1.0 - smoothstep(-0.5, 0.5, p.x - xp);
        o = over(o, c.fg.rgb, trk * (0.28 + 0.67 * played) * al);
        let knob = cov(length(p - vec2<f32>(xp, c.prog.z)) - 6.0 * s * c.prog_p.y, 0.75);
        o = over(o, c.fg.rgb, knob * c.prog_p.y * al);
        // times: shadow first, then the text, same technique as the title
        let toff = vec2<f32>(0.0, max(1.0, 1.2 * s));
        let ta = c.prog_p.z * al;
        let sh = max(text_cov(tex_el, c.t_el, p - toff), text_cov(tex_du, c.t_du, p - toff)) * 0.7 * ta;
        o = over(o, vec3<f32>(0.0), sh);
        let te = max(text_cov(tex_el, c.t_el, p), text_cov(tex_du, c.t_du, p));
        o = over(o, c.fg.rgb, te * ta);
    }
    return o;
}
