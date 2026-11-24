// Scene 1 "Cinematic Spectrum": original procedural shader. All music-to-motion decisions are made on the
// CPU (mg-engine scene_map); this shader only draws the parameters it is given. Resolution independent:
// everything is expressed in screen fractions and aspect-corrected where shape matters.

const SLOTS: i32 = 96;
const HMAX: f32 = 0.36;      // tallest bar, as a fraction of screen height
const X0: f32 = 0.07;
const X1: f32 = 0.93;

struct U {
    res: vec2<f32>,
    time: f32,
    energy: f32,
    // glow, haze, warp, swell
    a: vec4<f32>,
    // tilt, scroll.x, scroll.y, mode (0 live, 1 synthetic, 2 no audio / error)
    b: vec4<f32>,
    // ring1 (age, amp), ring2 (age, amp)
    c: vec4<f32>,
    pal0: vec4<f32>, // haze / base
    pal1: vec4<f32>, // bar body
    pal2: vec4<f32>, // accent / tips
    bars: array<vec4<f32>, 24>,
    peaks: array<vec4<f32>, 24>,
    bright: array<vec4<f32>, 24>,
};
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VOut;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2<f32>(1.0, 0.0)), w.x),
               mix(hash21(i + vec2<f32>(0.0, 1.0)), hash21(i + vec2<f32>(1.0, 1.0)), w.x), w.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    for (var k = 0; k < 3; k++) {
        s += a * vnoise(q);
        q = q * 2.03 + vec2<f32>(17.1, 9.3);
        a *= 0.5;
    }
    return s;
}

fn bar_h(i: i32) -> f32 { return u.bars[i >> 2][i & 3]; }
fn peak_h(i: i32) -> f32 { return u.peaks[i >> 2][i & 3]; }
fn bar_b(i: i32) -> f32 { return u.bright[i >> 2][i & 3]; }

// Smooth envelope between slot centres (the "light curtain" top edge), 0..1.
fn envelope(x: f32) -> f32 {
    let f = (x - X0) / (X1 - X0) * f32(SLOTS) - 0.5;
    let i0 = clamp(i32(floor(f)), 0, SLOTS - 1);
    let i1 = clamp(i0 + 1, 0, SLOTS - 1);
    let t = smoothstep(0.0, 1.0, clamp(f - floor(f), 0.0, 1.0));
    let inside = smoothstep(-0.5, 2.0, f + 0.5) * smoothstep(-0.5, 2.0, f32(SLOTS) - 0.5 - f);
    return mix(bar_h(i0), bar_h(i1), t) * inside;
}

// Spectrum colour at horizontal position x and height d above the horizon (d > 0), as additive light.
fn spectrum(x: f32, d: f32, glow: f32) -> vec3<f32> {
    if (d < -0.002) { return vec3<f32>(0.0); }
    let f = (x - X0) / (X1 - X0) * f32(SLOTS);
    let env_h = envelope(x) * HMAX;
    var col = vec3<f32>(0.0);
    // light curtain under the envelope: soft fill that brightens toward the horizon
    let curtain = (1.0 - smoothstep(0.0, env_h + 0.03, d)) * step(0.0, d);
    col += u.pal1.rgb * curtain * (0.10 + 0.16 * glow) * smoothstep(0.0, 0.04, env_h);
    // envelope line and halo
    col += u.pal2.rgb * (1.0 - smoothstep(0.0012, 0.0045, abs(d - env_h))) * 0.55 * smoothstep(0.004, 0.05, env_h);
    col += u.pal2.rgb * exp(-abs(d - env_h) * 22.0) * 0.20 * glow * smoothstep(0.004, 0.08, env_h);
    // bars
    if (f >= 0.0 && f < f32(SLOTS)) {
        let idx = i32(floor(f));
        let lx = abs(fract(f) - 0.5);
        let mask_x = 1.0 - smoothstep(0.30, 0.36, lx);
        let h = bar_h(idx) * HMAX;
        let inside = step(0.0, d) * (1.0 - smoothstep(h - 0.0025, h + 0.0025, d));
        let tip = smoothstep(0.62, 1.0, d / max(h, 0.002));
        let b = mix(0.55, 1.0, bar_b(idx));
        let fpos = abs(f - f32(SLOTS) * 0.5) / (f32(SLOTS) * 0.5);
        col += mix(u.pal1.rgb, u.pal2.rgb, clamp(tip + 0.45 * fpos, 0.0, 1.0)) * b * 0.95 * mask_x * inside * smoothstep(0.0015, 0.01, h);
        let ph = peak_h(idx) * HMAX;
        let cap = 1.0 - smoothstep(0.0012, 0.0035, abs(d - ph));
        col += u.pal2.rgb * cap * mask_x * 0.8 * smoothstep(0.01, 0.05, ph);
    }
    return col;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let aspect = u.res.x / u.res.y;
    var p = frag.xy / u.res;                       // 0..1, y down
    let t = u.time;
    let glow = u.a.x;
    let haze = u.a.y;
    let warp = u.a.z;
    let swell = u.a.w;
    let scroll = vec2<f32>(u.b.y, u.b.z);
    let mode = u.b.w;

    // idle-safe slow camera drift (time based, not audio driven) and horizon breathing
    p.x += 0.015 * sin(t * 6.2831853 / 40.0);
    let breathe = 1.0 + 0.03 * sin(t * 6.2831853 * 0.05);
    let yh = 0.60 - 0.02 * swell;

    // ---- atmosphere -------------------------------------------------------------------------
    let up = clamp((yh - p.y) / yh, 0.0, 1.0);      // 1 at the top of the screen, 0 at the horizon
    let sky_t = pow(1.0 - up, 1.7);                 // 0 top -> 1 horizon
    var col = u.pal0.rgb * mix(0.035, 0.34, sky_t);
    let q = vec2<f32>(p.x * aspect, p.y) * 1.5;
    let far = fbm(q * 0.7 + scroll * 0.2 + vec2<f32>(3.1, 1.7));
    let w = vec2<f32>(fbm(q + scroll + vec2<f32>(1.7, 9.2)), fbm(q + scroll * 1.1 + vec2<f32>(8.3, 2.8)));
    let near = fbm(q + warp * 2.2 * (w - 0.5) + scroll * 0.8);
    var haze_amt = (0.55 * near + 0.45 * far) * haze;
    haze_amt *= 1.0 + 0.25 * swell * exp(-pow((p.y - yh) / 0.15, 2.0));   // bass bloom near the horizon
    let below = step(yh, p.y);
    haze_amt *= mix(1.0, 0.45, below);
    col += mix(u.pal0.rgb, u.pal2.rgb, 0.25 + 0.2 * u.b.x) * haze_amt * mix(0.35, 0.95, sky_t);
    // horizon glow, lifted by the bass swell
    col += u.pal0.rgb * exp(-abs(p.y - yh) * 13.0) * (0.22 + 0.55 * swell);
    col += u.pal2.rgb * (1.0 - smoothstep(0.0008, 0.0024, abs(p.y - yh))) * 0.20 * breathe;

    // ---- spectrum above the horizon -----------------------------------------------------------
    let d_up = yh - p.y;
    col += spectrum(p.x, d_up, glow);

    // ---- reflection and ring ripples below the horizon -------------------------------------
    if (p.y > yh) {
        let dr = p.y - yh;
        let ring_amp = max(u.c.y * exp(-u.c.x / 0.8), u.c.w * exp(-u.c.z / 0.8));
        let ripple = 0.0022 * sin(dr * 70.0 - t * 1.6) * (0.25 + 3.0 * ring_amp);
        let fade = exp(-dr * 5.0) * 0.35;
        col += spectrum(p.x + ripple, dr * 1.15, glow * 0.6) * fade;
        // expanding elliptical rings on the water plane (impulses affect geometry/glow only, never colour)
        let qd = vec2<f32>((p.x - 0.5) * aspect, dr * aspect * 2.4);
        let r = length(qd);
        let r1 = u.c.x * 0.55;
        let r2 = u.c.z * 0.55;
        let ring1 = exp(-pow((r - r1) / 0.035, 2.0)) * u.c.y * exp(-u.c.x / 0.8);
        let ring2 = exp(-pow((r - r2) / 0.035, 2.0)) * u.c.w * exp(-u.c.z / 0.8);
        col += u.pal2.rgb * (ring1 + ring2) * 1.6 * smoothstep(0.0, 0.02, dr);
    }

    // ---- vignette, mode markers, tone map -------------------------------------------------------
    let vc = (frag.xy / u.res - 0.5) * vec2<f32>(1.0, 1.1);
    col *= 1.0 - 0.34 * smoothstep(0.25, 0.85, length(vc) * 1.25);
    if (mode > 1.5) {
        // no live audio: slow dashed amber line on the horizon (clearly not music activity)
        let dash = step(0.5, fract(p.x * aspect * 14.0));
        let pulse = 0.5 + 0.5 * sin(t * 1.5);
        col += vec3<f32>(0.9, 0.5, 0.1) * dash * (1.0 - smoothstep(0.0015, 0.004, abs(p.y - yh))) * (0.25 + 0.35 * pulse);
    } else if (mode > 0.5) {
        // synthetic test signal: small marker block, bottom-left corner
        let m = step(frag.x / u.res.y, 0.03) * step(1.0 - 0.03, frag.y / u.res.y);
        col += vec3<f32>(0.85, 0.85, 0.2) * m;
    }
    var outc = aces(col * 1.05);
    outc += (hash21(frag.xy) - 0.5) / 255.0;        // dither against gradient banding
    return vec4<f32>(outc, 1.0);
}
