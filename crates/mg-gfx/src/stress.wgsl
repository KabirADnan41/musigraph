// Musigraph stress shader (P3.3). Original procedural scene sized like the planned P4 look:
// 4-octave haze, 64 edge-to-edge spectrum bars with LED segmentation, mirrored floor reflection
// with ripple, analytic glow from neighbouring bars, vignette. Driven by one uniform buffer.

struct U {
    res: vec2<f32>,
    time: f32,
    level: f32,
    bands: array<vec4<f32>, 16>,
};
@group(0) @binding(0) var<uniform> u: U;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    let r = q + dot(q, q + 45.32);
    return fract(r.x * r.y);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), s.x),
               mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), s.x), s.y);
}

fn fbm(p0: vec2<f32>) -> f32 {
    var p = p0;
    var a = 0.5;
    var v = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        v = v + a * vnoise(p);
        p = p * 2.03 + vec2<f32>(17.0, 9.0);
        a = a * 0.5;
    }
    return v;
}

fn band(i: i32) -> f32 {
    let c = clamp(i, 0, 63);
    return u.bands[c / 4][c % 4];
}

fn palette(x: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.28318 * (vec3<f32>(0.0, 0.33, 0.67) + x * 0.6 + 0.55));
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(frag.x / u.res.x, frag.y / u.res.y);
    let aspect = u.res.x / u.res.y;
    let t = u.time;
    let floor_y = 0.62;

    // Atmospheric background: drifting haze, brighter with level.
    let hp = vec2<f32>(uv.x * aspect, uv.y) * 2.2 + vec2<f32>(t * 0.03, t * 0.012);
    let haze = fbm(hp) * fbm(hp * 0.6 + 7.0);
    var col = mix(vec3<f32>(0.01, 0.015, 0.04), vec3<f32>(0.06, 0.03, 0.12), uv.y)
        + haze * (0.10 + 0.25 * u.level) * palette(uv.x * 0.5 + t * 0.02);

    // Bars across the whole width.
    let fi = uv.x * 64.0;
    let bi = i32(floor(fi));
    let bx = fract(fi);
    let h = band(bi) * 0.42;
    let top = floor_y - h;
    let bar_mask = smoothstep(0.08, 0.14, bx) * (1.0 - smoothstep(0.86, 0.92, bx));
    let seg = smoothstep(0.1, 0.3, fract((floor_y - uv.y) * 60.0));
    let bcol = palette(f32(bi) / 64.0);
    if (uv.y <= floor_y && uv.y >= top) {
        col = col + bcol * bar_mask * (0.55 + 0.45 * seg);
    }

    // Analytic glow from this and neighbouring bars.
    var glow = 0.0;
    for (var d = -2; d <= 2; d = d + 1) {
        let hh = floor_y - band(bi + d) * 0.42;
        let dx = (fi - (f32(bi + d) + 0.5)) / 64.0 * aspect;
        let dy = uv.y - hh;
        glow = glow + exp(-(dx * dx * 900.0 + dy * dy * 140.0)) * band(bi + d);
    }
    col = col + palette(uv.x) * glow * 0.35;

    // Mirrored floor reflection with ripple and fade.
    if (uv.y > floor_y) {
        let ry = floor_y - (uv.y - floor_y) * 0.9 + sin(uv.x * 90.0 + t * 2.0) * 0.004;
        let rh = floor_y - ry;
        let inside = step(rh, band(bi) * 0.42) * bar_mask;
        let fade = exp(-(uv.y - floor_y) * 6.0);
        col = col + bcol * inside * fade * 0.35;
        col = col * (1.0 - 0.25 * (uv.y - floor_y));
    }

    // Beat-lit vignette.
    let v = distance(uv, vec2<f32>(0.5, 0.55));
    col = col * (1.0 - smoothstep(0.35, 0.95, v) * 0.7) + palette(0.2) * u.level * 0.04;
    return vec4<f32>(col, 1.0);
}
