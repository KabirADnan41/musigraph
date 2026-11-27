// Shared prelude for every scene shader (concatenated in front of the scene's own WGSL at build time).
// One uniform block for all scenes: the CPU mapper fills the common fields every frame and the
// scene-specific ones (ex0..ex5); each scene reads only what it needs. Fluid reuses NONE of ex0..ex5 or c: it
// reads its own block `FU` at @group(1) (fluid.wgsl, packed by fluid::FluidParams), so a crossfade between Fluid
// and Aurora/Focus/Orb can never feed one scene the other's fields. Fluid reads res, time, mode and pal0..pal3 here.

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
    pal0: vec4<f32>, // haze / base (w = dust phase 0..1)
    pal1: vec4<f32>, // bar body
    pal2: vec4<f32>, // accent / tips
    pal3: vec4<f32>, // secondary (haze tint, curtain)
    bars: array<vec4<f32>, 24>,
    peaks: array<vec4<f32>, 24>,
    bright: array<vec4<f32>, 24>,
    ex0: vec4<f32>,  // aurora: ribbon phases (cycles)
    ex1: vec4<f32>,  // aurora: band levels, bass to treble
    ex2: vec4<f32>,  // focus: artwork centre (uv x, uv y), half size (screen heights)
    ex3: vec4<f32>,  // focus: rotation phase, onset-front origin x, artwork present (0/1), unused
    ex4: vec4<f32>,  // orb: flow phase, rotation phase (cycles), ripple 0 and 1 impact azimuths
    ex5: vec4<f32>,  // orb: bass, mid, treble, breath
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

// x*x instead of pow(x, 2.0): pow with a negative base is undefined in WGSL and NaN on some drivers.
fn sq(x: f32) -> f32 { return x * x; }

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

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sd_round_box(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// No-audio / synthetic markers shared by every scene (never confusable with music activity).
fn mode_marker(col_in: vec3<f32>, frag: vec4<f32>, aspect: f32, line_y: f32) -> vec3<f32> {
    var col = col_in;
    let t = u.time;
    let mode = u.b.w;
    let p = frag.xy / u.res;
    if (mode > 1.5) {
        let dash = step(0.5, fract(p.x * aspect * 14.0));
        let pulse = 0.5 + 0.5 * sin(t * 1.5);
        col += vec3<f32>(0.9, 0.5, 0.1) * dash * (1.0 - smoothstep(0.0015, 0.004, abs(p.y - line_y))) * (0.25 + 0.35 * pulse);
    } else if (mode > 0.5) {
        let m = step(frag.x / u.res.y, 0.03) * step(1.0 - 0.03, frag.y / u.res.y);
        col += vec3<f32>(0.85, 0.85, 0.2) * m;
    }
    return col;
}
