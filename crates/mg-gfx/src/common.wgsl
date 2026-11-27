// Shared prelude for every scene shader (concatenated in front of the scene's own WGSL at build time).
// One uniform block for all scenes: the CPU mapper fills the common fields every frame and the
// scene-specific ones (ex0..ex5); each scene reads only what it needs. Fluid reuses none of ex0, ex1, ex3.y, ex4, ex5 or c
// (ex2 and ex3.xzw are the transition fx every scene shares): it
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
    ex2: vec4<f32>,  // transition veils: atmosphere, geometry, particles, global (see the fx helpers below)
    ex3: vec4<f32>,  // aurora: onset-front origin x in .y; transition: bridge luma .x, breathing .z, iris .w
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

// ---- scene transition fx (CPU side: mg-engine scene_switch.rs, packed from SceneParams::fx) -------------------------
// ex2 = veils 0 (visible) .. 1 (hidden): x atmosphere, y geometry, z particles, w global (scenes without separable
// layers); ex3.x = bridge relative luminance, ex3.z = bridge breathing amplitude, ex3.w = iris strength. All zero:
// every helper below returns its input unchanged, so a scene without a transition is bit-identical.
fn keep_atmo() -> f32 { return 1.0 - u.ex2.x; }
fn keep_geom() -> f32 { return 1.0 - u.ex2.y; }
fn keep_part() -> f32 { return 1.0 - u.ex2.z; }

fn oklab_of(c: vec3<f32>) -> vec3<f32> {
    let l = pow(max(0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b, 0.0), 0.3333333);
    let m = pow(max(0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b, 0.0), 0.3333333);
    let s = pow(max(0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b, 0.0), 0.3333333);
    return vec3<f32>(0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s);
}

fn lin_of_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    let m = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    let s = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
    let l3 = l * l * l;
    let m3 = m * m * m;
    let s3 = s * s * s;
    return vec3<f32>(4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
                     -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
                     -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3);
}

// The quiet bridge: a diagonal OKLab gradient from the palette's base to its secondary colour at 60 % chroma, set to
// the CPU's luminance (never black), +6 % lightness radial lift in the middle (falloff 0.6 screen heights), optional
// +-breathing at 0.25 Hz, dithered. No noise, no scene content.
fn bridge_colour(fragxy: vec2<f32>) -> vec3<f32> {
    let uv = fragxy / u.res;
    let t = clamp(0.5 * (uv.x + uv.y), 0.0, 1.0);
    let lab = mix(oklab_of(u.pal0.rgb), oklab_of(u.pal3.rgb), t);
    let r = length((uv - vec2<f32>(0.5)) * vec2<f32>(u.res.x / u.res.y, 1.0));
    let lift = 1.0 - smoothstep(0.0, 0.6, r);
    let ll = pow(max(u.ex3.x, 0.0), 0.3333333) * (1.0 + 0.06 * lift + u.ex3.z * sin(u.time * 1.5707963));
    let rgb = clamp(lin_of_oklab(vec3<f32>(ll, lab.y * 0.6, lab.z * 0.6)), vec3<f32>(0.0), vec3<f32>(1.0));
    return rgb + vec3<f32>((hash21(fragxy) - 0.5) / 255.0);
}

// Layered scenes: their layers fade themselves, the rest of the picture follows the atmosphere veil to the bridge.
fn veil_layers(c: vec3<f32>, fragxy: vec2<f32>) -> vec3<f32> {
    let v = clamp(u.ex2.x, 0.0, 1.0);
    if (v <= 0.0) { return c; }
    return mix(c, bridge_colour(fragxy), v);
}

// Scenes without separable layers: one global veil v: desaturate towards luma (0.7 v), an iris vignette closes from
// the corners (mask = smoothstep(r0, r0 + 0.35, distance); r0 reaches -0.35 at v = 1 so the centre is covered too),
// then the bridge replaces the picture. Iris strength 0 (reduced motion) leaves a plain fade.
fn veil_global(c: vec3<f32>, fragxy: vec2<f32>) -> vec3<f32> {
    let v = clamp(u.ex2.w, 0.0, 1.0);
    if (v <= 0.0) { return c; }
    let iris = clamp(u.ex3.w, 0.0, 1.0);
    let half = vec2<f32>(u.res.x / u.res.y, 1.0) * 0.5;
    let d = length((fragxy / u.res - vec2<f32>(0.5)) * vec2<f32>(u.res.x / u.res.y, 1.0)) / length(half);
    let r0 = 1.35 * (1.0 - v) - 0.35;
    let mask = mix(1.0, smoothstep(r0, r0 + 0.35, d), iris);
    let grey = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(mix(c, vec3<f32>(grey), 0.7 * v * iris), bridge_colour(fragxy), clamp(mask * v, 0.0, 1.0));
}
