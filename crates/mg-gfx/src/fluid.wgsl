// Scene "Fluid" (F1, Experimental): dye-only liquid. Concatenated after common.wgsl (vs, U, hash21, aces,
// mode_marker). Three entry points share this file:
//   fs_sim  - one fixed 1/60 s simulation step: ping-pong RGBA16Float, x/y/z = dye for pal1/pal2/pal3, w = heat.
//   fs_copy - resamples the field into a new grid (aspect change): the picture is kept, never cleared.
//   fs      - the scene: glossy render of the dye over a dark palette base.
// Velocity is ANALYTIC and divergence-free by construction (the curl of a stream function: 2-octave value
// noise + emitter swirl and push + onset vortices), so there is no pressure solve. All music mapping and
// smoothing happens on the CPU (mg-engine scene_map.rs); this file only integrates what it is given.

struct FU {
    emit: array<vec4<f32>, 3>, // bass/mid/treble emitter: uv x, uv y (down), strength (already gated), sigma (screen heights)
    swirl: vec4<f32>,          // swirl per emitter (1/s, signed), bass push (upward jet, screen heights/s)
    vort: array<vec4<f32>, 2>, // onset vortex: uv x, uv y, peak swirl speed (sh/s, signed), age (s; >= 100 none)
    flow: vec4<f32>,           // noise flow speed (sh/s), noise frequency (1/sh), evolution phase (lattice units, 0..256), buoyancy (upward drift, sh/s)
    look: vec4<f32>,           // dye memory tau (s), heat rate (1/s), exposure, jet width (sh; <= 0.03: twice the bass splat)
    misc: vec4<f32>,           // vortex dye hue (0 dark .. 1 bright onset: pal3 towards pal2) for slots 0 and 1, flow detail (0 laminar .. 1 turbulent), treble spread (0..1)
    sim: vec4<f32>,            // round-trip correction on (1) / plain RK2 semi-Lagrangian (0), grid width, grid height, unused
};
@group(1) @binding(0) var<uniform> fu: FU;
@group(1) @binding(1) var dye: texture_2d<f32>;
@group(1) @binding(2) var dye_smp: sampler;

const FL_DT: f32 = 0.016666668;
const FL_MAX_ADD: f32 = 0.012;      // flash guard: most dye one step may add to any texel
const FL_ZONE_TOP: f32 = 0.56;      // injection fades out between these uv heights:
const FL_ZONE_BOTTOM: f32 = 0.62;   // nothing is injected in the bottom 38% (artwork, lyrics, control bar)
const FL_VORTEX_SIGMA: f32 = 0.07;
const FL_VORTEX_PEAK_S: f32 = 0.18; // alpha-function envelope peak (slow in, slow out)

// Integer hash on a 256-periodic lattice: exact on every backend and the evolution phase can wrap at 256
// without a seam. The constant seed fixes the field (determinism).
fn fl_hash(i: vec2<f32>) -> f32 {
    let w = vec2<u32>(vec2<i32>(i) & vec2<i32>(255));
    var h = w.x * 1664525u + w.y * 1013904223u + 0x9e3779b9u;
    h = (h ^ (h >> 16u)) * 2246822519u;
    h = (h ^ (h >> 13u)) * 3266489917u;
    h = h ^ (h >> 16u);
    return f32(h) * (1.0 / 4294967295.0);
}

fn fl_dir(i: vec2<f32>) -> vec2<f32> {
    let a = fl_hash(i) * 6.2831853;
    return vec2<f32>(cos(a), sin(a));
}

// Gradient (Perlin-style) noise with its analytic gradient: (value, d/dx, d/dy), quintic fade. Value noise was
// tried first: its derivative vanishes on every lattice line, so the curl flow ran parallel to the lattice and
// drew straight horizontal and vertical fronts into the ink. Gradient noise has no such lines.
fn fl_noise_grad(q: vec2<f32>) -> vec3<f32> {
    let i = floor(q);
    let f = q - i;
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let dw = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let ga = fl_dir(i);
    let gb = fl_dir(i + vec2<f32>(1.0, 0.0));
    let gc = fl_dir(i + vec2<f32>(0.0, 1.0));
    let gd = fl_dir(i + vec2<f32>(1.0, 1.0));
    let va = dot(ga, f);
    let vb = dot(gb, f - vec2<f32>(1.0, 0.0));
    let vc = dot(gc, f - vec2<f32>(0.0, 1.0));
    let vd = dot(gd, f - vec2<f32>(1.0, 1.0));
    let k = va - vb - vc + vd;
    let value = va + w.x * (vb - va) + w.y * (vc - va) + w.x * w.y * k;
    let grad = ga + w.x * (gb - ga) + w.y * (gc - ga) + w.x * w.y * (ga - gb - gc + gd)
             + dw * (w.yx * k + vec2<f32>(vb, vc) - va);
    return vec3<f32>(value, grad);
}

// Octave rotations (lattice never aligned with the screen axes, nor with each other).
const FL_ROT1: mat2x2<f32> = mat2x2<f32>(0.8253, 0.5646, -0.5646, 0.8253);   // 34.4 degrees
const FL_ROT2: mat2x2<f32> = mat2x2<f32>(0.4536, -0.8912, 0.8912, 0.4536);   // -63.0 degrees

// v = (d psi/dy, -d psi/dx) for any stream function psi: divergence-free whatever psi is.
fn fl_curl(grad_psi: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(grad_psi.y, -grad_psi.x);
}

// Gaussian swirl around `c` with angular speed `s` at the centre (psi = s sigma^2 g).
fn fl_swirl(p: vec2<f32>, c: vec2<f32>, sigma: f32, s: f32) -> vec2<f32> {
    let d = p - c;
    let g = exp(-dot(d, d) / (2.0 * sigma * sigma));
    return fl_curl(-s * g * d);
}

// Upward jet (a vortex dipole: psi = P dx g): speed P straight up at the centre, return flow at the sides.
fn fl_jet(p: vec2<f32>, c: vec2<f32>, sigma: f32, push: f32) -> vec2<f32> {
    let d = p - c;
    let is2 = 1.0 / (sigma * sigma);
    let g = exp(-dot(d, d) * 0.5 * is2);
    return fl_curl(vec2<f32>(push * g * (1.0 - d.x * d.x * is2), -push * d.x * d.y * g * is2));
}

fn fl_env(age: f32) -> f32 {
    let x = max(age, 0.0) / FL_VORTEX_PEAK_S;
    return x * exp(1.0 - x);
}

// Velocity in screen heights per second at p (screen-height units, y down; x spans 0..aspect).
fn fl_vel(p: vec2<f32>, aspect: f32) -> vec2<f32> {
    let fq = fu.flow.y;
    let ph = fu.flow.z;
    // two rotated octaves; the second moves with twice the phase so both wrap seamlessly at 256
    let g1 = fl_noise_grad(FL_ROT1 * p * fq + vec2<f32>(ph, 0.0));
    let g2 = fl_noise_grad(FL_ROT2 * p * fq * 2.03 + vec2<f32>(31.0, 17.0 + 2.0 * ph));
    // psi = (speed / fq) (n1 + 0.35 d n2)  ->  grad_p psi = speed (R1^T grad_q n1 + 0.35 * 2.03 d R2^T grad_q n2);
    // d = flow detail: sustained sound flows laminar (the fine octave nearly off), impulsive sound churns
    let gp = g1.yz * FL_ROT1 + 0.7105 * clamp(fu.misc.z, 0.0, 1.0) * (g2.yz * FL_ROT2); // row-vector product = R^T g
    var v = fl_curl(fu.flow.x * 1.5 * gp);
    for (var i = 0; i < 3; i++) {
        let e = fu.emit[i];
        let c = vec2<f32>(e.x * aspect, e.y);
        v += fl_swirl(p, c, max(e.w, 0.03) * 1.6, fu.swirl[i]);
    }
    let b = fu.emit[0];
    // jet width: narrow for bass impacts, broad for a sustained low end (look.w; 0 = twice the bass splat)
    let jw = select(max(b.w, 0.03) * 2.0, fu.look.w, fu.look.w > 0.03);
    v += fl_jet(p, vec2<f32>(b.x * aspect, b.y), jw, fu.swirl.w);
    for (var j = 0; j < 2; j++) {
        let o = fu.vort[j];
        if (o.w < 50.0) {
            let s = o.z * fl_env(o.w) / (FL_VORTEX_SIGMA * 0.60653);
            v += fl_swirl(p, vec2<f32>(o.x * aspect, o.y), FL_VORTEX_SIGMA, s);
        }
    }
    // gentle uniform buoyancy (divergence-free); the soft top sink fades the dye as it rises. A broad convection
    // roll was tried and dropped: its fixed stagnation point drew straight lines into the ink.
    return v + vec2<f32>(0.0, -fu.flow.w);
}

fn fl_load(i: vec2<i32>, dims: vec2<i32>) -> vec4<f32> {
    return textureLoad(dye, clamp(i, vec2<i32>(0), dims - vec2<i32>(1)), 0);
}

fn fl_finite(v: vec4<f32>) -> vec4<f32> {
    // NaN and inf fail the comparison and become 0
    return select(vec4<f32>(0.0), v, abs(v) < vec4<f32>(1.0e4));
}

@fragment
fn fs_sim(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dims_i = vec2<i32>(textureDimensions(dye));
    let dims = vec2<f32>(dims_i);
    let uv = frag.xy / dims;
    let aspect = dims.x / dims.y;
    let to_uv = vec2<f32>(1.0 / aspect, 1.0);
    let p = uv / to_uv;

    // RK2 back-trace (semi-Lagrangian)
    let v1 = fl_vel(p, aspect);
    let pb = p - FL_DT * fl_vel(p - 0.5 * FL_DT * v1, aspect);
    let uvb = pb * to_uv;
    var s = textureSampleLevel(dye, dye_smp, uvb, 0.0);
    if (fu.sim.x > 0.5) {
        // Single-pass round-trip correction, MacCormack-INSPIRED but not MacCormack: true MacCormack/BFECC would
        // advect the back-traced result forward again (a second pass over the advected field). Here the forward
        // trace from the departure point samples the PREVIOUS field, so the correction 0.5 (here - back) only
        // compensates part of the interpolation smearing. Clamped to the departure cell's range (no overshoot).
        let w1 = fl_vel(pb, aspect);
        let pf = pb + FL_DT * fl_vel(pb + 0.5 * FL_DT * w1, aspect);
        let back = textureSampleLevel(dye, dye_smp, pf * to_uv, 0.0);
        let here = fl_load(vec2<i32>(frag.xy), dims_i);
        let i0 = vec2<i32>(floor(uvb * dims - 0.5));
        let a = fl_load(i0, dims_i);
        let b = fl_load(i0 + vec2<i32>(1, 0), dims_i);
        let c = fl_load(i0 + vec2<i32>(0, 1), dims_i);
        let d = fl_load(i0 + vec2<i32>(1, 1), dims_i);
        s = clamp(s + 0.5 * (here - back), min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
    }

    // injection: three palette-role emitters, heat from the bass, a little onset-coloured dye in each vortex
    let tau = max(fu.look.x, 0.5);
    var add = vec4<f32>(0.0);
    // high, airy sound spreads as a thin, wide veil: the treble splat widens horizontally (up to 3x)
    let spread = vec3<f32>(1.0, 1.0, 1.0 + 2.0 * clamp(fu.misc.w, 0.0, 1.0));
    for (var i = 0; i < 3; i++) {
        let e = fu.emit[i];
        let d = (p - vec2<f32>(e.x * aspect, e.y)) / vec2<f32>(spread[i], 1.0);
        let sg = max(e.w, 0.03);
        let g = exp(-dot(d, d) / (2.0 * sg * sg));
        // normalised by memory: the steady-state peak is set by strength, not by tau
        add[i] += min(max(e.z, 0.0) * 3.2 / tau, 0.72) * FL_DT * g;
        if (i == 0) {
            add.w += max(fu.look.y, 0.0) * FL_DT * g;
        }
    }
    for (var j = 0; j < 2; j++) {
        let o = fu.vort[j];
        if (o.w < 50.0) {
            let d = p - vec2<f32>(o.x * aspect, o.y);
            let g = exp(-dot(d, d) / (2.0 * 0.05 * 0.05)) * fl_env(o.w) * abs(sign(o.z));
            let h = clamp(fu.misc[j], 0.0, 1.0);
            // onsets add a little dye in the secondary role (pal3), towards the accent (pal2) for bright onsets
            add += vec4<f32>(0.0, 0.6 * h, 1.0 - 0.6 * h, 0.0) * 0.006 * g;
        }
    }
    let zone = 1.0 - smoothstep(FL_ZONE_TOP, FL_ZONE_BOTTOM, uv.y);
    s += min(add, vec4<f32>(FL_MAX_ADD)) * zone;

    // memory, edge sink, floor, soft ceiling (identity below 2, never above 3)
    s *= vec4<f32>(vec3<f32>(exp(-FL_DT / tau)), exp(-FL_DT / 1.5));
    // soft sinks (rates, so the fade is a gradient, not a line): sides and bottom, the top where plumes rise,
    // and the lyrics / control band, where anything the flow carries in fades within about a second
    let edge = min(min(uv.x, 1.0 - uv.x), 1.0 - uv.y);
    let sink = 6.0 * sq(1.0 - smoothstep(0.0, 0.08, edge))
             + 3.0 * sq(1.0 - smoothstep(0.0, 0.18, uv.y))
             + 1.4 * smoothstep(FL_ZONE_TOP, FL_ZONE_BOTTOM + 0.1, uv.y);
    s *= exp(-FL_DT * sink);
    s = select(s, vec4<f32>(0.0), s < vec4<f32>(1.0e-4));
    s = select(s, 2.0 + tanh(s - 2.0), s > vec4<f32>(2.0));
    return fl_finite(s);
}

// The field resampled into a new grid (fu.sim.yz = the new grid size); bilinear, so the mean dye is kept.
@fragment
fn fs_copy(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    return fl_finite(textureSampleLevel(dye, dye_smp, frag.xy / max(fu.sim.yz, vec2<f32>(1.0)), 0.0));
}

// Smooth upsampling of the low-resolution dye: cubic B-spline from four bilinear taps.
fn fl_tex(uv: vec2<f32>, dims: vec2<f32>) -> vec4<f32> {
    let tc = uv * dims - 0.5;
    let fc = floor(tc);
    let r = tc - fc;
    let r2 = r * r;
    let r3 = r2 * r;
    let w0 = (-r3 + 3.0 * r2 - 3.0 * r + 1.0) / 6.0;
    let w1 = (3.0 * r3 - 6.0 * r2 + 4.0) / 6.0;
    let w2 = (-3.0 * r3 + 3.0 * r2 + 3.0 * r + 1.0) / 6.0;
    let w3 = r3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let h0 = (fc - 0.5 + w1 / g0) / dims;
    let h1 = (fc + 1.5 + w3 / g1) / dims;
    return g0.y * (g0.x * textureSampleLevel(dye, dye_smp, vec2<f32>(h0.x, h0.y), 0.0)
                 + g1.x * textureSampleLevel(dye, dye_smp, vec2<f32>(h1.x, h0.y), 0.0))
         + g1.y * (g0.x * textureSampleLevel(dye, dye_smp, vec2<f32>(h0.x, h1.y), 0.0)
                 + g1.x * textureSampleLevel(dye, dye_smp, vec2<f32>(h1.x, h1.y), 0.0));
}

fn fl_total(uv: vec2<f32>) -> f32 {
    let d = textureSampleLevel(dye, dye_smp, uv, 0.0);
    return d.x + d.y + d.z;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let aspect = u.res.x / u.res.y;
    let p = frag.xy / u.res;
    let dims = vec2<f32>(textureDimensions(dye));
    let d = max(fl_tex(p, dims), vec4<f32>(0.0));
    let tot = d.x + d.y + d.z;
    let hue = (d.x * u.pal1.rgb + d.y * u.pal2.rgb + d.z * u.pal3.rgb) / max(tot, 1.0e-4);
    // Beer-Lambert coverage: thin dye is a translucent veil, thick dye saturates (never a hard edge)
    let cov = 1.0 - exp(-2.0 * tot);

    // glossy surface: normal from the density gradient (one dye texel apart, both directions)
    let e = 1.5 / dims;
    let gx = fl_total(p + vec2<f32>(e.x, 0.0)) - fl_total(p - vec2<f32>(e.x, 0.0));
    let gy = fl_total(p + vec2<f32>(0.0, e.y)) - fl_total(p - vec2<f32>(0.0, e.y));
    let n = normalize(vec3<f32>(-gx * 6.0, gy * 6.0, 1.0));
    let key = normalize(vec3<f32>(-0.45, 0.6, 0.65));
    let diff = clamp(dot(n, key), 0.0, 1.0);
    let hv = normalize(key + vec3<f32>(0.0, 0.0, 1.0));
    let spec = pow(clamp(dot(n, hv), 0.0, 1.0), 24.0);

    // dark palette base: a slow vertical wash, independent of the music
    var col = u.pal0.rgb * (0.45 + 0.35 * (1.0 - p.y));
    let expo = clamp(fu.look.z, 0.5, 1.5);
    var lit = hue * (0.55 + 0.6 * diff) * expo;
    lit += hue * clamp(d.w, 0.0, 1.0) * 0.35;           // heat: the bass-fed dye glows warmer for ~1.5 s
    col = mix(col, lit, cov);
    col += mix(u.pal2.rgb, vec3<f32>(1.0), 0.5) * spec * cov * 0.4 * expo;

    col = aces(col * 1.05);
    let vc = (p - vec2<f32>(0.5)) * vec2<f32>(1.0, 1.1);
    col *= 1.0 - 0.3 * smoothstep(0.3, 0.9, length(vc) * 1.25);
    col = mode_marker(col, frag, aspect, 0.62);
    // triangular dither (two uniform hashes): no banding in the dark gradients
    col += (hash21(frag.xy) + hash21(frag.xy + vec2<f32>(17.3, 41.9)) - 1.0) / 255.0;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
