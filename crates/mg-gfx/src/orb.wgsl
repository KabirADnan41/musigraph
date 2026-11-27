// Scene 2 "Orb": one solid, softly lit sphere (raymarched, <= 24 steps) over a palette-coloured atmosphere.
// No lines, ribbons or rings. Audio-driven values all come from the CPU mapper (never invented here):
//   ex1 = (bass, mid, treble, breath), each smoothed on the CPU
//   ex0 = (flow phase, rotation phase [cycles, integrated on the CPU], ripple 0 / 1 impact azimuth 0..1)
//   c   = onset ripples (age s, amplitude 0..0.15 = 0.15 * onset strength), at most two alive
// bass -> displacement amplitude + slow glow breath, mid -> flow/rotation speed (CPU), treble -> fine detail and
// sparse slow glints (<= 0.4 Hz envelope), onset -> one soft surface ripple (0.9 s, eased in, never a size pop).
// The sphere radius itself is constant; displacement is zero-mean. Uses the shared prelude (common.wgsl).

const ORB_CENTRE = vec2<f32>(0.57, 0.44);   // uv, y down: right of centre, clear of the bottom-left corner layout
const ORB_RADIUS = 0.22;                    // screen heights
const RIPPLE_LIFE = 0.9;

fn h31(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.zyx + 31.32);
    return fract((q.x + q.y) * q.z);
}

fn vn3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let a = mix(mix(h31(i), h31(i + vec3<f32>(1.0, 0.0, 0.0)), w.x),
                mix(h31(i + vec3<f32>(0.0, 1.0, 0.0)), h31(i + vec3<f32>(1.0, 1.0, 0.0)), w.x), w.y);
    let b = mix(mix(h31(i + vec3<f32>(0.0, 0.0, 1.0)), h31(i + vec3<f32>(1.0, 0.0, 1.0)), w.x),
                mix(h31(i + vec3<f32>(0.0, 1.0, 1.0)), h31(i + vec3<f32>(1.0, 1.0, 1.0)), w.x), w.y);
    return mix(a, b, w.z);
}

// Noise domain: rotated by the CPU-integrated rotation phase and carried along by the flow phase.
fn orb_domain(p: vec3<f32>) -> vec3<f32> {
    let a = u.ex4.y * 6.2831853;
    let ca = cos(a);
    let sa = sin(a);
    let r = vec3<f32>(p.x * ca + p.z * sa, p.y, -p.x * sa + p.z * ca);
    return r * 1.25 + u.ex4.x * vec3<f32>(0.37, 0.61, 0.23);
}

// Height of one onset ripple on the unit sphere: a soft wave travelling away from the impact direction.
fn ripple(n: vec3<f32>, az: f32, age: f32, amp: f32) -> f32 {
    let life = age / RIPPLE_LIFE;
    if (amp <= 0.0005 || life >= 1.0) { return 0.0; }
    let s = clamp(amp * 6.667, 0.0, 1.0);
    let phi = az * 6.2831853;
    let lat = (hash21(vec2<f32>(az, 3.7)) * 2.0 - 1.0) * 0.7;
    let dir = vec3<f32>(cos(lat) * cos(phi), sin(lat), -abs(cos(lat) * sin(phi)) - 0.2);
    let ang = acos(clamp(dot(n, normalize(dir)), -1.0, 1.0));
    let front = age * 3.4;
    let env = smoothstep(0.0, 0.08, age) * sq(1.0 - life);
    return 0.03 * s * env * sin((ang - front) * 7.0) * exp(-sq((ang - front) / 0.5));
}

fn orb_march_sdf(p: vec3<f32>) -> f32 {
    let n = normalize(p);
    let big = 0.015 + 0.04 * u.ex5.x;
    let h = (vn3(orb_domain(p)) - 0.5) * 2.0 * big;
    return length(p) - 1.0 - h - ripple(n, u.ex4.z, u.c.x, u.c.y) - ripple(n, u.ex4.w, u.c.z, u.c.w);
}

// Full surface (shading only): adds a domain-warped second layer and treble-driven fine detail.
fn orb_sdf(p: vec3<f32>) -> f32 {
    let n = normalize(p);
    let q = orb_domain(p);
    let wq = q + 0.5 * vec3<f32>(vn3(q * 0.9 + 7.0) - 0.5, vn3(q * 0.9 + 13.0) - 0.5, 0.0);
    let big = 0.015 + 0.04 * u.ex5.x;
    let fine = 0.004 + 0.012 * u.ex5.z;
    let h = (vn3(wq) - 0.5) * 2.0 * big + (vn3(q * 4.1 + 5.0) - 0.5) * 2.0 * fine;
    return length(p) - 1.0 - h - ripple(n, u.ex4.z, u.c.x, u.c.y) - ripple(n, u.ex4.w, u.c.z, u.c.w);
}

fn orb_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(1.0, -1.0) * 0.006;
    return normalize(e.xyy * orb_sdf(p + e.xyy) + e.yyx * orb_sdf(p + e.yyx)
                   + e.yxy * orb_sdf(p + e.yxy) + e.xxx * orb_sdf(p + e.xxx));
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let aspect = u.res.x / u.res.y;
    let p = frag.xy / u.res;
    let t = u.time;
    let glow = u.a.x;
    let haze = u.a.y;
    let bass = u.ex5.x;
    let treble = u.ex5.z;
    let breath = u.ex5.w;
    // silence breathes very slowly (0.06 Hz) and dimly; music brightens it
    let idle = 0.5 + 0.5 * sin(t * 0.35);
    let expo = (0.35 + 0.65 * glow) * (0.6 + 0.4 * u.energy) * (1.0 + 0.25 * breath + 0.05 * idle);

    // screen -> orb space (y up, orb radius = 1)
    let s = vec2<f32>((p.x - ORB_CENTRE.x) * aspect, ORB_CENTRE.y - p.y) / ORB_RADIUS;
    let r = length(s);

    // atmosphere: palette wash + soft halo that breathes with the bass
    let wash = fbm(vec2<f32>(p.x * aspect, p.y) * 1.1 + vec2<f32>(u.b.y, u.b.z) * 0.25);
    var col = mix(u.pal0.rgb, u.pal1.rgb * 0.30, wash) * 0.6 * (0.35 + 0.65 * haze);
    col += u.pal1.rgb * exp(-max(r - 1.0, 0.0) * 1.5) * (0.07 + 0.10 * glow + 0.14 * breath) * (0.5 + 0.5 * u.energy);
    col += u.pal3.rgb * exp(-r * 0.7) * 0.05 * (0.5 + 0.5 * u.energy);

    // ray vs. bounding sphere (radius 1.2), then <= 24 sphere-tracing steps inside it
    let ro = vec3<f32>(0.0, 0.0, -4.0);
    let rd = normalize(vec3<f32>(s * 0.2582, 1.0));
    let b = dot(ro, rd);
    let hh = b * b - (dot(ro, ro) - 1.44);
    if (hh > 0.0) {
        var tt = -b - sqrt(hh);
        var d = 1.0;
        for (var i = 0; i < 24; i++) {
            d = orb_march_sdf(ro + rd * tt);
            if (d < 0.004) { break; }
            tt += d * 0.85;
        }
        let ph = ro + rd * tt;
        if (d < 0.02 && length(ph) < 1.2) {
            let n = orb_normal(ph);
            let v = -rd;
            let key = normalize(vec3<f32>(-0.5, 0.55, -0.7));
            let diff = sq(clamp(dot(n, key) * 0.5 + 0.5, 0.0, 1.0));
            let fres = pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 3.0);
            let albedo = mix(u.pal1.rgb, u.pal3.rgb, smoothstep(-0.7, 0.8, n.y + 0.3 * n.x));
            var lit = albedo * (0.10 + 0.95 * diff);
            lit += u.pal2.rgb * fres * (0.30 + 0.35 * glow);            // palette rim light
            lit += u.pal3.rgb * 0.18 * clamp(dot(n, vec3<f32>(0.6, -0.4, -0.3)) * 0.5 + 0.5, 0.0, 1.0) * fres; // cool fill
            // sparse specular glints: only on the treble, slow smooth envelopes (never a flicker)
            let cell = floor(normalize(orb_domain(n)) * 9.0);
            let seed = h31(cell);
            let env = pow(0.5 + 0.5 * sin(t * 2.4 + seed * 6.2831853), 6.0);
            let sparkle = step(0.9, h31(cell + 17.0)) * env;
            let spec = pow(clamp(dot(reflect(-key, n), v), 0.0, 1.0), 36.0);
            lit += mix(u.pal2.rgb, vec3<f32>(1.0), 0.6) * (spec * 0.25 + sparkle * (0.2 + 0.8 * spec) * 0.9) * treble;
            // soft ring of light on the ripple front (bounded; never a full-screen flash)
            let rf = min(u.c.y * exp(-u.c.x * 2.0) * smoothstep(0.0, 0.08, u.c.x)
                   + u.c.w * exp(-u.c.z * 2.0) * smoothstep(0.0, 0.08, u.c.z), 0.2);
            lit += u.pal2.rgb * rf * 0.6 * fres;
            // soft silhouette: grazing rays that end within 0.004..0.02 of the surface blend in (no hard shimmering edge)
            col = mix(col, lit * expo * 1.25, 1.0 - smoothstep(0.004, 0.02, d));
        }
    }

    col = aces(col * 1.05);
    let vc = (p - vec2<f32>(0.5)) * vec2<f32>(1.0, 1.1);
    col *= 1.0 - 0.34 * smoothstep(0.25, 0.85, length(vc) * 1.25);
    col = mode_marker(col, frag, aspect, 0.92);
    col += (hash21(frag.xy) - 0.5) / 255.0;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
