// Scene 1 "Cinematic Spectrum": original procedural shader. All music-to-motion decisions are made on the
// CPU (mg-engine scene_map); this shader only draws the parameters it is given. Resolution independent:
// everything is expressed in screen fractions and aspect-corrected where shape matters.

const SLOTS: i32 = 96;
const HMAX: f32 = 0.36;      // tallest bar, as a fraction of screen height
const X0: f32 = 0.07;
const X1: f32 = 0.93;


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
    col += mix(u.pal1.rgb, u.pal3.rgb, 0.5) * curtain * (0.05 + 0.10 * glow) * smoothstep(0.0, 0.04, env_h);
    // envelope line and halo
    col += u.pal2.rgb * (1.0 - smoothstep(0.0012, 0.0045, abs(d - env_h))) * 0.55 * smoothstep(0.004, 0.05, env_h);
    col += u.pal2.rgb * exp(-abs(d - env_h) * 22.0) * 0.20 * glow * smoothstep(0.004, 0.08, env_h);
    // bars
    if (f >= 0.0 && f < f32(SLOTS)) {
        let idx = i32(floor(f));
        let lx = abs(fract(f) - 0.5);
        // thin needle core plus a soft side halo (not a solid equaliser bar)
        let mask_x = (1.0 - smoothstep(0.07, 0.15, lx)) + 0.30 * exp(-lx * 7.0);
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

// Falling dust on a hash grid. Density comes from the real adaptive energy; the fall phase is integrated on
// the CPU. The grid is periodic in 128 cells, so wrapping the phase is seamless and the hash input stays
// small for any run length. The grid period is 128 x lanes cells, so every layer wraps seamlessly.
fn dust_layer(p: vec2<f32>, aspect: f32, cell: f32, phase: f32, lanes: f32, dens: f32, seed: f32) -> f32 {
    var g = vec2<f32>(p.x * aspect, p.y) / cell;
    g.y -= phase * 128.0 * lanes;
    var id = floor(g);
    let f = fract(g) - 0.5;
    // The grid repeats every `period` cells, and one phase unit shifts exactly `period` cells, so the wrap is seamless.
    let period = 128.0 * lanes;
    id.y = id.y - period * floor(id.y / period);
    let off = vec2<f32>(hash21(id + seed) - 0.5, hash21(id * 1.7 + seed + 11.0) - 0.5) * 0.6;
    let on = step(1.0 - dens, hash21(id + seed + 5.0));
    return on * (1.0 - smoothstep(0.0, 0.10, length(f - off))) * (0.5 + 0.5 * hash21(id + seed + 2.0));
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    if (layers_veil() >= 1.0) { return vec4<f32>(bridge_colour(frag.xy), 1.0); }
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
    haze_amt *= 1.0 + 0.25 * swell * exp(-sq((p.y - yh) / 0.15));   // bass bloom near the horizon
    let below = step(yh, p.y);
    haze_amt *= mix(1.0, 0.45, below);
    col += mix(u.pal0.rgb, mix(u.pal3.rgb, u.pal2.rgb, 0.3), 0.25 + 0.2 * u.b.x) * haze_amt * mix(0.35, 0.95, sky_t);
    // horizon glow, lifted by the bass swell
    col += u.pal0.rgb * exp(-abs(p.y - yh) * 13.0) * (0.22 + 0.55 * swell);
    // hot horizon line: breathes at idle, brightens a little on real onsets (ring amplitude), never recolours
    let onset_lift = max(u.c.y * exp(-u.c.x / 0.35), u.c.w * exp(-u.c.z / 0.35));
    col += mix(u.pal2.rgb, vec3<f32>(1.0), 0.45) * (1.0 - smoothstep(0.0008, 0.0026, abs(p.y - yh))) * (0.30 * breathe + 1.6 * onset_lift);
    col += u.pal2.rgb * exp(-abs(p.y - yh) * 70.0) * 0.10 * (0.5 + glow);

    // ---- spectrum above the horizon -----------------------------------------------------------
    let d_up = yh - p.y;
    let dens = 0.012 + 0.05 * u.energy;
    let dust = dust_layer(p, aspect, 0.022, u.pal0.w, 1.0, dens, 3.0)
             + 0.6 * dust_layer(p, aspect, 0.040, u.pal0.w, 0.25, dens * 0.8, 9.0);
    col += mix(u.pal2.rgb, vec3<f32>(1.0), 0.5) * dust * (0.12 + 0.55 * glow) * mix(1.0, 0.3, below) * keep_part();
    col += spectrum(p.x, d_up, glow) * keep_geom();

    // ---- reflection and ring ripples below the horizon -------------------------------------
    if (p.y > yh) {
        let dr = p.y - yh;
        let ring_amp = max(u.c.y * exp(-u.c.x / 0.8), u.c.w * exp(-u.c.z / 0.8));
        let ripple = 0.0022 * sin(dr * 70.0 - t * 1.6) * (0.25 + 3.0 * ring_amp);
        let fade = exp(-dr * 9.0) * 0.30;
        col += spectrum(p.x + ripple, dr * 1.15, glow * 0.6) * fade * keep_geom();
        // expanding elliptical rings on the water plane (impulses affect geometry/glow only, never colour)
        let qd = vec2<f32>((p.x - 0.5) * aspect, dr * aspect * 2.4);
        let r = length(qd);
        let r1 = u.c.x * 0.55;
        let r2 = u.c.z * 0.55;
        let ring1 = exp(-sq((r - r1) / 0.035)) * u.c.y * exp(-u.c.x / 0.8);
        let ring2 = exp(-sq((r - r2) / 0.035)) * u.c.w * exp(-u.c.z / 0.8);
        col += u.pal2.rgb * (ring1 + ring2) * 1.0 * smoothstep(0.0, 0.02, dr) * keep_geom();
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
    outc = veil_layers(outc, frag.xy);
    return vec4<f32>(outc, 1.0);
}
