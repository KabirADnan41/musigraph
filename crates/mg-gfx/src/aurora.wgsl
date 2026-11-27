// Scene 2 "Aurora": four layered light ribbons over a dark ridge. No bars, no horizon, no water: the music
// lives in the ribbons' height, undulation and brightness. Composition keeps the lower-left clear for the
// artwork corner layout. All audio-driven motion is integrated on the CPU (ex0 = ribbon phases in cycles) so
// speed changes never make the shader jump. Onsets send a soft travelling front along the ribbons (never a
// flash), bounded and decaying. Uses the shared prelude (common.wgsl).

fn ribbon(i: i32, xw: f32, p: vec2<f32>, glow: f32, swell: f32) -> vec3<f32> {
    let fi = f32(i);
    let band = u.ex1[i];
    let ph = u.ex0[i] * 6.2831853;
    let k = 5.0 + 1.6 * fi;
    let wave = sin(k * xw + ph) + 0.5 * sin(2.3 * k * xw - ph);
    let yc = 0.60 - 0.085 * fi - (0.035 + 0.07 * band) * wave;
    let d = yc - p.y;                                   // > 0 above the ribbon's lower edge
    let spread = 0.05 + 0.10 * band;
    // sharp lower edge, long soft tail upward
    let prof = smoothstep(-0.012, 0.0, d) * exp(-max(d, 0.0) / spread);
    let rays = 0.6 + 0.4 * vnoise(vec2<f32>(xw * 26.0, fi * 7.0));
    // onset fronts travel along the ribbons and fade out; amplitude is already bounded by the mapper
    let f1 = u.c.y * exp(-u.c.x * 2.0) * exp(-sq((xw - (u.ex3.y + 0.12 * u.c.x)) / 0.09));
    let f2 = u.c.w * exp(-u.c.z * 2.0) * exp(-sq((xw - (fract(u.ex3.y + 0.37) + 0.12 * u.c.z)) / 0.09));
    let front = min(f1 + f2, 0.2);
    var gain = 0.20 + 0.45 * glow + 0.8 * front;
    if (i == 0) { gain += 0.30 * swell; }
    let intensity = prof * rays * gain * (0.35 + 0.65 * u.energy);
    var c = mix(u.pal2.rgb, u.pal1.rgb, clamp(d * 8.0, 0.0, 1.0));
    c = mix(c, u.pal3.rgb, smoothstep(0.03, 0.22, d));
    c = mix(c, mix(u.pal2.rgb, u.pal3.rgb, 0.5), 0.2 * fi / 3.0);
    return c * intensity;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    if (u.ex2.x >= 1.0) { return vec4<f32>(bridge_colour(frag.xy), 1.0); }
    let aspect = u.res.x / u.res.y;
    let p = frag.xy / u.res;                    // 0..1, y down
    let t = u.time;
    let glow = u.a.x;
    let warp = u.a.z;
    let swell = u.a.w;
    let scroll = vec2<f32>(u.b.y, u.b.z);

    // night sky: dark, tinted by the album palette
    var col = u.pal0.rgb * mix(0.03, 0.20, smoothstep(0.1, 0.9, p.y)) * keep_atmo();

    // one shared domain warp bends all ribbons together
    let w = fbm(vec2<f32>(p.x * aspect * 1.2, p.y * 0.6) + scroll) - 0.5;
    let xw = p.x + 0.18 * w * (0.5 + warp);
    var total = vec3<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        total += ribbon(i, xw, p, glow, swell);
    }
    let lum = dot(total, vec3<f32>(0.3333));

    // sparse stars, slow twinkle (<= 0.1 Hz), hidden behind bright ribbons
    let sg = vec2<f32>(p.x * aspect, p.y) * 70.0;
    let sid = floor(sg);
    let sf = fract(sg) - vec2<f32>(0.5);
    let star = step(0.992, hash21(sid)) * (1.0 - smoothstep(0.0, 0.12, length(sf)))
        * (0.5 + 0.5 * sin(t * 0.6 + hash21(sid + 3.0) * 6.2831853));
    col += mix(u.pal2.rgb, vec3<f32>(1.0), 0.6) * star * 0.5 * (1.0 - clamp(lum * 3.0, 0.0, 1.0)) * keep_part();
    col += total * keep_geom();

    // ridge silhouette
    let ridge = 0.86 + 0.03 * (vnoise(vec2<f32>(p.x * aspect * 3.0, 1.0)) - 0.5) * 2.0
        + 0.015 * (vnoise(vec2<f32>(p.x * aspect * 9.0, 4.0)) - 0.5) * 2.0;
    let land = smoothstep(ridge - 0.002, ridge + 0.002, p.y);
    col = mix(col, u.pal0.rgb * 0.012, land);

    // tone map (keeps additive ribbons from clipping to white), vignette, markers, dither
    col = vec3<f32>(1.0) - exp(-1.4 * col);
    let vc = (p - vec2<f32>(0.5)) * vec2<f32>(1.0, 1.1);
    col *= 1.0 - 0.30 * smoothstep(0.25, 0.85, length(vc) * 1.25);
    col = mode_marker(col, frag, aspect, 0.9);
    col += (hash21(frag.xy) - 0.5) / 255.0;
    col = veil_layers(col, frag.xy);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
