// Scene 3 "Album Focus": the artwork (drawn by the overlay pass) is the light source. This shader draws only
// the atmosphere around it, emanating from the artwork rectangle (ex2 = centre uv, half size in screen
// heights; ex3.z = 1 when artwork is present, otherwise a soft orb in the middle): a halo, slow angular
// rays, rounded-rectangle onset rings, two layers of orbiting motes and an ambient wash. Luminance next to
// the rectangle is capped so the artwork always stays the brightest thing. Uses the shared prelude.

fn mote_layer(q: vec2<f32>, ang: f32, rot: f32, layer: f32, seed: f32) -> f32 {
    // Polar grid with square cells: radial cell size 1/sc, and per ring an integer number of cells around the
    // circle (so the grid is seamless and motes stay round instead of stretching with the radius).
    let sc = 16.0 + 9.0 * layer;
    let rr = length(q) * sc;
    let row = floor(rr);
    let n = max(floor(6.2831853 * (row + 0.5)), 6.0);
    let g = vec2<f32>(ang / 6.2831853 * n + rot * (2.0 + layer), rr);
    var id = floor(g);
    let f = fract(g) - vec2<f32>(0.5);
    id.x = id.x - n * floor(id.x / n);
    let off = vec2<f32>(hash21(id + seed) - 0.5, hash21(id * 1.7 + seed + 11.0) - 0.5) * 0.6;
    let on = step(0.93, hash21(id + seed + 5.0));
    return on * (1.0 - smoothstep(0.0, 0.14, length(f - off))) * (0.5 + 0.5 * hash21(id + seed + 2.0));
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let aspect = u.res.x / u.res.y;
    let p = frag.xy / u.res;
    let p2 = vec2<f32>(p.x * aspect, p.y);
    let glow = u.a.x;
    let haze = u.a.y;
    let swell = u.a.w;
    let scroll = vec2<f32>(u.b.y, u.b.z);
    let rot = u.ex3.x;
    let present = u.ex3.z;

    let c = vec2<f32>(u.ex2.x * aspect, u.ex2.y);
    let h = max(u.ex2.zw, vec2<f32>(0.02));
    let d = sd_round_box(p2 - c, h, 0.012);
    let q = p2 - c;
    let ang = atan2(q.y, q.x + 0.00001);   // never atan2(0, 0)
    let gap = smoothstep(0.0, 0.012, d);               // thin dark gap: the halo never bleeds onto the edge

    // ambient wash
    let n = fbm(p2 * 1.2 + scroll * 0.25);
    var col = mix(u.pal0.rgb, u.pal1.rgb * 0.35, n) * 0.55 * (0.35 + 0.65 * haze);

    // halo breathing with the bass
    let halo = (0.15 + 0.25 * glow + 0.30 * swell) * exp(-max(d, 0.0) / (0.06 + 0.05 * swell)) * gap;
    col += u.pal1.rgb * halo;

    // slow angular rays (periodic noise on the unit circle), fading with distance
    let dir = vec2<f32>(cos(ang), sin(ang));
    let rn = vnoise(dir * 2.5 + vec2<f32>(rot * 0.7, rot * 0.4));
    col += u.pal3.rgb * (rn * rn * rn) * exp(-max(d, 0.0) / 0.35) * (0.10 + 0.20 * u.energy) * gap;

    // onset rings in the artwork's own rounded-rectangle metric (bounded amplitude, decaying)
    let r1 = u.c.y * exp(-u.c.x * 2.5) * exp(-sq((d - 0.35 * u.c.x) / 0.012));
    let r2 = u.c.w * exp(-u.c.z * 2.5) * exp(-sq((d - 0.35 * u.c.z) / 0.012));
    col += u.pal2.rgb * min(r1 + r2, 0.2) * 0.9 * gap;

    // orbiting motes, kept away from the artwork itself
    let motes = mote_layer(q, ang, rot, 0.0, 3.0) + 0.7 * mote_layer(q, ang, rot, 1.0, 9.0);
    col += mix(u.pal2.rgb, vec3<f32>(1.0), 0.5) * motes * (0.3 + 0.4 * glow) * smoothstep(0.04, 0.12, d);

    // keep the artwork the brightest thing: cap luminance near its rectangle
    let l = dot(col, vec3<f32>(0.3333));
    let cap = mix(0.4, 10.0, smoothstep(0.10, 0.20, d));
    col *= min(1.0, cap / max(l, 0.0001));
    // black behind the artwork (the overlay covers it) when it is present
    col *= mix(1.0, smoothstep(-0.004, 0.0, d), present);

    col = aces(col * 1.05);
    let vc = (p - vec2<f32>(0.5)) * vec2<f32>(1.0, 1.1);
    col *= 1.0 - 0.34 * smoothstep(0.25, 0.85, length(vc) * 1.25);
    col = mode_marker(col, frag, aspect, 0.92);
    col += (hash21(frag.xy) - 0.5) / 255.0;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
