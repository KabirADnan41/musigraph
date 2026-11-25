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
};
@group(0) @binding(0) var<uniform> o: O;
@group(0) @binding(1) var art_tex: texture_2d<f32>;
@group(0) @binding(2) var title_tex: texture_2d<f32>;
@group(0) @binding(3) var artist_tex: texture_2d<f32>;
@group(0) @binding(4) var smp: sampler;

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

    return vec4<f32>(rgb, a);
}
