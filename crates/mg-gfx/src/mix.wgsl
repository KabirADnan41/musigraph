// Scene crossfade: blends the two freshly rendered scene targets onto the swap chain (bilinear, so it also
// performs the render-scale upscale). Only used during the ~1 s transition between two scenes.
@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var<uniform> m: vec4<f32>; // x = blend 0..1 (already eased on the CPU)

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VOut;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let a = textureSample(tex_a, smp, in.uv);
    let b = textureSample(tex_b, smp, in.uv);
    return vec4<f32>(mix(a.rgb, b.rgb, clamp(m.x, 0.0, 1.0)), 1.0);
}
