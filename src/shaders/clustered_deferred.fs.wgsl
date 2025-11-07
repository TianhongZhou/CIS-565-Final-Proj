// TODO-3: implement the Clustered Deferred G-buffer fragment shader

// This shader should only store G-buffer information and should not do any shading.
@group(${bindGroup_material}) @binding(0) var baseColorTex : texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var baseColorSmp : sampler;

struct FSIn {
    @location(0) posWorld : vec3<f32>,
    @location(1) nor      : vec3<f32>,
    @location(2) uv       : vec2<f32>,
}

struct FSOut {
    @location(0) posOut : vec4<f32>,
    @location(1) norOut : vec4<f32>,
    @location(2) albOut : vec4<f32>,
}

@fragment
fn main(in: FSIn) -> FSOut {
    var out : FSOut;

    let alb = textureSample(baseColorTex, baseColorSmp, in.uv);
    if (alb.a < 0.5) { 
        discard; 
    }

    out.posOut = vec4<f32>(in.posWorld, 1.0);
    out.norOut = vec4<f32>(normalize(in.nor), 1.0);
    out.albOut = vec4<f32>(alb.rgb, 1.0);

    return out;
}