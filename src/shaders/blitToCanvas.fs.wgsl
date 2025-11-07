@group(0) @binding(0) var blitTex : texture_2d<f32>;
@group(0) @binding(1) var blitSmp : sampler;

struct FSIn { @location(0) uv: vec2<f32> }

@fragment
fn main(in: FSIn) -> @location(0) vec4<f32> {
  let uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
  return textureSample(blitTex, blitSmp, uv);
}