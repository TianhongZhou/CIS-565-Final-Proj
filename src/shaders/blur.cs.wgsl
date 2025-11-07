struct BlurParams {
  texSize   : vec2<u32>,
  radius    : u32, 
  direction : u32, 
}

@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var srcSmp : sampler;
@group(0) @binding(2) var dstTex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params : BlurParams;

fn gauss(x: f32, sigma: f32) -> f32 {
  return exp(-0.5 * (x * x) / (sigma * sigma));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.texSize.x || gid.y >= params.texSize.y) { return; }

  let sizeF = vec2<f32>(params.texSize);
  let uv0   = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / sizeF;

  let radius = i32(params.radius);
  let sigma  = max(f32(params.radius), 1.0) * 0.5 + 0.5;

  var acc = vec3<f32>(0.0);
  var wsum = 0.0;

  for (var i: i32 = -radius; i <= radius; i = i + 1) {
    let w = gauss(f32(i), sigma);
    var offs = vec2<f32>(0.0);
    if (params.direction == 0u) {
        offs = vec2<f32>(f32(i) / sizeF.x, 0.0);
    } else {
        offs = vec2<f32>(0.0, f32(i) / sizeF.y);
    }

    let c = textureSampleLevel(srcTex, srcSmp, uv0 + offs, 0.0).rgb;
    acc  += c * w;
    wsum += w;
  }

  let rgb = acc / max(wsum, 1e-6);
  textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(rgb, 1.0));
}