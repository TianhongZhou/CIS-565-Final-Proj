@group(0) @binding(0) var<uniform> camera : CameraUniforms;

struct HeightConsts {
  uvTexel      : vec2<f32>,
  worldScaleXY : vec2<f32>,
  heightScale  : f32,
  baseLevel    : f32
};

@group(1) @binding(0) var heightSampler : sampler;
@group(1) @binding(1) var heightTex     : texture_2d<f32>;
@group(1) @binding(2) var<uniform> hC   : HeightConsts;

struct VSIn  { @location(0) uv : vec2<f32> };
struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) worldPos   : vec3<f32>,
  @location(1) normal     : vec3<f32>,
  @location(2) uv         : vec2<f32>
};

fn H(uv: vec2<f32>) -> f32 {
  return textureSampleLevel(heightTex, heightSampler, uv, 0.0).r * hC.heightScale;
}

@vertex
fn vs_main(v: VSIn) -> VSOut {
  let h = H(v.uv);

  let world = vec3<f32>(
    (v.uv.x * 2.0 - 1.0) * hC.worldScaleXY.x,
    hC.baseLevel + h,
    (v.uv.y * 2.0 - 1.0) * hC.worldScaleXY.y
  );

  let du = vec2<f32>(hC.uvTexel.x, 0.0);
  let dv = vec2<f32>(0.0, hC.uvTexel.y);
  let hx = H(v.uv + du) - H(v.uv - du);
  let hz = H(v.uv + dv) - H(v.uv - dv);
  let dx = 2.0 * hC.worldScaleXY.x * hC.uvTexel.x;
  let dz = 2.0 * hC.worldScaleXY.y * hC.uvTexel.y;
  let n  = normalize(vec3<f32>(-hx / dx, 1.0, -hz / dz));

  var o : VSOut;
  o.clip     = camera.viewProjMat * vec4<f32>(world, 1.0);
  o.worldPos = world;
  o.normal   = n;
  o.uv       = v.uv;
  return o;
}