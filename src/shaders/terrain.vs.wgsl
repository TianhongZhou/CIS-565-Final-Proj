// Terrain heightmap vertex shader: displaces a grid using terrain height texture.
// Bind groups:
//  - @group(0): Camera uniforms (see common.wgsl)
//  - @group(1): terrain sampler + height texture + HeightConsts

struct HeightConsts {
  uvTexel      : vec2<f32>,
  worldScaleXY : vec2<f32>,
  heightScale  : f32,
  baseLevel    : f32
};

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(1) @binding(0) var terrainSampler : sampler;
@group(1) @binding(1) var terrainTex     : texture_2d<f32>;
@group(1) @binding(2) var<uniform> hC    : HeightConsts;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) worldPos   : vec3<f32>,
  @location(1) normal     : vec3<f32>,
  @location(2) uv         : vec2<f32>
};

fn terrainHeight(uv: vec2<f32>) -> f32 {
  return textureSampleLevel(terrainTex, terrainSampler, uv, 0.0).r * hC.heightScale + hC.baseLevel;
}

@vertex
fn vs_main(@location(0) uv : vec2<f32>) -> VSOut {
  let h = terrainHeight(uv);
  let world = vec3<f32>(
    (uv.x * 2.0 - 1.0) * hC.worldScaleXY.x,
    h,
    (uv.y * 2.0 - 1.0) * hC.worldScaleXY.y
  );

  let du = vec2<f32>(hC.uvTexel.x, 0.0);
  let dv = vec2<f32>(0.0, hC.uvTexel.y);
  let hx = terrainHeight(uv + du) - terrainHeight(uv - du);
  let hz = terrainHeight(uv + dv) - terrainHeight(uv - dv);
  let dx = 2.0 * hC.worldScaleXY.x * hC.uvTexel.x;
  let dz = 2.0 * hC.worldScaleXY.y * hC.uvTexel.y;
  let n  = normalize(vec3<f32>(-hx / dx, 1.0, -hz / dz));

  var o : VSOut;
  o.clip     = camera.viewProjMat * vec4<f32>(world, 1.0);
  o.worldPos = world;
  o.normal   = n;
  o.uv       = uv;
  return o;
}
