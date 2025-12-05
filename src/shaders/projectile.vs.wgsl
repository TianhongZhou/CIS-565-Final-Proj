// Vertex shader for projectile spheres.
@group(0) @binding(0) var<uniform> camera : CameraUniforms;

struct ProjectileUniforms {
  modelMat : mat4x4<f32>,
  color    : vec3<f32>,
  _pad     : f32,
};

@group(1) @binding(0) var<uniform> projUbo : ProjectileUniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) worldPos   : vec3<f32>,
  @location(1) normal     : vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) nor: vec3<f32>,
    @location(2) uv : vec2<f32> // unused, matches layout
) -> VSOut {
  let worldPos = (projUbo.modelMat * vec4<f32>(pos, 1.0)).xyz;
  let normal   = normalize((projUbo.modelMat * vec4<f32>(nor, 0.0)).xyz);
  var o: VSOut;
  o.clip = camera.viewProjMat * vec4<f32>(worldPos, 1.0);
  o.worldPos = worldPos;
  o.normal = normal;
  return o;
}
