// Fragment shader for projectile spheres.
@group(0) @binding(0) var<uniform> camera : CameraUniforms;

struct ProjectileUniforms {
  modelMat : mat4x4<f32>,
  color    : vec3<f32>,
  _pad     : f32,
};

@group(1) @binding(0) var<uniform> projUbo : ProjectileUniforms;

struct FSIn {
  @location(0) worldPos : vec3<f32>,
  @location(1) normal   : vec3<f32>,
};

@fragment
fn fs_main(in: FSIn) -> @location(0) vec4<f32> {
  let N = normalize(in.normal);
  let L = normalize(vec3<f32>(0.4, 1.0, 0.2));
  let camPos = (camera.invViewMat * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
  let V = normalize(camPos - in.worldPos);
  let base = projUbo.color;
  let ambient = 0.2 * base;
  let diffuse = max(dot(N, L), 0.0) * base;
  let spec = pow(max(dot(reflect(-L, N), V), 0.0), 16.0) * 0.2;
  let color = ambient + diffuse + spec;
  return vec4<f32>(color, 1.0);
}
