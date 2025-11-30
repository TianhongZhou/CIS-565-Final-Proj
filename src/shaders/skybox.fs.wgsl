@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(1) @binding(0) var envSampler : sampler;
@group(1) @binding(1) var envTex : texture_2d<f32>;

struct FSIn {
  @location(0) ndc : vec2<f32>,
};

fn dirToEquirectUV(dir: vec3<f32>) -> vec2<f32> {
  let d = normalize(dir);
  let u = atan2(d.z, d.x) * (0.15915494) + 0.5; // 1/(2*pi)
  let v = acos(clamp(d.y, -1.0, 1.0)) * 0.31830989; // 1/pi
  return vec2<f32>(u, v);
}

@fragment
fn fs_main(in: FSIn) -> @location(0) vec4<f32> {
  // Reconstruct view-space direction from NDC
  let ndc = vec3<f32>(in.ndc, 1.0);
  let viewDirH = camera.invProjMat * vec4<f32>(ndc, 1.0);
  let viewDir = normalize(viewDirH.xyz / viewDirH.w);
  // Transform to world
  let worldDir = normalize((camera.invViewMat * vec4<f32>(viewDir, 0.0)).xyz);

  let uv = dirToEquirectUV(worldDir);
  let color = textureSample(envTex, envSampler, uv).rgb;
  return vec4<f32>(color, 1.0);
}
