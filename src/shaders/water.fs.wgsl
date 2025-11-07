struct FSIn {
  @location(0) worldPos : vec3<f32>,
  @location(1) normal   : vec3<f32>,
  @location(2) uv       : vec2<f32>
};

@fragment
fn fs_main(in: FSIn) -> @location(0) vec4<f32> {
  let N = normalize(in.normal);
  let color = 0.5 * (N + 1.0);
  return vec4<f32>(color, 1.0);
}