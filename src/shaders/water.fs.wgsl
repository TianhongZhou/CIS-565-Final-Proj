// Fragment-stage input interpolated from the vertex shader.
// worldPos : world-space position of the surface point (handy for debugging/lighting)
// normal   : world-space normal (already normalized in VS, but we normalize again for safety)
// uv       : original mesh UV (not used here, but useful for texturing)
struct FSIn {
  @location(0) worldPos : vec3<f32>,
  @location(1) normal   : vec3<f32>,
  @location(2) uv       : vec2<f32>
};

@fragment
fn fs_main(in: FSIn) -> @location(0) vec4<f32> {
  // Ensure the normal is unit length
  let N = normalize(in.normal);
  
  let color = 0.5 * (N + 1.0);
  return vec4<f32>(color, 1.0);
}