@group(0) @binding(0) var<uniform> camera : CameraUniforms;

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
  // Normalize interpolated normal
  let N = normalize(in.normal);

  // --- Water base color ---
  let baseColor = vec3<f32>(100.0/255.0, 190.0/255.0, 255.0/255.0);

  // Directional light
  let L = normalize(vec3<f32>(0.4, 1.0, 0.2)); 
  let lightColor = vec3<f32>(1.0, 1.0, 1.0); 

  // View vector: from surface point to camera
  // camera.invViewMat * (0,0,0,1) gives camera position in world space
  let camPos = (camera.invViewMat * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
  let V = normalize(camPos - in.worldPos);

  // --- Simple Phong shading ---
  // Ambient term
  let ambientStrength = 0.2;
  let ambient = ambientStrength * baseColor;

  // Diffuse term 
  let NdotL = max(dot(N, L), 0.0);
  let diffuse = NdotL * baseColor;

  // Specular term
  let shininess = 64.0;   
  let specStrength = 0.4;

  let R = reflect(-L, N);  
  let specFactor = pow(max(dot(R, V), 0.0), shininess);
  let specular = specStrength * specFactor * lightColor;

  // Final color
  let color = ambient + diffuse + specular;
  
  // Keep water semi-transparent (for blending)
  let alpha = 0.8;
  
  return vec4<f32>(color, alpha);
}