@group(0) @binding(0) var<uniform> camera : CameraUniforms;

// Per-water-surface constants packed into one UBO.
// uvTexel      : (1/texWidth, 1/texHeight)
// worldScaleXY : half-extent of the grid in world X/Z (final size is 2*sx by 2*sz)
// heightScale  : scales the sampled height value (amplitude)
// baseLevel    : lifts the entire surface up/down in world Y
struct HeightConsts {
  uvTexel      : vec2<f32>,
  worldScaleXY : vec2<f32>,
  heightScale  : f32,
  baseLevel    : f32
};

// Height texture + sampler (R32Float, sampled as unfilterable + level 0)
@group(1) @binding(0) var heightSampler : sampler;
@group(1) @binding(1) var heightTex     : texture_2d<f32>;
@group(1) @binding(2) var<uniform> hC   : HeightConsts;
@group(1) @binding(3) var terrainTexture: texture_2d<f32>;
// Vertex input: we only feed a regular grid of UVs (0..1)
struct VSIn  { @location(0) uv : vec2<f32> };

// Vertex output to the rasterizer and fragment stage
struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) worldPos   : vec3<f32>,
  @location(1) normal     : vec3<f32>,
  @location(2) uv         : vec2<f32>
};

// Sample water height (meters, units) at a given UV and scale it by heightScale
fn H(uv: vec2<f32>) -> f32 {
  return textureSampleLevel(heightTex, heightSampler, uv, 0.0).r * hC.heightScale;
}

// Sample terrain height at a given UV
fn T(uv: vec2<f32>) -> f32 {
  return textureSampleLevel(terrainTexture, heightSampler, uv, 0.0).r * hC.heightScale;
}

// Sample total height (water + terrain) at a given UV
fn totalHeight(uv: vec2<f32>) -> f32 {
  return H(uv);
}

@vertex
fn vs_main(v: VSIn) -> VSOut {
  // Get total height (water + terrain)
  let h = totalHeight(v.uv);
  
  // Map UV in [0,1]^2 to world XZ in [-sx,+sx]×[-sz,+sz]
  let world = vec3<f32>(
    (v.uv.x * 2.0 - 1.0) * hC.worldScaleXY.x,
    h,
    (v.uv.y * 2.0 - 1.0) * hC.worldScaleXY.y
  );
  var clipPos = camera.viewProjMat * vec4<f32>(world, 1.0);
  
  // Central differences in texture space to estimate slope
  // Step size in UV is exactly one texel in each axis
  let du = vec2<f32>(hC.uvTexel.x, 0.0);
  let dv = vec2<f32>(0.0, hC.uvTexel.y);
  
  // Height differences at +-1 texel (using total height)
  let hx = totalHeight(v.uv + du) - totalHeight(v.uv - du);
  let hz = totalHeight(v.uv + dv) - totalHeight(v.uv - dv);
  
  // Convert UV-step to world-space distances (meters) on X and Z
  let dx = 2.0 * hC.worldScaleXY.x * hC.uvTexel.x;
  let dz = 2.0 * hC.worldScaleXY.y * hC.uvTexel.y;
  
  // Build geometric normal from surface gradient; Y points up.
  // Tangents: Tx = (dx, hx, 0), Tz = (0, hz, dz).
  let n  = normalize(vec3<f32>(-hx / dx, 1.0, -hz / dz));
  
  // Pack outputs
  var o : VSOut;
  o.clip     = clipPos;
  o.worldPos = world;
  o.normal   = n;
  o.uv       = v.uv;
  return o;
}