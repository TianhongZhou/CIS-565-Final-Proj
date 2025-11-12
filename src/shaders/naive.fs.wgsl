@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct PassFlags {
  isReflection : f32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
};

@group(3) @binding(0) var<uniform> passFlags : PassFlags;

struct FragmentInput
{
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f
{
    // If reflection pass, discard face below water surface
    let isReflection = (passFlags.isReflection > 0.5);
    if (isReflection && (in.pos.y < ${water_base_level})) {
        discard;
    }

    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv);
    if (diffuseColor.a < 0.5f) {
        discard;
    }

    var finalColor = diffuseColor.rgb;
    return vec4(finalColor, 1);
}
