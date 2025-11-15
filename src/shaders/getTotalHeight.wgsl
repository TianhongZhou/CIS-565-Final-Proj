//The newly calculated
@group(0) @binding(0) var lowFreqIn: texture_storage_2d<r32float, read_write>;



@group(1) @binding(0) var highFreqIn: texture_storage_2d<r32float, read_write>;


@group(2) @binding(0) var heightOut: texture_storage_2d<r32float, read_write>;


@group(3) @binding(0) var<uniform> timeStep: f32;
@group(3) @binding(1) var<uniform> gridScale: f32;

@group(3) @binding(2) var terrainHeightIn: texture_storage_2d<r32float, read>;

@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn getTotalHeight(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(lowFreqIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }
    let terrainHeight = textureLoad(terrainHeightIn, vec2u(globalIdx.x, globalIdx.y)).x;
    let lowFreqHeight = textureLoad(lowFreqIn, vec2u(globalIdx.x, globalIdx.y)).x;
    let highFreqHeight = textureLoad(highFreqIn, vec2u(globalIdx.x, globalIdx.y)).x;
    
    textureStore(heightOut, vec2u(globalIdx.x, globalIdx.y), vec4f(lowFreqHeight + highFreqHeight, 0, 0, 0));

}