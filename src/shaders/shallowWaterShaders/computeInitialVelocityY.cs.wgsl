@group(0) @binding(0) var previousHeightIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var fluxIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var velocityOut: texture_storage_2d<r32float, read_write>;

//It says we should upwind height, but we don't have the velocity yet, so I think we can just use the flux?
fn upWindHeight(vel: f32) -> u32 {
    if(vel <= 0.0)
    {
        return 1;
    }
    else
    {
        return 0;
    }

}
//Need to verify that the behavior here is correct, but should probably use this.
fn sampleTextures(tex: texture_storage_2d<r32float, read_write>, pos: vec2u, size: vec2u) -> f32 {
    var value: f32;
    //Note: unecessary to check for negative on unsigned ints, it's just a small precaution. Negatives should wrap around to large positive numbers.
    if(pos.x >= size.x || pos.y >= size.y || pos.x < 0 || pos.y < 0) {
        value = 0.001;
       
    }
    else
    {
       value = textureLoad(tex, vec2u(pos.x, pos.y)).x;
    }
    return value;
    

}

@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn computeInitialVelocityY(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(previousHeightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let flux = textureLoad(fluxIn, vec2u(globalIdx.x, globalIdx.y)).x;
    let prevHeight = sampleTextures(previousHeightIn, vec2u(globalIdx.x, globalIdx.y + upWindHeight(flux)), size);
    

    textureStore(velocityOut, vec2u(globalIdx.x, globalIdx.y), vec4f(flux / prevHeight, 0, 0, 0));
    
}