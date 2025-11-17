@group(0) @binding(0) var previousHeightIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var fluxIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var velocityOut: texture_storage_2d<r32float, read_write>;

//It says we should upwind height, but we don't have the velocity yet, so I think we can just use the flux?
fn upWindHeight(vel: f32) -> int {
    if(vel <= 0.0)
    {
        return 1;
    }
    else
    {
        return 0;
    }

}


#compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn computeInitialVelocityY(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(previousHeightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let flux = textureLoad(fluxIn, vec2u(globalIdx.x, globalIdx.y)).x;
    let prevHeight = textureLoad(previousHeightIn, vec2u(globalIdx.x, globalIdx.y + upWindHeight(flux))).x;
    

    textureStore(velocityOut, vec2u(globalIdx.x, globalIdx.y), vec4f(flux / prevHeight, 0, 0, 0));
    
}