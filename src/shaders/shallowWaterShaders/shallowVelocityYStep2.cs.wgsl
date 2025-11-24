@group(0) @binding(0) var velocityYIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(1) var fluxYIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var heightIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var changeInVelocityYOut: texture_storage_2d<r32float, read_write>;


@group(1) @binding(0) var<uniform> timeStep: f32;
@group(1) @binding(1) var<uniform> gridScale: f32;


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
fn shallowVelocityYStep2(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    
    let velYUp = sampleTextures(velocityYIn, vec2u(globalIdx.x, globalIdx.y), size);
    let velYDown = sampleTextures(velocityYIn, vec2u(globalIdx.x, globalIdx.y - 1), size);
    
    let fluxYUp = sampleTextures(fluxYIn, vec2u(globalIdx.x, globalIdx.y), size);
    let fluxYDown= sampleTextures(fluxYIn, vec2u(globalIdx.x, globalIdx.y - 1), size);
    let fluxYCenter = (fluxYUp + fluxYDown) / 2.0;
    
    let heightStaggeredUp = sampleTextures(heightIn, vec2u(globalIdx.x , globalIdx.y + upWindHeight(velYUp)), size);
    let heightCenter = sampleTextures(heightIn, vec2u(globalIdx.x, globalIdx.y), size);
    let heightUp = sampleTextures(heightIn, vec2u(globalIdx.x + 1, globalIdx.y), size);


    let gravity = 9.80665;

    var changeInVelocity = -(fluxYCenter * (velYUp - velYDown)) / (gridScale * heightStaggeredUp) - gravity * (heightUp - heightCenter) / gridScale;
    
    
    

    let prevChangeInVel = textureLoad(changeInVelocityYOut, vec2u(globalIdx.x, globalIdx.y)).x;
    textureStore(changeInVelocityYOut, vec2u(globalIdx.x, globalIdx.y), vec4f(changeInVelocity + prevChangeInVel, 0, 0, 0));
    

}