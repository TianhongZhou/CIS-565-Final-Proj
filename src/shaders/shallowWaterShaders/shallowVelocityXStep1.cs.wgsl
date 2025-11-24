@group(0) @binding(0) var velocityXIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var fluxXIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var heightIn: texture_storage_2d<r32float, read_write>;
@group(3) @binding(0) var changeInVelocityXOut: texture_storage_2d<r32float, read_write>;


@group(4) @binding(0) var<uniform> timeStep: f32;
@group(4) @binding(1) var<uniform> gridScale: f32;


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
fn shallowVelocityXStep1(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }
    
    let velXRight = sampleTextures(velocityXIn, vec2u(globalIdx.x, globalIdx.y), size);
    let velXLeft = sampleTextures(velocityXIn, vec2u(globalIdx.x - 1, globalIdx.y), size);


    let fluxXRight = sampleTextures(fluxXIn, vec2u(globalIdx.x, globalIdx.y), size);
    let fluxXLeft = sampleTextures(fluxXIn, vec2u(globalIdx.x - 1, globalIdx.y), size);
    let fluxXCenter = (fluxXLeft + fluxXRight) / 2.0;

    let heightRight = sampleTextures(heightIn, vec2u(globalIdx.x + upWindHeight(velXRight), globalIdx.y), size);
    let changeInVelocity = -(fluxXCenter * (velXRight - velXLeft)) / (gridScale * heightRight);
    
    //Assuming this is the first step. Subsequent steps will add to this value, so will need to both read and write.
    textureStore(changeInVelocityXOut, vec2u(globalIdx.x, globalIdx.y), vec4f(changeInVelocity, 0, 0, 0));
    

}