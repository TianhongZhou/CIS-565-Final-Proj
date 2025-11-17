@group(0) @binding(0) var velocityYIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var fluxXIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var heightIn: texture_storage_2d<r32float, read_write>;
@group(3) @binding(0) var changeInVelocityYOut: texture_storage_2d<r32float, read_write>;


@group(4) @binding(0) var<uniform> timeStep: f32;
@group(4) @binding(1) var<uniform> gridScale: f32;


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

@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn shallowVelocityYStep1(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }
    
    let velYUp = textureLoad(velocityXIn, vec2u(globalIdx.x, globalIdx.y)).x;
    let velYUpLeft = textureLoad(velocityXIn, vec2u(globalIdx.x - 1, globalIdx.y)).x;

    let fluxXLeft = textureLoad(fluxXIn, vec2u(globalIdx.x - 1, globalIdx.y)).x;

    let heightUp = textureLoad(heightIn, vec2u(globalIdx.x, globalIdx.y + upWindHeight(velYUp))).x;

    let changeInVelocity = -(fluxXLeft * (velYUp - velYUpLeft)) / (gridScale * heightUp);
    
    //Assuming this is the first step. Subsequent steps will add to this value, so will need to both read and write.
    textureStore(changeInVelocityYOut, vec2u(globalIdx.x, globalIdx.y), vec4f(changeInVelocity, 0, 0, 0));
    

}