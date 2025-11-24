
@group(0) @binding(0) var changeInVelocityIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var heightIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var velocityInOut: texture_storage_2d<r32float, read_write>;
@group(3) @binding(0) var fluxOut: texture_storage_2d<r32float, read_write>;


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
fn updateVelocityAndFluxY(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let vel = sampleTextures(velocityInOut, vec2u(globalIdx.x, globalIdx.y), size);
    let changeInVel = sampleTextures(changeInVelocityIn, vec2u(globalIdx.x, globalIdx.y), size);

    let newVel = vel + changeInVel * timeStep;

    let height = sampleTextures(heightIn, vec2u(globalIdx.x, globalIdx.y + upWindHeight(newVel)), size);

    let newFlux = newVel * height;

    textureStore(velocityInOut, vec2u(globalIdx.x, globalIdx.y), vec4f(newVel, 0, 0, 0));
    textureStore(fluxOut, vec2u(globalIdx.x, globalIdx.y), vec4f(newFlux, 0, 0, 0));
}