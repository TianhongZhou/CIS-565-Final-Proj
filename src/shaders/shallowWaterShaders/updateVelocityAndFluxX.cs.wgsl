
@group(0) @binding(0) var changeInVelocityIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(1) var heightIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var velocityInOut: texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var fluxOut: texture_storage_2d<r32float, read_write>;


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
fn clampI(v: i32, a: i32, b: i32) -> i32 {
  if (v < a) { return a; }
  if (v > b) { return b; }
  return v;
}


//Need to verify that the behavior here is correct, but should probably use this.
fn sampleTextures(tex: texture_storage_2d<r32float, read_write>, pos: vec2i, size: vec2u, isHeight: bool) -> f32 {
    var value: f32;
    //Note: unecessary to check for negative on unsigned ints, it's just a small precaution. Negatives should wrap around to large positive numbers.
    if(pos.x >= i32(size.x) || pos.y >= i32(size.y) || pos.x < 0 || pos.y < 0) {
        if(isHeight) {
            value = 0.0;
        } else {
            value = 0.0;
        }
       
    }
    else
    {
       value = textureLoad(tex, pos).x;
    }
    return value;
    

}
@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn updateVelocityAndFluxX(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let ix = i32(globalIdx.x);
    let iy = i32(globalIdx.y);

    let vel = textureLoad(velocityInOut, vec2i(ix, iy)).x;
    let changeInVel = textureLoad(changeInVelocityIn, vec2i(ix, iy)).x;

    var newVel = vel + changeInVel * timeStep;
    newVel = clamp(newVel, -0.25 * gridScale / timeStep, 0.25 * gridScale / timeStep);


    let height = textureLoad(heightIn, vec2i(ix + i32(upWindHeight(newVel)), iy)).x;
    var newFlux = newVel * height;
    newFlux = clamp(newFlux, -0.25 * gridScale * height / timeStep, 0.25 * gridScale * height / timeStep);

    textureStore(velocityInOut, vec2u(globalIdx.x, globalIdx.y), vec4f(newVel, 0, 0, 0));
    textureStore(fluxOut, vec2u(globalIdx.x, globalIdx.y), vec4f(newFlux, 0, 0, 0));
}