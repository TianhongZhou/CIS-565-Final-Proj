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
fn shallowVelocityYStep2(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let ix = i32(globalIdx.x);
    let iy = i32(globalIdx.y);


    let velYUp = textureLoad(velocityYIn, vec2i(ix, iy)).x;
    let velYDown = textureLoad(velocityYIn, vec2i(ix, clampI(iy - 1, 0, i32(size.y) - 1))).x;
    
    let fluxYUp = textureLoad(fluxYIn, vec2i(ix, iy)).x;
    let fluxYDown= textureLoad(fluxYIn, vec2i(ix, clampI(iy - 1, 0, i32(size.y) - 1))).x;
    let fluxYCenter = (fluxYUp + fluxYDown) / 2.0;
    
    let heightStaggeredUp = textureLoad(heightIn, vec2i(ix , clampI(iy + i32(upWindHeight(velYUp)), 0, i32(size.y) - 1))).x;
    let heightCenter = textureLoad(heightIn, vec2i(ix, iy)).x;
    let heightUp = textureLoad(heightIn, vec2i(ix, clampI(iy + 1, 0, i32(size.y) - 1))).x;


    let gravity = 9.80665;

    var changeInVelocity = -(fluxYCenter * (velYUp - velYDown)) / (gridScale * heightStaggeredUp) - gravity * (heightUp - heightCenter) / gridScale;
    
    
    

    let prevChangeInVel = textureLoad(changeInVelocityYOut, vec2u(globalIdx.x, globalIdx.y)).x;

    var newVel = changeInVelocity + prevChangeInVel;
    //Clamp to avoid extreme velocities
    //newVel = clamp(newVel, -0.25 * gridScale / timeStep, 0.25 * gridScale / timeStep);
    textureStore(changeInVelocityYOut, vec2u(globalIdx.x, globalIdx.y), vec4f(newVel, 0, 0, 0));
    

}