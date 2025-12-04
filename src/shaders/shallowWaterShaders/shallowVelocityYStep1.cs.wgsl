@group(0) @binding(0) var velocityYIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(1) var fluxXIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var heightIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var changeInVelocityYOut: texture_storage_2d<r32float, read_write>;


@group(1) @binding(0) var<uniform> timeStep: f32;
@group(1) @binding(1) var<uniform> gridScale: f32;


fn upWindHeight(vel: f32) -> i32 {
    if(vel < 0.0)
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


fn sampleFluxVel(tex: texture_storage_2d<r32float, read_write>, pos: vec2i, size: vec2u) -> f32 {
    var value: f32;
    
    if(pos.x >= i32(size.x) || pos.y >= i32(size.y) || pos.x < 0 || pos.y < 0) {
        value = 0.0;
        return value;
    }
    
    let clampedPos = vec2i(clampI(pos.x, 0, i32(size.x) - 1), clampI(pos.y, 0, i32(size.y) - 1));

    value = textureLoad(tex, clampedPos).x;
    
    return value;
    

}



@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn shallowVelocityYStep1(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }
    
    let ix = i32(globalIdx.x);
    let iy = i32(globalIdx.y);
    /*
    let velYUp = textureLoad(velocityYIn, vec2i(ix, iy)).x;
    let velYUpLeft = textureLoad(velocityYIn, vec2i(clampI(ix - 1, 0, i32(size.x) - 1), iy)).x;

    let fluxXLeft = textureLoad(fluxXIn, vec2i(clampI(ix - 1, 0, i32(size.x) - 1), iy)).x;
    */
    let velYUp = sampleFluxVel(velocityYIn, vec2i(ix, iy), size);
    let velYUpLeft = sampleFluxVel(velocityYIn, vec2i(ix - 1, iy), size);
    let fluxXLeft = sampleFluxVel(fluxXIn, vec2i(ix - 1, iy), size);

    let H_EPS : f32 = 1e-4;
    let heightUp = max(textureLoad(heightIn, vec2i(ix, clampI(iy + i32(upWindHeight(velYUp)), 0, i32(size.y) - 1))).x, H_EPS);
    let changeInVelocity = -(fluxXLeft * (velYUp - velYUpLeft)) / (gridScale * heightUp);
    
    //Assuming this is the first step. Subsequent steps will add to this value, so will need to both read and write.
    textureStore(changeInVelocityYOut, vec2u(globalIdx.x, globalIdx.y), vec4f(changeInVelocity, 0, 0, 0));
    

}