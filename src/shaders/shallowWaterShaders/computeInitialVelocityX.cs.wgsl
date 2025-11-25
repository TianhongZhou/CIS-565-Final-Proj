@group(0) @binding(0) var previousHeightIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var fluxIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var velocityOut: texture_storage_2d<r32float, read_write>;


@group(3) @binding(0) var<uniform> timeStep: f32;
@group(3) @binding(1) var<uniform> gridScale: f32;

fn clampI(v: i32, a: i32, b: i32) -> i32 {
  if (v < a) { return a; }
  if (v > b) { return b; }
  return v;
}

//It says we should upwind height, but we don't have the velocity yet, so I think we can just use the flux?
fn upWindHeight(vel: f32) -> i32 {
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
fn sampleTextures(tex: texture_storage_2d<r32float, read_write>, pos: vec2i, size: vec2u, isHeight: bool) -> f32 {
    var value: f32;
    //Note: unecessary to check for negative on unsigned ints, it's just a small precaution. Negatives should wrap around to large positive numbers.
    if(pos.x >= i32(size.x) || pos.y >= i32(size.y) || pos.x < 0 || pos.y < 0) {
        if(isHeight) {
            value = 4.0;
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
fn computeInitialVelocityX(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(previousHeightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }
    let ix = i32(globalIdx.x);
    let iy = i32(globalIdx.y);


    let flux = textureLoad(fluxIn, vec2i(ix, iy)).x;
    
    let prevHeight = textureLoad(previousHeightIn, vec2i(clampI(ix + upWindHeight(flux), 0, i32(size.x) - 1), iy)).x;
    
    let velocity = clamp(flux / prevHeight, -0.5 * gridScale / timeStep, 0.5 * gridScale / timeStep);
    
    textureStore(velocityOut, vec2i(ix, iy), vec4f(velocity, 0, 0, 0));
    
}