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
    if(vel < 0.0)
    {
        return 1;
    }
    else
    {
        return 0;
    }

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
fn computeInitialVelocityY(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(previousHeightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }
    let ix = i32(globalIdx.x);
    let iy = i32(globalIdx.y);

    let H_EPS : f32 = 1e-4;
    let flux = textureLoad(fluxIn, vec2i(ix, iy)).x;
    
    let prevHeight = max(textureLoad(previousHeightIn, vec2i(ix, clampI(iy + upWindHeight(flux), 0, i32(size.y) - 1))).x, H_EPS);
    let velocity = clamp(flux / prevHeight, -0.25 * gridScale / timeStep, 0.25 * gridScale / timeStep);

    textureStore(velocityOut, vec2i(ix, iy), vec4f(velocity, 0, 0, 0));
    
}