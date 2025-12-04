@group(0) @binding(0) var heightIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(1) var velocityXIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var velocityYIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var heightOut: texture_storage_2d<r32float, read_write>;


@group(1) @binding(0) var<uniform> timeStep: f32;
@group(1) @binding(1) var<uniform> gridScale: f32;

@group(2) @binding(0) var terrainTexture: texture_2d<f32>;


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
fn shallowHeight(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let H_EPS : f32 = 1e-4;
    let ix = i32(globalIdx.x);
    let iy = i32(globalIdx.y);
    /*
    let velXRightIndex = vec2i(ix, iy);
    let velXRight = textureLoad(velocityXIn, velXRightIndex).x;
    let velXUpIndex = vec2i(ix, iy);
    let velYUp = textureLoad(velocityYIn, velXUpIndex).x;
    */
    let velXRight = sampleFluxVel(velocityXIn, vec2i(ix, iy), size);
    let velYUp = sampleFluxVel(velocityYIn, vec2i(ix, iy), size);

    /*
    let velXLeftIndex = vec2i(clampI(ix - 1, 0, i32(size.x) - 1), iy);
    let velXLeft = textureLoad(velocityXIn, velXLeftIndex).x;
    let velYDownIndex = vec2i(ix, clampI(iy - 1, 0, i32(size.y) - 1));

    let velYDown = textureLoad(velocityYIn, velYDownIndex).x;
    */
    let velXLeft = sampleFluxVel(velocityXIn, vec2i(ix - 1, iy), size);
    let velYDown = sampleFluxVel(velocityYIn, vec2i(ix, iy - 1), size);


    let heightRightIndex = vec2i(clampI(ix + upWindHeight(velXRight), 0, i32(size.x) - 1), iy);
    let heightLeftIndex = vec2i(clampI(ix - 1 + upWindHeight(velXLeft), 0, i32(size.x) - 1), iy);
    let heightRight = max(textureLoad(heightIn, heightRightIndex).x, H_EPS);
    let heightLeft = max(textureLoad(heightIn, heightLeftIndex).x, H_EPS);

    let heightUpIndex = vec2i(ix, clampI(iy + upWindHeight(velYUp), 0, i32(size.y) - 1));
    let heightDownIndex = vec2i(ix, clampI(iy - 1 + upWindHeight(velYDown), 0, i32(size.y) - 1));
    let heightUp = max(textureLoad(heightIn, heightUpIndex).x, H_EPS);
    let heightDown = max(textureLoad(heightIn, heightDownIndex).x, H_EPS);

    var changeInHeight = -(heightRight * velXRight - heightLeft * velXLeft + heightUp * velYUp - heightDown * velYDown) / gridScale;
    let height = textureLoad(heightIn, vec2i(ix, iy)).x;
    var newHeight = height + changeInHeight * timeStep;

    //Clamp height to be non-negative
    newHeight = max(newHeight, H_EPS);

    textureStore(heightOut, vec2i(ix, iy), vec4f(newHeight, 0, 0, 0));

}