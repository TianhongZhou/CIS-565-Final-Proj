@group(0) @binding(0) var heightIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var velocityXIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var velocityYIn: texture_storage_2d<r32float, read_write>;
@group(3) @binding(0) var heightOut: texture_storage_2d<r32float, read_write>;


@group(4) @binding(0) var<uniform> timeStep: f32;
@group(4) @binding(1) var<uniform> gridScale: f32;

//TODO: Do proper boundary check for textures


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
    return values;
    

}

@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn shallowHeightStep(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let velXRight = sampleTextures(velocityXIn, vec2u(globalIdx.x, globalIdx.y), size).x;
    let velYUp = sampleTextures(velocityYIn, vec2u(globalIdx.x, globalIdx.y), size).x;

    //Will need to change this so we don't load textures outside of their bounds
    let velXLeft = sampleTextures(velocityXIn, vec2u(int(globalIdx.x) - 1, globalIdx.y), size).x;
    let velYDown = sampleTextures(velocityYIn, vec2u(globalIdx.x, int(globalIdx.y) - 1), size).x;
    
    let heightRight = sampleTextures(heightIn, vec2u(globalIdx.x + upWindHeight(velXRight), globalIdx.y), size).x;
    let heightLeft = sampleTextures(heightIn, vec2u(globalIdx.x - 1 + upWindHeight(velXLeft), globalIdx.y), size).x;

    let heightUp = sampleTextures(heightIn, vec2u(globalIdx.x, globalIdx.y + upWindHeight(velYUp)), size).x;
    let heightDown = sampleTextures(heightIn, vec2u(globalIdx.x, globalIdx.y - 1 + upWindHeight(velYDown)), size).x;

    let changeInHeight = -(heightRight * velXRight - heightLeft * velXLeft + heightUp * velYUp - heightDown * velYDown) / gridScale;

    let height = sampleTextures(heightIn, vec2u(globalIdx.x, globalIdx.y), size).x;
    let newHeight = height + changeInHeight * timeStep;

    textureStore(heightOut, vec2u(globalIdx.x, globalIdx.y), vec4f(newHeight, 0, 0, 0));

}