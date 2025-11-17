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

@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn shallowHeightStep(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let velXRight = textureLoad(velocityXIn, vec2u(globalIdx.x, globalIdx.y)).x;
    let velYUp = textureLoad(velocityYIn, vec2u(globalIdx.x, globalIdx.y)).x;

    //Will need to change this so we don't load textures outside of their bounds
    let velXLeft = textureLoad(velocityXIn, vec2u(int(globalIdx.x) - 1, globalIdx.y)).x;
    let velYDown = textureLoad(velocityYIn, vec2u(globalIdx.x, int(globalIdx.y) - 1)).x;
    
    let heightRight = textureLoad(heightIn, vec2u(globalIdx.x + upWindHeight(velXRight), globalIdx.y)).x;
    let heightLeft = textureLoad(heightIn, vec2u(globalIdx.x - 1 + upWindHeight(velXLeft), globalIdx.y)).x;

    let heightUp = textureLoad(heightIn, vec2u(globalIdx.x, globalIdx.y + upWindHeight(velYUp))).x;
    let heightDown = textureLoad(heightIn, vec2u(globalIdx.x, globalIdx.y - 1 + upWindHeight(velYDown))).x;

    let changeInHeight = -(heightRight * velXRight - heightLeft * velXLeft + heightUp * velYUp - heightDown * velYDown) / gridScale;

    let height = textureLoad(heightIn, vec2u(globalIdx.x, globalIdx.y)).x;
    let newHeight = height + changeInHeight * timeStep;

    textureStore(heightOut, vec2u(globalIdx.x, globalIdx.y), vec4f(newHeight, 0, 0, 0));

}