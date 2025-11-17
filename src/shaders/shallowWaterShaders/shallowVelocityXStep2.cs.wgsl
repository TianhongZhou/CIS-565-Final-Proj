@group(0) @binding(0) var velocityXIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var fluxYIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var heightIn: texture_storage_2d<r32float, read_write>;
@group(3) @binding(0) var changeInVelocityXOut: texture_storage_2d<r32float, read_write>;


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
fn shallowVelocityXStep2(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    
    let velXRight = textureLoad(velocityXIn, vec2u(globalIdx.x, globalIdx.y)).x;
    let velXDownRight = textureLoad(velocityXIn, vec2u(globalIdx.x, globalIdx.y - 1));
    
    let fluxYDown = textureLoad(fluxYIn, vec2u(globalIdx.x, globalIdx.y - 1));

    
    let heightStaggeredRight = textureLoad(heightIn, vec2u(globalIdx.x + upWindHeight(velXRight), globalIdx.y));
    let heightCenter = textureLoad(heightIn, vec2u(globalIdx.x, globalIdx.y));
    let heightRight = textureLoad(heightIn, vec2u(globalIdx.x + 1, globalIdx.y));


    let gravity = 9.80665;

    var changeInVelocity = -(fluxYDown * (velXRight - velXDownRight)) / (gridScale * heightStaggeredRight) - gravity * (heightRight - heightCenter) / gridScale;
    
    
    

    let prevChangeInVel = textureLoad(changeInVelocityXOut, vec2u(globalIdx.x, globalIdx.y));
    textureStore(changeInVelocityXOut, vec2u(globalIdx.x, globalIdx.y), vec4f(changeInVelocity + prevChangeInVel, 0, 0, 0));
    

}