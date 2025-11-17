@group(0) @binding(0) var velocityYIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var fluxYIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var heightIn: texture_storage_2d<r32float, read_write>;
@group(3) @binding(0) var changeInVelocityYOut: texture_storage_2d<r32float, read_write>;


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
fn shallowVelocityYStep2(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    
    let velYUp = textureLoad(velocityYIn, vec2u(globalIdx.x, globalIdx.y)).x;
    let velYDown = textureLoad(velocityYIn, vec2u(globalIdx.x, globalIdx.y - 1));
    
    let fluxYUp = textureLoad(fluxYIn, vec2u(globalIdx.x, globalIdx.y));
    let fluxYDown= textureLoad(fluxYIn, vec2u(globalIdx.x, globalIdx.y - 1));
    let fluxYCenter = (fluxYUp + fluxYDown) / 2.0;
    
    let heightStaggeredUp = textureLoad(heightIn, vec2u(globalIdx.x , globalIdx.y + upWindHeight(velXRight)));
    let heightCenter = textureLoad(heightIn, vec2u(globalIdx.x, globalIdx.y));
    let heightUp = textureLoad(heightIn, vec2u(globalIdx.x + 1, globalIdx.y));


    let gravity = 9.80665;

    var changeInVelocity = -(fluxYCenter * (velYUp - velYDown)) / (gridScale * heightStaggeredUp) - gravity * (heightUp - heightCenter) / gridScale;
    
    
    

    let prevChangeInVel = textureLoad(changeInVelocityYOut, vec2u(globalIdx.x, globalIdx.y));
    textureStore(changeInVelocityXOut, vec2u(globalIdx.x, globalIdx.y), vec4f(changeInVelocity + prevChangeInVel, 0, 0, 0));
    

}