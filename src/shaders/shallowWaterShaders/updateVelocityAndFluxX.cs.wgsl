
@group(0) @binding(0) var changeInVelocityInt: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var heightIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var velocityInOut: texture_storage_2d<r32float, read_write>;
@group(3) @binding(0) var fluxOut: texture_storage_2d<r32float, read_write>;


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

#compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn updateVelocityAndFluxY(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let vel = textureLoad(velocityInOut, vec2u(globalIdx.x, globalIdx.y)).x;
    let changeInVel = textureLoad(changeInVelocity, vec2u(globalIdx.x, globalIdx.y)).x;

    let newVel = vel + changeInVel * timeStep;

    let height = textureLoad(heightIn, vec2u(globalIdx.x, globalIdx.y + upWindHeight(newVel)))

    let newFlux = newVel * height;

    textureStore(velocityInOut, vec2u(globalIdx.x, globalIdx.y), vec4f(newVel, 0, 0, 0));
    textureStore(fluxOut, vec2u(globalIdx.x, globalIdx.y), vec4f(newFlux, 0, 0, 0))
}