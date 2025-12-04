@group(0) @binding(0) var velocityYIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(1) var fluxYIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var heightIn: texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var changeInVelocityYOut: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var<uniform> timeStep: f32;
@group(1) @binding(1) var<uniform> gridScale: f32;
@group(2) @binding(0) var terrainTexture: texture_2d<f32>;

fn upWindHeight(vel: f32) -> i32 {
    if(vel <= 0.0) {
        return 1;
    } else {
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
fn shallowVelocityYStep2(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(heightIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }

    let ix = i32(globalIdx.x);
    let iy = i32(globalIdx.y);

    let velYUp = sampleFluxVel(velocityYIn, vec2i(ix, iy), size);
    let velYDown = sampleFluxVel(velocityYIn, vec2i(ix, iy - 1), size);
    let fluxYUp = sampleFluxVel(fluxYIn, vec2i(ix, iy), size);
    let fluxYDown = sampleFluxVel(fluxYIn, vec2i(ix, iy - 1), size);
    let fluxYCenter = (fluxYUp + fluxYDown) / 2.0;

    let H_EPS : f32 = 1e-4;

    let staggeredIdx = vec2i(ix, clampI(iy + i32(upWindHeight(velYUp)), 0, i32(size.y) - 1));
    let terrainStaggered = textureLoad(terrainTexture, staggeredIdx, 0).x;
    let heightStaggeredUp =
        max(textureLoad(heightIn, staggeredIdx).x + terrainStaggered, H_EPS);

    let centerIdx = vec2i(ix, iy);
    let terrainCenter = textureLoad(terrainTexture, centerIdx, 0).x;
    let heightCenter =
        max(textureLoad(heightIn, centerIdx).x + terrainCenter, H_EPS);

    let upIdx = vec2i(ix, clampI(iy + 1, 0, i32(size.y) - 1));
    let terrainUp = textureLoad(terrainTexture, upIdx, 0).x;
    let heightUp =
        max(textureLoad(heightIn, upIdx).x + terrainUp, H_EPS);

    let gravity = 9.80665;

    var changeInVelocity =
        -(fluxYCenter * (velYUp - velYDown)) / (gridScale * heightStaggeredUp)
        - gravity * (heightUp - heightCenter) / gridScale;

    let prevChangeInVel =
        textureLoad(changeInVelocityYOut, vec2u(globalIdx.x, globalIdx.y)).x;

    var newVel = changeInVelocity + prevChangeInVel;

    textureStore(changeInVelocityYOut, vec2u(globalIdx.x, globalIdx.y),
                 vec4f(newVel, 0, 0, 0));
}
