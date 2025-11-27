@group(0) @binding(0) var hPrevTex : texture_2d<f32>;        // h^{t+Δt/2}
@group(0) @binding(1) var qxTex    : texture_2d<f32>;        // q_x^{t+Δt} 
@group(0) @binding(2) var qyTex    : texture_2d<f32>;        // q_y^{t+Δt}
@group(0) @binding(3) var hSurfTex : texture_2d<f32>;        // ˜h^{t+Δt} 
@group(0) @binding(4) var uXTex    : texture_2d<f32>;        // bulk ū_x^{t+Δt} on faces
@group(0) @binding(5) var uYTex    : texture_2d<f32>;        // bulk ū_y^{t+Δt}
@group(0) @binding(6) var hOutTex  : texture_storage_2d<r32float, write>;

@group(1) @binding(0) var<uniform> dt       : f32;
@group(1) @binding(1) var<uniform> gridScale: f32; // Δx = Δy = gridScale

fn loadOrZero(tex : texture_2d<f32>, coord : vec2<i32>) -> f32 {
    let dims = textureDimensions(tex, 0u);
    if (coord.x < 0 || coord.y < 0 ||
        coord.x >= i32(dims.x) || coord.y >= i32(dims.y)) {
        return 0.0;
    }
    return textureLoad(tex, coord, 0).x;
}

fn upwindOffset(vel : f32) -> i32 {
    if (vel <= 0.0) {
        return 1;
    }
    return 0;
}

@compute @workgroup_size(8, 8)
fn updateHeight(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(hPrevTex, 0u);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let coord = vec2<i32>(ix, iy);

    // h^{t+Δt/2}
    let hPrev = loadOrZero(hPrevTex, coord);

    let uXRight = loadOrZero(uXTex, coord);                    // (i+1/2, j)
    let uXLeft  = loadOrZero(uXTex, coord + vec2<i32>(-1, 0)); // (i-1/2, j)
    let uYUp    = loadOrZero(uYTex, coord);                    // (i, j+1/2)
    let uYDown  = loadOrZero(uYTex, coord + vec2<i32>(0, -1)); // (i, j-1/2)

    // surface height ˜h^{t+3Δt/2} 
    let hSurfRight = loadOrZero(
        hSurfTex,
        vec2<i32>(ix + upwindOffset(uXRight), iy)
    );
    let hSurfLeft = loadOrZero(
        hSurfTex,
        vec2<i32>(ix - 1 + upwindOffset(uXLeft), iy)
    );
    let hSurfUp = loadOrZero(
        hSurfTex,
        vec2<i32>(ix, iy + upwindOffset(uYUp))
    );
    let hSurfDown = loadOrZero(
        hSurfTex,
        vec2<i32>(ix, iy - 1 + upwindOffset(uYDown))
    );

    // bulk flux q^{t+Δt}
    let qxRightBase = loadOrZero(qxTex, coord);                    // (i+1/2, j)
    let qxLeftBase  = loadOrZero(qxTex, coord + vec2<i32>(-1, 0)); // (i-1/2, j)
    let qyUpBase    = loadOrZero(qyTex, coord);                    // (i, j+1/2)
    let qyDownBase  = loadOrZero(qyTex, coord + vec2<i32>(0, -1)); // (i, j-1/2)

    // ˘q^{t+Δt} = ˜h^{t+Δt} \bar u^{t+Δt}
    let qxRight = qxRightBase + hSurfRight * uXRight;
    let qxLeft  = qxLeftBase  + hSurfLeft  * uXLeft;
    let qyUp    = qyUpBase    + hSurfUp    * uYUp;
    let qyDown  = qyDownBase  + hSurfDown  * uYDown;

    // ∇·( q^{t+Δt} + ˘q^{t+Δt} )
    let inv_dx = 1.0 / gridScale;    // dx = dy = gridScale
    let div_q  = (qxRight - qxLeft + qyUp - qyDown) * inv_dx;

    // ∂h/∂t + ∇·q = 0  =>  h_new = h_prev - dt * ∇·q
    let hNew = hPrev - dt * div_q;

    textureStore(hOutTex, coord, vec4<f32>(hNew, 0.0, 0.0, 0.0));
}