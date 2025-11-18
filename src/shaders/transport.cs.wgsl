@group(0) @binding(0) var qIn       : texture_storage_2d<r32float, read>;
@group(0) @binding(1) var qOut      : texture_storage_2d<r32float, write>;
@group(0) @binding(2) var uXTex     : texture_storage_2d<r32float, read>;
@group(0) @binding(3) var uYTex     : texture_storage_2d<r32float, read>;

@group(1) @binding(0) var<uniform> dt : f32;
@group(1) @binding(1) var<uniform> h  : f32;        // cell size

//
// --- Utility functions ---
//

fn sample_q(x: f32, y: f32) -> f32 {
    let ix = clamp(i32(x), 0, textureDimensions(qIn).x-1);
    let iy = clamp(i32(y), 0, textureDimensions(qIn).y-1);
    return textureLoad(qIn, vec2<i32>(ix, iy), 0).x;
}

// Cubic interpolation kernel
fn cubic(w: f32) -> f32 {
    let a = -0.5;
    let w2 = w*w;
    let w3 = w2*w;
    return (a+2.0)*w3 - (a+3.0)*w2 + 1.0;
}

fn cubicInterp(f0: f32, f1: f32, f2: f32, f3: f32, t: f32) -> f32 {
    let w0 = cubic(1.0+t);
    let w1 = cubic(t);
    let w2 = cubic(1.0-t);
    let w3 = cubic(2.0-t);
    return f0*w0 + f1*w1 + f2*w2 + f3*w3;
}

// Bicubic sampling
fn bicubicSample(x: f32, y: f32) -> f32 {
    let ix = floor(x);
    let iy = floor(y);
    let tx = x - ix;
    let ty = y - iy;

    var col: array<f32,4>;
    for (var dy = -1; dy <= 2; dy = dy + 1) {
        let row0 = sample_q(ix-1.0, iy+f32(dy));
        let row1 = sample_q(ix+0.0, iy+f32(dy));
        let row2 = sample_q(ix+1.0, iy+f32(dy));
        let row3 = sample_q(ix+2.0, iy+f32(dy));
        col[u32(dy+1)] = cubicInterp(row0,row1,row2,row3,tx);
    }

    return cubicInterp(col[0], col[1], col[2], col[3], ty);
}

//
// --- MAIN ---
//

@compute @workgroup_size(8,8)
fn transport(@builtin(global_invocation_id) gid : vec3<u32>) {

    let W = textureDimensions(qIn).x;
    let H = textureDimensions(qIn).y;
    if (gid.x >= W || gid.y >= H) { return; }

    let xc = f32(gid.x) + 0.5;
    let yc = f32(gid.y) + 0.5;

    // sample face-centered velocity:
    // u_x at (i+1/2 , j)
    let ux = textureLoad(uXTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x;

    // u_y at (i , j+1/2)
    let uy = textureLoad(uYTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x;

    // Backtrace
    // TODO : may need advection with higher accuracy here
    let x_prev = xc - dt * ux / h;
    let y_prev = yc - dt * uy / h;

    // Bicubic interpolation
    let q_val = bicubicSample(x_prev, y_prev);

    textureStore(qOut, vec2<i32>(gid.xy), vec4<f32>(q_val,0,0,0));
}
