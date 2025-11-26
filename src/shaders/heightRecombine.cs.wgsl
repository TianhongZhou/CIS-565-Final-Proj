@group(0) @binding(0) var hPrevTex : texture_2d<f32>;
@group(0) @binding(1) var qxTex    : texture_2d<f32>;
@group(0) @binding(2) var qyTex    : texture_2d<f32>;
@group(0) @binding(3) var hSurfTex : texture_2d<f32>;
@group(0) @binding(4) var uXTex    : texture_2d<f32>;
@group(0) @binding(5) var uYTex    : texture_2d<f32>;
@group(0) @binding(6) var hOutTex  : texture_storage_2d<r32float, write>;

@group(1) @binding(0) var<uniform> dt       : f32;
@group(1) @binding(1) var<uniform> gridScale: f32;

fn loadClampX(tex: texture_2d<f32>, coord: vec2<i32>) -> f32 {
    let dims = textureDimensions(tex, 0u);   
    let x = clamp(coord.x, 0, i32(dims.x) - 1);
    let y = clamp(coord.y, 0, i32(dims.y) - 1);
    return textureLoad(tex, vec2<i32>(x, y), 0).x;
}

@compute @workgroup_size(8, 8)
fn updateHeight(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(hPrevTex, 0u);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    let hPrev = textureLoad(hPrevTex, coord, 0).x;
    let hSurf = textureLoad(hSurfTex, coord, 0).x;
    let uX    = textureLoad(uXTex,    coord, 0).x;
    let uY    = textureLoad(uYTex,    coord, 0).x;

    // extra flux: \breve q = \tilde h * \bar u
    let qxBreve = hSurf * uX;
    let qyBreve = hSurf * uY;

    // flux = q + \breve q
    let qxCenter = textureLoad(qxTex, coord, 0).x + qxBreve;
    let qyCenter = textureLoad(qyTex, coord, 0).x + qyBreve;

    let qxL = loadClampX(qxTex, coord + vec2<i32>(-1, 0)) + hSurf * loadClampX(uXTex, coord + vec2<i32>(-1, 0));
    let qxR = loadClampX(qxTex, coord + vec2<i32>( 1, 0)) + hSurf * loadClampX(uXTex, coord + vec2<i32>( 1, 0));
    let qyD = loadClampX(qyTex, coord + vec2<i32>( 0,-1)) + hSurf * loadClampX(uYTex, coord + vec2<i32>( 0,-1));
    let qyU = loadClampX(qyTex, coord + vec2<i32>( 0, 1)) + hSurf * loadClampX(uYTex, coord + vec2<i32>( 0, 1));

    let inv2dx = 1.0 / (2.0 * gridScale);
    let div = (qxR - qxL) * inv2dx + (qyU - qyD) * inv2dx;

    // ∂h/∂t + ∇·(q) = 0 → h_new = h_prev - dt * div
    let hNew = hPrev - dt * div;

    textureStore(hOutTex, coord, vec4<f32>(hNew, 0.0, 0.0, 0.0));
}