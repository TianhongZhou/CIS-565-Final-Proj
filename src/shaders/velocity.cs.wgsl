@group(0) @binding(0) var hTex  : texture_2d<f32>;   // h_low
@group(0) @binding(1) var qxTex : texture_2d<f32>;   // qx_low
@group(0) @binding(2) var qyTex : texture_2d<f32>;   // qy_low
@group(0) @binding(3) var uXTex : texture_storage_2d<r32float, write>;  // u_x
@group(0) @binding(4) var uYTex : texture_storage_2d<r32float, write>;  // u_y

struct VelConst {
    eps : f32,
};
@group(1) @binding(0) var<uniform> velConst : VelConst;

@compute @workgroup_size(8, 8)
fn updateVelocity(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(hTex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    let h  = textureLoad(hTex,  coord, 0).x;
    let qx = textureLoad(qxTex, coord, 0).x;
    let qy = textureLoad(qyTex, coord, 0).x;

    let depth = max(h, velConst.eps);

    var ux = qx / depth;
    var uy = qy / depth;

    textureStore(uXTex, coord, vec4<f32>(ux, 0.0, 0.0, 0.0));
    textureStore(uYTex, coord, vec4<f32>(uy, 0.0, 0.0, 0.0));
}