@group(0) @binding(0) var hBarTex   : texture_storage_2d<r32float, read>;
@group(0) @binding(1) var hHighTex  : texture_storage_2d<r32float, read>;
@group(0) @binding(2) var hOutTex   : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn recombineHeight(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(hBarTex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let hBar  = textureLoad(hBarTex,  coord).x;
    let hHigh = textureLoad(hHighTex, coord).x;

    let hTotal = hBar + hHigh;

    textureStore(hOutTex, coord, vec4<f32>(hTotal, 0.0, 0.0, 0.0));
}