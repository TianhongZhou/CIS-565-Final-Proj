// low-freq or transported low-freq q (e.g. \bar{q}^{t+Δt})
@group(0) @binding(0) var qBarTex   : texture_storage_2d<r32float, read>;
// high-freq or transported high-freq q (e.g. \tilde{q}^{t+Δt})
@group(0) @binding(1) var qHighTex  : texture_storage_2d<r32float, read>;
// output total q^{t+Δt}
@group(0) @binding(2) var qOutTex   : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn recombineFlow(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(qBarTex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    let qBar  = textureLoad(qBarTex,  coord).x;
    let qHigh = textureLoad(qHighTex, coord).x;

    let qTotal = qBar + qHigh;

    textureStore(qOutTex, coord, vec4<f32>(qTotal, 0.0, 0.0, 0.0));
}