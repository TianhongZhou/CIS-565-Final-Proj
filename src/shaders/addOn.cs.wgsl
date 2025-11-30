struct AddParams {
    size : vec2<u32>, 
    _pad : vec2<u32>, 
};

@group(0) @binding(0)
var<uniform> params : AddParams;

@group(0) @binding(1)
var baseTex : texture_storage_2d<r32float, read_write>;

@group(0) @binding(2)
var addTex  : texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let w = params.size.x;
    let h = params.size.y;

    if (gid.x >= w || gid.y >= h) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    let baseVal = textureLoad(baseTex, coord).r;
    let addVal  = textureLoad(addTex,  coord, 0).r;

    textureStore(baseTex, coord, vec4<f32>(baseVal + addVal, 0.0, 0.0, 0.0));
}