@group(0) @binding(0) var terrainIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var heightIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(0) var fluxXOut: texture_storage_2d<r32float, read_write>;
@group(3) @binding(0) var fluxYOut: texture_storage_2d<r32float, read_write>;

fn clampI(v: i32, a: i32, b: i32) -> i32 {
  if (v < a) { return a; }
  if (v > b) { return b; }
  return v;
}

@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn terrainCheck(@builtin(global_invocation_id) globalIdx: vec3u) {
    let size = textureDimensions(terrainIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(i32(globalIdx.x), i32(globalIdx.y));

    let terrainHeight = textureLoad(terrainIn, coord).x;
    let height = textureLoad(heightIn, coord).x;
    let heightUp = textureLoad(heightIn, vec2i(coord.x, clampI(coord.y + 1, 0, i32(size.y) - 1))).x;
    let heightRight = textureLoad(heightIn, vec2i(clampI(coord.x + 1, 0, i32(size.x) - 1), coord.y)).x;
    let terrainUp = textureLoad(terrainIn, vec2i(coord.x, clampI(coord.y + 1, 0, i32(size.y) - 1))).x;
    let terrainRight = textureLoad(terrainIn, vec2i(clampI(coord.x + 1, 0, i32(size.x) - 1), coord.y)).x;

    let heightMidU = 0.5 * (height + heightUp);
    let heightMidR = 0.5 * (height + heightRight);
    let terrainMidU = 0.5 * (terrainHeight + terrainUp);
    let terrainMidR = 0.5 * (terrainHeight + terrainRight);

    if(heightMidU < terrainMidU)
    {
        textureStore(fluxYOut, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
    if(heightMidR < terrainMidR)
    {
        textureStore(fluxXOut, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
    /*
    if(height < terrainHeight) {
        // Adjust height to be at least terrain height
        //textureStore(heightIn, coord, vec4<f32>(terrainHeight, 0.0, 0.0, 0.0));
        // Zero out fluxes
        textureStore(fluxXOut, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(fluxYOut, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
    */
}