struct Heights {
    height: f32,
    terrain: f32,
    totalHeight: f32
}




@group(0) @binding(0) var diffusionIn: texture_storage_2d<r32float, read_write>;
@group(1) @binding(0) var lowFreqOut: texture_storage_2d<r32float, read_write>;

@group(2) @binding(0) var<uniform> timeStep: f32;
@group(2) @binding(1) var<uniform> gridScale: f32;
@group(2) @binding(2) var terrainHeightIn: texture_storage_2d<r32float, read>;


//Note: pos will never be less than 0, so there's a chance that the boundary isn't being handled correctly
//However, since I'm subtracting 1, the theoretical behavior should make the int wrap, in which case it will for sure be higher than the size.
fn loadField(tex: texture_storage_2d<r32float, read_write>, coord: vec2<i32>) -> f32 {
    let size = textureDimensions(tex);
    let x = clamp(coord.x, 0, i32(size.x) - 1);
    let y = clamp(coord.y, 0, i32(size.y) - 1);
    return textureLoad(tex, vec2<u32>(u32(x), u32(y))).x;
}

fn loadTerrain(coord: vec2<i32>) -> f32 {
    let size = textureDimensions(terrainHeightIn);
    let x = clamp(coord.x, 0, i32(size.x) - 1);
    let y = clamp(coord.y, 0, i32(size.y) - 1);
    return textureLoad(terrainHeightIn, vec2<u32>(u32(x), u32(y))).x;
}

fn calculateHeights(coord: vec2<i32>) -> Heights {
    var h: Heights;
    let hVal = loadField(diffusionIn, coord);
    let tVal = loadTerrain(coord);
    h.height = hVal;
    h.terrain = tVal;
    h.totalHeight = hVal + tVal;
    return h;
}

fn calculateAlpha(pos: vec3u, height: f32, deltaHeight: f32) -> f32 {
    var alpha: f32;
    let size = textureDimensions(diffusionIn);
    if(pos.x >= size.x || pos.y >= size.y || pos.x < 0 || pos.y < 0) {
        alpha = 0.0;
       
    }
    else {
        let e = 2.718281828459045;
        let d = 0.01;
        alpha = (height * height) / 64.0 * pow(e, -d * deltaHeight * deltaHeight);
        //alpha = (height * height) / 64.0;
    }
    return alpha;
}

fn new_calculateAlpha(pos: vec3u, height: f32, deltaHeightX: f32, deltaHeightY: f32) -> f32 {
    var alpha: f32;
    let size = textureDimensions(diffusionIn);
    if(pos.x >= size.x || pos.y >= size.y || pos.x < 0 || pos.y < 0) {
        alpha = 0.0;
       
    }
    else {
        let e = 2.718281828459045;
        let d = 0.01;
        let norm = deltaHeightX * deltaHeightX + deltaHeightY + deltaHeightY;
        alpha = (height * height) / 64.0 * pow(e, -d * norm);
    }
    return alpha;
}

@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn diffuse(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(diffusionIn);
    if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
        return;
    }
    
    //let tileIdx = globalIdx.x + globalIdx.y * ${threadsInDiffusionBlockX};
    
    var heightsCenter: Heights;
    var heightsLeft: Heights;
    var heightsRight: Heights;
    var heightsUp: Heights;
    var heightsDown: Heights;

    let ix = i32(globalIdx.x);
    let iy = i32(globalIdx.y);

    heightsCenter = calculateHeights(vec2i(ix, iy));
    heightsLeft = calculateHeights(vec2i(ix - 1, iy));
    heightsRight = calculateHeights(vec2i(ix + 1, iy));
    heightsUp = calculateHeights(vec2i(ix, iy + 1));
    heightsDown = calculateHeights(vec2i(ix, iy - 1));

    //Right side of diffusion Equation
    //Note: If grid is now square, this will need to be updated
    var finiteDifferenceHeightX = (heightsRight.totalHeight + heightsLeft.totalHeight - 2.0 * heightsCenter.totalHeight) / (gridScale * gridScale);
    var finiteDifferenceHeightY = (heightsUp.totalHeight + heightsDown.totalHeight - 2.0 * heightsCenter.totalHeight) / (gridScale * gridScale);

    //Centered Difference Calculation
    var deltaHeightx = (heightsRight.totalHeight - heightsLeft.totalHeight) / (2.0 * gridScale);
    var deltaHeighty = (heightsUp.totalHeight - heightsDown.totalHeight) / (2.0 * gridScale);

    var xAlpha =  calculateAlpha(globalIdx, heightsCenter.totalHeight, deltaHeightx);
    var yAlpha =  calculateAlpha(globalIdx, heightsCenter.totalHeight, deltaHeighty);
    // var alpha = new_calculateAlpha(globalIdx, heightsCenter.Height, deltaHeightx, deltaHeighty);

    let diffusedHeight = heightsCenter.totalHeight + timeStep * (xAlpha * finiteDifferenceHeightX + yAlpha * finiteDifferenceHeightY);
    // let diffusedHeight = heightsCenter.totalHeight + timeStep * alpha * (finiteDifferenceHeightX + finiteDifferenceHeightY);

    let lowFreq = diffusedHeight - heightsCenter.terrain;

    textureStore(lowFreqOut, vec2u(globalIdx.x, globalIdx.y), vec4f(lowFreq, 0, 0, 0));
    

}
