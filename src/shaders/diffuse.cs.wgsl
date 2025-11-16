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

fn calculateHeights(pos: vec2u) -> Heights {
    var heights: Heights; 
    let size = textureDimensions(diffusionIn);
    if(pos.x >= size.x || pos.y >= size.y || pos.x < 0 || pos.y < 0) {
        heights.height = 0;
        heights.terrain = 0;
        heights.totalHeight = 0;
       
    }
    else
    {
       heights.height = textureLoad(diffusionIn, vec2u(pos.x, pos.y)).x;
       heights.terrain = textureLoad(terrainHeightIn, vec2u(pos.x, pos.y)).x;
       heights.totalHeight = heights.height + heights.terrain;
    }
    return heights;
    

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

    heightsCenter = calculateHeights(vec2u(globalIdx.x, globalIdx.y));
    heightsLeft = calculateHeights(vec2u(globalIdx.x, globalIdx.y) - vec2u(1, 0));
    heightsRight = calculateHeights(vec2u(globalIdx.x, globalIdx.y) + vec2u(1, 0));
    heightsUp = calculateHeights(vec2u(globalIdx.x, globalIdx.y) + vec2u(0, 1));
    heightsDown = calculateHeights(vec2u(globalIdx.x, globalIdx.y) - vec2u(0, 1));

    //Right side of diffusion Equation
    var finiteDifferenceHeightX = (heightsRight.totalHeight + heightsLeft.totalHeight - 2.0 * heightsCenter.totalHeight) / (gridScale * gridScale);
    var finiteDifferenceHeightY = (heightsUp.totalHeight + heightsDown.totalHeight - 2.0 * heightsCenter.totalHeight) / (gridScale * gridScale);

    //Centered Difference Calculation
    var deltaHeightx = (heightsRight.totalHeight - heightsLeft.totalHeight) / (2 * gridScale);
    var deltaHeighty = (heightsUp.totalHeight - heightsDown.totalHeight) / (2 * gridScale);

    var xAlpha =  calculateAlpha(globalIdx, heightsCenter.totalHeight, deltaHeightx);
    var yAlpha =  calculateAlpha(globalIdx, heightsCenter.totalHeight, deltaHeighty);


    let diffusedHeight = heightsCenter.totalHeight + timeStep * (xAlpha * finiteDifferenceHeightX + yAlpha * finiteDifferenceHeightY);

    //Currently adds the terrain height for rendering purposes
    let lowFreq = diffusedHeight - heightsCenter.terrain;
    let highFreq = lowFreq - heightsCenter.height;

    
    textureStore(lowFreqOut, vec2u(globalIdx.x, globalIdx.y), vec4f(lowFreq, 0, 0, 0));
    //textureStore(highFreqOut, vec2u(globalIdx.x, globalIdx.y), vec4f(highFreq, 0, 0, 0));
    
    
    /*
    let highFreqHeight = textureLoad(highFreqHeightIn, vec2u(globalIdx.x, globalIdx.y)).x;
    textureStore(lowFreqHeightOut, vec2u(globalIdx.x, globalIdx.y), vec4f(highFreqHeight, 0, 0, 0));
    */
}
