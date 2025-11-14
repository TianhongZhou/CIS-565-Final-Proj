struct Heights {
    lowFreq: f32,
    highFreq: f32,
    terrain: f32,
    totalHeight: f32
}




// Ping ponged inputs and outputs (note that they have the terrain height added to them for rendering purposes right now)
@group(0) @binding(0) var lowFreqHeightIn: texture_storage_2d<r32float, read_write>;
//Commented out since we can only load 4 storage textures at a time per compute shader
//@group(0) @binding(1) var highFreqHeightIn: texture_storage_2d<r32float, read_write>; 

@group(1) @binding(0) var lowFreqHeightOut: texture_storage_2d<r32float, read_write>;
//@group(1) @binding(1) var highFreqHeightOut: texture_storage_2d<r32float, read_write>;

@group(2) @binding(0) var<uniform> timeStep: f32;
@group(2) @binding(1) var<uniform> gridScale: f32;
@group(2) @binding(2) var highFreqHeightIn: texture_storage_2d<r32float, read_write>;
@group(2) @binding(3) var terrainHeightIn: texture_storage_2d<r32float, read>;

fn calculateHeights(pos: vec2u) -> Heights {
    var heights: Heights; 
    let size = textureDimensions(lowFreqHeightIn);
    if(pos.x >= size.x || pos.y >= size.y || pos.x < 0 || pos.y < 0) {
        heights.lowFreq = 0;
        heights.highFreq = 0;
        heights.terrain = 0;
        heights.totalHeight = 0;
       
    }
    else
    {
        heights.terrain = textureLoad(terrainHeightIn, pos).x;
        heights.lowFreq = textureLoad(lowFreqHeightIn, pos).x - heights.terrain;
        heights.highFreq = textureLoad(highFreqHeightIn, pos).x - heights.terrain;
        
        heights.totalHeight = heights.lowFreq + heights.highFreq + heights.terrain;
    }
    return heights;
    

}

fn calculateAlpha(pos: vec3u, height: f32, deltaHeight: f32) -> f32 {
    var alpha: f32;
    let size = textureDimensions(lowFreqHeightIn);
    if(pos.x >= size.x || pos.y >= size.y || pos.x < 0 || pos.y < 0) {
        alpha = 0.0;
       
    }
    else {
        let e = 2.718281828459045;
        let d = 0.01;
        alpha = (height * height) / 64.0 * pow(e, -d * deltaHeight * deltaHeight);
    }
    return alpha;
}

@compute
@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn diffuse(@builtin(global_invocation_id) globalIdx: vec3u) {

    let size = textureDimensions(lowFreqHeightIn);
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
    var deltaHeightx = heightsRight.totalHeight - heightsLeft.totalHeight / (2 * gridScale);
    var deltaHeighty = heightsUp.totalHeight - heightsDown.totalHeight / (2 * gridScale);

    var xAlpha =  calculateAlpha(globalIdx, heightsCenter.totalHeight, deltaHeightx);
    var yAlpha =  calculateAlpha(globalIdx, heightsCenter.totalHeight, deltaHeighty);


    let diffusedHeight = heightsCenter.totalHeight + timeStep * (xAlpha * finiteDifferenceHeightX + yAlpha * finiteDifferenceHeightY);

    

    //lowFreqHeight - terrainHeight (+ terrainHeight)
    textureStore(lowFreqHeightOut, vec2u(globalIdx.x, globalIdx.y), vec4f(diffusedHeight, 0, 0, 0));

    
    //height - (lowFreqHeight - terrainHeight) (+ terrainHeight)
    //textureStore(highFreqHeightOut, vec2u(globalIdx.x, globalIdx.y), vec4f((heightsCenter.lowFreq + heightsCenter.highFreq) - diffusedHeight, 0, 0, 0));
    /*
    let highFreqHeight = textureLoad(highFreqHeightIn, vec2u(globalIdx.x, globalIdx.y)).x;
    textureStore(lowFreqHeightOut, vec2u(globalIdx.x, globalIdx.y), vec4f(highFreqHeight, 0, 0, 0));
    */
}
