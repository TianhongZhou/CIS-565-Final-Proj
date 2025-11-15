import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';

export class DiffuseRenderer extends renderer.Renderer {
    sceneUniformsBindGroupLayout: GPUBindGroupLayout;
    sceneUniformsBindGroup: GPUBindGroup;

    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    pipeline: GPURenderPipeline;

    // --- Heightmap state (CPU & GPU resources) ---
    // current heightmap width & height (texels)
    private heightW = 256;
    private heightH = 256;

    // R32Float texture storing heights (1 float per texel)
    heightWaveTexture: GPUTexture;
    lowFreqWaveTexture: GPUTexture;
    highFreqWaveTexture: GPUTexture;
    heightTerrainTexture: GPUTexture;

    
    
    // non-filtering sampler
    heightSampler: GPUSampler;
    // UBO for HeightConsts (uvTexel, worldScale, heightScale, baseLevel)
    heightConsts: GPUBuffer;

    lowFreqColor: GPUBuffer;
    highFreqColor: GPUBuffer;
    terrainColor: GPUBuffer;
    // layout for { sampler, heightTex, heightConsts }
    heightBindGroupLayout: GPUBindGroupLayout;

    waveColorBindGroupLayout: GPUBindGroupLayout;
    lowFreqWaveColorBindGroup: GPUBindGroup;
    highFreqWaveColorBindGroup: GPUBindGroup;
    terrainColorBindGroup: GPUBindGroup;
    // bound view/sampler/UBO used by the water vertex shader
    lowFrequencyBindGroup: GPUBindGroup;
    highFrequencyBindGroup: GPUBindGroup;
    terrainBindGroup: GPUBindGroup;


    // pipeline that draws the displaced grid (water surface)
    heightPipeline: GPURenderPipeline;

    
    // vertex buffer storing per-vertex UVs
    heightVBO: GPUBuffer;
    // index buffer for the grid
    heightIBO: GPUBuffer;
    // how many indices to draw
    heightIndexCount = 0;


    // Compute shader resources
    diffusionComputePipeline: GPUComputePipeline;
    reconstructHeightPipeline: GPUComputePipeline;
    diffusionHeightComputeBindGroupLayout: GPUBindGroupLayout; //Ping pong low/high freq
    diffusionConstantsComputeBindGroupLayout: GPUBindGroupLayout; 
    diffusionHeightComputeBindGroup: GPUBindGroup;
    diffusionLowFreqComputeBindGroup: GPUBindGroup;
    diffusionHighFreqComputeBindGroup: GPUBindGroup;
    diffusionConstantsComputeBindGroup: GPUBindGroup;
    

    // --- Upload helpers for writeTexture() row alignment ---
    private rowBytes = 0;
    private paddedBytesPerRow = 0;
    private uploadScratch: Uint8Array | null = null;

    private lowFrequencyArray?: Float32Array; 
    private highFrequencyArray?: Float32Array; 
    private terrainArray?: Float32Array;

    private updater?: (dtSec: number, out: Float32Array) => void; 
    private _t = 0;

    constructor(stage: Stage) {
        super(stage);

        this.sceneUniformsBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "scene uniforms bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" }
                }
            ]
        });

        this.sceneUniformsBindGroup = renderer.device.createBindGroup({
            label: "scene uniforms bind group",
            layout: this.sceneUniformsBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.camera.uniformsBuffer }
                }
            ]
        });

        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        this.depthTextureView = this.depthTexture.createView();

        this.pipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                label: "naive pipeline layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    renderer.materialBindGroupLayout
                ]
            }),
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus"
            },
            vertex: {
                module: renderer.device.createShaderModule({
                    label: "naive vert shader",
                    code: shaders.naiveVertSrc
                }),
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "naive frag shader",
                    code: shaders.naiveFragSrc,
                }),
                targets: [
                    {
                        format: renderer.canvasFormat,
                    }
                ]
            }
        });

        {
            // Build a (nu × nv) regular grid in UV space. Each vertex stores (u,v) in [0,1].
            // The vertex shader converts (u,v) -> world (x,z) using worldScale and displaces y using the heightmap.
            const grid = makeGrid(512, 512, 0);
            

            this.heightIndexCount = grid.indices.length;

            this.heightVBO = renderer.device.createBuffer({
                size: grid.uvs.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            renderer.device.queue.writeBuffer(this.heightVBO, 0, grid.uvs);
            

            this.heightIBO = renderer.device.createBuffer({
                size: grid.indices.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
            });
            renderer.device.queue.writeBuffer(this.heightIBO, 0, grid.indices);
        }

        // R32Float height texture (unfilterable). One float per texel.
        this.lowFreqWaveTexture = renderer.device.createTexture({
            size: [256, 256], 
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
        });

        this.highFreqWaveTexture = renderer.device.createTexture({
            size: [256, 256], 
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
        });

        this.heightTerrainTexture = renderer.device.createTexture({
            size: [256, 256], 
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
        });
        this.heightWaveTexture = renderer.device.createTexture({
            size: [256, 256],
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
        });


        // Non-filtering sampler (required for unfilterable formats like R32Float).
        this.heightSampler = renderer.device.createSampler({
            minFilter: "nearest",
            magFilter: "nearest",
            mipmapFilter: "nearest",
        });

        // UBO for HeightConsts (32 bytes total, 16-byte aligned):
        // layout: [uvTexel(8B), worldScale(8B), heightScale(4B), baseLevel(4B), padding(8B)]
        this.heightConsts = renderer.device.createBuffer({
            size: 4*8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        
        /*
        Defaults for HeightConsts:
        0 = 1/width        (uvTexel.x)
        1 = 1/height       (uvTexel.y)
        2 = worldScale.x   (half extent in world X; final width = 2*sx)
        3 = worldScale.y   (half extent in world Z; final depth = 2*sz)
        4 = heightScale    (amplitude multiplier)
        5 = baseLevel      (world Y offset)
        6..7 = unused (padding)
        */
        const defaults = new Float32Array([1/256,1/256, 500,500, 10, 0, 0, 0]);
        renderer.device.queue.writeBuffer(this.heightConsts, 0, defaults);

        // Bind group layout for { sampler, heightTex, heightConsts } used by the water VS.
        // Note: sampleType: "unfilterable-float" + sampler type "non-filtering" for R32Float.
        this.heightBindGroupLayout = renderer.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, sampler: { type: "non-filtering" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "unfilterable-float" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            ]
        });

        // Concrete bind group connecting the sampler, the texture view, and the UBO.
        this.lowFrequencyBindGroup = renderer.device.createBindGroup({
            layout: this.heightBindGroupLayout,
            entries: [
                { binding: 0, resource: this.heightSampler },
                { binding: 1, resource: this.lowFreqWaveTexture.createView() },
                { binding: 2, resource: { buffer: this.heightConsts } },
            ]
        });
        
        this.highFrequencyBindGroup = renderer.device.createBindGroup({
            layout: this.heightBindGroupLayout,
            entries: [
                { binding: 0, resource: this.heightSampler },
                { binding: 1, resource: this.highFreqWaveTexture.createView() },
                { binding: 2, resource: { buffer: this.heightConsts } },
            ]
        });

        this.terrainBindGroup = renderer.device.createBindGroup({
            layout: this.heightBindGroupLayout,
            entries: [
                { binding: 0, resource: this.heightSampler },
                { binding: 1, resource: this.heightTerrainTexture.createView() },
                { binding: 2, resource: { buffer: this.heightConsts } },
            ]
        });


        this.waveColorBindGroupLayout = renderer.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });
        this.lowFreqColor = renderer.device.createBuffer({
            size: 3 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        renderer.device.queue.writeBuffer(this.lowFreqColor, 0, new Float32Array([1.0, 0.3, 0.6])); //Low frequency wave color
        this.lowFreqWaveColorBindGroup = renderer.device.createBindGroup({
            layout: this.waveColorBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.lowFreqColor } },
            ]
        });

        this.highFreqColor = renderer.device.createBuffer({
            size: 3 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        renderer.device.queue.writeBuffer(this.highFreqColor, 0, new Float32Array([0.2, 0.5, 1.0])); //High frequency wave color
        this.highFreqWaveColorBindGroup = renderer.device.createBindGroup({
            layout: this.waveColorBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.highFreqColor } },
            ]
        });

        this.terrainColor = renderer.device.createBuffer({
            size: 3 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        renderer.device.queue.writeBuffer(this.terrainColor, 0, new Float32Array([0.3, 0.8, 0.3])); //Terrain color
        this.terrainColorBindGroup = renderer.device.createBindGroup({
            layout: this.waveColorBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.terrainColor } },
            ]
        });

        // Pipeline used to render the water surface (height-displaced grid).
        // Vertex shader samples the height texture at UV, computes world position & normal.
        // Fragment shader can do lighting.
        this.heightPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    // camera UBO
                    this.sceneUniformsBindGroupLayout, 
                    // height sampler/texture/UBO
                    this.heightBindGroupLayout,
                    this.waveColorBindGroupLayout    
                ]
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.waterVertSrc }),
                entryPoint: "vs_main",
                buffers: [ heightVertexLayout ],
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.waterFragSrc }),
                entryPoint: "fs_main",
                targets: [{ format: renderer.canvasFormat }],
            },
            depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
            primitive: { topology: "triangle-list" }
        });

        //Compute shader setup

        this.diffusionHeightComputeBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "diffusion low/high freq compute bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: {format: "r32float", access: "read-write", viewDimension: "2d"} },
                //{ binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: {format: "r32float", access: "read-write", viewDimension: "2d"} }
            ]
        });

        this.diffusionConstantsComputeBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "diffusion constants compute bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: {format: "r32float", access: "read-only", viewDimension: "2d"} }
            ]
        });

        this.diffusionHeightComputeBindGroup = renderer.device.createBindGroup({
            layout: this.diffusionHeightComputeBindGroupLayout,
            entries: [
                { binding: 0, resource: this.heightWaveTexture.createView() }
            ]
        });
        this.diffusionLowFreqComputeBindGroup = renderer.device.createBindGroup({
            layout: this.diffusionHeightComputeBindGroupLayout,
            entries: [
                { binding: 0, resource: this.lowFreqWaveTexture.createView() }
            ]
        });
        this.diffusionHighFreqComputeBindGroup = renderer.device.createBindGroup({
            layout: this.diffusionHeightComputeBindGroupLayout,
            entries: [
                { binding: 0, resource: this.highFreqWaveTexture.createView() }
            ]
        });
        
                
                


        const timeStepBuffer = renderer.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const gridScaleBuffer = renderer.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        renderer.device.queue.writeBuffer(timeStepBuffer, 0, Float32Array.from([0.25])); //Currently defaulting the time step to 16ms
        renderer.device.queue.writeBuffer(gridScaleBuffer, 0, Float32Array.from([1.0]));
        this.diffusionConstantsComputeBindGroup = renderer.device.createBindGroup({
            label: "diffusion constants compute bind group",
            layout: this.diffusionConstantsComputeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: timeStepBuffer } },
                { binding: 1, resource: { buffer: gridScaleBuffer } },
                { binding: 2, resource: this.heightTerrainTexture.createView() }
            ]
        });
        
        this.diffusionComputePipeline = renderer.device.createComputePipeline({
            label: "diffusion compute pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.diffusionHeightComputeBindGroupLayout,
                    this.diffusionHeightComputeBindGroupLayout,
                    this.diffusionHeightComputeBindGroupLayout,
                    this.diffusionConstantsComputeBindGroupLayout
                ]
            }),
            compute: {
                module: renderer.device.createShaderModule({
                    label: "diffuse compute shader",
                    code: shaders.diffuseComputeSrc
                }),
                entryPoint: "diffuse"
            }
        });

        this.reconstructHeightPipeline = renderer.device.createComputePipeline({
            label: "reconstruct height compute pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.diffusionHeightComputeBindGroupLayout,
                    this.diffusionHeightComputeBindGroupLayout,
                    this.diffusionHeightComputeBindGroupLayout,
                    this.diffusionConstantsComputeBindGroupLayout
                ]
            }),
            compute: {
                module: renderer.device.createShaderModule({
                    label: "reconstruct height compute shader",
                    code: shaders.getTotalHeightSrc
                }),
                entryPoint: "getTotalHeight"
            }
        });



        this.setHeightParams(1, 1, 1, 0);
        //Set initial values of textures
        const dt = 0 / 1000; 
        this.initRowInfoIfNeeded();
        const terrainArr = this.terrainArray!;
        const lowFreqArr = this.lowFrequencyArray!;
        const highFreqArr = this.highFrequencyArray!;


        if (this.updater) {
            // External simulation writes into 'arr' (W*H floats)
            this.updater(dt, terrainArr);
        } else {
            this._t += dt;
            const W = this.heightW, H = this.heightH;
            const s = 0.05, w = this._t;
            for (let y = 0; y < H; y++) {
                for (let x = 0; x < W; x++) {
                terrainArr[y*W + x] = Math.sin(x*s + w) * Math.cos(y*s + 0.5*w);
                lowFreqArr[y*W + x] = 2;
                //highFreqArr[y*W + x] = terrainArr[y*W + x] + 0;
                }
            }
        }

        // Upload into the existing height texture
        //this.updateHeightInPlace(arr);
        
        this.updateTexture(terrainArr, this.heightTerrainTexture);
        this.updateTexture(lowFreqArr, this.heightWaveTexture);
        //this.updateTexture(lowFreqArr, this.lowFreqWaveTexture);
        //this.updateTexture(lowFreqArr, this.highFreqWaveTexture);
        //TODO: run compute shader 128 times, ping ponging between the two sets of textures for the low/high frequency waves
        const encoder = renderer.device.createCommandEncoder();
        for(let i = 0; i < 1280; i++) {
        
            const diffusePass = encoder.beginComputePass();
            diffusePass.setPipeline(this.diffusionComputePipeline);
            diffusePass.setBindGroup(0, this.diffusionHeightComputeBindGroup);
            diffusePass.setBindGroup(1, this.diffusionLowFreqComputeBindGroup);
            diffusePass.setBindGroup(2, this.diffusionHighFreqComputeBindGroup);
            diffusePass.setBindGroup(3, this.diffusionConstantsComputeBindGroup);
            diffusePass.dispatchWorkgroups(Math.ceil(this.heightW / shaders.constants.threadsInDiffusionBlockX), Math.ceil(this.heightH / shaders.constants.threadsInDiffusionBlockY));
            diffusePass.end();
            
            const reconstructPass = encoder.beginComputePass();
            reconstructPass.setPipeline(this.reconstructHeightPipeline);
            reconstructPass.setBindGroup(0, this.diffusionLowFreqComputeBindGroup);
            reconstructPass.setBindGroup(1, this.diffusionHighFreqComputeBindGroup);
            reconstructPass.setBindGroup(2, this.diffusionHeightComputeBindGroup);
            reconstructPass.setBindGroup(3, this.diffusionConstantsComputeBindGroup);
            reconstructPass.dispatchWorkgroups(Math.ceil(this.heightW / shaders.constants.threadsInDiffusionBlockX), Math.ceil(this.heightH / shaders.constants.threadsInDiffusionBlockY));
            reconstructPass.end();
        }
        renderer.device.queue.submit([encoder.finish()]);

    }

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = renderer.context.getCurrentTexture().createView();

        const renderPass = encoder.beginRenderPass({
            label: "naive render pass",
            colorAttachments: [
                {
                    view: canvasTextureView,
                    clearValue: [0, 0, 0, 0],
                    loadOp: "clear",
                    storeOp: "store"
                }
            ],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            }
        });
        renderPass.setPipeline(this.pipeline);

        renderPass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);

        this.scene.iterate(node => {
            renderPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            renderPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            renderPass.setVertexBuffer(0, primitive.vertexBuffer);
            renderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            renderPass.drawIndexed(primitive.numIndices);
        });

        // Add pipeline for height map in render pass
        renderPass.setPipeline(this.heightPipeline);
        renderPass.setBindGroup(0, this.sceneUniformsBindGroup);
       
        renderPass.setVertexBuffer(0, this.heightVBO);
        renderPass.setIndexBuffer(this.heightIBO, "uint32");

        
        renderPass.setBindGroup(1, this.terrainBindGroup); //Terrain
        renderPass.setBindGroup(2, this.terrainColorBindGroup);
        renderPass.drawIndexed(this.heightIndexCount);
        
        
        renderPass.setBindGroup(1, this.lowFrequencyBindGroup); //Low Frequency waves
        renderPass.setBindGroup(2, this.lowFreqWaveColorBindGroup);
        renderPass.drawIndexed(this.heightIndexCount);
        
        renderPass.setBindGroup(1, this.highFrequencyBindGroup); //High Frequency waves
        renderPass.setBindGroup(2, this.highFreqWaveColorBindGroup);
        renderPass.drawIndexed(this.heightIndexCount);
        
        renderPass.end();

        renderer.device.queue.submit([encoder.finish()]);
    }

    // Update world scale (sx, sz), height amplitude, and base level (Y offset).
    // Offsets match the HeightConsts layout described above.
    setHeightParams(sx: number, sz: number, heightScale: number, baseLevel: number) {
        renderer.device.queue.writeBuffer(this.heightConsts, 8,  new Float32Array([sx, sz]));
        renderer.device.queue.writeBuffer(this.heightConsts, 16, new Float32Array([heightScale]));
        renderer.device.queue.writeBuffer(this.heightConsts, 20, new Float32Array([baseLevel]));
    }
    setHeightUpdater(fn: (dtSec: number, out: Float32Array) => void) {
        this.updater = fn;
    }

    // Prepare per-frame upload helpers based on current (heightW, heightH).
    // Ensures we have:
    //  - rowBytes & paddedBytesPerRow computed (256-byte alignment for WebGPU)
    //  - uploadScratch allocated if padding is needed
    //  - heightArray allocated for the simulation output
    private initRowInfoIfNeeded() {
        //For writing buffers to textures, byte rows must be multiples of 256
        //when it's not, we must create a new buffer that is that size and later copy the old buffer into it (with padding)
        if (!this.rowBytes) {
            this.rowBytes = this.heightW * 4;
            const a = 256;
            this.paddedBytesPerRow = Math.ceil(this.rowBytes / a) * a;
            if (this.paddedBytesPerRow !== this.rowBytes) {
                this.uploadScratch = new Uint8Array(this.paddedBytesPerRow * this.heightH);
            }
        }
        if (!this.terrainArray) {
            this.terrainArray = new Float32Array(this.heightW * this.heightH);
        }
        if (!this.lowFrequencyArray) {
            this.lowFrequencyArray = new Float32Array(this.heightW * this.heightH);
        }
        if (!this.highFrequencyArray) {
            this.highFrequencyArray = new Float32Array(this.heightW * this.heightH);
        }
    }

    private updateTexture(heightData: Float32Array, heightTexture: GPUTexture)
    {
        const data = new Float32Array(heightData);
        this.initRowInfoIfNeeded();

        // Aligned -> upload directly
        if (this.paddedBytesPerRow === this.rowBytes) {
            renderer.device.queue.writeTexture(
                { texture: heightTexture },
                data,
                { bytesPerRow: this.rowBytes },
                [this.heightW, this.heightH, 1]
            );
        } else {
            // Not aligned -> copy row-by-row into the padded scratch buffer
            const tmp = new Uint8Array(this.uploadScratch!);
            const src = new Uint8Array(heightData.buffer, heightData.byteOffset, heightData.byteLength);
            for (let y = 0; y < this.heightH; y++) {
                //Copies the non padded buffer to the padded buffer, leaving padding before each row.
                const s = y * this.rowBytes, d = y * this.paddedBytesPerRow;
                tmp.set(src.subarray(s, s + this.rowBytes), d);
            }
            renderer.device.queue.writeTexture(
                { texture: heightTexture },
                tmp,
                { bytesPerRow: this.paddedBytesPerRow },
                [this.heightW, this.heightH, 1]
            );
        }
    }


    // Called every frame before drawing. Fills heightArray either via user-provided
    // simulation callback (setHeightUpdater) or with a demo wave when no callback is set.
    // Then uploads the new height field into the GPU texture.
    protected override onBeforeDraw(dtMs: number): void {
        /*
        const dt = dtMs / 1000; 
        this.initRowInfoIfNeeded();
        const terrainArr = this.terrainArray!;
        const lowFreqArr = this.lowFrequencyArray!;
        const highFreqArr = this.highFrequencyArray!;


        if (this.updater) {
            // External simulation writes into 'arr' (W*H floats)
            this.updater(dt, terrainArr);
        } else {
            this._t += dt;
            const W = this.heightW, H = this.heightH;
            const s = 0.05, w = this._t;
            for (let y = 0; y < H; y++) {
                for (let x = 0; x < W; x++) {
                terrainArr[y*W + x] = Math.sin(x*s + w) * Math.cos(y*s + 0.5*w);
                lowFreqArr[y*W + x] = terrainArr[y*W + x] + 3;
                highFreqArr[y*W + x] = terrainArr[y*W + x] + 6;
                }
            }
        }

        // Upload into the existing height texture
        //this.updateHeightInPlace(arr);
        this.updateTexture(terrainArr, this.heightTerrainTexture);
        this.updateTexture(lowFreqArr, this.lowFreqWaveTexture);
        this.updateTexture(highFreqArr, this.highFreqWaveTexture);
        */
    }
}

// Build a (nu × nv) grid of UVs and triangle indices.
// UVs are in [0,1]; the vertex shader maps them to world space and applies displacement.
function makeGrid(nu: number, nv: number, indexOffset: number) {
    const uvs: number[] = [];
    for (let j = 0; j <= nv; j++) {
        for (let i = 0; i <= nu; i++) {
        uvs.push(i/nu, j/nv);
        }
    }
    const idx: number[] = [];
    const W = nu + 1;
    for (let j = 0; j < nv; j++) {
        for (let i = 0; i < nu; i++) {
        const a = j*W + i + indexOffset, b = a+1 + indexOffset, c = a+W + indexOffset, d = c+1 + indexOffset;
        idx.push(a,c,b, b,c,d);
        }
    }
    return { uvs: new Float32Array(uvs), indices: new Uint32Array(idx) };
}

// Vertex layout for the water grid: only (u,v) per vertex.
const heightVertexLayout: GPUVertexBufferLayout = {
    arrayStride: 2*4,
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
};


