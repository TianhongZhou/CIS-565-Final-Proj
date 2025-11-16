import * as shaders from '../shaders/shaders';

export class DiffuseCS {
    private device: GPUDevice;
    private width: number;
    private height: number;

    //TODO: Consider renaming to diffuseTex? Cause it can represent both height and flux
    private diffuseTex: GPUTexture; 
    private lowFreqTex: GPUTexture;
    private lowFreqTexPingpong: GPUTexture;
    private highFreqTex: GPUTexture;
    private terrainTex: GPUTexture;

    private ioBindGroupLayout!: GPUBindGroupLayout;
    private constsBindGroupLayout!: GPUBindGroupLayout;

    private heightBindGroup!: GPUBindGroup;
    private lowFreqBindGroup!: GPUBindGroup;
    private lowFreqBindGroupPingpong!: GPUBindGroup;
    private highFreqBindGroup!: GPUBindGroup;
    private constsBindGroup!: GPUBindGroup;

    private timeStepBuffer!: GPUBuffer;
    private gridScaleBuffer!: GPUBuffer;

    private diffusePipeline!: GPUComputePipeline;
    //TODO: Rename to highFrequencyPipeline (and other files/classes related to the reconstructHeight step)
    private reconstructPipeline!: GPUComputePipeline;

    constructor(
        device: GPUDevice,
        width: number,
        height: number,
        fieldTex: GPUTexture,       
        lowFreqTex: GPUTexture,     
        lowFreqTexPingpong: GPUTexture, 
        highFreqTex: GPUTexture,      
        terrainTex: GPUTexture          
    ) {
        this.device   = device;
        this.width    = width;
        this.height   = height;
        this.diffuseTex   = fieldTex;
        this.lowFreqTex = lowFreqTex;
        this.lowFreqTexPingpong = lowFreqTexPingpong;
        this.highFreqTex = highFreqTex;
        this.terrainTex  = terrainTex;

        this.createBuffersAndLayouts();
        this.createBindGroups();
        this.createPipeline();
    }

    private createBuffersAndLayouts() {
        this.timeStepBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.gridScaleBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.device.queue.writeBuffer(
            this.timeStepBuffer,
            0,
            new Float32Array([0.25])
        );
        this.device.queue.writeBuffer(
            this.gridScaleBuffer,
            0,
            new Float32Array([1.0])
        );

        // Diffuse (group 0 / 1)) + Reconstruct (group 0 / 1 / 2): read_write storage texture
        this.ioBindGroupLayout = this.device.createBindGroupLayout({
            label: 'diffusion IO BGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'r32float',
                        access: 'read-write',
                        viewDimension: '2d',
                    },
                },
            ],
        });

        // Diffuse (group 2)) + Reconstruct (group 3)：dt, gridScale, highFreq, terrain
        this.constsBindGroupLayout = this.device.createBindGroupLayout({
            label: 'diffusion consts BGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' },   // timeStep
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' },   // gridScale
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'r32float',
                        access: 'read-only',        // terrainHeightIn
                        viewDimension: '2d',
                    },
                },
            ],
        });
    }

    private createBindGroups() {


        // Reconstruct group(1) + Copy to lowFreq: height (diffusion)


        this.heightBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.diffuseTex.createView(),
                },
            ],
        });

        // Diffuse group(0) + Reconstruct group(0) + PingPonged with lowFreqPingPong：lowFreq
        this.lowFreqBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.lowFreqTex.createView(),
                },
            ],
        });
        // Diffuse group(1) + PingPonged with lowFreq：lowFreqPingPong
        this.lowFreqBindGroupPingpong = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.lowFreqTexPingpong.createView(),
                },
            ],
        });
        

        // Diffuse group(2) + Reconstruct group(3)：dt, gridScale, highFreq, terrain
        this.constsBindGroup = this.device.createBindGroup({
            layout: this.constsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.timeStepBuffer } },
                { binding: 1, resource: { buffer: this.gridScaleBuffer } },
                { binding: 2, resource: this.terrainTex.createView() },
            ],
        });

        // Reconstruct group(2)：highFreq
        this.highFreqBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.highFreqTex.createView(),
                },
            ],
        });
    }

    private createPipeline() {
        this.diffusePipeline = this.device.createComputePipeline({
            label: 'diffusion compute pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioBindGroupLayout,       // group(0)  DiffusionInput
                    this.ioBindGroupLayout,       // group(1)  lowFreq
                    this.constsBindGroupLayout,   // group(2)  dt, gridScale, terrain
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'diffuse compute shader',
                    code: shaders.diffuseComputeSrc,
                }),
                entryPoint: 'diffuse',
            },
        });
        //I have it organized inputs first then outputs, 
        this.reconstructPipeline = this.device.createComputePipeline({
            label: 'reconstruct compute pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioBindGroupLayout,       // group(0)  lowFreq
                    this.ioBindGroupLayout,       // group(1)  diffusion
                    this.ioBindGroupLayout,       // group(2)  highFreq
                    this.constsBindGroupLayout,   // group(3)  dt, gridScale, terrain
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'reconstruct compute shader',
                    code: shaders.reconstructHeightSrc,
                }),
                entryPoint: 'reconstructHeight',
            },
        });

    }

    step(dt: number) {

        this.device.queue.writeBuffer(
            this.timeStepBuffer,
            0,
            new Float32Array([dt])
        );

        const encoder = this.device.createCommandEncoder();

        const wgX = Math.ceil(
            this.width / shaders.constants.threadsInDiffusionBlockX
        );
        const wgY = Math.ceil(
            this.height / shaders.constants.threadsInDiffusionBlockY
        );
        //Inital copy of height to lowFreq
        encoder.copyTextureToTexture(
            { texture: this.diffuseTex },
            { texture: this.lowFreqTex },
            {
                width: this.width,
                height: this.height,
                depthOrArrayLayers: 1,
            }
        );
        //Note: since we're pingponging, must have an even number of substeps
        for(let i = 0; i < 128; i++)
        {
            const diffusePass = encoder.beginComputePass();
            diffusePass.setPipeline(this.diffusePipeline);
            diffusePass.setBindGroup(0, this.lowFreqBindGroup);             // group(0) lowFreq
            diffusePass.setBindGroup(1, this.lowFreqBindGroupPingpong);     // group(1) lowFreqPingpong
            diffusePass.setBindGroup(2, this.constsBindGroup);              // group(2) dt, gridScale, terrain

        
            diffusePass.dispatchWorkgroups(wgX, wgY);
            diffusePass.end();
            //Pingpong
            const temp = this.lowFreqBindGroup;
            this.lowFreqBindGroup = this.lowFreqBindGroupPingpong;
            this.lowFreqBindGroupPingpong = temp;
        }
        

        const reconstructPass = encoder.beginComputePass();
        reconstructPass.setPipeline(this.reconstructPipeline);
        reconstructPass.setBindGroup(0, this.lowFreqBindGroup);     // group(0) lowFreq
        reconstructPass.setBindGroup(1, this.heightBindGroup);      // group(1) height
        reconstructPass.setBindGroup(2, this.highFreqBindGroup);    // group(2) highFreq
        reconstructPass.setBindGroup(3, this.constsBindGroup);      // group(3) dt, gridScale, terrain

        reconstructPass.dispatchWorkgroups(wgX, wgY);
        reconstructPass.end();
        
        this.device.queue.submit([encoder.finish()]);
    }
}