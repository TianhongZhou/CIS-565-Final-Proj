import * as shaders from '../shaders/shaders';

export class DiffuseCS {
    private device: GPUDevice;
    private width: number;
    private height: number;

    private heightTex: GPUTexture;
    private lowFreqTex: GPUTexture;
    private highFreqTex: GPUTexture;
    private terrainTex: GPUTexture;

    private ioBindGroupLayout!: GPUBindGroupLayout;
    private constsBindGroupLayout!: GPUBindGroupLayout;

    private heightBindGroup!: GPUBindGroup;
    private lowFreqBindGroup!: GPUBindGroup;
    private highFreqBindGroup!: GPUBindGroup;
    private constsBindGroup!: GPUBindGroup;

    private timeStepBuffer!: GPUBuffer;
    private gridScaleBuffer!: GPUBuffer;

    private diffusePipeline!: GPUComputePipeline;
    private reconstructPipeline!: GPUComputePipeline;

    constructor(
        device: GPUDevice,
        width: number,
        height: number,
        heightTex: GPUTexture,       
        lowFreqTex: GPUTexture,      
        highFreqTex: GPUTexture,      
        terrainTex: GPUTexture          
    ) {
        this.device   = device;
        this.width    = width;
        this.height   = height;
        this.heightTex   = heightTex;
        this.lowFreqTex = lowFreqTex;
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

        // group(0) / group(1) / group(2): read_write storage texture
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

        // group(3)：dt, gridScale, highFreq, terrain
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
        // group(0): height


        this.heightBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.heightTex.createView(),
                },
            ],
        });

        // group(1)：lowFreq
        this.lowFreqBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.lowFreqTex.createView(),
                },
            ],
        });

        // group(2): highFreq
        this.highFreqBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.highFreqTex.createView(),
                },
            ],
        });

        // group(3)：dt, gridScale, highFreq, terrain
        this.constsBindGroup = this.device.createBindGroup({
            layout: this.constsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.timeStepBuffer } },
                { binding: 1, resource: { buffer: this.gridScaleBuffer } },
                { binding: 2, resource: this.terrainTex.createView() },
            ],
        });
    }

    private createPipeline() {
        this.diffusePipeline = this.device.createComputePipeline({
            label: 'diffusion compute pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioBindGroupLayout,       // group(0)  height
                    this.ioBindGroupLayout,       // group(1)  lowFreq
                    this.ioBindGroupLayout,       // group(2)  highFreq
                    this.constsBindGroupLayout,   // group(3)  dt, gridScale, terrain
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
        //I have it organized inputs first then outputs, so that's why height's after in this case. 
        this.reconstructPipeline = this.device.createComputePipeline({
            label: 'reconstruct compute pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioBindGroupLayout,       // group(0)  lowFreq
                    this.ioBindGroupLayout,       // group(1)  highFreq
                    this.ioBindGroupLayout,       // group(2)  height
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
        
        const diffusePass = encoder.beginComputePass();
        diffusePass.setPipeline(this.diffusePipeline);
        diffusePass.setBindGroup(0, this.heightBindGroup);      // group(0) height
        diffusePass.setBindGroup(1, this.lowFreqBindGroup);     // group(1) lowFreq
        diffusePass.setBindGroup(2, this.highFreqBindGroup);    // group(2) highFreq
        diffusePass.setBindGroup(3, this.constsBindGroup);      // group(3) dt, gridScale, terrain

        
        diffusePass.dispatchWorkgroups(wgX, wgY);
        diffusePass.end();

        const reconstructPass = encoder.beginComputePass();
        reconstructPass.setPipeline(this.reconstructPipeline);
        reconstructPass.setBindGroup(0, this.lowFreqBindGroup);     // group(0) lowFreq
        reconstructPass.setBindGroup(1, this.highFreqBindGroup);    // group(1) highFreq
        reconstructPass.setBindGroup(2, this.heightBindGroup);      //group(2) height
        reconstructPass.setBindGroup(3, this.constsBindGroup);      //group(3) dt, gridScale, terrain

        reconstructPass.dispatchWorkgroups(wgX, wgY);
        reconstructPass.end();
        
        this.device.queue.submit([encoder.finish()]);
    }
}