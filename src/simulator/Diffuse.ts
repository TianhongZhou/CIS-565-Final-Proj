import * as shaders from '../shaders/shaders';

export class DiffuseCS {
    private device: GPUDevice;
    private width: number;
    private height: number;

    private lowFreqInTex: GPUTexture;
    private lowFreqOutTex: GPUTexture;

    private highFreqTex: GPUTexture;
    private terrainTex: GPUTexture;

    private ioBindGroupLayout!: GPUBindGroupLayout;
    private constsBindGroupLayout!: GPUBindGroupLayout;

    private inBindGroup!: GPUBindGroup;
    private outBindGroup!: GPUBindGroup;
    private constsBindGroup!: GPUBindGroup;

    private timeStepBuffer!: GPUBuffer;
    private gridScaleBuffer!: GPUBuffer;

    private pipeline!: GPUComputePipeline;

    constructor(
        device: GPUDevice,
        width: number,
        height: number,
        lowFreqInTex: GPUTexture,       
        lowFreqOutTex: GPUTexture,      
        highFreqTex: GPUTexture,      
        terrainTex: GPUTexture          
    ) {
        this.device   = device;
        this.width    = width;
        this.height   = height;
        this.lowFreqInTex  = lowFreqInTex;
        this.lowFreqOutTex = lowFreqOutTex;
        this.highFreqTex   = highFreqTex;
        this.terrainTex    = terrainTex;

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
            new Float32Array([0.016])
        );
        this.device.queue.writeBuffer(
            this.gridScaleBuffer,
            0,
            new Float32Array([1.0])
        );

        // group(0) / group(1): read_write storage texture
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

        // group(2)：dt, gridScale, highFreq, terrain
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
                        access: 'read-write',       // highFreqHeightIn
                        viewDimension: '2d',
                    },
                },
                {
                    binding: 3,
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
        // group(0)：lowFreqHeightIn
        this.inBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.lowFreqInTex.createView(),
                },
            ],
        });

        // group(1)：lowFreqHeightOut
        this.outBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.lowFreqOutTex.createView(),
                },
            ],
        });

        // group(2)：dt, gridScale, highFreq, terrain
        this.constsBindGroup = this.device.createBindGroup({
            layout: this.constsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.timeStepBuffer } },
                { binding: 1, resource: { buffer: this.gridScaleBuffer } },
                { binding: 2, resource: this.highFreqTex.createView() },
                { binding: 3, resource: this.terrainTex.createView() },
            ],
        });
    }

    private createPipeline() {
        this.pipeline = this.device.createComputePipeline({
            label: 'diffusion compute pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioBindGroupLayout,       // group(0)  lowFreqHeightIn
                    this.ioBindGroupLayout,       // group(1)  lowFreqHeightOut
                    this.constsBindGroupLayout,   // group(2)  dt, gridScale, highFreq, terrain
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
    }

    step(dt: number) {
        this.device.queue.writeBuffer(
            this.timeStepBuffer,
            0,
            new Float32Array([dt])
        );

        const encoder = this.device.createCommandEncoder();

        encoder.copyTextureToTexture(
            { texture: this.lowFreqOutTex },
            { texture: this.lowFreqInTex },
            {
                width: this.width,
                height: this.height,
                depthOrArrayLayers: 1,
            }
        );

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.inBindGroup);        // group(0) lowFreqHeightIn
        pass.setBindGroup(1, this.outBindGroup);       // group(1) lowFreqHeightOut
        pass.setBindGroup(2, this.constsBindGroup);    // group(2) consts + highFreq + terrain

        const wgX = Math.ceil(
            this.width / shaders.constants.threadsInDiffusionBlockX
        );
        const wgY = Math.ceil(
            this.height / shaders.constants.threadsInDiffusionBlockY
        );
        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();

        this.device.queue.submit([encoder.finish()]);
    }
}