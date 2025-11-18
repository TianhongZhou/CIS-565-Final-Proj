import * as shaders from '../shaders/shaders';

export class TransportCS {
    private device: GPUDevice;
    private width: number;
    private height: number;

    private qTex: GPUTexture;
    private uTex: GPUTexture;
    private updatedQTex: GPUTexture;

    private ioRWLayout!: GPUBindGroupLayout;
    private ioROLayout!: GPUBindGroupLayout;
    private constsLayout!: GPUBindGroupLayout;

    private qROBindGroup!: GPUBindGroup;
    private uROBindGroup!: GPUBindGroup;
    private updatedQRWBindGroup!: GPUBindGroup;
    private constsBindGroup!: GPUBindGroup;

    private dtBuffer!: GPUBuffer;
    private gridScaleBuffer!: GPUBuffer;

    private transportPipeline!: GPUComputePipeline;

    constructor(
        device: GPUDevice,
        width: number,
        height: number,
        qTex: GPUTexture,
        uTex: GPUTexture,
        updatedQTex: GPUTexture
    ) {
        this.device = device;
        this.width = width;
        this.height = height;
        this.qTex = qTex;
        this.uTex = uTex;
        this.updatedQTex = updatedQTex;

        this.createBuffersAndLayouts();
        this.createBindGroups();
        this.createPipeline();
    }

    private createBuffersAndLayouts() {
        this.dtBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.gridScaleBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.device.queue.writeBuffer(this.dtBuffer, 0, new Float32Array([0.25]));
        this.device.queue.writeBuffer(this.gridScaleBuffer, 0, new Float32Array([1.0]));

        // group(0): updated q (read-write)
        this.ioRWLayout = this.device.createBindGroupLayout({
            label: 'transport rw output',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: 'read-write',
                        format: 'r32float',
                        viewDimension: '2d',
                    },
                },
            ],
        });

        // group(1/2): q, u (read-only)
        this.ioROLayout = this.device.createBindGroupLayout({
            label: 'transport read-only input',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: 'read-only',
                        format: 'r32float',
                        viewDimension: '2d',
                    },
                },
            ],
        });

        // group(3): dt, gridScale
        this.constsLayout = this.device.createBindGroupLayout({
            label: 'transport consts',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
    }

    private createBindGroups() {
        this.qROBindGroup = this.device.createBindGroup({
            layout: this.ioROLayout,
            entries: [{ binding: 0, resource: this.qTex.createView() }],
        });

        this.uROBindGroup = this.device.createBindGroup({
            layout: this.ioROLayout,
            entries: [{ binding: 0, resource: this.uTex.createView() }],
        });

        this.updatedQRWBindGroup = this.device.createBindGroup({
            layout: this.ioRWLayout,
            entries: [{ binding: 0, resource: this.updatedQTex.createView() }],
        });

        this.constsBindGroup = this.device.createBindGroup({
            layout: this.constsLayout,
            entries: [
                { binding: 0, resource: { buffer: this.dtBuffer } },
                { binding: 1, resource: { buffer: this.gridScaleBuffer } },
            ],
        });
    }

    private createPipeline() {
        this.transportPipeline = this.device.createComputePipeline({
            label: 'transport compute pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioRWLayout,     // group(0): updated q (RW)
                    this.ioROLayout,     // group(1): q (RO)
                    this.ioROLayout,     // group(2): u (RO)
                    this.constsLayout,   // group(3): dt + gridScale
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    code: shaders.transportComputeSrc,
                }),
                entryPoint: 'transport',
            },
        });
    }

    step(dt: number) {
        this.device.queue.writeBuffer(
            this.dtBuffer,
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

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.transportPipeline);

        // Bind groups
        pass.setBindGroup(0, this.updatedQRWBindGroup); // output q
        pass.setBindGroup(1, this.qROBindGroup);         // input q
        pass.setBindGroup(2, this.uROBindGroup);         // velocity field
        pass.setBindGroup(3, this.constsBindGroup);      // dt, gridScale

        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();

        this.device.queue.submit([encoder.finish()]);
    }
}
