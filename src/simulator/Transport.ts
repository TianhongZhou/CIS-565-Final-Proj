import * as shaders from "../shaders/shaders";

export class TransportCS {
    private device: GPUDevice;
    private width: number;
    private height: number;

    private qTex: GPUTexture;
    private uXTex: GPUTexture;
    private uYTex: GPUTexture;
    private qNextTex: GPUTexture;

    private ioBGL!: GPUBindGroupLayout;
    private constBGL!: GPUBindGroupLayout;

    private qBindGroup!: GPUBindGroup;
    private constBindGroup!: GPUBindGroup;

    private dtBuffer!: GPUBuffer;
    private gridScaleBuffer!: GPUBuffer;

    private pipeline!: GPUComputePipeline;

    constructor(
        device: GPUDevice,
        width: number,
        height: number,
        qTex: GPUTexture,
        uXTex: GPUTexture,
        uYTex: GPUTexture,
        qNextTex: GPUTexture
    ) {
        this.device = device;
        this.width = width;
        this.height = height;
        this.qTex = qTex;
        this.uXTex = uXTex;
        this.uYTex = uYTex;
        this.qNextTex = qNextTex;

        this.createResources();
        this.createPipeline();
    }

    private createResources() {
        this.dtBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.gridScaleBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.device.queue.writeBuffer(this.gridScaleBuffer, 0, new Float32Array([1.0]));

        this.ioBGL = this.device.createBindGroupLayout({
            entries: [
                {   // q
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { format: "r32float", access: "read-only" },
                },
                {   // qNext
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { format: "r32float", access: "write-only" },
                },
                {   // uX
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { format: "r32float", access: "read-only" },
                },
                {   // uY
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { format: "r32float", access: "read-only" },
                },
            ]
        });

        this.constBGL = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ]
        });

        this.qBindGroup = this.device.createBindGroup({
            layout: this.ioBGL,
            entries: [
                { binding: 0, resource: this.qTex.createView() },
                { binding: 1, resource: this.qNextTex.createView() },
                { binding: 2, resource: this.uXTex.createView() },
                { binding: 3, resource: this.uYTex.createView() },
            ]
        });

        this.constBindGroup = this.device.createBindGroup({
            layout: this.constBGL,
            entries: [
                { binding: 0, resource: { buffer: this.dtBuffer } },
                { binding: 1, resource: { buffer: this.gridScaleBuffer } },
            ]
        });
    }

    private createPipeline() {
        this.pipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.ioBGL, this.constBGL]
            }),
            compute: {
                module: this.device.createShaderModule({
                    code: shaders.transportComputeSrc
                }),
                entryPoint: "transport"
            }
        });
    }

    step(dt: number) {
        this.device.queue.writeBuffer(this.dtBuffer, 0, new Float32Array([dt]));

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.qBindGroup);
        pass.setBindGroup(1, this.constBindGroup);

        // TODO : needs substeps here

        const tgX = 8, tgY = 8;
        pass.dispatchWorkgroups(
            Math.ceil(this.width / tgX),
            Math.ceil(this.height / tgY)
        );

        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }
}
