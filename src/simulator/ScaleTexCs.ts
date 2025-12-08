import { scaleTexSrc } from "../shaders/shaders";

export class ScaleTexCs {
    private device: GPUDevice;
    private pipeline: GPUComputePipeline;
    private paramsBuffer: GPUBuffer;
    private scaleBuffer: GPUBuffer;

    constructor(device: GPUDevice) {
        this.device = device;

        this.paramsBuffer = device.createBuffer({
            label: "height-add params",
            size: 16, // vec2<u32> + padding = 16 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.scaleBuffer = device.createBuffer({
            label: "scale factor buffer",
            size: 4, // f32 = 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.pipeline = device.createComputePipeline({
            label: "height-add pipeline",
            layout: "auto",
            compute: {
                module: device.createShaderModule({
                    label: "height-add cs module",
                    code: scaleTexSrc,
                }),
                entryPoint: "cs_main",
            },
        });
    }
    
    run(
        baseTex: GPUTexture,
        scaleFactor: number,
        width: number,
        height: number,
    ) {
        const params = new Uint32Array([width, height, 0, 0]);
        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);
        this.device.queue.writeBuffer(this.scaleBuffer, 0, new Float32Array([scaleFactor]));

        const bindGroup = this.device.createBindGroup({
            label: "height-add bind group",
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.paramsBuffer },
                },
                {
                    binding: 1,
                    resource: baseTex.createView(), // storage_2d<r32float, read_write>
                },
                {
                    binding: 2,
                    resource: { buffer: this.scaleBuffer },  // uniform scale factor
                },
            ],
        });

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass({ label: "height-add pass" });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
            Math.ceil(width / 8),
            Math.ceil(height / 8),
        );
        pass.end();

        this.device.queue.submit([encoder.finish()]);
    }
}