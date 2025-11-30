import { addOnSrc } from "../shaders/shaders";

export class AddOnCS {
    private device: GPUDevice;
    private pipeline: GPUComputePipeline;
    private paramsBuffer: GPUBuffer;

    constructor(device: GPUDevice) {
        this.device = device;

        this.paramsBuffer = device.createBuffer({
            label: "height-add params",
            size: 16, // vec2<u32> + padding = 16 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.pipeline = device.createComputePipeline({
            label: "height-add pipeline",
            layout: "auto",
            compute: {
                module: device.createShaderModule({
                    label: "height-add cs module",
                    code: addOnSrc,
                }),
                entryPoint: "cs_main",
            },
        });
    }
    
    run(
        baseTex: GPUTexture,
        addTex: GPUTexture,
        width: number,
        height: number,
    ) {
        const params = new Uint32Array([width, height, 0, 0]);
        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

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
                    resource: addTex.createView(),  // texture_2d<f32>
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