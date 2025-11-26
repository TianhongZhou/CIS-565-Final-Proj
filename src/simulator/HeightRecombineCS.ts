import * as shaders from "../shaders/shaders";

export class HeightRecombineCS {
    private device: GPUDevice;
    private width: number;
    private height: number;

    private pipeline: GPUComputePipeline;
    private ioBindGroupLayout: GPUBindGroupLayout;
    private constsBindGroupLayout: GPUBindGroupLayout;
    private bindGroupIO: GPUBindGroup;
    private bindGroupConsts: GPUBindGroup;

    private dtBuffer: GPUBuffer;
    private gridScaleBuffer: GPUBuffer;

    constructor(
        device: GPUDevice,
        width: number,
        height: number,
        hPrevTex: GPUTexture,
        qxTex: GPUTexture,
        qyTex: GPUTexture,
        hSurfTex: GPUTexture,
        uXTex: GPUTexture,
        uYTex: GPUTexture,
        hOutTex: GPUTexture,
        gridScale: number,
    ) {
        this.device = device;
        this.width = width;
        this.height = height;

        this.ioBindGroupLayout = device.createBindGroupLayout({
          label: "height recombine bgl",
          entries: [
            // hPrev
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
            },
            // qx
            {
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
            },
            // qy
            {
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
            },
            // hSurf
            {
              binding: 3,
              visibility: GPUShaderStage.COMPUTE,
              texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
            },
            // uX
            {
              binding: 4,
              visibility: GPUShaderStage.COMPUTE,
              texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
            },
            // uY
            {
              binding: 5,
              visibility: GPUShaderStage.COMPUTE,
              texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
            },
            // hOut storage texture（write-only）
            {
              binding: 6,
              visibility: GPUShaderStage.COMPUTE,
              storageTexture: {
                format: "r32float",
                access: "write-only",
                viewDimension: "2d",
              },
            },
          ],
        });

        this.constsBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this.pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.ioBindGroupLayout, this.constsBindGroupLayout],
            }),
            compute: {
                module: device.createShaderModule({
                    code: shaders.heightRecombineComputeSrc,
                }),
                entryPoint: 'updateHeight',
            },
        });

        this.bindGroupIO = device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                { binding: 0, resource: hPrevTex.createView() },
                { binding: 1, resource: qxTex.createView() },
                { binding: 2, resource: qyTex.createView() },
                { binding: 3, resource: hSurfTex.createView() },
                { binding: 4, resource: uXTex.createView() },
                { binding: 5, resource: uYTex.createView() },
                { binding: 6, resource: hOutTex.createView() },
            ],
        });

        this.dtBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.gridScaleBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupConsts = device.createBindGroup({
            layout: this.constsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.dtBuffer } },
                { binding: 1, resource: { buffer: this.gridScaleBuffer } },
            ],
        });

        device.queue.writeBuffer(this.gridScaleBuffer, 0, new Float32Array([gridScale]));
    }

    step(dt: number) {
        this.device.queue.writeBuffer(this.dtBuffer, 0, new Float32Array([dt]));

        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroupIO);
        pass.setBindGroup(1, this.bindGroupConsts);

        const workgroupSize = 8;
        const wgX = Math.ceil(this.width  / workgroupSize);
        const wgY = Math.ceil(this.height / workgroupSize);
        pass.dispatchWorkgroups(wgX, wgY);
        pass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}