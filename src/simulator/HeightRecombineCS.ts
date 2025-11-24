import * as shaders from "../shaders/shaders";

export class HeightRecombineCS {
  private device: GPUDevice;
  private width: number;
  private height: number;

  private hBarTex: GPUTexture;
  private hHighTex: GPUTexture;
  private hOutTex: GPUTexture;

  private ioBGL: GPUBindGroupLayout;
  private ioBG: GPUBindGroup;
  private pipeline: GPUComputePipeline;

  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    hBarTex: GPUTexture,
    hHighTex: GPUTexture,
    hOutTex: GPUTexture,
  ) {
    this.device  = device;
    this.width   = width;
    this.height  = height;
    this.hBarTex = hBarTex;
    this.hHighTex = hHighTex;
    this.hOutTex = hOutTex;

    this.ioBGL = device.createBindGroupLayout({
      label: "height recombine bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { format: "r32float", access: "read-only" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { format: "r32float", access: "read-only" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { format: "r32float", access: "write-only" } },
      ],
    });

    this.ioBG = device.createBindGroup({
      layout: this.ioBGL,
      entries: [
        { binding: 0, resource: this.hBarTex.createView() },
        { binding: 1, resource: this.hHighTex.createView() },
        { binding: 2, resource: this.hOutTex.createView() },
      ],
    });

    this.pipeline = device.createComputePipeline({
      label: "height recombine pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.ioBGL],
      }),
      compute: {
        module: device.createShaderModule({
          code: shaders.heightRecombineComputeSrc,
        }),
        entryPoint: "recombineHeight",
      },
    });
  }

  step() {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.ioBG);
    pass.dispatchWorkgroups(
      Math.ceil(this.width / 8),
      Math.ceil(this.height / 8)
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}