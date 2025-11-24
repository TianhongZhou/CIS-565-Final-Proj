import * as shaders from "../shaders/shaders";

export class FlowRecombineCS {
  private device: GPUDevice;
  private width: number;
  private height: number;

  private qBarTex: GPUTexture;
  private qHighTex: GPUTexture;
  private qOutTex: GPUTexture;

  private ioBGL: GPUBindGroupLayout;
  private ioBG: GPUBindGroup;
  private pipeline: GPUComputePipeline;

  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    qBarTex: GPUTexture,
    qHighTex: GPUTexture,
    qOutTex: GPUTexture,
  ) {
    this.device   = device;
    this.width    = width;
    this.height   = height;
    this.qBarTex  = qBarTex;
    this.qHighTex = qHighTex;
    this.qOutTex  = qOutTex;

    this.ioBGL = device.createBindGroupLayout({
      label: "flow recombine bgl",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { format: "r32float", access: "read-only" }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { format: "r32float", access: "read-only" }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { format: "r32float", access: "write-only" }
        },
      ],
    });

    this.ioBG = device.createBindGroup({
      layout: this.ioBGL,
      entries: [
        { binding: 0, resource: this.qBarTex.createView() },
        { binding: 1, resource: this.qHighTex.createView() },
        { binding: 2, resource: this.qOutTex.createView() },
      ],
    });

    this.pipeline = device.createComputePipeline({
      label: "flow recombine pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.ioBGL],
      }),
      compute: {
        module: device.createShaderModule({
          code: shaders.flowRecombineComputeSrc,
        }),
        entryPoint: "recombineFlow",
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