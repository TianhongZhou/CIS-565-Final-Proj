import * as shaders from '../shaders/shaders';

export class VelocityCS {
  private device: GPUDevice;
  private width: number;
  private height: number;

  private hTex: GPUTexture;
  private qxTex: GPUTexture;
  private qyTex: GPUTexture;
  private uXTex: GPUTexture;
  private uYTex: GPUTexture;

  private velConstBuf: GPUBuffer;
  private ioBGL: GPUBindGroupLayout;
  private constBGL: GPUBindGroupLayout;
  private ioBG: GPUBindGroup;
  private constBG: GPUBindGroup;
  private pipeline: GPUComputePipeline;

  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    hTex: GPUTexture,
    qxTex: GPUTexture,
    qyTex: GPUTexture,
    uXTex: GPUTexture,
    uYTex: GPUTexture,
  ) {
    this.device  = device;
    this.width   = width;
    this.height  = height;
    this.hTex    = hTex;
    this.qxTex   = qxTex;
    this.qyTex   = qyTex;
    this.uXTex   = uXTex;
    this.uYTex   = uYTex;

    this.velConstBuf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.velConstBuf, 0, new Float32Array([1e-3]));

    this.ioBGL = device.createBindGroupLayout({
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float' } }, // hTex
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float' } }, // qxTex
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float' } }, // qyTex
        { binding: 3, visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: 'r32float', access: 'write-only'} }, // uX
        { binding: 4, visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: 'r32float', access: 'write-only'} }, // uY
    ],
    });

    this.constBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    this.ioBG = device.createBindGroup({
      layout: this.ioBGL,
      entries: [
        { binding: 0, resource: hTex.createView()  },
        { binding: 1, resource: qxTex.createView() },
        { binding: 2, resource: qyTex.createView() },
        { binding: 3, resource: uXTex.createView() },
        { binding: 4, resource: uYTex.createView() },
      ],
    });

    this.constBG = device.createBindGroup({
      layout: this.constBGL,
      entries: [
        { binding: 0, resource: { buffer: this.velConstBuf } },
      ],
    });

    const module = device.createShaderModule({
      code: shaders.velocityComputeSrc,
    });

    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.ioBGL, this.constBGL] }),
      compute: {
        module,
        entryPoint: 'updateVelocity',
      },
    });
  }

  step() {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.ioBG);
    pass.setBindGroup(1, this.constBG);
    pass.dispatchWorkgroups(
      Math.ceil(this.width / 8),
      Math.ceil(this.height / 8),
      1
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}