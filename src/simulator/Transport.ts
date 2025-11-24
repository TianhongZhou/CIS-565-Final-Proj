import * as shaders from '../shaders/shaders';

export type LambdaMode = 'q' | 'h';

export class TransportCS {
  private device: GPUDevice;
  private width: number;
  private height: number;

  private lambdaIn: GPUTexture;   // cell-centered (W x H) r32float
  private uX: GPUTexture;         // face-centered x (W+1 x H) r32float
  private uY: GPUTexture;         // face-centered y (W x H+1) r32float
  private lambdaOut: GPUTexture;  // cell-centered (W x H) r32float (output)

  private ioBGL!: GPUBindGroupLayout;
  private constBGL!: GPUBindGroupLayout;
  private ioBindGroup!: GPUBindGroup;
  private constBindGroup!: GPUBindGroup;

  private dtBuf!: GPUBuffer;
  private hBuf!: GPUBuffer;
  private gammaBuf!: GPUBuffer;
  private modeBuf!: GPUBuffer; // int32: 0->q,1->h (for bookkeeping only)

  private pipeline!: GPUComputePipeline;

  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    lambdaIn: GPUTexture,
    uX: GPUTexture,
    uY: GPUTexture,
    lambdaOut: GPUTexture,
    mode: LambdaMode = 'q',
    gamma: number = 0.25,
    cellSize: number = 1.0
  ) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.lambdaIn = lambdaIn;
    this.uX = uX;
    this.uY = uY;
    this.lambdaOut = lambdaOut;

    this.createLayoutsAndBuffers(mode, gamma, cellSize);
    this.createBindGroups();
    this.createPipeline();
  }

  private createLayoutsAndBuffers(mode: LambdaMode, gamma: number, cellSize: number) {
    // uniforms
    this.dtBuf = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.hBuf = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.gammaBuf = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.modeBuf = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // initial uniform values
    this.device.queue.writeBuffer(this.hBuf, 0, new Float32Array([cellSize]));
    this.device.queue.writeBuffer(this.gammaBuf, 0, new Float32Array([gamma]));
    this.device.queue.writeBuffer(this.modeBuf, 0, new Int32Array([mode === 'q' ? 0 : 1]));

    // bind group layouts
    this.ioBGL = this.device.createBindGroupLayout({
      label: 'transport io bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'r32float', access: 'read-only', viewDimension: '2d' } }, // lambdaIn
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'r32float', access: 'write-only', viewDimension: '2d' } }, // lambdaOut
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'r32float', access: 'read-only', viewDimension: '2d' } }, // uX
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'r32float', access: 'read-only', viewDimension: '2d' } }, // uY
      ],
    });

    this.constBGL = this.device.createBindGroupLayout({
      label: 'transport const bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // dt
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // h
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // gamma
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // mode
      ],
    });
  }

  private createBindGroups() {
    this.ioBindGroup = this.device.createBindGroup({
      layout: this.ioBGL,
      entries: [
        { binding: 0, resource: this.lambdaIn.createView() },
        { binding: 1, resource: this.lambdaOut.createView() },
        { binding: 2, resource: this.uX.createView() },
        { binding: 3, resource: this.uY.createView() },
      ],
    });

    this.constBindGroup = this.device.createBindGroup({
      layout: this.constBGL,
      entries: [
        { binding: 0, resource: { buffer: this.dtBuf } },
        { binding: 1, resource: { buffer: this.hBuf } },
        { binding: 2, resource: { buffer: this.gammaBuf } },
        { binding: 3, resource: { buffer: this.modeBuf } },
      ],
    });
  }

  private createPipeline() {
    this.pipeline = this.device.createComputePipeline({
      label: 'transport compute pipeline unified',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.ioBGL, this.constBGL] }),
      compute: {
        module: this.device.createShaderModule({ code: shaders.transportComputeSrc }),
        entryPoint: 'transport',
      },
    });
  }

  // external API to change parameters
  setGamma(g: number) { this.device.queue.writeBuffer(this.gammaBuf, 0, new Float32Array([g])); }
  setCellSize(h: number) { this.device.queue.writeBuffer(this.hBuf, 0, new Float32Array([h])); }
  setMode(mode: LambdaMode) { this.device.queue.writeBuffer(this.modeBuf, 0, new Int32Array([mode === 'q' ? 0 : 1])); }

  step(dt: number) {
    // update dt
    this.device.queue.writeBuffer(this.dtBuf, 0, new Float32Array([dt]));

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.ioBindGroup);
    pass.setBindGroup(1, this.constBindGroup);

    const tgX = 8, tgY = 8;
    pass.dispatchWorkgroups(Math.ceil(this.width / tgX), Math.ceil(this.height / tgY));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
