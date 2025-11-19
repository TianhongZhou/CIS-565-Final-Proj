import * as shaders from '../shaders/shaders';

const FFT_FLAG_INVERSE = 1;
const FFT_FLAG_USE_REAL = 2;

// Utility so we consistently ceil-divide thread groups.
function workgroupCount(value: number, block: number): number {
    return Math.ceil(value / block);
}

// GPU implementation of the Airy-wave step (Algorithm 2).
export class AiryWaveCS {
    private device: GPUDevice;
    private width: number;
    private height: number;

    // Shared textures written by DiffuseCS that represent the high-frequency fields.
    private heightTexture: GPUTexture;
    private qxTexture: GPUTexture;
    private qyTexture: GPUTexture;

    // Intermediate textures used to run FFTs on height.
    private hMidTexture: GPUTexture;
    private hTempTexture: GPUTexture;
    private hFreqTexture: GPUTexture;

    // Intermediate textures used for the qx FFT passes.
    private qxTempTexture: GPUTexture;
    private qxFreqTexture: GPUTexture;
    private qxFreqUpdatedTexture: GPUTexture;

    // Intermediate textures used for the qy FFT passes.
    private qyTempTexture: GPUTexture;
    private qyFreqTexture: GPUTexture;
    private qyFreqUpdatedTexture: GPUTexture;

    // Frequency-space derivatives (∂h/∂x, ∂h/∂y).
    private derivXTexture: GPUTexture;
    private derivYTexture: GPUTexture;

    private averagePipeline: GPUComputePipeline;
    private fftRowsPipeline: GPUComputePipeline;
    private fftColsPipeline: GPUComputePipeline;
    private derivativePipeline: GPUComputePipeline;
    private fluxPipeline: GPUComputePipeline;
    private writePipeline: GPUComputePipeline;

    private gridUniformBuffer: GPUBuffer;
    private fftUniformBuffer: GPUBuffer;
    private fluxUniformBuffer: GPUBuffer;

    // Bind groups representing each compute stage (average, FFT, derivative, flux, write-back).
    private averageBindGroup: GPUBindGroup;
    private heightRowBindGroup: GPUBindGroup;
    private heightColBindGroup: GPUBindGroup;

    private qxRowBindGroup: GPUBindGroup;
    private qxColBindGroup: GPUBindGroup;
    private qxColInverseBindGroup: GPUBindGroup;
    private qxRowInverseBindGroup: GPUBindGroup;

    private qyRowBindGroup: GPUBindGroup;
    private qyColBindGroup: GPUBindGroup;
    private qyColInverseBindGroup: GPUBindGroup;
    private qyRowInverseBindGroup: GPUBindGroup;

    private derivativeBindGroup: GPUBindGroup;
    private qxFluxBindGroup: GPUBindGroup;
    private qyFluxBindGroup: GPUBindGroup;

    private qxWriteBindGroup: GPUBindGroup;
    private qyWriteBindGroup: GPUBindGroup;

    // CPU copies backing the FFT/flux uniform buffers so we can rewrite per-dispatch state quickly.
    private fftUniformArray: Uint32Array;
    private fluxUniformArray: Float32Array;
    // Bulk solver still needs an “effective” depth – currently approximated by a simple average.
    private averageDepth: number;

    constructor(
        device: GPUDevice,
        width: number,
        height: number,
        heightTexture: GPUTexture,
        qxTexture: GPUTexture,
        qyTexture: GPUTexture,
        smoothDepth: Float32Array
    ) {
        this.device = device;
        this.width = width;
        this.height = height;
        this.heightTexture = heightTexture;
        this.qxTexture = qxTexture;
        this.qyTexture = qyTexture;

        // TODO: replace single averaged depth with per-sample depth tables + spatial interpolation.
        this.averageDepth = this.computeAverageDepth(smoothDepth);

        const realUsage =
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.COPY_DST;
        const complexUsage =
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.COPY_DST;

        // Scratch textures for height FFT pipeline (real mid + complex ping-pong).
        this.hMidTexture = this.createTexture(width, height, 'r32float', realUsage);
        this.hTempTexture = this.createTexture(width, height, 'rgba32float', complexUsage);
        this.hFreqTexture = this.createTexture(width, height, 'rgba32float', complexUsage);

        // Scratch textures for qx FFT pipeline.
        this.qxTempTexture = this.createTexture(width, height, 'rgba32float', complexUsage);
        this.qxFreqTexture = this.createTexture(width, height, 'rgba32float', complexUsage);
        this.qxFreqUpdatedTexture = this.createTexture(width, height, 'rgba32float', complexUsage);

        // Scratch textures for qy FFT pipeline.
        this.qyTempTexture = this.createTexture(width, height, 'rgba32float', complexUsage);
        this.qyFreqTexture = this.createTexture(width, height, 'rgba32float', complexUsage);
        this.qyFreqUpdatedTexture = this.createTexture(width, height, 'rgba32float', complexUsage);

        // Frequency-space derivative storage.
        this.derivXTexture = this.createTexture(width, height, 'rgba32float', complexUsage);
        this.derivYTexture = this.createTexture(width, height, 'rgba32float', complexUsage);

        this.gridUniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.fftUniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.fluxUniformBuffer = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Record the resolution once since most shaders need it.
        const gridData = new Uint32Array([width, height, 0, 0]);
        device.queue.writeBuffer(
            this.gridUniformBuffer,
            0,
            gridData.buffer,
            gridData.byteOffset,
            gridData.byteLength
        );

        // FFT uniforms store width/height/count/flags for current pass.
        this.fftUniformArray = new Uint32Array(4);
        // Flux uniforms: [dt, gravity, domainX, domainY, depth, padding...].
        this.fluxUniformArray = new Float32Array([
            0.0,
            9.81,
            width,
            height,
            this.averageDepth,
            0,
            0,
            0,
        ]);
        device.queue.writeBuffer(
            this.fluxUniformBuffer,
            0,
            this.fluxUniformArray.buffer,
            this.fluxUniformArray.byteOffset,
            this.fluxUniformArray.byteLength
        );

        // Layout #0: combine two half-step height textures into a mid-step texture.
        const averageLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d' },
                },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        // Layout #1: shared between row/column FFT passes (input can be R32 or complex, output always complex).
        const fftLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' },
                },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        // Layout #2: derivative pass (read height spectrum, write ∂h/∂x and ∂h/∂y).
        const derivativeLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' },
                },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        // Layout #3: exponential integrator (q spectrum + derivative spectrum + dt/uniforms).
        const fluxLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' },
                },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        // Layout #4: convert complex inverse FFT output back into the shared R32 textures.
        const writeLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d' },
                },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        const module = device.createShaderModule({
            code: shaders.airyWaveComputeSrc,
        });

        // Pipelines map each WGSL entry point to the matching bind group layout defined above.
        this.averagePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [averageLayout] }),
            compute: { module, entryPoint: 'airyAverage' },
        });
        this.fftRowsPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [fftLayout] }),
            compute: { module, entryPoint: 'fftRows' },
        });
        this.fftColsPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [fftLayout] }),
            compute: { module, entryPoint: 'fftCols' },
        });
        this.derivativePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [derivativeLayout] }),
            compute: { module, entryPoint: 'airyDerivatives' },
        });
        this.fluxPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [fluxLayout] }),
            compute: { module, entryPoint: 'airyUpdateFlux' },
        });
        this.writePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [writeLayout] }),
            compute: { module, entryPoint: 'airyWriteReal' },
        });

        // TODO: heightMinusTex, heightPlusTex are same for now
        const heightMinusView = this.heightTexture.createView();
        const heightPlusView = this.heightTexture.createView();
        const hMidView = this.hMidTexture.createView();
        const hTempView = this.hTempTexture.createView();
        const hFreqView = this.hFreqTexture.createView();

        const qxRealView = this.qxTexture.createView();
        const qxTempView = this.qxTempTexture.createView();
        const qxFreqView = this.qxFreqTexture.createView();
        const qxFreqUpdatedView = this.qxFreqUpdatedTexture.createView();

        const qyRealView = this.qyTexture.createView();
        const qyTempView = this.qyTempTexture.createView();
        const qyFreqView = this.qyFreqTexture.createView();
        const qyFreqUpdatedView = this.qyFreqUpdatedTexture.createView();

        const derivXView = this.derivXTexture.createView();
        const derivYView = this.derivYTexture.createView();

        this.averageBindGroup = device.createBindGroup({
            layout: averageLayout,
            entries: [
                { binding: 0, resource: heightMinusView },
                { binding: 1, resource: heightPlusView },
                { binding: 2, resource: hMidView },
                { binding: 3, resource: { buffer: this.gridUniformBuffer } },
            ],
        });

        // FFT bind groups share the same layout but swap the input/output views depending on the stage.
        this.heightRowBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: hMidView },
                { binding: 1, resource: hFreqView },
                { binding: 2, resource: hTempView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.heightColBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: hMidView },
                { binding: 1, resource: hTempView },
                { binding: 2, resource: hFreqView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.qxRowBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: qxRealView },
                { binding: 1, resource: qxFreqView },
                { binding: 2, resource: qxTempView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.qxColBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: qxRealView },
                { binding: 1, resource: qxTempView },
                { binding: 2, resource: qxFreqView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.qxColInverseBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: qxRealView },
                { binding: 1, resource: qxFreqView },
                { binding: 2, resource: qxTempView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.qxRowInverseBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: qxRealView },
                { binding: 1, resource: qxTempView },
                { binding: 2, resource: qxFreqView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.qyRowBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: qyRealView },
                { binding: 1, resource: qyFreqView },
                { binding: 2, resource: qyTempView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.qyColBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: qyRealView },
                { binding: 1, resource: qyTempView },
                { binding: 2, resource: qyFreqView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.qyColInverseBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: qyRealView },
                { binding: 1, resource: qyFreqView },
                { binding: 2, resource: qyTempView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.qyRowInverseBindGroup = device.createBindGroup({
            layout: fftLayout,
            entries: [
                { binding: 0, resource: qyRealView },
                { binding: 1, resource: qyTempView },
                { binding: 2, resource: qyFreqView },
                { binding: 3, resource: { buffer: this.fftUniformBuffer } },
            ],
        });

        this.derivativeBindGroup = device.createBindGroup({
            layout: derivativeLayout,
            entries: [
                { binding: 0, resource: hFreqView },
                { binding: 1, resource: derivXView },
                { binding: 2, resource: derivYView },
                { binding: 3, resource: { buffer: this.gridUniformBuffer } },
            ],
        });

        this.qxFluxBindGroup = device.createBindGroup({
            layout: fluxLayout,
            entries: [
                { binding: 0, resource: qxFreqView },
                { binding: 1, resource: derivXView },
                { binding: 2, resource: qxFreqUpdatedView },
                { binding: 3, resource: { buffer: this.fluxUniformBuffer } },
                { binding: 4, resource: { buffer: this.gridUniformBuffer } },
            ],
        });

        this.qyFluxBindGroup = device.createBindGroup({
            layout: fluxLayout,
            entries: [
                { binding: 0, resource: qyFreqView },
                { binding: 1, resource: derivYView },
                { binding: 2, resource: qyFreqUpdatedView },
                { binding: 3, resource: { buffer: this.fluxUniformBuffer } },
                { binding: 4, resource: { buffer: this.gridUniformBuffer } },
            ],
        });

        this.qxWriteBindGroup = device.createBindGroup({
            layout: writeLayout,
            entries: [
                { binding: 0, resource: qxFreqView },
                { binding: 1, resource: qxRealView },
                { binding: 2, resource: { buffer: this.gridUniformBuffer } },
            ],
        });

        this.qyWriteBindGroup = device.createBindGroup({
            layout: writeLayout,
            entries: [
                { binding: 0, resource: qyFreqView },
                { binding: 1, resource: qyRealView },
                { binding: 2, resource: { buffer: this.gridUniformBuffer } },
            ],
        });
    }

    step(dt: number): void {
        this.fluxUniformArray[0] = dt;
        this.device.queue.writeBuffer(
            this.fluxUniformBuffer,
            0,
            this.fluxUniformArray.buffer,
            this.fluxUniformArray.byteOffset,
            this.fluxUniformArray.byteLength
        );

        const encoder = this.device.createCommandEncoder();

        // 1) Build h_t from half steps.
        this.runAveragePass(encoder);

        // 2) Forward FFTs for h, qx, qy (row then column).
        this.runRows(encoder, this.heightRowBindGroup, false, true);
        this.runCols(encoder, this.heightColBindGroup, false);

        this.runRows(encoder, this.qxRowBindGroup, false, true);
        this.runCols(encoder, this.qxColBindGroup, false);

        this.runRows(encoder, this.qyRowBindGroup, false, true);
        this.runCols(encoder, this.qyColBindGroup, false);

        // 3) Frequency-space derivatives + exponential integrator (Eq. 17 in paper).
        this.runDerivativePass(encoder);

        this.runFluxPass(encoder, this.qxFluxBindGroup);
        this.copyTextureInternal(encoder, this.qxFreqUpdatedTexture, this.qxFreqTexture);

        this.runFluxPass(encoder, this.qyFluxBindGroup);
        this.copyTextureInternal(encoder, this.qyFreqUpdatedTexture, this.qyFreqTexture);

        // 4) Inverse FFTs back to spatial domain.
        this.runCols(encoder, this.qxColInverseBindGroup, true);
        this.runRows(encoder, this.qxRowInverseBindGroup, true, false);

        this.runCols(encoder, this.qyColInverseBindGroup, true);
        this.runRows(encoder, this.qyRowInverseBindGroup, true, false);

        // 5) Write real q̃ back into the shared R32 textures for rendering/bulk stages.
        this.runWritePass(encoder, this.qxWriteBindGroup);
        this.runWritePass(encoder, this.qyWriteBindGroup);

        this.device.queue.submit([encoder.finish()]);
    }

    // First pass that combines the two half-step heights into h_t.
    private runAveragePass(encoder: GPUCommandEncoder) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.averagePipeline);
        pass.setBindGroup(0, this.averageBindGroup);
        pass.dispatchWorkgroups(
            workgroupCount(this.width, 8),
            workgroupCount(this.height, 8),
            1
        );
        pass.end();
    }
    // Executes row FFT. When `useReal` is true we read the original R32 texture instead of complex data.
    private runRows(
        encoder: GPUCommandEncoder,
        bindGroup: GPUBindGroup,
        inverse: boolean,
        useReal: boolean
    ) {
        this.updateFftUniforms(this.width, this.height, this.width, inverse, useReal);
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.fftRowsPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(this.height, 1, 1);
        pass.end();
    }


    // Executes column FFT pass.
    private runCols(
        encoder: GPUCommandEncoder,
        bindGroup: GPUBindGroup,
        inverse: boolean
    ) {
        this.updateFftUniforms(this.width, this.height, this.height, inverse, false);
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.fftColsPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(this.width, 1, 1);
        pass.end();
    }

    // Launch derivative pass (computes ∂h/∂x and ∂h/∂y in frequency space).
    private runDerivativePass(encoder: GPUCommandEncoder) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.derivativePipeline);
        pass.setBindGroup(0, this.derivativeBindGroup);
        pass.dispatchWorkgroups(
            workgroupCount(this.width, 8),
            workgroupCount(this.height, 8),
            1
        );
        pass.end();
    }

    // Launch exponential integrator pass for either qx or qy.
    private runFluxPass(encoder: GPUCommandEncoder, bindGroup: GPUBindGroup) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.fluxPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
            workgroupCount(this.width, 8),
            workgroupCount(this.height, 8),
            1
        );
        pass.end();
    }

    // Writes the real component of the inverse FFT result back to the shared R32 textures.
    private runWritePass(encoder: GPUCommandEncoder, bindGroup: GPUBindGroup) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.writePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
            workgroupCount(this.width, 8),
            workgroupCount(this.height, 8),
            1
        );
        pass.end();
    }

    // Writes current FFT parameters (dimensions + flags) into shared uniform buffer.
    private updateFftUniforms(
        width: number,
        height: number,
        count: number,
        inverse: boolean,
        useReal: boolean
    ) {
        this.fftUniformArray[0] = width;
        this.fftUniformArray[1] = height;
        this.fftUniformArray[2] = count;
        let flags = 0;
        if (inverse) {
            flags |= FFT_FLAG_INVERSE;
        }
        if (useReal) {
            flags |= FFT_FLAG_USE_REAL;
        }
        this.fftUniformArray[3] = flags;
        this.device.queue.writeBuffer(
            this.fftUniformBuffer,
            0,
            this.fftUniformArray.buffer,
            this.fftUniformArray.byteOffset,
            this.fftUniformArray.byteLength
        );
    }

    // GPU-side copy helper so we can ping-pong spectra without CPU readback.
    private copyTextureInternal(
        encoder: GPUCommandEncoder,
        srcTex: GPUTexture,
        dstTex: GPUTexture
    ) {
        encoder.copyTextureToTexture(
            { texture: srcTex },
            { texture: dstTex },
            { width: this.width, height: this.height, depthOrArrayLayers: 1 }
        );
    }

    // Helper that standardizes texture creation so all scratch buffers match format/usage.
    private createTexture(
        width: number,
        height: number,
        format: GPUTextureFormat,
        usage: GPUTextureUsageFlags
    ): GPUTexture {
        return this.device.createTexture({
            size: [width, height, 1],
            format,
            usage,
        });
    }

    // Collapse smooth depth field to a single scalar (temporary simplification of the paper).
    private computeAverageDepth(data: Float32Array): number {
        if (data.length === 0) {
            return 1.0;
        }
        let sum = 0.0;
        for (let i = 0; i < data.length; i++) {
            sum += data[i];
        }
        return Math.max(sum / data.length, 0.01);
    }
}
