import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';

export class ClusteredDeferredBlurRenderer extends renderer.Renderer {
    gbufPosTex!: GPUTexture;  gbufPosView!: GPUTextureView;
    gbufNorTex!: GPUTexture;  gbufNorView!: GPUTextureView;
    gbufAlbTex!: GPUTexture;  gbufAlbView!: GPUTextureView;
    gbufSampler!: GPUSampler;

    depthTex!: GPUTexture; depthView!: GPUTextureView;

    sceneBindGroupLayout!: GPUBindGroupLayout;
    sceneBindGroup!: GPUBindGroup;

    gbufferReadBindGroupLayout!: GPUBindGroupLayout;
    gbufferReadBindGroup!: GPUBindGroup;

    gbufferPipeline!: GPURenderPipeline;
    fullscreenPipeline!: GPURenderPipeline;

    postSceneTex!: GPUTexture;  postSceneView!: GPUTextureView; 
    blurATex!: GPUTexture;      blurAView!: GPUTextureView;  
    blurBTex!: GPUTexture;      blurBView!: GPUTextureView;   
    postSampler!: GPUSampler;

    blurBGL!: GPUBindGroupLayout;
    blurParamsBuf!: GPUBuffer;
    blurPipe!: GPUComputePipeline;

    blurBindA!: GPUBindGroup; 
    blurBindB!: GPUBindGroup; 
    blurBindC!: GPUBindGroup;

    blitBGL!: GPUBindGroupLayout;
    blitBG!: GPUBindGroup;
    blitPipe!: GPURenderPipeline;

    constructor(stage: Stage) {
        super(stage);

        this.gbufPosTex = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height], format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            label: 'gbuf-position'
        });
        this.gbufPosView = this.gbufPosTex.createView();

        this.gbufNorTex = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height], format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            label: 'gbuf-normal'
        });
        this.gbufNorView = this.gbufNorTex.createView();

        this.gbufAlbTex = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height], format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            label: 'gbuf-albedo'
        });
        this.gbufAlbView = this.gbufAlbTex.createView();

        this.depthTex = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height], format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
            label: 'gbuf-depth'
        });
        this.depthView = this.depthTex.createView();

        this.gbufSampler = renderer.device.createSampler({
            label: 'gbuf-sampler',
            magFilter: 'nearest',
            minFilter: 'nearest',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge'
        });

        this.sceneBindGroupLayout = renderer.device.createBindGroupLayout({
            label: 'clustered-deferred scene bgl',
            entries: [
                { binding: 0, visibility:GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            ]
        });

        this.sceneBindGroup = renderer.device.createBindGroup({
            label: 'clustered-deferred scene bg',
            layout: this.sceneBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer } },
                { binding: 2, resource: { buffer: this.lights.clusterBuffer } },
            ]
        });

        this.gbufferReadBindGroupLayout = renderer.device.createBindGroupLayout({
            label: 'gbuffer read bgl',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, 
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, 
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ]
        });

        this.gbufferReadBindGroup = renderer.device.createBindGroup({
            label: 'gbuffer read bg',
            layout: this.gbufferReadBindGroupLayout,
            entries: [
                { binding: 0, resource: this.gbufPosView },
                { binding: 1, resource: this.gbufNorView },
                { binding: 2, resource: this.gbufAlbView },
                { binding: 3, resource: this.gbufSampler },
            ]
        });

        this.gbufferPipeline = renderer.device.createRenderPipeline({
            label: 'clustered-deferred gbuffer pipeline',
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.sceneBindGroupLayout,
                    renderer.modelBindGroupLayout,   
                    renderer.materialBindGroupLayout,   
                ],
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.naiveVertSrc, label: 'gbuffer vert (reuse naive)' }),
                buffers: [renderer.vertexBufferLayout],
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.clusteredDeferredFragSrc, label: 'gbuffer frag' }),
                targets: [
                    { format: 'rgba16float' },  
                    { format: 'rgba16float' },  
                    { format: 'rgba8unorm'  }, 
                ],
            },
            depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
        });

        this.fullscreenPipeline = renderer.device.createRenderPipeline({
            label: 'clustered-deferred fullscreen pipeline',
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [ this.sceneBindGroupLayout, this.gbufferReadBindGroupLayout ],
            }),
            vertex: {
                module: renderer.device.createShaderModule({
                code: shaders.clusteredDeferredFullscreenVertSrc,
                label: 'deferred fsq vert'
                }),
            },
            fragment: {
                module: renderer.device.createShaderModule({
                code: shaders.clusteredDeferredFullscreenFragSrc,
                label: 'deferred fsq frag'
                }),
                targets: [{ format: 'rgba8unorm' }],
            },
        });

        this.postSceneTex = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
            label: 'post-scene'
        });
        this.postSceneView = this.postSceneTex.createView();

        const mkStorageRT = (label: string) => renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
            label
        });

        this.blurATex = mkStorageRT('blurA'); this.blurAView = this.blurATex.createView();
        this.blurBTex = mkStorageRT('blurB'); this.blurBView = this.blurBTex.createView();

        this.postSampler = renderer.device.createSampler({
            label: 'post-sampler',
            magFilter: 'linear', minFilter: 'linear',
            addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'
        });

        this.blurBGL = renderer.device.createBindGroupLayout({
        label: 'blur bgl',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },  
            { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } }, 
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]
        });

        this.blurParamsBuf = renderer.device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'blur-params'
        });

        this.blurPipe = renderer.device.createComputePipeline({
            label: 'blur compute',
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [ this.blurBGL ] }),
            compute: {
                module: renderer.device.createShaderModule({ code: shaders.blurComputeSrc, label: 'blur cs' }),
                entryPoint: 'main'
            }
        });

        const mkBlurBG = (srcView: GPUTextureView, dstView: GPUTextureView) =>
        renderer.device.createBindGroup({
            label: 'blur bg',
            layout: this.blurBGL,
            entries: [
                { binding: 0, resource: srcView },
                { binding: 1, resource: this.postSampler },
                { binding: 2, resource: dstView },
                { binding: 3, resource: { buffer: this.blurParamsBuf } },
            ]
        });

        this.blurBindA = mkBlurBG(this.postSceneView, this.blurAView);
        this.blurBindB = mkBlurBG(this.blurAView, this.blurBView);
        this.blurBindC = mkBlurBG(this.blurBView, this.postSceneView);

        this.blitBGL = renderer.device.createBindGroupLayout({
            label: 'blit bgl',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ]
        });

        this.blitBG = renderer.device.createBindGroup({
            label: 'blit bg',
            layout: this.blitBGL,
            entries: [
                { binding: 0, resource: this.blurBView }, 
                { binding: 1, resource: this.postSampler },
            ]
        });

        this.blitPipe = renderer.device.createRenderPipeline({
            label: 'blit pipeline',
            layout: renderer.device.createPipelineLayout({ bindGroupLayouts: [ this.blitBGL ] }),
            vertex:  { module: renderer.device.createShaderModule({ code: shaders.clusteredDeferredFullscreenVertSrc }) },
            fragment:{ module: renderer.device.createShaderModule({ code: shaders.blitToCanvasFragSrc }),
                        targets: [{ format: renderer.canvasFormat }] },
        });
    }

    override draw() {
        const encoder = renderer.device.createCommandEncoder();

        this.lights.doLightClustering(encoder);

        {
        const pass = encoder.beginRenderPass({
            label: 'gbuffer pass',
            colorAttachments: [
            { view: this.gbufPosView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
            { view: this.gbufNorView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
            { view: this.gbufAlbView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
            ],
            depthStencilAttachment: {
            view: this.depthView,
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
            }
        });

        pass.setPipeline(this.gbufferPipeline);

        pass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneBindGroup);

        this.scene.iterate(node => {
            pass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);   
        }, material => {
            pass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            pass.setVertexBuffer(0, primitive.vertexBuffer);
            pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            pass.drawIndexed(primitive.numIndices);
        });

        pass.end();
        }

        {
        const pass = encoder.beginRenderPass({
            label: 'deferred fullscreen blur pass',
            colorAttachments: [{
            view: this.postSceneView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
            }],
        });

        pass.setPipeline(this.fullscreenPipeline);
        pass.setBindGroup(0, this.sceneBindGroup);        
        pass.setBindGroup(1, this.gbufferReadBindGroup); 
        pass.draw(3); 
        pass.end();
        }

        {
        const pass = encoder.beginComputePass({ label: 'blur compute' });
        pass.setPipeline(this.blurPipe);

        renderer.device.queue.writeBuffer(
            this.blurParamsBuf, 0,
            new Uint32Array([renderer.canvas.width, renderer.canvas.height, shaders.constants.blurRadius, 0])
        );
        pass.setBindGroup(0, this.blurBindA);
        pass.dispatchWorkgroups(Math.ceil(renderer.canvas.width / 8), Math.ceil(renderer.canvas.height / 8));

        renderer.device.queue.writeBuffer(
            this.blurParamsBuf, 0,
            new Uint32Array([renderer.canvas.width, renderer.canvas.height, shaders.constants.blurRadius, 1])
        );
        pass.setBindGroup(0, this.blurBindB);
        pass.dispatchWorkgroups(Math.ceil(renderer.canvas.width / 8), Math.ceil(renderer.canvas.height / 8));

        pass.end();
        }

        {
        const canvasView = renderer.context.getCurrentTexture().createView();
        const pass = encoder.beginRenderPass({
            label: 'present blit',
            colorAttachments: [{
            view: canvasView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
            }],
        });
        pass.setPipeline(this.blitPipe);
        pass.setBindGroup(0, this.blitBG);
        pass.draw(3);
        pass.end();
        }

        renderer.device.queue.submit([encoder.finish()]);
    }
}
