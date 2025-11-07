import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';

export class ClusteredDeferredRenderer extends renderer.Renderer {
    // TODO-3: add layouts, pipelines, textures, etc. needed for Forward+ here
    // you may need extra uniforms such as the camera view matrix and the canvas resolution
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

    constructor(stage: Stage) {
        super(stage);

        // TODO-3: initialize layouts, pipelines, textures, etc. needed for Forward+ here
        // you'll need two pipelines: one for the G-buffer pass and one for the fullscreen pass
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
                bindGroupLayouts: [
                this.sceneBindGroupLayout,        
                this.gbufferReadBindGroupLayout,  
                ],
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.clusteredDeferredFullscreenVertSrc, label: 'deferred fsq vert' }),
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.clusteredDeferredFullscreenFragSrc, label: 'deferred fsq frag' }),
                targets: [{ format: renderer.canvasFormat }],
            },
        });
    }

    override draw() {
        // TODO-3: run the Forward+ rendering pass:
        // - run the clustering compute shader
        // - run the G-buffer pass, outputting position, albedo, and normals
        // - run the fullscreen pass, which reads from the G-buffer and performs lighting calculations
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
        const canvasView = renderer.context.getCurrentTexture().createView();
        const pass = encoder.beginRenderPass({
            label: 'deferred fullscreen pass',
            colorAttachments: [{
            view: canvasView,
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

        renderer.device.queue.submit([encoder.finish()]);
    }
}
