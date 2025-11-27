import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { mat4, vec3 } from "wgpu-matrix";
import { Camera, CameraUniforms } from '../stage/camera';
import { DiffuseCS } from '../simulator/Diffuse';
import { Simulator } from '../simulator/simulator';
// import { AiryWave } from '../simulator/AiryWave';
import { ShallowWater } from '../simulator/ShallowWater';
import { AiryWaveCS } from '../simulator/AiryWaveCS';
import { TransportCS } from '../simulator/Transport';
import { VelocityCS } from '../simulator/Velocity';
import { FlowRecombineCS } from '../simulator/FlowRecombineCS';
import { HeightRecombineCS } from '../simulator/HeightRecombineCS';

export class NaiveRenderer extends renderer.Renderer {
    sceneUniformsBindGroupLayout: GPUBindGroupLayout;
    sceneUniformsBindGroup: GPUBindGroup;

    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    pipeline: GPURenderPipeline;

    // --- Heightmap state (CPU & GPU resources) ---
    // current heightmap width & height (texels)
    private heightW = 256; 
    private heightH = 256;

    // R32Float texture storing heights (1 float per texel)
    heightTexture: GPUTexture;
    heightPrevTexture: GPUTexture;
    terrainTexture: GPUTexture;
    terrainZeroTexture: GPUTexture;
    lowFreqTexture: GPUTexture;
    lowFreqTexturePingpong: GPUTexture;
    highFreqTexture: GPUTexture;
    // non-filtering sampler
    heightSampler: GPUSampler;
    // UBO for HeightConsts (uvTexel, worldScale, heightScale, baseLevel)
    heightConsts: GPUBuffer;
    // layout for { sampler, heightTex, heightConsts }
    heightBindGroupLayout: GPUBindGroupLayout;
    // bound view/sampler/UBO used by the water vertex shader
    heightBindGroup: GPUBindGroup;

    // pipeline that draws the displaced grid (water surface)
    heightPipeline: GPURenderPipeline;

    // vertex buffer storing per-vertex UVs
    heightVBO: GPUBuffer;
    // index buffer for the grid
    heightIBO: GPUBuffer;
    // how many indices to draw
    heightIndexCount = 0;

    // --- Flux decompose state ---
    qxTexture: GPUTexture;
    qxLowFreqTexture: GPUTexture;
    qxLowFreqTexturePingpong: GPUTexture;
    qxHighFreqTexture: GPUTexture;

    qyTexture: GPUTexture;
    qyLowFreqTexture: GPUTexture;
    qyLowFreqTexturePingpong: GPUTexture;
    qyHighFreqTexture: GPUTexture;


    // --- lowFreq previous height + Velocity textures for shallow water

    private lowFreqPrevHeightTexture: GPUTexture;
    private lowFreqVelocityXTexture: GPUTexture;
    private lowFreqVelocityYTexture: GPUTexture;
    private changeInLowFreqVelocityXTexture: GPUTexture;
    private changeInLowFreqVelocityYTexture: GPUTexture;



    // --- Velocity & transport lambda texture ---
    // u = (u_x, u_y)，cell-centered for now
    private uXTex: GPUTexture;
    private uYTex: GPUTexture;

    // Transport output: lambda（ping-pong）
    private qHighTransportTexX: GPUTexture; 
    private qHighTransportTexY: GPUTexture;
    private hHighTransportTex: GPUTexture;   // λ=h

    // --- Upload helpers for writeTexture() row alignment ---
    private rowBytes = 0;
    private paddedBytesPerRow = 0;
    private uploadScratch: Uint8Array | null = null;

    // Diffuse compute
    private diffuseHeight: DiffuseCS;
    private diffuseFluxX: DiffuseCS;
    private diffuseFluxY: DiffuseCS;

    private velocityCS: VelocityCS;

    private simulator: Simulator;


    private shallowWater: ShallowWater;
    // AiryWave compute
    // private airyWave: AiryWave;
    private airyWaveCS: AiryWaveCS;

    // Transport
    private transportFlowRateX: TransportCS;
    private transportFlowRateY: TransportCS;
    private transportHeight: TransportCS;

    // Combine
    private flowRecombineX: FlowRecombineCS;
    private flowRecombineY: FlowRecombineCS;
    private heightRecombine: HeightRecombineCS;

    // --- Reflection state (CPU & GPU resources) ---
    // Color texture where we render the mirrored scene (used later by water shader)
    private reflectionTexture: GPUTexture;
    private reflectionTextureView: GPUTextureView;
    // Depth buffer used only for the reflection camera
    private reflectionDepthTexture: GPUTexture;
    private reflectionDepthTextureView: GPUTextureView;
    // Sampler used when sampling the reflection texture in the water fragment shader
    private reflectionSampler: GPUSampler;

    // UBO holding camera data for the reflection camera (same layout as CameraUniforms)
    private reflectionCameraUniforms: CameraUniforms;
    private reflectionCameraUniformsBuffer: GPUBuffer;
    private reflectionSceneUniformsBindGroup: GPUBindGroup;

    // UBO that stores only the reflection viewProj matrix
    private reflectionViewProjBuffer: GPUBuffer;
    private reflectionBindGroupLayout: GPUBindGroupLayout;
    private reflectionBindGroup: GPUBindGroup;

    private waterBaseLevel: number = 0;

    // --- Pass flags ---
    // UBO for the reflection pass (isReflection = 1)
    private passFlagsBufferReflection: GPUBuffer;
    // UBO for the main pass (isReflection = 0)
    private passFlagsBufferMain: GPUBuffer;
    private passFlagsBindGroupLayout: GPUBindGroupLayout;
    private passFlagsBindGroupReflection: GPUBindGroup;
    private passFlagsBindGroupMain: GPUBindGroup;

    constructor(stage: Stage) {
        super(stage);

        this.sceneUniformsBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "scene uniforms bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                }
            ]
        });

        this.sceneUniformsBindGroup = renderer.device.createBindGroup({
            label: "scene uniforms bind group",
            layout: this.sceneUniformsBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.camera.uniformsBuffer }
                }
            ]
        });

        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        this.depthTextureView = this.depthTexture.createView();

        // --- PassFlags: tiny UBO telling the material shader which pass is active ---
        this.passFlagsBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "pass flags bgl",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" },
                },
            ],
        });

        // Reflection pass version: isReflection = 1
        this.passFlagsBufferReflection = renderer.device.createBuffer({
            label: "pass flags buffer (reflection)",
            size: 4 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        renderer.device.queue.writeBuffer(
            this.passFlagsBufferReflection,
            0,
            new Float32Array([1.0, 0, 0, 0]) 
        );
        this.passFlagsBindGroupReflection = renderer.device.createBindGroup({
            label: "pass flags bg (reflection)",
            layout: this.passFlagsBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.passFlagsBufferReflection },
                },
            ],
        });

        // Main pass version: isReflection = 0
        this.passFlagsBufferMain = renderer.device.createBuffer({
            label: "pass flags buffer (main)",
            size: 4 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        renderer.device.queue.writeBuffer(
            this.passFlagsBufferMain,
            0,
            new Float32Array([0.0, 0, 0, 0]) 
        );
        this.passFlagsBindGroupMain = renderer.device.createBindGroup({
            label: "pass flags bg (main)",
            layout: this.passFlagsBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.passFlagsBufferMain },
                },
            ],
        });

        // Main scene pipeline (used for glTF models).
        this.pipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                label: "naive pipeline layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    renderer.materialBindGroupLayout,
                    this.passFlagsBindGroupLayout
                ]
            }),
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus"
            },
            vertex: {
                module: renderer.device.createShaderModule({
                    label: "naive vert shader",
                    code: shaders.naiveVertSrc
                }),
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "naive frag shader",
                    code: shaders.naiveFragSrc,
                }),
                targets: [
                    {
                        format: renderer.canvasFormat,
                    }
                ]
            }
        });

        // --- Height Map Pipeline --- 
        {
            // Build a (nu × nv) regular grid in UV space. Each vertex stores (u,v) in [0,1].
            // The vertex shader converts (u,v) -> world (x,z) using worldScale and displaces y using the heightmap.
            const grid = makeGrid(512, 512);
            this.heightIndexCount = grid.indices.length;

            this.heightVBO = renderer.device.createBuffer({
                size: grid.uvs.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            renderer.device.queue.writeBuffer(this.heightVBO, 0, grid.uvs);

            this.heightIBO = renderer.device.createBuffer({
                size: grid.indices.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
            });
            renderer.device.queue.writeBuffer(this.heightIBO, 0, grid.indices);
        }

        const texUsage =
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.STORAGE_BINDING;

        // R32Float height texture (unfilterable). One float per texel.
        this.heightTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });

        this.heightPrevTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });

        // lowFreq IN
        this.lowFreqTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.lowFreqTexturePingpong = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        // highFreq
        this.highFreqTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });

        // terrain
        this.terrainTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        
        this.terrainZeroTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });

        // flux x
        this.qxTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.qxLowFreqTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.qxLowFreqTexturePingpong = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.qxHighFreqTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });

        // flux y
        this.qyTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.qyLowFreqTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.qyLowFreqTexturePingpong = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.qyHighFreqTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });

        //Shallow water extra textures
        this.lowFreqPrevHeightTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.lowFreqVelocityXTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.lowFreqVelocityYTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.changeInLowFreqVelocityXTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.changeInLowFreqVelocityYTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });


        // velocity u & transport output λ texture
        this.uXTex = renderer.device.createTexture({
            size: [this.heightW, this.heightH],   // Simple version for now
            format: "r32float",
            usage: texUsage,
        });
        this.uYTex = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });

        this.qHighTransportTexX = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.qHighTransportTexY = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });
        this.hHighTransportTex = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: texUsage,
        });

        // Non-filtering sampler (required for unfilterable formats like R32Float).
        this.heightSampler = renderer.device.createSampler({
            minFilter: "nearest",
            magFilter: "nearest",
            mipmapFilter: "nearest",
        });

        // UBO for HeightConsts (32 bytes total, 16-byte aligned):
        // layout: [uvTexel(8B), worldScale(8B), heightScale(4B), baseLevel(4B), padding(8B)]
        this.heightConsts = renderer.device.createBuffer({
            size: 4*8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        
        /*
        Defaults for HeightConsts:
        0 = 1/width        (uvTexel.x)
        1 = 1/height       (uvTexel.y)
        2 = worldScale.x   (half extent in world X; final width = 2*sx)
        3 = worldScale.y   (half extent in world Z; final depth = 2*sz)
        4 = heightScale    (amplitude multiplier)
        5 = baseLevel      (world Y offset)
        6..7 = unused (padding)
        */
        const defaults = new Float32Array([1/256,1/256, 500,500, 10, 0, 0, 0]);
        renderer.device.queue.writeBuffer(this.heightConsts, 0, defaults);

        // Bind group layout for { sampler, heightTex, heightConsts } used by the water VS.
        // Note: sampleType: "unfilterable-float" + sampler type "non-filtering" for R32Float.
        this.heightBindGroupLayout = renderer.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, sampler: { type: "non-filtering" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "unfilterable-float" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            ]
        });

        // Concrete bind group connecting the sampler, the texture view, and the UBO.
        this.heightBindGroup = renderer.device.createBindGroup({
            layout: this.heightBindGroupLayout,
            entries: [
                { binding: 0, resource: this.heightSampler },
                { binding: 1, resource: this.heightTexture.createView() },
                { binding: 2, resource: { buffer: this.heightConsts } },
            ]
        });

        // --- Reflection Pipeline ---
        const w = renderer.canvas.width;
        const h = renderer.canvas.height;

        // Color texture where the reflection pass renders the mirrored scene.
        // The water shader later samples from this.
        this.reflectionTexture = renderer.device.createTexture({
            label: "reflection color",
            size: [w, h],
            format: renderer.canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.reflectionTextureView = this.reflectionTexture.createView();

        // Depth texture used only in the reflection pass
        this.reflectionDepthTexture = renderer.device.createTexture({
            label: "reflection depth",
            size: [w, h],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.reflectionDepthTextureView = this.reflectionDepthTexture.createView();

        // Sampler used to sample the reflection texture in the water fragment shader
        this.reflectionSampler = renderer.device.createSampler({
            label: "reflection sampler",
            magFilter: "linear",
            minFilter: "linear",
        });

        // Separate camera uniforms instance for the reflection camera
        this.reflectionCameraUniforms = new CameraUniforms();
        this.reflectionCameraUniformsBuffer = renderer.device.createBuffer({
            label: "reflection camera uniforms buffer",
            size: this.reflectionCameraUniforms.buffer.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind group that uses the reflection camera UBO for the reflection pass.
        // Same layout as the main sceneUniformsBindGroup, just bound to a different buffer.
        this.reflectionSceneUniformsBindGroup = renderer.device.createBindGroup({
            label: "reflection scene uniforms bind group",
            layout: this.sceneUniformsBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.reflectionCameraUniformsBuffer }
                }
            ]
        });

        // UBO that stores only the reflection viewProj matrix,
        // used by the water shader's ReflectionUniforms struct.
        this.reflectionViewProjBuffer = renderer.device.createBuffer({
            label: "reflection viewProj buffer",
            size: 16 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind group layout for water reflection:
        //   binding(0): sampler
        //   binding(1): reflection color texture
        //   binding(2): ReflectionUniforms (viewProj matrix)
        this.reflectionBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "reflection bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: "filtering" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" },
                },
            ],
        });

        // Concrete bind group for the water pass:
        //   sampler + reflection texture view + viewProj UBO.
        this.reflectionBindGroup = renderer.device.createBindGroup({
            label: "reflection bind group",
            layout: this.reflectionBindGroupLayout,
            entries: [
                { binding: 0, resource: this.reflectionSampler },
                { binding: 1, resource: this.reflectionTextureView },
                { binding: 2, resource: { buffer: this.reflectionViewProjBuffer } },
            ],
        });

        // Pipeline used to render the water surface (height-displaced grid).
        // Vertex shader samples the height texture at UV, computes world position & normal.
        // Fragment shader can do lighting.
        this.heightPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    // camera UBO
                    this.sceneUniformsBindGroupLayout, 
                    // height sampler/texture/UBO
                    this.heightBindGroupLayout,         
                    // reflection sampler/texture/UBO
                    this.reflectionBindGroupLayout
                ]
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.waterVertSrc }),
                entryPoint: "vs_main",
                buffers: [ heightVertexLayout ],
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.waterFragSrc }),
                entryPoint: "fs_main",
                targets: [{
                    format: renderer.canvasFormat,
                    blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                        alpha: {
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                    },
                }],
            },
            depthStencil: { 
                format: "depth24plus", 
                depthWriteEnabled: true, 
                depthCompare: "less" 
            },
            primitive: { topology: "triangle-list"}
        });

        // --- Diffuse step ---
        this.initRowInfoIfNeeded();

        const terrainArr = new Float32Array(this.heightW * this.heightH);
        const terrainZeroArr = new Float32Array(this.heightW * this.heightH); 
        const lowArr = new Float32Array(this.heightW * this.heightH);
        const highArr = new Float32Array(this.heightW * this.heightH);

        const Wtex = this.heightW;
        const Htex = this.heightH;
        const centerX = Wtex / 2;
        const centerY = Htex / 2;
        const bumpRadius = Math.min(Wtex, Htex) * 0.1; // Bump covers 20% of domain
        const bumpHeight = 3.5; // Height of the bump
        
        const fluxInitX = new Float32Array(this.heightW * this.heightH);
        const fluxInitY = new Float32Array(this.heightW * this.heightH);
        
        for (let y = 0; y < Htex; y++) {
            for (let x = 0; x < Wtex; x++) {
                const idx = y * Wtex + x;
                
                // Distance from center
                const dx = x - centerX;
                const dy = y - centerY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                // Gaussian bump: h = A * exp(-(r/R)^2)
                const bump = bumpHeight * Math.exp(-(dist * dist) / (bumpRadius * bumpRadius));
                

                terrainArr[idx] = 0.0; // Flat terrain
                lowArr[idx] = bump + 10; // Water height = bump + base level
                highArr[idx] = 0.0;
                fluxInitX[idx] = bump * 0.0;
                fluxInitY[idx] = 0.0;
            }
        }

        this.updateTexture(terrainArr, this.terrainTexture);
        this.updateTexture(terrainZeroArr,  this.terrainZeroTexture);
        this.updateTexture(lowArr, this.heightTexture);  
        this.updateTexture(lowArr, this.lowFreqTexture);
        this.updateTexture(lowArr, this.lowFreqTexturePingpong);
        this.updateTexture(highArr, this.highFreqTexture);

        this.updateTexture(fluxInitX, this.qxTexture);
        this.updateTexture(fluxInitX, this.qxLowFreqTexture);
        this.updateTexture(fluxInitX, this.qxLowFreqTexturePingpong);
        this.updateTexture(fluxInitX, this.qxHighFreqTexture);

        this.updateTexture(fluxInitY, this.qyTexture);
        this.updateTexture(fluxInitY, this.qyLowFreqTexture);
        this.updateTexture(fluxInitY, this.qyLowFreqTexturePingpong);
        this.updateTexture(fluxInitY, this.qyHighFreqTexture);
        
        this.updateTexture(fluxInitX, this.uXTex);
        this.updateTexture(fluxInitY, this.uYTex);

        this.diffuseHeight = new DiffuseCS(
            renderer.device,
            this.heightW,
            this.heightH,
            this.heightTexture,           // fieldTex: height
            this.lowFreqTexture,          // lowFreqTex
            this.lowFreqTexturePingpong,  // lowFreqTexPingpong
            this.highFreqTexture,         // highFreqTex
            this.terrainTexture
        );

        this.diffuseFluxX = new DiffuseCS(
            renderer.device,
            this.heightW,
            this.heightH,
            this.qxTexture,               // qx
            this.qxLowFreqTexture,
            this.qxLowFreqTexturePingpong,
            this.qxHighFreqTexture,
            this.terrainZeroTexture   
        );

        this.diffuseFluxY = new DiffuseCS(
            renderer.device,
            this.heightW,
            this.heightH,
            this.qyTexture,               // qy
            this.qyLowFreqTexture,
            this.qyLowFreqTexturePingpong,
            this.qyHighFreqTexture,
            this.terrainZeroTexture
        );

        this.velocityCS = new VelocityCS(
            renderer.device,
            this.heightW,
            this.heightH,
            this.lowFreqTexture,      // h_low
            this.qxLowFreqTexture,    // qx_low
            this.qyLowFreqTexture,    // qy_low
            this.uXTex,
            this.uYTex
        );

        this.shallowWater = new ShallowWater(
            renderer.device,
            this.heightW,
            this.heightW,
            this.lowFreqTexture,
            this.lowFreqPrevHeightTexture,
            this.qxLowFreqTexture,
            this.qyLowFreqTexture,
            this.lowFreqVelocityXTexture,
            this.lowFreqVelocityYTexture,
            this.changeInLowFreqVelocityXTexture,
            this.changeInLowFreqVelocityYTexture
        );
        
        // this.shallowWater = new ShallowWater(
        //     renderer.device,
        //     this.heightW,
        //     this.heightH,
        //     this.heightTexture,
        //     this.lowFreqPrevHeightTexture,
        //     this.qxTexture,
        //     this.qyTexture,
        //     this.lowFreqVelocityXTexture,
        //     this.lowFreqVelocityYTexture,
        //     this.changeInLowFreqVelocityXTexture,
        //     this.changeInLowFreqVelocityYTexture
        // );

        
        /**
         * Smooth depth field \bar{h}(x,y).
         * In a full implementation, this should come from the low-frequency
         * water height minus terrain, i.e. the actual water depth.
         */
        const smoothDepth = new Float32Array(this.heightW * this.heightH);
        for (let i = 0; i < smoothDepth.length; ++i) {
            // Element-wise difference: water height minus terrain height
            const depth = lowArr[i] - terrainArr[i];
            // Clamp to a small positive value to avoid zero / negative depth,
            // which would cause problems in the dispersion relation.
            smoothDepth[i] = Math.max(depth, 0.01);
        }
        
        this.airyWaveCS = new AiryWaveCS(
            renderer.device,
            this.heightW,
            this.heightH,
            this.highFreqTexture,
            this.qxHighFreqTexture,
            this.qyHighFreqTexture,
            smoothDepth
        );

        this.transportFlowRateX = new TransportCS(
            renderer.device,
            this.heightW,
            this.heightH,
            this.qxHighFreqTexture,  // λ_in = qx_high
            this.uXTex,
            this.uYTex,
            this.qHighTransportTexX, // λ_out_x
            'q',
            0.25,
            1.0
        );

        this.transportFlowRateY = new TransportCS(
            renderer.device,
            this.heightW,
            this.heightH,
            this.qyHighFreqTexture,  // λ_in = qy_high
            this.uXTex,
            this.uYTex,
            this.qHighTransportTexY, // λ_out_y
            'q',
            0.25,
            1.0
        );

        this.transportHeight = new TransportCS(
            renderer.device,
            this.heightW,
            this.heightH,
            this.highFreqTexture,    
            this.uXTex,
            this.uYTex,
            this.hHighTransportTex, 
            'h',
            0.25,
            1.0
        );

        this.flowRecombineX = new FlowRecombineCS(
            renderer.device,
            this.heightW, this.heightH,
            this.qxLowFreqTexture,     // \bar{q}_x^{t+Δt}
            this.qHighTransportTexX,    // \tilde{q}_x^{t+Δt}
            this.qxTexture             // q_x^{t+Δt}
        );

        this.flowRecombineY = new FlowRecombineCS(
            renderer.device,
            this.heightW, this.heightH,
            this.qyLowFreqTexture,     // \bar{q}_x^{t+Δt}
            this.qHighTransportTexY,    // \tilde{q}_x^{t+Δt}
            this.qyTexture             // q_x^{t+Δt}
        );

        this.heightRecombine = new HeightRecombineCS(
            renderer.device,
            this.heightW, this.heightH,
            this.heightPrevTexture,   // h_prev
            this.qxTexture,        // q_x^{t+Δt}
            this.qyTexture,        // q_y^{t+Δt}
            this.hHighTransportTex,  // \tilde h
            this.uXTex,            // bulk velocity x
            this.uYTex,            // bulk velocity y
            this.heightTexture,    // h^{t+3Δt/2}
            1.0   
        );

        this.simulator = new Simulator(
            this.heightW,
            this.heightH,
            this.diffuseHeight,
            this.diffuseFluxX,
            this.diffuseFluxY,
            this.velocityCS,
            this.shallowWater,
            this.airyWaveCS,
            this.transportFlowRateX,
            this.transportFlowRateY,
            this.transportHeight,
            this.flowRecombineX,
            this.flowRecombineY,
            this.heightRecombine
        );
    }

    override draw() {
        const canvasTextureView = renderer.context.getCurrentTexture().createView();
        const encoder = renderer.device.createCommandEncoder();

        this.updateReflectionUniforms();

        // --- Reflection pass ---
        // Render the scene from the mirrored camera into reflectionTexture.
        const reflectionPass = encoder.beginRenderPass({
            label: "reflection pass",
            colorAttachments: [
                {
                    view: this.reflectionTextureView,
                    clearValue: [0, 0, 0, 1],
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
            depthStencilAttachment: {
                view: this.reflectionDepthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store",
            },
        });

        reflectionPass.setPipeline(this.pipeline);
        // Use the reflection camera instead of the main camera
        reflectionPass.setBindGroup(
            shaders.constants.bindGroup_scene,
            this.reflectionSceneUniformsBindGroup
        );
        // Tell the material shader "this is the reflection pass"
        // so it can discard fragments below the water plane.
        reflectionPass.setBindGroup(3, this.passFlagsBindGroupReflection);

        // Draw the glTF scene from the mirrored camera
        this.scene.iterate(node => {
            reflectionPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            reflectionPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            reflectionPass.setVertexBuffer(0, primitive.vertexBuffer);
            reflectionPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            reflectionPass.drawIndexed(primitive.numIndices);
        });

        reflectionPass.end();

        // --- Main pass ---
        // Render the scene normally to the swapchain, then draw the water on top
        const renderPass = encoder.beginRenderPass({
            label: "naive render pass",
            colorAttachments: [
                {
                    view: canvasTextureView,
                    clearValue: [0, 0, 0, 0],
                    loadOp: "clear",
                    storeOp: "store"
                }
            ],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            }
        });
        renderPass.setPipeline(this.pipeline);

        renderPass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
        // Tell the material shader "this is the main pass"
        renderPass.setBindGroup(3, this.passFlagsBindGroupMain);

        this.scene.iterate(node => {
            renderPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            renderPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            renderPass.setVertexBuffer(0, primitive.vertexBuffer);
            renderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            renderPass.drawIndexed(primitive.numIndices);
        });

        // Add pipeline for height map in render pass
        renderPass.setPipeline(this.heightPipeline);
        renderPass.setBindGroup(0, this.sceneUniformsBindGroup);
        renderPass.setBindGroup(1, this.heightBindGroup);
        renderPass.setBindGroup(2, this.reflectionBindGroup);
        renderPass.setVertexBuffer(0, this.heightVBO);
        renderPass.setIndexBuffer(this.heightIBO, "uint32");
        renderPass.drawIndexed(this.heightIndexCount);

        renderPass.end();

        renderer.device.queue.submit([encoder.finish()]);
    }

    // Update world scale (sx, sz), height amplitude, and base level (Y offset).
    // Offsets match the HeightConsts layout described above.
    setHeightParams(sx: number, sz: number, heightScale: number, baseLevel: number) {
        renderer.device.queue.writeBuffer(this.heightConsts, 8,  new Float32Array([sx, sz]));
        renderer.device.queue.writeBuffer(this.heightConsts, 16, new Float32Array([heightScale]));
        renderer.device.queue.writeBuffer(this.heightConsts, 20, new Float32Array([baseLevel]));

        this.waterBaseLevel = baseLevel;
    }

    // Prepare per-frame upload helpers based on current (heightW, heightH).
    // Ensures we have:
    //  - rowBytes & paddedBytesPerRow computed (256-byte alignment for WebGPU)
    //  - uploadScratch allocated if padding is needed
    //  - heightArray allocated for the simulation output
    private initRowInfoIfNeeded() {
        //For writing buffers to textures, byte rows must be multiples of 256
        //when it's not, we must create a new buffer that is that size and later copy the old buffer into it (with padding)
        if (!this.rowBytes) {
            this.rowBytes = this.heightW * 4;
            const a = 256;
            this.paddedBytesPerRow = Math.ceil(this.rowBytes / a) * a;
            if (this.paddedBytesPerRow !== this.rowBytes) {
                this.uploadScratch = new Uint8Array(this.paddedBytesPerRow * this.heightH);
            }
        }
    }

    private updateTexture(heightData: Float32Array, heightTexture: GPUTexture)
    {
        const data = new Float32Array(heightData);
        this.initRowInfoIfNeeded();

        // Aligned -> upload directly
        if (this.paddedBytesPerRow === this.rowBytes) {
            renderer.device.queue.writeTexture(
                { texture: heightTexture },
                data,
                { bytesPerRow: this.rowBytes },
                [this.heightW, this.heightH, 1]
            );
        } else {
            // Not aligned -> copy row-by-row into the padded scratch buffer
            const tmp = new Uint8Array(this.uploadScratch!);
            const src = new Uint8Array(heightData.buffer, heightData.byteOffset, heightData.byteLength);
            for (let y = 0; y < this.heightH; y++) {
                //Copies the non padded buffer to the padded buffer, leaving padding before each row.
                const s = y * this.rowBytes, d = y * this.paddedBytesPerRow;
                tmp.set(src.subarray(s, s + this.rowBytes), d);
            }
            renderer.device.queue.writeTexture(
                { texture: heightTexture },
                tmp,
                { bytesPerRow: this.paddedBytesPerRow },
                [this.heightW, this.heightH, 1]
            );
        }
    }

    // Called every frame before drawing.
    protected override onBeforeDraw(dtMs: number): void {
        const dt = dtMs / 1000; 

        const encoder = renderer.device.createCommandEncoder();

        encoder.copyTextureToTexture(
            { texture: this.heightTexture },      // src: h^n
            { texture: this.heightPrevTexture },  // dst: h_prev
            {
                width:  this.heightW,
                height: this.heightH,
                depthOrArrayLayers: 1,
            }
        );

        renderer.device.queue.submit([encoder.finish()]);

        this.simulator.simulate(dt);
    }

    private updateReflectionUniforms() {
        const mainCam = this.camera; 
        const eye = mainCam.cameraPos;     
        const front = mainCam.cameraFront; 
        const up = mainCam.cameraUp;        

        const h = this.waterBaseLevel; 

        // Mirror the camera position and orientation across the plane y = h.
        // For a horizontal plane, reflection is:
        //   y' = 2*h - y, and we flip the Y component of forward & up.
        const eyeRef = vec3.fromValues(eye[0], 2 * h - eye[1], eye[2]);
        const frontRef = vec3.fromValues(front[0], -front[1], front[2]);
        const upRef = vec3.fromValues(up[0], -up[1], up[2]);

        // Build the "look at" target for the reflection camera
        const lookRef = vec3.add(eyeRef, frontRef);

        // Build view matrix from mirrored position and orientation
        const viewRef = mat4.lookAt(eyeRef, lookRef, upRef);
        const proj = mainCam.projMat; 
        const viewProjRef = mat4.mul(proj, viewRef);
        const invViewRef = mat4.inverse(viewRef);
        const invProj = mat4.inverse(proj);

        // Fill the reflection camera uniforms (used by reflection pass)
        this.reflectionCameraUniforms.viewProjMat = viewProjRef;
        this.reflectionCameraUniforms.viewMat = viewRef;
        this.reflectionCameraUniforms.invProjMat = invProj;
        this.reflectionCameraUniforms.invViewMat = invViewRef;
        this.reflectionCameraUniforms.canvasResolution = [renderer.canvas.width, renderer.canvas.height];
        this.reflectionCameraUniforms.nearPlane = Camera.nearPlane;
        this.reflectionCameraUniforms.farPlane = Camera.farPlane;

        // Upload reflection camera UBO for the reflection pass
        renderer.device.queue.writeBuffer(
            this.reflectionCameraUniformsBuffer,
            0,
            this.reflectionCameraUniforms.buffer
        );

        // Upload the reflection viewProj matrix for the water shader (ReflectionUniforms)
        renderer.device.queue.writeBuffer(
            this.reflectionViewProjBuffer,
            0,
            viewProjRef.buffer,  
            viewProjRef.byteOffset,   
            viewProjRef.byteLength
        );
    }
}

// Build a (nu × nv) grid of UVs and triangle indices.
// UVs are in [0,1]; the vertex shader maps them to world space and applies displacement.
function makeGrid(nu: number, nv: number) {
    const uvs: number[] = [];
    for (let j = 0; j <= nv; j++) {
        for (let i = 0; i <= nu; i++) {
        uvs.push(i/nu, j/nv);
        }
    }
    const idx: number[] = [];
    const W = nu + 1;
    for (let j = 0; j < nv; j++) {
        for (let i = 0; i < nu; i++) {
        const a = j*W + i, b = a+1, c = a+W, d = c+1;
        idx.push(a,c,b, b,c,d);
        }
    }
    return { uvs: new Float32Array(uvs), indices: new Uint32Array(idx) };
}

// Vertex layout for the water grid: only (u,v) per vertex.
const heightVertexLayout: GPUVertexBufferLayout = {
    arrayStride: 2*4,
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
};
