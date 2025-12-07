import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { mat4, vec3, Vec3 } from "wgpu-matrix";
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
import { AddOnCS } from "../simulator/AddOnCs";

export class NaiveRenderer extends renderer.Renderer {
    sceneUniformsBindGroupLayout: GPUBindGroupLayout;
    sceneUniformsBindGroup: GPUBindGroup;

    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    pipeline: GPURenderPipeline;

    // --- Heightmap state (CPU & GPU resources) ---
    // current heightmap width & height (texels)
    private heightW = 512; 
    private heightH = 512;

    // R32Float texture storing heights (1 float per texel)
    heightTexture: GPUTexture;
    heightPrevTexture: GPUTexture;
    terrainTexture: GPUTexture;
    terrainZeroTexture: GPUTexture;
    lowFreqTexture: GPUTexture;
    lowFreqTexturePingpong: GPUTexture;
    highFreqTexture: GPUTexture;
    private highFreqPrevTexture: GPUTexture;
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

    // CPU-side caches for click scene perturbations
    private heightArr: Float32Array;
    private lowArr: Float32Array;

    // Cached water extents for screen->world mapping
    private waterScaleX = 10;
    private waterScaleZ = 10;
    private navTerrainMax = shaders.constants.water_base_level + 0.02;
    private initialized = false;

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

    // Environment map (equirectangular HDR)
    private envTexture: GPUTexture;
    private envSampler: GPUSampler;
    private envBindGroupLayout: GPUBindGroupLayout;
    private envBindGroup: GPUBindGroup;
    private skyboxPipeline: GPURenderPipeline;

    // --- Pass flags ---
    // UBO for the reflection pass (isReflection = 1)
    private passFlagsBufferReflection: GPUBuffer;
    // UBO for the main pass (isReflection = 0)
    private passFlagsBufferMain: GPUBuffer;
    private passFlagsBindGroupLayout: GPUBindGroupLayout;
    private passFlagsBindGroupReflection: GPUBindGroup;
    private passFlagsBindGroupMain: GPUBindGroup;

    private heightAddCS: AddOnCS;
    private bumpTexture: GPUTexture;

    // Terrain surface rendering (terrain heightmap -> mesh)
    private terrainSurfacePipeline: GPURenderPipeline;
    private terrainSurfaceBindGroupLayout: GPUBindGroupLayout;
    private terrainSurfaceBindGroup: GPUBindGroup;

    // Fixed plane used for ship height and reflection
    private fixedWaterPlaneHeight: number;

    // Projectile rendering state
    private projectilePipeline!: GPURenderPipeline;
    private projectileBindGroupLayout!: GPUBindGroupLayout;
    private projectileBindGroup!: GPUBindGroup;
    private projectileUniformBuffer!: GPUBuffer;
    private projectileVBO!: GPUBuffer;
    private projectileIBO!: GPUBuffer;
    private projectileIndexCount = 0;

    private projectiles: Array<{
        start: Vec3;
        dir: Vec3;
        pos: Vec3;
        totalDist: number;
        travelTime: number;
        elapsed: number;
        amp: number;
        sigma: number;
        uv: { u: number; v: number };
    }> = [];

    private npcShips: Array<{
        pos: Vec3;
        forward: Vec3;
        modelBuffer: GPUBuffer;
        modelBindGroup: GPUBindGroup;
        waypoints: Vec3[];
        waypointIndex: number;
        bumpCur: Float32Array;
        bumpPrev: Float32Array;
        bumpDelta: Float32Array;
        warmup: number;
    }> = [];

    private followMode = false;

    private initMode: 'default' | 'terrain' | 'ship' | 'click';

    private terrainArr: Float32Array;

    // --- Ship imagined ball state ---
    private shipPos: Vec3 = vec3.fromValues(3, shaders.constants.water_base_level, 3);
    private shipRadius: number = 0.1; 
    private shipSpeed: number = 1.0;
    private shipProjectileSpeed: number = 6.0;
    private shipProjectileRadius: number = 0.05;
    private shipForward: Vec3 = vec3.fromValues(0, 0, 1);
    private shipModelScale: number = 0.02;

    private shipKeys: Record<string, boolean> = {
        ArrowUp: false,
        ArrowDown: false,
        ArrowLeft: false,
        ArrowRight: false,
    };
    private shipKeyDownHandler?: (e: KeyboardEvent) => void;
    private shipKeyUpHandler?: (e: KeyboardEvent) => void;

    private shipBumpArr: Float32Array | null = null;
    private shipBumpArrPrev: Float32Array | null = null;
    private shipBumpArrDelta: Float32Array | null = null;

    constructor(stage: Stage, initMode: 'default' | 'terrain' | 'ship' | 'click' = 'default') {
        super(stage);
        this.initMode = initMode;
        if (initMode === 'ship' || initMode === 'default') {
            this.shipKeyDownHandler = (e: KeyboardEvent) => this.handleShipKey(e.key, true);
            this.shipKeyUpHandler = (e: KeyboardEvent) => this.handleShipKey(e.key, false);
            window.addEventListener('keydown', this.shipKeyDownHandler);
            window.addEventListener('keyup', this.shipKeyUpHandler);
            window.addEventListener('keydown', this.handleFollowToggle);
        }

        const base = shaders.constants.water_base_level;
        this.fixedWaterPlaneHeight = base;

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

        this.bumpTexture = renderer.device.createTexture({
            size: [this.heightW, this.heightH],
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.STORAGE_BINDING,
        });

        this.heightAddCS = new AddOnCS(renderer.device);

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

        this.highFreqPrevTexture = renderer.device.createTexture({
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
        const defaults = new Float32Array([1/this.heightW,1/this.heightH, 500,500, 10, 0, 0, 0]);
        renderer.device.queue.writeBuffer(this.heightConsts, 0, defaults);

        // Bind group layout for { sampler, heightTex, heightConsts } used by the water VS.
        // Note: sampleType: "unfilterable-float" + sampler type "non-filtering" for R32Float.
        this.heightBindGroupLayout = renderer.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, sampler: { type: "non-filtering" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "unfilterable-float" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
                { binding: 3, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "unfilterable-float" } },
            ]
        });

        // Concrete bind group connecting the sampler, the texture view, and the UBO.
        this.heightBindGroup = renderer.device.createBindGroup({
            layout: this.heightBindGroupLayout,
            entries: [
                { binding: 0, resource: this.heightSampler },
                { binding: 1, resource: this.heightTexture.createView() },
                { binding: 2, resource: { buffer: this.heightConsts } },
                { binding: 3, resource: this.terrainTexture.createView() },
            ]
        });

        // --- Reflection Pipeline ---
        const w = renderer.canvas.width;
        const h = renderer.canvas.height;

        // Color texture where the reflection pass renders the mirrored scene.
        // Slightly higher resolution to reduce edge sampling artifacts after UV shrink.
        const reflectionScale = 2.0;
        const reflW = Math.max(1, Math.floor(w * reflectionScale));
        const reflH = Math.max(1, Math.floor(h * reflectionScale));
        this.reflectionTexture = renderer.device.createTexture({
            label: "reflection color",
            size: [reflW, reflH],
            format: renderer.canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.reflectionTextureView = this.reflectionTexture.createView();

        // Depth texture used only in the reflection pass
        this.reflectionDepthTexture = renderer.device.createTexture({
            label: "reflection depth",
            size: [reflW, reflH],
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

        // Environment map bindings (equirectangular 2D)
        this.envSampler = renderer.device.createSampler({
            label: "env sampler",
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
        });
        this.envTexture = this.createPlaceholderEnvTexture();
        this.envBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "env bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            ],
        });
        this.envBindGroup = renderer.device.createBindGroup({
            label: "env bind group",
            layout: this.envBindGroupLayout,
            entries: [
                { binding: 0, resource: this.envSampler },
                { binding: 1, resource: this.envTexture.createView() },
            ],
        });
        // Kick async HDR load; placeholder is used until finished.
        const envUrl = new URL("../../scenes/envmap/Frozen_Waterfall_Ref.hdr", import.meta.url).href;
        this.initEnvironmentMap(envUrl);

        // Skybox pipeline (uses camera + env)
        this.skyboxPipeline = renderer.device.createRenderPipeline({
            label: "skybox pipeline",
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [this.sceneUniformsBindGroupLayout, this.envBindGroupLayout],
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.skyboxVertSrc }),
                entryPoint: "vs_main",
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.skyboxFragSrc }),
                entryPoint: "fs_main",
                targets: [{ format: renderer.canvasFormat }],
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: false,
                depthCompare: "less-equal",
            },
            primitive: { topology: "triangle-list" },
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

        // --- Terrain surface pipeline (renders terrainArr as a height map mesh) ---
        this.terrainSurfaceBindGroupLayout = renderer.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, sampler: { type: "non-filtering" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "unfilterable-float" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            ],
        });

        this.terrainSurfaceBindGroup = renderer.device.createBindGroup({
            layout: this.terrainSurfaceBindGroupLayout,
            entries: [
                { binding: 0, resource: this.heightSampler },
                { binding: 1, resource: this.terrainTexture.createView() },
                { binding: 2, resource: { buffer: this.heightConsts } },
            ],
        });

        this.terrainSurfacePipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    this.terrainSurfaceBindGroupLayout,
                ],
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.terrainVertSrc }),
                entryPoint: "vs_main",
                buffers: [heightVertexLayout],
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.terrainFragSrc }),
                entryPoint: "fs_main",
                targets: [{ format: renderer.canvasFormat }],
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
            primitive: { topology: "triangle-list" },
        });

        // --- Projectile pipeline (simple colored sphere) ---
        this.createProjectileResources();

        // --- Diffuse step and simulation setup ---
        this.initRowInfoIfNeeded();

        const terrainArr = new Float32Array(this.heightW * this.heightH);
        const terrainZeroArr = new Float32Array(this.heightW * this.heightH); 
        this.heightArr = new Float32Array(this.heightW * this.heightH);
        this.lowArr = new Float32Array(this.heightW * this.heightH);
        const highArr = new Float32Array(this.heightW * this.heightH);

        const Wtex = this.heightW;
        const Htex = this.heightH;
        
        const fluxInitX = new Float32Array(this.heightW * this.heightH);
        const fluxInitY = new Float32Array(this.heightW * this.heightH);
        
        this.seedHeightFields(this.initMode, terrainArr, highArr, Wtex, Htex);

        this.terrainArr = terrainArr;

        this.updateTexture(terrainArr, this.terrainTexture);
        this.updateTexture(terrainZeroArr,  this.terrainZeroTexture);
        this.updateTexture(this.heightArr, this.heightTexture);  
        this.updateTexture(this.lowArr, this.lowFreqTexture);
        this.updateTexture(this.lowArr, this.lowFreqTexturePingpong);
        this.updateTexture(highArr, this.highFreqTexture);
        this.updateTexture(highArr, this.highFreqPrevTexture);

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
            this.heightH,
            this.heightTexture,
            this.lowFreqPrevHeightTexture,
            this.qxTexture,
            this.qyTexture,
            this.lowFreqVelocityXTexture,
            this.lowFreqVelocityYTexture,
            this.changeInLowFreqVelocityXTexture,
            this.changeInLowFreqVelocityYTexture,
            this.terrainTexture
        );

        const smoothDepth = new Float32Array(this.heightW * this.heightH);
        for (let i = 0; i < smoothDepth.length; ++i) {
            const depth = this.lowArr[i];
            smoothDepth[i] = Math.max(depth, 0.01);
        }
        
        this.airyWaveCS = new AiryWaveCS(
            renderer.device,
            this.heightW,
            this.heightH,
            this.highFreqPrevTexture,
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

        // NPCs depend on terrainArr, so init after it is ready.
        this.initNpcShips();

        this.initialized = true;
    }

    private createProjectileResources() {
        const sphere = makeSphere(16, 16);
        this.projectileIndexCount = sphere.indices.length;

        this.projectileVBO = renderer.device.createBuffer({
            size: sphere.verts.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        renderer.device.queue.writeBuffer(this.projectileVBO, 0, sphere.verts);

        this.projectileIBO = renderer.device.createBuffer({
            size: sphere.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        renderer.device.queue.writeBuffer(this.projectileIBO, 0, sphere.indices);

        this.projectileUniformBuffer = renderer.device.createBuffer({
            size: 4 * 16 + 4 * 4, // modelMat (16 floats) + color (vec3 + pad)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.projectileBindGroupLayout = renderer.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ],
        });

        this.projectileBindGroup = renderer.device.createBindGroup({
            layout: this.projectileBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.projectileUniformBuffer } },
            ],
        });

        this.projectilePipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    this.projectileBindGroupLayout,
                ],
            }),
            vertex: {
                module: renderer.device.createShaderModule({ code: shaders.projectileVertSrc }),
                entryPoint: "vs_main",
                buffers: [renderer.vertexBufferLayout],
            },
            fragment: {
                module: renderer.device.createShaderModule({ code: shaders.projectileFragSrc }),
                entryPoint: "fs_main",
                targets: [{ format: renderer.canvasFormat }],
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
            primitive: { topology: "triangle-list" },
        });
    }

    private seedHeightFields(
        mode: 'default' | 'terrain' | 'ship' | 'click',
        terrainArr: Float32Array,
        highArr: Float32Array,
        Wtex: number,
        Htex: number
    ) {
        const centerX = Wtex / 2;
        const centerY = Htex / 2;
        const bumpRadius = Math.min(Wtex, Htex) * 0.1;
        const bumpHeight = 3.5;
        const highAmp = 0.5;
        const freq = 16.0;
        // Keep terrain for default/click; only ship scene stays flat.
        const flattenTerrain = mode === 'ship';

        for (let y = 0; y < Htex; y++) {
            for (let x = 0; x < Wtex; x++) {
                const idx = y * Wtex + x;
                

                const u = x / Wtex;
                const v = y / Htex;

                let baseWater = shaders.constants.water_base_level;
                let high = 0.0;

                switch (mode) {
                    case 'terrain': {
                        baseWater += 0.0;
                        high = 0.0;
                        break;
                    }
                    case 'ship': {
                        baseWater += 0.0;
                        high = 0.0;
                        break;
                    }
                    case 'click': {
                        // flat water for click interactions
                        baseWater += 0.0;
                        high = 0.0;
                        break;
                    }
                    default: {
                        // flat like ship
                        baseWater += 0.0;
                        high = 0.0;
                        break;
                    }
                }

                this.lowArr[idx] = baseWater;
                highArr[idx] = high;
                this.heightArr[idx] = baseWater;

                // For ship/click scenes, terrain stays completely flat (all zeros).
                if (flattenTerrain) {
                    terrainArr[idx] = 0.0;
                    continue;
                }

                const baseTerrain = baseWater - 0.08;
                const baseWaterHeight = baseWater;

                // Beach gradients along left and top edges (smooth ramp up).
                const beachWidth = Math.min(Wtex, Htex) * 0.25;
                const beachHeight = 0.25;
                const leftFactor = Math.max(0, 1 - x / beachWidth);
                const topFactor = Math.max(0, 1 - y / beachWidth);
                const beachFactor = Math.max(leftFactor, topFactor);

                let terrainH = baseTerrain + beachHeight * beachFactor;
                let waterH   = baseWaterHeight;

                // Gentle undulation across the beach so it isn't perfectly flat.
                if (beachFactor > 0.0) {
                    const waviness =
                        0.05 * beachFactor *
                        (Math.sin(u * 2.0 * Math.PI * 4.0) + Math.cos(v * 2.0 * Math.PI * 3.0)) * 0.5;
                    terrainH += waviness;
                }

                // Islands: soft bumps in the middle.
                const islands = [
                    { cx: 0.42, cy: 0.45, amp: 0.45, sigma: 0.05 },
                    { cx: 0.58, cy: 0.55, amp: 0.35, sigma: 0.04 },
                    { cx: 0.50, cy: 0.38, amp: 0.30, sigma: 0.035 },
                ];
                for (const island of islands) {
                    const dx = u - island.cx;
                    const dy = v - island.cy;
                    const dist2 = dx * dx + dy * dy;
                    const bump = island.amp * Math.exp(-dist2 / (2 * island.sigma * island.sigma));
                    terrainH += bump;
                }

                terrainArr[idx] = terrainH;
                this.heightArr[idx] = waterH;
            }
        }
    }

    protected override draw() {
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

        // Draw skybox into reflection target
        reflectionPass.setPipeline(this.skyboxPipeline);
        reflectionPass.setBindGroup(0, this.reflectionSceneUniformsBindGroup);
        reflectionPass.setBindGroup(1, this.envBindGroup);
        reflectionPass.draw(3);

        if (this.initMode !== 'click') {
            // Draw glTF scene into reflection target
            reflectionPass.setPipeline(this.pipeline);
            reflectionPass.setBindGroup(shaders.constants.bindGroup_scene, this.reflectionSceneUniformsBindGroup);
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
            // Draw NPC ships in reflection
            this.drawNpcShips(reflectionPass, this.passFlagsBindGroupReflection, this.reflectionSceneUniformsBindGroup);
        }

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
        // Draw skybox first (no depth write)
        renderPass.setPipeline(this.skyboxPipeline);
        renderPass.setBindGroup(0, this.sceneUniformsBindGroup);
        renderPass.setBindGroup(1, this.envBindGroup);
        renderPass.draw(3);

        if (this.initMode !== 'click') {
            // Draw glTF scene
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
            // Draw NPC ships in main pass
            this.drawNpcShips(renderPass, this.passFlagsBindGroupMain, this.sceneUniformsBindGroup);
        }

        // Draw projectiles (simple spheres)
        if (this.projectiles.length > 0) {
            renderPass.setPipeline(this.projectilePipeline);
            renderPass.setBindGroup(0, this.sceneUniformsBindGroup);
            renderPass.setBindGroup(1, this.projectileBindGroup);
            renderPass.setVertexBuffer(0, this.projectileVBO);
            renderPass.setIndexBuffer(this.projectileIBO, "uint32");

            const modelMat = new Float32Array(16);

            for (const p of this.projectiles) {
                const scl   = mat4.scaling([
                    this.shipProjectileRadius,
                    this.shipProjectileRadius,
                    this.shipProjectileRadius
                ]);
                const trans = mat4.translation(p.pos);

                mat4.mul(trans, scl, modelMat);

                renderer.device.queue.writeBuffer(
                    this.projectileUniformBuffer,
                    0,
                    modelMat
                );

                const color = new Float32Array([1.0, 0.8, 0.2, 0.0]);
                renderer.device.queue.writeBuffer(
                    this.projectileUniformBuffer,
                    64,
                    color
                );

                renderPass.drawIndexed(this.projectileIndexCount);
            }
        }

        // Draw terrain height map as a displaced mesh (terrainArr).
        renderPass.setPipeline(this.terrainSurfacePipeline);
        renderPass.setBindGroup(0, this.sceneUniformsBindGroup);
        renderPass.setBindGroup(1, this.terrainSurfaceBindGroup);
        renderPass.setVertexBuffer(0, this.heightVBO);
        renderPass.setIndexBuffer(this.heightIBO, "uint32");
        renderPass.drawIndexed(this.heightIndexCount);

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

    override stop(): void {
        if (this.shipKeyDownHandler) {
            window.removeEventListener('keydown', this.shipKeyDownHandler);
            this.shipKeyDownHandler = undefined;
        }
        if (this.shipKeyUpHandler) {
            window.removeEventListener('keyup', this.shipKeyUpHandler);
            this.shipKeyUpHandler = undefined;
        }
        window.removeEventListener('keydown', this.handleFollowToggle);
        super.stop();
    }

    // Update world scale (sx, sz), height amplitude, and base level (Y offset).
    // Offsets match the HeightConsts layout described above.
    setHeightParams(sx: number, sz: number, heightScale: number) {
        renderer.device.queue.writeBuffer(this.heightConsts, 8,  new Float32Array([sx, sz]));
        renderer.device.queue.writeBuffer(this.heightConsts, 16, new Float32Array([heightScale]));
        renderer.device.queue.writeBuffer(this.heightConsts, 20, new Float32Array([0.0]));
        this.waterScaleX = sx;
        this.waterScaleZ = sz;
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
        if (!this.initialized || !this.simulator) return;

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

        encoder.copyTextureToTexture(
            { texture: this.highFreqTexture },
            { texture: this.highFreqPrevTexture },
            { width: this.heightW, height: this.heightH, depthOrArrayLayers: 1 }
        );

        renderer.device.queue.submit([encoder.finish()]);

        if (this.initMode === 'ship' || this.initMode === 'default') {
            this.updateShipInteraction(dt);
            this.updateProjectiles(dt);
            this.updateNpcShips(dt);
            if (this.followMode) {
                this.updateFollowCamera();
            }
        }

        this.simulator.simulate(1.0/240.0);
    }

    private updateReflectionUniforms() {
        const mainCam = this.camera; 
        const eye = mainCam.cameraPos;     
        const front = mainCam.cameraFront; 
        const up = mainCam.cameraUp;        

        const h = this.fixedWaterPlaneHeight; 

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
        // Use a slightly wider FOV for reflection to reduce edge clipping/stretch
        const fovScale = 3.0;
        const proj = mat4.perspective(
            renderer.fovYDegrees * fovScale * Math.PI / 180.0,
            renderer.aspectRatio,
            Camera.nearPlane,
            Camera.farPlane
        );
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
        this.reflectionCameraUniforms.farPlane = Camera.farPlane * 10.0;

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

    // --- Public helpers for click scene ---
    // Resets water to flat base level with no bumps/high frequency content.
    public resetWaterFlat() {
        const base = shaders.constants.water_base_level;
        this.heightArr.fill(base);
        this.lowArr.fill(base);
        const zeroHigh = new Float32Array(this.heightW * this.heightH);

        this.updateTexture(this.heightArr, this.heightTexture);
        this.updateTexture(this.heightArr, this.heightPrevTexture);
        this.updateTexture(this.lowArr, this.lowFreqTexture);
        this.updateTexture(this.lowArr, this.lowFreqTexturePingpong);
        this.updateTexture(zeroHigh, this.highFreqTexture);
        this.updateTexture(zeroHigh, this.highFreqPrevTexture);
    }

    // Adds a Gaussian bump at normalized UV (0..1) into the height/low fields.
    public addClickBump(u: number, v: number, amplitude = 3.0, sigma = 8.0) {
        const cx = u * (this.heightW - 1);
        const cy = v * (this.heightH - 1);
        const radius = Math.ceil(3 * sigma);

        const bumpArr = new Float32Array(this.heightW * this.heightH);

        for (let y = Math.max(0, Math.floor(cy - radius));
                y <= Math.min(this.heightH - 1, Math.ceil(cy + radius)); y++) {
            for (let x = Math.max(0, Math.floor(cx - radius));
                    x <= Math.min(this.heightW - 1, Math.ceil(cx + radius)); x++) {

                const dx = x - cx;
                const dy = y - cy;
                const dist2 = dx * dx + dy * dy;

                const bump = amplitude * Math.exp(-dist2 / (2 * sigma * sigma));
                const idx = y * this.heightW + x;

                bumpArr[idx] += bump;
            }
        }

        this.updateTexture(bumpArr, this.bumpTexture);
        
        this.heightAddCS.run(this.heightTexture, this.bumpTexture, this.heightW, this.heightH);
        this.heightAddCS.run(this.lowFreqTexture, this.bumpTexture, this.heightW, this.heightH);
        this.heightAddCS.run(this.heightPrevTexture, this.bumpTexture, this.heightW, this.heightH);
    }

    // Sinusoidal wave injection for click scene (flat plane wave with Gaussian envelope).
    private addClickWave(u: number, v: number, amplitude = 1.2, wavelength = 6.0, envelopeRadius = 1.0) {
        const W = this.heightW;
        const H = this.heightH;
        const waveArr = new Float32Array(W * H);

        // Convert center from uv to world space on the water plane.
        const centerX = (u - 0.5) * 2 * this.waterScaleX;
        const centerZ = (v - 0.5) * 2 * this.waterScaleZ;

        const k = (2 * Math.PI) / Math.max(1e-4, wavelength);
        const spreadZ2 = Math.max(1e-4, envelopeRadius * envelopeRadius);
        // Make the crest span almost the full width in X so it feels like a wall of water.
        const spreadX2 = Math.max(1e-4, Math.pow(this.waterScaleX * 0.95, 2));

        for (let y = 0; y < H; y++) {
            const vy = y / (H - 1);
            const worldZ = (vy - 0.5) * 2 * this.waterScaleZ;
            for (let x = 0; x < W; x++) {
                const vx = x / (W - 1);
                const worldX = (vx - 0.5) * 2 * this.waterScaleX;

                const dx = worldX - centerX;
                const dz = worldZ - centerZ;

                // Wide envelope in X, tight in Z to form a horizontal wall moving along +Z.
                const envelopeX = Math.exp(-(dx * dx) / (2 * spreadX2));
                const envelopeZ = Math.exp(-(dz * dz) / (2 * spreadZ2));
                const envelope = envelopeX * envelopeZ;

                // Plane wave traveling along +Z with localized depth envelope.
                // Only positive displacement (crest), no trough.
                const crest = Math.max(0, Math.sin(k * dz));
                const wave = amplitude * crest * envelope;

                waveArr[y * W + x] = wave;
            }
        }

        this.updateTexture(waveArr, this.bumpTexture);
        
        this.heightAddCS.run(this.heightTexture, this.bumpTexture, this.heightW, this.heightH);
        this.heightAddCS.run(this.lowFreqTexture, this.bumpTexture, this.heightW, this.heightH);
        this.heightAddCS.run(this.heightPrevTexture, this.bumpTexture, this.heightW, this.heightH);
    }

    // Adds bump based on screen coordinates (clientX/Y) -> water plane intersection
    public addClickBumpFromScreen(clientX: number, clientY: number, rect: DOMRect, amp = 3.0, sigma = 8.0) {
        const target = this.screenToWaterUVAndWorld(clientX, clientY, rect);
        if (!target) return;
        const { u, v } = target;

        // if (this.initMode === 'click') {
        //     // Click scene: disable Gaussian bump and inject a localized sinusoidal wave instead.
        //     this.addClickWave(u, v, Math.max(amp, 1.2), 6.0, Math.max(0.6, sigma * 0.05));
        //     return;
        // }

        this.addClickBump(u, v, amp, sigma);
    }

    private mulMat4Vec4(m: Float32Array, v: [number, number, number, number]): [number, number, number, number] {
        const x = v[0], y = v[1], z = v[2], w = v[3];
        return [
            m[0] * x + m[4] * y + m[8]  * z + m[12] * w,
            m[1] * x + m[5] * y + m[9]  * z + m[13] * w,
            m[2] * x + m[6] * y + m[10] * z + m[14] * w,
            m[3] * x + m[7] * y + m[11] * z + m[15] * w,
        ];
    }

    private createPlaceholderEnvTexture(): GPUTexture {
        const tex = renderer.device.createTexture({
            label: "env placeholder",
            size: [1, 1, 1],
            format: "rgba16float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        const data = new Uint16Array([this.toFloat16(0.1), this.toFloat16(0.2), this.toFloat16(0.3), this.toFloat16(1.0)]);
        renderer.device.queue.writeTexture(
            { texture: tex },
            data,
            { bytesPerRow: 8 },
            [1, 1, 1]
        );
        return tex;
    }

    private async initEnvironmentMap(url: string) {
        try {
            const hdr = await this.loadHDR(url);
            const tex = renderer.device.createTexture({
                label: "env hdr texture",
                size: [hdr.width, hdr.height, 1],
                format: "rgba16float",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });

            const data16 = this.toFloat16Array(hdr.data);
            renderer.device.queue.writeTexture(
                { texture: tex },
                data16 as BufferSource,
                { bytesPerRow: hdr.width * 4 * 2 },
                [hdr.width, hdr.height, 1]
            );

            // Swap in the loaded texture.
            this.envTexture = tex;
            this.envBindGroup = renderer.device.createBindGroup({
                label: "env bind group (loaded)",
                layout: this.envBindGroupLayout,
                entries: [
                    { binding: 0, resource: this.envSampler },
                    { binding: 1, resource: this.envTexture.createView() },
                ],
            });
        } catch (err) {
            console.warn("Failed to load HDR env map:", err);
        }
    }

    private async loadHDR(url: string): Promise<{ width: number; height: number; data: Float32Array }> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch HDR: ${response.statusText}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        return this.parseHDR(arrayBuffer);
    }

    private toFloat16(v: number): number {
        // Half float conversion (round-to-nearest-even)
        const floatView = new Float32Array(1);
        const intView = new Uint32Array(floatView.buffer);
        floatView[0] = v;
        const x = intView[0];
        const sign = (x >> 16) & 0x8000;
        const mantissa = x & 0x7fffff;
        const exp = (x >> 23) & 0xff;
        if (exp === 0xff) {
            // NaN or Inf
            return sign | 0x7c00 | (mantissa ? 1 : 0);
        }
        if (exp === 0) {
            // subnormal in f32
            if (mantissa === 0) {
                return sign;
            }
            let e = -1;
            let m = mantissa;
            while ((m & 0x800000) === 0) {
                m <<= 1;
                e--;
            }
            m &= 0x7fffff;
            const exp16 = e + 1 + 127 - 15;
            if (exp16 <= 0) {
                return sign;
            }
            return sign | (exp16 << 10) | (m >> 13);
        }
        const exp16 = exp - 127 + 15;
        if (exp16 >= 0x1f) {
            return sign | 0x7c00;
        }
        return sign | (exp16 << 10) | (mantissa >> 13);
    }

    private toFloat16Array(src: Float32Array): Uint16Array {
        const out = new Uint16Array(src.length);
        for (let i = 0; i < src.length; i++) {
            out[i] = this.toFloat16(src[i]);
        }
        return out;
    }

    // Minimal Radiance .hdr (RGBE) loader, adapted from three.js utilities.
    private parseHDR(buffer: ArrayBuffer): { width: number; height: number; data: Float32Array } {
        const uint8 = new Uint8Array(buffer);
        let pos = 0;
        const NEWLINE = 10;
        const decoder = new TextDecoder();

        const readLine = () => {
            let start = pos;
            while (pos < uint8.length && uint8[pos] !== NEWLINE) pos++;
            const line = decoder.decode(uint8.subarray(start, pos));
            pos++; // skip newline
            return line;
        };

        let width = 0;
        let height = 0;
        // Header
        while (pos < uint8.length) {
            const line = readLine().trim();
            if (line.length === 0) {
                // empty line; resolution typically follows
                break;
            }
        }
        // Read until we find resolution line "-Y H +X W"
        while (pos < uint8.length && (width === 0 || height === 0)) {
            const line = readLine().trim();
            if (line.length === 0) {
                continue;
            }
            if (line.startsWith("-Y") || line.startsWith("+Y")) {
                const parts = line.split(/\s+/);
                // e.g. "-Y 1024 +X 2048"
                const hIndex = parts.findIndex(p => p === "-Y" || p === "+Y");
                const wIndex = parts.findIndex(p => p === "+X" || p === "-X");
                if (hIndex >= 0 && hIndex + 1 < parts.length) {
                    height = parseInt(parts[hIndex + 1], 10);
                }
                if (wIndex >= 0 && wIndex + 1 < parts.length) {
                    width = parseInt(parts[wIndex + 1], 10);
                }
            }
        }

        if (width === 0 || height === 0) {
            throw new Error("Invalid HDR header (width/height missing)");
        }

        const out = new Float32Array(width * height * 4);
        const scanline = new Uint8Array(width * 4);

        for (let y = 0; y < height; y++) {
            // Expecting RLE header per scanline
            if (uint8[pos++] !== 2 || uint8[pos++] !== 2) {
                throw new Error("Unsupported HDR scanline format");
            }
            const scanWidth = (uint8[pos++] << 8) | uint8[pos++];
            if (scanWidth !== width) {
                throw new Error("HDR scanline width mismatch");
            }
            // Read RLE for each of 4 components
            for (let c = 0; c < 4; c++) {
                let i = 0;
                while (i < width) {
                    const count = uint8[pos++];
                    const value = uint8[pos++];
                    if (count > 128) {
                        const reps = count - 128;
                        for (let r = 0; r < reps; r++) {
                            scanline[i * 4 + c] = value;
                            i++;
                        }
                    } else {
                        let reps = count;
                        scanline[i * 4 + c] = value;
                        i++;
                        reps--;
                        for (let r = 0; r < reps; r++) {
                            scanline[i * 4 + c] = uint8[pos++];
                            i++;
                        }
                    }
                }
            }
            // Convert RGBE to float
            for (let x = 0; x < width; x++) {
                const r = scanline[x * 4 + 0];
                const g = scanline[x * 4 + 1];
                const b = scanline[x * 4 + 2];
                const e = scanline[x * 4 + 3];
                const outIndex = (y * width + x) * 4;
                if (e > 0) {
                    const scale = Math.pow(2.0, e - 128.0) / 255.0;
                    out[outIndex + 0] = r * scale;
                    out[outIndex + 1] = g * scale;
                    out[outIndex + 2] = b * scale;
                    out[outIndex + 3] = 1.0;
                } else {
                    out[outIndex + 0] = 0;
                    out[outIndex + 1] = 0;
                    out[outIndex + 2] = 0;
                    out[outIndex + 3] = 1.0;
                }
            }
        }

        return { width, height, data: out };
    }

    public handleShipKey(key: string, pressed: boolean) {
        if (this.initMode !== 'ship' && this.initMode !== 'default') return;
        if (key in this.shipKeys) {
            this.shipKeys[key] = pressed;
        }
    }

    public getShipPosition(): Vec3 {
        return vec3.clone(this.shipPos);
    }

    private handleFollowToggle = (e: KeyboardEvent) => {
        if (e.key.toLowerCase() === 'z' && (this.initMode === 'default' || this.initMode === 'ship')) {
            this.followMode = !this.followMode;
        }
    };

    // --- Ship interaction: imagined ball pressing into height field ---
    private updateShipInteraction(dt: number) {
        if (this.initMode !== 'ship' && this.initMode !== 'default') return;

        const camFront = this.camera.cameraFront;
        const forwardCam: Vec3 = vec3.fromValues(camFront[0], 0, camFront[2]);

        if (vec3.length(forwardCam) < 1e-4) {
            return;
        }
        vec3.normalize(forwardCam, forwardCam);

        const rightCam: Vec3 = vec3.fromValues(forwardCam[2], 0, -forwardCam[0]);

        let moveX = 0;
        let moveZ = 0;
        let turn = 0;

        if (this.followMode) {
            // In follow mode: left/right rotate ship in place; up/down move along ship forward.
            if (this.shipKeys['ArrowLeft']) turn += 1;
            if (this.shipKeys['ArrowRight']) turn -= 1;
            if (this.shipKeys['ArrowUp']) {
                moveX += this.shipForward[0];
                moveZ += this.shipForward[2];
            }
            if (this.shipKeys['ArrowDown']) {
                moveX -= this.shipForward[0];
                moveZ -= this.shipForward[2];
            }
        } else {
            if (this.shipKeys['ArrowUp']) {
                moveX += forwardCam[0];
                moveZ += forwardCam[2];
            }
            if (this.shipKeys['ArrowDown']) {
                moveX -= forwardCam[0];
                moveZ -= forwardCam[2];
            }
            if (this.shipKeys['ArrowLeft']) {
                moveX += rightCam[0];
                moveZ += rightCam[2];
            }
            if (this.shipKeys['ArrowRight']) {
                moveX -= rightCam[0];
                moveZ -= rightCam[2];
            }
        }

        const move: Vec3 = vec3.fromValues(moveX, 0, moveZ);
        const len = vec3.length(move);

        if (len > 1e-4) {
            vec3.scale(move, this.shipSpeed * dt / len, move);
            const nextX = this.shipPos[0] + move[0];
            const nextZ = this.shipPos[2] + move[2];
            const clamped = this.clampToTerrain(nextX, nextZ, this.shipPos[0], this.shipPos[2]);
            this.shipPos[0] = clamped[0];
            this.shipPos[2] = clamped[1];

            const moveDir: Vec3 = vec3.fromValues(move[0], 0, move[2]);
            vec3.normalize(moveDir, moveDir);

            if (!this.followMode) {
                vec3.lerp(this.shipForward, moveDir, 0.01, this.shipForward);
                vec3.normalize(this.shipForward, this.shipForward);
            }

            this.applyShipBump(dt);
        }

        // In follow mode, left/right turns rotate ship in place.
        if (this.followMode && (this.shipKeys['ArrowLeft'] || this.shipKeys['ArrowRight'])) {
            const turnDir = (this.shipKeys['ArrowLeft'] ? 1 : 0) + (this.shipKeys['ArrowRight'] ? -1 : 0);
            if (turnDir !== 0) {
                const yawSpeed = 1.5; // rad/s
                const yaw = turnDir * yawSpeed * dt;
                const rot = mat4.rotationY(yaw);
                const f = vec3.transformMat4(this.shipForward, rot);
                vec3.normalize(f, f);
                this.shipForward = f;
            }
        }
        
        this.shipPos[0] = Math.min(this.waterScaleX,  Math.max(-this.waterScaleX,  this.shipPos[0]));
        this.shipPos[2] = Math.min(this.waterScaleZ,  Math.max(-this.waterScaleZ,  this.shipPos[2]));

        // Keep ship at a fixed water plane height (shared with reflection plane).
        this.shipPos[1] = this.fixedWaterPlaneHeight;

        // Update model transform to follow imagined ship position/orientation
        const angle = Math.atan2(this.shipForward[0], this.shipForward[2]);
        const rot   = mat4.rotationY(angle);
        const scl   = mat4.scaling([this.shipModelScale, this.shipModelScale, this.shipModelScale]);
        const trans = mat4.translation([this.shipPos[0], this.shipPos[1], this.shipPos[2]]);

        const modelMat = new Float32Array(16);     
        mat4.mul(trans, mat4.mul(rot, scl), modelMat);          

        // Apply to all mesh nodes (simple assumption: single ship model loaded)
        this.scene.iterate(
            (node) => {
                if (node.mesh && node.modelMatUniformBuffer) {
                    renderer.device.queue.writeBuffer(
                        node.modelMatUniformBuffer,
                        0,
                        modelMat                          
                    );
                }
            },
            () => {},
            () => {},
        );
    }

    private applyShipBump(dt: number) {
        if (this.initMode !== 'ship' && this.initMode !== 'default') return;

        const W = this.heightW;
        const H = this.heightH;
        const N = W * H;

        if (!this.shipBumpArr) {
            this.shipBumpArr      = new Float32Array(N);
            this.shipBumpArrPrev  = new Float32Array(N);
            this.shipBumpArrDelta = new Float32Array(N);
        }

        this.applyShipBumpAt(this.shipPos, this.shipForward, dt, this.shipBumpArr!, this.shipBumpArrPrev!, this.shipBumpArrDelta!, 0.5);
    }

    private updateProjectiles(dt: number) {
        if (this.projectiles.length === 0) return;

        const toRemove: number[] = [];
        for (let i = 0; i < this.projectiles.length; i++) {
            const p = this.projectiles[i];
            p.elapsed += dt;
            const traveled = Math.min(p.elapsed * this.shipProjectileSpeed, p.totalDist);
            const step = vec3.scale(p.dir, traveled);
            p.pos = vec3.add(p.start, step);

            if (p.elapsed >= p.travelTime) {
                this.addClickBump(p.uv.u, p.uv.v, p.amp, p.sigma);
                toRemove.push(i);
            }
        }
        for (let j = toRemove.length - 1; j >= 0; j--) {
            this.projectiles.splice(toRemove[j], 1);
        }
    }

    private initNpcShips() {
        const count = 0;
        const radiusMin = this.waterScaleX * 0.2;
        const radiusMax = this.waterScaleX * 0.8;
        const N = this.heightW * this.heightH;
        for (let i = 0; i < count; i++) {
            this.npcShips.push(this.createNpcShip(i, count, radiusMin, radiusMax, N));
        }
    }

    private createNpcShip(idx: number, count: number, radiusMin: number, radiusMax: number, N: number) {
        const theta = (idx / count) * Math.PI * 2;
        const r = radiusMin + Math.random() * (radiusMax - radiusMin);
        const pos = vec3.fromValues(Math.cos(theta) * r, this.fixedWaterPlaneHeight, Math.sin(theta) * r);
        const forward = vec3.normalize(vec3.fromValues(-Math.sin(theta), 0, Math.cos(theta)));
        const waypoints: Vec3[] = [];
        const waypointCount = 5 + Math.floor(Math.random() * 3);
        for (let w = 0; w < waypointCount; w++) {
            const ang = Math.random() * Math.PI * 2;
            const rad = radiusMin + Math.random() * (radiusMax - radiusMin);
            const wx = Math.cos(ang) * rad;
            const wz = Math.sin(ang) * rad;
            const clamped = this.clampToTerrain(wx, wz, pos[0], pos[2]);
            waypoints.push(vec3.fromValues(clamped[0], this.fixedWaterPlaneHeight, clamped[1]));
        }
        waypoints.push(vec3.clone(pos)); // loop back near start

        const modelBuffer = renderer.device.createBuffer({
            label: `npc ship model buffer ${idx}`,
            size: 16 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const modelBindGroup = renderer.device.createBindGroup({
            layout: renderer.modelBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: modelBuffer } }],
        });

        return {
            pos,
            forward,
            modelBuffer,
            modelBindGroup,
            waypoints,
            waypointIndex: 0,
            bumpCur: new Float32Array(N),
            bumpPrev: new Float32Array(N),
            bumpDelta: new Float32Array(N),
            warmup: 2.0, // seconds to skip applying bumps/motion
        };
    }

    private updateNpcShips(dt: number) {
        if (this.npcShips.length === 0) return;
        const speed = this.shipSpeed * 0.6;
        for (const npc of this.npcShips) {
            if (npc.warmup > 0) {
                npc.warmup = Math.max(0, npc.warmup - dt);
                continue;
            }
            const target = npc.waypoints[npc.waypointIndex];
            const toTarget = vec3.sub(target, npc.pos);
            const dist = vec3.length(toTarget);
            if (dist < 0.05) {
                npc.waypointIndex = (npc.waypointIndex + 1) % npc.waypoints.length;
                continue;
            }
            const dir = vec3.scale(toTarget, 1 / dist);
            const step = Math.min(dist, speed * dt);
            if (step <= 1e-4) continue;

            vec3.scale(dir, step, dir);
            const nextX = npc.pos[0] + dir[0];
            const nextZ = npc.pos[2] + dir[2];
            if (!this.isTerrainNavigable(nextX, nextZ)) {
                // If blocked, reset waypoints from current position.
                npc.waypoints = this.createNpcShip(0, 1, this.waterScaleX * 0.2, this.waterScaleX * 0.8, this.heightW * this.heightH).waypoints;
                npc.waypointIndex = 0;
                continue;
            }
            npc.pos[0] = nextX;
            npc.pos[2] = nextZ;
            npc.pos[1] = this.fixedWaterPlaneHeight;
            npc.forward = vec3.normalize(vec3.fromValues(dir[0], 0, dir[2]));

            const angle = Math.atan2(npc.forward[0], npc.forward[2]);
            const rot = mat4.rotationY(angle);
            const scl = mat4.scaling([this.shipModelScale, this.shipModelScale, this.shipModelScale]);
            const trans = mat4.translation(npc.pos);
            const modelMat = new Float32Array(16);
            mat4.mul(trans, mat4.mul(rot, scl), modelMat);
            renderer.device.queue.writeBuffer(npc.modelBuffer, 0, modelMat);

            // Apply water bump only when movement is non-trivial
            if (step > 1e-3) {
                this.applyShipBumpAt(npc.pos, npc.forward, dt, npc.bumpCur, npc.bumpPrev, npc.bumpDelta, 0.3);
            }
        }
    }

    private drawNpcShips(pass: GPURenderPassEncoder, passFlags: GPUBindGroup, sceneBG: GPUBindGroup) {
        if (this.npcShips.length === 0) return;
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(shaders.constants.bindGroup_scene, sceneBG);
        pass.setBindGroup(3, passFlags);
        this.scene.iterate(node => {
            // override per npc bind group below
        }, material => {
            pass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            pass.setVertexBuffer(0, primitive.vertexBuffer);
            pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            for (const npc of this.npcShips) {
                pass.setBindGroup(shaders.constants.bindGroup_model, npc.modelBindGroup);
                pass.drawIndexed(primitive.numIndices);
            }
        });
    }

    // Fire a projectile from ship position toward a screen click; apply bump on impact with water plane.
    public fireClickProjectileToScreen(clientX: number, clientY: number, rect: DOMRect, amp = 3.0, sigma = 8.0) {
        if (this.initMode !== 'default' && this.initMode !== 'ship') return;

        const target = this.screenToWaterUVAndWorld(clientX, clientY, rect);
        if (!target) return;
        const { u, v, world } = target;

        const shipPos = this.getShipPosition();
        const start = vec3.fromValues(shipPos[0], this.fixedWaterPlaneHeight + 0.05, shipPos[2]);
        const dir = vec3.sub(world, start);
        const dist = vec3.length(dir);
        if (dist < 1e-4) {
            this.addClickBump(u, v, amp, sigma);
            return;
        }
        const travelTime = dist / this.shipProjectileSpeed; // seconds
        vec3.scale(dir, 1 / dist, dir);

        this.projectiles.push({
            start: vec3.clone(start),
            dir: vec3.clone(dir),
            pos: vec3.clone(start),
            totalDist: dist,
            travelTime,
            elapsed: 0,
            amp,
            sigma,
            uv: { u, v },
        });
    }

    // Convert screen click to water-plane uv + world position on fixed plane.
    private screenToWaterUVAndWorld(clientX: number, clientY: number, rect: DOMRect): { u: number; v: number; world: Vec3 } | null {
        const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
        const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;

        const invViewProj = this.camera.invViewProjMat;

        const clipNear: [number, number, number, number] = [ndcX, ndcY, 0, 1];
        const clipFar:  [number, number, number, number] = [ndcX, ndcY, 1, 1];

        const worldNear4 = this.mulMat4Vec4(invViewProj, clipNear);
        const worldFar4  = this.mulMat4Vec4(invViewProj, clipFar);

        const worldNear: Vec3 = vec3.fromValues(
            worldNear4[0] / worldNear4[3],
            worldNear4[1] / worldNear4[3],
            worldNear4[2] / worldNear4[3],
        );
        const worldFar: Vec3 = vec3.fromValues(
            worldFar4[0] / worldFar4[3],
            worldFar4[1] / worldFar4[3],
            worldFar4[2] / worldFar4[3],
        );

        const worldDir = vec3.normalize(vec3.sub(worldFar, worldNear));
        const rayOrigin: Vec3 = this.camera.cameraPos;

        const denom = worldDir[1];
        if (Math.abs(denom) < 1e-4) return null;

        const t = (this.fixedWaterPlaneHeight - rayOrigin[1]) / denom;
        if (t <= 0) return null;

        const hit = vec3.add(rayOrigin, vec3.scale(worldDir, t));

        const u = hit[0] / (2 * this.waterScaleX) + 0.5;
        const v = hit[2] / (2 * this.waterScaleZ) + 0.5;
        if (u < 0 || u > 1 || v < 0 || v > 1) return null;
        return { u, v, world: vec3.fromValues(hit[0], this.fixedWaterPlaneHeight, hit[2]) };
    }

    private clampToTerrain(x: number, z: number, fallbackX: number, fallbackZ: number): [number, number] {
        if (!this.isTerrainNavigable(x, z)) {
            return [fallbackX, fallbackZ];
        }
        return [x, z];
    }

    private isTerrainNavigable(x: number, z: number): boolean {
        const u = (x / (2 * this.waterScaleX)) + 0.5;
        const v = (z / (2 * this.waterScaleZ)) + 0.5;
        if (u < 0 || u > 1 || v < 0 || v > 1) return false;
        const texX = Math.min(this.heightW - 1, Math.max(0, Math.floor(u * (this.heightW - 1))));
        const texY = Math.min(this.heightH - 1, Math.max(0, Math.floor(v * (this.heightH - 1))));
        const idx = texY * this.heightW + texX;
        const terrainH = this.terrainArr[idx];
        return terrainH <= this.navTerrainMax;
    }

    private updateFollowCamera() {
        const shipPos = this.getShipPosition();
        const forward = vec3.normalize(vec3.clone(this.shipForward));
        const backDist = 1.8;
        const height = 1.3;
        const camPos = vec3.fromValues(
            shipPos[0] - forward[0] * backDist,
            shipPos[1] + height,
            shipPos[2] - forward[2] * backDist
        );
        const lookPos = shipPos;

        this.camera.cameraPos = camPos;
        this.camera.cameraFront = vec3.normalize(vec3.sub(lookPos, camPos));
        this.camera.cameraUp = vec3.fromValues(0, 1, 0);
        this.camera.viewMat = mat4.lookAt(camPos, lookPos, this.camera.cameraUp);
        this.camera.viewProjMat = mat4.mul(this.camera.projMat, this.camera.viewMat);
        this.camera.invViewProjMat = mat4.inverse(this.camera.viewProjMat);
        this.camera.uniforms.viewProjMat = this.camera.viewProjMat;
        this.camera.uniforms.viewMat = this.camera.viewMat;
        this.camera.uniforms.invViewMat = mat4.inverse(this.camera.viewMat);
        renderer.device.queue.writeBuffer(this.camera.uniformsBuffer, 0, this.camera.uniforms.buffer);
    }

    private applyShipBumpAt(pos: Vec3, forward: Vec3, dt: number, cur: Float32Array, prev: Float32Array, delta: Float32Array, strengthScale = 1.0) {
        const W = this.heightW;
        const H = this.heightH;
        const N = W * H;

        prev.set(cur);
        cur.fill(0);

        const sx = this.waterScaleX;
        const sz = this.waterScaleZ;

        const cx = pos[0];
        const cz = pos[2];

        const forwardXZ: Vec3 = vec3.fromValues(forward[0], 0, forward[2]);
        if (vec3.length(forwardXZ) < 1e-4) {
            return;
        }
        vec3.normalize(forwardXZ, forwardXZ);

        const right: Vec3 = vec3.fromValues(forwardXZ[2], 0, -forwardXZ[0]);

        const L = this.shipRadius * 3.0;  
        const baseHalfWidth = this.shipRadius * 1.0; 

        const strength = 20.0 * strengthScale; 

        for (let j = 0; j < H; j++) {
            const v = j / (H - 1);
            const worldZ = (v - 0.5) * 2 * sz;

            for (let i = 0; i < W; i++) {
                const u = i / (W - 1);
                const worldX = (u - 0.5) * 2 * sx;

                const dx = worldX - cx;
                const dz = worldZ - cz;

                const s = dx * forwardXZ[0] + dz * forwardXZ[2];
                const t = dx * right[0]     + dz * right[2];

                if (s < 0 || s > L) continue;

                const halfWidthAtS = baseHalfWidth * (1.0 - s / L);
                if (halfWidthAtS <= 0.0) continue;
                if (Math.abs(t) > halfWidthAtS) continue;

                const wAlong = 1.0 - s / L;              
                const wSide  = 1.0 - Math.abs(t) / halfWidthAtS;
                const w = wAlong * wSide;

                if (w <= 0.0) continue;

                const d = -strength * w * 1.1;

                const idx = j * W + i;
                cur[idx] += d;
            }
        }

        let sum = 0;
        for (let i = 0; i < N; i++) {
            delta[i] = cur[i] - prev[i];
            sum += delta[i];
        }

        const mean = sum / N;
        for (let i = 0; i < N; i++) {
            delta[i] = (delta[i] - mean) * dt;
        }

        this.updateTexture(delta, this.bumpTexture);

        this.heightAddCS.run(this.heightTexture,     this.bumpTexture, W, H);
        this.heightAddCS.run(this.lowFreqTexture,    this.bumpTexture, W, H);
        this.heightAddCS.run(this.heightPrevTexture, this.bumpTexture, W, H);
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

// Simple UV sphere geometry matching the default vertex layout (pos, nor, uv).
function makeSphere(segmentsX: number, segmentsY: number) {
    const verts: number[] = [];
    const indices: number[] = [];

    for (let y = 0; y <= segmentsY; y++) {
        const v = y / segmentsY;
        const theta = v * Math.PI;
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);

        for (let x = 0; x <= segmentsX; x++) {
            const u = x / segmentsX;
            const phi = u * Math.PI * 2;
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            const px = sinTheta * cosPhi;
            const py = cosTheta;
            const pz = sinTheta * sinPhi;

            // pos
            verts.push(px, py, pz);
            // normal
            verts.push(px, py, pz);
            // uv
            verts.push(u, 1 - v);
        }
    }

    const stride = segmentsX + 1;
    for (let y = 0; y < segmentsY; y++) {
        for (let x = 0; x < segmentsX; x++) {
            const a = y * stride + x;
            const b = a + 1;
            const c = a + stride;
            const d = c + 1;
            indices.push(a, c, b, b, c, d);
        }
    }

    return { verts: new Float32Array(verts), indices: new Uint32Array(indices) };
}
