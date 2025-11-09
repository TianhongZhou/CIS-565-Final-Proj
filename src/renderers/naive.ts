import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';

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

    // --- Upload helpers for writeTexture() row alignment ---
    private rowBytes = 0;
    private paddedBytesPerRow = 0;
    private uploadScratch: Uint8Array | null = null;

    private heightArray?: Float32Array; 
    private updater?: (dtSec: number, out: Float32Array) => void; 
    private _t = 0;

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

        this.pipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                label: "naive pipeline layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    renderer.materialBindGroupLayout
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

        // R32Float height texture (unfilterable). One float per texel.
        this.heightTexture = renderer.device.createTexture({
            size: [256, 256], 
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
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

        // Pipeline used to render the water surface (height-displaced grid).
        // Vertex shader samples the height texture at UV, computes world position & normal.
        // Fragment shader can do lighting.
        this.heightPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                bindGroupLayouts: [
                    // camera UBO
                    this.sceneUniformsBindGroupLayout, 
                    // height sampler/texture/UBO
                    this.heightBindGroupLayout         
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
    }

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = renderer.context.getCurrentTexture().createView();

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
    }

    // Initialize (or resize) the height texture and update dependent state.
    // Also uploads the given height data for the current frame.
    updateHeight(heightData: Float32Array, w: number, h: number) {
        const data = new Float32Array(heightData);

        const sizeChanged = (w !== this.heightW) || (h !== this.heightH);

        //If the size has changed, the texture size will change, so we need to recreate the texture and update the bind group layout it's in
        if (sizeChanged) {
            this.heightW = w; this.heightH = h;

            this.heightTexture = renderer.device.createTexture({
            size: [w, h],
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });

            // Update uvTexel = (1/w, 1/h) in the UBO
            renderer.device.queue.writeBuffer(this.heightConsts, 0, new Float32Array([1/w, 1/h]));
            
            // Recreate the bind group because the texture view changed
            this.heightBindGroup = renderer.device.createBindGroup({
            layout: this.heightBindGroupLayout,
            entries: [
                { binding: 0, resource: this.heightSampler },
                { binding: 1, resource: this.heightTexture.createView() },
                { binding: 2, resource: { buffer: this.heightConsts } },
            ],
            });
        }

        // Upload the data (handles 256-byte row alignment as needed)
        const rowBytes = w * 4;
        const align = 256;
        const padded = Math.ceil(rowBytes / align) * align;

        if (padded === rowBytes) {
            // Direct upload when already aligned
            renderer.device.queue.writeTexture(
                { texture: this.heightTexture },
                data,
                { bytesPerRow: rowBytes },
                [w, h, 1]
            );
        } else {
            // Row-padded upload path
            const tmp = new Uint8Array(padded * h);
            const src = new Uint8Array(heightData.buffer, heightData.byteOffset, heightData.byteLength);
            for (let y = 0; y < h; y++) {
                tmp.set(src.subarray(y*rowBytes, y*rowBytes + rowBytes), y*padded);
            }
            renderer.device.queue.writeTexture(
                { texture: this.heightTexture },
                tmp,
                { bytesPerRow: padded },
                [w, h, 1]
            );
        }
    }

    setHeightUpdater(fn: (dtSec: number, out: Float32Array) => void) {
        this.updater = fn;
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
        if (!this.heightArray) {
            this.heightArray = new Float32Array(this.heightW * this.heightH);
        }
    }

    /**
     * Writes the height data array to the height texture
     * @param heightData The height array
     */
    // Fast path: upload a W×H Float32Array into the existing height texture.
    // Avoids re-creating textures/bind groups; handles 256-byte row alignment.
    private updateHeightInPlace(heightData: Float32Array) {
        const data = new Float32Array(heightData);
        this.initRowInfoIfNeeded();

        // Aligned -> upload directly
        if (this.paddedBytesPerRow === this.rowBytes) {
            renderer.device.queue.writeTexture(
                { texture: this.heightTexture },
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
                { texture: this.heightTexture },
                tmp,
                { bytesPerRow: this.paddedBytesPerRow },
                [this.heightW, this.heightH, 1]
            );
        }
    }

    // Called every frame before drawing. Fills heightArray either via user-provided
    // simulation callback (setHeightUpdater) or with a demo wave when no callback is set.
    // Then uploads the new height field into the GPU texture.
    protected override onBeforeDraw(dtMs: number): void {
        const dt = dtMs / 1000; 
        this.initRowInfoIfNeeded();
        const arr = this.heightArray!;

        if (this.updater) {
            // External simulation writes into 'arr' (W*H floats)
            this.updater(dt, arr);
        } else {
            this._t += dt;
            const W = this.heightW, H = this.heightH;
            const s = 0.05, w = this._t;
            for (let y = 0; y < H; y++) {
                for (let x = 0; x < W; x++) {
                arr[y*W + x] = 0.1*Math.sin(x*s + w) * Math.cos(y*s + 0.5*w);
                }
            }
        }

        // Upload into the existing height texture
        this.updateHeightInPlace(arr);
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
