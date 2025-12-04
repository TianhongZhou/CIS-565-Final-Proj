
import * as shaders from "../shaders/shaders";

export class ShallowWater {
    private device: GPUDevice;
    private width: number;
    private height: number;

    //Note: These heights are the low frequency heights gotten from the diffuse step. 
    //First instantiation of the previousHeight texture should be the same as height texture, and not all zeroes.
    private heightMap: GPUTexture; 
    private previousHeightMap: GPUTexture;
    private fluxXMap: GPUTexture;
    private fluxYMap: GPUTexture;
    private velocityXMap: GPUTexture;
    private velocityYMap: GPUTexture;
    private changeInVelocityXMap: GPUTexture;
    private changeInVelocityYMap: GPUTexture;
    private terrainTex: GPUTexture;


    private ioBindGroupLayout!: GPUBindGroupLayout;
    private ioCombinedBindGroupLayout!: GPUBindGroupLayout;
    private constsBindGroupLayout!: GPUBindGroupLayout;
    private terrainBindGroupLayout!: GPUBindGroupLayout;

    private heightBindGroup!: GPUBindGroup;
    private previousHeightBindGroup!: GPUBindGroup;
    private fluxXBindGroup!: GPUBindGroup;
    private fluxYBindGroup!: GPUBindGroup;
    private velocityXBindGroup!: GPUBindGroup;
    private velocityYBindGroup!: GPUBindGroup;
    private changeInVelocityXBindGroup!: GPUBindGroup;
    private changeInVelocityYBindGroup!: GPUBindGroup;
    private constsBindGroup!: GPUBindGroup;
    private terrainBindGroup!: GPUBindGroup;

    //Redone bind groups
    private shallowHeightBindGroup!: GPUBindGroup;
    private shallowVelocityXStep1BindGroup!: GPUBindGroup;
    private shallowVelocityXStep2BindGroup!: GPUBindGroup;
    private shallowVelocityYStep1BindGroup!: GPUBindGroup;
    private shallowVelocityYStep2BindGroup!: GPUBindGroup;
    private updateVelocityAndFluxXBindGroup!: GPUBindGroup;
    private updateVelocityAndFluxYBindGroup!: GPUBindGroup;

    private timeStepBuffer!: GPUBuffer;
    private gridScaleBuffer!: GPUBuffer;


    private initialVelocityXPipeline!: GPUComputePipeline;
    private initialVelocityYPipeline!: GPUComputePipeline;

    private shallowHeightPipeline!: GPUComputePipeline;
    private shallowVelocityXStep1Pipeline!: GPUComputePipeline;
    private shallowVelocityXStep2Pipeline!: GPUComputePipeline;

    private shallowVelocityYStep1Pipeline!: GPUComputePipeline;
    private shallowVelocityYStep2Pipeline!: GPUComputePipeline;

    private updateVelocityAndFluxXPipeline!: GPUComputePipeline;
    private updateVelocityAndFluxYPipeline!: GPUComputePipeline;


    constructor(
        device: GPUDevice,
        width: number,
        height: number,
        heightTex: GPUTexture,  
        previousHeightTex: GPUTexture,     
        fluxXTex: GPUTexture,     
        fluxYTex: GPUTexture, 
        velocityXTex: GPUTexture,      
        velocityYTex: GPUTexture,     
        changeInVelocityXTex: GPUTexture,      
        changeInVelocityYTex: GPUTexture,     
        terrainTex: GPUTexture
    ) {
        this.device   = device;
        this.width    = width;
        this.height   = height;
        this.heightMap   = heightTex;
        this.previousHeightMap = previousHeightTex;
        this.fluxXMap = fluxXTex;
        this.fluxYMap = fluxYTex;
        this.velocityXMap = velocityXTex;
        this.velocityYMap  = velocityYTex;
        this.changeInVelocityXMap = changeInVelocityXTex;
        this.changeInVelocityYMap = changeInVelocityYTex;
        this.terrainTex = terrainTex;

        this.createBuffersAndLayouts();
        this.createBindGroups();
        this.createPipeline();

        // Initialize prev height
        const encoder = this.device.createCommandEncoder();
        encoder.copyTextureToTexture(
            { texture: this.heightMap },
            { texture: this.previousHeightMap },
            { width: this.width, height: this.height, depthOrArrayLayers: 1 }
        );
        this.device.queue.submit([encoder.finish()]);
    }

    private createBuffersAndLayouts() {
        this.timeStepBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.gridScaleBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.device.queue.writeBuffer(
            this.timeStepBuffer,
            0,
            new Float32Array([0.25])
        );
        this.device.queue.writeBuffer(
            this.gridScaleBuffer,
            0,
            new Float32Array([1.0])
        );

        // Read write storage textures
        this.ioBindGroupLayout = this.device.createBindGroupLayout({
            label: 'diffusion IO BGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'r32float',
                        access: 'read-write',
                        viewDimension: '2d',
                    },
                },
            ],
        });

        this.ioCombinedBindGroupLayout = this.device.createBindGroupLayout({
            label: 'diffusion IO combined BGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'r32float',
                        access: 'read-write',
                        viewDimension: '2d',
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'r32float',
                        access: 'read-write',
                        viewDimension: '2d',
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'r32float',
                        access: 'read-write',
                        viewDimension: '2d',
                    },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'r32float',
                        access: 'read-write',
                        viewDimension: '2d',
                    },
                }
            ],
        });




        // Constants
        this.constsBindGroupLayout = this.device.createBindGroupLayout({
            label: 'diffusion consts BGL',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' },   // timeStep
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' },   // gridScale
                }
            ],
        });
        
        this.terrainBindGroupLayout = this.device.createBindGroupLayout({
            label: 'terrain',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "unfilterable-float" } },
            ],
        });
    }

    private createBindGroups() {
        this.heightBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.heightMap.createView(),
                },
            ],
        });

        this.previousHeightBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.previousHeightMap.createView(),
                },
            ],
        });
        
        this.fluxXBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.fluxXMap.createView(),
                },
            ],
        });
        
        
        this.fluxYBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.fluxYMap.createView(),
                },
            ],
        });

        this.velocityXBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.velocityXMap.createView(),
                },
            ],
        });
        
        
        this.velocityYBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.velocityYMap.createView(),
                },
            ],
        });

         this.changeInVelocityXBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.changeInVelocityXMap.createView(),
                },
            ],
        });
        
        
        this.changeInVelocityYBindGroup = this.device.createBindGroup({
            layout: this.ioBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.changeInVelocityYMap.createView(),
                },
            ],
        });
        this.constsBindGroup = this.device.createBindGroup({
            layout: this.constsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.timeStepBuffer } },
                { binding: 1, resource: { buffer: this.gridScaleBuffer } }
            ],
        });
        this.terrainBindGroup = this.device.createBindGroup({
            label: 'terrain bind group',
            layout: this.terrainBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.terrainTex.createView(),
                },
            ],
        });
        //Redone bind groups
        this.shallowHeightBindGroup = this.device.createBindGroup({
            layout: this.ioCombinedBindGroupLayout,
            entries: [
                { binding: 0, resource: this.previousHeightMap.createView() },
                { binding: 1, resource: this.velocityXMap.createView() },
                { binding: 2, resource: this.velocityYMap.createView() },
                { binding: 3, resource: this.heightMap.createView() },
            ],
        });
        this.shallowVelocityXStep1BindGroup = this.device.createBindGroup({
            layout: this.ioCombinedBindGroupLayout,
            entries: [
                { binding: 0, resource: this.velocityXMap.createView() },
                { binding: 1, resource: this.fluxXMap.createView() },
                { binding: 2, resource: this.heightMap.createView() },
                { binding: 3, resource: this.changeInVelocityXMap.createView() },
            ],
        });
        this.shallowVelocityYStep1BindGroup = this.device.createBindGroup({
            layout: this.ioCombinedBindGroupLayout,
            entries: [
                { binding: 0, resource: this.velocityYMap.createView() },
                { binding: 1, resource: this.fluxXMap.createView() },
                { binding: 2, resource: this.heightMap.createView() },
                { binding: 3, resource: this.changeInVelocityYMap.createView() },
            ],
        });
        this.shallowVelocityXStep2BindGroup = this.device.createBindGroup({
            layout: this.ioCombinedBindGroupLayout,
            entries: [
                { binding: 0, resource: this.velocityXMap.createView() },
                { binding: 1, resource: this.fluxYMap.createView() },
                { binding: 2, resource: this.heightMap.createView() },
                { binding: 3, resource: this.changeInVelocityXMap.createView() },
            ],
        });
        this.shallowVelocityYStep2BindGroup = this.device.createBindGroup({
            layout: this.ioCombinedBindGroupLayout,
            entries: [
                { binding: 0, resource: this.velocityYMap.createView() },
                { binding: 1, resource: this.fluxYMap.createView() },
                { binding: 2, resource: this.heightMap.createView() },
                { binding: 3, resource: this.changeInVelocityYMap.createView() },
            ],
        });
        this.updateVelocityAndFluxXBindGroup = this.device.createBindGroup({
            layout: this.ioCombinedBindGroupLayout,
            entries: [
                { binding: 0, resource: this.changeInVelocityXMap.createView() },
                { binding: 1, resource: this.heightMap.createView() },
                { binding: 2, resource: this.velocityXMap.createView() },
                { binding: 3, resource: this.fluxXMap.createView() },
            ],
        });
        this.updateVelocityAndFluxYBindGroup = this.device.createBindGroup({
            layout: this.ioCombinedBindGroupLayout,
            entries: [
                { binding: 0, resource: this.changeInVelocityYMap.createView() },
                { binding: 1, resource: this.heightMap.createView() },
                { binding: 2, resource: this.velocityYMap.createView() },
                { binding: 3, resource: this.fluxYMap.createView() },
            ],
        });
    }

    private createPipeline() {
        this.initialVelocityXPipeline = this.device.createComputePipeline({
            label: 'Shallow Initial Velocity X Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioBindGroupLayout,       // group(0)  previousHeight
                    this.ioBindGroupLayout,       // group(1)  fluxX
                    this.ioBindGroupLayout,       // group(2)  velocityX
                    this.constsBindGroupLayout,   // group(3)  dt, gridScale
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'ShallowInitialVelocityX',
                    code: shaders.computeInitialVelocityXSrc,
                }),
                entryPoint: 'computeInitialVelocityX',
            },
        });
        this.initialVelocityYPipeline = this.device.createComputePipeline({
            label: 'Shallow Initial Velocity Y Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioBindGroupLayout,       // group(0)  previousHeight
                    this.ioBindGroupLayout,       // group(1)  fluxY
                    this.ioBindGroupLayout,       // group(2)  velocityY
                    this.constsBindGroupLayout,   // group(3)  dt, gridScale
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'ShallowInitialVelocityY',
                    code: shaders.computeInitialVelocityYSrc,
                }),
                entryPoint: 'computeInitialVelocityY',
            },
        });

        this.shallowHeightPipeline = this.device.createComputePipeline({
            label: 'Shallow Height Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioCombinedBindGroupLayout, // group(0)  previousHeight, velocityXIn, velocityYIn, heightOut
                    this.constsBindGroupLayout,   // group(1)  dt, gridScale
                    this.terrainBindGroupLayout,  // group(2)  terrain
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'ShallowHeight',
                    code: shaders.shallowHeightSrc,
                }),
                entryPoint: 'shallowHeight',
            },
        });
        this.shallowVelocityXStep1Pipeline = this.device.createComputePipeline({
            label: 'Shallow Velocity X Step 1 Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioCombinedBindGroupLayout, // group(0)  velocityX, fluxX, previousHeight, changeInVelocityX
                    this.constsBindGroupLayout,   // group(1)  dt, gridScale
                    this.terrainBindGroupLayout,  // group(2)  terrain
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'ShallowVelocityXStep1',
                    code: shaders.shallowVelocityXStep1Src,
                }),
                entryPoint: 'shallowVelocityXStep1',
            },
        });
        this.shallowVelocityXStep2Pipeline = this.device.createComputePipeline({
            label: 'Shallow Velocity X Step 2 Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioCombinedBindGroupLayout, // group(0)  velocityX, fluxY, previousHeight, changeInVelocityX
                    this.constsBindGroupLayout,   // group(1)  dt, gridScale
                    this.terrainBindGroupLayout,  // group(2)  terrain
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'ShallowVelocityXStep2',
                    code: shaders.shallowVelocityXStep2Src,
                }),
                entryPoint: 'shallowVelocityXStep2',
            },
        });
        this.shallowVelocityYStep1Pipeline = this.device.createComputePipeline({
            label: 'Shallow Velocity Y Step 1 Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioCombinedBindGroupLayout, // group(0)  velocityY, fluxX, previousHeight, changeInVelocityY
                    this.constsBindGroupLayout,   // group(1)  dt, gridScale
                    this.terrainBindGroupLayout,  // group(2)  terrain
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'ShallowVelocityYStep1',
                    code: shaders.shallowVelocityYStep1Src,
                }),
                entryPoint: 'shallowVelocityYStep1',
            },
        });
        this.shallowVelocityYStep2Pipeline = this.device.createComputePipeline({
            label: 'Shallow Velocity Y Step 2 Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioCombinedBindGroupLayout, // group(0)  velocityY, fluxY, previousHeight, changeInVelocityY
                    this.constsBindGroupLayout,   // group(1)  dt, gridScale
                    this.terrainBindGroupLayout,  // group(2)  terrain
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'ShallowVelocityYStep2',
                    code: shaders.shallowVelocityYStep2Src,
                }),
                entryPoint: 'shallowVelocityYStep2',
            },
        });
        this.updateVelocityAndFluxXPipeline = this.device.createComputePipeline({
            label: 'Update Velocity And Flux X Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioCombinedBindGroupLayout, // group(0)  changeInVelocityX, height, velocityX, fluxX
                    this.constsBindGroupLayout,   // group(1)  dt, gridScale
                    this.terrainBindGroupLayout,  // group(2)  terrain
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'UpdateVelocityAndFluxX',
                    code: shaders.updateVelocityAndFluxXSrc,
                }),
                entryPoint: 'updateVelocityAndFluxX',
            },
        });
        this.updateVelocityAndFluxYPipeline = this.device.createComputePipeline({
            label: 'Update Velocity And Flux Y Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.ioCombinedBindGroupLayout, // group(0)  changeInVelocityY, height, velocityY, fluxY
                    this.constsBindGroupLayout,   // group(1)  dt, gridScale
                    this.terrainBindGroupLayout,  // group(2)  terrain
                ],
            }),
            compute: {
                module: this.device.createShaderModule({
                    label: 'UpdateVelocityAndFluxY',
                    code: shaders.updateVelocityAndFluxYSrc,
                }),
                entryPoint: 'updateVelocityAndFluxY',
            },
        });
    }

    step(dt: number) {

        this.device.queue.writeBuffer(
            this.timeStepBuffer,
            0,
            new Float32Array([dt])
        );

        const encoder = this.device.createCommandEncoder();

        const wgX = Math.ceil(
            this.width / shaders.constants.threadsInDiffusionBlockX
        );
        const wgY = Math.ceil(
            this.height / shaders.constants.threadsInDiffusionBlockY
        );
        
        //Initial Velocity X
        {
            const initialVelocityXPass = encoder.beginComputePass();
            initialVelocityXPass.setPipeline(this.initialVelocityXPipeline);
            initialVelocityXPass.setBindGroup(0, this.previousHeightBindGroup);
            initialVelocityXPass.setBindGroup(1, this.fluxXBindGroup);
            initialVelocityXPass.setBindGroup(2, this.velocityXBindGroup);
            initialVelocityXPass.setBindGroup(3, this.constsBindGroup);
            initialVelocityXPass.dispatchWorkgroups(wgX, wgY);
            initialVelocityXPass.end();
        }

        //Initial Velocity Y
        {
            const initialVelocityYPass = encoder.beginComputePass();
            initialVelocityYPass.setPipeline(this.initialVelocityYPipeline);
            initialVelocityYPass.setBindGroup(0, this.previousHeightBindGroup);
            initialVelocityYPass.setBindGroup(1, this.fluxYBindGroup);
            initialVelocityYPass.setBindGroup(2, this.velocityYBindGroup);
            initialVelocityYPass.setBindGroup(3, this.constsBindGroup);
            initialVelocityYPass.dispatchWorkgroups(wgX, wgY);
            initialVelocityYPass.end();
        }
        
        //Copy Texture(height -> previousHeight)
        
        
        {
            
            encoder.copyTextureToTexture(
                {
                    texture: this.heightMap,
                },
                {
                    texture: this.previousHeightMap,
                },
                [this.width, this.height, 1]
            );

        }

        
        //Shallow Height
        {
            const shallowHeightPass = encoder.beginComputePass();
            shallowHeightPass.setPipeline(this.shallowHeightPipeline);
            shallowHeightPass.setBindGroup(0, this.shallowHeightBindGroup);
            shallowHeightPass.setBindGroup(1, this.constsBindGroup);
            shallowHeightPass.setBindGroup(2, this.terrainBindGroup);
            shallowHeightPass.dispatchWorkgroups(wgX, wgY);
            shallowHeightPass.end();
        }


        //Shallow Velocity X Step 1 and 2
        {
            const shallowVelocityXStep1Pass = encoder.beginComputePass();
            shallowVelocityXStep1Pass.setPipeline(this.shallowVelocityXStep1Pipeline);
            shallowVelocityXStep1Pass.setBindGroup(0, this.shallowVelocityXStep1BindGroup);
            shallowVelocityXStep1Pass.setBindGroup(1, this.constsBindGroup);
            shallowVelocityXStep1Pass.setBindGroup(2, this.terrainBindGroup);
            shallowVelocityXStep1Pass.dispatchWorkgroups(wgX, wgY);
            shallowVelocityXStep1Pass.end();
        }
        {
            const shallowVelocityXStep2Pass = encoder.beginComputePass();
            shallowVelocityXStep2Pass.setPipeline(this.shallowVelocityXStep2Pipeline);
            shallowVelocityXStep2Pass.setBindGroup(0, this.shallowVelocityXStep2BindGroup);
            shallowVelocityXStep2Pass.setBindGroup(1, this.constsBindGroup);
            shallowVelocityXStep2Pass.setBindGroup(2, this.terrainBindGroup);
            shallowVelocityXStep2Pass.dispatchWorkgroups(wgX, wgY);
            shallowVelocityXStep2Pass.end();
        }
        //Shallow Velocity Y Step 1 and 2
        {
            const shallowVelocityYStep1Pass = encoder.beginComputePass();
            shallowVelocityYStep1Pass.setPipeline(this.shallowVelocityYStep1Pipeline);
            shallowVelocityYStep1Pass.setBindGroup(0, this.shallowVelocityYStep1BindGroup);
            shallowVelocityYStep1Pass.setBindGroup(1, this.constsBindGroup);
            shallowVelocityYStep1Pass.setBindGroup(2, this.terrainBindGroup);
            shallowVelocityYStep1Pass.dispatchWorkgroups(wgX, wgY);
            shallowVelocityYStep1Pass.end();
        }
        {
            const shallowVelocityYStep2Pass = encoder.beginComputePass();
            shallowVelocityYStep2Pass.setPipeline(this.shallowVelocityYStep2Pipeline);
            shallowVelocityYStep2Pass.setBindGroup(0, this.shallowVelocityYStep2BindGroup);
            shallowVelocityYStep2Pass.setBindGroup(1, this.constsBindGroup);
            shallowVelocityYStep2Pass.setBindGroup(2, this.terrainBindGroup);
            shallowVelocityYStep2Pass.dispatchWorkgroups(wgX, wgY);
            shallowVelocityYStep2Pass.end();
        }  

        //Update Velocity and Flux X and Y 
        
        {
            const updateVelocityAndFluxXPass = encoder.beginComputePass();
            updateVelocityAndFluxXPass.setPipeline(this.updateVelocityAndFluxXPipeline);
            updateVelocityAndFluxXPass.setBindGroup(0, this.updateVelocityAndFluxXBindGroup);
            updateVelocityAndFluxXPass.setBindGroup(1, this.constsBindGroup);
            updateVelocityAndFluxXPass.setBindGroup(2, this.terrainBindGroup);
            updateVelocityAndFluxXPass.dispatchWorkgroups(wgX, wgY);
            updateVelocityAndFluxXPass.end();
        }
        
        {
            const updateVelocityAndFluxYPass = encoder.beginComputePass();
            updateVelocityAndFluxYPass.setPipeline(this.updateVelocityAndFluxYPipeline);
            updateVelocityAndFluxYPass.setBindGroup(0, this.updateVelocityAndFluxYBindGroup);
            updateVelocityAndFluxYPass.setBindGroup(1, this.constsBindGroup);
            updateVelocityAndFluxYPass.setBindGroup(2, this.terrainBindGroup);
            updateVelocityAndFluxYPass.dispatchWorkgroups(wgX, wgY);
            updateVelocityAndFluxYPass.end();
        }

        this.device.queue.submit([encoder.finish()]);
    }

}
/*
Pseudo-code

Textures needed:

height
previousHeight

fluxX
fluxY
velocityX
velocityY
changeInVelocityX
changeInvelocityY

Constants:
timestep
gridsize


Steps:

computeInitialVelocityX(previousHeight, fluxX, velocityX);
computeInitialVelocityY(previousHeight, fluxY, velocityY);

copyTexture(height -> previousHeight)

shallowHeight(previousHeight, velocityX, velocityY, height, constants);
shallowVelocityXStep1(velocityX, fluxX, previousHeight, changeInVelocityX, constants);
shallowVelocityXStep2(velocityX, fluxY, previousHeight, changeInVelocityX, constants);

shallowVelocityYStep1(velocityY, fluxX, previousHeight, changeInvelocityY, constants);
shallowVelocityYStep2(velocityY, fluxY, previousHeight, changeInvelocityY, constants);

updateVelocityAndFluxX(changeInVelocityX, height, velocity, fluxX, constants);
updateVelocityAndFluxY(changeInVelocityY, height, velocity, fluxY, constants);



TODO: debug shaders, fix texture sampling to account for texture boundary, start research on boundary conditions.
*/