import { Camera } from '../stage/camera';

import commonRaw from './common.wgsl?raw';

import naiveVertRaw from './naive.vs.wgsl?raw';
import naiveFragRaw from './naive.fs.wgsl?raw';

import waterVertRaw from './water.vs.wgsl?raw';
import waterFragRaw from './water.fs.wgsl?raw';

import diffuseComputeRaw from './diffuse.cs.wgsl?raw';
import reconstructHeightRaw from './reconstructHeight.wgsl?raw';
import airyWaveComputeRaw from './airywave.cs.wgsl?raw';
import transportComputeRaw from './transport.cs.wgsl?raw';
import velocityComputeRaw from './velocity.cs.wgsl?raw';
import flowRecombineComputeRaw from './flowRecombine.cs.wgsl?raw';
import heightRecombineComputeRaw from './heightRecombine.cs.wgsl?raw';
import skyboxVertRaw from './skybox.vs.wgsl?raw';
import skyboxFragRaw from './skybox.fs.wgsl?raw';
import projectileVertRaw from './projectile.vs.wgsl?raw';
import projectileFragRaw from './projectile.fs.wgsl?raw';

import computeInitialVelocityX from './shallowWaterShaders/computeInitialVelocityX.cs.wgsl?raw';
import computeInitialVelocityY from './shallowWaterShaders/computeInitialVelocityY.cs.wgsl?raw';
import shallowHeight from './shallowWaterShaders/shallowHeight.cs.wgsl?raw';
import shallowVelocityXStep1 from './shallowWaterShaders/shallowVelocityXStep1.cs.wgsl?raw';
import shallowVelocityXStep2 from './shallowWaterShaders/shallowVelocityXStep2.cs.wgsl?raw';
import shallowVelocityYStep1 from './shallowWaterShaders/shallowVelocityYStep1.cs.wgsl?raw';
import shallowVelocityYStep2 from './shallowWaterShaders/shallowVelocityYStep2.cs.wgsl?raw';
import updateVelocityAndFluxX from './shallowWaterShaders/updateVelocityAndFluxX.cs.wgsl?raw';
import updateVelocityAndFluxY from './shallowWaterShaders/updateVelocityAndFluxY.cs.wgsl?raw';
import terrainCheck from './shallowWaterShaders/terrainCheck.cs.wgsl?raw';

import addOn from './addOn.cs.wgsl?raw';
import terrainVertRaw from './terrain.vs.wgsl?raw';
import terrainFragRaw from './terrain.fs.wgsl?raw';


// CONSTANTS (for use in shaders)
export const constants = {
    bindGroup_scene: 0,
    bindGroup_model: 1,
    bindGroup_material: 2,
    threadsInDiffusionBlockX: 16,
    threadsInDiffusionBlockY: 16,
    water_base_level: 10,
    flux_damp: 0.995
};

// =================================

console.log(constants);

function evalShaderRaw(raw: string) {
    return eval('`' + raw.replaceAll('${', '${constants.') + '`');
}

const commonSrc: string = evalShaderRaw(commonRaw);

function processShaderRaw(raw: string) {
    return commonSrc + evalShaderRaw(raw);
}

export const naiveVertSrc: string = processShaderRaw(naiveVertRaw);
export const naiveFragSrc: string = processShaderRaw(naiveFragRaw);

export const waterVertSrc: string = processShaderRaw(waterVertRaw);
export const waterFragSrc: string = processShaderRaw(waterFragRaw);
export const skyboxVertSrc: string = processShaderRaw(skyboxVertRaw);
export const skyboxFragSrc: string = processShaderRaw(skyboxFragRaw);
export const terrainVertSrc: string = processShaderRaw(terrainVertRaw);
export const terrainFragSrc: string = processShaderRaw(terrainFragRaw);
export const projectileVertSrc: string = processShaderRaw(projectileVertRaw);
export const projectileFragSrc: string = processShaderRaw(projectileFragRaw);

export const diffuseComputeSrc: string = processShaderRaw(diffuseComputeRaw);
export const reconstructHeightSrc: string = processShaderRaw(reconstructHeightRaw);
export const airyWaveComputeSrc: string = processShaderRaw(airyWaveComputeRaw);
export const transportComputeSrc: string = processShaderRaw(transportComputeRaw);
export const velocityComputeSrc: string = processShaderRaw(velocityComputeRaw);
export const flowRecombineComputeSrc: string = processShaderRaw(flowRecombineComputeRaw);
export const heightRecombineComputeSrc: string = processShaderRaw(heightRecombineComputeRaw);

export const computeInitialVelocityXSrc: string = processShaderRaw(computeInitialVelocityX);
export const computeInitialVelocityYSrc: string = processShaderRaw(computeInitialVelocityY);
export const shallowHeightSrc: string = processShaderRaw(shallowHeight);
export const shallowVelocityXStep1Src: string = processShaderRaw(shallowVelocityXStep1);
export const shallowVelocityXStep2Src: string = processShaderRaw(shallowVelocityXStep2);
export const shallowVelocityYStep1Src: string = processShaderRaw(shallowVelocityYStep1);
export const shallowVelocityYStep2Src: string = processShaderRaw(shallowVelocityYStep2);
export const updateVelocityAndFluxXSrc: string = processShaderRaw(updateVelocityAndFluxX);
export const updateVelocityAndFluxYSrc: string = processShaderRaw(updateVelocityAndFluxY);
export const terrainCheckSrc: string = processShaderRaw(terrainCheck);

export const addOnSrc: string = processShaderRaw(addOn);
