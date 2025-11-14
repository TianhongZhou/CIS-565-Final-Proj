import { Camera } from '../stage/camera';

import commonRaw from './common.wgsl?raw';

import naiveVertRaw from './naive.vs.wgsl?raw';
import naiveFragRaw from './naive.fs.wgsl?raw';

import waterVertRaw from './water.vs.wgsl?raw';
import waterFragRaw from './water.fs.wgsl?raw';

import diffuseComputeRaw from './diffuse.cs.wgsl?raw';


// CONSTANTS (for use in shaders)
export const constants = {
    bindGroup_scene: 0,
    bindGroup_model: 1,
    bindGroup_material: 2,
    threadsInDiffusionBlockX: 16,
    threadsInDiffusionBlockY: 16,
    water_base_level: 0
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

export const diffuseComputeSrc: string = processShaderRaw(diffuseComputeRaw);