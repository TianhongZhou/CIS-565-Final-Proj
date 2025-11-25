import Stats from 'stats.js';
import { GUI } from 'dat.gui';

import { initWebGPU, Renderer } from './renderer';
import { NaiveRenderer } from './renderers/naive';
import { DiffuseRenderer } from './renderers/DiffuseRender';
import { Simulator } from './simulator/simulator';

import { setupLoaders, Scene } from './stage/scene';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';
import { constants } from './shaders/shaders';

import { DiffuseCS } from './simulator/Diffuse';
import { ShallowRenderer } from './renderers/ShallowRender';

await initWebGPU();
setupLoaders();

let scene = new Scene();
await scene.loadGltf('./scenes/sponza/Sponza.gltf');

const camera = new Camera();

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const gui = new GUI();

const stage = new Stage(scene, camera, stats);

// NaiveRenderer
var renderer: NaiveRenderer | undefined;

renderer?.stop();
renderer = new NaiveRenderer(stage);

// // DiffuseRender
// var renderer: DiffuseRenderer | undefined;

// renderer?.stop();
// renderer = new DiffuseRenderer(stage);

// Set water-surface parameters:
// worldScaleXY -> the grid spans [-a,+b] in X and Z (width/depth = 2 units)
// heightScale  -> amplitude multiplier for heights sampled from the texture
// baseLevel    -> lifts the whole plane in world Y (0 = centered at origin)
renderer.setHeightParams(5, 2, 1, constants.water_base_level);

// Initialize GPU height texture and bind groups with the first frame’s data.
//renderer.updateHeight(arr, W, H);
